import os
import platform
import threading

import torch
from ultralytics import YOLO

from ai_services.process_camera import CameraProcessor
from ai_services.reid import ReIDModel
from config.camera import CameraConfig
from db.analytics import AnalyticsDB

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MultiCameraTracker:
    """Main class for multi-camera person tracking system with threading."""

    def __init__(self, reid_model_path: str, camera_configs: list[CameraConfig]):
        self.analytics_db = AnalyticsDB()

        print("Initializing ReID Model...")
        self.reid_model = ReIDModel(reid_model_path, db=self.analytics_db)

        # --- YOLO: export to CoreML (Apple Silicon) or OpenVINO (Linux/x86) ---
        base_model_name = "models/best"
        pytorch_model_path = f"{base_model_name}.pt"

        is_apple_silicon = (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        )

        if is_apple_silicon:
            # Load .pt directly — PyTorch routes inference to MPS automatically.
            # CoreML is skipped: its baked NMS threshold + resources.bin issues
            # cause poor detection on M4. PyTorch MPS matches Colab results.
            print(f"Loading detector from {pytorch_model_path} (MPS)...")
            self.detector = YOLO(pytorch_model_path)
        else:
            exported_model_path = f"{base_model_name}_openvino_model"
            if not os.path.exists(exported_model_path):
                print(f"Exporting {pytorch_model_path} to OpenVINO...")
                YOLO(pytorch_model_path).export(format="openvino", imgsz=640)
                print("Export complete.")
            print(f"Loading detector from {exported_model_path}...")
            self.detector = YOLO(exported_model_path, task="detect")
            try:
                self.detector.model.fuse()
            except Exception:
                pass

        # Locks: allow different cameras to run YOLO / ReID concurrently
        self._yolo_lock = threading.Lock()
        self._reid_lock = threading.Lock()

        print("Initializing camera processors...")
        self.cameras: dict[int, CameraProcessor] = {
            cfg.camera_id: CameraProcessor(
                cfg, self.analytics_db, self.detector, self.reid_model
            )
            for cfg in camera_configs
        }
        self.threads: list[threading.Thread] = []

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start one thread per camera and wait for all to finish."""
        try:
            for cam_id, camera in self.cameras.items():
                t = threading.Thread(
                    target=self._process_camera_loop,
                    args=(cam_id, camera),
                    daemon=True,
                    name=f"camera-{cam_id}",
                )
                t.start()
                self.threads.append(t)

            for t in self.threads:
                t.join()

        finally:
            self._cleanup()

    # ------------------------------------------------------------------

    def _process_camera_loop(self, cam_id: int, camera: CameraProcessor) -> None:
        print(f"[Camera {cam_id}] started")
        while True:
            with self._yolo_lock:
                frame = camera.process_frame()  # detector injected in __init__

            if frame is None:
                print(f"[Camera {cam_id}] stream ended")
                break

            with self._reid_lock:
                annotated_frame = camera.process_tracks(
                    frame
                )  # reid injected in __init__

            camera.write_frame(annotated_frame)

        camera.cleanup()

    def _cleanup(self) -> None:
        """Safety-net cleanup — CameraProcessor.cleanup() is idempotent."""
        for camera in self.cameras.values():
            camera.cleanup()
