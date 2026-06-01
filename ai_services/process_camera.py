import logging
import queue
import threading
import time
import torch
import numpy as np

from config.camera import CameraConfig
from ai_services.reid import ReIDModel
from ai_services.video_capture import VideoSource
from ai_services.video_writer import VideoOutput
from ai_services.frame_processor import FrameProcessor
from ai_services.person_detector import PersonDetector
from ai_services.tracker_manager import TrackerManager

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


class CameraProcessor:
    """Orchestrator using a 2-thread architecture for frame processing."""

    def __init__(self, config_camera: CameraConfig, detector, reid_model: ReIDModel):
        self.config = config_camera
        self.detector = PersonDetector(detector)
        self.reid_model = reid_model

        self.logger = logging.getLogger(f"CameraProcessor-{config_camera.camera_id}")
        self.logger.setLevel(logging.INFO)

        self.source = VideoSource(config_camera.video_path, config_camera.stream_url)

        self.is_stream = bool(config_camera.stream_url)
        self.logger.info(
            f"Initialized in {'STREAM (RTSP)' if self.is_stream else 'FILE (MP4)'} mode."
        )

        output_path = (
            self._generate_output_filename() if config_camera.video_path else None
        )
        self.output = VideoOutput(
            width=self.source.width,
            height=self.source.height,
            fps=self.source.fps,
            output_path=output_path,
            stream_url=config_camera.output_url,
        )

        self.processor = FrameProcessor()
        self.tracker = TrackerManager()

        self.stop_event = threading.Event()

        self.inference_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)

        self.threads = []

        self.frame_count = 0
        self.max_frames = (
            float("inf")
            if self.is_stream
            else int(self.source.fps * config_camera.max_duration_seconds)
        )

    def _warmup_models(self):
        """Run dummy data through the models to ensure they're loaded and optimized before real processing starts."""
        self.logger.info("Warming up AI models...")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.detect(dummy_frame)

        dummy_crop = np.zeros((256, 128, 3), dtype=np.uint8)
        self.reid_model.extract_embedding(dummy_crop)

        self.logger.info("Models warmed up successfully.")

    def _generate_output_filename(self) -> str:
        return f"output_osnet_x1_0_{self.config.camera_id}.mp4"

    def start(self):
        self._warmup_models()
        self.logger.info(
            "Models are warmed up and ready. Starting processing pipeline."
        )
        t_main = threading.Thread(
            target=self._main_thread, name=f"Main-{self.config.camera_id}"
        )
        t_inference = threading.Thread(
            target=self._inference_thread, name=f"Inference-{self.config.camera_id}"
        )

        self.threads.extend([t_main, t_inference])
        for t in self.threads:
            t.daemon = True
            t.start()

    def _main_thread(self):
        """Thread 1: Reads frames, routes 1/10 to inference, applies results, and writes."""
        # Calculate expected time per frame (e.g., 1 / 30.0 = 0.033s) -> Only used for MP4s
        frame_delay = (
            1.0 / self.source.fps
            if (self.source.fps > 0 and not self.is_stream)
            else 0.0
        )

        current_results = None

        while not self.stop_event.is_set():
            start_time = time.time()

            if not self.source.grab() or self.frame_count >= self.max_frames:
                self.logger.info("Finished video file or RTSP stream disconnected.")
                self.stop_event.set()
                try:
                    self.inference_queue.put_nowait((None, None))
                except queue.Full:
                    pass
                break

            self.frame_count += 1
            frame = self.source.retrieve()
            if frame is None:
                continue

            frame = self.processor.preprocess(frame)

            if self.frame_count % self.config.detection_interval == 0:
                if self.inference_queue.full():
                    try:
                        self.inference_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.inference_queue.put_nowait((self.frame_count, frame.copy()))
                except queue.Full:
                    pass

            try:
                while not self.result_queue.empty():
                    current_results = self.result_queue.get_nowait()
            except queue.Empty:
                pass

            if current_results is not None:
                for person in current_results:
                    l, t, r, b = person["bbox"]
                    global_id = person["global_id"]
                    self.processor.annotate(frame, l, t, r, b, str(global_id))
                self.processor.draw_person_count(frame, len(current_results))

            self.output.write(frame)

            if frame_delay > 0:
                elapsed = time.time() - start_time
                time_to_sleep = frame_delay - elapsed
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

    def _inference_thread(self):
        """Thread 2: Receives frames, runs detection + tracking + ReID, and sends results back."""
        while not self.stop_event.is_set():
            try:
                frame_count, frame = self.inference_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                break

            detections = self.detector.detect(frame)
            if not detections:
                detections = []

            tracked_results = self.tracker.update(
                frame,
                detections,
                self.reid_model,
                frame_count,
                self.config.detection_interval,
                self.config.camera_id,
            )

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.result_queue.put_nowait(tracked_results)
            except queue.Full:
                pass

    def cleanup(self):
        self.stop_event.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        self.source.release()
        self.output.release()
