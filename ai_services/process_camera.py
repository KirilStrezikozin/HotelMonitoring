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
            self._generate_output_filename()
            if config_camera.video_path and not config_camera.output_url
            else None
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

        self.person_count = 0

        self.data_lock = threading.Lock()
        self.inference_event = threading.Event()
        self.latest_inference_frame = None
        self.latest_results = None

        self.threads = []

        self.frame_count = 0
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.total_detection_time = 0.0
        self.total_tracking_time = 0.0
        self.total_reid_time = 0.0

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
        frame_delay = (
            1.0 / self.source.fps
            if (self.source.fps > 0 and not self.is_stream)
            else 0.0
        )

        persistent_results = []

        while not self.stop_event.is_set():
            start_time = time.time()

            if not self.source.grab() or self.frame_count >= self.max_frames:
                self.logger.info("Finished video file or RTSP stream disconnected.")
                self.stop_event.set()
                with self.data_lock:
                    self.latest_inference_frame = (None, None)
                self.inference_event.set()
                break

            self.frame_count += 1
            frame = self.source.retrieve()
            if frame is None:
                continue

            frame = self.processor.preprocess(frame)

            if self.frame_count % self.config.detection_interval == 0:
                with self.data_lock:
                    self.latest_inference_frame = (self.frame_count, frame.copy())
                self.inference_event.set()

            with self.data_lock:
                if self.latest_results is not None:
                    persistent_results = self.latest_results

            for person in persistent_results:
                l, t, r, b = person["bbox"]

                label = str(person.get("global_id", "Person"))

                self.processor.annotate(frame, l, t, r, b, label)

            self.processor.draw_person_count(frame, self.person_count)

            self.output.write(frame)

            if frame_delay > 0:
                elapsed = time.time() - start_time
                time_to_sleep = frame_delay - elapsed
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

    def _inference_thread(self):
        """Thread 2: Receives frames, runs detection + tracking + ReID, and sends results back."""
        while not self.stop_event.is_set():
            if not self.inference_event.wait(timeout=1.0):
                continue

            self.inference_event.clear()

            with self.data_lock:
                inference_data = self.latest_inference_frame

            if inference_data is None or inference_data[0] is None:
                break

            frame_count, frame = inference_data

            if frame is None:
                break
            try:
                frame_started = time.time()
                self.logger.info(f"Frame {frame_count}: starting inference")
                yolo_start = time.time()
                detections = self.detector.detect(frame)
                yolo_elapsed = time.time() - yolo_start
                self.logger.info(
                    f"Frame {frame_count}: detection complete with {len(detections)} detections in {yolo_elapsed:.3f}s"
                )
                self.total_detection_time += yolo_elapsed

                self.person_count = len(detections)
                if self.config.panorama:
                    tracked_results = []
                    for bbox, _, _ in detections:
                        x, y, w, h = bbox
                        l, t, r, b = x, y, x + w, y + h
                        tracked_results.append({"bbox": (l, t, r, b), "global_id": ""})
                else:
                    if not detections:
                        detections = []

                    tracker_result = self.tracker.update(
                        frame,
                        detections,
                        self.reid_model,
                        frame_count,
                        self.config.detection_interval,
                        self.config.camera_id,
                    )
                    tracked_results = tracker_result.get("render_data", [])
                    tracking_elapsed = tracker_result.get("tracking_time", 0.0)
                    reid_elapsed = tracker_result.get("reid_time", 0.0)
                    self.total_tracking_time += tracking_elapsed
                    self.total_reid_time = (
                        getattr(self, "total_reid_time", 0.0) + reid_elapsed
                    )
                    self.logger.info(
                        f"Frame {frame_count}: tracking complete ({len(tracked_results)} tracked) in {tracking_elapsed:.3f}s"
                    )
                    self.logger.info(
                        f"Frame {frame_count}: ReID complete in {reid_elapsed:.3f}s"
                    )

                inference_elapsed = time.time() - frame_started
                self.inference_count += 1
                self.total_inference_time += inference_elapsed
                self.logger.info(
                    f"Frame {frame_count}: total inference time {inference_elapsed:.3f}s"
                )

                with self.data_lock:
                    self.latest_results = tracked_results

            except Exception as e:
                self.logger.error(
                    f"Inference crashed on Frame {frame_count}: {e}", exc_info=True
                )
                continue

        if self.inference_count > 0:
            avg_inference = self.total_inference_time / self.inference_count
            avg_detection = self.total_detection_time / self.inference_count
            avg_tracking = self.total_tracking_time / self.inference_count
            avg_reid = self.total_reid_time / self.inference_count
            self.logger.info(
                f"Inference summary: {self.inference_count} frames, avg total {avg_inference:.3f}s, "
                f"avg detection {avg_detection:.3f}s, avg tracking {avg_tracking:.3f}s, "
                f"avg ReID {avg_reid:.3f}s"
            )

    def cleanup(self):
        self.stop_event.set()
        self.inference_event.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        self.source.release()
        self.output.release()
