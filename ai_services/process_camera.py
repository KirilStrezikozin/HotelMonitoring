import logging
import queue
import threading
import time
import torch
from ultralytics import YOLO

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
    """Orchestrator that automatically adapts its pipeline behavior for MP4 vs RTSP."""

    def __init__(
        self, config_camera: CameraConfig, detector: YOLO, reid_model: ReIDModel
    ):
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

        self.queue_capacity = 4
        self.input_queue = queue.Queue(maxsize=self.queue_capacity)
        self.output_queue = queue.Queue(maxsize=self.queue_capacity)
        self.threads = []

        self.frame_count = 0
        self.max_frames = (
            float("inf")
            if self.is_stream
            else int(self.source.fps * config_camera.max_duration_seconds)
        )

    def _generate_output_filename(self) -> str:
        return f"output_osnet_x1_0_{self.config.camera_id}.mp4"

    def start(self):
        self.logger.info("Starting processing pipeline.")
        t_reader = threading.Thread(
            target=self._reader_thread, name=f"Reader-{self.config.camera_id}"
        )
        t_processor = threading.Thread(
            target=self._processor_thread, name=f"Processor-{self.config.camera_id}"
        )
        t_writer = threading.Thread(
            target=self._writer_thread, name=f"Writer-{self.config.camera_id}"
        )

        self.threads.extend([t_reader, t_processor, t_writer])
        for t in self.threads:
            t.daemon = True
            t.start()

    def _reader_thread(self):
        """Thread 1: Reads frames. Blocks/paces for files; drops/evicts for streams."""
        # Calculate expected time per frame (e.g., 1 / 30.0 = 0.033s) -> Only used for MP4s
        frame_delay = (
            1.0 / self.source.fps
            if (self.source.fps > 0 and not self.is_stream)
            else 0.0
        )

        while not self.stop_event.is_set():
            start_time = time.time()

            if not self.source.grab() or self.frame_count >= self.max_frames:
                self.logger.info("Finished video file or RTSP stream disconnected.")
                self.stop_event.set()
                break

            self.frame_count += 1
            frame = self.source.retrieve()
            if frame is None:
                continue

            if self.is_stream:
                if self.input_queue.full():
                    try:
                        dropped_id, _ = self.input_queue.get_nowait()
                        self.logger.debug(f"RTSP Lag: Dropped old frame {dropped_id}")
                    except queue.Empty:
                        pass
                try:
                    self.input_queue.put_nowait((self.frame_count, frame))
                except queue.Full:
                    pass
            else:
                try:
                    self.input_queue.put((self.frame_count, frame), timeout=2)
                except queue.Full:
                    self.logger.warning(
                        "Pipeline severely stalled during MP4 playback."
                    )

            if frame_delay > 0:
                elapsed = time.time() - start_time
                time_to_sleep = frame_delay - elapsed
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        try:
            self.input_queue.put((None, None), timeout=2)
        except queue.Full:
            pass

    def _processor_thread(self):
        """Thread 2: Processes frames and pushes them forward conditionally."""
        while not self.stop_event.is_set():
            try:
                frame_count, frame = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                try:
                    self.output_queue.put(None, timeout=2)
                except queue.Full:
                    pass
                break

            frame = self.processor.preprocess(frame)

            if frame_count % self.config.detection_interval == 0:
                detections = self.detector.detect(frame)
                if not detections or len(detections) == 0:
                    self._push_to_output_queue(frame)
                    continue
            else:
                detections = []

            self.tracker.update(
                frame,
                detections,
                self.reid_model,
                frame_count,
                self.config.detection_interval,
                self.config.camera_id,
            )

            self._push_to_output_queue(frame)

    def _push_to_output_queue(self, frame):
        """Handles routing to the output/writer thread safely depending on mode."""
        if self.is_stream:
            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.output_queue.put_nowait(frame)
            except queue.Full:
                pass
        else:
            try:
                self.output_queue.put(frame, timeout=2)
            except queue.Full:
                pass

    def _writer_thread(self):
        """Thread 3: Writes out or streams out frames."""
        while not self.stop_event.is_set():
            try:
                frame = self.output_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                break

            self.output.write(frame)

    def cleanup(self):
        self.stop_event.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        self.source.release()
        self.output.release()
