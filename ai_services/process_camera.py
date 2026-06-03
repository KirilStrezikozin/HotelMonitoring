import cv2
import os
from datetime import datetime

import numpy as np
from ultralytics import YOLO

from config.camera import CameraConfig
from config.logging import setup_camera_logger
from config.tracking import TrackingConfig
from ai_services.reid import ReIDModel, ReIDTracker
from ai_services.video_capture import VideoSource
from ai_services.video_writer import VideoOutput
from ai_services.person_detector import PersonDetector
from ai_services.frame_processor import FrameProcessor
from ai_services.tracker_manager import TrackerManager
from ai_services.event_detector import EventDetector
from db.analytics import AnalyticsDB
from schema.embedding import TrackState

tracking_config = TrackingConfig()


class CameraProcessor:
    """
    Orchestrates the full per-camera pipeline:
        VideoSource → FrameProcessor → PersonDetector → TrackerManager
        → ReID → events → DB → FrameProcessor (annotation) → VideoOutput
    """

    def __init__(
        self,
        config_camera: CameraConfig,
        analytics_db: AnalyticsDB,
        detector: YOLO,
        reid_model: ReIDModel,
    ):
        self.config = config_camera
        self.db = analytics_db
        self.reid = ReIDTracker(reid_model)

        # ---- sub-class instances ----
        self.source = VideoSource(config_camera.video_path, config_camera.stream_url)
        self.processor = FrameProcessor()
        self.detector = PersonDetector(
            detector,
            min_box_area=config_camera.min_box_area,
            is_elevator=(config_camera.camera_id == 3),
        )
        self.tracker = TrackerManager()

        # ---- video output ----
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

        self.fps = self.source.fps
        self.width = self.source.width
        self.height = self.source.height
        self.min_box_area = config_camera.min_box_area

        self.frame_count = config_camera.detection_interval - 1
        self.max_frames = (
            float("inf")
            if config_camera.stream_url
            else int(self.fps * config_camera.max_duration_seconds)
        )
        self.last_tracks: list = []
        self.last_detection: list = []

        self.logger = setup_camera_logger(config_camera.camera_id)

        # ---- event detection (entry/exit + elevator) ----
        self.events = EventDetector(
            camera_id=config_camera.camera_id,
            db=analytics_db,
            logger=self.logger,
            width=self.width,
            height=self.height,
        )

        # ---- per-track state machine ----
        self.track_states: dict[int, TrackState] = {}

        # ---- staff IDs (loaded once; never change during a run) ----
        self.staff_ids: set[str] = set(self.db.get_staff_ids())

    def _generate_output_filename(self) -> str:
        return f"output_osnet_x1_0_{self.config.camera_id}.mp4"

    def get_current_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")[:-3]

    # ------------------------------------------------------------------
    # Frame reading & detection
    # ------------------------------------------------------------------

    def process_frame(self) -> np.ndarray | None:
        """Grab frame → preprocess → detect → track. Returns annotated frame or None."""
        if not self.source.grab() or self.frame_count >= self.max_frames:
            return None

        frame = self.source.retrieve()
        if frame is None:
            return None

        self.frame_count += 1

        # FrameProcessor: optional preprocessing (bilateral filter disabled by default)

        if self.frame_count % self.config.detection_interval == 0:
            detections = self.detector.detect(frame)  # PersonDetector
            self.last_detection = detections

        self.last_tracks = self.tracker.update(
            frame, self.last_detection
        )  # TrackerManager
        return frame

    # ------------------------------------------------------------------
    # Track state helpers
    # ------------------------------------------------------------------

    def _get_or_create_track_state(
        self, local_id: int, global_id: str | None = None
    ) -> TrackState:
        if local_id not in self.track_states:
            self.track_states[local_id] = TrackState(
                local_id=local_id,
                global_id=global_id,
                last_seen_frame=self.frame_count,
                elevator=False,
            )
        else:
            self.track_states[local_id].last_seen_frame = self.frame_count
            if global_id:
                self.track_states[local_id].global_id = global_id
        return self.track_states[local_id]

    def _cleanup_dead_tracks(self) -> None:
        active_ids = {t.track_id for t in self.last_tracks}
        dead = [tid for tid in self.track_states if tid not in active_ids]
        for tid in dead:
            self.track_states.pop(tid, None)
            self.reid.cleanup_track(tid)
            self.events.cleanup_track(tid)

    # ------------------------------------------------------------------
    # Main track processing
    # ------------------------------------------------------------------

    def process_tracks(self, frame: np.ndarray) -> np.ndarray:
        """ReID → events → DB → annotation via FrameProcessor."""
        is_detection_frame = self.frame_count % self.config.detection_interval == 0

        # Clean copy used for body crops — made ONCE before any cv2.rectangle/putText
        # so that annotations from other tracks never bleed into saved images.
        # save_crops_this_frame = (
        #     is_detection_frame
        #     and self.frame_count % (self.config.detection_interval * 3) == 0
        # )
        # clean_frame = frame.copy() if save_crops_this_frame else None

        if is_detection_frame:
            self._cleanup_dead_tracks()

        # ------------------------------------------------------------------
        # Build reid_skip_ids: tracks whose ReID should be skipped this frame
        # (edge-of-frame or confirmed inside elevator). Their existing global
        # ID is preserved in track_to_global by ReIDTracker.
        # ------------------------------------------------------------------
        # Edge margin: 5% of frame dimension (scales with resolution).
        # e.g. 1920×1080 → ~54px,  596×336 → ~17px  (old hardcoded 100px was
        # cutting off most detections on small preview videos).
        edge_h = int(self.height * 0.05)

        reid_skip_ids: set[int] = set()
        if is_detection_frame:
            for track in self.last_tracks:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                l, t = max(0, l), max(0, t)
                r, b = min(self.width, r), min(self.height, b)
                if r <= l or b <= t:
                    reid_skip_ids.add(local_id)
                    continue
                is_bottom_frame = (
                    False if self.config.camera_id == 3 else (self.height - b) < edge_h
                )
                is_top_frame = False if self.config.camera_id == 3 else t < edge_h
                ts = self.track_states.get(local_id)
                in_elevator = (
                    ts.elevator if ts and self.config.camera_id == 3 else False
                )
                if is_bottom_frame or is_top_frame or in_elevator:
                    reid_skip_ids.add(local_id)

        # ------------------------------------------------------------------
        # Delegate embedding extraction + global-ID assignment to ReIDTracker
        # ------------------------------------------------------------------
        self.reid.process_tracks(
            self.last_tracks,
            frame,
            self.config.camera_id,
            is_detection_frame,
            reid_skip_ids,
            self.min_box_area,
            self.logger,
        )

        # ------------------------------------------------------------------
        # Annotation + event detection + DB recording
        # ------------------------------------------------------------------
        active_track_count = 0

        for track in self.last_tracks:
            try:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())  # noqa: E741
                l, t = max(0, l), max(0, t)
                r, b = min(self.width, r), min(self.height, b)

                if r <= l or b <= t:
                    continue

                vertical = (
                    False if self.config.camera_id == 3 else (r - l) / (b - t) > 1.6
                )
                if (r - l) * (b - t) <= self.min_box_area or vertical:
                    continue

                active_track_count += 1
                assigned_gid = self.reid.get_global_id(local_id)
                track_state = self._get_or_create_track_state(local_id, assigned_gid)

                # Camera 3: elevator event — only when ReID was run this frame
                if (
                    self.config.camera_id == 3
                    and is_detection_frame
                    and local_id not in reid_skip_ids
                    and assigned_gid
                ):
                    elevator_result = self.events.check_elevator_event(
                        track_id=local_id,
                        global_id=assigned_gid,
                        bbox=(l, t, r, b),
                        frame_number=self.frame_count,
                    )
                    if elevator_result is not None:
                        track_state.elevator = elevator_result

                if assigned_gid:
                    timestamp = self.get_current_timestamp()

                    # Save body crop from the pre-annotation clean copy
                    # if clean_frame is not None:
                    #     crop = clean_frame[t:b, l:r]
                    #     if crop.size > 0:
                    #         self.save_body_crop(assigned_gid, crop, timestamp)

                    FrameProcessor.annotate(
                        frame,
                        l,
                        t,
                        r,
                        b,
                        assigned_gid,
                        is_staff=(assigned_gid in self.staff_ids),
                    )

                    # Entry-line overlay for camera 1
                    if self.config.camera_id == 1:
                        cv2.line(
                            frame,
                            (0, self.events.entry_line_y),
                            (self.width, self.events.entry_line_y),
                            (255, 0, 0),
                            2,
                        )

                    self.events.check_entry_event(
                        track_id=local_id,
                        global_id=assigned_gid,
                        bbox=(l, t, r, b),
                        frame_number=self.frame_count,
                    )

                    if is_detection_frame:
                        self.db.record_sighting(
                            global_id=assigned_gid,
                            camera_id=self.config.camera_id,
                            timestamp=timestamp,
                            frame_number=self.frame_count,
                            bbox=(l, t, r, b),
                        )

            except Exception as e:
                self.logger.error(
                    {
                        "event": "error",
                        "frame": self.frame_count,
                        "error": str(e),
                    }
                )
                continue

        FrameProcessor.draw_person_count(frame, active_track_count)
        return frame

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write_frame(self, frame: np.ndarray) -> None:
        self.output.write(frame)

    def save_body_crop(self, global_id: str, crop: np.ndarray, timestamp: str) -> str:
        os.makedirs(f"data_analysis/body_crop/{global_id}/", exist_ok=True)
        path = (
            f"data_analysis/body_crop/{global_id}/"
            f"{timestamp.replace('.', '_')}_id_{global_id}.jpg"
        )
        cv2.imwrite(path, crop)
        return path

    def cleanup(self) -> None:
        """Release resources (idempotent)."""
        try:
            self.source.release()
        except Exception:
            pass
        try:
            self.output.release()
        except Exception:
            pass
