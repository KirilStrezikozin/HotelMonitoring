import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from config.tracking import TrackingConfig


class TrackerManager:
    """
    Thin wrapper around DeepSort.

    Responsibility: tracking only — no ReID, no annotation.
    ReID and events are handled by CameraProcessor.
    """

    def __init__(self):
        config = TrackingConfig()
        self.tracker = DeepSort(
            max_age=config.max_age,  # 60 — трек живе довше при зникненні
            max_iou_distance=0.7,  # 0.5→0.7 — гнучкіше зіставлення при русі
            n_init=3,  # 5→3 — трек підтверджується вже на 3-му кадрі
            max_cosine_distance=0.4,  # 0.2→0.3 — м'якше appearance matching
        )

    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Feed detections into DeepSort and return updated tracks.

        Args:
            frame:      current frame (used by DeepSort for appearance features)
            detections: list of ([x, y, w, h], conf, "person") tuples

        Returns:
            List of DeepSort Track objects.
        """
        return self.tracker.update_tracks(detections, frame=frame)
