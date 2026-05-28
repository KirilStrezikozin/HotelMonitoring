import numpy as np
from ultralytics import YOLO
import torch

from ai_services.frame_processor import FrameProcessor

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


class PersonDetector:
    """Wraps YOLO person detection."""

    CONF_THRESHOLD = 0.3

    def __init__(self, detector: YOLO):
        self.detector = detector

    def detect(self, frame: np.ndarray) -> list[tuple[list[int], float, str]]:
        """
        Detect persons in a frame.

        Returns:
            List of tuples: ([x, y, w, h], confidence, "person")
        """
        results = self.detector(frame, imgsz=1088, iou=0.6)
        detections = []

        for box, cls, conf in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        ):
            if int(cls) != 1 or float(conf) < self.CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if (
                w * h > FrameProcessor.MIN_BOX_AREA
                and w / h <= FrameProcessor.MAX_VERTICAL_RATIO
            ):
                detections.append(([x1, y1, w, h], float(conf), "person"))

        return detections
