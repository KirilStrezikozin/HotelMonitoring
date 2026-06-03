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
    """Wraps YOLO person detection with a tight 4-sided ROI crop for a true zoom effect."""

    CONF_THRESHOLD = 0.15

    # Define our tight bounding box around the actual street
    TOP_CROP = 0.35  # Remove sky/clock tower
    BOTTOM_CROP = 0.0  # Keep the immediate foreground
    LEFT_CROP = 0.15  # Remove left buildings
    RIGHT_CROP = 0.15  # Remove right buildings

    def __init__(self, detector: YOLO):
        self.detector = detector

    def detect(self, frame: np.ndarray) -> list[tuple[list[int], float, str]]:
        """Detect persons inside a focused geographic zone."""
        h_orig, w_orig = frame.shape[:2]

        # 1. Calculate pixel boundaries for the crop
        y_start = int(h_orig * self.TOP_CROP)
        y_end = int(h_orig * (1.0 - self.BOTTOM_CROP))
        x_start = int(w_orig * self.LEFT_CROP)
        x_end = int(w_orig * (1.0 - self.RIGHT_CROP))

        # 2. Slice the frame tightly around the crowd
        roi_frame = frame[y_start:y_end, x_start:x_end]

        # 3. Run inference (YOLO will now scale this smaller box up to 1088!)
        results = self.detector(roi_frame, imgsz=1088, iou=0.6)
        detections = []

        for box, cls, conf in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        ):
            if int(cls) != 1 or float(conf) < self.CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)

            # 4. CRITICAL: Translate coordinates back to the full 1080p frame space
            x1 += x_start
            x2 += x_start
            y1 += y_start
            y2 += y_start

            w, h = x2 - x1, y2 - y1

            if (
                w * h > FrameProcessor.MIN_BOX_AREA
                and w / h <= FrameProcessor.MAX_VERTICAL_RATIO
            ):
                detections.append(([x1, y1, w, h], float(conf), "person"))

        return detections
