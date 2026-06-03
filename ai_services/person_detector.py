import numpy as np
import torch
from ultralytics import YOLO

from config.tracking import TrackingConfig

tracking_config = TrackingConfig()

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


class PersonDetector:
    """Wraps YOLO person-only detection with NMS deduplication."""

    def __init__(self, detector: YOLO, min_box_area: int, is_elevator: bool = False):
        self.detector = detector
        self.min_box_area = min_box_area
        self.is_elevator = is_elevator

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect persons in a frame.
        Returns list of ([x, y, w, h], confidence, "person") tuples.
        """
        results = self.detector.predict(
            frame,
            classes=[1],  # person only
            conf=tracking_config.conf_threshold,
            imgsz=tracking_config.yolo_imgsz,
            iou=0.3,  # агресивно прибирає дублікати ще всередині YOLO
            device=device,
            verbose=False,
        )

        detections: list[tuple] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        for i, box in enumerate(boxes.xyxy):
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            if h == 0:
                continue
            vertical = False if self.is_elevator else (w / h) > 1.6
            if w * h > self.min_box_area and not vertical:
                detections.append(([x1, y1, w, h], conf, "person"))

        # iou_threshold=0.35 — другий прохід NMS для відео з малою роздільністю
        return self._nms(detections, iou_threshold=0.35)

    # ------------------------------------------------------------------
    # NMS — extra deduplication pass on top of YOLO's built-in NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(a: list, b: list) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _nms(detections: list[tuple], iou_threshold: float = 0.8) -> list[tuple]:
        if len(detections) <= 1:
            return detections
        detections = sorted(detections, key=lambda d: d[1], reverse=True)
        kept = []
        suppressed = set()
        for i, det_i in enumerate(detections):
            if i in suppressed:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue
                if (
                    PersonDetector._compute_iou(det_i[0], detections[j][0])
                    > iou_threshold
                ):
                    suppressed.add(j)
        return kept
