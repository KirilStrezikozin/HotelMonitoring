import cv2
import numpy as np


class FrameProcessor:
    """Preprocesses frames and draws annotations."""

    MIN_BOX_AREA = 5
    MAX_VERTICAL_RATIO = 1.6

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction and enhance brightness/contrast."""
        # frame = cv2.bilateralFilter(frame, 5, 75, 75)
        # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        return frame

    @staticmethod
    def annotate(
        frame: np.ndarray,
        l: int,
        t: int,
        r: int,
        b: int,
        global_id: str,
        is_staff: bool = False,
    ):
        """Draw annotation. Staff: box + name. Clients: text only, no box."""
        if is_staff:
            color = (0, 200, 255)  # amber
            cv2.rectangle(frame, (l, t), (r, b), color, 1)
            cv2.putText(
                frame,
                global_id,
                (l, t - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        else:
            # Client — only the word "client" centered above the detection, no box
            cx = (l + r) // 2
            (tw, _), _ = cv2.getTextSize("client", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(
                frame,
                "client",
                (cx - tw // 2, t - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 0),
                1,
            )

    @staticmethod
    def draw_person_count(frame: np.ndarray, count: int):
        """Draw the total person count in the top right corner."""
        height, width = frame.shape[:2]
        text = f"People: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x = width - text_width - 20
        y = text_height + 20

        cv2.rectangle(
            frame,
            (x - 10, y - text_height - 10),
            (x + text_width + 10, y + 10),
            (0, 0, 0),
            -1,
        )

        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
