from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import time

from config.tracking import TrackingConfig
from ai_services.reid import ReIDModel
from ai_services.frame_processor import FrameProcessor


class TrackerManager:
    """Manages DeepSort tracking and ReID assignments. Returns data for rendering."""

    def __init__(self):
        tracking_config = TrackingConfig()
        self.tracker = DeepSort(
            max_age=tracking_config.max_age,
            max_iou_distance=0.8,
            n_init=2,
            max_cosine_distance=0.2,
            embedder=None,
            # max_age=5,
            # max_iou_distance=0.5,
            # n_init=1,
            # max_cosine_distance=0.2,
        )
        self.track_to_global: dict[int, str] = {}

    def update(
        self,
        frame: np.ndarray,
        detections: list,
        reid_model: ReIDModel,
        frame_count: int,
        detection_interval: int,
        camera_id: str,
    ) -> dict:
        """
        Update tracker with pre-computed batched ReID embeddings and assign global IDs.
        """
        frame_h, frame_w = frame.shape[:2]

        detection_crops = []
        valid_detections = []

        for det in detections:
            bbox, conf, cl = det
            x, y, w, h = map(int, bbox)
            l, t, r, b = x, y, x + w, y + h
            l, t = max(0, l), max(0, t)
            r, b = min(frame_w, r), min(frame_h, b)

            if r <= l or b <= t:
                continue

            detection_crops.append(frame[t:b, l:r])
            valid_detections.append(det)

        reid_start = time.time()
        detection_embeddings = []
        if detection_crops:
            detection_embeddings = reid_model.extract_embeddings_batch(detection_crops)
        reid_time = time.time() - reid_start

        tracking_start = time.time()

        tracks = self.tracker.update_tracks(
            valid_detections, frame=frame, embeds=detection_embeddings
        )
        tracking_elapsed = time.time() - tracking_start

        used_gids: set[str] = set()
        render_data = []

        is_interval_frame = frame_count % detection_interval == 0

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            try:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())

                l, t = max(0, l), max(0, t)
                r, b = min(frame_w, r), min(frame_h, b)

                if r <= l or b <= t:
                    continue

                vertical = (r - l) / (b - t) > FrameProcessor.MAX_VERTICAL_RATIO

                if (r - l) * (b - t) <= FrameProcessor.MIN_BOX_AREA or vertical:
                    continue

                current_gid = self.track_to_global.get(local_id)
                assigned_gid = current_gid

                if is_interval_frame:
                    if track.features:
                        embedding = track.features[-1]

                        assigned_gid = reid_model.assign_global_id(
                            embedding,
                            camera_id,
                            current_gid,
                            active_ids=used_gids,
                        )

                        if assigned_gid in used_gids:
                            assigned_gid = reid_model._create_new_identity(
                                embedding, camera_id
                            )

                        if assigned_gid:
                            self.track_to_global[local_id] = assigned_gid
                            used_gids.add(assigned_gid)

                if assigned_gid:
                    render_data.append(
                        {"bbox": (l, t, r, b), "global_id": assigned_gid}
                    )

            except Exception as e:
                print(f"Tracking error: {e}")
                continue

        return {
            "render_data": render_data,
            "tracking_time": tracking_elapsed,
            "reid_time": reid_time,
        }

    def update_no_embedding(
        self,
        frame: np.ndarray,
        detections: list,
        reid_model: ReIDModel,
        frame_count: int,
        detection_interval: int,
        camera_id: str,
    ) -> dict:
        """
        Update tracker using dummy embeddings to force math-only IoU tracking.
        Returns a list of dictionaries containing bbox and local track id.
        """

        frame_h, frame_w = frame.shape[:2]

        # Create non-zero dummy embeddings to prevent division-by-zero crashes
        dummy_embeds = np.ones((len(detections), 128)) if detections else None

        # Feed the dummy embeddings so the tracker relies solely on IoU and Kalman filters
        tracking_start = time.time()
        tracks = self.tracker.update_tracks(
            detections, frame=frame, embeds=dummy_embeds
        )
        tracking_elapsed = time.time() - tracking_start

        render_data = []

        for track in tracks:
            try:
                # Only render tracks that the Kalman filter has confirmed
                if not track.is_confirmed():
                    continue

                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())

                # Clamp coordinates to the frame boundaries
                l, t = max(0, l), max(0, t)
                r, b = min(frame_w, r), min(frame_h, b)

                if r <= l or b <= t:
                    continue

                vertical = (r - l) / (b - t) > FrameProcessor.MAX_VERTICAL_RATIO

                if (r - l) * (b - t) <= FrameProcessor.MIN_BOX_AREA or vertical:
                    continue

                # Pass the spatial tracker's local ID to the renderer
                render_data.append({"bbox": (l, t, r, b), "global_id": str(local_id)})

            except Exception as e:
                print(f"Tracking error: {e}")
                continue

        return {
            "render_data": render_data,
            "tracking_time": tracking_elapsed,
            "reid_time": 0,
        }

    def update_no_tracking(
        self,
        frame: np.ndarray,
        detections: list,
        reid_model: ReIDModel,
        frame_count: int,
        detection_interval: int,
        camera_id: str,
    ) -> dict:
        """
        Processes raw detections directly into render data without tracking.
        Returns a list of dictionaries containing bbox and a placeholder global_id.
        """
        frame_h, frame_w = frame.shape[:2]
        render_data = []

        for detection in detections:
            try:
                # PersonDetector returns: ([l, t, w, h], confidence, "person")
                bbox, conf, cls = detection
                x, y, w, h = bbox

                l, t = int(x), int(y)
                r, b = int(x + w), int(y + h)

                # Clamp coordinates to the frame boundaries
                l, t = max(0, l), max(0, t)
                r, b = min(frame_w, r), min(frame_h, b)

                if r <= l or b <= t:
                    continue

                vertical = (r - l) / (b - t) > FrameProcessor.MAX_VERTICAL_RATIO

                if (r - l) * (b - t) <= FrameProcessor.MIN_BOX_AREA or vertical:
                    continue

                # Pass the raw bounding box directly to the renderer
                render_data.append({"bbox": (l, t, r, b), "global_id": "Person"})

            except Exception as e:
                print(f"Tracking error: {e}")
                continue

        return {
            "render_data": render_data,
            "tracking_time": 0,
            "reid_time": 0,
        }
