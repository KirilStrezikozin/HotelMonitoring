from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

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
    ) -> list[dict]:
        """
        Update tracker and assign global IDs using ReID.
        Returns a list of dictionaries containing bbox and global_id for drawing.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        used_gids: set[str] = set()

        frame_h, frame_w = frame.shape[:2]

        render_data = []

        for track in tracks:
            try:
                local_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())

                # Clamp coordinates to the frame boundaries
                l, t = max(0, l), max(0, t)
                r, b = min(frame_w, r), min(frame_h, b)

                if r <= l or b <= t:
                    continue

                crop = frame[t:b, l:r]
                vertical = (r - l) / (b - t) > FrameProcessor.MAX_VERTICAL_RATIO

                if (r - l) * (b - t) <= FrameProcessor.MIN_BOX_AREA or vertical:
                    continue

                current_gid = self.track_to_global.get(local_id)
                assigned_gid = current_gid

                if frame_count % detection_interval == 0:
                    embedding = reid_model.extract_embedding(crop)
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

        return render_data
