import uuid
import cv2
import time
from collections import deque
from typing import Deque

import torch
import numpy as np
import torchreid


from config.tracking import TrackingConfig
from schema.embedding import EmbeddingEntry
from db.analytics import AnalyticsDB

tracking_config = TrackingConfig()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ImageNet normalisation constants as tensors (computed once)
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)


class ReIDModel:
    """Improved ReID model with consistent ID assignment and threshold management."""

    def __init__(self, model_path: str, db: AnalyticsDB | None = None):
        self.model_path = model_path
        self.db = db
        self.model = self._load_model()
        self.threshold = tracking_config.reid_threshold
        self.buffer_size = tracking_config.embedding_buffer_size
        self.ttl = tracking_config.embedding_ttl_seconds

        # global_id -> deque[EmbeddingEntry]
        self.embedding_db: dict[str, Deque[EmbeddingEntry]] = {}
        # global_id -> cached mean np.ndarray  (invalidated on buffer update)
        self._mean_cache: dict[str, np.ndarray] = {}
        # staff IDs are never expired by TTL
        self.staff_ids: set[str] = set()

        # Restore identities from DB (survives restarts)
        self._load_embeddings_from_db()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the ReID model."""
        model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )

        torchreid.utils.load_pretrained_weights(model, self.model_path)
        model.eval()
        model.to(device)
        return model

    def _load_embeddings_from_db(self) -> None:
        """Restore embedding buffers from DB on startup (skips stale entries)."""
        if not self.db:
            return
        min_ts = (time.time() - self.ttl) if self.ttl > 0 else 0.0
        stored = self.db.load_embeddings(min_timestamp=min_ts)
        for gid, entries in stored.items():
            # Keep only the newest buffer_size entries
            entries = entries[-self.buffer_size :]
            buf: Deque[EmbeddingEntry] = deque(maxlen=self.buffer_size)
            for emb, ts, cam_id in entries:
                buf.append(
                    EmbeddingEntry(embedding=emb, timestamp=ts, camera_id=cam_id)
                )
            self.embedding_db[gid] = buf
        if stored:
            print(f"[ReID] Restored {len(stored)} identities from DB")
        if self.db:
            self.staff_ids = self.db.get_staff_ids()
            if self.staff_ids:
                print(
                    f"[ReID] {len(self.staff_ids)} staff member(s) loaded: {self.staff_ids}"
                )

    # ------------------------------------------------------------------
    # Embedding extraction  (OpenCV pipeline, no PIL)
    # ------------------------------------------------------------------

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract L2-normalised feature vector from a BGR crop."""
        # Resize with OpenCV (much faster than PIL BICUBIC for small crops)
        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        # ImageNet normalisation
        tensor = (tensor.to(device) - _MEAN) / _STD
        input_batch = tensor.unsqueeze(0)

        with torch.no_grad():
            feature = self.model(input_batch)
            normalised = torch.nn.functional.normalize(feature, p=2, dim=1)
        return normalised.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # ID assignment
    # ------------------------------------------------------------------

    def assign_global_id(
        self,
        embedding: np.ndarray,
        camera_id: int,
        current_id: str | None,
        active_ids: set[str],
        logger=None,
    ) -> str:
        self._cleanup_stale_identities()
        # Tracks re-appearing after a drop (current_id=None) get a looser threshold
        # so they can match their previous identity even with slight appearance change.
        candidates = self._find_best_match(
            embedding, camera_id, current_id, loose=current_id is None
        )

        for best_gid, dist in candidates:
            # Same track as before — keep the ID
            if current_id == best_gid:
                self._update_embedding_buffer(best_gid, embedding, camera_id)
                if logger:
                    logger.info(
                        {
                            "event": "reuse_global_id",
                            "camera_id": camera_id,
                            "global_id": best_gid,
                            "distance": float(dist),
                        }
                    )
                return best_gid

            # Candidate is free — assign it
            if best_gid not in active_ids:
                self._update_embedding_buffer(best_gid, embedding, camera_id)
                if logger:
                    logger.info(
                        {
                            "event": "match_found",
                            "camera_id": camera_id,
                            "global_id": best_gid,
                            "distance": float(dist),
                        }
                    )
                return best_gid

            # Candidate already taken by another track this frame — try next
            if logger:
                logger.info(
                    {
                        "event": "id_conflict",
                        "camera_id": camera_id,
                        "candidate_global_id": best_gid,
                        "distance": float(dist),
                        "note": "Candidate already taken — trying next",
                    }
                )

        new_id = self._create_new_identity(embedding, camera_id)
        if logger:
            logger.info(
                {
                    "event": "new_identity",
                    "camera_id": camera_id,
                    "global_id": new_id,
                    "note": "No valid match found",
                }
            )
        return new_id

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _find_best_match(
        self,
        embedding: np.ndarray,
        camera_id: int,
        current_id: str,
        loose: bool = False,
    ) -> list[tuple[str, float]]:
        # loose=True when track has no prior ID (re-appeared after drop):
        # use 1.3× threshold so slight appearance changes still match.
        base = self.threshold * 1.3 if loose else self.threshold
        now = time.time()
        candidates: list[tuple[str, float]] = []

        for gid, buf in self.embedding_db.items():
            avg = self._get_mean_embedding(gid)

            # Cosine distance for L2-normalised vectors = 1 - dot product
            distance = 1.0 - float(np.dot(avg, embedding))

            last_entry = buf[-1]
            if last_entry.camera_id != camera_id:
                # Person was last seen on a different camera less than 1s ago
                if (now - last_entry.timestamp) < 1.0:
                    continue
                threshold = base
            else:
                # Same camera → tighter threshold
                threshold = base * 0.9

            if distance < threshold:
                candidates.append((gid, distance))

        # If no candidate found but current_id exists, keep it regardless of distance
        if not candidates and current_id and current_id in self.embedding_db:
            avg = self._get_mean_embedding(current_id)
            distance = 1.0 - float(np.dot(avg, embedding))
            candidates.append((current_id, distance))

        candidates.sort(key=lambda x: x[1])
        return candidates

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _update_embedding_buffer(
        self, gid: str, embedding: np.ndarray, camera_id: int
    ) -> None:
        # Staff embeddings are fixed at registration — never overwrite them
        # so that live video noise cannot corrupt the reference buffer.
        if gid in self.staff_ids:
            return

        ts = time.time()
        entry = EmbeddingEntry(embedding=embedding, timestamp=ts, camera_id=camera_id)
        buf = self.embedding_db[gid]
        buf.append(entry)  # deque(maxlen=N) auto-removes oldest on overflow
        # Invalidate cached mean
        self._mean_cache.pop(gid, None)
        # Persist to DB and trim old rows
        if self.db:
            self.db.save_embedding(gid, embedding, camera_id, ts)
            self.db.prune_old_embeddings(gid, self.buffer_size)

    def _create_new_identity(self, embedding: np.ndarray, camera_id: int) -> str:
        ts = time.time()
        new_gid = str(uuid.uuid4())[:8]
        entry = EmbeddingEntry(embedding=embedding, timestamp=ts, camera_id=camera_id)
        self.embedding_db[new_gid] = deque([entry], maxlen=self.buffer_size)
        self._mean_cache.pop(new_gid, None)
        # Persist to DB
        if self.db:
            self.db.save_embedding(new_gid, embedding, camera_id, ts)
        return new_gid

    def _get_mean_embedding(self, gid: str) -> np.ndarray:
        """Return cached mean embedding, computing it only when cache is cold."""
        if gid not in self._mean_cache:
            buf = self.embedding_db[gid]
            self._mean_cache[gid] = np.mean([e.embedding for e in buf], axis=0)
        return self._mean_cache[gid]

    def _cleanup_stale_identities(self) -> None:
        """Remove identities that haven't been seen for longer than TTL."""
        if self.ttl <= 0:
            return
        now = time.time()
        stale = [
            gid
            for gid, buf in self.embedding_db.items()
            if gid not in self.staff_ids  # staff embeddings never expire
            and (now - buf[-1].timestamp) > self.ttl
        ]
        for gid in stale:
            del self.embedding_db[gid]
            self._mean_cache.pop(gid, None)
            if self.db:
                self.db.delete_identity_embeddings(gid)


class ReIDTracker:
    """
    Orchestrates per-frame ReID across all tracks of a single camera.

    Responsibilities:
        - extract embeddings from track crops
        - assign / maintain global IDs via ReIDModel
        - own the track_id → global_id mapping

    CameraProcessor passes skip_ids (elevator, edge-of-frame) so this class
    stays camera-agnostic.
    """

    def __init__(self, model: ReIDModel):
        self.model = model
        self.track_to_global: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Main entry point (called once per frame)
    # ------------------------------------------------------------------

    def process_tracks(
        self,
        tracks: list,
        frame: np.ndarray,
        camera_id: int,
        is_detection_frame: bool,
        skip_ids: set[int],
        min_box_area: int,
        logger=None,
    ) -> None:
        """
        Extract embeddings and assign global IDs for all eligible tracks.
        Updates track_to_global in-place; call get_global_id() afterwards.

        Args:
            tracks:             DeepSort track list for this frame
            frame:              current BGR frame
            camera_id:          camera identifier
            is_detection_frame: run ReID only on detection frames
            skip_ids:           track IDs to exclude (elevator / edge conditions)
            min_box_area:       minimum crop area to process
            logger:             optional logger
        """
        frame_h, frame_w = frame.shape[:2]
        used_gids: set[str] = set(self.track_to_global.values())

        for track in tracks:
            local_id = track.track_id
            current_gid = self.track_to_global.get(local_id)

            # Run ReID if:
            #   a) it's a regular detection frame (periodic update for all tracks), OR
            #   b) this track has no global_id yet — run every frame so a re-appeared
            #      person gets re-identified immediately instead of waiting N frames.
            if not is_detection_frame and current_gid is not None:
                continue

            if local_id in skip_ids:
                continue

            l, t, r, b = map(int, track.to_ltrb())
            l, t = max(0, l), max(0, t)
            r, b = min(frame_w, r), min(frame_h, b)

            if r <= l or b <= t:
                continue
            if (r - l) * (b - t) <= min_box_area:
                continue

            crop = frame[t:b, l:r]

            try:
                embedding = self.model.extract_embedding(crop)
                assigned_gid = self.model.assign_global_id(
                    embedding, camera_id, current_gid, used_gids, logger
                )
                if assigned_gid:
                    # Resolve conflict: ID already taken by another track this frame
                    if assigned_gid in used_gids and assigned_gid != current_gid:
                        assigned_gid = self.model._create_new_identity(
                            embedding, camera_id
                        )
                    self.track_to_global[local_id] = assigned_gid
                    used_gids.add(assigned_gid)

            except Exception as e:
                if logger:
                    logger.error(
                        {"event": "reid_error", "track_id": local_id, "error": str(e)}
                    )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_global_id(self, track_id: int) -> str | None:
        return self.track_to_global.get(track_id)

    def cleanup_track(self, track_id: int) -> None:
        """Call when DeepSort drops a track."""
        self.track_to_global.pop(track_id, None)
