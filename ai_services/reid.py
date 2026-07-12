import uuid
import time
from collections import deque
from typing import Deque

import torch
import numpy as np
import torchreid
from scipy.spatial.distance import cosine
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config.tracking import TrackingConfig
from schema.embedding import EmbeddingEntry

tracking_config = TrackingConfig()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ReIDModel:
    """Improved ReID model with consistent ID assignment and threshold management."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._create_transform()
        self.embedding_db: dict[str, Deque[EmbeddingEntry]] = {}
        self.threshold = tracking_config.reid_threshold
        self.buffer_size = tracking_config.embedding_buffer_size

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the ReID model."""
        model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )

        torchreid.utils.load_pretrained_weights(model, self.model_path)
        model.eval()
        model.to(device)
        return model

    def _create_transform(self) -> transforms.Compose:
        """Create image transformation pipeline."""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract embedding for a single image crop."""
        with torch.no_grad():
            input_tensor = self.transform(crop).unsqueeze(0).to(device)
            feature = self.model(input_tensor)
            normalized_feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            return normalized_feature.cpu().numpy().flatten()

    def extract_embeddings_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """Extract embeddings for a list of image crops in a single forward pass."""
        if not crops:
            return []
        with torch.no_grad():
            tensors = [self.transform(crop) for crop in crops]
            input_tensor = torch.stack(tensors).to(device)
            features = self.model(input_tensor)
            normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
            return [feat for feat in normalized_features.cpu().numpy()]

    def assign_global_id(
        self,
        embedding: np.ndarray,
        camera_id: int,
        current_id: str,
        active_ids: set[str],
    ) -> str:
        """
        Assign a global ID to an embedding, prioritizing ID persistence.
        
        Strategy:
        1. If current_id exists and is a strong match, reuse it
        2. Otherwise, find best matching ID from database
        3. If no match found, create new identity
        """
        candidates = self._find_best_match(embedding, camera_id, current_id)

        # Strategy 1: If we have a current_id and it's in the candidates, prefer it
        if current_id is not None:
            for best_gid, dist in candidates:
                if best_gid == current_id:
                    self._update_embedding_buffer(best_gid, embedding, camera_id)
                    return best_gid

        # Strategy 2: Use the best matching ID if not already active in this frame
        for best_gid, dist in candidates:
            if best_gid not in active_ids:
                self._update_embedding_buffer(best_gid, embedding, camera_id)
                return best_gid

        # Strategy 3: No match found, create new identity
        new_id = self._create_new_identity(embedding, camera_id)
        return new_id

    def _find_best_match(
        self, embedding: np.ndarray, camera_id: int, current_id: str
    ) -> list[tuple[str, float]]:
        """Find best matching identities for an embedding.
        
        Returns a sorted list of (gid, distance) tuples, sorted by distance (closest first).
        If current_id is valid and matches, it will be included in candidates.
        """
        threshold = self.threshold
        candidates: list[tuple[str, float]] = []

        # Search all known identities for similarity matches
        for gid, embedding_buffer in self.embedding_db.items():
            avg_embedding = np.mean([e.embedding for e in embedding_buffer], axis=0)
            distance = cosine(avg_embedding, embedding)

            if distance < threshold:
                candidates.append((gid, distance))

        candidates.sort(key=lambda x: x[1])
        
        # Fallback: If no candidates but current_id is valid and in database,
        # add it as a fallback even if it's below threshold (ID persistence)
        if not candidates and current_id is not None and current_id in self.embedding_db:
            avg_emb = np.mean(
                [e.embedding for e in self.embedding_db[current_id]], axis=0
            )
            best_distance = cosine(avg_emb, embedding)
            candidates.append((current_id, best_distance))

        return candidates

    def _update_embedding_buffer(
        self, gid: str, embedding: np.ndarray, camera_id: int
    ) -> None:
        """Update the embedding buffer for a given global ID."""
        entry = EmbeddingEntry(
            embedding=embedding, timestamp=time.time(), camera_id=camera_id
        )
        if len(self.embedding_db[gid]) >= self.buffer_size:
            self.embedding_db[gid].popleft()
        self.embedding_db[gid].append(entry)

    def _create_new_identity(self, embedding: np.ndarray, camera_id: int) -> str:
        """Create a new global identity."""
        new_gid = str(uuid.uuid4())[:8]
        entry = EmbeddingEntry(
            embedding=embedding, timestamp=time.time(), camera_id=camera_id
        )
        self.embedding_db[new_gid] = deque([entry], maxlen=self.buffer_size)
        return new_gid
