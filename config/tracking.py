from config.base import BaseConfig


class TrackingConfig(BaseConfig):
    reid_threshold: float = 0.3
    max_age: int = 40
    embedding_buffer_size: int = 10
    min_crop_height: int = 50
    min_crop_width: int = 25
    conf_threshold: float = 0.35  # YOLO detection confidence threshold
    yolo_imgsz: int = 640  # must match CoreML export imgsz
    embedding_ttl_seconds: float = (
        300.0  # drop identity from DB after N seconds of absence
    )
