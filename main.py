import time

from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig
import logging
import torch
import sys

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(
    *args, **{**kwargs, "weights_only": False}
)

sys.modules["numpy._core"] = sys.modules["numpy.core"]
sys.modules["numpy._core.multiarray"] = sys.modules["numpy.core.multiarray"]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the tracking system."""

    # Configuration

    reid_model_path = "models/model.pth.tar-40"

    camera_configs = [
        # CameraConfig(camera_id=1, video_path='15546948_1080_1920_50fps.mp4'),
        # CameraConfig(camera_id=2, video_path="VIRAT_S_050201_05_000890_000944.mp4"),
        # CameraConfig(camera_id=3, video_path="VIRAT_S_010204_05_000856_000890.mp4"),
        CameraConfig(camera_id=3, video_path="videos/yard_test.mp4"),
        # CameraConfig(camera_id=4, stream_url="rtsp://admin:tEsTgfhjkm1729@192.168.12.20:554/cam/realmonitor?channel=1&subtype=1", output_url="rtsp://mediamtx:8554/cam/reception"),
    ]

    print("--- Phase 1: Loading Models & Environments ---")
    tracker = MultiCameraTracker(reid_model_path, camera_configs)

    print("\n--- Phase 2: Processing Video Streams ---")
    start_time = time.time()

    tracker.run()

    end_time = time.time()
    print(
        f"\nProcessing complete. Pure execution time: {end_time - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
