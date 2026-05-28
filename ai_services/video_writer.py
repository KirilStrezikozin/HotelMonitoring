import subprocess

import cv2
import numpy as np


class VideoOutput:
    """Handles writing frames to a file or RTSP stream."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: str | None = None,
        stream_url: str | None = None,
    ):
        self.stream = False
        if output_path:
            self.writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        elif stream_url:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(int(fps)),
                "-i",
                "-",
                "-c:v",
                "h264_videotoolbox",
                "-b:v",
                "4000k",
                "-tune",
                "zerolatency",
                "-preset",
                "superfast",
                "-f",
                "rtsp",
                stream_url,
            ]
            self.writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            self.stream = True
        else:
            raise ValueError("Either output_path or stream_url must be provided.")

    def write(self, frame: np.ndarray):
        if self.stream:
            self.writer.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def release(self):
        if self.stream:
            self.writer.stdin.close()
            self.writer.wait()
        else:
            self.writer.release()
