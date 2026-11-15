"""Video reading helpers for stage 1 (PyAV, ffprobe and a PySceneDetect adapter)."""
import json
import logging
import subprocess
from pathlib import Path

import av
import numpy as np
from scenedetect.backends.pyav import VideoStreamAv

logger = logging.getLogger(__name__)


def open_video(video_path: Path | str) -> av.container.InputContainer:
    """Open a video for software decoding with PyAV."""
    return av.open(str(video_path))


def get_metadata(video_path: Path | str) -> dict:
    """Return ffprobe's JSON metadata (format, streams, chapters) for a video."""
    command = ["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", "-show_chapters", str(video_path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def sample_frames(video_path: Path | str, num_frames: int = 10) -> tuple[list[np.ndarray], int, int]:
    """Decode `num_frames` evenly spaced frames (RGB uint8), excluding the very start and end.

    Each frame is the first one decoded after seeking to its timestamp.

    Returns:
        (frames, width, height)
    """
    container = open_video(video_path)
    try:
        stream = container.streams.video[0]

        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = stream.duration * stream.time_base
        if duration is None or duration <= 0:
            duration = container.duration / av.time_base
        if duration is None or duration <= 0:
            if stream.frames > 0 and stream.average_rate > 0:
                duration = stream.frames / float(stream.average_rate)
            else:
                frame_count = sum(1 for _ in container.decode(stream))
                container.seek(0)
                duration = frame_count / float(stream.average_rate)

        if num_frames == 1:
            positions = [duration / 2]
        else:
            positions = np.linspace(0, duration, num_frames + 2)[1:-1]

        frames = []
        for pos in positions:
            container.seek(int(pos / stream.time_base), stream=stream)
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                break
    finally:
        container.close()

    if not frames:
        return [], 0, 0
    height, width, _ = frames[0].shape
    return frames, width, height


class ContainerVideoStream(VideoStreamAv):
    """PySceneDetect video stream built from an already-open PyAV container.

    PySceneDetect's own PyAV backend failed on some AV1 streams, so the container is opened by us and injected here.
    """

    def __init__(self, container, stream):  # noqa: super().__init__ is intentionally not called
        self._container = container
        self._stream = stream
        self._stream.thread_type = "AUTO"
        self._duration = self._get_duration()
        self._pix_fmt = self._stream.codec_context.format.name
        self._position = None
        self._frame = None
        self._frame_index = 0
        self._frame_buffer = []
        self._seek_mode = "accurate"
        self._reopened = True
        self._duration_frames = self._get_duration()

    @property
    def frame_rate(self):
        return float(self._stream.average_rate)

    @property
    def frame_number(self) -> int:
        """Current frame number; 0 until the first frame has been read."""
        if self._frame:
            return self.position.frame_num + 1
        return 0
