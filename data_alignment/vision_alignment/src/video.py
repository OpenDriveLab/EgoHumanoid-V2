"""Video decoding shared by geometry preparation and validation."""

from pathlib import Path

import cv2


def iter_video_rgb(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {index}/{count}")
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
