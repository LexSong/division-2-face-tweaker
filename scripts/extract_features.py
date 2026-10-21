"""Measure every frame of each clip and cache the results.

Per frame this stores the slider values read from the panel, the ArcFace
embedding, and the head yaw and pitch from MediaPipe. Later steps then run on
these files in seconds instead of decoding video.

A clip where the slider panel scrolls is rejected and gets no file, because
its rows are then not the first 9 sliders.

Usage: uv run scripts/extract_features.py <out_dir> <video>...
Writes <out_dir>/<video name>.npz per accepted video.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_tweaker.division2_sliders import SLIDER_NAMES
from face_tweaker.division2_sliders import PanelScrolledError
from face_tweaker.division2_sliders import read_sliders
from face_tweaker.face_features import ArcFace
from face_tweaker.face_features import Landmarks


def extract(video: Path, out: Path) -> None:
    """Write the per-frame file. Raises PanelScrolledError if the panel scrolls."""
    landmarks = Landmarks()
    arcface = ArcFace()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = {
        "sliders": [],
        "ok": [],
        "selected": [],
        "face": [],
        "arcface": [],
        "yaw": [],
        "pitch": [],
    }
    start = time.perf_counter()
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            reading = read_sliders(frame)
            values = []
            for v in reading.values:
                if v is None:
                    values.append(-1)
                else:
                    values.append(v)
            rows["sliders"].append(values)
            rows["ok"].append(reading.ok)
            if reading.selected is None:
                rows["selected"].append(-1)
            else:
                rows["selected"].append(reading.selected)

            found = landmarks.detect(frame, int(frame_no * 1000 / fps))
            if found is None:
                rows["face"].append(False)
                rows["arcface"].append(np.full(512, np.nan, np.float32))
                rows["yaw"].append(np.nan)
                rows["pitch"].append(np.nan)
            else:
                points, yaw, pitch = found
                rows["face"].append(True)
                rows["arcface"].append(arcface.embed(frame, points))
                rows["yaw"].append(yaw)
                rows["pitch"].append(pitch)
            frame_no += 1
    finally:
        cap.release()
        landmarks.close()

    arrays = {key: np.array(value) for key, value in rows.items()}
    np.savez(out, fps=fps, slider_names=np.array(SLIDER_NAMES), **arrays)
    seconds = time.perf_counter() - start
    print(
        f"{video.name}: {frame_no} frames, {1000 * seconds / frame_no:.1f} ms per frame, face in {arrays['face'].sum()}"
    )


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected = 0
    for name in sys.argv[2:]:
        video = Path(name)
        try:
            extract(video, out_dir / (video.stem + ".npz"))
        except PanelScrolledError as e:
            print(f"{video.name}: rejected: {e}")
            rejected += 1
    if rejected:
        sys.exit(1)


if __name__ == "__main__":
    main()
