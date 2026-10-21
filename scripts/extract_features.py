"""Measure every frame of each new clip and cache the results.

Per frame this stores the slider values read from the panel, the ArcFace
embedding, and the head yaw and pitch from MediaPipe. Later steps then run on
these files in seconds instead of decoding video.

Clips live in head folders (see face_tweaker.layout). Only clips without a
cache file are extracted, so earlier extractions are reused. A clip where the
slider panel scrolls is rejected and gets no file, because its rows are then
not the first 9 sliders, so it is reported again on every run until removed.

Usage: uv run scripts/extract_features.py <recordings_dir> <features_dir>
Writes <features_dir>/HeadNN/<clip name>.npz per accepted clip.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import cv2
import numpy as np

from face_tweaker.division2_sliders import SLIDER_NAMES
from face_tweaker.division2_sliders import PanelScrolledError
from face_tweaker.division2_sliders import read_sliders
from face_tweaker.face_features import ArcFace
from face_tweaker.face_features import Landmarks
from face_tweaker.layout import head_folders

# Clips extracted at once. One process takes about 30 ms per frame, and four in
# parallel take about 43 ms each, so four give about 2.8 times the throughput.
WORKERS = 4


def extract(video: Path, out: Path) -> str:
    """Write the per-frame file and return a summary line. Raises
    PanelScrolledError if the panel scrolls."""
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
    return f"{frame_no} frames, {1000 * seconds / frame_no:.1f} ms per frame, face in {arrays['face'].sum()}"


def main():
    recordings = Path(sys.argv[1])
    features = Path(sys.argv[2])
    loose = list(recordings.glob("*.mp4"))
    if loose:
        print(f"{len(loose)} clips sit outside a head folder and are skipped")

    jobs = []
    cached = 0
    for _, folder in head_folders(recordings):
        for video in sorted(folder.glob("*.mp4")):
            out = features / folder.name / (video.stem + ".npz")
            if out.exists():
                cached += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((video, out))
    print(f"{len(jobs)} clips to extract, {cached} already cached")

    rejected = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(extract, video, out): video for video, out in jobs}
        for future in as_completed(futures):
            video = futures[future]
            label = f"{video.parent.name}/{video.name}"
            try:
                print(f"{label}: {future.result()}")
            except PanelScrolledError as e:
                print(f"{label}: rejected: {e}")
                rejected += 1
    if rejected:
        sys.exit(1)


if __name__ == "__main__":
    main()
