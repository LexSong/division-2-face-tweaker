"""Check each new clip, then measure every frame of the clips that pass and
cache the results.

Per frame this stores the slider values read from the panel, the ArcFace
embedding, and the head yaw and pitch from MediaPipe. Later steps then run on
these files in seconds instead of decoding video.

Clips live in head folders (see face_tweaker.layout). A new clip is one
without a cache file, or one sitting outside a head folder. Earlier
extractions are reused.

The check comes first and reads three frames per clip, so it reports in
seconds, before minutes of extraction. It prints each clip's length and the
slider values half a second after the start, in the middle, and half a second
before the end. A clip passes only if all three agree, because the sliders
must not move while recording. A clip that fails is skipped and gets no file,
so it is reported again on every run until fixed or removed.

A clip where the slider panel scrolls is rejected, because its rows are then
not the first 9 sliders. The check catches this if it shows in one of its
three frames, and extraction catches it in any other frame.

Usage: uv run scripts/extract_features.py <recordings_dir> <features_dir>
Writes <features_dir>/HeadNN/<clip name>.npz per accepted clip.
Exits with code 1 if any clip has a problem.
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

# Checks sit this many frames in from each end, half a second at 60 fps, so
# the press that starts or stops the recording does not show.
EDGE_FRAMES = 30

# A clip should cover at least one idle loop, which lasts about 17.6 s.
MIN_SECONDS = 17.0


def sliders_at(cap: cv2.VideoCapture, frame_no: int) -> str:
    """The slider values in one frame, or the reason they cannot be read."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        return "unreadable frame"
    try:
        reading = read_sliders(frame)
    except PanelScrolledError:
        return "panel scrolled"
    if not reading.ok:
        return "slider unclear"
    return " ".join(str(v) for v in reading.values)


def new_clips(recordings: Path, features: Path) -> list:
    """(head folder name or None, video) for each clip not yet extracted."""
    clips = [(None, video) for video in sorted(recordings.glob("*.mp4"))]
    for _, folder in head_folders(recordings):
        for video in sorted(folder.glob("*.mp4")):
            if not (features / folder.name / (video.stem + ".npz")).exists():
                clips.append((folder.name, video))
    return clips


def check(folder: str | None, video: Path) -> bool:
    """Print one line on the clip and return whether it passes."""
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = frames / cap.get(cv2.CAP_PROP_FPS)
    start = sliders_at(cap, EDGE_FRAMES)
    middle = sliders_at(cap, frames // 2)
    end = sliders_at(cap, max(frames - EDGE_FRAMES, 0))
    cap.release()

    issues = []
    if folder is None:
        folder = "no head folder"
        issues.append("not in a head folder")
    if seconds < MIN_SECONDS:
        issues.append("shorter than one idle loop")
    if not (start == middle == end):
        issues.append("sliders change")
    for reading in (start, middle, end):
        if not reading[0].isdigit():
            issues.append(reading)
            break
    if issues:
        verdict = "PROBLEM: " + ", ".join(issues)
    else:
        verdict = "ok"
    print(f"{folder} | {video.name} | {seconds:5.1f} s | {start} | {middle} | {end} | {verdict}")
    return not issues


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
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, fps=fps, slider_names=np.array(SLIDER_NAMES), **arrays)
    seconds = time.perf_counter() - start
    return f"{frame_no} frames, {1000 * seconds / frame_no:.1f} ms per frame, face in {arrays['face'].sum()}"


def main():
    recordings = Path(sys.argv[1])
    features = Path(sys.argv[2])
    clips = new_clips(recordings, features)
    print(f"{len(clips)} new clips")

    jobs = []
    problems = 0
    for folder, video in clips:
        if check(folder, video):
            jobs.append((video, features / folder / (video.stem + ".npz")))
        else:
            problems += 1
    if problems:
        print(f"{problems} clips with a problem are skipped")
    print(f"{len(jobs)} clips to extract")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(extract, video, out): video for video, out in jobs}
        for future in as_completed(futures):
            video = futures[future]
            label = f"{video.parent.name}/{video.name}"
            try:
                print(f"{label}: {future.result()}")
            except PanelScrolledError as e:
                print(f"{label}: rejected: {e}")
                problems += 1
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
