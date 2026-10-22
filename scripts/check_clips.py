"""Quick check of new clips before extracting them.

A new clip is one without a cache file, or one sitting outside a head folder.
For each, this prints its length and the slider values half a second after
the start, in the middle, and half a second before the end. A clip is only
useful if all three agree: the sliders must not move while recording. It
reads three frames per clip, so it takes seconds, not minutes.

Usage: uv run scripts/check_clips.py <recordings_dir> <features_dir>
Exits with code 1 if any clip has a problem.
"""

import sys
from pathlib import Path

import cv2

from face_tweaker.division2_sliders import PanelScrolledError
from face_tweaker.division2_sliders import read_sliders
from face_tweaker.layout import head_folders

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


def main():
    clips = new_clips(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{len(clips)} new clips")
    problems = 0
    for folder, video in clips:
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
            problems += 1
            verdict = "PROBLEM: " + ", ".join(issues)
        else:
            verdict = "ok"
        print(f"{folder} | {video.name} | {seconds:5.1f} s | {start} | {middle} | {end} | {verdict}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
