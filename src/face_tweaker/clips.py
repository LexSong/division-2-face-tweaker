"""Pick the frames and clips the fit uses from one head's feature folder.

Each clip has one features file of per-frame measurements, written by
scripts/extract_features.py. A frame is good when its sliders were read and
its face counts for ArcFace. A clip counts when its sliders stay the same over
its good frames, and clip names do not matter.

A control clip is a clip with every slider at 6, recorded with the mouse
untouched. In it the face is fixed, so every change in ArcFace comes from the
idle animation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_VALUE = 6

# Frames with MediaPipe yaw beyond this are left out. Within about 10 degrees
# of the resting pose ArcFace barely moves, but past -20 degrees it drifts
# fast, down to a cosine of about 0.72 at the idle loop's turn to -40 degrees.
# On the control clips, limits from 15 to 30 degrees all give the steadiest
# clip averages, and 25 keeps 84% of frames.
YAW_LIMIT = 25.0

# A clip's average stands for whole idle loops, and frames with unread sliders
# tend to come in runs, so a clip that loses many would average over only parts
# of the loop. Clean clips read 100% of frames, and a cursor over the panel
# costs a few. A knob lost to compression once cost 75%.
MIN_SLIDERS_READ = 0.95


@dataclass
class SliderFeatures:
    """One slider setting and the features measured at it."""

    sliders: np.ndarray  # slider values, the same in every good frame
    arcface: np.ndarray  # ArcFace of the good frames
    feature_file: Path  # where it was loaded from, named after its clip; for printing only


def good_frames(d) -> np.ndarray:
    """Mask of the good frames in one clip's features: sliders read, and a
    face found turned no more than YAW_LIMIT."""
    return d["ok"] & d["face"] & (np.abs(d["yaw"]) <= YAW_LIMIT)


def load_feature_folder(folder: Path) -> list:
    """One SliderFeatures per clip in folder that counts."""
    samples = []
    for path in sorted(folder.glob("*.npz")):
        d = np.load(path)
        read = d["ok"].mean()
        if read < MIN_SLIDERS_READ:
            print(f"skipped {path.stem}: sliders read in only {read:.0%} of frames")
            continue
        good = good_frames(d)
        values = np.unique(d["sliders"][good], axis=0)
        if len(values) != 1:
            print(f"skipped {path.stem}: its sliders change during the clip")
            continue
        samples.append(SliderFeatures(values[0], d["arcface"][good].astype(np.float64), path))
    return samples


def control_frames(samples: list) -> np.ndarray:
    """ArcFace of the samples with every slider at 6, pooled."""
    frames = [s.arcface for s in samples if (s.sliders == BASE_VALUE).all()]
    if not frames:
        raise ValueError("no control clip")
    return np.concatenate(frames)
