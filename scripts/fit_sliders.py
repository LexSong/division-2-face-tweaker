"""Suggest slider values that make one head look like the target photos.

All clips in the head's feature folder are data points: the control clip with
every slider at 6, and clips with sliders moved. Each clip has one features
file of per-frame measurements, written by scripts/extract_features.py. A
frame is good when its sliders were read and its face counts for ArcFace. A
clip counts when its sliders stay the same over its good frames, and clip
names do not matter. Each clip that counts gives one point, its ArcFace
average over good frames. A local model then predicts ArcFace from the
sliders. Per slider it has a slope, plus a curvature where that slider was
recorded at 3 or more values, and the sliders are assumed to add up.

The suggestion maximizes the cosine similarity between the predicted ArcFace
and the target's, over each slider's full 0-12 range. Beyond the values
recorded for a slider, the model extrapolates, so those suggestions are
marked. Record the suggestion as a new clip in the same head folder: the next
run fits it in, which measures the model exactly where it was guessing.

The head needs a control clip: every slider at 6, recorded with the mouse
untouched. In it the face is fixed, so every change in ArcFace comes from the
idle animation.

Usage: uv run scripts/fit_sliders.py <features_dir> <head> <photo>...
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from bareface.arcface import ArcFace

from face_tweaker.division2_sliders import SLIDER_CENTER
from face_tweaker.division2_sliders import SLIDER_MAX
from face_tweaker.division2_sliders import SLIDER_MIN
from face_tweaker.division2_sliders import SLIDER_NAMES
from face_tweaker.face_features import FIVE_POINTS
from face_tweaker.face_features import Landmarks

# Frames with MediaPipe yaw beyond this are left out. Within about 10 degrees
# of the resting pose ArcFace barely moves, but past -20 degrees it drifts
# fast, down to a cosine of about 0.72 at the idle loop's turn to -40 degrees.
# On the control clips, limits from 15 to 30 degrees all give the steadiest
# clip averages, and 25 keeps 84% of frames.
YAW_LIMIT = 25.0

# A clip's average stands for whole idle loops, and frames with unread sliders
# tend to come in runs, so a clip that loses many would average over only parts
# of the loop. Clips without problems read 100% of frames, and a cursor over the panel
# costs a few. A knob lost to compression once cost 75%.
MIN_SLIDERS_READ = 0.95

# Idle animation axes removed from every vector before comparing. A head's
# first 30 axes hold about 87% of its idle animation, and removing more risks
# removing what makes the target recognizable.
IDLE_ANIMATION_AXES = 30


@dataclass
class SliderFeatures:
    """One slider setting and the features measured at it."""

    sliders: np.ndarray  # slider values, the same in every good frame
    arcface: np.ndarray  # ArcFace of the good frames


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
        samples.append(SliderFeatures(values[0], d["arcface"][good].astype(np.float64)))
    return samples


def target_vector(photos: list) -> np.ndarray:
    """Average ArcFace vector of the photos."""
    arcface = ArcFace()
    vectors = []
    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None:
            raise ValueError(f"cannot read {photo}")
        # Landmarks runs in video mode, which tracks the face from the previous
        # frame, so each photo gets its own.
        landmarks = Landmarks()
        found = landmarks.detect(image, 0)
        landmarks.close()
        if found is None:
            raise ValueError(f"no face in {photo}")
        vectors.append(arcface.embed(image, found[0][FIVE_POINTS, :2]))
    return np.mean(vectors, axis=0)


def idle_animation_axes(samples: list) -> np.ndarray:
    """The main axes of the idle animation's spread in ArcFace, as rows, from
    a PCA of the frames of the samples with every slider at 6. The idle
    animation is the same for every head, but these axes are not: a head's own
    60 axes capture 92% of its idle animation, while 60 axes from the other
    heads capture 46%."""
    controls = [s.arcface for s in samples if (s.sliders == SLIDER_CENTER).all()]
    if not controls:
        raise ValueError("no idle control clips")
    frames = np.concatenate(controls)
    _, _, vt = np.linalg.svd(frames - frames.mean(0), full_matrices=False)
    return vt[:IDLE_ANIMATION_AXES]


def remove_axes(array: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """array with its parts along the axes removed, per row. The axes are
    orthonormal rows."""
    return array - (array @ axes.T) @ axes


def design(values: np.ndarray, curved: np.ndarray) -> np.ndarray:
    """Model inputs per row of slider values: 1, the offsets from 6, and the
    squared offsets of the sliders with a curvature term."""
    offset = values - SLIDER_CENTER
    return np.concatenate([np.ones((len(values), 1)), offset, offset[:, curved] ** 2], axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    features = Path(sys.argv[1])
    head = int(sys.argv[2])
    photos = sys.argv[3:]

    samples = load_feature_folder(features / f"Head{head:02d}")
    idle_axes = idle_animation_axes(samples)

    def remove_idle_animation(array):
        return remove_axes(array, idle_axes)

    target = remove_idle_animation(target_vector(photos))
    values = np.array([c.sliders for c in samples])
    means = remove_idle_animation(np.array([c.arcface.mean(0) for c in samples]))

    print(f"{len(samples)} clips for head {head}, {len(photos)} target photo(s)")
    print("Measured similarity to the target, per clip:")
    for c, m in zip(samples, means, strict=True):
        moved = [f"{SLIDER_NAMES[s]} {v}" for s, v in enumerate(c.sliders) if v != SLIDER_CENTER]
        label = ", ".join(moved)
        if not label:
            label = "all at 6"
        print(f"  {cosine(m, target):+.3f}  {label}")

    # A curvature term needs the slider recorded at 3 or more values.
    curved = np.array([len(np.unique(values[:, s])) >= 3 for s in range(values.shape[1])])
    x = design(values, curved)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        sys.exit("not enough clips to fit every slider")
    coef, *_ = np.linalg.lstsq(x, means, rcond=None)

    def predict(v):
        return design(v[None], curved)[0] @ coef

    # Coordinate ascent over whole-number values: set one slider at a time to
    # its best value, until no slider changes.
    best = np.full(values.shape[1], SLIDER_CENTER)
    best_score = cosine(predict(best), target)
    changed = True
    while changed:
        changed = False
        for s in range(len(best)):
            for value in range(SLIDER_MIN, SLIDER_MAX + 1):
                trial = best.copy()
                trial[s] = value
                score = cosine(predict(trial), target)
                if score > best_score + 1e-9:
                    best = trial
                    best_score = score
                    changed = True

    base_score = cosine(predict(np.full(values.shape[1], SLIDER_CENTER)), target)
    print()
    low = values.min(axis=0)
    high = values.max(axis=0)
    print("Suggested values:")
    for s, name in enumerate(SLIDER_NAMES):
        note = ""
        if best[s] < low[s] or best[s] > high[s]:
            note = f"   beyond the recorded {low[s]} to {high[s]}, extrapolated"
        print(f"  {name:<12} {best[s]:2d}{note}")
    print(f"Predicted similarity: {best_score:+.3f}, against {base_score:+.3f} with every slider at 6")


if __name__ == "__main__":
    main()
