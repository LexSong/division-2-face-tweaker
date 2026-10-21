"""Suggest slider values that make one head look like the target photos.

All clips in the head's feature folder are data points: the control clip with
every slider at 6, and clips with sliders moved. Each clip gives one point,
its ArcFace average over frames near the usual head angle. A local model then
predicts ArcFace from the sliders. Per slider it has a slope, plus a
curvature where that slider was recorded on both sides of 6, and the sliders
are assumed to add up.

The suggestion maximizes the cosine similarity between the predicted ArcFace
and the target's, over each slider's full 0-12 range. Beyond the values
recorded for a slider, the model extrapolates, so those suggestions are
marked. Record the suggestion as a new clip in the same head folder: the next
run fits it in, which measures the model exactly where it was guessing.

Usage: uv run scripts/fit_sliders.py <features_dir> <control.npz> <head> <photo>...
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from face_tweaker.division2_sliders import SLIDER_NAMES
from face_tweaker.face_features import ArcFace
from face_tweaker.face_features import Landmarks

BASE_VALUE = 6
SLIDER_MIN = 0
SLIDER_MAX = 12

# Animation axes removed from every vector before comparing. A head's first 30
# axes hold about 87% of its animation, and removing more risks removing what
# makes the target recognizable.
CLEAN_DIRECTIONS = 30


def target_vector(photos: list) -> np.ndarray:
    """Average ArcFace vector of the photos."""
    landmarks = Landmarks()
    arcface = ArcFace()
    vectors = []
    for photo in photos:
        image = cv2.imread(str(photo))
        if image is None:
            raise ValueError(f"cannot read {photo}")
        found = landmarks.detect(image, 0)
        if found is None:
            raise ValueError(f"no face in {photo}")
        vectors.append(arcface.embed(image, found[0]))
    landmarks.close()
    return np.mean(vectors, axis=0)


def clip_points(folder: Path, yaw_center: float, yaw_limit: float) -> list:
    """(name, slider values, ArcFace average) for each clip whose sliders
    stay constant."""
    points = []
    for path in sorted(folder.glob("*.npz")):
        d = np.load(path)
        good = d["ok"] & d["face"] & (np.abs(d["yaw"] - yaw_center) <= yaw_limit)
        values = np.unique(d["sliders"][good], axis=0)
        if len(values) != 1:
            print(f"skipped {path.stem}: its sliders change during the clip")
            continue
        points.append((path.stem, values[0], d["arcface"][good].astype(np.float64).mean(0)))
    return points


def design(values: np.ndarray, curved: np.ndarray) -> np.ndarray:
    """Model inputs per row of slider values: 1, the offsets from 6, and the
    squared offsets of the sliders with a curvature term."""
    offset = values - BASE_VALUE
    return np.concatenate([np.ones((len(values), 1)), offset, offset[:, curved] ** 2], axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    features = Path(sys.argv[1])
    control = np.load(sys.argv[2])
    head = int(sys.argv[3])
    photos = sys.argv[4:]

    heads = list(control["heads"])
    i = heads.index(head)
    directions = control["directions"][i][:CLEAN_DIRECTIONS]

    def clean(x):
        return x - (x @ directions.T) @ directions

    points = clip_points(features / f"Head{head:02d}", float(control["yaw_center"]), float(control["yaw_limit"]))
    target = clean(target_vector(photos))
    values = np.array([v for _, v, _ in points])
    means = np.array([clean(m) for _, _, m in points])

    # Every clip should match its own head's control best.
    control_means = np.array([clean(m) for m in control["mean"]])
    print(f"{len(points)} clips for head {head}, {len(photos)} target photo(s)")
    print("Measured similarity to the target, per clip:")
    for (name, v, _), m in zip(points, means, strict=True):
        nearest = heads[int(np.argmin(np.linalg.norm(control_means - m, axis=1)))]
        moved = [f"{SLIDER_NAMES[s]} {v[s]}" for s in range(len(v)) if v[s] != BASE_VALUE]
        label = ", ".join(moved)
        if not label:
            label = "all at 6"
        warning = ""
        if nearest != head:
            warning = f"   WARNING: looks more like head {nearest}"
        print(f"  {cosine(m, target):+.3f}  {label:<20} {name[-11:]}{warning}")

    # A curvature term needs the slider recorded on both sides of 6.
    curved = np.array(
        [(values[:, s] < BASE_VALUE).any() and (values[:, s] > BASE_VALUE).any() for s in range(values.shape[1])]
    )
    x = design(values, curved)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        sys.exit("not enough clips to fit every slider")
    coef, *_ = np.linalg.lstsq(x, means, rcond=None)

    def predict(v):
        return design(v[None], curved)[0] @ coef

    # Coordinate ascent over whole-number values: set one slider at a time to
    # its best value, until no slider changes.
    best = np.full(values.shape[1], BASE_VALUE)
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

    base_score = cosine(predict(np.full(values.shape[1], BASE_VALUE)), target)
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
