"""Build the per-head control features from the control clips.

A control clip shows one head type with all 9 sliders at 6 and the mouse
untouched, for about two loops of the idle animation (17-18 s per loop). In
it the face is fixed, so every change in ArcFace comes from the animation.

Per head this stores:
- mean: the ArcFace average over frames near the usual head angle, the head's
  starting point for a search.
- directions: the main axes of the animation's spread in ArcFace, from a PCA
  of the clip's frames. The animation is the same for every head, but these
  axes are not: a head's own 60 axes capture 92% of its animation, while 60
  axes from the other heads capture 46%. So each head keeps its own.
- spread: how far the animation moves ArcFace along each axis, so a search can
  remove axes or weight them down.
- noise: spread of the average of one half of the clip, about one loop.

Usage: uv run scripts/build_control.py <control_features_dir> <out.npz>
The feature files come from extract_features.py, named HeadNN after the head.
"""

import re
import sys
from pathlib import Path

import numpy as np

# Frames turned more than this from the median yaw of all control clips are
# left out. The idle loop turns to about -40 degrees, and dropping those turns
# cleaned the slider effects slightly better. This keeps about 90% of frames.
YAW_LIMIT = 25.0

# Animation axes kept per head. 60 capture about 92% of a head's animation.
DIRECTIONS = 60


def load(folder: Path) -> dict:
    clips = {}
    for path in sorted(folder.glob("*.npz")):
        match = re.match(r"[Hh]ead(\d+)", path.name)
        if match is None:
            raise ValueError(f"{path.name}: name does not start with HeadNN")
        d = np.load(path)
        keep = d["ok"] & d["face"]
        clips[int(match.group(1))] = {"arcface": d["arcface"][keep].astype(np.float64), "yaw": d["yaw"][keep]}
    return clips


def main():
    clips = load(Path(sys.argv[1]))
    heads = sorted(clips)
    yaw_center = float(np.median(np.concatenate([clips[h]["yaw"] for h in heads])))

    means = []
    directions = []
    spreads = []
    noises = []
    used = []
    for h in heads:
        c = clips[h]
        x = c["arcface"][np.abs(c["yaw"] - yaw_center) <= YAW_LIMIT]
        mean = x.mean(0)
        _, sig, vt = np.linalg.svd(x - mean, full_matrices=False)
        half = len(x) // 2
        noise = np.linalg.norm(x[:half].mean(0) - x[half:].mean(0)) / np.sqrt(2)
        means.append(mean)
        directions.append(vt[:DIRECTIONS])
        # RMS distance of a frame from the mean along each axis.
        spreads.append(sig[:DIRECTIONS] / np.sqrt(len(x)))
        noises.append(noise)
        used.append(len(x) / len(c["yaw"]))

    means = np.array(means)
    np.savez(
        sys.argv[2],
        heads=np.array(heads),
        mean=means,
        directions=np.array(directions),
        spread=np.array(spreads),
        noise=np.array(noises),
        yaw_center=yaw_center,
        yaw_limit=YAW_LIMIT,
    )

    print(f"{len(heads)} heads, median yaw {yaw_center:.1f} degrees, frames within {YAW_LIMIT:.0f} degrees kept")
    print(f"{'head':>4} {'frames kept':>11} {'noise':>7} {'nearest head':>12} {'distance':>8} {'in noise units':>14}")
    for i, h in enumerate(heads):
        dist = np.linalg.norm(means - means[i], axis=1)
        dist[i] = np.inf
        j = int(np.argmin(dist))
        print(f"{h:4d} {used[i]:11.0%} {noises[i]:7.4f} {heads[j]:12d} {dist[j]:8.3f} {dist[j] / noises[i]:14.0f}")
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
