"""Calibrate the slider grid corners used by face_tweaker.division2_sliders.

The slider panel is a flat plane drawn in perspective. On that plane the 9
bars are evenly spaced, and so are the 13 ticks on each bar. So one homography
maps grid coordinates (tick, row) to screen pixels. The selected bar is drawn
larger and in orange, and it follows its own homography. This script finds the
ticks drawn in each frame, fits both homographies, and prints the pixel
positions of the four grid corners for each.

Usage: uv run scripts/calibrate_sliders.py <video>...
Uses frame 0 of each video. Together the videos must show every white bar at
least once, and the orange bar on at least 3 rows.
"""

import sys

import cv2
import numpy as np

from face_tweaker.division2_sliders import GRID_CORNERS
from face_tweaker.division2_sliders import KNOB_COLOR
from face_tweaker.division2_sliders import ORANGE_BAR_COLOR
from face_tweaker.division2_sliders import TICKS

# Area searched for knobs and ticks. The scrollbar starts at x 1137.
PANEL = (640, 1134, 230, 1190)  # x0, x1, y0, y1

# Rows above and below the knob that the tick search sees. The lowest bar
# tilts about 12 px from its center to each end, and ticks reach 10 px off
# the axis, so 25 px holds the whole bar.
AXIS_BAND = 25

# Steepest axis slope tried. The lowest bar tilts about 0.05.
MAX_SLOPE = 0.1

# Ticks are found 7 px off the axis. There only ticks cross, while the knob
# reaches 4.5 px and the axis line 1 px.
TICK_OFFSET = 7

# Row heights of the bars at value 6, used only to group knobs by row here.
ROW_Y = np.array([270, 383, 496, 608, 718, 827, 934, 1041, 1147])


def knob_masks(frame_bgr: np.ndarray) -> dict:
    """Per state, a uint8 mask of knob-colored pixels over PANEL."""
    x0, x1, y0, y1 = PANEL
    p = frame_bgr[y0:y1, x0:x1]
    white = cv2.inRange(p, *KNOB_COLOR["white"])
    black = cv2.inRange(p, *KNOB_COLOR["orange"])
    # A black knob counts only inside the orange bar, which spans the rows
    # with more than 100 orange pixels.
    orange = cv2.inRange(p, *ORANGE_BAR_COLOR)
    black[np.count_nonzero(orange, axis=1) <= 100] = 0
    return {"white": white, "orange": black}


def knobs(mask: np.ndarray) -> list:
    """Frame pixel centers of components the size of a knob, about 8 by 8 px."""
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        if 40 <= area <= 120 and 6 <= w <= 12 and 6 <= h <= 12:
            out.append((centers[i][0] + PANEL[0], centers[i][1] + PANEL[2]))
    return out


def _runs(mask: np.ndarray) -> list:
    """(start, end) of each run of True values, end exclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True))


def find_ticks(gray: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """(x, y) of each tick of the bar whose knob is at (kx, ky), knob included."""
    x0, x1 = PANEL[0], PANEL[1]
    top = round(float(ky)) - AXIS_BAND
    band = gray[top : top + 2 * AXIS_BAND + 1, x0:x1]

    # Contrast against each column's own background. Ticks and the axis cover
    # less than half of a column, so the column median is the background. This
    # works for light ticks on the dark bar and dark ticks on the orange bar.
    contrast = np.abs(band - np.median(band, axis=0))

    # The axis line passes through the knob center, so only its slope is
    # unknown. Try slopes through the knob and keep the line with the most
    # contrast. The bar outline never passes through the knob, so it cannot win.
    height = band.shape[0]
    cols = np.arange(band.shape[1])
    knob_col = kx - x0
    knob_row = ky - top
    best_score = -1.0
    slope = 0.0
    for s in np.linspace(-MAX_SLOPE, MAX_SLOPE, 401):
        rows = np.round(knob_row + s * (cols - knob_col)).astype(int)
        inside = (rows >= 0) & (rows < height)
        score = contrast[rows[inside], cols[inside]].sum()
        if score > best_score:
            best_score = score
            slope = s
    axis_rows = knob_row + slope * (cols - knob_col)

    # The axis runs from tick 0 to tick 12. Its extent is the run of bright
    # axis pixels that holds the knob, with gaps of up to 4 px closed.
    on_axis = contrast[np.clip(np.round(axis_rows).astype(int), 0, height - 1), cols]
    bright = np.convolve(on_axis > np.median(on_axis) / 2, np.ones(5), "same") > 0
    lo = None
    hi = None
    for s, e in _runs(bright):
        if s <= knob_col < e:
            lo = s
            hi = e
    if lo is None or hi is None:
        return np.zeros((0, 2))

    # Tick profile along the axis, sampled on both sides of it.
    xs = np.arange(lo, hi)
    axis_y = axis_rows[lo:hi]
    above = np.clip(np.round(axis_y - TICK_OFFSET).astype(int), 0, height - 1)
    below = np.clip(np.round(axis_y + TICK_OFFSET).astype(int), 0, height - 1)
    profile = (contrast[above, xs] + contrast[below, xs]) / 2

    base = np.median(profile)
    peak = np.percentile(profile, 99)
    tick_x = []
    for s, e in _runs(profile > base + (peak - base) / 2):
        w = profile[s:e] - base
        tick_x.append(x0 + lo + (np.arange(s, e) * w).sum() / w.sum())

    # The knob sits on one tick and hides it, so it stands in for that tick.
    tick_x = [x for x in tick_x if abs(x - kx) > 10]
    tick_x.append(kx)
    tick_x = np.sort(np.array(tick_x))
    tick_y = ky + slope * (tick_x - kx)
    return np.stack([tick_x, tick_y], axis=1)


def collect(videos: list) -> dict:
    """Per state, rows of (tick, row, x, y) from frame 0 of each video."""
    points: dict[str, list] = {"white": [], "orange": []}
    for path in videos:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read {path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        for state, mask in knob_masks(frame).items():
            for kx, ky in knobs(mask):
                row = int(np.argmin(abs(ROW_Y - ky)))
                if abs(ROW_Y[row] - ky) > 20:
                    continue
                ticks = find_ticks(gray, kx, ky)
                if len(ticks) != TICKS:
                    print(f"skipped {state} row {row}: found {len(ticks)} ticks")
                    continue
                for t, (x, y) in enumerate(ticks):
                    points[state].append((t, row, x, y))
    return {state: np.array(p) for state, p in points.items()}


def main():
    points = collect(sys.argv[1:])
    print("CORNERS = {")
    for state, p in points.items():
        rows = sorted(set(p[:, 1].astype(int).tolist()))
        h, _ = cv2.findHomography(p[:, :2], p[:, 2:], 0)
        err = cv2.perspectiveTransform(p[None, :, :2], h)[0] - p[:, 2:]
        corners = cv2.perspectiveTransform(GRID_CORNERS[None].astype(float), h)[0]
        print(
            f"    # {len(p)} ticks on rows {rows}. Worst fit error:"
            f" x {abs(err[:, 0]).max():.2f} px, y {abs(err[:, 1]).max():.2f} px."
        )
        print(f'    "{state}": [')
        for (t, row), (x, y) in zip(GRID_CORNERS, corners, strict=True):
            print(f"        ({x:.2f}, {y:.2f}),  # tick {t:.0f}, row {row:.0f}")
        print("    ],")
    print("}")


if __name__ == "__main__":
    main()
