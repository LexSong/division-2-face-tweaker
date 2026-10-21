"""Read The Division 2 face slider values from a recorded frame.

The slider panel is a flat plane drawn in perspective. On that plane the 9
bars are evenly spaced, and so are the 13 ticks on each bar, values 0 to 12.
So one homography maps grid coordinates (tick, row) to screen pixels. It is
stored as the pixel positions of the four grid corners, which
scripts/calibrate_sliders.py measures.

An unselected bar draws a white square knob. The selected bar is drawn larger
and in orange, with a black knob, and it follows its own homography.

A knob always sits on its bar's axis line, and the homography gives that line.
So the reader samples pixels along each axis line only, and finds the knob as
a short run of knob-colored samples there.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# The corners below hold for NVIDIA app recordings at this size.
FRAME_SIZE = (1566, 2880)

# Rows from top to bottom. Jawline sits below Cheeks and is off screen unless
# the panel scrolls, so it is not readable here.
SLIDER_NAMES = [
    "Brow Height",
    "Brow Depth",
    "Eyeline",
    "Eye Spacing",
    "Nose Width",
    "Nose Height",
    "Nose Bridge",
    "Mouth Height",
    "Cheeks",
]

TICKS = 13

# Grid coordinates (tick, row) of the four corners.
GRID_CORNERS = np.array(
    [
        [0, 0],
        [TICKS - 1, 0],
        [0, len(SLIDER_NAMES) - 1],
        [TICKS - 1, len(SLIDER_NAMES) - 1],
    ],
    dtype=np.float32,
)

# Pixel positions of GRID_CORNERS, from scripts/calibrate_sliders.py.
# Calibrated on pilot clips: white on all 9 rows, orange on rows 0, 1 and 4.
# Ticks miss these corners' homographies by 0.39 px at most in x.
CORNERS = {
    "white": [
        (672.40, 260.81),  # tick 0, row 0
        (1103.69, 279.45),  # tick 12, row 0
        (705.72, 1159.82),  # tick 0, row 8
        (1119.21, 1135.29),  # tick 12, row 8
    ],
    "orange": [
        (664.91, 254.24),  # tick 0, row 0
        (1103.30, 273.65),  # tick 12, row 0
        (699.18, 1165.78),  # tick 0, row 8
        (1118.67, 1140.12),  # tick 12, row 8
    ],
}

# Knob colors, inclusive bounds in BGR order.
KNOB_COLOR = {
    "white": ((221, 221, 221), (255, 255, 255)),
    "orange": ((0, 0, 0), (49, 49, 49)),
}
ORANGE_BAR_COLOR = ((0, 91, 201), (79, 169, 255))

# Scrollbar thumb area. When the panel scrolls, the thumb leaves this box and
# the corners above no longer apply. In place, the box averages 236-237
# brightness on every clip so far. Scrolled by one row, it drops to 37.
SCROLL_THUMB = (1139, 1147, 250, 478)  # x0, x1, y0, y1
THUMB_MIN_BRIGHTNESS = 200

# Samples along each axis line, in tick units. They reach half a tick past
# both ends, so a knob at 0 or 12 lies fully inside. 700 samples over about
# 450 px keep every sample under 1 px from the next.
SAMPLE_T = np.linspace(-0.5, TICKS - 0.5, 700)

# A knob is about 8 px wide and ticks are 33 to 38 px apart, so a knob covers
# about 0.23 tick. Runs of knob color outside this width are not a knob.
KNOB_WIDTH_T = (0.1, 0.4)

# A knob farther than this from the nearest tick is mid-move or misread.
# Half a tick spacing is about 17 px, so 6 px leaves a wide margin.
MAX_RESIDUAL_PX = 6.0


class PanelScrolledError(Exception):
    """The panel has scrolled, so a clip with this frame must be rejected."""


def _grid_to_pixel(state: str, t: np.ndarray, row: float) -> np.ndarray:
    """Pixel (x, y) of grid points (t, row) on one row."""
    h = cv2.getPerspectiveTransform(GRID_CORNERS, np.array(CORNERS[state], np.float32))
    grid = np.stack([t, np.full_like(t, row)], axis=1)
    return cv2.perspectiveTransform(grid[None].astype(np.float64), h)[0]


# Per state and row: float pixel position of each sample, its rounded pixel,
# and the x of each tick.
LINE = {}
LINE_PIXEL = {}
TICK_X = {}
for _state in CORNERS:
    for _row in range(len(SLIDER_NAMES)):
        LINE[_state, _row] = _grid_to_pixel(_state, SAMPLE_T, _row)
        LINE_PIXEL[_state, _row] = np.round(LINE[_state, _row]).astype(int)
        TICK_X[_state, _row] = _grid_to_pixel(_state, np.arange(TICKS, dtype=float), _row)[:, 0]


@dataclass
class Reading:
    values: list  # int per row, or None where no knob was read
    selected: int | None  # row of the orange bar
    residual: float  # largest distance in px from a knob to its tick

    @property
    def ok(self) -> bool:
        return None not in self.values and self.residual <= MAX_RESIDUAL_PX


def _runs(mask: np.ndarray) -> list:
    """(start, end) of each run of True values, end exclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True))


def _in_color(pixels: np.ndarray, color: tuple) -> np.ndarray:
    lo, hi = color
    return np.all((pixels >= lo) & (pixels <= hi), axis=1)


def _samples(frame_bgr: np.ndarray, state: str, row: int) -> np.ndarray:
    xy = LINE_PIXEL[state, row]
    return frame_bgr[xy[:, 1], xy[:, 0]]


def panel_in_place(frame_bgr: np.ndarray) -> bool:
    """True while the panel shows its first 9 sliders, where the corners apply."""
    x0, x1, y0, y1 = SCROLL_THUMB
    return frame_bgr[y0:y1, x0:x1].mean() >= THUMB_MIN_BRIGHTNESS


def read_sliders(frame_bgr: np.ndarray) -> Reading:
    if frame_bgr.shape[:2] != FRAME_SIZE:
        raise ValueError(f"frame is {frame_bgr.shape[:2]}, corners are for {FRAME_SIZE}")
    if not panel_in_place(frame_bgr):
        raise PanelScrolledError("panel scrolled, so the rows are not the first 9 sliders")

    values: list[int | None] = [None] * len(SLIDER_NAMES)
    residual = 0.0
    selected = None
    for row in range(len(SLIDER_NAMES)):
        # The selected bar is the one whose orange axis line runs over orange.
        orange_samples = _samples(frame_bgr, "orange", row)
        if _in_color(orange_samples, ORANGE_BAR_COLOR).mean() > 0.5:
            state = "orange"
            samples = orange_samples
            selected = row
        else:
            state = "white"
            samples = _samples(frame_bgr, "white", row)

        knob_runs = []
        for s, e in _runs(_in_color(samples, KNOB_COLOR[state])):
            width = SAMPLE_T[e - 1] - SAMPLE_T[s]
            if KNOB_WIDTH_T[0] <= width <= KNOB_WIDTH_T[1]:
                knob_runs.append((s, e))
        if len(knob_runs) != 1:
            # No knob, or more than one: leave None, so the frame is not ok.
            continue

        s, e = knob_runs[0]
        t = SAMPLE_T[s:e].mean()
        knob_x = LINE[state, row][s:e, 0].mean()
        v = int(np.clip(round(t), 0, TICKS - 1))
        values[row] = v
        residual = max(residual, abs(knob_x - TICK_X[state, row][v]))
    return Reading(values, selected, residual)


def read_video(path: str) -> list:
    """One Reading per frame. Raises PanelScrolledError naming the first
    scrolled frame."""
    cap = cv2.VideoCapture(path)
    readings = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                readings.append(read_sliders(frame))
            except PanelScrolledError as e:
                raise PanelScrolledError(f"frame {len(readings)}: {e}") from e
    finally:
        cap.release()
    return readings
