"""Read The Division 2 face slider values from a recorded frame.

The slider panel is a flat plane drawn in perspective. On that plane the 9
bars are evenly spaced, and so are the 13 ticks on each bar, values 0 to 12.
So one homography maps grid coordinates (tick, row) to screen pixels. It is
stored as the pixel positions of the four grid corners.

The game places the panel by the window's aspect ratio, and for one aspect
ratio the panel scales with the frame height. So each aspect ratio has its own
calibration, and a frame of another size with that ratio scales it.

An unselected bar draws a white square knob. The selected bar is drawn larger
and in orange, with a black knob, and it follows its own homography.

A knob always sits on its bar's axis line, and the homography gives that line.
So the reader samples pixels along each axis line only, and finds the knob as
a short run of knob-colored samples there.
"""

import math
from dataclasses import dataclass
from functools import cache
from functools import cached_property

import cv2
import numpy as np

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


@dataclass(frozen=True)
class PanelLayout:
    """Where the slider panel sits in frames of one size."""

    size: tuple  # (height, width) of the frames
    corners: dict  # per state, pixel positions of GRID_CORNERS
    scroll_thumb: tuple  # x0, x1, y0, y1 of a box inside the scrollbar thumb

    @staticmethod
    @cache
    def for_size(height: int, width: int) -> PanelLayout:
        """The calibration with this frame's aspect ratio, scaled to its size.

        Within one aspect ratio the panel scales with the frame, so one scale,
        from the heights, applies to x and y alike.
        """
        cal = None
        for candidate in CALIBRATIONS:
            ratio = candidate.size[1] / candidate.size[0]
            if abs(width / height - ratio) <= ASPECT_TOLERANCE * ratio:
                cal = candidate
                break
        if cal is None:
            raise ValueError(f"no slider calibration for a {width}x{height} frame")

        scale = height / cal.size[0]
        corners = {}
        for state, points in cal.corners.items():
            # Corners are pixel centers, which sit half a pixel in from pixel edges.
            corners[state] = [((x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5) for x, y in points]
        # Shrink the thumb box onto whole pixels inside its scaled edges.
        x0, x1, y0, y1 = cal.scroll_thumb
        thumb = (math.ceil(x0 * scale), int(x1 * scale), math.ceil(y0 * scale), int(y1 * scale))
        return PanelLayout((height, width), corners, thumb)

    @cached_property
    def line_pixels(self) -> dict:
        """Per (state, row), the pixel (x, y) of each sample on the axis line."""
        out = {}
        for state, points in self.corners.items():
            h = cv2.getPerspectiveTransform(GRID_CORNERS, np.array(points, np.float32))
            for row in range(len(SLIDER_NAMES)):
                grid = np.stack([SAMPLE_T, np.full_like(SAMPLE_T, row)], axis=1)
                line = cv2.perspectiveTransform(grid[None], h)[0]
                out[state, row] = np.round(line).astype(int)
        return out

    def samples(self, frame_bgr: np.ndarray, state: str, row: int) -> np.ndarray:
        """Frame pixels along one row's axis line."""
        xy = self.line_pixels[state, row]
        return frame_bgr[xy[:, 1], xy[:, 0]]

    def panel_in_place(self, frame_bgr: np.ndarray) -> bool:
        """True while the panel shows its first 9 sliders, where the corners apply."""
        x0, x1, y0, y1 = self.scroll_thumb
        return bool(frame_bgr[y0:y1, x0:x1].mean() >= THUMB_MIN_BRIGHTNESS)


# One layout per aspect ratio, at the frame size it was measured on.
CALIBRATIONS = [
    # Game window at 2880x1566, recorded at that size. From
    # scripts/calibrate_sliders.py on pilot clips: white on all 9 rows, orange
    # on rows 0, 1 and 4. Ticks miss these corners' homographies by 0.39 px at
    # most in x.
    PanelLayout(
        size=(1566, 2880),
        corners={
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
        },
        scroll_thumb=(1139, 1147, 250, 478),
    ),
    # Game window at 16:9, recorded at 3840x2160 and 1280x720. Mapped from the
    # corners above by an affine alignment (OpenCV ECC) of the panel area in a
    # 4K frame to a 2880x1566 frame. The same alignment on a 720p frame gives
    # corners within 0.1 px of these, at 4K scale.
    PanelLayout(
        size=(2160, 3840),
        corners={
            "white": [
                (854.22, 360.13),  # tick 0, row 0
                (1449.58, 385.78),  # tick 12, row 0
                (900.57, 1599.89),  # tick 0, row 8
                (1471.34, 1566.00),  # tick 12, row 8
            ],
            "orange": [
                (843.88, 351.07),  # tick 0, row 0
                (1449.04, 377.78),  # tick 12, row 0
                (891.54, 1608.10),  # tick 0, row 8
                (1470.59, 1572.66),  # tick 12, row 8
            ],
        },
        scroll_thumb=(1499, 1509, 346, 659),
    ),
]

# Aspect ratios within this fraction of a calibration's ratio use it. The two
# ratios above, 1.839 and 1.778, differ by 3%.
ASPECT_TOLERANCE = 0.005

# Knob colors, inclusive bounds in BGR order. The black knob's bound sits at
# about half the orange bar's color (25, 110, 245), so a pixel counts when it is
# more than half knob. At 720p the knob is 4 px wide and its edge pixels are
# mixed, so a tighter bound loses them in some frames and the knob goes unread.
KNOB_COLOR = {
    "white": ((221, 221, 221), (255, 255, 255)),
    "orange": ((0, 0, 0), (60, 60, 125)),
}
ORANGE_BAR_COLOR = ((0, 91, 201), (79, 169, 255))

# When the panel scrolls, the scrollbar thumb leaves its box and the corners no
# longer apply. In place, the box averages 236-237 brightness on every
# 2880x1566 clip so far. Scrolled by one row, it drops to 37.
THUMB_MIN_BRIGHTNESS = 200

# Samples along each axis line, in tick units. They reach half a tick past
# both ends, so a knob at 0 or 12 lies fully inside. The line is about 450 px
# long at height 1566 and 620 px at 2160, so 700 samples stay under 1 px apart.
SAMPLE_T = np.linspace(-0.5, TICKS - 0.5, 700)

# A knob covers about 0.23 tick. Runs of knob color outside this width are not
# a knob.
KNOB_WIDTH_T = (0.1, 0.4)

# A knob farther than this from the nearest tick, in tick units, is mid-move or
# misread. Half a tick is 0.5, so 0.17 (6 px at height 1566) leaves a wide
# margin.
MAX_RESIDUAL_T = 0.17


class PanelScrolledError(Exception):
    """The panel has scrolled, so a clip with this frame must be rejected."""


@dataclass
class Reading:
    values: list  # int per row, or None where no knob was read
    selected: int | None  # row of the orange bar
    residual: float  # largest distance in ticks from a knob to its tick

    @property
    def ok(self) -> bool:
        return None not in self.values and self.residual <= MAX_RESIDUAL_T


def _runs(mask: np.ndarray) -> list:
    """(start, end) of each run of True values, end exclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True))


def _in_color(pixels: np.ndarray, color: tuple) -> np.ndarray:
    lo, hi = color
    return np.all((pixels >= lo) & (pixels <= hi), axis=1)


def read_sliders(frame_bgr: np.ndarray) -> Reading:
    lay = PanelLayout.for_size(*frame_bgr.shape[:2])
    if not lay.panel_in_place(frame_bgr):
        raise PanelScrolledError("panel scrolled, so the rows are not the first 9 sliders")

    values: list[int | None] = [None] * len(SLIDER_NAMES)
    residual = 0.0
    selected = None
    for row in range(len(SLIDER_NAMES)):
        # The selected bar is the one whose orange axis line runs over orange.
        orange_samples = lay.samples(frame_bgr, "orange", row)
        if _in_color(orange_samples, ORANGE_BAR_COLOR).mean() > 0.5:
            state = "orange"
            samples = orange_samples
            selected = row
        else:
            state = "white"
            samples = lay.samples(frame_bgr, "white", row)

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
        v = int(np.clip(round(t), 0, TICKS - 1))
        values[row] = v
        residual = max(residual, abs(t - v))
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
