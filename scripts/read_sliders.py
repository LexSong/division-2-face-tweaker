"""Print each run of frames that share the same slider values.

A clip where the panel scrolls is rejected, because its rows are then not
the first 9 sliders.

Usage: uv run scripts/read_sliders.py <video>
"""

import sys

from face_tweaker.division2_sliders import SLIDER_NAMES
from face_tweaker.division2_sliders import PanelScrolledError
from face_tweaker.division2_sliders import read_video


def main():
    try:
        readings = read_video(sys.argv[1])
    except PanelScrolledError as e:
        print(f"rejected: {e}")
        sys.exit(1)
    print(f"{len(readings)} frames")
    print("Runs of frames with the same values. Columns are the sliders, top to bottom:")
    print("  " + ", ".join(SLIDER_NAMES))
    start = 0
    for i in range(1, len(readings) + 1):
        if i < len(readings) and readings[i].values == readings[start].values:
            continue
        run = readings[start:i]
        bad = sum(not r.ok for r in run)
        worst = max(r.residual for r in run)
        sel = run[0].selected
        if sel is None:
            sel_name = "none"
        else:
            sel_name = SLIDER_NAMES[sel]
        cells = []
        for v in run[0].values:
            if v is None:
                cells.append(" -")
            else:
                cells.append(f"{v:2d}")
        vals = " ".join(cells)
        print(
            f"frames {start:5d}-{i - 1:5d} ({len(run):4d})  [{vals}]"
            f"  selected {sel_name:<12} worst {worst:4.1f} px  bad {bad}"
        )
        start = i


if __name__ == "__main__":
    main()
