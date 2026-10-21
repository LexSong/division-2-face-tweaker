"""Where clips and their features live.

Recordings sit in one folder per head type, named Head00 to Head19, under
the NVIDIA app's folder for the game. The folder name gives the head type, so
clips keep the names the NVIDIA app gives them. The feature cache mirrors that
layout: <features>/HeadNN/<clip name>.npz.
"""

import re
from pathlib import Path

HEAD_FOLDER = re.compile(r"Head(\d\d)")


def head_folders(root: Path) -> list:
    """(head number, folder) for each head folder in root, in head order."""
    out = []
    for folder in sorted(root.iterdir()):
        match = HEAD_FOLDER.fullmatch(folder.name)
        if folder.is_dir() and match is not None:
            out.append((int(match.group(1)), folder))
    return out
