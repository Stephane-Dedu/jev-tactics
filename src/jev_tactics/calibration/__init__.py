"""Calibration geometrique : homographie ecran <-> grille iso.

La projection iso est une transformation projective fixe. On l'estime UNE fois a partir
de >=4 points de reference (cellules connues), puis la conversion est closed-form et
exacte. En 3.x la camera zoome -> re-estimer l'homographie (toujours des points, jamais
un reseau). La table cellId->(x,y) est l'algorithme officiel Ankama (cf. homography.py).
"""

from jev_tactics.calibration.homography import (
    CELL_COORDS,
    GRID_CELLS,
    MAP_HEIGHT,
    MAP_WIDTH,
    calibration_error,
    cell_coord,
    cell_to_screen,
    estimate_homography,
    project,
    screen_to_cell,
)
from jev_tactics.calibration.io import load_homography, save_homography

__all__ = [
    "CELL_COORDS",
    "GRID_CELLS",
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "calibration_error",
    "cell_coord",
    "cell_to_screen",
    "estimate_homography",
    "load_homography",
    "project",
    "save_homography",
    "screen_to_cell",
]
