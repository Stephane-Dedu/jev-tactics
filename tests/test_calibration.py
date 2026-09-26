"""La calibration doit reconstruire exactement la grille depuis quelques points."""

import numpy as np

from jev_tactics.calibration import (
    CELL_COORDS,
    GRID_CELLS,
    calibration_error,
    cell_to_screen,
    estimate_homography,
    screen_to_cell,
)


def test_table_shape_and_bounds():
    assert CELL_COORDS.shape == (GRID_CELLS, 2)
    assert GRID_CELLS == 560
    # Cellule 0 = origine du repere diagonal Ankama.
    assert tuple(CELL_COORDS[0]) == (0.0, 0.0)


def _synthetic_iso_projection() -> np.ndarray:
    """Projection iso plausible (demi-largeur 43px, demi-hauteur 21.5px) + offset."""
    m = np.array([[43.0, -43.0], [21.5, 21.5]])
    t = np.array([600.0, 100.0])
    return CELL_COORDS @ m.T + t


def test_homography_round_trips_every_cell():
    screen = _synthetic_iso_projection()
    # Calibration a partir de 5 cellules seulement (>=4 requis).
    sample = [0, 13, 279, 546, 559]
    h = estimate_homography(CELL_COORDS[sample], screen[sample])

    # Reprojection quasi parfaite sur les 560 cellules.
    err = calibration_error(h, CELL_COORDS, screen)
    assert err.max() < 1e-6

    # Chaque cellule : centre ecran -> retour a la meme cellule.
    for cell in range(GRID_CELLS):
        u, v = cell_to_screen(h, cell)
        assert screen_to_cell(h, u, v) == cell


def test_requires_four_points():
    try:
        estimate_homography([(0, 0), (1, 1), (2, 0)], [(0, 0), (1, 1), (2, 0)])
    except ValueError:
        return
    raise AssertionError("aurait du exiger >=4 correspondances")
