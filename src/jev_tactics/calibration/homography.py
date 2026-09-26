"""Calibration geometrique ecran <-> grille iso Dofus.

Coeur analytique du projet : la projection iso est fixe, on l'estime une fois par
homographie (DLT/SVD) a partir de >=4 correspondances (x,y)_grille <-> (u,v)_ecran,
puis les conversions sont closed-form et exactes.

La table cellId -> (x, y) reproduit EXACTEMENT l'algorithme officiel d'Ankama
(com.ankamagames.jerakine.types.positions.MapPoint, boucle init() du client
decompile). Les (x, y) sont les coordonnees diagonales du jeu : elles vivent sur un
plan, donc plan->ecran est affine et l'homographie l'absorbe -- aucune constante pixel
n'est codee en dur. Format 14x20/560 : legacy Dofus, stable depuis 1.29 ; on le valide
neanmoins via les residus de calibration (cf. calibration_error).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MAP_WIDTH = 14
MAP_HEIGHT = 20
GRID_CELLS = MAP_WIDTH * MAP_HEIGHT * 2  # 560


def _build_cell_coords() -> NDArray[np.float64]:
    """Reproduit MapPoint.init() : remplit CELLPOS[cell] = (startX + b, startY + b)."""
    coords: list[tuple[int, int]] = []
    start_x = 0
    start_y = 0
    for _row in range(MAP_HEIGHT):
        for b in range(MAP_WIDTH):  # sous-ligne paire
            coords.append((start_x + b, start_y + b))
        start_x += 1
        for b in range(MAP_WIDTH):  # sous-ligne impaire (decalee d'une demi-cellule)
            coords.append((start_x + b, start_y + b))
        start_y -= 1
    return np.asarray(coords, dtype=np.float64)


# (560, 2) : coordonnees diagonales Ankama (nX, nY) de chaque cellule, indexees 0..559.
CELL_COORDS: NDArray[np.float64] = _build_cell_coords()


def cell_coord(cell_id: int) -> NDArray[np.float64]:
    """Coordonnee-grille (x, y) d'une cellule (repere diagonal Ankama)."""
    return CELL_COORDS[cell_id]


def estimate_homography(
    grid_pts: NDArray[np.float64] | list[tuple[float, float]],
    screen_pts: NDArray[np.float64] | list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Estime H (3x3) : espace-grille -> pixels ecran, par DLT + SVD.

    Requiert >=4 correspondances non colineaires. Robuste au zoom camera 3.x
    (il suffit de re-estimer H apres un changement de zoom).
    """
    grid = np.asarray(grid_pts, dtype=np.float64)
    screen = np.asarray(screen_pts, dtype=np.float64)
    if len(grid) < 4 or len(grid) != len(screen):
        raise ValueError("il faut au moins 4 correspondances grille/ecran de meme longueur")

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(grid, screen):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def project(h: NDArray[np.float64], pts: NDArray[np.float64] | list[tuple[float, float]]) -> NDArray[np.float64]:
    """Applique une homographie a un ou plusieurs points (retourne toujours (N, 2))."""
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    hom = np.hstack([p, np.ones((len(p), 1))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


def cell_to_screen(h: NDArray[np.float64], cell_id: int) -> tuple[float, float]:
    """Centre ecran (px) d'une cellule -> cible d'un clic."""
    u, v = project(h, CELL_COORDS[cell_id])[0]
    return float(u), float(v)


def screen_to_cell(h: NDArray[np.float64], x: float, y: float) -> int:
    """Cellule sous un point ecran : projection inverse puis plus proche cellule."""
    grid = project(np.linalg.inv(h), (x, y))[0]
    return int(np.linalg.norm(CELL_COORDS - grid, axis=1).argmin())


def calibration_error(
    h: NDArray[np.float64],
    grid_pts: NDArray[np.float64] | list[tuple[float, float]],
    screen_pts: NDArray[np.float64] | list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Erreur de reprojection par point (px). C'est le juge de qualite d'une calibration
    ET la sonde qui detecterait une table cellId->(x,y) erronee (residus qui explosent)."""
    proj = project(h, grid_pts)
    return np.linalg.norm(proj - np.asarray(screen_pts, dtype=np.float64), axis=1)
