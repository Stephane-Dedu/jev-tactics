"""Calibrateur interactif ecran <-> grille iso Dofus, a lancer sur une vraie capture.

Usage :
    python scripts/calibrate.py "C:\\...\\dofus.png"
    python scripts/calibrate.py "C:\\...\\dofus.png" --refine 6

Etape 1 (obligatoire, zero connaissance requise) :
    Clique les 4 sommets d'UN SEUL losange de la grille, dans l'ordre :
    HAUT, DROITE, BAS, GAUCHE. On en derive la base iso (e_x, e_y) analytiquement.

Etape 2 (optionnelle, --refine N) :
    Clique le centre de N cellules n'importe ou sur la carte. Elles sont
    auto-etiquetees via la calibration de l'etape 1, puis tout est re-ajuste
    par homographie. Cliquer loin revele si la camera 3.x a de la perspective
    (l'affine d'une seule cellule derivera aux bords) ou reste orthographique.

Sorties :
    configs/calibration.json   : H (3x3) + base + residu
    scripts/_calib_overlay.png : grille Ankama reprojetee sur l'image (verif visuelle)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from jev_tactics.calibration import (
    CELL_COORDS,
    calibration_error,
    estimate_homography,
    project,
    save_homography,
    screen_to_cell,
)

ROOT = Path(__file__).resolve().parents[1]


def _clicks(img_rgb: np.ndarray, n: int, title: str) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.tight_layout()
    pts = plt.ginput(n, timeout=0)  # bloque jusqu'a n clics
    plt.close(fig)
    if len(pts) < n:
        raise SystemExit(f"il fallait {n} clics, {len(pts)} recu(s)")
    return np.asarray(pts, dtype=np.float64)


def _homography_from_diamond(top, right, bottom, left):
    """4 sommets d'un losange -> H (grille->ecran) + base iso (e_x, e_y).

    La cellule cliquee est ancree provisoirement en (0, 0). L'ancrage absolu
    (quelle cellId c'est) se fixera plus tard via un etat de jeu connu.
    """
    center = np.mean([top, right, bottom, left], axis=0)
    # right/left = center +/- (e_x - e_y)/2 ; top/bottom = center +/- (e_x + e_y)/2
    e_x = (right - center) + (bottom - center)  # = (right + bottom - 2*center)
    e_y = (bottom - center) - (right - center)  # = (bottom - right)
    grid = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    screen = np.array([center, center + e_x, center + e_y, center + e_x + e_y])
    h = estimate_homography(grid, screen)
    return h, e_x, e_y


def _draw_overlay(img_bgr, h, e_x, e_y, out_path):
    vis = img_bgr.copy()
    half_rl = (e_x - e_y) / 2.0  # centre -> sommet droit
    half_tb = (e_x + e_y) / 2.0  # centre -> sommet bas
    centers = project(h, CELL_COORDS)
    hh, ww = vis.shape[:2]
    for cx, cy in centers:
        if -200 < cx < ww + 200 and -200 < cy < hh + 200:
            verts = np.array([
                [cx, cy] - half_tb,  # haut
                [cx, cy] + half_rl,  # droite
                [cx, cy] + half_tb,  # bas
                [cx, cy] - half_rl,  # gauche
            ], dtype=np.int32)
            cv2.polylines(vis, [verts], isClosed=True, color=(0, 255, 255), thickness=1)
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--refine", type=int, default=0, help="nb de cellules a cliquer pour affiner")
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"image illisible : {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    quad = _clicks(img_rgb, 4, "Clique les 4 sommets d'UN losange : HAUT, DROITE, BAS, GAUCHE")
    h, e_x, e_y = _homography_from_diamond(*quad)
    print(f"base iso : e_x={e_x.round(1)}  e_y={e_y.round(1)}")

    if args.refine > 0:
        extra = _clicks(img_rgb, args.refine, f"Clique le CENTRE de {args.refine} cellules (n'importe ou)")
        grid_pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        screen_pts = [project(h, g)[0] for g in grid_pts]  # garde l'ancrage initial
        for px in extra:
            cell = screen_to_cell(h, px[0], px[1])  # auto-etiquetage
            grid_pts.append(CELL_COORDS[cell])
            screen_pts.append(px)
        h = estimate_homography(np.array(grid_pts), np.array(screen_pts))
        err = calibration_error(h, np.array(grid_pts), np.array(screen_pts))
        print(f"residu apres refine (px) : moy={err.mean():.2f}  max={err.max():.2f}")

    out_json = ROOT / "configs" / "calibration.json"
    save_homography(out_json, h, meta={
        "image": args.image,
        "e_x": e_x.tolist(),
        "e_y": e_y.tolist(),
        "note": "cellule cliquee ancree en (0,0) ; ancrage absolu a fixer via etat de jeu",
    })
    out_png = ROOT / "scripts" / "_calib_overlay.png"
    _draw_overlay(img_bgr, h, e_x, e_y, out_png)
    print(f"ecrit : {out_json}")
    print(f"ecrit : {out_png}  <- verifie l'alignement de la grille")


if __name__ == "__main__":
    main()
