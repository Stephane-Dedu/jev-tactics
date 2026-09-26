"""Valide une calibration sur donnees reelles, sans verite terrain.

Principe : sur l'herbe, les lignes de la grille du jeu sont plus sombres que le sol.
Si la calibration est correcte, les sommets de cellules predits tombent sur ces
rainures -> luminance(sommets) < luminance(centres). On balaie ensuite la phase
(a population de cellules FIGEE) pour mesurer le residu sous-pixel : l'optimum doit
etre proche de (0,0).

Usage :
    python scripts/validate_calibration.py <image> [--calib configs/calibration.json]

Note : le detecteur d'herbe (is_grass) est cale sur une map type prairie ; adapter le
seuil couleur pour un autre decor. C'est un controle qualite, pas un test unitaire.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.calibration import CELL_COORDS, load_homography, project

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--calib", default=str(ROOT / "configs" / "calibration.json"))
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"image illisible : {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bgr = img.astype(np.float32)
    hh, ww = gray.shape

    h, meta = load_homography(args.calib)
    e_x = np.array(meta["e_x"])
    e_y = np.array(meta["e_y"])
    half_rl = (e_x - e_y) / 2.0  # centre -> sommet droit/gauche
    half_tb = (e_x + e_y) / 2.0  # centre -> sommet haut/bas

    def lum(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        xi = np.clip(pts[:, 0].round().astype(int), 0, ww - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, hh - 1)
        return gray[yi, xi]

    def is_grass(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        xi = np.clip(pts[:, 0].round().astype(int), 0, ww - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, hh - 1)
        b, g, r = bgr[yi, xi, 0], bgr[yi, xi, 1], bgr[yi, xi, 2]
        return (g > r + 15) & (g > b + 30) & (g > 90) & (g < 235)

    # Population de cellules figee : herbe visible, loin des bords.
    base = project(h, CELL_COORDS)
    inside = (base[:, 0] > 60) & (base[:, 0] < ww - 60) & (base[:, 1] > 60) & (base[:, 1] < hh - 60)
    base = base[inside]
    base = base[is_grass(base)]
    if len(base) < 20:
        raise SystemExit(f"trop peu de cellules d'herbe ({len(base)}) : ajuster is_grass ou l'image")

    def phase_gap(off: np.ndarray) -> float:
        c = base + off
        v = np.stack(
            [lum(c + half_tb), lum(c - half_tb), lum(c + half_rl), lum(c - half_rl)], axis=1
        ).mean(axis=1)
        return float(lum(c).mean() - v.mean())

    gap0 = phase_gap(np.zeros(2))
    best = (0.0, 0.0, gap0)
    for dx in np.linspace(-0.5, 0.5, 21):  # au-dela d'1/2 maille on retombe sur le reseau
        for dy in np.linspace(-0.5, 0.5, 21):
            g = phase_gap(dx * half_rl + dy * half_tb)
            if g > best[2]:
                best = (dx, dy, g)
    res_px = float(np.hypot(*(best[0] * half_rl + best[1] * half_tb)))

    print(f"cellules d'herbe testees   : {len(base)}")
    print(f"contraste centre-sommet    : {gap0:+.2f}  (positif = sommets sur les rainures du jeu)")
    print(f"phase optimale             : dx={best[0]:+.2f}  dy={best[1]:+.2f} demi-cellule")
    print(f"residu de calibration      : {res_px:.1f} px")
    verdict = "OK (< 3 px)" if res_px < 3 else "a affiner (relance calibrate.py --refine)"
    print(f"verdict                    : {verdict}")


if __name__ == "__main__":
    main()
