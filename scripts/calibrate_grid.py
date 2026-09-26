"""Calibration AUTOMATIQUE depuis la grille visible d'une capture de combat.

    python scripts/calibrate_grid.py combat.png                  # estimer et afficher
    python scripts/calibrate_grid.py combat.png --out over.png   # + overlay de controle
    python scripts/calibrate_grid.py combat.png --save configs/calibration_auto.json
    python scripts/calibrate_grid.py a.png b.png c.png           # comparer plusieurs

Remplace le calibrage manuel a 4 clics (scripts/calibrate.py) : celui-ci ne passait pas
a l'echelle (une carte, un clic) alors que le detecteur d'entites a besoin d'une
homographie PAR CARTE.

L'origine placee est coherente mais pas absolue : suffisant pour toute la planification
tactique (invariance par translation des coordonnees diagonales), cf. §4.4 du doc d'archi.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from jev_tactics.calibration.grid import board_coverage, estimate_grid
from jev_tactics.calibration.io import save_homography
from jev_tactics.perception.entities import cell_outline_points


def report(path: str, frame, estimate) -> None:
    if estimate is None:
        print(f"{path}: ECHEC — aucun reseau exploitable "
              f"(hors combat ? grille masquee ?)")
        return
    w, h = estimate.cell_size
    coverage = board_coverage(estimate.homography(), frame.shape[:2])
    print(f"{path}")
    print(f"  e_x = ({estimate.e_x[0]:7.2f}, {estimate.e_x[1]:6.2f})")
    print(f"  e_y = ({estimate.e_y[0]:7.2f}, {estimate.e_y[1]:6.2f})")
    print(f"  cellule = {w:.1f} x {h:.1f} px    force = {estimate.strength:.2f}")
    print(f"  couverture = {coverage * 100:.0f}% des 560 cellules dans l'image")


def overlay(frame, estimate, out: str, step: int = 1) -> None:
    """Dessine les contours de cellules projetes : controle visuel de l'alignement."""
    canvas = frame.copy()
    homography = estimate.homography()
    height, width = frame.shape[:2]
    for cell in range(0, 560, step):
        pts = cell_outline_points(homography, cell).astype(np.int32)
        if pts[:, 0].min() < 0 or pts[:, 0].max() >= width:
            continue
        if pts[:, 1].min() < 0 or pts[:, 1].max() >= height:
            continue
        cv2.polylines(canvas, [pts], True, (0, 255, 255), 1)
    cv2.imwrite(out, canvas)
    print(f"  overlay -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="PNG de controle (premiere image seulement)")
    ap.add_argument("--save", help="ecrire l'homographie estimee en JSON")
    ap.add_argument("--crop", type=float, default=0.55,
                    help="part centrale analysee (defaut 0.55)")
    args = ap.parse_args()

    first = None
    for path in args.images:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: illisible")
            continue
        estimate = estimate_grid(frame, args.crop)
        report(path, frame, estimate)
        if first is None and estimate is not None:
            first = (path, frame, estimate)

    if first is None:
        raise SystemExit("aucune estimation exploitable")

    path, frame, estimate = first
    if args.out:
        overlay(frame, estimate, args.out)
    if args.save:
        save_homography(args.save, estimate.homography(), {
            "image": path,
            "methode": "grille (autocorrelation)",
            "e_x": list(estimate.e_x),
            "e_y": list(estimate.e_y),
            "force": estimate.strength,
            "note": "origine coherente mais NON absolue ; ancrage absolu a fixer",
        })
        print(f"  homographie -> {args.save}")


if __name__ == "__main__":
    main()
