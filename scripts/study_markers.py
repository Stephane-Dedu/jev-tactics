"""Outil de calibration des marqueurs de cellule sur une capture de COMBAT.

    # Sonder un rectangle pris SUR la bordure d'un marqueur (releve a la main) :
    python scripts/study_markers.py combat.png --probe 730,203,40,7
    #   -> bande HSV a coller dans TEAM_BANDS (perception/entities.py)

    # Echantillonner la grille avec l'homographie courante et visualiser :
    python scripts/study_markers.py combat.png --calib configs/calibration.json --out o.png

Remplace study_bars.py : les barres de vie flottantes N'EXISTENT PAS en Dofus 3.6
(verifie sur 5 captures). L'equipe se lit sur le marqueur de cellule, les PV sur la
timeline. Cf. l'en-tete de perception/entities.py.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from jev_tactics.calibration.io import load_homography
from jev_tactics.perception.entities import (
    MIN_VOTE_RATIO,
    TEAM_BANDS,
    cell_outline_points,
    detect_cell_markers,
)


def probe(frame: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sel = hsv[(hsv[:, 1] > 100) & (hsv[:, 2] > 100)]
    print(f"rectangle {w}x{h} @ ({x},{y}) — {len(sel)}/{len(hsv)} px satures")
    if not len(sel):
        print("  aucun pixel sature : viser precisement la bordure du marqueur.")
        return
    for name, i in (("H", 0), ("S", 1), ("V", 2)):
        v = sel[:, i]
        print(f"  {name}: p5={int(np.percentile(v, 5)):3d} med={int(np.median(v)):3d} "
              f"p95={int(np.percentile(v, 95)):3d}")
    lo = tuple(max(0, int(np.percentile(sel[:, i], 5)) - m) for i, m in ((0, 5), (1, 30), (2, 30)))
    hi = tuple(min(c, int(np.percentile(sel[:, i], 95)) + m)
               for i, m, c in ((0, 5, 179), (1, 30, 255), (2, 30, 255)))
    print(f"\nbande suggeree : (({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}))")


def scan(frame: np.ndarray, calib: str, out: str | None) -> None:
    homography, meta = load_homography(calib)
    print(f"calibration : {meta.get('image', '?')}")
    print(f"bandes : {TEAM_BANDS}\nseuil de vote : {MIN_VOTE_RATIO}\n")

    markers = detect_cell_markers(frame, homography)
    for m in markers:
        print(f"  [MARQUEUR] cellule {m.cell:3d}  {m.team.value:<5} vote={m.vote_ratio:.2f}")
    if not markers:
        print("  aucun marqueur — verifier que l'homographie correspond A CETTE carte "
              "(sonde de coherence : scripts/bench.py) puis ajuster TEAM_BANDS via --probe.")

    if out:
        overlay = frame.copy()
        for m in markers:
            pts = cell_outline_points(homography, m.cell).astype(np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            cv2.putText(overlay, str(m.cell), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        cv2.imwrite(out, overlay)
        print(f"\noverlay -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--probe", help="x,y,w,h sur la bordure d'un marqueur")
    ap.add_argument("--calib", default="configs/calibration.json")
    ap.add_argument("--out", help="PNG annote (mode scan)")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"image illisible : {args.image}")

    if args.probe:
        probe(frame, tuple(int(v) for v in args.probe.split(",")))
    else:
        scan(frame, args.calib, args.out)


if __name__ == "__main__":
    main()
