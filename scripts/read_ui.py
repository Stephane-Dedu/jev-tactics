"""Lit PV/PA/PM sur une capture de combat via l'OCR de l'UI.

Usage :
    python scripts/read_ui.py "C:\\...\\dofus.png"

Necessite Tesseract installe (cf. perception/ui.py pour le chemin / la var TESSERACT_CMD).
Les ROIs par defaut valent pour la disposition 1919x1079.
"""

import argparse

import cv2

from jev_tactics.perception import read_ui


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"image illisible : {args.image}")
    print(read_ui(frame).model_dump())


if __name__ == "__main__":
    main()
