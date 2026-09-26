"""Mesure la latence de chaque etage du pipeline, separement.

Le projet a une regle : on n'optimise pas sans profiler. Ce script est l'instrument
qui rend cette regle applicable -- il donne l'avant/apres de toute optimisation.

    python scripts/bench.py                          # sur une capture de reference
    python scripts/bench.py --image "C:\\...\\x.png"  # sur une autre image
    python scripts/bench.py --no-capture             # sans toucher a l'ecran

Necessite Tesseract tant que l'OCR n'est pas remplace (cf. docs/ARCHITECTURE_PERFORMANCE.md).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import cv2

from jev_tactics.calibration import CELL_COORDS, project, screen_to_cell
from jev_tactics.calibration.io import load_homography
from jev_tactics.perception import DEFAULT_ROIS, read_ui
from jev_tactics.perception.ui import _ocr, _preprocess

DEFAULT_IMAGE = Path.home() / "Pictures" / "Screenshots" / "dofus.png"
DEFAULT_CALIB = Path(__file__).resolve().parents[1] / "configs" / "calibration.json"


def bench(label: str, fn: Callable[[], object], n: int = 20) -> float:
    """Chronometre `fn` sur n iterations apres un warmup. Retourne les ms/appel."""
    fn()  # warmup : exclut l'import paresseux et le premier cache-miss
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<44} {ms:8.2f} ms")
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--no-capture", action="store_true", help="ne pas capturer l'ecran")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"image illisible : {args.image}")
    print(f"image : {frame.shape[1]}x{frame.shape[0]}\n")

    # --- Perception : OCR de l'UI (aujourd'hui le poste dominant) ---
    print("PERCEPTION (OCR UI)")
    total_ocr = bench("read_ui() complet (4 ROIs)", lambda: read_ui(frame), n=3)
    crop = frame[DEFAULT_ROIS[0].y0:DEFAULT_ROIS[0].y1, DEFAULT_ROIS[0].x0:DEFAULT_ROIS[0].x1]
    pre = bench("  _preprocess() seul, 1 ROI (OpenCV)", lambda: _preprocess(crop), n=50)
    binimg = _preprocess(crop)
    tess = bench("  _ocr() 1 ROI (Tesseract, FALLBACK seul)", lambda: _ocr(binimg), n=5)
    print(f"  -> le fallback Tesseract coute {tess / max(total_ocr / 4, 1e-9):.0f}x "
          f"le chemin template par ROI\n")

    # --- Calibration : closed-form, doit rester negligeable ---
    print("CALIBRATION (homographie)")
    h, meta = load_homography(str(DEFAULT_CALIB))
    bench("project() 560 cases -> pixels", lambda: project(h, CELL_COORDS), n=200)
    bench("screen_to_cell() 1 point", lambda: screen_to_cell(h, 960.0, 540.0), n=200)

    # Sonde de coherence : combien de cases tombent reellement dans l'ecran ?
    pts = project(h, CELL_COORDS)
    hgt, wid = frame.shape[:2]
    inside = int(((pts[:, 0] >= 0) & (pts[:, 0] < wid) & (pts[:, 1] >= 0) & (pts[:, 1] < hgt)).sum())
    print(f"  cases projetees dans l'image : {inside}/560")
    if inside < 560:
        print(f"  /!\\ calibration partielle (bbox x[{pts[:, 0].min():.0f},{pts[:, 0].max():.0f}]) "
              f"-- ancrage absolu a fixer : {meta.get('note', '')}")
    print()

    # --- Capture ecran ---
    if not args.no_capture:
        print("CAPTURE")
        from jev_tactics.capture import ScreenCapture

        with ScreenCapture() as cap:
            full = bench("grab() plein ecran", lambda: cap.grab().image.shape, n=15)
            roi = {"left": 500, "top": 940, "width": 140, "height": 120}
            small = bench("grab() ROI 140x120", lambda: cap.grab(roi).image.shape, n=15)
        # mss a un cout plancher : restreindre la ROI rapporte peu.
        print(f"  -> ROI 100x plus petite mais seulement {full / small:.1f}x plus rapide "
              f"(cout plancher ~{small:.0f} ms)\n")
        print(f"BOUCLE COMPLETE (capture + OCR) ~ {full + total_ocr:.0f} ms")


if __name__ == "__main__":
    main()
