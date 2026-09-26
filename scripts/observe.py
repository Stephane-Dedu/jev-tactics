"""Assemble une Observation depuis une image ou une capture d'ecran live.

    python scripts/observe.py --image "C:\\...\\combat.png"    # depuis un fichier
    python scripts/observe.py --live                           # capture ecran principal
    python scripts/observe.py --live --region 0,0,1920,1080    # region ecran
    python scripts/observe.py --image combat.png --no-board    # UI seule (sans grille)

La geometrie du plateau est estimee une fois (~150 ms) puis reutilisee ; l'observation
elle-meme coute ~20 ms. En mode --live --loop, on estime une seule fois et on observe
en boucle, ce qui est le rythme reel du bot en combat.
"""

from __future__ import annotations

import argparse
import time

import cv2

from jev_tactics.pipeline import build_board, observe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--region", help="left,top,width,height (avec --live)")
    ap.add_argument("--no-board", action="store_true",
                    help="ne pas estimer la grille (lecture d'UI seule)")
    ap.add_argument("--loop", type=int, default=1,
                    help="nombre d'observations (--live) ; la grille n'est estimee qu'une fois")
    args = ap.parse_args()

    if args.live:
        from jev_tactics.capture import ScreenCapture

        region = None
        if args.region:
            left, top, width, height = (int(v) for v in args.region.split(","))
            region = {"left": left, "top": top, "width": width, "height": height}
        capture = ScreenCapture()
        grab = lambda: capture.grab(region)
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"image illisible : {args.image}")

        class _Static:
            image = frame
            timestamp = 0.0

        grab = lambda: _Static()
    else:
        raise SystemExit("preciser --image <path> ou --live")

    first = grab()
    board = None
    if not args.no_board:
        t0 = time.perf_counter()
        board = build_board(first.image)
        elapsed = (time.perf_counter() - t0) * 1000
        if board is None:
            print(f"# aucune grille detectee ({elapsed:.0f} ms) — hors combat ?")
        else:
            print(f"# plateau : {len(board)} cases ({elapsed:.0f} ms)")

    for i in range(max(1, args.loop)):
        frame = first if i == 0 else grab()
        t0 = time.perf_counter()
        obs = observe(frame.image, board=board, timestamp=frame.timestamp or None)
        print(f"# observation en {(time.perf_counter() - t0) * 1000:.0f} ms")
        print(obs.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
