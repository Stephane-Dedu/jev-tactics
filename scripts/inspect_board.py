"""Montrer au bot ou est un monstre, et lui faire dire POURQUOI il ne l'a pas vu.

    python scripts/inspect_board.py combat1.png          # sur une capture
    python scripts/inspect_board.py --live               # sur le jeu

Clic GAUCHE sur une case  -> ce que la perception a mesure pour elle, et sa conclusion.
Touches d'etiquetage, apres un clic :
    e  « il y a un ENNEMI ici »        a  « un ALLIE »        v  « c'est VIDE »
    q / Echap  quitter

Chaque etiquette est enregistree avec TOUTES les mesures de la case. C'est ce qui rend le
desaccord exploitable : « tu dis ennemi, j'ai mesure un score de ligne de 17,2 pour un
seuil bas a 15,6, et aucun marqueur au-dessus de 0,04 » -- on sait alors lequel des deux
etages a lache, sans avoir a le deviner.

Pourquoi cet outil existe : une iteration entiere a ete passee a mesurer a la main
pourquoi deux monstres n'etaient pas detectes. Le diagnostic final tenait en une phrase,
que ce script produit en un clic. Et il s'etait d'abord trompe -- j'avais releve les cases
a l'oeil, en oubliant qu'un sprite isometrique est dessine AU-DESSUS de la sienne. Cliquer
la case et la voir surlignee supprime cette erreur-la.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.perception.inspect import BoardInspector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "cell_labels.jsonl"
WINDOW = "plateau — clic sur une case, [e]nnemi [a]llie [v]ide [q]uitter"

LABEL_KEYS = {ord("e"): "enemy", ord("a"): "ally", ord("v"): "empty"}
COLOURS = {"enemy": (0, 90, 255), "ally": (255, 180, 0), "empty": (150, 150, 150)}


def draw(inspector: BoardInspector, selection, labels) -> np.ndarray:
    """Plateau detecte en magenta, reseau alentour en gris, selection en jaune."""
    canvas = inspector.frame.copy()
    for key, index in inspector._index.items():
        on_board = key in inspector.board
        if not on_board and inspector.scores[index] < inspector.low_threshold:
            continue        # ni plateau ni candidat : ne pas noyer l'image
        colour = (255, 0, 255) if on_board else (90, 90, 90)
        cv2.polylines(canvas, [inspector.lattice.outline(index, per_edge=6, inset=0.85)
                               .astype(np.int32)], True, colour, 1)
    for entry in labels:
        cv2.circle(canvas, tuple(entry["centre"]), 14, COLOURS[entry["label"]], 2)
    if selection is not None:
        index = inspector._index[selection.lattice]
        cv2.polylines(canvas, [inspector.lattice.outline(index, per_edge=6, inset=0.9)
                               .astype(np.int32)], True, (0, 255, 255), 3)
    return canvas


def describe(diagnosis) -> str:
    markers = " ".join(f"{team}={ratio:.2f}" for team, ratio in diagnosis.markers.items())
    return (f"\nmaille {diagnosis.lattice} @ {diagnosis.centre}\n"
            f"  score de ligne {diagnosis.line_score:6.1f}   "
            f"(seuil bas {diagnosis.low_threshold:.1f}, amorce {diagnosis.seed_threshold:.1f})\n"
            f"  marqueurs      {markers}   (seuil de vote {diagnosis.vote_threshold})\n"
            f"  centre         {diagnosis.centre_response:6.1f}   "
            f"(une case de plateau VIDE est creuse : mediane ~2, max mesure 24)\n"
            f"  phase locale   {diagnosis.local_phase:6.2f}   "
            f"(< 1 : le centre repond plus que les aretes — un sprite est pose dessus)\n"
            f"  -> {diagnosis.explain()}")


def show_neighbourhood(inspector: BoardInspector, diagnosis) -> None:
    """Le voisinage, parce qu'un ilot ne s'explique jamais par sa seule case.

    Le defaut le plus long a comprendre n'etait pas dans la case du monstre mais dans le
    couloir de cases faibles qui la coupait du plateau.
    """
    print("  voisinage (score, * = dans le plateau) :")
    cells = {d.lattice: d for d in inspector.neighbourhood(*diagnosis.centre, radius=2)}
    ci, cj = diagnosis.lattice
    for i in range(ci - 2, ci + 3):
        row = "    "
        for j in range(cj - 2, cj + 3):
            found = cells.get((i, j))
            if found is None:
                row += "    -   "
            else:
                row += f"{found.line_score:6.1f}{'*' if found.on_board else ' '}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="capture a inspecter (sinon --live)")
    ap.add_argument("--live", action="store_true", help="capturer l'ecran du jeu")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS),
                    help="fichier d'etiquettes ('off' pour ne rien ecrire)")
    args = ap.parse_args()

    if args.live:
        from jev_tactics.capture import ScreenCapture, grab_when_focused

        with ScreenCapture() as capture:
            frame = grab_when_focused(capture, announce=print).image
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"image illisible : {args.image}")
    else:
        raise SystemExit("preciser une image ou --live")

    try:
        inspector = BoardInspector(frame=frame)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(f"reseau {len(inspector.lattice)} mailles | plateau {len(inspector.board)} cases")
    print(f"seuils : amorce {inspector.seed_threshold:.1f}, bas {inspector.low_threshold:.1f}")
    print("clic sur une case ; [e]nnemi [a]llie [v]ide pour etiqueter ; [q] pour quitter")

    state = {"selection": None}
    labels: list[dict] = []

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        diagnosis = inspector.diagnose(x, y)
        if diagnosis is None:
            print("\nhors du reseau")
            return
        state["selection"] = diagnosis
        print(describe(diagnosis))
        show_neighbourhood(inspector, diagnosis)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        cv2.imshow(WINDOW, draw(inspector, state["selection"], labels))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in LABEL_KEYS and state["selection"] is not None:
            diagnosis = state["selection"]
            entry = {
                "label": LABEL_KEYS[key],
                "lattice": list(diagnosis.lattice),
                "centre": list(diagnosis.centre),
                "on_board": diagnosis.on_board,
                "line_score": round(diagnosis.line_score, 2),
                "low_threshold": round(diagnosis.low_threshold, 2),
                "seed_threshold": round(diagnosis.seed_threshold, 2),
                "markers": {k: round(v, 3) for k, v in diagnosis.markers.items()},
                "centre_response": round(diagnosis.centre_response, 2),
                "local_phase": round(diagnosis.local_phase, 2),
                "source": args.image or "live",
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            labels.append(entry)
            agreed = (diagnosis.best_team[0] or "empty").lower().startswith(
                entry["label"][:3])
            print(f"  etiquette « {entry['label']} » enregistree"
                  f"{'' if agreed else '  <- DESACCORD avec la perception'}")

    cv2.destroyAllWindows()

    if labels and args.labels != "off":
        path = Path(args.labels)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ajout seul : les etiquettes s'accumulent d'une session a l'autre, et une
        # reecriture perdrait le travail deja fait.
        with path.open("a", encoding="utf-8") as stream:
            for entry in labels:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n{len(labels)} etiquette(s) -> {path}")


if __name__ == "__main__":
    main()
