"""Lire les etiquettes de cases et dire OU la perception lache, et combien de fois.

    python scripts/study_cells.py                      # data/cell_labels.jsonl
    python scripts/study_cells.py mon_fichier.jsonl

Les etiquettes viennent de `scripts/inspect_board.py` : on clique une case, on dit ce
qu'on y voit, et toutes les mesures de la case sont enregistrees avec.

Ce script ferme la boucle. Sans lui, etiqueter serait un geste sans suite -- le defaut
que ce projet a deja corrige trois fois : une mesure produite, puis jetee.

Ce qu'il repond, et qui ne se devine pas :

  - sur les cases ou l'on a dit « ennemi », COMBIEN sont effectivement vues, et pour les
    autres, a quel ETAGE ca a lache. Un seuil, une connexite et une bande de couleur
    appellent trois corrections differentes ; savoir laquelle mord le plus souvent dit
    par quoi commencer ;
  - et le RECOUVREMENT des mesures entre les cases pleines et les cases vides. Un critere
    dont les plages se chevauchent ne separe rien : le regler ne ferait que deplacer les
    erreurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "cell_labels.jsonl"
# En dessous, aucune plage n'est representative : l'annoncer evite de conclure sur trois
# clics et de croire la mesure solide.
MIN_SAMPLES = 8


def load(path: Path) -> list[dict]:
    """Lignes illisibles IGNOREES : une session interrompue en laisse une tronquee, et
    perdre tout l'etiquetage pour cela serait absurde."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def stage_of(row: dict) -> str:
    """A quel etage cette case a-t-elle ete perdue ? Meme decoupage que le verdict de
    `BoardInspector.explain`, pour que les deux racontent la meme histoire."""
    best = max(row["markers"].values()) if row["markers"] else 0.0
    if not row["on_board"]:
        if row["line_score"] >= row["low_threshold"]:
            return "connexite (score suffisant, mais case isolee)"
        return "score de ligne (sprite masquant les aretes)"
    if best >= 0.2:
        return "VUE"
    if best > 0:
        return "seuil de vote (marqueur present mais faible)"
    return "bande de couleur (aucun marqueur)"


def spans(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return ordered[0], ordered[len(ordered) // 2], ordered[-1]


def overlap(a: list[float], b: list[float]) -> float:
    lo_a, _, hi_a = spans(a)
    lo_b, _, hi_b = spans(b)
    inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    union = max(hi_a, hi_b) - min(lo_a, lo_b)
    return inter / union if union else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="?", default=str(DEFAULT))
    args = ap.parse_args()

    path = Path(args.labels)
    if not path.exists():
        sys.exit(f"aucune etiquette : {path}\n"
                 f"En produire : python scripts/inspect_board.py <capture>")
    rows = load(path)
    if not rows:
        sys.exit(f"fichier vide : {path}")

    occupied = [r for r in rows if r["label"] in ("enemy", "ally")]
    empty = [r for r in rows if r["label"] == "empty"]
    print(f"{len(rows)} etiquette(s) : {len(occupied)} case(s) occupee(s), "
          f"{len(empty)} vide(s), sur {len({r['source'] for r in rows})} capture(s)")

    if occupied:
        print("\nCASES OCCUPEES — a quel etage la perception lache :")
        stages = Counter(stage_of(r) for r in occupied)
        for stage, count in stages.most_common():
            share = count / len(occupied)
            print(f"  {count:3} ({share:4.0%})  {stage}")
        seen = stages.get("VUE", 0)
        print(f"\n  -> {seen}/{len(occupied)} vues. Le premier etage de la liste est "
              f"celui par lequel commencer.")

    if empty:
        wrong = [r for r in empty if max(r["markers"].values(), default=0.0) >= 0.2]
        print(f"\nCASES VIDES — {len(wrong)}/{len(empty)} portent pourtant un marqueur "
              f"au-dessus du seuil de vote (ce sont des faux positifs en puissance)")

    if occupied and empty and len(rows) >= 2:
        print(f"\n{'critere':<16}{'occupees (min/med/max)':>26}{'vides':>26}{'recouvr.':>10}")
        criteria = [("score de ligne", lambda r: r["line_score"]),
                    ("marqueur max", lambda r: max(r["markers"].values(), default=0.0))]
        if all("centre_response" in r for r in rows):
            criteria.append(("reponse centre", lambda r: r["centre_response"]))
        for name, get in criteria:
            a, b = [get(r) for r in occupied], [get(r) for r in empty]
            print(f"{name:<16}{'/'.join(f'{v:.2f}' for v in spans(a)):>26}"
                  f"{'/'.join(f'{v:.2f}' for v in spans(b)):>26}{overlap(a, b):>9.0%}")
        print("\nRecouvrement faible = critere qui separe, donc seuil utilisable.")
        print("Recouvrement fort   = ce critere ne distingue rien ; le durcir couterait")
        print("                      des cases manquees sans supprimer les faux positifs.")

    if len(rows) < MIN_SAMPLES:
        print(f"\n/!\\ moins de {MIN_SAMPLES} etiquettes : ne rien regler sur si peu.")


if __name__ == "__main__":
    main()
