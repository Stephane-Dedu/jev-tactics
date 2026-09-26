"""Fabrique les gabarits de chiffres de la pastille de REMPLISSAGE d'inventaire.

    python scripts/build_pods_templates.py hors_combat.png=59 plein.png=87

Troisieme jeu de gabarits du projet, et pour la meme raison que le deuxieme : mesure sur
hors_combat.png, les gabarits de COORDONNEES y plafonnent a 0,66-0,71 quand le seuil
d'acceptation est a 0,90, et les chiffres y font 9 px de haut contre 13. Ce n'est pas un
seuil trop strict, c'est une autre police.

On n'INVENTE pas les etiquettes, on les fournit : chaque capture est passee avec la valeur
lue a l'oeil. Le script refuse si le nombre de glyphes ne correspond pas, plutot que
d'aligner au hasard -- une etiquette decalee produirait des gabarits faux, donc des
lectures fausses PARTOUT.

La pastille n'apparait QUE HORS COMBAT : les captures doivent l'etre.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from jev_tactics.perception.digits import DigitTemplates, normalize_glyph
from jev_tactics.perception.pods import glyphs_of

DEFAULT_OUT = "src/jev_tactics/perception/data/pods_templates.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", metavar="IMAGE=POURCENTAGE")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    exemplars, digits = [], []
    for pair in args.pairs:
        if "=" not in pair:
            sys.exit(f"attendu « fichier.png=59 », recu : {pair!r}")
        path, value = pair.split("=", 1)
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f"image illisible : {path}")
        group = glyphs_of(frame)
        text = value.strip().rstrip("%")
        if len(group) != len(text):
            sys.exit(f"{path} : {len(group)} glyphe(s) pour « {text} » "
                     f"({len(text)} chiffres) — etiquette fausse, pastille absente "
                     f"(capture EN COMBAT ?), ou ROI a cote")
        for (_x, _y, _w, _h, patch), char in zip(group, text):
            exemplars.append(normalize_glyph(patch))
            digits.append(int(char))
        print(f"  {path} -> {text}%")

    if not exemplars:
        sys.exit("aucun glyphe collecte")
    templates = DigitTemplates(exemplars=np.array(exemplars, dtype=np.float32),
                               digits=np.array(digits, dtype=np.int64))
    templates.save(args.out)
    covered = templates.covered_digits()
    missing = sorted(set(range(10)) - covered)
    print(f"\n{len(exemplars)} exemplaires -> {args.out}")
    print(f"chiffres couverts : {sorted(covered)}")
    if missing:
        print(f"MANQUANTS : {missing} — tout taux en contenant sera REFUSE, pas devine.")
        print("Ajouter des captures HORS COMBAT a d'autres taux de remplissage.")


if __name__ == "__main__":
    main()
