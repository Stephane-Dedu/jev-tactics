"""Fabrique les gabarits de chiffres de la police des COORDONNEES DE CARTE.

    python scripts/build_coord_templates.py tacle.png=3,8 combat1.png=1,29

Pourquoi un second jeu de gabarits. Ceux de l'UI (PV/PA/PM) ne lisent PAS les
coordonnees : mesure sur `tacle.png`, les scores plafonnent entre 0,61 et 0,82 la ou le
seuil d'acceptation est a 0,90. Ce n'est pas un reglage trop strict -- c'est une autre
police, et abaisser le seuil ferait accepter des confusions au lieu de les rejeter.

Le principe est celui de `build_digit_templates.py` : on n'INVENTE pas les etiquettes, on
les fournit. Chaque capture est passee avec sa valeur lue a l'oeil ; le script en deduit
le decoupage et associe chaque glyphe a son chiffre. Une capture mal etiquetee produirait
des gabarits faux, donc le script verifie que le nombre de glyphes correspond a ce qui est
annonce, et refuse sinon.

Couverture : elle depend entierement des captures fournies. Les chiffres absents seront
REJETES a la lecture (tout-ou-rien), pas devines -- une coordonnee a moitie lue est une
position fausse, et une position fausse est pire que pas de position.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.perception.coordinates import (
    COORD_ROI,
    MAX_DIGIT_HEIGHT,
    MIN_DIGIT_HEIGHT,
    WHITE_BAND,
    _blobs,
    _group_by_gap,
)
from jev_tactics.perception.digits import DigitTemplates, normalize_glyph

DEFAULT_OUT = (Path(__file__).resolve().parents[1] / "src" / "jev_tactics" / "perception"
               / "data" / "coord_templates.npz")


def glyphs_of(frame: np.ndarray) -> list[list]:
    x0, y0, x1, y1 = COORD_ROI
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(WHITE_BAND[0], np.uint8),
                       np.array(WHITE_BAND[1], np.uint8))
    tall = [(x, y, w, h, patch) for x, y, w, h, _a, patch in _blobs(mask)
            if MIN_DIGIT_HEIGHT <= h <= MAX_DIGIT_HEIGHT]
    return _group_by_gap(tall)[:2]


def collect(pairs: list[str]) -> tuple[list[np.ndarray], list[int]]:
    exemplars: list[np.ndarray] = []
    digits: list[int] = []

    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"attendu « fichier.png=x,y », recu : {pair!r}")
        path, value = pair.split("=", 1)
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f"image illisible : {path}")
        expected = [v.strip() for v in value.split(",")]
        if len(expected) != 2:
            sys.exit(f"coordonnees attendues sous la forme « x,y », recu : {value!r}")

        groups = glyphs_of(frame)
        if len(groups) != 2:
            sys.exit(f"{path} : {len(groups)} groupe(s) trouve(s) au lieu de 2")

        for group, label in zip(groups, expected):
            text = label.lstrip("-")
            if len(group) != len(text):
                # Refuser plutot que d'aligner au hasard : une etiquette decalee
                # produirait des gabarits faux, et donc des lectures fausses PARTOUT.
                sys.exit(f"{path} : {len(group)} glyphes pour « {label} » "
                         f"({len(text)} chiffres) — etiquette ou ROI incorrecte")
            for (_x, _y, _w, _h, patch), char in zip(group, text):
                exemplars.append(normalize_glyph(patch))
                digits.append(int(char))
        print(f"  {path} -> {expected}")

    return exemplars, digits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", metavar="IMAGE=x,y")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    # COMPLETER plutot que refaire. Sans cela, ajouter le seul chiffre qui manque oblige a
    # repasser TOUTES les captures d'origine dans la meme commande -- et si l'une d'elles
    # a disparu, on perd sa couverture sans s'en apercevoir : le jeu ecrit est celui de la
    # derniere commande, pas la reunion de ce qu'on a appris.
    ap.add_argument("--extend", action="store_true",
                    help="ajouter aux gabarits existants au lieu de les remplacer")
    args = ap.parse_args()

    exemplars, digits = collect(args.pairs)
    if not exemplars:
        sys.exit("aucun glyphe collecte")

    if args.extend and Path(args.out).exists():
        ancien = DigitTemplates.load(args.out)
        avant = sorted(ancien.covered_digits())
        exemplars = list(ancien.exemplars) + exemplars
        digits = list(ancien.digits) + digits
        print(f"  (complete {len(ancien.digits)} exemplaires deja connus, "
              f"chiffres {avant})")

    templates = DigitTemplates(exemplars=np.array(exemplars, dtype=np.float32),
                               digits=np.array(digits, dtype=np.int64))
    templates.save(args.out)

    covered = templates.covered_digits()
    missing = sorted(set(range(10)) - covered)
    print(f"\n{len(exemplars)} exemplaires -> {args.out}")
    print(f"chiffres couverts : {sorted(covered)}")
    if missing:
        print(f"MANQUANTS : {missing} — toute coordonnee en contenant sera REJETEE "
              f"(lecture tout-ou-rien).")
        print("Ajouter des captures ou ces chiffres apparaissent pour completer.")


if __name__ == "__main__":
    main()
