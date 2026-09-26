"""Fabrique les exemplaires de chiffres de l'HDV (police 10 px), offline, une fois.

    python scripts/build_hdv_templates.py
    python scripts/build_hdv_templates.py --review out/   # dump PNG des glyphes etiquetes

POURQUOI PAS TESSERACT. `build_digit_templates.py` (police 14 px du HUD) utilise
Tesseract comme oracle d'etiquetage : personne ne connaissait les valeurs affichees sur
les captures, il fallait bien les faire lire par quelqu'un. Ici la situation est
differente -- les captures de reference ont ete relues et VALIDEES a l'oeil, nombre par
nombre. La verite terrain est donc ecrite ci-dessous.

C'est strictement meilleur qu'un oracle : etiquettes exactes, aucune erreur d'OCR a
relire ensuite, et le projet ne gagne pas une dependance externe (Tesseract n'est pas
installe sur cette machine) pour un travail fait une seule fois.

Le garde-fou reste le meme que dans l'autre constructeur : si le nombre de glyphes
segmentes ne correspond pas au nombre de chiffres attendu, la ROI est SAUTEE bruyamment.
Un desaccord veut dire que la geometrie ou la segmentation a bouge, et etiqueter quand
meme melangerait les classes -- exactement le genre d'erreur qui ne se voit qu'au moment
ou un prix faux est deja dans le CSV.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.perception.digits import GLYPH_SIZE, DigitTemplates
from jev_tactics.perception.hdv import (
    DETAIL_AVERAGE,
    DETAIL_LOT_X,
    DETAIL_PRICE_X,
    DETAIL_ROW_TOPS,
    LIST_LEVEL_X,
    LIST_PRICE_X,
    LIST_ROW_TOPS,
    LOT_VALUES,
    MIN_VALUE_BRIGHT,
    MIN_VALUE_DIM,
    row_roi,
    segment,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA = ROOT / "src" / "jev_tactics" / "perception" / "data" / "hdv_digit_templates.npz"

# Verite terrain, relue et validee sur les captures. `None` = ligne absente ou vide.
#
# Couverture visee : les dix chiffres. HDV.png et 5.png donnent 0-4 et 6-9 ; le 5 vient
# de 5.png (450, 50, 1359) et cherche.png ajoute de la variete sur les grands nombres
# (jusqu'a 499 997, six chiffres).
REFERENCES: dict[str, dict[str, object]] = {
    "HDV.png": {
        "detail_prices": (791, 7989, 69989),
        "detail_average": 909,
        "list_levels": (90, 130, 100, 86, 110),
        "list_prices": (2194, 646, 909, 322, 1781),
    },
    "5.png": {
        "detail_prices": (450, 4000, 89993),
        "detail_average": 1022,
        "list_levels": (50, 50, 50, 50, 200, 200, 114, 170),
        "list_prices": (727, 311, 1022, 120, 38347, 1359, 691, 1018182),
    },
    "cherche.png": {
        "detail_prices": None,   # panneau de detail ferme sur cette capture
        "detail_average": None,
        "list_levels": (122, 41, 150, 190, 1, 1, 1, 1),
        "list_prices": (3998, 5947, 45979, 47981, 397999, 499992, 499996, 499997),
    },
    # --- Captures prises PAR LE BOT LUI-MEME (mss), et non a l'outil de capture de
    # Windows. La distinction n'est pas cosmetique : sur une zone d'interface pourtant
    # immobile, les deux chemins ne donnent que 97,2 % de pixels identiques. Les gabarits
    # tires des captures Windows decrivaient donc un rendu que le bot ne rencontre jamais,
    # et un « 8 » de niveau a ete rejete en jeu pour 0,867 -- la taille du glyphe est
    # ouverte au milieu dans le rendu mss, fermee dans l'autre.
    #
    # Toute capture ajoutee ici doit venir de `doctor.py --live --save-frame` ou d'une
    # frame conservee par le releve : c'est le seul rendu qui compte.
    "data/hdv/ref/live_ivoire.png": {
        "detail_prices": None,      # panneau de detail ferme sur cette capture
        "detail_average": None,
        "list_levels": (80,),
        "list_prices": (168,),
    },
    "data/hdv/ref/live_ivoire2.png": {
        "detail_prices": (257, 2124, 17998),
        "detail_average": 168,
        "list_levels": (80,),
        "list_prices": (168,),
    },
    # QUATRE lots : le jeu en affiche un de 1 000 quand les offres le permettent.
    "data/hdv/ref/live1.png": {
        "detail_prices": (57, 479, 9000, 65000),
        "detail_average": 94,
        "list_levels": (86,),
        "list_prices": (94,),
    },
    "data/hdv/ref/live_coque.png": {
        "detail_prices": (105, 1095, 12995, 149995),
        "detail_average": 150,
        "list_levels": (),          # « Aucun element ne correspond a votre recherche »
        "list_prices": (),
    },
}


def _harvest_roi(frame: np.ndarray, roi: tuple[int, int, int, int], expected: int,
                 label: str, out: list, digits: list, review: str | None,
                 min_value: int = MIN_VALUE_BRIGHT) -> None:
    x0, y0, x1, y1 = roi
    glyphs = segment(frame[y0:y1, x0:x1], min_value)
    text = str(expected)
    if len(glyphs) != len(text):
        print(f"  [saute] {label}: {len(glyphs)} glyphes pour '{text}' "
              f"({len(text)} attendus)")
        return
    for position, (char, glyph) in enumerate(zip(text, glyphs)):
        out.append(glyph)
        digits.append(int(char))
        if review:
            os.makedirs(review, exist_ok=True)
            name = f"{label.replace('/', '_')}_{position}_{char}.png"
            cv2.imwrite(os.path.join(review, name), (glyph * 255).astype(np.uint8))


def harvest(review: str | None) -> tuple[np.ndarray, np.ndarray]:
    exemplars: list[np.ndarray] = []
    digits: list[int] = []
    for name, truth in REFERENCES.items():
        path = ROOT / name
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  [saute] capture illisible : {path}")
            continue
        print(f"{name} :")

        detail_prices = truth["detail_prices"]
        if detail_prices:
            for lot, top, price in zip(LOT_VALUES, DETAIL_ROW_TOPS, detail_prices):
                # La colonne Lot est elle aussi une source d'exemplaires : 1, 10, 100
                # sont connus par construction et donnent des 0 et des 1 propres.
                _harvest_roi(frame, row_roi(top, DETAIL_LOT_X), lot,
                             f"{name}/lot{lot}", exemplars, digits, review)
                _harvest_roi(frame, row_roi(top, DETAIL_PRICE_X), price,
                             f"{name}/prix{lot}", exemplars, digits, review)
        average = truth["detail_average"]
        if average:
            _harvest_roi(frame, DETAIL_AVERAGE, average,
                         f"{name}/moyen", exemplars, digits, review)

        for index, (top, level, price) in enumerate(
                zip(LIST_ROW_TOPS, truth["list_levels"], truth["list_prices"])):
            # La colonne Niveau est en gris : son propre seuil, cf. hdv.MIN_VALUE_DIM.
            _harvest_roi(frame, row_roi(top, LIST_LEVEL_X), level,
                         f"{name}/niveau{index}", exemplars, digits, review,
                         MIN_VALUE_DIM)
            _harvest_roi(frame, row_roi(top, LIST_PRICE_X), price,
                         f"{name}/prixliste{index}", exemplars, digits, review)

    return (np.asarray(exemplars, dtype=np.float32).reshape(-1, GLYPH_SIZE, GLYPH_SIZE),
            np.asarray(digits, dtype=np.int64))


def report(templates: DigitTemplates) -> None:
    """Couverture, puis classification EN EXCLUANT SOI-MEME (leave-one-out).

    La v1 de ce rapport donnait l'accord intra-classe minimal, comme le constructeur
    14 px. Ici cette statistique est trompeuse et annoncait une marge NEGATIVE alors que
    la lecture etait parfaite : l'HDV melange deux populations de texte (prix blancs
    seuillés a 180, niveaux gris seuilles a 140), donc deux epaisseurs de trait pour un
    meme chiffre. Comparer un « 0 » gris a un « 0 » blanc donne un accord bas sans que ce
    soit un probleme -- `classify` prend le MAXIMUM sur les exemplaires d'une classe, il
    lui suffit qu'UN exemplaire ressemble.

    Ce qui predit vraiment un rejet, c'est : en retirant un exemplaire du jeu, est-il
    encore classe correctement, et avec quelle marge ? C'est ce qu'on mesure.
    """
    covered = templates.covered_digits()
    missing = sorted(set(range(10)) - covered)
    print(f"\ncouverture : {sorted(covered)}"
          + (f"  MANQUANTS : {missing}" if missing else "  (complete)"))

    ex, dg = templates.exemplars, templates.digits
    correct, rejected, wrong = 0, 0, 0
    worst = 1.0
    for i in range(len(ex)):
        keep = np.arange(len(ex)) != i
        reduced = DigitTemplates(exemplars=ex[keep], digits=dg[keep])
        digit, score = reduced.classify(ex[i])
        if digit is None:
            rejected += 1
        elif digit == dg[i]:
            correct += 1
            worst = min(worst, score)
        else:
            wrong += 1
            print(f"  !! exemplaire {i} etiquete {dg[i]} classe {digit} (score {score:.3f})")
    total = len(ex)
    print(f"leave-one-out : {correct}/{total} exacts, {rejected} rejetes, {wrong} FAUX")
    print(f"score minimal des reconnaissances exactes = {worst:.3f} "
          f"(seuil MIN_SCORE = 0.90)")
    if wrong:
        print("  un seul FAUX invalide le jeu : un chiffre confondu est un prix faux.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PACKAGE_DATA))
    parser.add_argument("--review", help="dossier ou dumper les glyphes etiquetes")
    args = parser.parse_args()

    exemplars, digits = harvest(args.review)
    if not len(exemplars):
        print("aucun exemplaire recolte")
        return 1
    templates = DigitTemplates(exemplars=exemplars, digits=digits)
    templates.save(args.out)
    print(f"\n{len(exemplars)} exemplaires -> {args.out}")
    report(templates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
