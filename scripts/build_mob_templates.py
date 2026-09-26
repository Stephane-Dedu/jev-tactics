"""Fabrique les gabarits de chiffres de l'INFOBULLE DE MONSTRE.

    python scripts/build_mob_templates.py data/runs/20260807-224524.png=139,186                                           data/runs/20260807-221715.png=200,

Une etiquette VIDE ignore le nombre correspondant : ci-dessus, le maximum « 200 » de la
seconde capture est en gris et se fragmente, la ou sa valeur courante segmente bien.

Quatrieme jeu de gabarits du projet, meme methode que les trois autres : les etiquettes
sont FOURNIES, jamais devinees, et le script refuse si le compte de glyphes ne correspond
pas -- une etiquette decalee produirait des gabarits faux, donc des lectures fausses
PARTOUT.

`--roi` designe la ligne de vie de l'infobulle. Le defaut est celui de la capture
ci-dessus ; l'infobulle SUIT LE CURSEUR, donc ce rectangle n'a rien d'universel et devra
etre ancre (sur l'icone de coeur, par exemple) avant tout usage en session.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from jev_tactics.perception.digits import DigitTemplates, normalize_glyph
from jev_tactics.perception.mobinfo import glyph_groups

DEFAULT_OUT = "src/jev_tactics/perception/data/mob_templates.npz"
DEFAULT_ROI = (740, 820, 836, 845)
# Largeur minimale d'un chiffre de cette police, le « 1 » excepte. Mesure :
# 5 a 7 px pour 0, 3, 6, 8, 9 ; 3 px pour le 1.
MIN_DIGIT_WIDTH = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", metavar="IMAGE=COURANT,MAX")
    ap.add_argument("--roi", nargs=4, type=int, default=list(DEFAULT_ROI))
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    exemplars, digits = [], []
    for pair in args.pairs:
        if "=" not in pair:
            sys.exit(f"attendu « fichier.png=139,186 », recu : {pair!r}")
        path, values = pair.split("=", 1)
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f"image illisible : {path}")
        groups = glyph_groups(frame, tuple(args.roi))
        labels = values.split(",")
        if len(groups) != len(labels):
            sys.exit(f"{path} : {len(groups)} nombre(s) segmente(s) pour "
                     f"{len(labels)} etiquette(s) — ROI a cote ?")
        for group, text in zip(groups, labels, strict=True):
            # Etiquette VIDE = groupe volontairement ignore. Sert quand un des deux
            # nombres ne se segmente pas proprement : sur data/runs/20260807-221715.png,
            # le maximum « 200 » est en GRIS et son « 2 » se fragmente en deux
            # composantes de 3 et 4 px, a tous les seuils essayes (150, 130, 110, 90).
            # La valeur COURANTE, en blanc, y segmente parfaitement -- il serait absurde
            # de perdre ses trois chiffres parce que son voisin resiste.
            if not text:
                continue
            if len(group) != len(text):
                sys.exit(f"{path} : {len(group)} glyphe(s) pour « {text} » "
                         f"({len(text)} chiffres) — etiquette fausse ou ROI a cote")
            # Le compte de glyphes ne suffit PAS, et cela s'est vu. Sur une seconde
            # capture, « 200 » s'est segmente en trois fragments de 3, 4 et 5 px de large
            # -- le controle de compte est passe par coincidence, et le gabarit du « 2 »
            # obtenu etait une BARRE VERTICALE. Elle matchait ensuite le « 1 » a egalite,
            # et la marge du classifieur rejetait alors toute lecture contenant un 1.
            #
            # Une capture mal etiquetee ne degrade pas le lecteur : elle le casse pour
            # tous les autres chiffres. D'ou ce controle de LARGEUR : les chiffres de
            # cette police font 5 a 7 px, le « 1 » excepte.
            widths = [patch.shape[1] for _left, patch in group]
            suspects = [(c, w) for c, w in zip(text, widths, strict=True)
                        if c != "1" and w < MIN_DIGIT_WIDTH]
            if suspects:
                sys.exit(f"{path} : glyphe(s) trop etroit(s) pour leur etiquette "
                         f"{suspects} — segmentation fragmentee, gabarits refuses")
            for (_left, patch), char in zip(group, text, strict=True):
                exemplars.append(normalize_glyph(patch))
                digits.append(int(char))
        print(f"  {path} -> {' / '.join(labels)}")

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
        print(f"MANQUANTS : {missing} — toute lecture en contenant sera REFUSEE.")


if __name__ == "__main__":
    main()
