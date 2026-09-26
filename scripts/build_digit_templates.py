"""Fabrique les exemplaires de chiffres avec Tesseract en oracle (offline, une fois).

    python scripts/build_digit_templates.py                # captures dofus*.png par defaut
    python scripts/build_digit_templates.py img1.png ...   # captures explicites
    python scripts/build_digit_templates.py --review out/  # dump PNG des glyphes etiquetes

Pour chaque ROI de chaque capture : Tesseract lit la valeur (verite terrain), on extrait
les glyphes tries de gauche a droite, et si les comptes concordent chaque glyphe recoit
son chiffre. Les exemplaires sont sauves DANS le paquet
(src/jev_tactics/perception/data/digit_templates.npz) puis committes.

Le rapport final donne la couverture (chiffres jamais vus -> ils seront REJETES puis lus
par le fallback Tesseract a l'execution : capturer des ecrans contenant ces chiffres et
relancer ce script pour completer) et la separation intra/inter-classe qui justifie les
seuils MIN_SCORE / MIN_MARGIN de digits.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.perception.digits import (
    MIN_MARGIN,
    MIN_SCORE,
    DigitTemplates,
    extract_glyphs,
)
from jev_tactics.perception.ui import DEFAULT_ROIS, HEART_ROI, _glyph_mask, _ocr, _preprocess

# LA MOISSON NE VOYAIT QU'UN SEUL ATH. `DEFAULT_ROIS` decrit l'orbe a deux nombres ; sur
# l'ATH ou les PV tiennent dans un seul nombre centre, ces boites ne rendent rien et
# aucun glyphe n'en sortait. Les chiffres 3 et 8 manquaient d'ailleurs a la couverture,
# ce qui renvoyait chaque nombre les contenant vers Tesseract -- 203 ms par lecture au
# lieu des 1,3 ms annonces. Moissonner les DEUX dispositions ferme les deux trous.
HARVEST_ROIS = (*DEFAULT_ROIS, HEART_ROI)

PACKAGE_DATA = (
    Path(__file__).resolve().parents[1]
    / "src" / "jev_tactics" / "perception" / "data" / "digit_templates.npz"
)
DEFAULT_GLOB = str(Path.home() / "Pictures" / "Screenshots" / "dofus*.png")


def harvest(image_paths: list[str], review_dir: str | None) -> tuple[np.ndarray, np.ndarray]:
    exemplars: list[np.ndarray] = []
    digits: list[int] = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  [skip] illisible : {path}")
            continue
        for roi in HARVEST_ROIS:
            crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
            truth = re.findall(r"\d+", _ocr(_preprocess(crop)))  # oracle Tesseract
            value = "".join(truth)
            glyphs = extract_glyphs(_glyph_mask(crop))
            if not value or len(glyphs) != len(value):
                print(f"  [skip] {Path(path).name}/{roi.name}: oracle='{value}' "
                      f"vs {len(glyphs)} glyphes")
                continue
            for pos, (char, glyph) in enumerate(zip(value, glyphs)):
                exemplars.append(glyph)
                digits.append(int(char))
                if review_dir:
                    os.makedirs(review_dir, exist_ok=True)
                    out = f"{review_dir}/{char}_{Path(path).stem}_{roi.name}_{pos}.png"
                    cv2.imwrite(out, (glyph * 255).astype(np.uint8))
            print(f"  [ok]   {Path(path).name}/{roi.name}: '{value}' ({len(glyphs)} glyphes)")
    return np.asarray(exemplars, dtype=np.float32), np.asarray(digits, dtype=np.int64)


def report(templates: DigitTemplates) -> None:
    covered = templates.covered_digits()
    missing = set(range(10)) - covered
    print(f"\ncouverture : {sorted(covered)}")
    if missing:
        print(f"/!\\ chiffres jamais vus : {sorted(missing)} -> rejetes a l'execution "
              f"(fallback Tesseract) ; capturer des ecrans les contenant puis relancer.")

    # LE CONTROLE COMPARAIT UN MINIMUM GLOBAL A UN MAXIMUM GLOBAL, sur deux exemplaires
    # DIFFERENTS. « intra min 0,855 <= inter max 0,883 » ne decrit alors aucun conflit
    # reel : le glyphe le plus isole de sa classe et la paire inter-classes la plus
    # ressemblante n'ont rien a voir l'un avec l'autre. Le controle tenait tant que le
    # jeu etait petit (33 exemplaires) et criait des qu'il s'enrichissait -- exactement
    # a rebours de ce qu'on veut d'un garde-fou.
    #
    # Deux mesures le remplacent, toutes deux PAR EXEMPLAIRE :
    #   - la separation : un exemplaire doit ressembler davantage a sa classe qu'a une
    #     autre. Un echec ici signale un glyphe MAL ETIQUETE par l'oracle ;
    #   - le leave-one-out : ce que le classifieur repondrait vraiment, regle de rejet
    #     comprise. C'est la seule mesure qui parle de son comportement a l'execution.
    ex, dg = templates.exemplars, templates.digits
    mal_etiquetes = []
    bons = rejetes = faux = 0
    for i in range(len(ex)):
        scores = 1.0 - np.abs(ex - ex[i]).mean(axis=(1, 2))
        scores[i] = -1.0  # ne pas se comparer a soi-meme
        same = scores[dg == dg[i]]
        other = scores[dg != dg[i]]
        if same.size and other.size and other.max() > same.max():
            mal_etiquetes.append((i, int(dg[i]), int(dg[int(scores.argmax())])))
        best_by_class: dict[int, float] = {}
        for digit, score in zip(dg, scores):
            d = int(digit)
            if score > best_by_class.get(d, -1.0):
                best_by_class[d] = float(score)
        ranked = sorted(best_by_class.items(), key=lambda kv: kv[1], reverse=True)
        best_digit, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best < MIN_SCORE or best - second < MIN_MARGIN:
            rejetes += 1
        elif best_digit == int(dg[i]):
            bons += 1
        else:
            faux += 1

    print(f"separation : {len(ex) - len(mal_etiquetes)}/{len(ex)} exemplaires plus proches "
          f"de leur propre classe que d'une autre")
    for i, etiquette, voisin in mal_etiquetes[:10]:
        print(f"    /!\\ exemplaire {i} etiquete {etiquette} ressemble surtout a un {voisin} "
              f"— oracle probablement fautif")
    print(f"leave-one-out (regle de rejet comprise) : {bons} corrects, {rejetes} rejetes "
          f"-> fallback Tesseract, {faux} FAUX")
    if faux:
        print("/!\\ des lectures FAUSSES passent la regle de rejet — ne pas committer : "
              "un nombre faux est pire qu'un nombre absent.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="captures (defaut : dofus*.png)")
    ap.add_argument("--review", help="repertoire de dump PNG des glyphes etiquetes")
    ap.add_argument("--out", default=str(PACKAGE_DATA))
    args = ap.parse_args()

    paths = args.images or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        raise SystemExit(f"aucune capture trouvee ({DEFAULT_GLOB})")
    print(f"captures : {[Path(p).name for p in paths]}")

    exemplars, digits = harvest(paths, args.review)
    if not len(exemplars):
        raise SystemExit("aucun glyphe recolte — verifier les ROIs / captures")

    templates = DigitTemplates(exemplars=exemplars, digits=digits)
    templates.save(args.out)
    print(f"\n{len(exemplars)} exemplaires -> {args.out}")
    report(templates)


if __name__ == "__main__":
    main()
