"""Classifieur maison des chiffres de l'UI (remplace Tesseract sur le chemin chaud).

La police du jeu est fixe et le domaine ferme (10 glyphes) : un plus proche voisin sur
bitmaps normalises suffit. Mesure : Tesseract coutait ~110 ms par ROI ; ce module vise
< 1 ms. Tesseract reste l'oracle hors-ligne qui fabrique les exemplaires
(scripts/build_digit_templates.py) et le fallback quand un glyphe est rejete.

Regle de rejet : un glyphe n'est accepte que si son meilleur score depasse MIN_SCORE
ET domine le deuxieme candidat d'au moins MIN_MARGIN. Un chiffre jamais vu a la
construction des templates (jeu d'exemplaires incomplet) doit etre REJETE, pas devine —
le rejet declenche le fallback Tesseract dans ui.py, et la valeur lue permet d'enrichir
les exemplaires.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

GLYPH_SIZE = 16  # bitmaps normalises GLYPH_SIZE x GLYPH_SIZE

# Seuils de la regle de rejet, calibres sur le rapport de build_digit_templates.py :
# accord intra-classe min = 0.969, inter-classe max = 0.840 -> seuil au milieu de la marge.
MIN_SCORE = 0.90   # accord minimal avec le meilleur exemplaire
MIN_MARGIN = 0.04  # ecart minimal entre meilleure et deuxieme classe

_DEFAULT_TEMPLATES = "digit_templates.npz"  # embarque dans le paquet (perception/data/)


def normalize_glyph(component: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Bitmap binaire d'un glyphe (bbox) -> GLYPH_SIZE^2 en {0.0, 1.0}.

    L'etirement de la bbox vers un carre deforme les glyphes etroits (le '1') mais de
    facon identique a la construction des templates et a la classification — seule la
    coherence compte.
    """
    resized = cv2.resize(component, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)
    return (resized > 127).astype(np.float32)


def extract_glyphs(mask: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
    """Masque binaire (glyphes blancs sur noir, cf. ui._glyph_mask) -> bitmaps normalises,
    tries de gauche a droite (ordre de lecture du nombre)."""
    if mask.size == 0:
        # Masque vide (ROI hors capture) : rien a lire n'est pas une erreur de
        # programme. Voir ui.read_stat pour le pourquoi complet.
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    glyphs: list[tuple[int, NDArray[np.float32]]] = []
    for i in range(1, n):
        x, y, w, h = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                            cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        component = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        glyphs.append((int(x), normalize_glyph(component)))
    return [g for _, g in sorted(glyphs, key=lambda t: t[0])]


@dataclass
class DigitTemplates:
    """Exemplaires par chiffre. On garde TOUS les exemplaires observes (pas une moyenne) :
    le score d'une classe est le max sur ses exemplaires, plus robuste aux variantes."""

    exemplars: NDArray[np.float32]  # (N, GLYPH_SIZE, GLYPH_SIZE)
    digits: NDArray[np.int64]       # (N,) chiffre de chaque exemplaire

    @classmethod
    def load(cls, path: str | Path | None = None) -> DigitTemplates:
        if path is None:
            ref = resources.files("jev_tactics.perception") / "data" / _DEFAULT_TEMPLATES
            with resources.as_file(ref) as p:
                data = np.load(p)
        else:
            data = np.load(path)
        return cls(exemplars=data["exemplars"].astype(np.float32), digits=data["digits"])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, exemplars=self.exemplars, digits=self.digits)

    def covered_digits(self) -> set[int]:
        return set(int(d) for d in self.digits)

    def classify(self, glyph: NDArray[np.float32]) -> tuple[int | None, float]:
        """-> (chiffre, score) ; (None, score) si rejete (score faible ou marge floue)."""
        # Accord pixel a pixel dans [0, 1] contre chaque exemplaire.
        scores = 1.0 - np.abs(self.exemplars - glyph).mean(axis=(1, 2))
        # Meilleur score PAR CLASSE (max sur les exemplaires de la classe).
        best_by_class: dict[int, float] = {}
        for digit, score in zip(self.digits, scores):
            d = int(digit)
            if score > best_by_class.get(d, -1.0):
                best_by_class[d] = float(score)
        ranked = sorted(best_by_class.items(), key=lambda kv: kv[1], reverse=True)
        best_digit, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best < MIN_SCORE or best - second < MIN_MARGIN:
            return None, best
        return best_digit, best


def read_digits_from_mask(mask: NDArray[np.uint8], templates: DigitTemplates) -> int | None:
    """Masque de glyphes -> entier, ou None si un glyphe est rejete (fallback appelant).

    Tout-ou-rien volontaire : un nombre avec UN chiffre douteux est un nombre faux
    (1597 lu 1_97) — mieux vaut rejeter la lecture entiere que renvoyer une valeur sure
    en apparence.
    """
    glyphs = extract_glyphs(mask)
    if not glyphs:
        return None
    out = 0
    for glyph in glyphs:
        digit, _ = templates.classify(glyph)
        if digit is None:
            return None
        out = out * 10 + digit
    return out
