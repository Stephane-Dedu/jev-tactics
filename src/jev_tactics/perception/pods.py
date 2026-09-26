"""Lecture du taux de remplissage de l'inventaire, affiche en haut a gauche.

Ce badge a longtemps ete tenu pour inexistant. `--pods-full` a meme ete RETIRE parce
qu aucun lecteur ne pouvait exister -- on croyait devoir ouvrir l'inventaire. Il est en
fait affiche en permanence : une pastille sombre portant une icone et « 59% », mesuree a
x 163..218, y 65..94 sur une capture 1920x1080.

TROIS CHOSES MESUREES, et chacune compte :

  - le badge n'apparait QUE HORS COMBAT. Sur sept captures reelles, une seule le montre --
    la seule prise hors combat. C'est sans consequence ici : la recolte, qui en a besoin,
    tourne hors combat ;
  - sa police n'est NI celle de l'UI, NI celle des coordonnees. Les gabarits de
    coordonnees y plafonnent a 0,66-0,71 pour un seuil a 0,90, et ses chiffres font 9 px
    de haut contre 13. D'ou un troisieme jeu de gabarits ;
  - le fond de la pastille est SOMBRE, contrairement aux coordonnees qui sont posees sur
    le decor. Le masque blanc y est donc bien plus net.

COUVERTURE HONNETE : une seule capture disponible, donc seuls les chiffres 5 et 9 sont
couverts. Tout le reste est REFUSE, pas devine -- une lecture fausse ferait croire
l'inventaire plein et arreterait la session, ou l'inverse. Deux captures a des taux
differents suffisent a elargir :

    python scripts/build_pods_templates.py plein.png=87 vide.png=12
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.coordinates import (
    MAX_DIGIT_HEIGHT,
    WHITE_BAND,
    _blobs,
    _group_by_gap,
)
from jev_tactics.perception.digits import MIN_SCORE, DigitTemplates, normalize_glyph

# Pastille du taux de remplissage, mesuree sur hors_combat.png (1920x1080).
PODS_ROI = (160, 63, 222, 96)
# Les chiffres y font 9 px, contre 13 pour les coordonnees : le plancher est plus bas.
MIN_DIGIT_HEIGHT = 7
# La pastille a un FOND SOMBRE, contrairement au reste du bandeau qui est pose sur le
# decor. C'est ce qui dit si elle est LA, avant meme de chercher des chiffres. Mesure sur
# sept captures : 0,50 quand elle est affichee (hors combat), 0,01 a 0,02 sinon. Le seuil
# tombe dans un intervalle vide, ce qui est rare et confortable.
BADGE_DARKNESS = 0.20
_PODS_TEMPLATES = "pods_templates.npz"


@dataclass(frozen=True)
class PodsReading:
    """Taux lu, ou POURQUOI il ne l'a pas ete.

    Meme contrat que les coordonnees : « illisible » ne se corrige pas, la raison si.
    """

    ratio: float | None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ratio is not None


def load_pods_templates() -> DigitTemplates:
    from importlib import resources

    ref = resources.files("jev_tactics.perception") / "data" / _PODS_TEMPLATES
    with resources.as_file(ref) as path:
        return DigitTemplates.load(path)


def glyphs_of(frame: NDArray[np.uint8]) -> list:
    """Glyphes de la pastille, de gauche a droite. Partage par le lecteur et le
    fabricant de gabarits, pour qu'ils ne puissent pas decouper differemment."""
    x0, y0, x1, y1 = PODS_ROI
    height, width = frame.shape[:2]
    if x1 > width or y1 > height:
        return []
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(WHITE_BAND[0], np.uint8),
                       np.array(WHITE_BAND[1], np.uint8))
    # `_blobs` rend desormais l'ordonnee : elle sert cote coordonnees a exiger qu'un signe
    # negatif soit A LA HAUTEUR du nombre qu'il signe. Ici on ne s'en sert pas, mais la
    # forme doit suivre -- `_group_by_gap` lit la largeur par sa position.
    tall = [(x, y, w, h, patch) for x, y, w, h, _a, patch in _blobs(mask)
            if MIN_DIGIT_HEIGHT <= h <= MAX_DIGIT_HEIGHT]
    groups = _group_by_gap(tall)
    # Le dernier groupe est le nombre : ce qui precede est l'icone de la pastille, dont
    # les traits clairs ont parfois la hauteur d'un chiffre.
    return groups[-1] if groups else []


def _darkness(frame: NDArray[np.uint8]) -> float:
    """Part de pixels sombres dans la zone de la pastille."""
    x0, y0, x1, y1 = PODS_ROI
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    return float((crop.max(axis=2) < 70).mean())


def diagnose_pods(
    frame: NDArray[np.uint8], templates: DigitTemplates | None = None
) -> PodsReading:
    """Frame BGR -> taux de remplissage dans [0, 1], avec la raison d'un echec."""
    _, _, x1, y1 = PODS_ROI
    height, width = frame.shape[:2]
    if x1 > width or y1 > height:
        return PodsReading(None, f"capture {width}x{height} trop petite pour {PODS_ROI}")

    # La presence de la PASTILLE se verifie AVANT de lire des chiffres. Sans cela, le
    # texte « - Niveau 30 » qui passe dans la meme zone en combat etait pris pour un taux
    # et rejete comme « chiffre illisible » -- un refus juste, pour une raison fausse.
    if _darkness(frame) < BADGE_DARKNESS:
        return PodsReading(None, "pastille absente — EN COMBAT (elle n'y est pas "
                                 "affichee), ou disposition differente")
    group = glyphs_of(frame)
    if not group:
        return PodsReading(None, "pastille presente mais aucun chiffre lisible dedans")

    templates = templates or load_pods_templates()
    missing = sorted(set(range(10)) - templates.covered_digits())
    digits = []
    for _x, _y, _w, _h, patch in group:
        digit, score = templates.classify(normalize_glyph(patch))
        if digit is None:
            return PodsReading(
                None, f"chiffre rejete (accord {score:.2f}, seuil {MIN_SCORE})"
                      + (f" — gabarits MANQUANTS : {missing}" if missing else ""))
        digits.append(digit)

    value = int("".join(str(d) for d in digits))
    if value > 100:
        # Un pourcentage superieur a 100 est une lecture fausse, pas un inventaire tres
        # plein. La rendre ferait arreter la session pour rien.
        return PodsReading(None, f"{value}% lu : impossible, lecture rejetee")
    return PodsReading(value / 100.0)


def read_pods_ratio(frame: NDArray[np.uint8]) -> float | None:
    """Taux de remplissage, ou None. Voir `diagnose_pods` pour la RAISON."""
    return diagnose_pods(frame).ratio


def blind_spots(threshold: float,
                templates: DigitTemplates | None = None) -> list[int]:
    """Taux entiers >= `threshold` que les gabarits actuels ne savent PAS lire.

    « Chiffres manquants : [0, 1, 3, 4, 7, 8] » est vrai mais ne dit pas ce qui compte.
    Ce qui compte est la BANDE DE DECISION : les taux au-dela du seuil, seuls capables de
    declencher quoi que ce soit. Mesure avec les quatre chiffres couverts et un seuil a
    0,90 : 4 valeurs lisibles sur 11 (92, 95, 96, 99). Le lecteur est donc aveugle
    precisement la ou il sert -- un inventaire qui se remplit passe de 89 a 91 a 93 sans
    qu'aucun ne soit lu, et la session continue a recolter dans le vide.

    Ce n'est pas une panne : c'est une couverture incomplete, qui se comble avec deux
    captures. Mais il fallait pouvoir le DIRE, plutot que de le decouvrir en session.
    """
    covered = (templates or load_pods_templates()).covered_digits()
    start = max(0, min(100, round(threshold * 100)))
    return [value for value in range(start, 101)
            if not {int(c) for c in str(value)} <= covered]
