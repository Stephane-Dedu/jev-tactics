"""Lecture de l'infobulle d'un MONSTRE survole : ses points de vie exacts.

Raison d'etre, chiffree. Sur les 32 captures du depot, 72 ennemis sont detectes et AUCUN
n'a de PV connus : ni l'OCR (reserve au joueur) ni la timeline (qui ne donne qu'un ratio
sans maximum) ne les fournit. Sans eux, `evaluate` ne peut ni plafonner les degats ni
accorder la prime de mise a mort -- le terme le plus lourd du score. Mesure en arene :

    PV ennemis connus      85,8 %      70,0 % a trois ennemis
    PV ennemis inconnus    36,7 %      50,0 %

Quarante-neuf points. C'est de loin le plus gros ecart mesure dans ce projet, et il tient
entierement a une lecture qui n'existait pas.

Ce que l'infobulle donne, releve sur data/runs/20260807-224524.png :

    Bouftou
    139 / 186 (74%)        <- ce qu'on lit ici
    Niv. 24                <- disponible, non lu

QUATRIEME JEU DE GABARITS du projet, et pour la meme raison que les trois autres : la
police differe. Les chiffres y font 8 a 10 px de haut, sur fond sombre, et la valeur
courante est en BLANC quand le maximum est en GRIS -- deux teintes, une seule forme.

Les etiquettes sont FOURNIES, jamais devinees (cf. scripts/build_mob_templates.py), et
toute lecture partielle est refusee : un maximum de PV faux ferait croire un monstre
mourant ou invulnerable, et la prime de mise a mort deciderait sur cette base.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.digits import DigitTemplates, normalize_glyph

# Texte de l'infobulle : clair et peu sature, sur fond sombre.
MIN_VALUE = 150
MAX_SATURATION = 90
# Un glyphe de chiffre mesure 8 a 10 px de haut ; en dessous c'est un point ou une bordure.
MIN_DIGIT_HEIGHT = 7
MAX_DIGIT_HEIGHT = 14
MAX_DIGIT_WIDTH = 14
# Ecart horizontal au-dela duquel deux glyphes appartiennent a deux NOMBRES differents.
GROUP_GAP = 8
MIN_SCORE = 0.90
_TEMPLATES_FILE = "mob_templates.npz"
_TEMPLATES: DigitTemplates | None = None


@dataclass(frozen=True)
class MobReading:
    """PV courants et maximum d'un monstre survole, ou la RAISON du refus."""

    hp: int | None = None
    hp_max: int | None = None
    reason: str = ""


def load_templates() -> DigitTemplates:
    global _TEMPLATES
    if _TEMPLATES is None:
        ref = resources.files("jev_tactics.perception") / "data" / _TEMPLATES_FILE
        with resources.as_file(ref) as path:
            _TEMPLATES = DigitTemplates.load(str(path))
    return _TEMPLATES


def glyph_groups(frame: NDArray[np.uint8],
                 roi: tuple[int, int, int, int]) -> list[list[tuple[int, NDArray]]]:
    """Glyphes de la ROI, regroupes en NOMBRES par leurs espacements.

    Partage avec le constructeur de gabarits : segmenter deux fois, c'est se donner deux
    occasions de diverger -- le projet l'a deja paye ailleurs.
    """
    x0, y0, x1, y1 = roi
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > MIN_VALUE) & (hsv[:, :, 1] < MAX_SATURATION)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8) * 255, connectivity=8)
    glyphs = []
    for index in range(1, count):
        left, top, width, height = (int(stats[index, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if not MIN_DIGIT_HEIGHT <= height <= MAX_DIGIT_HEIGHT:
            continue
        if width > MAX_DIGIT_WIDTH:
            continue
        patch = (labels[top:top + height, left:left + width] == index)
        glyphs.append((left, width, patch.astype(np.uint8) * 255))
    glyphs.sort(key=lambda g: g[0])

    groups: list[list[tuple[int, NDArray]]] = []
    end = -10_000
    for left, width, patch in glyphs:
        if groups and left - end <= GROUP_GAP:
            groups[-1].append((left, patch))
        else:
            groups.append([(left, patch)])
        end = left + width
    return groups


def _number(group, templates) -> int | None:
    digits = []
    for _left, patch in group:
        digit, score = templates.classify(normalize_glyph(patch))
        if digit is None or score < MIN_SCORE:
            return None
        digits.append(digit)
    return int("".join(str(d) for d in digits)) if digits else None


def has_tooltip(frame: NDArray[np.uint8],
                roi: tuple[int, int, int, int],
                minimum: int = 2) -> bool:
    """Une infobulle de monstre est-elle ouverte ? -> booleen, SANS lire les chiffres.

    POURQUOI SEPAREMENT DE `read_mob_hp`. Ce que la chasse doit trancher n'est pas
    « combien de PV », c'est « y a-t-il un monstre sous le curseur ». Les deux questions
    ont un cout tres different :

        survoler un candidat, puis lire        0,35 s     TOOLTIP_DELAY
        cliquer un candidat, puis constater    8 s        ENGAGE_TIMEOUT

    Vingt fois moins cher, et le journal reel ne contient QUE des clics rates. Or
    `read_mob_hp` refuse aujourd'hui la moitie des infobulles qu'il voit, faute de
    gabarits de chiffres (4, 5 et 7 manquent) : s'en servir pour repondre a la question
    de la chasse ferait passer un monstre pour du decor a cause d'un gabarit absent.

    Cette fonction ne classe donc AUCUN glyphe. Elle compte les nombres segmentes, ce qui
    ne depend d'aucun jeu de gabarits et reste vrai le jour ou ils seront completes.

    MESURE, sur les 24 captures de `data/runs/` et la ROI de combat (740, 820, 836, 845) :

        22 captures sans survol   ->  0 nombre     aucun faux positif
         2 captures avec survol   ->  2 nombres    dont une que `read_mob_hp` refuse

    CE QUE LA MESURE NE DIT PAS. `minimum` vaut 2 par ARGUMENT, pas par mesure : sur ces
    24 captures, 1 et 2 rendent exactement le meme verdict (contre-epreuve faite, la suite
    reste verte avec 1). Deux est retenu parce qu'un glyphe clair isole -- un chiffre
    d'interface, une valeur de degats qui passe -- suffirait a tromper un seuil a un. Mais
    c'est un raisonnement, et le jour ou l'infobulle hors combat ne portera qu'un nombre,
    c'est ce raisonnement qu'il faudra revoir : d'ou le parametre.

    LIMITE A CONNAITRE : ces 24 captures sont EN COMBAT, et la ROI ci-dessus est celle de
    l'infobulle de combat. Hors combat, sur la carte, l'infobulle d'un groupe s'affiche
    ailleurs -- la mesure vaut pour le PRINCIPE, pas pour cette ROI-la. C'est pourquoi la
    ROI est un argument et non une constante : elle doit etre relevee sur une capture hors
    combat avant de servir a la chasse.
    """
    return len(glyph_groups(frame, roi)) >= minimum


def read_mob_hp(frame: NDArray[np.uint8],
                roi: tuple[int, int, int, int]) -> MobReading:
    """Frame + ROI de la ligne de vie -> PV courants et maximum.

    Tout-ou-rien, comme les trois autres lecteurs de chiffres du projet. Un maximum faux
    ferait croire un monstre mourant ou invulnerable, et la prime de mise a mort -- le
    terme le plus lourd du score -- deciderait sur cette base.
    """
    groups = glyph_groups(frame, roi)
    if len(groups) < 2:
        return MobReading(reason=f"{len(groups)} nombre(s) lu(s), deux attendus "
                                 f"(courant et maximum)")
    templates = load_templates()
    current, maximum = _number(groups[0], templates), _number(groups[1], templates)
    missing = sorted(set(range(10)) - templates.covered_digits())
    if current is None or maximum is None:
        return MobReading(reason="chiffre rejete"
                                 + (f" — gabarits MANQUANTS : {missing}" if missing else ""))
    if not 0 < current <= maximum:
        return MobReading(reason=f"{current}/{maximum} : incoherent, lecture rejetee")
    return MobReading(hp=current, hp_max=maximum)
