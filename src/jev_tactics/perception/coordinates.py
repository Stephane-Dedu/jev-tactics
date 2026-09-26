"""Lecture des COORDONNEES DE CARTE affichees en haut a gauche.

Ce que ca debloque, et qui manquait a tout le farming :

  - le circuit de recolte est aujourd'hui une suite AVEUGLE de directions. Il est robuste
    -- aucune lecture ratee ne peut le derailler -- mais il ne sait jamais ou il est, donc
    il ne peut ni rejoindre une banque, ni se rattraper apres un deplacement rate ;
  - la verification de changement de carte se fait par comparaison d'images, dont le seuil
    n'a jamais ete valide et qui echouerait sur deux cartes semblables. Comparer deux
    couples de coordonnees ne souffre pas de cette faiblesse.

Le jeu affiche « Amakna (Coin des Bouftous) » puis « 3, 8 - Niveau 20 ». On lit la
seconde ligne, avant le mot « Niveau ».

Mesure sur trois captures reelles (masque blanc, saturation < 60 et valeur > 200) :

    tacle.png          3, 8    chiffres a y=8, hauteur 13, puis « Niveau » a y=13, h=8
    combat1.png        1, 29   le « 1 » ne fait que 5 px de large, les autres 9 a 11
    bugcarreblanc.png  0, 7

Les deux tailles se separent nettement : les chiffres de coordonnees sont HAUTS (12-13 px)
et le texte qui suit est bas (8 px). C'est ce qui permet de s'arreter avant « Niveau »
sans avoir a lire le mot.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.digits import (
    MIN_SCORE,
    DigitTemplates,
    normalize_glyph,
)

# Zone de recherche (x0, y0, x1, y1) pour une disposition 1919x1079.
COORD_ROI = (0, 70, 170, 100)
# Le texte est blanc cerne de noir : peu sature, tres lumineux.
WHITE_BAND = ((0, 0, 200), (179, 60, 255))

# Hauteur des chiffres de coordonnees. Le texte « Niveau NN » qui suit sur la meme ligne
# est plus bas (8 px mesures) : c'est ce qui permet de l'ignorer sans le lire.
MIN_DIGIT_HEIGHT = 10
MAX_DIGIT_HEIGHT = 18
MIN_DIGIT_AREA = 18
# Ecart horizontal au-dela duquel on passe de l'abscisse a l'ordonnee. Mesure : ~3 px
# entre deux chiffres d'un meme nombre, ~15 px entre les deux coordonnees.
COORD_GAP = 9
# Profil d'un tiret : large et plat. MESURE sur hors_combat.png, ou le separateur de
# « 2, 9 - Niveau 20 » fait 5x1 pixels, soit une aire de 5 -- tres en dessous de
# MIN_DIGIT_AREA. Un signe negatif est le MEME glyphe, donc il etait purement et
# simplement invisible : la logique de signe qui existait ici n'a jamais pu s'executer.
MINUS_MAX_HEIGHT = 5
MINUS_MIN_ASPECT = 1.8
# Largeur minimale pour distinguer un tiret du bruit de rendu (mesure : des composantes
# de 1 a 2 px de large trainent en bord de ROI).
MINUS_MIN_WIDTH = 4
# Ecart maximal entre un SIGNE NEGATIF et le chiffre qu'il signe. Mesure sur les captures
# d'Incarnam (coordonnees negatives) : le signe est a 2-3 px de son chiffre, quand le
# separateur de « - Niveau » est a 8 px du « N ». Le seuil est place entre les deux.
MINUS_ATTACH_GAP = 5


_COORD_TEMPLATES = "coord_templates.npz"


def load_coord_templates() -> DigitTemplates:
    """Gabarits de la police des COORDONNEES, distincte de celle de l'UI.

    Mesure qui impose ce second jeu : sur `tacle.png`, les gabarits d'UI donnent des
    scores de 0,61 a 0,82 la ou le seuil d'acceptation est a 0,90. Ce n'est pas un seuil
    trop strict -- c'est une autre police, et l'abaisser ferait accepter des confusions
    au lieu de les rejeter.
    """
    from importlib import resources

    ref = resources.files("jev_tactics.perception") / "data" / _COORD_TEMPLATES
    with resources.as_file(ref) as path:
        return DigitTemplates.load(path)


@dataclass(frozen=True)
class PositionReading:
    """Position lue, ou POURQUOI elle ne l'a pas ete.

    « Coordonnees illisibles » ne se corrige pas : les causes possibles appellent des
    gestes opposes -- fabriquer un gabarit, deplacer une ROI, accepter un signe negatif.
    Une session reelle a rapporte « 3 passages sans coordonnees lisibles » sans rien de
    plus, et il n'y avait aucun moyen de savoir lequel des cinq obstacles c'etait.
    """

    position: MapPosition | None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.position is not None


@dataclass(frozen=True)
class MapPosition:
    """Coordonnees de la carte courante."""

    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y


def _blobs(
    mask: NDArray[np.uint8],
) -> list[tuple[int, int, int, int, int, NDArray[np.uint8]]]:
    """Composantes -> (x, y, largeur, hauteur, aire, imagette), de gauche a droite."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for index in range(1, count):
        x, y, width, height, area = (int(stats[index, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        if area < MIN_DIGIT_AREA:
            continue
        patch = (labels[y:y + height, x:x + width] == index).astype(np.uint8) * 255
        found.append((x, y, width, height, area, patch))
    return sorted(found, key=lambda b: b[0])


def _minus_marks(mask: NDArray[np.uint8]) -> list[tuple[int, int, int, int]]:
    """Glyphes en forme de TIRET -> (x, y, largeur, hauteur), de gauche a droite.

    Ils sont cherches a part parce qu'ils sont minuscules : le separateur mesure a 5x1
    pixels, quand le seuil d'aire des chiffres est a 18. Les inclure dans `_blobs`
    ferait entrer du bruit de rendu partout ailleurs.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    marks = []
    for index in range(1, count):
        x, y, width, height = (int(stats[index, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if (height <= MINUS_MAX_HEIGHT and width >= MINUS_MIN_WIDTH
                and width / max(height, 1) >= MINUS_MIN_ASPECT):
            marks.append((x, y, width, height))
    return sorted(marks)


def _signe_a_gauche(
    group: list, marks: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    """Le groupe est-il precede d'un SIGNE NEGATIF ? -> le tiret consomme, ou None.

    Un signe est COLLE au nombre qu'il signe ; le separateur de « - Niveau » flotte entre
    deux mots. Mesure sur les cinq captures qui portent des coordonnees, dont trois
    negatives (Incarnam) :

        glyphe                       ecart au glyphe suivant
        signe de -1, -2, -4, -6              2 a 3 px
        separateur de « - Niveau »             8 px

    Le seuil est place a mi-chemin. Le tiret doit en outre etre a la HAUTEUR du nombre :
    sur `20260818-205954` la zone porte cinq autres composantes plates, au-dessus et en
    dessous de la ligne de texte, dont deux passeraient l'ecart seul.
    """
    gx = min(glyph[0] for glyph in group)
    top = min(glyph[1] for glyph in group)
    bottom = max(glyph[1] + glyph[3] for glyph in group)
    for mark in marks:
        mx, my, mw, mh = mark
        if not 0 <= gx - (mx + mw) <= MINUS_ATTACH_GAP:
            continue
        if top <= my + mh // 2 <= bottom:
            return mark
    return None


def read_position(
    frame: NDArray[np.uint8], templates: DigitTemplates | None = None
) -> MapPosition | None:
    """Frame BGR -> coordonnees, ou None. Voir `diagnose_position` pour la RAISON."""
    return diagnose_position(frame, templates).position


def diagnose_position(
    frame: NDArray[np.uint8], templates: DigitTemplates | None = None
) -> PositionReading:
    """Frame BGR -> coordonnees de la carte, ou None si la lecture n'est pas sure.

    Tout-ou-rien, comme la lecture d'UI : une coordonnee dont un chiffre est douteux est
    une position FAUSSE, et une position fausse est pire que pas de position -- le bot
    croirait savoir ou il est et calculerait un trajet vers nulle part.
    """
    templates = templates or load_coord_templates()
    x0, y0, x1, y1 = COORD_ROI
    height, width = frame.shape[:2]
    if x1 > width or y1 > height:
        return PositionReading(None, f"capture {width}x{height} trop petite pour la "
                                     f"zone {COORD_ROI}")

    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(WHITE_BAND[0], np.uint8),
                       np.array(WHITE_BAND[1], np.uint8))

    # Les minuscules (« iveau ») sont plus basses que les chiffres : on ne garde que les
    # glyphes hauts. Mais le « N » majuscule, lui, a la meme hauteur -- s'arreter sur la
    # hauteur seule ne suffit donc pas, et c'est le decoupage en groupes qui tranche.
    tall = [(x, y, w, h, patch) for x, y, w, h, _a, patch in _blobs(mask)
            if MIN_DIGIT_HEIGHT <= h <= MAX_DIGIT_HEIGHT]
    groups = _group_by_gap(tall)
    if not tall:
        return PositionReading(None, "aucun glyphe a la bonne hauteur dans la zone — "
                                     "coordonnees masquees, ou ROI a cote")
    if len(groups) < 2:
        return PositionReading(None, f"{len(groups)} groupe(s) de chiffres au lieu de 2 "
                                     f"— une seule coordonnee lisible")

    # Seuls les DEUX PREMIERS groupes sont les coordonnees. Ce qui suit est « - Niveau NN »,
    # dont le N majuscule est aussi haut qu'un chiffre : le lire donnerait une ordonnee
    # fantaisiste, et le niveau du personnage passerait pour une position.
    # LE SIGNE NEGATIF. Les coordonnees Dofus sont souvent negatives -- Bonta, Astrub, et
    # INCARNAM tout entier, ou commence tout personnage neuf. Lire « -2, 9 » comme « 2, 9 »
    # enverrait le bot calculer un trajet vers une carte a l'oppose.
    #
    # Ce chemin refusait donc TOUTE lecture des qu'un tiret precedait la fin des
    # coordonnees, faute de pouvoir distinguer un signe du separateur de « - Niveau », qui
    # est le meme glyphe. Le refus etait honnete, mais il rendait le bot AVEUGLE sur une
    # zone entiere : en session reelle a Incarnam, chaque carte sortait INCONNU, et
    # l'orchestrateur s'arretait apres avoir envoye sa touche de dernier recours.
    #
    # Le commentaire d'alors disait ce qui manquait -- « il faudrait mesurer l'ecart
    # tiret-chiffre, et le mesurer demande une capture ». Les captures existent
    # desormais, la mesure est faite, et elle separe franchement : cf. `_signe_a_gauche`.
    marks = list(_minus_marks(mask))
    end_of_coordinates = max(x + width for x, _y, width, _h, _p in groups[1])
    signs = []
    for group in groups[:2]:
        mark = _signe_a_gauche(group, marks)
        signs.append(-1 if mark is not None else 1)
        if mark is not None:
            marks.remove(mark)

    # ...ET SUR LA MEME LIGNE. La ROI mord le BAS DE LA LIGNE DU DESSUS -- le nom de la
    # zone -- et les fragments de lettres qu'elle y attrape ont la forme d'un tiret.
    #
    # Constate en session reelle, sur la frame conservee du 19/08 : « Incarnam (Cha...) »
    # au-dessus de « -3, -6 - Niveau 5 ». Les deux signes etaient correctement rattaches,
    # et la lecture echouait quand meme sur trois marques a y = 0 a 3, alors que les
    # chiffres tiennent entre y = 8 et 21. Toutes les cartes d'Incarnam ont des coordonnees
    # negatives : la position etait donc ILLISIBLE partout, le circuit ne pouvait plus
    # juger un deplacement, et le journal enregistrait `position: null` a chaque ligne.
    #
    # Un tiret qui n'est pas a la hauteur des chiffres ne peut etre ni un signe ni un
    # separateur : il appartient a une autre ligne de texte. Le refus garde son domaine --
    # une marque inexpliquee DANS la ligne des coordonnees -- et perd ce qui n'a jamais
    # ete le sien.
    haut = min(y for group in groups[:2] for _x, y, _w, _h, _p in group)
    bas = max(y + height for group in groups[:2] for _x, y, _w, height, _p in group)

    def _sur_la_ligne(mark) -> bool:
        _mx, my, _mw, mh = mark
        return haut <= my + mh // 2 <= bas

    if any(x < end_of_coordinates and _sur_la_ligne((x, _y, _w, _h))
           for x, _y, _w, _h in marks):
        return PositionReading(None, "tiret inexplique avant la fin des coordonnees : ni "
                                     "signe accroche a un nombre, ni separateur — lecture "
                                     "refusee plutot que de rendre une position douteuse")

    missing = sorted(set(range(10)) - templates.covered_digits())
    numbers = []
    for group, sign in zip(groups[:2], signs, strict=True):
        value, score = _read_number(group, templates)
        if value is None:
            return PositionReading(
                None, f"chiffre rejete (meilleur accord {score:.2f}, seuil "
                      f"{MIN_SCORE})" + (f" — gabarits MANQUANTS : {missing}"
                                         if missing else ""))
        numbers.append(sign * value)
    return PositionReading(MapPosition(x=numbers[0], y=numbers[1]))


def _group_by_gap(glyphs: list) -> list[list]:
    """Decoupe une suite de glyphes en groupes separes par un ecart horizontal."""
    groups: list[list] = []
    previous_end = None
    for glyph in glyphs:
        x, width = glyph[0], glyph[2]
        if previous_end is None or x - previous_end > COORD_GAP:
            groups.append([])
        groups[-1].append(glyph)
        previous_end = x + width
    return groups


def _read_number(group: list, templates: DigitTemplates) -> tuple[int | None, float]:
    """Groupe de glyphes -> entier POSITIF, ou None si un seul chiffre est douteux.

    Le signe est traite en amont, dans `diagnose_position` : un tiret ne survit pas au
    filtre d'aire des chiffres, et ne peut donc pas arriver jusqu'ici. Le code qui
    pretendait le gerer a cet endroit n'a jamais pu s'executer.

    Tout-ou-rien : une coordonnee dont un chiffre est incertain est une position FAUSSE,
    et une position fausse est pire que pas de position -- le bot croirait savoir ou il
    est et calculerait un trajet vers nulle part.
    """
    digits = []
    worst = 1.0
    for _x, _y, _width, _height, patch in group:
        digit, score = templates.classify(normalize_glyph(patch))
        if digit is None:
            return None, score
        worst = min(worst, score)
        digits.append(digit)
    if not digits:
        return None, 0.0
    return int("".join(str(d) for d in digits)), worst
