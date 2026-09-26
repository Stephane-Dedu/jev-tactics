"""Lecture de la barre de sorts : quels emplacements sont occupes, et lesquels sont
reellement lancables ce tour-ci.

Sans cette lecture, le planificateur suppose tous les sorts toujours disponibles et
programme des lancers que le jeu refusera (rechargement, cout en PA insuffisant,
conditions non remplies). C'est une perception ACTIVE de plus : le jeu grise deja ce qui
n'est pas jouable, on le lit au lieu de modeliser des regles de rechargement.

Signature relevee en comparant les MEMES emplacements sur deux captures (le meme
personnage, a deux moments) :

    etat            saturation   luminosite
    vide               ~47          ~60        <- ni couleur ni lumiere
    en recharge      121-162       66-95       <- garde sa COULEUR, perd sa LUMIERE
    pret              60-217      136-249

Un sort en recharge reste donc colore : c'est la luminosite qui tranche, pas la
saturation. Distinguer vide et indisponible exige les deux.

Geometrie relevee sur 1919x1079 : grille de 12 x 3 emplacements, pas de 46,2 px,
origine (655, 902). Comme les ROIs d'UI, c'est de la config liee a la disposition.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

# Grille de la barre de sorts (disposition 1919x1079).
ORIGIN_X, ORIGIN_Y = 655, 902
SLOT_STEP = 46.2
COLUMNS, ROWS = 12, 3
# Marge interne : on echantillonne le coeur de l'icone, pas son cadre.
INSET = 8
SAMPLE = 30

# Seuils issus des mesures ci-dessus, places dans les intervalles vides.
EMPTY_MAX_SATURATION = 70
EMPTY_MAX_VALUE = 75
READY_MIN_VALUE = 115
# PART DES EMPLACEMENTS OCCUPES LUS « EN RECHARGEMENT » AU-DELA DE LAQUELLE LA LECTURE
# N'EST PLUS CREDIBLE.
#
# Un panneau du jeu peut se dessiner PAR-DESSUS la barre de sorts, et le bot en ouvre un
# lui-meme : l'infobulle de monstre, qu'il affiche pour lire les PV ennemis
# (`enemy_hp_hovers`). Les emplacements couverts sont alors sombres, donc lus « en
# rechargement » -- pas « vides ». La garde existante de `bot/session.py` ne couvrait que
# le cas TOUT VIDE : sa surveillance etait plus etroite que sa promesse.
#
# Mesure sur les 24 captures de session, part des occupes lus en rechargement :
#
#     22 captures      4 % a 12 %      barre degagee
#      2 captures     81 %             infobulle de monstre par-dessus la barre
#
# L'intervalle entre 12 % et 81 % est VIDE : le seuil s'y pose sans arbitrage, comme celui
# du glyphe de mort. Physiquement, il est de toute facon invraisemblable que quatre sorts
# sur cinq soient en rechargement au meme instant.
#
# CE QUE COUTAIT LE DEFAUT : sur ces deux frames, 5 sorts restent lisibles sur 27. Le
# solveur planifie donc avec le cinquieme de son arsenal, ce qui produit au mieux un plan
# mediocre et au pire un tour passe -- le mode de panne le plus cher du projet, puisque le
# bilan rend « 30 tours joues », strictement le meme qu'un combat gagne.
MAX_PLAUSIBLE_COOLING_SHARE = 0.5


class SlotState(str, Enum):
    EMPTY = "empty"              # pas de sort dans l'emplacement
    COOLING = "cooling"          # sort present mais pas lancable (recharge, conditions)
    READY = "ready"              # lancable


def slot_index(row: int, column: int) -> int:
    """Index lineaire d'un emplacement (lecture par lignes, comme la barre)."""
    return row * COLUMNS + column


def slot_region(index: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) de la zone echantillonnee pour un emplacement."""
    row, column = divmod(index, COLUMNS)
    x = int(ORIGIN_X + column * SLOT_STEP) + INSET
    y = int(ORIGIN_Y + row * SLOT_STEP) + INSET
    return x, y, x + SAMPLE, y + SAMPLE


# Bornes d'une icone de sort, en pixels. Mesurees sur six captures : 42x42 partout, le
# pas valant 46,2. On accepte 30 a 52 pour absorber le cadre et l'anticrenelage.
ICON_SIDE = (30, 52)
ICON_AREA = (900, 3000)
# Hauteur fouillee au-dessus du bas de l'ecran. Trois rangees plus leur entete tiennent en
# 192 px sur les references ; 220 laisse la marge sans atteindre le plateau.
BAR_BAND_HEIGHT = 220
# Ecart vertical au-dela duquel deux icones appartiennent a des rangees differentes.
ROW_GAP = 25


@dataclass(frozen=True)
class BarGeometry:
    """Ou se trouve REELLEMENT la barre, et combien de rangees elle montre."""

    origin_x: int
    origin_y: int
    rows: int


def detect_bar_geometry(frame: NDArray[np.uint8]) -> BarGeometry | None:
    """Geometrie relevee sur la frame, ou None si la barre n'est pas identifiable.

    POURQUOI RELEVER PLUTOT QUE SUPPOSER. La barre se configure en jeu : une, deux ou
    trois rangees, empilees vers le haut depuis une rangee du bas qui ne bouge pas. La
    constante `ORIGIN_Y` decrit une barre a TROIS rangees ; sur une barre a une rangee,
    tous les prelevements tombent 68 px trop haut, dans le decor -- lu comme des
    emplacements occupes. Constate le 19/08 : « 30 emplacements occupes » sur un
    personnage de niveau 5 qui en a cinq.

    CE QUI REND LA MESURE FIABLE : elle retrouve la constante posee a la main. Sur les cinq
    captures de reference (trois rangees), elle rend (648, 901) contre (655, 902) releve a
    l'oeil -- un pixel d'ecart en ordonnee. Ce n'est donc pas une methode concurrente, c'est
    la meme mesure, faite a chaque frame au lieu d'une fois pour toutes.

    RIEN N'EST DEVINE QUAND RIEN N'EST VU : hors combat, l'inventaire remplace la barre et
    aucune icone n'est trouvee. On rend None, et l'appelant garde la constante -- l'ignorance
    ne doit pas paralyser, principe applique partout ailleurs.
    """
    height = frame.shape[0]
    haut = max(0, height - BAR_BAND_HEIGHT)
    hsv = cv2.cvtColor(frame[haut:height], cv2.COLOR_BGR2HSV)
    masque = ((hsv[:, :, 1] > EMPTY_MAX_SATURATION + 20)
              & (hsv[:, :, 2] > READY_MIN_VALUE - 5)).astype(np.uint8)
    nombre, _labels, stats, centres = cv2.connectedComponentsWithStats(masque, 8)
    icones = [(int(centres[i][0]), int(centres[i][1]) + haut)
              for i in range(1, nombre)
              if ICON_AREA[0] < stats[i, cv2.CC_STAT_AREA] < ICON_AREA[1]
              and ICON_SIDE[0] <= stats[i, cv2.CC_STAT_WIDTH] <= ICON_SIDE[1]
              and ICON_SIDE[0] <= stats[i, cv2.CC_STAT_HEIGHT] <= ICON_SIDE[1]]
    if not icones:
        return None
    ordonnees = sorted(y for _x, y in icones)
    rangees = [ordonnees[0]]
    for y in ordonnees[1:]:
        if y - rangees[-1] > ROW_GAP:
            rangees.append(y)
    demi = int(SLOT_STEP // 2)
    return BarGeometry(origin_x=min(x for x, _y in icones) - demi,
                       origin_y=rangees[0] - demi, rows=len(rangees))


# Luminosite en deca de laquelle un pixel appartient au BANDEAU de la barre et non au
# decor. Mesure sur six captures : le bandeau tient sous 90, le decor au-dessus depasse 110.
BAR_PANEL_DARKNESS = 90
# Ecart tolere entre le bandeau mesure et celui que la geometrie suppose. La rangee vaut
# 46 px : au-dela d'une demi-rangee, on ne parle plus de la meme disposition.
BAR_TOP_TOLERANCE = 25


def _panel_top(frame: NDArray[np.uint8]) -> int | None:
    """Haut du bandeau de la barre, ou None si aucun bandeau ne se detache du decor."""
    width = frame.shape[1]
    x0 = min(width, ORIGIN_X + int(COLUMNS * SLOT_STEP) // 2)
    colonne = frame[:, max(0, x0 - 120):x0]
    if colonne.size == 0:
        return None
    luminosite = colonne.max(axis=2).mean(axis=1)
    sombres = np.flatnonzero(luminosite[ORIGIN_Y - 60:] < BAR_PANEL_DARKNESS)
    if sombres.size == 0:
        return None
    haut = int(sombres[0]) + ORIGIN_Y - 60
    # Un bandeau se definit par son CONTRASTE avec ce qui est au-dessus : sans cela une
    # frame entierement sombre rend « bandeau des le premier pixel examine ».
    dessus = luminosite[max(0, haut - 40):haut]
    if dessus.size == 0 or float(dessus.mean()) < BAR_PANEL_DARKNESS + 20:
        return None
    return haut


# Ecart entre le haut du BANDEAU et l'origine des emplacements. Releve sur les captures
# de reference : bandeau a y=873, origine des icones a y=901.
BAR_PANEL_TO_ORIGIN = 28


def bar_layout(frame: NDArray[np.uint8]) -> BarGeometry | None:
    """Disposition REELLE de la barre, ou None si rien ne permet de trancher.

    DEUX SIGNAUX, CHACUN LA OU IL EST FORT. Le nombre de rangees vient du BANDEAU :
    `detect_bar_geometry` le deduit des icones, et le commentaire de `_describe_layout`
    dit pourquoi c'est insuffisant -- un panneau dessine par-dessus la barre cache les
    icones des rangees hautes, et le comptage y voit une barre a une rangee alors qu'elle
    est entiere. Le bandeau, lui, ne bouge pas sous ce qu'on dessine dessus.

    L'ORIGINE vient des icones quand elles s'accordent avec le bandeau sur le nombre de
    rangees : elle est alors exacte au pixel. Si elles ne s'accordent pas, c'est qu'une
    d'elles est masquee -- on retombe sur l'origine deduite du bandeau, qui ne l'est pas.

    Rend None quand le bandeau est introuvable : l'appelant garde alors les constantes.
    Mieux vaut la disposition de reference qu'une disposition devinee sur des icones dont
    on vient d'etablir qu'elles peuvent mentir.
    """
    haut = _panel_top(frame)
    if haut is None:
        return None
    rangees = max(1, min(ROWS, round(
        (ORIGIN_Y + (ROWS - 1) * SLOT_STEP - haut + 14) / SLOT_STEP)))
    icones = detect_bar_geometry(frame)
    if icones is not None and icones.rows == rangees:
        return icones
    return BarGeometry(origin_x=ORIGIN_X, origin_y=haut + BAR_PANEL_TO_ORIGIN,
                       rows=rangees)


def _describe_layout(frame: NDArray[np.uint8]) -> str | None:
    """Disposition relevee, quand elle DIFFERE de celle que decrivent les constantes.

    Elle n'est plus une accusation : `read_spell_slots` releve la geometrie et s'y adapte,
    donc la lecture est juste. Ce qui reste utile est de la NOMMER, parce qu'elle explique
    le reste du bilan -- sur une barre a une rangee, les emplacements 12 a 35 n'existent
    pas, et une config qui les declare sera signalee « manquants ». Sans cette ligne, ce
    diagnostic enverrait chercher un remappage qui n'a pas eu lieu.
    """
    geometrie = detect_bar_geometry(frame)
    if geometrie is None:
        return None
    # LE BANDEAU TRANCHE, PAS LES ICONES. Un panneau du jeu dessine PAR-DESSUS la barre --
    # l'infobulle de monstre, que le bot ouvre lui-meme -- cache les icones des rangees
    # hautes : le comptage d'icones y voit une barre a une rangee, alors que la barre est
    # entiere. Le bandeau, lui, reste a sa hauteur quoi qu'on dessine dessus. C'est le seul
    # discriminant, et une lecture d'occlusion prise pour une disposition ferait lire douze
    # emplacements au lieu de trente-six -- le solveur planifierait avec un tiers de
    # l'arsenal, exactement la panne que `MAX_PLAUSIBLE_COOLING_SHARE` existe pour attraper.
    haut = _panel_top(frame)
    if haut is None or abs(haut - (ORIGIN_Y - 14)) <= BAR_TOP_TOLERANCE:
        return None
    rangees = max(1, round((ORIGIN_Y + (ROWS - 1) * SLOT_STEP - haut + 14) / SLOT_STEP))
    return (f"barre a {rangees} rangee(s) au lieu de {ROWS} (bandeau releve a y={haut}, "
            f"icones a y={geometrie.origin_y}) : les emplacements au-dela de "
            f"{COLUMNS * rangees} n'existent pas sur cette disposition, et la lecture "
            f"les prend dans le decor")


def slot_region_for(index: int, geometry: BarGeometry | None) -> tuple[int, int, int, int]:
    """`slot_region`, mais sur une geometrie RELEVEE quand on en a une."""
    if geometry is None:
        return slot_region(index)
    row, column = divmod(index, COLUMNS)
    x = int(geometry.origin_x + column * SLOT_STEP) + INSET
    y = int(geometry.origin_y + row * SLOT_STEP) + INSET
    return x, y, x + SAMPLE, y + SAMPLE


def read_spell_slots(frame: NDArray[np.uint8]) -> list[SlotState]:
    """Etat de chaque emplacement de la barre, dans l'ordre de lecture.

    LE CABLAGE MANQUAIT, et le commentaire de `BarAudit.layout` affirmait le contraire :
    « `read_spell_slots` releve la geometrie a chaque frame et s'y adapte ». Cette
    fonction appelait `slot_region` -- la version CONSTANTE -- et parcourait les 36
    emplacements d'une barre a trois rangees quelle que soit la barre affichee.
    `slot_region_for`, ecrit precisement pour s'adapter, n'etait appele par personne.

    CE QUE CA DONNAIT, mesure sur les captures de session du 19/08 (barre a UNE rangee,
    cinq sorts) : « 14 prets, 16 en rechargement, 6 vides ». Les 24 emplacements des
    rangees inexistantes sont preleves dans le DECOR, et le decor d'Incarnam est clair --
    donc lu « pret ». Le solveur se voyait offrir des sorts qui n'existent pas, et
    l'audit de barre comparait la config a une lecture imaginaire.

    Les emplacements des rangees absentes sont rendus VIDES : ils n'existent pas, et
    « vide » est ce que l'audit sait deja interpreter.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]

    geometry = bar_layout(frame)
    # Rien de tranche (hors combat, inventaire a la place, bandeau noye dans le decor) :
    # on garde les constantes plutot que de tout declarer vide -- l'ignorance ne doit pas
    # paralyser. C'est aussi ce qui laisse `MAX_PLAUSIBLE_COOLING_SHARE` faire son travail
    # sur une barre recouverte, au lieu de lire douze emplacements et de trouver ca normal.
    rows = geometry.rows if geometry is not None else ROWS

    states: list[SlotState] = []
    for index in range(COLUMNS * ROWS):
        if index >= COLUMNS * rows:
            states.append(SlotState.EMPTY)
            continue
        x0, y0, x1, y1 = slot_region_for(index, geometry)
        if x1 > width or y1 > height:
            states.append(SlotState.EMPTY)
            continue
        patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
        saturation = float(np.median(patch[:, 1]))
        value = float(np.median(patch[:, 2]))

        if saturation < EMPTY_MAX_SATURATION and value < EMPTY_MAX_VALUE:
            states.append(SlotState.EMPTY)
        elif value < READY_MIN_VALUE:
            states.append(SlotState.COOLING)
        else:
            states.append(SlotState.READY)
    return states


def ready_slots(frame: NDArray[np.uint8]) -> set[int]:
    """Index des emplacements lancables."""
    return {i for i, state in enumerate(read_spell_slots(frame))
            if state is SlotState.READY}


@dataclass(frozen=True)
class BarAudit:
    """Confrontation entre la barre CONFIGUREE et celle affichee a l'ecran."""

    displayed: bool              # la barre est-elle visible ? (masquee hors combat)
    missing: tuple[int, ...]     # slots declares par la config, VIDES a l'ecran
    unmapped: tuple[int, ...]    # slots occupes que la config ignore
    # DISPOSITION RELEVEE, quand elle differe de celle que decrivent les constantes.
    #
    # Ce champ a d'abord ete une ACCUSATION : la lecture tombait a cote et il fallait le
    # dire. Depuis, `read_spell_slots` releve la geometrie a chaque frame et s'y adapte,
    # donc la lecture est juste -- ce qui reste utile est de NOMMER la disposition, parce
    # qu'elle explique le reste du bilan : sur une barre a une rangee, les emplacements
    # 12 a 35 n'existent pas, et une config qui les declare sera signalee « manquants ».
    # Sans cette ligne, ce diagnostic-la enverrait chercher un remappage inexistant.
    layout: str | None = None

    @property
    def consistent(self) -> bool:
        """Seuls les slots MANQUANTS sont une incoherence.

        Un slot occupe qu'on ignore n'est pas une erreur : la barre contient des objets,
        des consommables et des sorts qu'on a choisi de ne pas jouer. Mesure sur les
        captures reelles : 26 emplacements occupes pour 20 sorts configures, et les 20
        sont bien la.
        """
        return not self.displayed or not self.missing

    def describe(self) -> str:
        prefixe = f"{self.layout} — " if self.layout else ""
        if not self.displayed:
            return "barre non affichee (hors combat) — rien a confronter"
        if not self.missing:
            return (f"{prefixe}{len(self.unmapped)} emplacement(s) occupe(s) hors config "
                    f"(objets, sorts non joues) — tous les sorts configures sont la")
        return (f"{prefixe}emplacements {list(self.missing)} declares par la config mais "
                f"VIDES a l'ecran — la barre a ete remappee, la config est perimee, ou "
                f"ces emplacements n'existent pas sur cette disposition")


def audit_bar(frame: NDArray[np.uint8], expected: Iterable[int]) -> BarAudit:
    """La barre affichee contient-elle bien les sorts que la config y place ?

    Ce que ca attrape, et que RIEN ne verifiait : un remappage en jeu. La config associe
    un sort a un emplacement et en deduit son raccourci ; si la barre a bouge depuis, le
    bot appuie sur la touche en croyant lancer Decimation et lance ce qui s'y trouve.
    Le planificateur, lui, a calcule sa sequence en supposant l'inverse -- donc le sort
    part, sur la cible prevue pour un autre.

    Ce n'est pas theorique : la barre de ce projet a deja ete remappee a la main une fois,
    et seule une relecture attentive de la config l'avait rattrape.

    `displayed` distingue « la barre est vide » de « quelque chose occupe la barre ». Il
    NE distingue PAS le combat du hors-combat, contrairement a ce que ce docstring a
    longtemps laisse croire : mesure sur capture.png, une carte, il rend displayed=True
    avec les emplacements [0, 19] occupes et conclut au remappage. Hors combat l'inventaire
    prend la place de la barre, et son contenu ressemble assez a des sorts pour tromper le
    lecteur d'emplacements.

    L'APPELANT doit donc s'assurer d'etre en combat avant de conclure. Le discriminant
    etabli est la timeline (`read_timeline`, 1 ms) ; `CombatRunner._audit_bar_once` s'en
    sert, et `scripts/play_turn.py` aussi.
    """
    states = read_spell_slots(frame)
    occupied = {i for i, state in enumerate(states) if state is not SlotState.EMPTY}
    wanted = {int(slot) for slot in expected}
    return BarAudit(
        displayed=bool(occupied),
        missing=tuple(sorted(wanted - occupied)),
        unmapped=tuple(sorted(occupied - wanted)),
        layout=_describe_layout(frame),
    )
