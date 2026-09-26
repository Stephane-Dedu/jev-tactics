"""Lecture de la timeline de combat : ordre de jeu et PV de chaque combattant.

En Dofus 3.6 il n'y a pas de barre de vie flottante au-dessus des sprites (cf.
perception/entities.py) : les PV se lisent sur la bande de portraits en haut de l'ecran,
chacun flanque d'une **barre verticale rouge** dont la hauteur est proportionnelle aux PV
restants.

Geometrie relevee sur les captures 1919x1079 (dofusscreen*.png) :
  - barres larges de 7 px, hautes de 70 px a pleine vie ;
  - emplacements espaces de 77 px (x = 1252, 1338, 1415, 1492, 1569, 1646...) ;
  - le portrait ACTIF est agrandi : sa barre est decalee vers le haut (bas a y=121
    contre 130 pour les autres), ce qui sert a le reperer.

Verification de l'hypothese hauteur <-> PV : sur dofusscreen.png l'OCR lit 566/1539
(36.8 %) et la barre mesure 24/70 = 34 %. L'ecart tient a l'antialiasing des extremites.

Cette lecture donne aussi l'**ordre d'initiative**, utile au planificateur.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from jev_tactics.state import Team

# Zone de recherche (x0, y0, x1, y1) pour une disposition 1919x1079.
TIMELINE_ROI = (1150, 30, 1700, 150)

# Bandes HSV du rouge des barres (le rouge chevauche 0 en teinte -> deux sous-bandes).
RED_BANDS = (((0, 120, 90), (12, 255, 255)), ((168, 120, 90), (179, 255, 255)))

BAR_WIDTH_MAX = 14            # une barre est fine
# Hauteur minimale d'une barre de vie pour etre retenue.
#
# LIMITE CONNUE, et elle a l'air grave : une barre COURTE est un combattant presque mort.
# Constate sur 20260807-222701.png -- la barre du monstre y fait 4 px de haut, soit ~6 %
# de vie, et tombe sous ce seuil. Le bot perd donc de vue un ennemi au moment precis ou il
# faudrait l'achever.
#
# J'allais baisser le seuil. Mesure faite d'abord, sur 30 captures :
#
#     seuil    comptages justes   un seul actif   equipes lues
#       10          27/30              29              28
#        8          27/30              30               2
#        6          24/30              29               2
#        4          23/30              30               2
#
# La lecture des EQUIPES s'effondre de 28 a 2 des qu'on descend a 8. Les barres courtes
# admises sont majoritairement du decor et de l'ATH, et leur teinte brouille l'attribution
# d'equipe de toute la timeline. Le remede coute cinq fois ce que vaut le defaut.
#
# La perte reste bornee : un ennemi a 6 % de vie disparait de la TIMELINE, pas du plateau
# -- ses marqueurs de case restent detectes, et c'est eux que vise le solveur.
BAR_HEIGHT_MIN = 10
FULL_BAR_HEIGHT = 70.0        # hauteur a pleine vie (liee a l'echelle d'UI des captures)
MIN_SLOT_SPACING = 60         # ecart minimal entre deux emplacements (reels : ~77 px)
# De combien le portrait actif remonte par rapport aux autres. Mesure : 9 px (121 vs 130).
# Le seuil est place bien en dessous, pour tolerer un arrondi de detection de barre, tout
# en restant au-dessus du bruit (les portraits normaux sont alignes au pixel pres).
ACTIVE_OFFSET_PX = 3
# Teinte du fond des portraits (mesure : joueur ~103, ennemis ~176-177).
ALLY_HUE_RANGE = (92, 118)
ENEMY_HUE_MIN = 160
# Hauteur du bandeau fouille par `locate_timeline`. Ce diagnostic doit chercher PARTOUT --
# c'est son objet -- et n'agit sur rien.
TOP_BAND_HEIGHT = 260
# Fenetre VERTICALE du repli de lecture (y0, y1), en pixels. Toute la largeur, mais pas
# toute la hauteur : la panne constatee etait horizontale (barres a x=937, rectangle a
# partir de 1150), et fouiller plus bas ne fait qu'inviter le bruit.
#
# Mesure sur les 24 captures de combat du depot : les vraies jauges occupent y de 27 a 130.
# Le rectangle en dur commence a y=30 -- il coupait donc deja de 3 px. La fenetre garde une
# marge de part et d'autre sans descendre dans l'ecran.
FALLBACK_BAND_Y = (15, 175)
# Barres « a equipe » exigees pour croire une timeline trouvee hors du rectangle.
FALLBACK_MIN_BARS = 2


class TimelineEntry(BaseModel):
    """Un combattant sur la timeline, dans l'ordre d'initiative."""

    slot: int = Field(ge=0)          # rang dans l'ordre de jeu (0 = premier)
    x: int                           # abscisse ecran de la barre (identifie l'emplacement)
    hp_ratio: float = Field(ge=0.0, le=1.0)
    is_active: bool = False          # portrait agrandi = c'est son tour
    team: Team | None = None         # None si le fond du portrait n'est pas concluant


@dataclass(frozen=True)
class _Bar:
    x: int
    y: int
    w: int
    h: int

    @property
    def bottom(self) -> int:
        return self.y + self.h


def _find_bars(
    frame: NDArray[np.uint8], roi: tuple[int, int, int, int] = TIMELINE_ROI
) -> list[_Bar]:
    """Barres verticales rouges de la timeline, triees de gauche a droite."""
    x0, y0, x1, y1 = roi
    x1 = min(x1, frame.shape[1])
    y1 = min(y1, frame.shape[0])
    if x0 >= x1 or y0 >= y1:
        return []
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in RED_BANDS:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bars = []
    for i in range(1, count):
        x, y, w, h = (int(stats[i, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if w <= BAR_WIDTH_MAX and h >= BAR_HEIGHT_MIN and h > w:
            bars.append(_Bar(x=x + x0, y=y + y0, w=w, h=h))
    if not bars:
        return []

    # Toutes les barres de PV ont la MEME LARGEUR (7 px a cette echelle d'UI) ; les
    # elements rouges parasites (icones, decorations) mesurent 9 a 12 px. Mesure sur
    # dofusscreen* : vraies barres 7x24 a 7x70, parasites 9x14, 11x22, 12x16.
    # L'alignement par le bas a ete essaye et ECARTE : le bas du portrait actif varie
    # (114 ou 121 selon la capture -- il est agrandi et semble animé), ce qui faisait
    # rejeter le joueur lui-meme.
    widths = np.array([b.w for b in bars])
    values, counts = np.unique(widths, return_counts=True)
    modal_width = int(values[counts.argmax()])
    kept = sorted((b for b in bars if abs(b.w - modal_width) <= 1), key=lambda b: b.x)

    # Les emplacements sont espaces de ~77 px. Deux barres plus proches que cela sont un
    # doublon : on garde la plus HAUTE (la vraie jauge ; le parasite restant est court --
    # mesure : hauteurs 10 et 11 px, contre 24 a 70 pour les vraies).
    result: list[_Bar] = []
    for bar in kept:
        if result and bar.x - result[-1].x < MIN_SLOT_SPACING:
            if bar.h > result[-1].h:
                result[-1] = bar
        else:
            result.append(bar)
    return result


def _top_band(frame: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    """Tout le bandeau superieur, quelle que soit la largeur d'ecran."""
    height, width = frame.shape[:2]
    return (0, 0, width, min(TOP_BAND_HEIGHT, height))


def _fallback_band(frame: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    """Toute la largeur, mais seulement la hauteur ou vit une timeline."""
    height, width = frame.shape[:2]
    y0, y1 = FALLBACK_BAND_Y
    return (0, y0, width, min(y1, height))


def _fallback_bars(frame: NDArray[np.uint8]) -> list[_Bar]:
    """Barres de timeline cherchees dans TOUT le bandeau, quand le rectangle n'a rien eu.

    POURQUOI CE REPLI EXISTE. `TIMELINE_ROI` est un rectangle en dur (x de 1150 a 1700)
    releve sur des combats a quatre et plus. La timeline n'occupe QUE la largeur de ses
    combattants : a deux, elle se retrouve bien plus a gauche. Mesure sur la capture du
    18/08 -- barres a x=937 et x=1023, soit 113 px avant le bord gauche du rectangle, et
    la premiere commence a y=27 quand le rectangle debute a y=30. Les deux axes coupaient.
    La consequence n'etait pas une erreur mais un SILENCE : `read_timeline` rendait [],
    donc `classify_screen` rendait INCONNU, donc le bot envoyait sa touche de dernier
    recours et s'arretait -- en plein combat. C'est le defaut que le docstring de
    `locate_timeline` annoncait mot pour mot, sans que rien ne le branche.

    POURQUOI IL FILTRE SUR L'EQUIPE. Chercher large invite les faux positifs, et ce n'est
    pas theorique : sur les 33 captures hors combat que le rectangle laisse vides, la
    recherche large rend 1 a 4 composantes rouges (decor, ATH, icones) -- dont les 12
    ecrans d'HDV, que la timeline classerait COMBAT avant meme que l'HDV soit teste. Le
    fond du portrait tranche, parce qu'une vraie barre est TOUJOURS collee a un panneau
    bleu (allie) ou rouge (ennemi), ce que le decor n'a pas.

    Chaque contrainte a ete pesee separement, sur les 34 captures ou le rectangle ne rend
    rien (1 combat a retrouver, 33 negatifs) :

        variante                              combat retrouve   faux positifs
        bandeau 260, sans filtre                    1/1              12
        bandeau 260, + equipe                       1/1               1
        fenetre timeline, sans filtre               1/1              10
        fenetre timeline, + equipe                  1/1               0

    Aucune des deux n'est decorative : le filtre d'equipe ecarte dix negatifs, la fenetre
    verticale en ecarte un que le filtre laissait passer -- `20260818-210621`, le PANNEAU
    DE CONTROLE du bot pose par-dessus le jeu, dont le fond bleu donne deux barres de 10 px
    a y=217 classees « alliees ». C'est le faux positif le plus dangereux du lot, puisqu'il
    ferait jouer un tour dans une fenetre d'interface.

    POURQUOI CE FILTRE NE S'APPLIQUE PAS AU CHEMIN NORMAL. `_portrait_team` rend None des
    que le fond n'est pas net, et cela arrive sur de VRAIES barres : sur
    `20260807-222701.png`, deux barres dans le rectangle, zero equipe concluante.
    L'appliquer partout perdrait ce combat-la. Le rectangle est une preuve de position
    suffisante ; la bande entiere ne l'est pas, et c'est elle seule qui doit payer.
    """
    bars = [bar for bar in _find_bars(frame, _fallback_band(frame))
            if _portrait_team(frame, bar) is not None]
    # Un combat compte au moins deux combattants. Exiger la paire ecarte les composantes
    # isolees, qui sont l'essentiel du bruit (1 barre sur 11 des 12 ecrans d'HDV).
    return bars if len(bars) >= FALLBACK_MIN_BARS else []


def locate_timeline(frame: NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
    """Cherche la timeline dans TOUT le bandeau superieur -> son emprise reelle, ou None.

    `TIMELINE_ROI` est un rectangle en dur, releve sur une disposition 1919x1079. C'est la
    dependance la plus fragile de la chaine : si l'ATH est deplace, redimensionne, ou si
    l'echelle d'interface differe, les barres tombent hors du rectangle, `read_timeline`
    rend une liste vide -- et le bot en conclut simplement **« hors combat »**. Aucune
    erreur, aucun symptome : il attend, indefiniment, un combat qui a lieu sous ses yeux.

    Cette fonction ne sert PAS a la lecture courante (chercher large invite les faux
    positifs de l'UI). Elle sert au diagnostic : comparer ou sont vraiment les barres a
    l'endroit ou on les cherche transforme une panne muette en ecart chiffre.
    """
    bars = _find_bars(frame, _top_band(frame))
    if len(bars) < 2:
        return None
    xs = [b.x for b in bars]
    ys = [b.y for b in bars]
    bottoms = [b.bottom for b in bars]
    return (min(xs), min(ys), max(xs) + BAR_WIDTH_MAX, max(bottoms))


def _portrait_team(frame: NDArray[np.uint8], bar: _Bar) -> Team | None:
    """Equipe d'un emplacement, d'apres la teinte du fond de son portrait.

    Mesure sur les captures : fond du joueur H ~103 (bleu), fond ennemi H ~176-177
    (rouge/rose). L'ecart est franc, mais le portrait ACTIF est agrandi et decale, ce qui
    brouille parfois la mesure -- d'ou le renvoi de None quand rien n'est net, plutot
    qu'une equipe devinee.
    """
    # BANDEAU HAUT du portrait uniquement. L'illustration de la creature occupe le centre
    # et domine la mediane : sur `tacle.png`, le Chef de Guerre est blanc-creme, ce qui
    # tirait la teinte de son portrait a H=110 -- en plein dans la bande ALLIE. La
    # timeline annoncait donc deux allies, la reconciliation avec le plateau se
    # desactivait, et l'identification du joueur devenait un tirage au sort.
    #
    # Le haut du portrait, lui, est du fond pur. Mesure, sur les deux captures reelles :
    #     zone echantillonnee    tacle.png        combat1.png
    #     tout le portrait       103, 110  (faux)  103, 177, 177
    #     bandeau haut           103, 177  (juste) 103, 177, 177
    x0, x1 = bar.x + 10, bar.x + 60
    y0, y1 = bar.bottom - 75, bar.bottom - 55
    height, width = frame.shape[:2]
    if x1 > width or y0 < 0 or y1 > height:
        return None

    patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lively = patch[patch[:, 1] > 30]
    if len(lively) < 50:
        return None
    hue = float(np.median(lively[:, 0]))

    if ALLY_HUE_RANGE[0] <= hue <= ALLY_HUE_RANGE[1]:
        return Team.ALLY
    if hue >= ENEMY_HUE_MIN:
        return Team.ENEMY
    return None


def read_timeline(frame: NDArray[np.uint8]) -> list[TimelineEntry]:
    """Frame BGR -> combattants de la timeline, dans l'ordre d'initiative.

    Le portrait actif est repere par le bas de sa barre : il est agrandi, donc son bas
    differe de celui de la majorite. Liste vide si la timeline n'est pas visible (hors
    combat).
    """
    bars = _find_bars(frame)
    if not bars:
        # Le rectangle en dur n'est pas la timeline : c'est l'endroit ou elle se trouvait
        # sur les captures de reference. Avant de conclure « hors combat » -- verdict dont
        # tout le reste depend -- on la cherche pour de bon. Cf. `_fallback_bars`.
        bars = _fallback_bars(frame)
    if not bars:
        return []

    # Le portrait actif est agrandi et DEBORDE VERS LE HAUT : le bas de sa barre est plus
    # haut que celui des autres. Mesure sur combat1.png -- actif bottom=121, les deux
    # autres bottom=130 -- confirmee par un second cas en jeu.
    #
    # La regle precedente cherchait le bas MAJORITAIRE, ce qui n'a de sens qu'a partir de
    # trois combattants. A deux, chaque valeur est vue une fois, `argmax` rendait
    # arbitrairement la plus petite -- donc celle de l'actif -- et le portrait actif se
    # retrouvait declare normal pendant que l'autre etait declare actif. **L'inverse
    # exact.** Constate en jeu : « slot 0 ally <- MOI / slot 1 enemy <- ACTIF » alors que
    # le tour etait au joueur.
    #
    # La reference est donc la ligne BASSE des portraits normaux, qui ne depend d'aucune
    # majorite et vaut des deux combattants.
    bottoms = [b.bottom for b in bars]
    baseline = max(bottoms)
    raised = [i for i, b in enumerate(bottoms) if baseline - b > ACTIVE_OFFSET_PX]
    # Un seul portrait peut etre actif : en cas d'egalite, le plus haut.
    active = min(raised, key=lambda i: bottoms[i]) if raised else None

    return [
        TimelineEntry(
            slot=index,
            x=bar.x,
            hp_ratio=min(1.0, bar.h / FULL_BAR_HEIGHT),
            is_active=(index == active),
            team=_portrait_team(frame, bar),
        )
        for index, bar in enumerate(bars)
    ]
