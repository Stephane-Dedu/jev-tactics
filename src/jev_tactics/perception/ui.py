"""Lecture OCR de l'UI de combat : PV (courant/max), PA, PM.

Approche (b) du projet : ces valeurs sont a la fois une entree de l'etat structure ET
la source de labels "gratuits" derives des pixels. Le pretraitement (seuillage du texte
blanc + nettoyage) rend les chiffres nets ; le moteur OCR est Tesseract, encapsule
derriere read_ui() pour rester swappable (ex. futur classifieur maison sur police fixe).

Les ROIs sont propres a une resolution/disposition (ici 1919x1079). Pour une autre
resolution, re-mesurer les boites (cf. scripts d'exploration) -- c'est de la config,
pas de la logique.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

# TESSERACT EST CHARGE A L'APPEL, PAS AU CHARGEMENT DU MODULE.
#
# Il ne sert plus sur le chemin chaud -- un classifieur maison lit l'ATH en 1,3 ms -- et
# `pyproject.toml` le declare en extra optionnel (`ocr`). L'import etait pourtant en tete
# de fichier : `perception/__init__` importe `ui`, donc TOUT le paquet exigeait une
# dependance annoncee comme facultative. Le depot s'installait selon son propre README
# (`pip install -e ".[dev]"`) et la suite entiere tombait en erreur de COLLECTE -- 31
# fichiers, pas un seul test execute.
#
# Invisible en local, ou le venv l'avait par habitude. Trouve par la premiere CI, qui
# n'installe que ce que le README annonce : c'est precisement ce qu'on lui demande.
_TESSERACT: Any = None


def _tesseract() -> Any:
    """L'oracle hors-ligne, charge au premier besoin.

    Leve un message actionnable plutot qu'un `ImportError` nu : on n'arrive ici que si le
    classifieur maison a rendu la main -- cas deja rare -- et il faut savoir que c'est un
    extra qui manque, non le paquet qui est casse.
    """
    global _TESSERACT
    if _TESSERACT is None:
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                'pytesseract absent : c\'est le repli OCR, installe par '
                '`pip install -e ".[ocr]"` (plus le binaire Tesseract). Le chemin normal '
                "est le classifieur maison ; si l'on arrive ici, ses gabarits n'ont pas "
                "pu etre charges."
            ) from exc
        cmd = os.environ.get(
            "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = cmd
        _TESSERACT = pytesseract
    return _TESSERACT


@dataclass(frozen=True)
class Roi:
    """Boite de lecture d'un entier (une seule ligne de chiffres)."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int


# Disposition combat 1919x1079 (mesuree sur les captures 3.6).
# PV = orbe courant SUR max, empiles verticalement -> deux ROIs distinctes.
DEFAULT_ROIS: tuple[Roi, ...] = (
    Roi("pv", 515, 947, 628, 975),
    Roi("pv_max", 515, 979, 628, 1007),
    Roi("pa", 510, 1022, 562, 1054),
    Roi("pm", 578, 1022, 628, 1055),
)

# DEUXIEME DISPOSITION DE L'ORBE, et elle n'etait pas lue. Les deux ROIs ci-dessus
# supposent DEUX nombres empiles (courant au-dessus du maximum). Sur un ATH ou les PV
# tiennent dans un SEUL nombre centre -- « 88 », sans maximum affiche -- ce nombre tombe
# a cheval sur les deux boites : `pv` en attrape le haut, `pv_max` un liseré, et les deux
# lectures echouent. Mesure sur les captures de session du 19/08 : pv=None, pv_max=None,
# alors que pa et pm se lisent parfaitement.
#
# Releve au composant connexe sur quatre captures de session : les chiffres occupent
# y 969..984, x 555..582 (deux chiffres). La boite ci-dessous les encadre en evitant le
# liseré clair de l'orbe, qui deborde a x<540 et y<963.
HEART_ROI = Roi("pv_coeur", 530, 963, 610, 990)

# L'ORBE EST UNE JAUGE, et c'est ce qui rend la part de PV lisible SANS maximum affiche.
# Le liquide rouge descend avec les PV ; seule sa hauteur compte.
#
# Bornes relevees sur un orbe PLEIN puis verifiees sur un orbe entame, dans les DEUX
# dispositions (elles partagent le meme widget) :
#
#     capture                        verite   jauge    ecart
#     combat1920.png                  1.000   1.000     0.000
#     hors_combat.png                 1.000   0.987    -0.013
#     tacle.png (1146/1637)           0.700   0.707    +0.007
#     20260819-115112 (69/88)         0.784   0.800    +0.016
#
# Soit 1,6 point au pire sur six captures, deux ATH et trois niveaux de PV. C'est
# l'ordre de grandeur d'un pixel : la jauge fait 75 px pour 100 %.
HEART_X0, HEART_X1 = 505, 630
HEART_TOP_Y, HEART_BOTTOM_Y = 938, 1013
# Le bas du liquide ne bouge pas : c'est la pointe de l'orbe. S'il manque, ce n'est pas
# un orbe entame mais un orbe ABSENT (autre disposition, panneau par-dessus) -- et le
# contrat est de rendre None, jamais une valeur de repli.
HEART_BOTTOM_TOLERANCE = 4
# UN ORBE ENTAME ET UN ORBE MASQUE SE RESSEMBLENT : dans les deux cas le rouge commence
# plus bas. Le panneau de chat, deplie, recouvre le haut de l'orbe -- constate sur
# `20260819-115106-combat-vide.png`, ou la jauge annoncait 0,68 pour 0,78 reels.
#
# Ce qui les separe est le LISERE clair de l'orbe : il borde la partie vide, et un
# panneau pose par-dessus l'efface. Mesure sur les rangees situees juste au-dessus du
# niveau de liquide :
#
#     orbe entame (0,784)      lisere 236 px
#     orbe entame (0,700)      lisere 301 px
#     orbe plein               lisere  91 px
#     orbe masque par le chat  lisere   0 px
#
# On n'exige la preuve que lorsqu'on s'apprete a annoncer un orbe ENTAME : un orbe plein
# n'a pas de partie vide a justifier, et son lisere depend alors du decor derriere.
HEART_RIM_ROWS = 12
HEART_RIM_MIN = 30
HEART_FULL_ENOUGH = 0.97


# Rectangles ecran occupes par l'ATH, en disposition de reference. Une case de combat ne
# peut pas s'y trouver : l'interface est dessinee PAR-DESSUS le jeu, donc meme si le
# plateau s'etend dessous, ces cases ne sont ni visibles ni cliquables.
#
# Ce filtrage vient d'un defaut constate en jeu : un « ennemi » detecte a (937, 919),
# c'est-a-dire en plein sur la barre de sorts -- une icone orange prise pour un marqueur.
# Le controle plateau/timeline ne l'avait pas vu, parce que le COMPTE tombait juste (un
# ennemi reel manque, un fantome ajoute) : un comptage ne detecte pas un echange.
UI_ZONES: tuple[tuple[int, int, int, int], ...] = (
    (655, 902, 1209, 1040),     # barre de sorts
    (1150, 30, 1700, 150),      # timeline
    (500, 940, 640, 1060),      # jauges PV/PA/PM
    # Barres d'icones du bandeau superieur. Absentes jusqu'ici, alors qu'elles sont le
    # seul element d'ATH que `ui_mask` ne peut PAS rattraper : leurs icones sont CLAIRES
    # (0,023 de pixels sombres, contre 0,82 pour le chat).
    #
    # Relevees colonne par colonne sur cinq captures, en combat et hors combat. Deux
    # plages sortent sur les CINQ -- (0..81) et (1850..1920) -- et la barre centrale
    # droite sur quatre ; la cinquieme (combat1920.png) la montre pourtant a l'oeil, ses
    # icones y etant simplement plus claires que le critere.
    #
    # Verifie avant de les ajouter : sur les huit captures, ces rectangles ne recouvrent
    # AUCUNE case de plateau, sauf 3 sur 294 hors combat -- situees sous la barre, donc
    # ni visibles ni cliquables.
    (0, 0, 145, 45),            # outils, en haut a gauche
    (1426, 0, 1820, 45),        # barre d'icones principale
    (1850, 0, 1920, 45),        # boutons de fenetre
    # LA MINI-CARTE, et c'etait un CABLAGE VERS RIEN. Le docstring de `ui_mask` ci-dessous
    # dit depuis toujours qu'elle « ne couvre pas [...] l'interieur de la minicarte [...] :
    # ceux-la restent du ressort de `UI_ZONES` ». UI_ZONES ne la couvrait pas. Une
    # responsabilite explicitement deleguee a quelque chose qui ne l'implementait pas.
    #
    # Ce que ca laissait passer, mesure sur `data/mobexemple1.png` (carte hors combat,
    # groupe de monstres visible) :
    #
    #     region                     couvert par ui_mask   + UI_ZONES
    #     chat                                      96 %        96 %
    #     panneau de quetes                         92 %        92 %
    #     barre de sorts                            72 %        98 %
    #     mini-carte                                 0 %         0 %
    #     terrain (temoin)                           4 %         4 %
    #
    # 9,7 % de l'ecran, entierement en icones CLAIRES et animees -- des marqueurs
    # d'evenement qui clignotent. La detection de groupes repose sur le mouvement : chaque
    # clignotement devenait un candidat cliquable, et un clic sur la mini-carte coute un
    # ENGAGE_TIMEOUT entier, soit huit secondes, sans jamais engager quoi que ce soit.
    #
    # `ui_mask` ne peut pas la rattraper et le dit : la bande d'en-tete plafonne a 40 % de
    # pixels sombres et l'interieur tombe a 0-11 %, loin de MIN_PANEL_DENSITY.
    #
    # BORD RELEVE, pas estime : la chute de luminance moyenne donne (1444, 657) sur les
    # TROIS captures plein ecran qui la montrent -- mobexemple1.png, hors_combat.png et
    # capture.png -- au pixel pres.
    #
    # Verifie comme pour les barres d'icones ci-dessus : ce rectangle recouvre 26 a 28
    # cases de plateau sur ~291 visibles. Elles sont SOUS le panneau, donc ni visibles ni
    # cliquables par le joueur non plus -- les exclure ne coute aucune case jouable.
    (1444, 657, 1920, 1080),    # mini-carte
)


class UiReading(BaseModel):
    """Etat lisible de l'UI a un instant t (None = lecture echouee)."""

    pv: int | None = None
    pv_max: int | None = None
    pa: int | None = None
    pm: int | None = None


def _glyph_mask(crop: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Masque binaire des glyphes (chiffres BLANCS sur fond noir, resolution native).

    Les chiffres sont BLANCS (faible saturation, forte valeur) ; les icones sont
    SATUREES (bleu/vert/rouge). On isole donc le texte en HSV (S bas & V haut), ce qui
    ecarte les corps d'icones, puis on retire les petits specks (glints, reflets).
    C'est l'entree commune du classifieur maison (digits.py) et de Tesseract.
    """
    if crop.size == 0:
        # Appele directement par scripts/build_digit_templates.py, donc garde ici aussi :
        # une zone vide n'a rien a lire. Voir read_stat pour le pourquoi complet.
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat < 70) & (val > 180)).astype(np.uint8) * 255

    # Un chiffre occupe une bande de hauteur mediane : ni un trait plein-cadre
    # (deco d'orbe), ni un speck (glint, reflet), ni une ligne fine (separateur).
    h = mask.shape[0]
    clean = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        ca = stats[i, cv2.CC_STAT_AREA]
        if 0.35 * h <= ch <= 0.9 * h and ca >= 40:
            clean[labels == i] = 255
    return clean


def _preprocess(crop: NDArray[np.uint8], scale: int = 4) -> NDArray[np.uint8]:
    """Chiffres NOIRS sur fond BLANC, agrandis (ce qu'attend Tesseract)."""
    if crop.size == 0:
        # Meme garde que `read_stat`, et pour la meme raison : une ROI hors capture est
        # une lecture impossible, pas une erreur de programme. `cv2.resize` levait ici
        # une assertion OpenCV -- ce qui faisait planter la fabrique de templates au
        # milieu de sa moisson, sur la premiere capture d'une autre taille.
        return np.full((1, 1), 255, dtype=np.uint8)
    clean = cv2.bitwise_not(_glyph_mask(crop))  # inversion : noir sur blanc
    big = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # Marge blanche autour (Tesseract lit mal le texte colle au bord).
    return cv2.copyMakeBorder(big, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)


def _ocr(binary: NDArray[np.uint8]) -> str:
    # 13 = « ligne brute », ajoute en DERNIER pour ne rien changer aux lectures qui
    # passaient deja. C'est le seul mode qui lit les nombres centres de l'ATH a un seul
    # nombre : mesure sur « 88 » et « 69 », les modes 6/7/8/10 rendent tous la chaine
    # vide. Le fallback etait donc muet exactement la ou le classifieur l'appelait.
    for psm in (7, 8, 10, 13):  # ligne, mot, caractere unique, ligne brute
        cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        txt = _tesseract().image_to_string(binary, config=cfg)
        if any(c.isdigit() for c in txt):
            return txt.strip()
    return ""


# Exemplaires du classifieur maison, charges paresseusement (None tant que non charges,
# False si indisponibles -> Tesseract seul).
_TEMPLATES: object = None


def _get_templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        try:
            from jev_tactics.perception.digits import DigitTemplates

            _TEMPLATES = DigitTemplates.load()
        except (FileNotFoundError, ModuleNotFoundError, KeyError):
            _TEMPLATES = False
    return _TEMPLATES or None


def _read_stat_tesseract(crop: NDArray[np.uint8]) -> int | None:
    digits = re.findall(r"\d+", _ocr(_preprocess(crop)))
    return int("".join(digits)) if digits else None


def read_stat(frame: NDArray[np.uint8], roi: Roi) -> int | None:
    """Lecture d'une ROI : classifieur maison (<1 ms) d'abord, Tesseract en fallback.

    Le fallback couvre les glyphes rejetes (ex. chiffre absent des exemplaires).
    DOFUS_OCR=tesseract force l'ancien chemin (oracle/deboggage).
    """
    crop = frame[roi.y0:roi.y1, roi.x0:roi.x1]
    if crop.size == 0:
        # ROI hors de la capture. Ce n'est pas un cas tordu : toutes les ROIs du projet
        # sont des pixels absolus cales sur 1919x1079, donc la moindre autre disposition
        # y mene. Chaque etage en aval levait une exception OpenCV differente
        # (cvtColor, connectedComponents, resize) -- et le DIAGNOSTIC s'effondrait ainsi
        # sur la situation qu'il existe precisement pour signaler, rendant une trace de
        # pile au lieu du mot « resolution ». On tranche ici, une fois, a la source :
        # une zone vide n'a rien a lire, c'est une lecture impossible.
        return None
    templates = _get_templates()
    if templates is not None and os.environ.get("DOFUS_OCR") != "tesseract":
        from jev_tactics.perception.digits import read_digits_from_mask

        value = read_digits_from_mask(_glyph_mask(crop), templates)
        if value is not None:
            return value
    return _read_stat_tesseract(crop)


def read_hp_gauge(frame: NDArray[np.uint8]) -> float | None:
    """Part de PV lue sur la HAUTEUR DE LIQUIDE de l'orbe, sans lire un seul chiffre.

    C'est le second lecteur, INDEPENDANT des chiffres, dont `read_hp_ratio` deplorait la
    disparition. Il fonctionne dans les deux dispositions d'ATH et ne demande pas que le
    maximum soit affiche -- une jauge porte la proportion par construction.

    Rend None si l'orbe n'est pas la : voir HEART_BOTTOM_TOLERANCE.
    """
    crop = frame[HEART_TOP_Y - 10:HEART_BOTTOM_Y + 6, HEART_X0:HEART_X1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(int)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Le rouge est a cheval sur 0 en teinte, d'ou les deux plages.
    red = ((hue < 12) | (hue > 168)) & (sat > 80) & (val > 50)
    # Trois pixels : une ligne isolee est un reflet, pas un niveau de liquide.
    rows = np.where(red.sum(axis=1) > 3)[0]
    if not rows.size:
        return None
    top = HEART_TOP_Y - 10 + int(rows[0])
    bottom = HEART_TOP_Y - 10 + int(rows[-1])
    if abs(bottom - HEART_BOTTOM_Y) > HEART_BOTTOM_TOLERANCE:
        return None
    ratio = float(min(1.0, max(0.0, (HEART_BOTTOM_Y - top) / (HEART_BOTTOM_Y - HEART_TOP_Y))))
    if ratio < HEART_FULL_ENOUGH:
        # Annoncer un orbe entame demande de prouver que la partie vide est bien de
        # l'orbe, et non un panneau pose dessus. Cf. HEART_RIM_MIN.
        band = slice(max(0, int(rows[0]) - HEART_RIM_ROWS), int(rows[0]))
        rim = ((sat < 60) & (val > 110))[band]
        if int(rim.sum()) < HEART_RIM_MIN:
            return None
    return ratio


def read_ui(frame: NDArray[np.uint8], rois: tuple[Roi, ...] = DEFAULT_ROIS) -> UiReading:
    """Lit toutes les ROIs d'une frame BGR -> UiReading."""
    values = {roi.name: read_stat(frame, roi) for roi in rois}
    pv, pv_max = values.get("pv"), values.get("pv_max")
    if pv is None and pv_max is None and rois is DEFAULT_ROIS:
        # Disposition « un seul nombre » : les deux boites empilees ont echoue ensemble,
        # ce qui est precisement la signature de cet ATH. `pv_max` reste None -- il n'est
        # pas affiche, et l'inventer serait pire que de l'ignorer.
        pv = read_stat(frame, HEART_ROI)
    return UiReading(pv=pv, pv_max=pv_max, pa=values.get("pa"), pm=values.get("pm"))


def read_hp_ratio(frame: NDArray[np.uint8]) -> float | None:
    """Part de PV restants, dans [0, 1], ou None si la lecture n'est pas sure.

    A quoi ca sert : decider s'il est raisonnable d'ENGAGER un combat. Un bot qui chasse
    a 10 % de PV meurt, se releve en etat de faiblesse, rechasse et remeurt -- la session
    ne produit rien et coute de l'energie.

    UN SEUL LECTEUR, ET CE DOCSTRING DISAIT LE CONTRAIRE. Il annoncait « deux lecteurs
    INDEPENDANTS : celui-ci mesure la longueur de la jauge, `read_ui` lit les chiffres ».
    Le lecteur de jauge n'existe plus -- le corps ci-dessous appelle `read_ui` et divise.
    Le test cense les croiser comparait donc la meme expression a elle-meme, et son « ecart
    maximal 0,000 » etait une identite, pas une concordance.

    CE QUE COUTAIT CETTE ABSENCE, constate le 19/08 : sur un ATH ou les PV tiennent dans
    un coeur -- « 69 », sans maximum affiche -- `pv_max` est introuvable, cette fonction
    rendait None, et le garde-fou « ne pas engager a bas PV » devenait aveugle sans que
    rien ne prenne le relais. Mesure sur les captures de session du 19/08 : `too_hurt`
    valait False a TOUS les PV, donc `--min-hp` ne s'est jamais declenche.

    LE SECOND LECTEUR EXISTE DE NOUVEAU, et c'est celui que ce docstring appelait :
    « une jauge, elle, se lit sans chiffres ». `read_hp_gauge` mesure la hauteur de
    liquide de l'orbe. Il sert de RECOURS quand les chiffres manquent -- pas de
    remplacement : les chiffres, quand ils sont la, sont exacts, la jauge est a 1,6 point
    pres (cf. le tableau de mesures pres de HEART_TOP_Y).

    Mesure sur les 31 captures du depot -- combat ET hors combat : la lecture repond
    partout ou `pv` et `pv_max` sont affiches.

    Deux lacunes se ferment ainsi. Le hors-combat, d'abord : aucune capture n'en existait
    a la premiere redaction, et c'est pourtant la que la question se pose puisque c'est la
    qu'on decide d'engager. `capture.png` et `hors_combat.png` donnent 1672/1672 et
    1667/1667, jauge et chiffres d'accord.

    Le pouvoir de DISCRIMINATION, ensuite : lire 1,0 partout serait aussi le comportement
    d'un lecteur casse, et toutes les captures de session montrent un personnage intact.
    `tacle.png` tranche -- 1146/1637, soit 0,700 par les chiffres comme par la jauge.

    D'ou le contrat : None quand la lecture echoue, JAMAIS une valeur de repli. L'appelant
    traite None comme « on ne sait pas » et ne bloque rien -- l'ignorance ne doit pas
    paralyser, principe deja applique aux pods et aux coordonnees.
    """
    reading = read_ui(frame)
    if reading.pv is not None and reading.pv_max:
        if reading.pv > reading.pv_max:
            # Deja rencontre : une lecture ou les PV depassent le maximum est fausse des
            # deux cotes, et un ratio > 1 laisserait croire a une pleine sante.
            return None
        return reading.pv / reading.pv_max
    # Pas de maximum affiche (ou chiffres illisibles) : la jauge repond quand meme.
    return read_hp_gauge(frame)


# Panneaux d'interface : sombres et peu satures, la ou la carte est claire et saturee.
# Mesure sur capture.png (part de pixels V<90) :
#
#     chat 0,82 | attitudes 0,84 | quetes 0,58   |   herbe 0,012 | chemin 0,017
#
# Deux ordres de grandeur separent les deux familles.
DARK_PANEL_VALUE = 90
DARK_PANEL_SATURATION = 150
# Aire minimale d'un panneau. Les taches sombres de la CARTE -- trous, ombres sous les
# arbres -- sont petites ; les panneaux mesures font 21 000 a 305 000 px.
MIN_PANEL_AREA = 15_000
PANEL_CLOSE = 25
# Au-dela, l'hypothese « sombre = panneau » est fausse : les panneaux mesures couvrent 0,22
# a 0,30 de l'ecran. Une image majoritairement sombre est autre chose -- un chargement, une
# scene de nuit, une frame de test -- et masquer 90 % de l'ecran rendrait les deux
# detecteurs de terrain aveugles sans rien dire.
MAX_PANEL_SHARE = 0.60
# Densite de pixels VRAIMENT sombres a l'interieur d'une composante retenue.
#
# La fermeture morphologique soude en une seule composante des pixels sombres EPARS -- les
# bordures de cases, l'ombre d'un sprite, un feuillage -- et produisait ainsi de fausses
# « fenetres » en plein milieu du plateau. Mesure sur quatre captures :
#
#     panneaux reels                       0,64 a 0,89
#     amas soudes (plateau, decor)         0,14 a 0,50
#
# Le seuil tombe dans l'intervalle vide. Sans lui, jusqu'a QUATRE cases de combat par
# capture etaient declarees cachees : le planificateur pouvait les viser, et le clic serait
# tombe sur ce que le masque croyait etre une fenetre.
MIN_PANEL_DENSITY = 0.55


def ui_mask(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Masque des pixels appartenant a un panneau d'interface OUVERT.

    `UI_ZONES` est une liste de trois rectangles fixes. Dessines sur une capture reelle,
    ils couvrent la barre d'objets, l'orbe de PV -- et un morceau de terrain JOUABLE en
    haut a droite. Pendant ce temps restent grand ouverts le chat qui defile, la carte des
    quetes, le panneau d'attitudes et la minicarte : quatre sources d'animation permanente,
    dans un projet dont les deux detecteurs de terrain reposent justement sur le mouvement.
    C'est la forme residuelle du « il hallucine completement les ressources ».

    Des rectangles de plus ne repareraient pas cela durablement : ces panneaux s'ouvrent,
    se ferment et se deplacent au gre du joueur. On les reconnait donc a ce qu'ils SONT --
    des surfaces sombres et peu saturees, la ou la carte est claire et saturee.

    On rend le MASQUE et non des boites englobantes : mesure sur les memes captures, le
    remplissage des boites descend a 0,18-0,28 pour les panneaux en L (chat + barre de
    sorts), donc une boite exclurait quatre fois trop de terrain.

    Ne couvre pas la barre d'icones du haut ni l'interieur de la minicarte, tous deux
    CLAIRS (0,02 et 0,04 de pixels sombres) : ceux-la restent du ressort de `UI_ZONES`.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark_raw = ((hsv[:, :, 2] < DARK_PANEL_VALUE)
                & (hsv[:, :, 1] < DARK_PANEL_SATURATION))
    dark = dark_raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (PANEL_CLOSE, PANEL_CLOSE))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(frame.shape[:2], dtype=np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < MIN_PANEL_AREA:
            continue
        member = labels == index
        # La fermeture a pu souder des pixels epars : on verifie que la composante est
        # VRAIMENT sombre, et pas seulement l'enveloppe d'un semis de points sombres.
        if dark_raw[member].mean() < MIN_PANEL_DENSITY:
            continue
        keep[member] = 255
    if keep.mean() / 255.0 > MAX_PANEL_SHARE:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    return keep


def hidden_by_ui(mask: NDArray[np.uint8] | None, x: float, y: float) -> bool:
    """Ce point tombe-t-il sur un panneau ? `None` = pas de masque, donc on ne sait pas,
    donc on ne bloque rien -- meme regle que partout ailleurs dans ce projet."""
    if mask is None:
        return False
    row, col = int(round(y)), int(round(x))
    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
        return False
    return bool(mask[row, col])
