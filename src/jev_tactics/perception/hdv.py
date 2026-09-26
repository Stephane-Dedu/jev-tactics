"""Lecture de l'Hotel de vente : prix par lot, prix moyen, niveau.

Trois choses distinguent cet ecran de tout le reste du projet, et chacune a dicte une
decision ici.

**1. La police est plus PETITE qu'ailleurs.** Les chiffres de l'HDV font 10 px de haut,
ceux du HUD de combat (orbes PV/PA/PM) en font 14. Mesure faite : les gabarits du HUD
rejettent 12 nombres sur 12 de l'HDV (scores 0,55-0,88 pour un seuil a 0,90). La regle de
rejet a donc fait exactement son travail -- elle a refuse de deviner. D'ou un SECOND jeu
de gabarits, `hdv_digit_templates.npz`, propre a cette taille.

**2. « Pas d'offre » ne se lit pas comme un prix nul.** Dans la liste, un prix
indisponible s'affiche `-----` ; ces tirets font 2 px de haut et le filtre de hauteur les
elimine, donc la ROI rend zero glyphe -- le cas sort gratuitement. Dans le panneau de
detail, c'est plus vicieux : le tableau Lot/Prix DISPARAIT et la ligne « Prix moyen »
affiche **0**. Un lecteur naif ecrirait 0 dans le CSV, ou 0 veut dire « inconnu » et
surement pas « gratuit ». C'est la panne silencieuse la plus couteuse de ce module :
elle ne casse rien, elle fausse toutes les moyennes ensuite.

**3. Les panneaux sont des fenetres, pas de l'ATH.** Le contenu du panneau de detail se
DECALE verticalement selon son etat : la ligne « Prix moyen » est a y=244 quand le tableau
est la, a y=238 quand il n'y est pas (panneau plus court). Les ROIs internes sont donc
donnees avec une bande de tolerance, jamais au pixel.

Ce qui permet de s'en sortir : la colonne `Lot` est une EMPREINTE CONSTANTE. Elle vaut
toujours 1, 10, 100 -- soit 1, 2 et 3 glyphes. On la lit d'abord, et elle tranche seule
entre « tableau present » (on peut lire les prix), « pas d'offre » (colonne vide) et
« geometrie fausse » (n'importe quoi d'autre). Un autotest gratuit, au meme endroit que
la donnee qu'il protege.

Geometrie relevee sur HDV.png / 5.png / cherche.png / padofre.png, disposition 1919x1079.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.digits import DigitTemplates, normalize_glyph

_TEMPLATES_FILE = "hdv_digit_templates.npz"
_TEMPLATES: DigitTemplates | None = None
_HEADER_FILE = "hdv_header.npz"
_HEADER: NDArray[np.uint8] | None = None


# --- Segmentation -----------------------------------------------------------------
#
# Memes bandes HSV que le reste du projet : les chiffres sont BLANCS (saturation basse,
# valeur haute), les icones sont SATUREES. Ce qui change ici, ce sont les bornes de
# hauteur, calees sur la police 10 px.

# DEUX seuils de luminosite, et c'est une mesure, pas un reglage de confort. L'HDV
# affiche deux populations de texte :
#
#   - les PRIX en blanc pur (V jusqu'a 255). Sous 170, les glyphes s'epaississent et se
#     TOUCHENT : `646` et `909` sortent en une seule composante de 25 px de large, qui
#     est alors ecartee par MAX_DIGIT_WIDTH -> zero chiffre lu. Il faut donc un seuil
#     HAUT pour les separer.
#   - les NIVEAUX en gris (V plafonne a 193). Au-dessus de 160, ils s'erodent et perdent
#     un chiffre : `90` devient `9`, `150` devient `15`. Il faut donc un seuil BAS.
#
# Les deux fenetres ne se recouvrent pas ([170, 200] contre [120, 160]) : aucun seuil
# unique ne lit correctement les deux colonnes, et en chercher un reviendrait a choisir
# laquelle des deux on accepte de mal lire. Le fond, lui, plafonne a V=111 sur une ligne
# selectionnee -- les deux seuils sont largement au-dessus.
MIN_VALUE_BRIGHT = 180   # prix (liste et panneau de detail), texte blanc
MIN_VALUE_DIM = 140      # colonne Niveau, texte gris
MAX_SATURATION = 70
# 8 : au-dessus des tirets « ----- » d'un prix indisponible, mesures a 2 px. C'est CE
# seuil qui fait sortir « aucune offre » sans une ligne de code de plus.
MIN_DIGIT_HEIGHT = 8
MAX_DIGIT_HEIGHT = 14
MAX_DIGIT_WIDTH = 20
MIN_DIGIT_AREA = 12


def glyph_mask(crop: NDArray[np.uint8],
               min_value: int = MIN_VALUE_BRIGHT) -> NDArray[np.uint8]:
    """Masque binaire des chiffres (blancs sur noir) d'une ROI de l'HDV."""
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] < MAX_SATURATION) & (hsv[:, :, 2] > min_value)
    return mask.astype(np.uint8) * 255


def segment(crop: NDArray[np.uint8],
            min_value: int = MIN_VALUE_BRIGHT) -> list[NDArray[np.float32]]:
    """ROI -> bitmaps normalises des chiffres, tries dans l'ordre de lecture.

    Le separateur de milliers n'apparait jamais ici : ce n'est pas une composante
    connexe assez haute. Verifie sur `1 018 182` (deux separateurs) -> 7 glyphes.
    """
    mask = glyph_mask(crop, min_value)
    if mask.size == 0:
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    glyphs: list[tuple[int, NDArray[np.float32]]] = []
    for index in range(1, count):
        left, top, width, height, area = (int(stats[index, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        if not MIN_DIGIT_HEIGHT <= height <= MAX_DIGIT_HEIGHT:
            continue
        if width > MAX_DIGIT_WIDTH or area < MIN_DIGIT_AREA:
            continue
        patch = (labels[top:top + height, left:left + width] == index)
        glyphs.append((left, normalize_glyph(patch.astype(np.uint8) * 255)))
    return [g for _, g in sorted(glyphs, key=lambda t: t[0])]


def load_templates() -> DigitTemplates:
    global _TEMPLATES
    if _TEMPLATES is None:
        ref = resources.files("jev_tactics.perception") / "data" / _TEMPLATES_FILE
        with resources.as_file(ref) as path:
            _TEMPLATES = DigitTemplates.load(str(path))
    return _TEMPLATES


def read_number(crop: NDArray[np.uint8],
                templates: DigitTemplates | None = None,
                min_value: int = MIN_VALUE_BRIGHT) -> int | None:
    """ROI -> entier, ou None si la ROI est vide OU si un chiffre est rejete.

    Tout-ou-rien, comme les autres lecteurs du projet : un prix dont un chiffre est
    douteux est un prix faux, et un prix faux dans un CSV ne se voit plus jamais.

    Attention a l'appelant : `None` recouvre ici DEUX cas qu'il faut separer -- rien a
    lire (pas d'offre) et lecture refusee (probleme). `count_glyphs` les distingue.
    """
    glyphs = segment(crop, min_value)
    if not glyphs:
        return None
    templates = templates or load_templates()
    value = 0
    for glyph in glyphs:
        digit, _score = templates.classify(glyph)
        if digit is None:
            return None
        value = value * 10 + digit
    return value


def count_glyphs(crop: NDArray[np.uint8],
                 min_value: int = MIN_VALUE_BRIGHT) -> int:
    """Nombre de chiffres presents, sans les classifier.

    Sert a lire la structure plutot que la valeur : l'empreinte de la colonne Lot, et la
    distinction « zero chiffre » (pas d'offre) / « chiffre rejete » (refus).
    """
    return len(segment(crop, min_value))


# --- Geometrie --------------------------------------------------------------------
#
# Disposition de reference 1919x1079. Ces valeurs sont les COINS de la fenetre HDV telle
# qu'elle s'ouvre par defaut ; `anchor` permet de les decaler si la fenetre bouge.

Roi = tuple[int, int, int, int]  # (x0, y0, x1, y1)

# Liste centrale : 8 lignes visibles, pas alterne 76/75 px. Le pas n'est pas entier, donc
# les hauts de ligne sont donnes tels que MESURES plutot que recalcules -- une erreur
# d'arrondi cumulee sur 8 lignes finirait par sortir de la bande.
LIST_ROW_TOPS: tuple[int, ...] = (312, 388, 463, 539, 614, 690, 766, 841)
# Bande verticale autour du haut de ligne. Large exprès : avec la recherche active, la
# grille est decalee de +2 px (mesure), et le haut d'un glyphe varie d'un pixel selon le
# chiffre. Rien d'autre ne tombe dans ces colonnes, donc la largeur ne coute rien.
ROW_BAND_ABOVE = 8
ROW_BAND_BELOW = 16

LIST_LEVEL_X: tuple[int, int] = (1112, 1168)
LIST_PRICE_X: tuple[int, int] = (1232, 1322)

# Panneau de detail (fenetre flottante). JUSQU'A QUATRE lignes de lot.
#
# La v1 n'en connaissait que trois, parce que les captures de reference n'en montraient
# que trois. En jeu, un objet suffisamment offert en propose un QUATRIEME : le lot de
# 1 000 (releve sur « Coque Endommagee » : 105 / 1 095 / 12 995 / 149 995). Le nombre de
# lignes depend donc des offres, et rien ne le garantit d'avance.
#
# Consequence directe sur la lecture : exiger une empreinte fixe est faux. Un objet qui
# n'a qu'un lot de 1 ne rendra jamais (1, 2, 3), et le releve l'attendait jusqu'au
# timeout en annoncant « panneau fige » -- alors que le panneau etait complet depuis le
# debut.
DETAIL_ROW_TOPS: tuple[int, ...] = (354, 396, 437, 479)
DETAIL_LOT_X: tuple[int, int] = (185, 228)
DETAIL_PRICE_X: tuple[int, int] = (330, 412)
# Chiffres de « Prix moyen : N ». Le libelle occupe x 204..277, les chiffres commencent a
# 291, l'icone kama est a 349 -- la fenetre x isole donc les chiffres seuls. La bande y
# couvre les deux etats du panneau (244 avec tableau, 238 sans).
DETAIL_AVERAGE: Roi = (285, 230, 342, 262)

# Lots possibles, dans l'ordre ou le jeu les affiche. Une ligne presente porte forcement
# l'une de ces valeurs, et les lignes presentes forment toujours un PREFIXE de cette
# suite : c'est ce qui permet de distinguer un tableau court (peu d'offres) d'un tableau
# en cours de dessin (trou au milieu).
LOT_VALUES: tuple[int, ...] = (1, 10, 100, 1000)
# Empreinte attendue si les quatre lots sont la. Conservee pour le diagnostic : elle ne
# sert plus de condition d'acceptation, cf. `read_detail`.
LOT_SIGNATURE: tuple[int, ...] = (1, 2, 3, 4)

# Bouton de vidage du champ de recherche. Il n'existe QUE lorsque le champ contient du
# texte (mesure : 0 pixel clair a cet endroit sur un champ vide, ~70 sur un champ rempli).
# On s'en sert donc a la fois pour vider et pour savoir s'il y a quelque chose a vider.
CLEAR_BUTTON: Roi = (770, 244, 800, 266)
CLEAR_BUTTON_MIN_PIXELS = 20


def search_has_text(frame: NDArray[np.uint8],
                    anchor: tuple[int, int] = (0, 0)) -> bool:
    """Le champ de recherche contient-il du texte ? (par la presence de sa croix)

    Sert a ne cliquer la croix que lorsqu'elle existe. `Ctrl+A` ne peut pas remplacer ce
    geste : le champ du jeu N'APPLIQUE PAS le modificateur, et recoit simplement la lettre
    « a ». Constate en jeu -- le champ contenait `aGroin de Dragon Cochon`, c'est-a-dire
    l'ancien texte, puis un « a », puis le nouveau nom. Aucune recherche ne pouvait
    aboutir, et rien ne le signalait.
    """
    crop = _crop(frame, _shift(CLEAR_BUTTON, anchor))
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 140)
    return bool(bright.sum() >= CLEAR_BUTTON_MIN_PIXELS)

# Points de clic, releves sur cherche.png.
SEARCH_FIELD: tuple[int, int] = (670, 254)   # champ « Rechercher... », loin de la croix
SEARCH_CLEAR: tuple[int, int] = (783, 254)   # croix de vidage du champ
# Abscisse de clic d'une ligne : dans la colonne Nom, loin des bords. L'ordonnee est
# celle du haut des glyphes de prix decalee vers le milieu de la ligne (haute de ~76 px).
ROW_CLICK_X = 1000
ROW_CLICK_DY = 5


def search_field(anchor: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    return (SEARCH_FIELD[0] + anchor[0], SEARCH_FIELD[1] + anchor[1])


def search_clear(anchor: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    return (SEARCH_CLEAR[0] + anchor[0], SEARCH_CLEAR[1] + anchor[1])


def row_click_point(index: int, anchor: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    """Ou cliquer pour selectionner la ligne `index` de la liste."""
    return (ROW_CLICK_X + anchor[0], LIST_ROW_TOPS[index] + ROW_CLICK_DY + anchor[1])


# BARRE DE TITRE de la fenetre (« Hotel de vente » et son icone), relevee sur HDV.png.
#
# POURQUOI CELLE-CI ET PAS L'EN-TETE DES COLONNES. La v1 ancrait sur la ligne « Nom /
# Niveau / Prix moyen », en la croyant fixe. Elle ne l'est pas : cette ligne n'existe QUE
# lorsque la liste a des resultats. A l'ouverture, l'HDV affiche « Veuillez selectionner
# une categorie ou rechercher un objet » et il n'y a pas d'en-tete du tout.
#
# Les quatre captures de reference avaient toutes des resultats affiches, donc le defaut
# etait invisible ici. Il est apparu au premier lancement en jeu, et sous un deguisement :
# correlation 0,45, c'est-a-dire le score exact d'un ecran SANS HDV. Le message accusait
# donc une fenetre fermee alors qu'elle etait ouverte, au bon endroit, et vide.
#
# La barre de titre, elle, est du decor de fenetre par construction -- c'est meme par elle
# qu'on la deplace. Mesure sur six captures d'HDV (dont l'ecran vide) : correlation 1,000
# partout, contre 0,45-0,58 sur les ecrans sans HDV, a toutes les echelles.
HEADER_PATCH: Roi = (860, 148, 1100, 180)
# Correlation minimale pour accepter l'ancrage. Le motif est du texte net sur fond uni :
# une vraie correspondance monte au-dessus de 0,95, une fausse plafonne bien plus bas.
HEADER_MIN_SCORE = 0.90
# Deplacement maximal cherche autour de la position de reference. Une fenetre deplacee
# de plus que ca sort de l'ecran ; au-dela on prefere echouer que trouver un sosie.
# Facteur de sous-echantillonnage de la passe grossiere, et son seuil de rejet.
#
# LE SEUIL EST BEAUCOUP PLUS BAS QUE HEADER_MIN_SCORE, ET C'EST MESURE. Reduire l'image
# d'un facteur 4 transforme un bandeau de texte de 405x22 px en 101x5 : le detail qui
# fait la correspondance disparait, et la separation s'effondre avec lui. Mesure sur les
# captures du depot, en cherchant dans TOUTE l'image :
#
#     echelle   avec HDV (min)   sans HDV (max)   ecart
#     1/1            1,000            0,493       +0,507
#     1/2            1,000            0,530       +0,470
#     1/4            1,000            0,582       +0,418
#
# Le seuil grossier est pose a 0,70, entre les deux nuages. Qu'il laisse passer un
# candidat douteux ne coute rien : la passe fine, en pleine resolution, tranche ensuite a
# 0,90. C'est l'inverse qui serait grave -- une passe grossiere trop severe eliminerait le
# bon candidat avant que quiconque puisse le regarder.
COARSE_SCALE = 4
COARSE_MIN_SCORE = 0.70


def load_header() -> NDArray[np.uint8]:
    global _HEADER
    if _HEADER is None:
        ref = resources.files("jev_tactics.perception") / "data" / _HEADER_FILE
        with resources.as_file(ref) as path:
            _HEADER = np.load(path)["patch"]
    return _HEADER


@dataclass(frozen=True)
class Anchor:
    """Ou se trouve la fenetre de l'HDV, par rapport a la disposition de reference."""

    offset: tuple[int, int] = (0, 0)
    score: float = 0.0
    found: bool = False
    message: str = ""


def find_anchor(frame: NDArray[np.uint8]) -> Anchor:
    """Localise la fenetre de l'HDV et rend le decalage a appliquer aux ROIs.

    POURQUOI CETTE FONCTION EXISTE. Le reste du projet lit l'ATH, qui est colle aux bords
    de l'ecran et ne bouge donc jamais. L'HDV, lui, est une FENETRE : le joueur peut la
    deplacer. Des ROIs en pixels ecran y sont fausses au premier glisser-deposer, et
    fausses SANS ERREUR -- on lirait le decor a la place des prix.

    Rendre `found=False` plutot qu'un decalage devine est volontaire : mieux vaut refuser
    de relever des prix que d'en relever de faux, puisque le CSV, lui, ne dit pas d'ou
    vient un chiffre.
    """
    patch = load_header()
    ph, pw = patch.shape[:2]
    x0, y0, x1, y1 = HEADER_PATCH

    # CHEMIN RAPIDE : la fenetre est-elle a sa position par defaut ? C'est le cas de
    # loin le plus frequent -- l'HDV s'ouvre toujours au meme endroit et rien ne la
    # deplace tant que le joueur ne la traine pas. Une seule comparaison a cette
    # position coute 0,1 ms contre ~50 ms pour la recherche plein cadre, et la boucle de
    # releve appelle cette fonction a CHAQUE capture.
    reference = frame[y0:y1, x0:x1]
    if reference.shape == patch.shape:
        score = float(cv2.matchTemplate(reference, patch, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score >= HEADER_MIN_SCORE:
            return Anchor(offset=(0, 0), score=score, found=True,
                          message="HDV a sa position de reference")

    # LA RECHERCHE COUVRE TOUTE L'IMAGE, ET C'EST UNE CORRECTION. Elle etait bornee a une
    # fenetre de +-320 px autour de la position de reference, sur l'idee qu'une fenetre ne
    # se deplace pas de beaucoup. Premier lancement en jeu : correlation 0,45, soit
    # exactement le score d'un ecran SANS HDV -- la fenetre etait ouverte, simplement
    # ailleurs, et la recherche ne regardait pas la ou elle etait. Une borne posee par
    # confort avait produit un diagnostic faux (« fenetre fermee ») sur une fenetre
    # ouverte. A quart de resolution, chercher partout coute 12 ms : il n'y avait rien a
    # economiser.
    zone = frame
    if zone.shape[0] < ph or zone.shape[1] < pw:
        return Anchor(message="capture trop petite pour contenir l'en-tete de l'HDV")
    sx0 = sy0 = 0

    # RECHERCHE EN DEUX TEMPS. Le plein cadre coute ~150 ms, et l'ecran le plus frequent
    # du bot -- une carte, sans HDV -- les paie INTEGRALEMENT puisqu'il n'y a rien a
    # trouver. On cherche donc d'abord a quart de resolution (~16x moins de travail).
    # Un ecran sans HDV s'y elimine tout de suite ; seul un candidat plausible est
    # ensuite affine en pleine resolution, sur une petite fenetre.
    small_zone = cv2.resize(zone, None, fx=1 / COARSE_SCALE, fy=1 / COARSE_SCALE,
                            interpolation=cv2.INTER_AREA)
    small_patch = cv2.resize(patch, None, fx=1 / COARSE_SCALE, fy=1 / COARSE_SCALE,
                             interpolation=cv2.INTER_AREA)
    if (small_zone.shape[0] < small_patch.shape[0]
            or small_zone.shape[1] < small_patch.shape[1]):
        return Anchor(message="capture trop petite pour contenir l'en-tete de l'HDV")
    coarse = cv2.matchTemplate(small_zone, small_patch, cv2.TM_CCOEFF_NORMED)
    _cmin, coarse_score, _cminl, coarse_loc = cv2.minMaxLoc(coarse)
    if coarse_score < COARSE_MIN_SCORE:
        return Anchor(score=float(coarse_score),
                      message=f"en-tete de l'HDV introuvable (correlation grossiere "
                              f"{coarse_score:.2f} < {COARSE_MIN_SCORE}) — fenetre "
                              f"fermee, masquee, ou disposition differente de 1919x1079")

    # Affinage : autour du candidat, en pleine resolution. La marge couvre l'incertitude
    # du sous-echantillonnage (COARSE_SCALE px) plus un peu.
    cx, cy = coarse_loc[0] * COARSE_SCALE, coarse_loc[1] * COARSE_SCALE
    margin = COARSE_SCALE * 2
    rx0, ry0 = max(0, cx - margin), max(0, cy - margin)
    rx1 = min(zone.shape[1], cx + pw + margin)
    ry1 = min(zone.shape[0], cy + ph + margin)
    window = zone[ry0:ry1, rx0:rx1]
    if window.shape[0] < ph or window.shape[1] < pw:
        return Anchor(score=float(coarse_score),
                      message="candidat trop pres du bord pour etre confirme")
    result = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
    _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(result)
    if max_v < HEADER_MIN_SCORE:
        return Anchor(score=float(max_v),
                      message=f"en-tete de l'HDV introuvable (correlation {max_v:.2f} < "
                              f"{HEADER_MIN_SCORE}) — fenetre fermee, masquee, ou "
                              f"disposition differente de 1919x1079")
    found_x = sx0 + rx0 + max_l[0]
    found_y = sy0 + ry0 + max_l[1]
    return Anchor(offset=(found_x - x0, found_y - y0), score=float(max_v), found=True,
                  message=f"HDV ancree, decalage {(found_x - x0, found_y - y0)}")


def _shift(roi: Roi, anchor: tuple[int, int]) -> Roi:
    dx, dy = anchor
    x0, y0, x1, y1 = roi
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def _inside(frame: NDArray[np.uint8], roi: Roi) -> bool:
    """La ROI tient-elle ENTIEREMENT dans la capture ?

    Distinguer « rien a lire » de « rien a voir ». Une fenetre trainee a moitie hors de
    l'ecran donne une colonne Lot vide, donc l'empreinte (0, 0, 0), donc le verdict
    « objet pas en vente » -- une absence de marche affirmee alors qu'on n'a simplement
    pas regarde. Dans un CSV de prix, cette confusion est indetectable apres coup.
    """
    x0, y0, x1, y1 = roi
    h, w = frame.shape[:2]
    return 0 <= x0 and 0 <= y0 and x1 <= w and y1 <= h


def _crop(frame: NDArray[np.uint8], roi: Roi) -> NDArray[np.uint8]:
    x0, y0, x1, y1 = roi
    h, w = frame.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame[y0:y1, x0:x1]


def row_roi(top: int, columns: tuple[int, int], anchor: tuple[int, int] = (0, 0)) -> Roi:
    """ROI d'une cellule : bande verticale autour de `top`, largeur de la colonne."""
    x0, x1 = columns
    return _shift((x0, top - ROW_BAND_ABOVE, x1, top + ROW_BAND_BELOW), anchor)


# --- Lecture de la liste ----------------------------------------------------------

# Vocabulaire de statut, partage avec le CSV. Trois cas et pas deux : « je n'ai pas pu
# lire » n'est pas « il n'y a pas d'offre », et les confondre reviendrait a inventer une
# absence de marche a chaque fois que la fenetre a bouge.
OK = "ok"
NO_OFFER = "aucune_offre"
REFUSED = "refus"


@dataclass(frozen=True)
class ListRow:
    """Une ligne de la liste centrale."""

    index: int
    level: int | None = None
    average_price: int | None = None
    status: str = REFUSED
    reason: str = ""

    @property
    def empty(self) -> bool:
        """Ligne sans contenu : au-dela du dernier resultat de la recherche."""
        return self.level is None and self.status == NO_OFFER


def read_list(frame: NDArray[np.uint8],
              anchor: tuple[int, int] = (0, 0),
              templates: DigitTemplates | None = None) -> list[ListRow]:
    """Frame -> les 8 lignes visibles de la liste.

    Une ligne sans niveau ET sans prix est une ligne VIDE (la recherche a rendu moins de
    8 resultats) -- pas une anomalie. Une ligne avec un niveau mais un prix illisible,
    elle, est un refus : il y a bien un objet, on n'a pas su lire son prix.
    """
    templates = templates or load_templates()
    rows: list[ListRow] = []
    for index, top in enumerate(LIST_ROW_TOPS):
        level_crop = _crop(frame, row_roi(top, LIST_LEVEL_X, anchor))
        price_crop = _crop(frame, row_roi(top, LIST_PRICE_X, anchor))
        level = read_number(level_crop, templates, MIN_VALUE_DIM)
        price_glyphs = count_glyphs(price_crop)
        price = read_number(price_crop, templates) if price_glyphs else None

        if level is None and price_glyphs == 0:
            rows.append(ListRow(index, status=NO_OFFER, reason="ligne vide"))
        elif level is None:
            rows.append(ListRow(index, average_price=price, status=REFUSED,
                                reason="niveau illisible"))
        elif price_glyphs == 0:
            # Les tirets « ----- » : l'objet existe, il n'est pas en vente.
            rows.append(ListRow(index, level=level, status=NO_OFFER,
                                reason="prix moyen indisponible"))
        elif price is None:
            rows.append(ListRow(index, level=level, status=REFUSED,
                                reason=f"prix illisible ({price_glyphs} chiffres)"))
        else:
            rows.append(ListRow(index, level=level, average_price=price, status=OK))
    return rows


# --- Lecture du panneau de detail -------------------------------------------------


@dataclass(frozen=True)
class DetailReading:
    """Prix par lot d'un objet, ou la RAISON du refus."""

    lots: dict[int, int] = field(default_factory=dict)
    average_price: int | None = None
    status: str = REFUSED
    reason: str = ""

    def unit_price(self, lot: int) -> float | None:
        price = self.lots.get(lot)
        return None if price is None else price / lot


# En-tete du panneau de detail : nom de l'objet, sa ligne « Niv. N » et sa categorie.
# Sert d'empreinte de fraicheur, pas de lecture -- voir `panel_fingerprint`.
DETAIL_HEADER: Roi = (190, 196, 540, 240)

# Corps du panneau de detail, et la luminosite au-dessus de laquelle on n'y est plus.
# Mesure des medianes de V sur les captures : panneau ouvert 62-76, decor de jeu 101-180.
# Le seuil est pose au milieu de l'ecart.
DETAIL_BODY: Roi = (120, 300, 545, 460)
PANEL_MAX_VALUE = 88


def panel_present(frame: NDArray[np.uint8],
                  anchor: tuple[int, int] = (0, 0)) -> bool:
    """Le panneau de detail est-il ouvert ? Sert au DIAGNOSTIC, pas a la decision.

    Un panneau ferme (aucun objet selectionne) et un panneau mal cadre produisent la meme
    chose du point de vue des prix : rien de lisible. Mais ils n'envoient pas chercher au
    meme endroit -- l'un est un etat NORMAL du deroule, l'autre une panne. Sans cette
    distinction, `doctor --hdv` accuse la geometrie alors qu'il n'y a simplement rien de
    selectionne.

    C'est volontairement un simple critere de luminosite, et il n'autorise JAMAIS une
    lecture : l'empreinte de la colonne Lot reste seule juge de ce qui peut etre lu. Au
    pire, ce test se trompe de message ; il ne peut pas faire ecrire un prix faux.
    """
    crop = _crop(frame, _shift(DETAIL_BODY, anchor))
    if crop.size == 0:
        return False
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    return bool(np.median(value) <= PANEL_MAX_VALUE)


# Zone des huit lignes de resultats. Sert d'empreinte de fraicheur de la LISTE, comme
# DETAIL_HEADER pour le panneau.
LIST_REGION: Roi = (850, 285, 1340, 880)

# Fond d'une ligne, lu dans une bande SANS TEXTE (entre les colonnes Niveau et Prix).
# Le jeu y met trois teintes, et elles sont exactes -- mesure sur quatre captures, toutes
# lignes confondues, sans une seule valeur intermediaire :
#
#     V = 111  ligne SELECTIONNEE
#     V =  76  ligne normale
#     V =  50  au-dela du dernier resultat (la liste est plus courte que huit)
#
# Ce que ca apporte : une confirmation IMMEDIATE qu'un clic a porte. Sans elle, un clic
# qui rate et un panneau lent a s'ouvrir produisent le meme symptome -- on attend, puis on
# abandonne -- et on ne sait pas lequel des deux corriger.
ROW_BACKGROUND_X: tuple[int, int] = (1170, 1225)
ROW_SELECTED_VALUE = 111
ROW_NORMAL_VALUE = 76
ROW_EMPTY_VALUE = 50
# Moitie de l'ecart entre deux teintes voisines (76 -> 111) : large, mais elles sont si
# nettes qu'aucune valeur reelle ne s'en approche.
ROW_VALUE_TOLERANCE = 12


def row_background(frame: NDArray[np.uint8], index: int,
                   anchor: tuple[int, int] = (0, 0)) -> float:
    """Luminosite mediane du fond de la ligne `index`."""
    top = LIST_ROW_TOPS[index]
    x0, x1 = ROW_BACKGROUND_X
    crop = _crop(frame, _shift((x0, top - 4, x1, top + 14), anchor))
    if crop.size == 0:
        return 0.0
    return float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]))


def row_selected(frame: NDArray[np.uint8], index: int,
                 anchor: tuple[int, int] = (0, 0)) -> bool:
    """La ligne `index` est-elle la ligne selectionnee ?"""
    return abs(row_background(frame, index, anchor) - ROW_SELECTED_VALUE) <= ROW_VALUE_TOLERANCE


def rows_shown(frame: NDArray[np.uint8], anchor: tuple[int, int] = (0, 0)) -> int:
    """Combien de lignes la recherche a rendues (0 a 8), par le fond et non les chiffres.

    Independant du lecteur de chiffres : une ligne dont le niveau serait illisible compte
    quand meme. Sert a distinguer « la recherche n'a rien rendu » de « je n'ai pas su
    lire », deux diagnostics qui envoient a des endroits opposes.
    """
    count = 0
    for index in range(len(LIST_ROW_TOPS)):
        if abs(row_background(frame, index, anchor) - ROW_EMPTY_VALUE) > ROW_VALUE_TOLERANCE:
            count += 1
    return count


def list_fingerprint(frame: NDArray[np.uint8],
                     anchor: tuple[int, int] = (0, 0)) -> int:
    """Empreinte de la liste de resultats. Deux recherches differentes, deux valeurs.

    MEME PIEGE QUE POUR LE PANNEAU, UN CRAN PLUS TOT. Apres la frappe, le jeu met un
    instant a refiltrer la liste. Lire tout de suite rend les resultats de la recherche
    PRECEDENTE, et le decalage est parfaitement silencieux : on cherche « Bombe de
    Graboule » (niveau 86), on lit la liste de « Coquille de Dragoeuf Charbon » (niveau
    70), et on conclut que l'objet n'existe pas.

    Constate en jeu sur 28 objets d'affilee, chaque diagnostic montrant le niveau de
    l'objet PRECEDENT. Le bilan disait « introuvable » -- le mot le plus trompeur
    possible, puisque la recherche, elle, avait parfaitement fonctionne.
    """
    crop = _crop(frame, _shift(LIST_REGION, anchor))
    if crop.size == 0:
        return 0
    return int(cv2.norm(crop.astype(np.float32), cv2.NORM_L1))


def panel_fingerprint(frame: NDArray[np.uint8],
                      anchor: tuple[int, int] = (0, 0)) -> int:
    """Empreinte de l'en-tete du panneau de detail. Deux objets differents, deux valeurs.

    LE PROBLEME QU'ELLE RESOUT. Apres avoir clique une ligne, le jeu met un instant a
    remplir le panneau. Lire trop tot rend les prix de l'objet PRECEDENT, qui sont alors
    enregistres sous le nom du nouveau. Rien ne le signale : trois prix parfaitement
    plausibles, ranges au mauvais endroit. C'est la panne la plus couteuse que ce module
    puisse produire, parce qu'elle survit dans le CSV.

    POURQUOI UNE EMPREINTE ET NON UNE LECTURE DU NIVEAU. Le panneau affiche « Niv. 100 »,
    ce qui semblerait suffire a verifier qu'on regarde le bon objet. Mesure faite : a
    cette taille les chiffres FUSIONNENT (deux composantes connexes pour « 100 », une
    seule pour « 50 »), aux deux seuils utiles. Un verificateur qui echoue sur des cas
    normaux ferait refuser des relevés justes.

    Comparer l'en-tete a elle-meme n'a aucun de ces defauts : le nom de l'objet y est
    ecrit en toutes lettres, deux objets distincts ne peuvent pas donner la meme image, et
    aucun caractere n'a besoin d'etre reconnu.

    Limite assumee : deux objets consecutifs au panneau rigoureusement identique feraient
    conclure « pas rafraichi ». Le cas est improbable (il faudrait le meme nom) et son
    issue est un REFUS journalise, pas un prix faux.
    """
    crop = _crop(frame, _shift(DETAIL_HEADER, anchor))
    if crop.size == 0:
        return 0
    return int(cv2.norm(crop.astype(np.float32), cv2.NORM_L1))


def lot_signature(frame: NDArray[np.uint8],
                  anchor: tuple[int, int] = (0, 0)) -> tuple[int, ...]:
    """Nombre de chiffres de chaque ligne de la colonne Lot.

    (1, 2, 3) = tableau present et bien cadre. (0, 0, 0) = pas d'offre. Autre chose =
    on ne sait pas ou on est, et c'est le seul moment ou on peut encore s'en rendre
    compte avant d'ecrire un prix faux.
    """
    return tuple(count_glyphs(_crop(frame, row_roi(top, DETAIL_LOT_X, anchor)))
                 for top in DETAIL_ROW_TOPS)


def read_average(frame: NDArray[np.uint8],
                 anchor: tuple[int, int] = (0, 0),
                 templates: DigitTemplates | None = None) -> int | None:
    """« Prix moyen : N » du panneau de detail. 0 est rendu None -- VOIR L'EN-TETE.

    Le jeu affiche litteralement `Prix moyen : 0` quand la statistique n'existe pas
    (l'infobulle de l'inventaire, elle, ecrit « indisponible » en toutes lettres). Zero
    kama n'est pas un prix ; le laisser passer contaminerait toute moyenne calculee sur
    le CSV, et rien ne le signalerait.
    """
    value = read_number(_crop(frame, _shift(DETAIL_AVERAGE, anchor)), templates)
    return None if value == 0 else value


def read_detail(frame: NDArray[np.uint8],
                anchor: tuple[int, int] = (0, 0),
                templates: DigitTemplates | None = None) -> DetailReading:
    """Frame -> prix des lots 1 / 10 / 100 de l'objet selectionne."""
    templates = templates or load_templates()

    # Avant toute lecture : le panneau est-il entierement visible ? Sinon on refuse, on
    # n'interprete pas (cf. `_inside`).
    required = [row_roi(top, column, anchor)
                for top in DETAIL_ROW_TOPS
                for column in (DETAIL_LOT_X, DETAIL_PRICE_X)]
    if not all(_inside(frame, roi) for roi in required):
        return DetailReading(status=REFUSED,
                             reason="panneau de detail hors cadre — fenetre a moitie "
                                    "sortie de l'ecran, ou capture partielle")

    average = read_average(frame, anchor, templates)

    # Panneau ferme d'abord : sans lui, les ROIs tombent sur le decor du jeu et n'importe
    # quel reflet compte comme un chiffre.
    if not panel_present(frame, anchor):
        return DetailReading(average_price=average, status=REFUSED,
                             reason="aucun objet selectionne, panneau de detail ferme")

    lots: dict[int, int] = {}
    ended = False   # une ligne vide a ete rencontree : les suivantes doivent l'etre aussi

    for lot, top in zip(LOT_VALUES, DETAIL_ROW_TOPS):
        lot_crop = _crop(frame, row_roi(top, DETAIL_LOT_X, anchor))
        price_crop = _crop(frame, row_roi(top, DETAIL_PRICE_X, anchor))
        lot_glyphs, price_glyphs = count_glyphs(lot_crop), count_glyphs(price_crop)

        # LA COLONNE LOT FAIT AUTORITE sur l'existence de la ligne. Un panneau court
        # s'arrete avant les dernieres lignes, et leurs ROIs tombent alors sur le decor :
        # on y a mesure un glyphe parasite dans la colonne Prix, qui aurait fait conclure
        # a une ligne a moitie dessinee. La colonne Lot, elle, ne contient jamais rien
        # d'autre que 1, 10, 100 ou 1000.
        if lot_glyphs == 0:
            ended = True          # tableau plus court : normal, peu d'offres
            continue
        if ended:
            # Une ligne pleine APRES une ligne vide : les lots forment toujours un
            # prefixe de 1/10/100/1000, donc c'est un tableau a moitie dessine.
            return DetailReading(average_price=average, status=REFUSED,
                                 reason=f"ligne de lot {lot} presente alors qu'une "
                                        f"precedente est vide — tableau en cours "
                                        f"d'affichage")
        if price_glyphs == 0:
            return DetailReading(average_price=average, status=REFUSED,
                                 reason=f"lot {lot} affiche sans son prix — tableau en "
                                        f"cours d'affichage")

        # La colonne Lot est relue POUR SA VALEUR : des nombres au bon nombre de chiffres
        # mais qui ne seraient pas 1/10/100/1000 signaleraient une geometrie juste par
        # hasard.
        read_lot = read_number(lot_crop, templates)
        if read_lot != lot:
            return DetailReading(average_price=average, status=REFUSED,
                                 reason=f"lot lu {read_lot}, attendu {lot}")
        price = read_number(price_crop, templates)
        if price is None:
            return DetailReading(average_price=average, status=REFUSED,
                                 reason=f"prix du lot {lot} illisible")
        lots[lot] = price

    if not lots:
        return DetailReading(average_price=average, status=NO_OFFER,
                             reason="tableau Lot/Prix absent — objet pas en vente")

    return DetailReading(lots=lots, average_price=average, status=OK)
