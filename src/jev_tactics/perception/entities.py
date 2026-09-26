"""Detecteur d'entites v1 : marqueurs de cellule, echantillonnes VIA LA GRILLE (zero ML).

DEUX CORRECTIONS issues de l'observation de vraies captures de combat 3.6 (phase 2.1) :

1. **Il n'y a pas de barres de vie flottantes.** Le doc d'archi supposait des barres au-
   dessus des sprites (heritage Dofus 2) : verifie sur 5 captures, elles n'existent pas.
   Dofus 3.6 signale l'equipe par un **marqueur de cellule** (losange colore sous
   l'entite) et les PV par la **timeline** en haut de l'ecran.

2. **La detection par blob colore est inexploitable ici.** Mesure : une bande HSV sur
   l'orange du marqueur + filtres de forme (taille cellule, ratio, remplissage) sort
   **70 a 120 candidats par frame** — parce que le terrain lui-meme est un pavage de
   losanges de meme taille et de teinte voisine. Aucun reglage de seuil ne separe.

D'ou l'algorithme retenu, qui exploite ce que le projet sait deja exactement : **la
geometrie**. Plutot que chercher des losanges dans les pixels, on projette les 560
cellules connues et on echantillonne le contour de chacune :

    pour chaque cellule -> 4 coins via l'homographie -> points le long des aretes
                        -> vote de couleur -> equipe (ou rien)

Avantages : pas de seuil de forme, pas d'inversion pixel->cellule (l'id est connu par
construction), robuste a l'occlusion partielle par le sprite (vote sur ~48 points), et
le pavage du terrain n'est plus un piege puisqu'on ne teste que les aretes de cellules.

/!\\ PREREQUIS : une homographie valide POUR LA CARTE COURANTE. C'est la phase 4
(ancrage) du plan, qui devient donc un prealable au detecteur — l'ordre initial du plan
est inverse par cette mesure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.calibration import CELL_COORDS, GRID_CELLS, project
from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception.timeline import TimelineEntry
from jev_tactics.state import Entity, Team

# --- Bandes HSV des marqueurs (calibrees sur dofusscreen*.png) -----------------------
# La teinte est le discriminant CRITIQUE : le marqueur ennemi est orange-ROUGE (H<=17)
# alors que le terrain de ces cartes est jaune (H~25-30). Balayage mesure sur
# dofusscreen3 (4 ennemis + 1 joueur, 220 cellules visibles) :
#     H<=14 -> 0 detection (bande trop etroite)
#     H<=16 -> 5   |  H<=18 -> 5      <- plateau stable, insensible a S_min
#     H<=20 -> 15  |  H<=22 -> 22  |  H<=25 -> 62  |  H<=28 -> 173 (le terrain passe)
# On se place au centre du plateau. Elargir la teinte au-dela de 18 fait exploser les
# faux positifs : c'est le terrain, pas un defaut de reglage.
#
# /!\\ Ces bandes ne sont PAS encore validees de bout en bout : le detecteur ne peut pas
# etre exerce sur du reel tant que l'ANCRAGE de la grille n'est pas resolu (cf.
# calibration/grid.py) -- les cellules 0..559 se projettent a cote du plateau, donc les
# detections obtenues jusqu'ici portaient sur le decor (feuillage roux) et non sur les
# marqueurs. La bande de teinte ci-dessus vient d'un balayage quantitatif, pas d'un
# controle visuel des marqueurs : a reverifier une fois l'ancrage en place.
#
# Marqueur du joueur : c'est le CONTOUR BLANC qui porte le signal, pas l'anneau bleu.
# Balayage de toutes les cases sur 3 captures, a inset 0.40 :
#     signature   case joueur      meilleure autre case
#     blanc       0.28 / 0.42 / 0.47      <= 0.11        <- retenu
#     bleu        0.05 / 0.14 / 0.17      <= 0.02        (anneau trop fin)
# Le blanc separe d'un facteur ~3 ; le bleu serait sous le seuil de vote.
#
# MARGE MESUREE DEPUIS, avec le libelle « MOI » des captures de session comme ORACLE -- il
# designe la case du personnage, on mesure ce que le detecteur y voit :
#
#     4 captures sur 6    vote 0,25 a 0,27      seuil 0,20
#     2 captures sur 6    vote 0,00             contour a H=36-43, S=170-180 (vert sature)
#
# CETTE MESURE EST FAUSSEE, ET IL FAUT LE SAVOIR AVANT DE TOUCHER AUX SEUILS. Les captures
# de `data/runs/` portent l'overlay du bot INCRUSTE, et `overlay.py` dessine le marqueur du
# joueur en (255, 140, 0) BGR -- soit HSV (104, 255, 255) -- sous forme d'un losange EPAIS
# pose sur le contour de la case. Or c'est exactement la que `EDGE_INSETS` echantillonne.
#
# Part des points echantillonnes tombant sur ce bleu, mesuree sur les 17 cases de joueur
# validees par la zone de deplacement :
#
#     mediane 52 %        et jusqu'a 58 %
#
# Le detecteur y lit donc une image dont le bot a REPEINT la moitie des preuves. La
# production capture l'ecran brut et n'a pas ce probleme.
#
# CE QUI SURVIT A LA CORRECTION. Sur ces 17 cases, 8 passent le seuil et 9 sont a zero ;
# mais parmi ces neuf, SIX portent ~52 % de bleu d'overlay et leur verdict est inexploitable.
# Trois seulement (220501, 222542, 224545) sont exemptes de tout overlay et donnent
# malgre tout 0 % de pixels dans la bande alliee : celles-la sont de vrais echecs. Le taux
# reel est donc quelque part entre 8/17 et 14/17, et ces captures ne permettent pas de
# trancher.
#
# LA MESURE REFAITE SUR DES PIXELS PROPRES, et elle renverse la conclusion. Les six
# captures de combat de la racine du depot ne portent AUCUN pixel d'overlay (verifie par
# egalite exacte avec SELF_COLOUR et ENEMY_COLOUR) :
#
#     tacle.png          0,292        bug.png            0,458
#     combat1.png        0,333        bug4.png           0,458
#     bugcarreblanc.png  0,354        combat1920.png     0,271
#
# SIX SUR SIX au-dessus du seuil de 0,20, avec 35 % a 130 % de marge. Le vote de couleur du
# joueur n'est donc pas marginal : c'est la mesure qui l'etait, parce qu'elle portait sur
# des captures que le bot avait repeintes. Tenu par un test sur ces six captures.
#
# Non corrige, et pour une raison qui tient toujours : elargir la bande vers le vert sature
# attraperait la zone de deplacement elle-meme, qui couvre justement les cases autour du
# joueur. Le secours par `movement_centre` traite deja le cas, et il est desormais mesure
# (`Entity.cell_confirmed`).
#
# LA BORNE BASSE VALAIT 3, ET C'ETAIT LE DEFAUT LE PLUS COUTEUX DE LA PERCEPTION. Mesure
# sur 24 captures d'une vraie session : le contour des cases ennemies y est a la teinte 0
# -- rouge pur, S=255, V=237 a 255 -- donc SOUS la bande. Il n'en restait qu'une fraction
# dans 3..17, et le vote tombait a 0,12 et 0,19 pour un seuil a 0,20. Juste en dessous.
#
#     bande        comptages justes   manquent   en trop      (30 captures)
#     3..17               16             13          1
#     0..17               27              2          1
#
# Ajouter la bande haute (172..179), l'autre cote du rouge pur, ne change rien : la teinte
# lue est bien 0 et non 179.
TEAM_BANDS: dict[Team, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    Team.ENEMY: [((0, 150, 120), (17, 255, 255))],
    Team.ALLY: [((0, 0, 190), (179, 70, 255))],  # contour blanc : S bas, V haut
}

PLACEHOLDER_HP = 1000      # PV de remplissage tant qu'ils ne sont pas mesures
SAMPLES_PER_EDGE = 12      # points echantillonnes par arete du losange (4 aretes)
# Seuil de vote. Marges mesurees a inset 0.40 : ennemi 0.63 vs terrain 0.00 ;
# joueur 0.28-0.47 vs meilleure autre case 0.11. 0.20 laisse passer le joueur le plus
# faible (0.28) tout en restant au-dessus du bruit (0.11).
MIN_VOTE_RATIO = 0.20
# Rayon d'echantillonnage, en fraction de la demi-diagonale. Le jeu dessine le marqueur
# EN RETRAIT du bord de case ; balayage mesure sur dofusscreen3 (4 cases ennemies contre
# 5 cases de terrain nu), bande H<=18 :
#   inset  0.30  0.35  0.40  0.44  0.46  0.48
#   marq.  0.10  0.38  0.63  0.53  0.33  0.12     <- pic net a 0.40
#   sol    0.00  0.00  0.00  0.00  0.00  0.00
# A 0.46 (valeur initiale, choisie a priori) le marqueur etait quasi manque : c'etait la
# cause des non-detections, pas la bande de couleur.
# DEMI-LARGEUR D'UNE CASE EN UNITES DE GRILLE ANKAMA, pour le chemin par
# homographie (`cell_outline_points`). A NE PAS CONFONDRE avec `EDGE_INSETS`
# quelques lignes plus bas, qui est une FRACTION DE LA DEMI-DIAGONALE en pixels,
# lue par `detect_markers_on_board`. Deux espaces differents, deux quantites
# differentes -- et le 0,40 commun les faisait passer pour la meme chose au point
# de tromper un commentaire ecrit dans ce fichier meme.
GRID_CORNER_INSET = 0.40
# ... mais le rayon du marqueur VARIE : il est anime pendant notre tour. Mesure du vote
# du marqueur joueur selon l'inset, sur deux captures reelles :
#
#     inset      0.30  0.35  0.40  0.45  0.48
#     tacle.png  0.02  0.04  0.29  0.25  0.08
#     combat1    0.02  0.21  0.33  0.12  0.00
#
# Le pic se DEPLACE (0.40 ici, 0.35-0.40 la), et hors de sa fenetre le vote s'effondre a
# 0.02. Un inset fixe ne tient donc que par chance : il suffit que l'animation soit a la
# mauvaise phase pour que le joueur devienne introuvable -- ce qui produit « etat
# inexploitable » en plein combat, constate en jeu.
#
# On echantillonne donc PLUSIEURS rayons et on retient le meilleur. Le cout est
# negligeable (quelques dizaines de points par case) et le gain est de ne plus dependre
# d'un instant precis de l'animation.
EDGE_INSETS = (0.34, 0.40, 0.46)
# Un marqueur est une LIGNE sur le pourtour de la case : son interieur reste du terrain.
# Une case simplement TEINTEE, elle, est coloree de bout en bout.
#
# Ce controle vient d'un defaut constate en jeu : quand le personnage est TACLE, Dofus
# colore sa zone de deplacement en ROUGE -- la teinte meme que porte le marqueur ennemi.
# Le bot y a lu **12 ennemis pour 3 combattants**, tous groupes dans la meme direction
# (la zone), et a depense un tour entier sur une case vide.
#
# On echantillonne donc aussi le coeur de la case. Le seuil n'a pas besoin d'etre fin :
# une case teintee y donne pres de 1,0, un marqueur pres de 0.
INTERIOR_INSET = 0.15
# Part de l'INTERIEUR d'une case pouvant etre a la couleur de l'equipe sans que la case
# soit tenue pour teintee plutot que marquee.
#
# DEFAUT MESURE, non corrige : sur 24 captures d'une vraie session, CINQ ne detectent AUCUN
# ennemi alors que la timeline en annonce 1 a 3. Cause identifiee sur 20260807-221351.png --
# six des sept cases rouges votent 0,31 a 0,77 au contour, largement au-dessus du seuil de
# 0,20, et sont rejetees ICI parce que leur coeur est rempli a 96-100 %.
#
# Ces captures affichent les libelles « MOI » et « ENNEMIS » au-dessus des cases : un mode
# d'affichage ou la case ennemie est un aplat rouge UNIFORME, pas un anneau. Le garde-fou,
# ecrit contre le rouge de tacle, ne distingue pas les deux.
#
# Pourquoi ce n'est pas corrige a la legere : le rouge de tacle et l'aplat ennemi sont
# indiscernables case par case. Ce qui les separe est GLOBAL -- un tacle ou une portee
# couvre un anneau contigu autour du joueur (cf. `looks_like_spell_range`), un ennemi est
# isole. Mais ce test demande la position du joueur, qui vient elle-meme des marqueurs :
# la dependance est circulaire.
#
# LA CIRCULARITE, ELLE, EST LEVEE : `movement_centre` donne la position du joueur depuis la
# zone de deplacement, sans passer par les marqueurs. Ce n'est donc plus l'obstacle. Ce qui
# manque est un test de « portee de sort » assez fin, et deux formes ont ete mesurees :
#
#   - CONTIGUITE des distances (le test existant) : il repond oui presque partout des qu'on
#     lui donne six cases, parce que six distances quelconques forment une bande contigue
#     par hasard. Ecrit pour un anneau de vingt cases, il ne transpose pas ;
#   - REMPLISSAGE de l'anneau -- part des cases de l'anneau qui sont teintees :
#
#         portees de sort connues   0,75  0,74  0,40  0,33
#         cases ennemies supposees  0,28  0,24  0,10
#
#     Cinq centiemes separent les deux classes. A comparer aux marges retenues ailleurs
#     dans ce projet : 36 % pour le crane du chat, 56 % pour le mot « Epuise ».
#
# Ce qui trancherait sans mesure : une capture SANS le mode d'affichage qui remplit les
# cases. Le marqueur y redeviendrait un anneau, et ce garde-fou ferait son travail.
MAX_INTERIOR_RATIO = 0.5
# -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellMarker:
    """Un marqueur detecte sur une cellule identifiee."""

    cell: int
    team: Team
    vote_ratio: float


def cell_outline_points(
    homography: NDArray[np.float64], cell_id: int, per_edge: int = SAMPLES_PER_EDGE
) -> NDArray[np.float64]:
    """Points ecran le long du contour d'une cellule (losange), via l'homographie.

    Les 4 coins d'une cellule sont a +/- GRID_CORNER_INSET le long de chaque axe diagonal
    Ankama ; on interpole ensuite `per_edge` points sur chacune des 4 aretes.
    """
    cx, cy = CELL_COORDS[cell_id]
    corners_grid = np.array([
        [cx + GRID_CORNER_INSET, cy], [cx, cy + GRID_CORNER_INSET],
        [cx - GRID_CORNER_INSET, cy], [cx, cy - GRID_CORNER_INSET],
    ])
    corners = project(homography, corners_grid)
    edges = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        t = np.linspace(0.0, 1.0, per_edge, endpoint=False)[:, None]
        edges.append(a + t * (b - a))
    return np.vstack(edges)


def _team_masks(frame: NDArray[np.uint8]) -> dict[Team, NDArray[np.uint8]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks: dict[Team, NDArray[np.uint8]] = {}
    for team, bands in TEAM_BANDS.items():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lo, hi in bands:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        masks[team] = mask
    return masks


def cursor_position() -> tuple[int, int] | None:
    """Position ecran du curseur, ou None hors Windows / en cas d'echec.

    EXTRAITE de `hovered_cell`, qui en etait le seul appelant. Un second en a eu besoin :
    chercher OU s'affiche l'infobulle d'un monstre survole. Balayer la frame entiere pour
    la trouver rend 71 a 78 candidats sur CHAQUE capture -- le chat, la barre de sorts et
    le panneau de quetes sont tous du texte clair sur fond sombre. L'infobulle, elle, suit
    le curseur : savoir ou il est ecarte tout l'ATH d'un coup.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        point = _Point()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return None
    except Exception:
        return None
    return int(point.x), int(point.y)


CURSEUR_REEL = object()
"""Sentinelle : « va lire la position reelle de la souris ». Voir `hovered_cell`."""


def hovered_cell(board: BoardMap, position: object = CURSEUR_REEL) -> int | None:
    """Case actuellement SURVOLEE par le curseur, ou None.

    `position` EST UN PARAMETRE DEPUIS QU'ON A COMPRIS CE QU'IL COUTAIT D'ALLER LE
    CHERCHER. La derniere ligne du paragraphe ci-dessous -- « elle n'existe pas sur une
    capture figee » -- etait vraie de la BOUCLE, et fausse de l'APPEL : le code lisait la
    souris du systeme meme quand l'image venait d'un fichier. Rejouer une capture
    retranchait donc une case choisie par l'endroit ou trainait la souris de
    l'utilisateur, sans aucun rapport avec l'image.

    C'est la cause des echecs intermittents jamais attribues de
    `TestTheEnemyCountAcrossEverySessionCapture` et de
    `TestTheSelfLocationRateAcrossTheWholeSession` : souris posee ailleurs, 22 captures sur
    24 ; souris sur la case du personnage d'une capture, 21. Deterministe dans une session
    (la souris ne bouge pas), changeant d'une session a l'autre -- de quoi resister a la
    recherche du bug pendant des semaines, et faire ecarter tour a tour les fils
    d'execution, le cache d'octets, les gabarits et la calibration.

    Passer `None` pour rejouer une image stockee ; le defaut lit la souris, ce qui est le
    comportement voulu quand la frame vient de l'ecran.

    Le jeu dessine un contour BLANC sous la case survolee -- exactement la signature du
    marqueur du personnage joue. Or le bot deplace lui-meme la souris : apres un clic, son
    curseur reste sur le plateau, le jeu y dessine un contour, et a la frame suivante le
    bot **prend son propre survol pour son personnage**.

    Constate en jeu, et visible sur le calque : « MOI 126 » designait une case vide portant
    un contour blanc, pendant que le vrai personnage etait ailleurs. Toute la geometrie du
    tour se calculait alors depuis une position ou il n'etait pas, et la zone de
    deplacement du jeu -- centree sur le VRAI personnage -- ne touchait plus la position
    supposee : d'ou « 0 cases atteignables ».

    Une boucle de retroaction, donc, que seule l'observation du bot en action pouvait
    reveler : elle n'existe pas sur une capture figee.
    """
    if position is CURSEUR_REEL:
        position = cursor_position()
    if position is None:
        return None
    x, y = position

    cell = board.nearest(float(x), float(y))
    # `nearest` rend toujours une case, meme si le curseur est hors du plateau. On ne
    # l'ecarte que s'il est effectivement DANS la case, sinon on perdrait une case
    # legitime chaque fois que la souris est ailleurs a l'ecran.
    centre = board.center(cell)
    half = (abs(board.e_x[0]) + abs(board.e_y[0])) / 2.0
    if abs(x - centre[0]) > half or abs(y - centre[1]) > half / 2.0:
        return None
    return cell


def detect_markers_on_board(frame: NDArray[np.uint8], board: BoardMap) -> list[CellMarker]:
    """Frame BGR + plateau detecte -> marqueurs de cellule.

    On n'echantillonne QUE les cases jouables du plateau (et non 560 positions
    arbitraires) : c'est ce qui evite de noter le decor. Une case est retenue pour
    l'equipe dont le vote depasse MIN_VOTE_RATIO.
    """
    masks = _team_masks(frame)
    height, width = frame.shape[:2]
    markers: list[CellMarker] = []

    for index in range(len(board)):
        best_team, best_ratio = None, 0.0
        visible = False
        for inset in EDGE_INSETS:
            pts = board.outline(index, per_edge=SAMPLES_PER_EDGE, inset=inset)
            xs = np.round(pts[:, 0]).astype(int)
            ys = np.round(pts[:, 1]).astype(int)
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if inside.sum() < len(pts) * 0.8:  # case majoritairement hors ecran
                continue
            visible = True
            xs, ys = xs[inside], ys[inside]
            for team, mask in masks.items():
                ratio = float((mask[ys, xs] > 0).mean())
                if ratio > best_ratio:
                    best_team, best_ratio = team, ratio
        if not visible:
            continue
        if best_team is None or best_ratio < MIN_VOTE_RATIO:
            continue

        # Case teintee plutot que marqueur : le coeur est colore lui aussi.
        #
        # Reserve au camp ENNEMI, seul dont la bande entre en collision avec un
        # surlignage de case : le rouge du tacle. Le jeu ne teinte aucune case en blanc,
        # donc appliquer le controle au marqueur du JOUEUR n'ecarterait aucun faux
        # positif -- et risquerait d'ecarter le joueur lui-meme quand son anneau anime
        # deborde vers le centre. Un ennemi de trop coute un coup ; un joueur perdu
        # coute tout le tour.
        if best_team is Team.ENEMY:
            core = board.outline(index, per_edge=SAMPLES_PER_EDGE, inset=INTERIOR_INSET)
            cxs = np.round(core[:, 0]).astype(int)
            cys = np.round(core[:, 1]).astype(int)
            within = (cxs >= 0) & (cxs < width) & (cys >= 0) & (cys < height)
            if within.any():
                filled = float((masks[best_team][cys[within], cxs[within]] > 0).mean())
                if filled >= MAX_INTERIOR_RATIO:
                    continue

        markers.append(CellMarker(cell=index, team=best_team, vote_ratio=best_ratio))
    return markers


def detect_cell_markers(
    frame: NDArray[np.uint8], homography: NDArray[np.float64]
) -> list[CellMarker]:
    """Variante sur les 560 cellules Ankama -- conservee pour les tests de geometrie.

    A NE PAS utiliser sur des captures reelles : le plateau de combat n'est pas la carte
    560 (cf. calibration/grid.py), donc ces positions tombent a cote. Utiliser
    `detect_markers_on_board`.
    """
    masks = _team_masks(frame)
    height, width = frame.shape[:2]
    markers: list[CellMarker] = []

    for cell in range(GRID_CELLS):
        pts = cell_outline_points(homography, cell)
        xs = np.round(pts[:, 0]).astype(int)
        ys = np.round(pts[:, 1]).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if inside.sum() < len(pts) * 0.8:
            continue
        xs, ys = xs[inside], ys[inside]

        best_team, best_ratio = None, 0.0
        for team, mask in masks.items():
            ratio = float((mask[ys, xs] > 0).mean())
            if ratio > best_ratio:
                best_team, best_ratio = team, ratio
        if best_team is not None and best_ratio >= MIN_VOTE_RATIO:
            markers.append(CellMarker(cell=cell, team=best_team, vote_ratio=best_ratio))
    return markers


def markers_to_entities(markers: list[CellMarker]) -> list[Entity]:
    """Marqueurs -> entites.

    Le marqueur ne porte que POSITION et EQUIPE. hp/ap/mp restent des valeurs par defaut :
    le pipeline y injecte les PV/PA/PM exacts (OCR) pour le joueur, et les PV approches
    (timeline) pour les autres.

    `is_self` : le marqueur blanc designe le personnage controle. **Un seul** marqueur
    peut l'etre, et c'est le mieux vote.

    Ce point n'est pas cosmetique. La regle precedente marquait `is_self` sur TOUT
    marqueur allie ; or le client reel en produit trois (mesure : « 3 joueur, 2 ennemis,
    ecart timeline 2 »). `CombatState.self_entity()` en choisissait alors un par ordre de
    CASE, pas par confiance -- un faux positif d'indice inferieur suffisait a faire
    calculer deplacements, portees et lignes de vue depuis une position ou le personnage
    n'est pas. Toutes les decisions du tour en decoulent, sans qu'aucune erreur ne soit
    levee.

    Les autres marqueurs allies sont conserves comme allies : en combat de groupe ils
    sont reels, et les effacer priverait le planificateur des cases qu'ils occupent.
    """
    best_ally = max((m for m in markers if m.team is Team.ALLY),
                    key=lambda m: m.vote_ratio, default=None)
    return [
        Entity(
            entity_id=f"{m.team.value}_{m.cell}",
            team=m.team,
            cell=m.cell,
            # PV inconnus a ce stade : le pipeline injecte les valeurs exactes (OCR) pour
            # le joueur. `hp_known=False` empeche le planificateur de raisonner sur des
            # seuils de mise a mort qui n'ont aucun fondement.
            hp=PLACEHOLDER_HP,
            hp_max=PLACEHOLDER_HP,
            ap=0,
            mp=0,
            is_self=(m is best_ally),
            hp_known=False,
        )
        for m in markers
    ]


def reconcile_with_timeline(
    markers: list[CellMarker], timeline: list[TimelineEntry]
) -> list[CellMarker]:
    """Ecarte les marqueurs excedentaires quand le plateau voit plus que la timeline.

    Les deux perceptions sont independantes : la timeline compte des PORTRAITS (ROI fixe,
    fort contraste, taille constante), le plateau echantillonne des couleurs sur du
    terrain. Quand elles divergent, c'est la timeline qui fait autorite sur le NOMBRE --
    le plateau, lui, reste seul a donner les POSITIONS.

    L'asymetrie est volontaire et c'est tout l'interet :

      - **plateau > timeline** : des marqueurs sont des faux positifs. On garde les mieux
        votes. Planifier une attaque sur un fantome gaspille des PA et perd le tour.
      - **plateau < timeline** : un marqueur manque. On ne touche a rien -- inventer une
        position serait pire que l'ignorer, et le plan reste valide, seulement incomplet.

    Ne rien faire si la timeline est vide (hors combat, ou lecture ratee) : elle ne fait
    autorite que lorsqu'elle a elle-meme repondu.

    COMBIEN CETTE TRONCATURE TRAVAILLE. Sur les six captures de combat SANS overlay de la
    racine du depot : 28 marqueurs bruts pour 17 combattants, soit 13 jetes -- 46 %. Le
    detecteur produit donc environ deux fois plus de marqueurs qu'il n'y a de combattants,
    et c'est le tri par vote qui decide lesquels survivent.
    #
    Le meme calcul sur `data/runs/` donne 48 %, mais il ne vaut RIEN : l'overlay du bot y
    dessine les ennemis en (0, 0, 255) BGR -- rouge pur, HSV (0, 255, 255) -- ce qui tombe
    en plein dans `TEAM_BANDS[ENEMY]`. Mesure : 98 % des ennemis detectes sur ce dossier
    portent ce rouge sur leur contour, a 85 % en mediane. Le detecteur y relit les
    conclusions d'un run precedent. Que les deux chiffres coincident est une coincidence
    heureuse, pas une validation -- c'est la mesure sur captures propres qui fait foi.
    """
    if not timeline:
        return markers

    kept: list[CellMarker] = []
    for team in (Team.ALLY, Team.ENEMY):
        seen = [m for m in markers if m.team is team]
        expected = sum(1 for e in timeline if e.team is team)
        if expected and len(seen) > expected:
            seen = sorted(seen, key=lambda m: m.vote_ratio, reverse=True)[:expected]
        kept.extend(seen)

    # Filet de securite sur le TOTAL, independant des equipes.
    #
    # Le filtrage par equipe ci-dessus se desactive entierement quand aucun portrait
    # n'est identifie (`expected` vaut alors 0 partout). Constate en jeu, et le resultat
    # est spectaculaire : timeline illisible -> « equipe ? » sur les trois portraits ->
    # AUCUN filtrage -> **12 ennemis detectes pour 3 combattants**, et un tour entier
    # depense sur une case vide.
    #
    # Or compter les barres ne demande pas de lire leur couleur : le total reste fiable
    # quand les equipes ne le sont plus. On s'en sert donc comme borne, en gardant les
    # mieux votes -- moins precis que le filtrage par equipe, mais toujours mieux que
    # rien, et c'est exactement quand tout le reste a echoue qu'il faut une borne.
    if len(kept) > len(timeline):
        kept = sorted(kept, key=lambda m: m.vote_ratio, reverse=True)[:len(timeline)]
    return sorted(kept, key=lambda m: m.cell)


def detect_entities(frame: NDArray[np.uint8], board: BoardMap) -> list[Entity]:
    """Frame BGR + plateau detecte -> entites. Point d'entree du pipeline."""
    return markers_to_entities(detect_markers_on_board(frame, board))


# Marge, en cases du reseau, exploree autour du plateau detecte pour retrouver un
# combattant que le plateau n'explique pas. Quatre suffisent : les deux monstres manques de
# combat1.png sont a 3 et 6 cases de bordure, et aller plus loin ne ramene que du decor.
OUTSIDE_MARGIN = 4


def enemy_candidates_outside(frame: NDArray[np.uint8],
                             board: BoardMap) -> list[tuple[int, int, float]]:
    """Marqueurs ennemis poses sur le RESEAU mais hors du plateau detecte.

    -> [(x, y, vote), ...] en pixels, tries du vote le plus fort au plus faible.

    A quoi cela sert, et a quoi cela ne sert PAS. Le plateau de combat1.png s'arrete avant
    deux monstres : le diagnostic annonce « 1 joueur, 0 ennemis, ecart timeline 2 » et la
    session refuse de jouer, sans jamais dire OU chercher. Or les deux monstres portent un
    marqueur d'equipe parfaitement ordinaire -- ils sont sur le reseau, pas sur le plateau.
    Verifie a l'oeil en zoomant : un Scarafeuille blanc en (431, 293) et un bleu en
    (569, 317), chacun sur une case cerclee d'orange.

    Cela REFORMULE la limitation connue. Ce n'est pas la detection de plateau qu'il faut
    reprendre : c'est que le decor porte des marqueurs indiscernables. En elargissant le
    reseau de quatre cases on ramene quatre candidats pour deux monstres -- les deux vrais,
    plus un arbre d'automne et le panneau lateral -- et AUCUNE des trois mesures dont
    dispose ce module ne les separe :

        vote de couleur    blanc 0,292 | ARBRE 0,292 | bleu 0,271 | panneau 0,250
        reponse de grille  ARBRE 19,4  | bleu 17,2   | blanc 16,0 | panneau 11,6
        anneau vs tache    blanc coeur 0,000 | bleu 0,000 | ARBRE 0,021 | panneau 0,146

    L'arbre egale ou devance un vrai monstre sur les deux premieres, et ses feuilles
    tombent sur l'anneau sans toucher le coeur -- coincidence geometrique qui le fait
    passer pour un marqueur. Seule la distance au plateau les ordonne (3 et 6 contre 9 et
    11), un ecart trop mince pour en faire un seuil sur une seule capture.

    D'ou une fonction qui SIGNALE sans CHOISIR. Le jour ou un discriminant existera -- une
    correspondance avec les barres de vie de la timeline, un gabarit de sprite -- il aura
    ces candidats en entree. En attendant, mieux vaut « deux candidats hors du plateau, en
    (431, 293) et (569, 317) » que « 0 ennemis » : le premier dit ou regarder.
    
    LA DISTANCE AU PLATEAU EST MORTE AUSSI, et elle etait la derniere piste nommee ici.
    Elle ordonnait les quatre candidats de combat1.png (3 et 4 cases pour les vrais
    monstres, 5 et 7 pour l'arbre et le panneau), ce qui suffisait a esperer un seuil.

    Le contre-exemple manquait, et il tient dans les CINQ AUTRES captures de combat propres
    du depot : leur ecart timeline est nul, donc tout candidat hors plateau y est FAUX par
    construction. Il y en a 51, a des distances de 1 a 6 cases -- soit exactement la plage
    ou se tiennent les deux vrais.

        captures sans ennemi manquant   51 candidats, tous faux, distances 1 a 6
        combat1.png                      4 candidats, 2 vrais, distances 3 a 7

    Aucun seuil sur cette distance ne separe donc quoi que ce soit. Et le chiffre dit aussi
    ce que couterait une regle qui promouvrait les candidats proches : une dizaine
    d'ennemis fantomes par capture, c'est-a-dire « il tape une case morte » a chaque tour.

    L'APPARIEMENT AVEC LES PORTRAITS DE LA TIMELINE A ETE ESSAYE, ET IL NE SEPARE PAS.
    C'etait la plus prometteuse des deux pistes nommees ci-dessous : les portraits ne sont
    pas des icones stylisees mais les SPRITES DE CORPS ENTIER des monstres, meme dessin et
    meme echelle a peu pres que sur le plateau.

    Methode : coeur du portrait (60 x 52 px) en gabarit, `TM_CCOEFF_NORMED` en niveaux de
    gris sur une fenetre de 96 x 98 autour du candidat, meilleur score sur tous les
    portraits ennemis.

    Sur combat1.png seul, cela paraissait marcher :

        VRAI blanc +0,467 | VRAI bleu +0,415 | faux panneau +0,288 | faux arbre +0,175

    Les 51 candidats des cinq autres captures -- faux par construction, leur ecart timeline
    etant nul -- refutent :

        mediane +0,299    p90 +0,485    MAX +0,576
        douze d'entre eux depassent +0,415, c'est-a-dire le plus faible des deux vrais

    Le meilleur faux bat donc les deux vrais. Ce n'est pas un seuil a affiner, c'est un
    recouvrement franc -- l'herbe et les feuillages produisent des textures qui correlent
    aussi bien qu'un insecte avec l'image d'un insecte.

    LA QUESTION ETAIT MAL POSEE, ET LA REFORMULER CHANGE LES CHIFFRES. Tout ce qui precede
    evalue les mesures comme des FILTRES : « ce candidat est-il un monstre ? ». Or la
    timeline donne un BUDGET -- elle dit combien il en manque -- et la vraie question est
    « les k meilleurs sont-ils les k manquants ? ». Les 51 faux ci-dessus n'y participent
    meme pas : leurs captures n'ont aucun ecart, donc la regle ne s'y declencherait jamais.
    C'est l'asymetrie que `reconcile_with_timeline` exploite deja dans l'autre sens.

    POUR MESURER, il fallait des cas ou l'on connait la reponse. On en fabrique en
    TRONQUANT le plateau des captures saines -- on retire une bande de bordure, ce qui fait
    tomber un ennemi connu au dehors, exactement comme la troncature reelle. Six cas ainsi
    obtenus, plus combat1.png qui est le cas reel :

        rang par le VOTE de couleur     0 / 6     (il choisit un arbre avant un monstre)
        rang par le GABARIT de portrait 3 / 6
        rang par la DISTANCE au plateau 2 / 6

    « k sur 6 » compte les cas ou TOUS les manquants sont retrouves. Trois enseignements.

    Le premier : le vote de couleur, seule mesure que cette fonction expose aujourd'hui,
    est le PIRE des trois. Qui construirait ce mecanisme dessus le construirait sur le
    mauvais signal.

    Le deuxieme : le gabarit de portrait, inutilisable comme filtre (cf. plus haut, le
    meilleur faux bat les deux vrais), devient le meilleur des trois comme RANG. Un mauvais
    seuil peut faire un bon classement.

    UN REGLAGE DU GABARIT A ETE ESSAYE, ET LE GAIN N'EST PAS ETABLI. En portant le jeu a
    25 troncatures et en balayant quatre variantes :

        gris, portrait complet      10 / 25        gris, coeur resserre       7 / 25
        couleur, portrait complet   10 / 25        COULEUR + coeur resserre  15 / 25

    L'interaction a du sens -- resserrer retire le fond violet du portrait, ce qui en gris
    ne fait que perdre du contraste et en couleur retire la teinte parasite dominante -- et
    le test des signes apparie donne p = 0,031, cinq gains contre zero perte.

    IL NE FAUT PAS LE CROIRE. Ventile par capture, les CINQ gains viennent tous de
    `bug4.png` ; les quatre autres captures ne bougent pas d'un cas. Les 25 troncatures ne
    sont pas 25 observations independantes, ce sont CINQ captures decoupees de 25 facons,
    et l'effet n'apparait que dans une seule. Le test des signes, qui les traite comme
    independantes, mesure ici la maniere dont on a decoupe, pas la methode.

    A retenir aussi : `bugcarreblanc.png` rend 0 / 9 avec les quatre variantes. Sur cette
    capture-la, aucun classement ne retrouve l'ennemi retire.

    Le troisieme, et il commande : trois sur six ne suffit pas pour CREER des entites que
    le solveur ira frapper. La moitie du temps on poserait un monstre sur un arbre, et
    c'est le defaut « il tape une case morte ». On garde donc le signalement sans choix --
    mais on sait desormais que le plafond n'est pas « aucun signal ne separe », il est
    « le meilleur rang disponible se trompe une fois sur deux ».
    """
    if not len(board):
        return []
    known = {(int(i), int(j)) for i, j in board.cells}
    lo_i = min(i for i, _ in known) - OUTSIDE_MARGIN
    hi_i = max(i for i, _ in known) + OUTSIDE_MARGIN
    lo_j = min(j for _, j in known) - OUTSIDE_MARGIN
    hi_j = max(j for _, j in known) + OUTSIDE_MARGIN
    grown = [(i, j) for i in range(lo_i, hi_i + 1) for j in range(lo_j, hi_j + 1)
             if (i, j) not in known]
    if not grown:
        return []
    lattice = BoardMap(cells=np.array(grown, dtype=np.int64), e_x=board.e_x,
                       e_y=board.e_y, origin=board.origin)
    found = []
    for marker in detect_markers_on_board(frame, lattice):
        if marker.team is not Team.ENEMY:
            continue
        x, y = lattice.center(marker.cell)
        found.append((round(float(x)), round(float(y)), marker.vote_ratio))
    return sorted(found, key=lambda row: -row[2])
