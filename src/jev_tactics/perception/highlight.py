"""Lecture de la zone de deplacement surlignee par le jeu.

Perception active, meme principe que la touche Y pour les ressources : plutot que de
recalculer ce que le jeu affiche deja, on le lit. Pendant notre tour, Dofus teinte en
VERT les cases atteignables avec les PM restants -- c'est-a-dire le resultat exact de son
propre pathfinding, obstacles compris.

Ce que la mesure a revele en comparant ce surlignage a notre BFS (qui supposait le
plateau vide) :
    dofusscreen3  22 vertes / 23 BFS -> 1 case que le BFS croyait libre : un OBSTACLE
    dofusscreen2  15 vertes /  8 BFS -> 7 cases vertes hors BFS : des cases de PLATEAU
                                        que la detection avait manquees
Les deux familles d'erreur sont donc detectees -- et corrigees d'un coup, puisqu'on
adopte la reponse du jeu plutot que la notre.

Limite assumee : le surlignage n'existe QUE pendant notre tour. Hors de la, on retombe
sur le BFS.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.rules.movement import grid_distance

# Vert du surlignage de deplacement (teinte large : la couleur varie avec le terrain
# sous-jacent, qui transparait).
GREEN_BAND = ((35, 90, 60), (85, 255, 255))
# Fraction de l'interieur de la case a echantillonner : on reste loin des bords pour ne
# pas mordre sur les cases voisines.
INNER_SCALE = 0.35
# Part de pixels verts au-dela de laquelle la case est consideree surlignee.
MIN_GREEN_RATIO = 0.5
# Rouge sombre des zones de placement (avant le debut du combat). Deux sous-bandes :
# la teinte rouge chevauche 0.
PLACEMENT_BANDS = (((0, 90, 40), (10, 255, 180)), ((170, 90, 40), (179, 255, 180)))
MIN_RED_RATIO = 0.5
# Part des cases du plateau teintees au-dela de laquelle on conclut au placement.
# Prudent : mieux vaut jouer un tour de placement (on perd un positionnement) que
# refuser de jouer un vrai tour (on perd le combat).
MIN_PLACEMENT_SHARE = 0.15


def _cell_polygon(board: BoardMap, index: int) -> NDArray[np.int32]:
    centre = board.center(index)
    return np.array([
        centre + (board.e_x - board.e_y) * INNER_SCALE,
        centre + (board.e_x + board.e_y) * INNER_SCALE,
        centre - (board.e_x - board.e_y) * INNER_SCALE,
        centre - (board.e_x + board.e_y) * INNER_SCALE,
    ], dtype=np.int32)


def _lit_cells(band: NDArray[np.uint8], board: BoardMap, minimum: float) -> set[int]:
    """Cases dont le losange interieur est couvert a `minimum` par le masque `band`.

    Le losange fait environ 90x46 px ; allouer un masque PLEIN ECRAN par case revenait a
    remplir 2 Mo pour en lire 4 Ko, deux cent fois par lecture et deux lectures par frame.
    On travaille donc dans le rectangle englobant, le polygone etant translate d'autant.

    Mesure sur une capture de session (206 cases) :
        plein ecran            219 ms par lecture
        rectangle englobant     11 ms par lecture

    Aucun changement de resultat : le losange decoupe est le meme, seul le support change.
    Verifie sur les 30 captures, ensembles compares un a un.
    """
    hauteur, largeur = band.shape[:2]
    allumees: set[int] = set()
    for index in range(len(board)):
        polygone = _cell_polygon(board, index)
        x0, y0 = polygone.min(axis=0)
        x1, y1 = polygone.max(axis=0)
        # Une case peut deborder de l'ecran : on la ramene dedans plutot que de la
        # rejeter, exactement comme le decoupage implicite du masque plein ecran le
        # faisait.
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1) + 1, largeur), min(int(y1) + 1, hauteur)
        if x1 <= x0 or y1 <= y0:
            continue
        petit = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(petit, [polygone - (x0, y0)], 255)
        inside = band[y0:y1, x0:x1][petit > 0]
        if len(inside) and float((inside > 0).mean()) >= minimum:
            allumees.add(index)
    return allumees


def read_movement_range(frame: NDArray[np.uint8], board: BoardMap) -> set[int]:
    """Cases surlignees en vert -> ensemble d'index de `board`.

    Ensemble vide si rien n'est surligne (tour adverse, ou aucun PM restant).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(GREEN_BAND[0], np.uint8),
                        np.array(GREEN_BAND[1], np.uint8))

    return _lit_cells(green, board, MIN_GREEN_RATIO)


def read_placement_zone(frame: NDArray[np.uint8], board: BoardMap) -> set[int]:
    """Cases teintees en ROUGE de placement -> ensemble d'index de `board`.

    Avant que le combat commence, Dofus colore les zones de placement des deux equipes et
    laisse quelques secondes pour se positionner. La timeline existe deja, l'UI affiche
    PV/PA/PM, et **le bot prenait donc cet ecran pour un tour ordinaire** : il planifiait,
    puis cliquait une case a plusieurs cases de la -- ce qui, vu du joueur, ressemble a un
    bot qui « vise trop loin ». Constate en jeu, avec le monstre colle a gauche et le clic
    parti en bas a droite.

    Mesure qui separe les deux situations (part de rouge de placement dans l'image) :
        placement       18,7 %
        tour normal      1,0 % et 2,5 %
    L'ecart est d'un ordre de grandeur, mais on compte ici en CASES DE PLATEAU plutot
    qu'en pixels d'ecran : une fraction d'ecran dependrait du zoom et de la taille de la
    carte, une fraction de cases non.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array(PLACEMENT_BANDS[0][0], np.uint8),
                      np.array(PLACEMENT_BANDS[0][1], np.uint8))
    red |= cv2.inRange(hsv, np.array(PLACEMENT_BANDS[1][0], np.uint8),
                       np.array(PLACEMENT_BANDS[1][1], np.uint8))

    return _lit_cells(red, board, MIN_RED_RATIO)


def looks_like_spell_range(board: BoardMap, zone: set[int], centre: int) -> bool:
    """La zone rouge est-elle la PORTEE D'UN SORT plutot qu'une zone de placement ?

    Ce test existe parce que le critere precedent -- « assez de rouge, donc placement » --
    a ete pris en flagrant delit sur une capture reelle. bugcarreblanc.png affiche
    « FIN DE TOUR » en toutes lettres : c'est un tour ordinaire. Il donnait pourtant 17,9 %
    de rouge, au-dessus du seuil de 15 %.

    Ce que le rouge etait vraiment, mesure en distances de grille au joueur :

        bugcarreblanc.png    rouge aux distances {2: 6, 3: 8, 4: 10}   vert {1: 3}
        tacle.png            rouge aux distances {2: 5, 3: 6, 4: 6}    vert {1: 2}

    Un anneau contigu de 2 a 4, centre sur le joueur, avec le deplacement en vert a
    distance 1 : c'est une portee de sort de portee minimale 2 et maximale 4. Le jeu
    l'affiche des qu'un sort est SELECTIONNE -- c'est-a-dire au moment precis ou le bot
    s'apprete a frapper.

    La consequence bouclait la boucle : le sort selectionne declenchait la fausse detection
    de placement, la session repondait PLACEMENT, et le runner envoyait la touche « pret »
    -- qui est celle de FIN DE TOUR. Le bot terminait donc son tour parce qu'il avait
    choisi un sort. « Entoure d'ennemis, il n'a pas frappe et a rate le tour. »

    LIMITE ASSUMEE : ce test n'est valide que du cote NEGATIF. Aucune capture de placement
    n'existe dans le depot -- celle qui servait de reference n'en etait pas une -- donc on
    ne peut pas verifier qu'une vraie zone de placement y echappe. Les couts sont
    heureusement asymetriques : se tromper ici fait jouer pendant le placement (le
    personnage se deplace dans sa zone, sans dommage), alors que l'erreur inverse, elle,
    est constatee et coute un tour entier.
    """
    if not zone:
        return False
    distances = sorted({grid_distance(board, centre, cell) for cell in zone})
    # Une portee de sort ne contient jamais la case du lanceur, et une zone de placement
    # contient celle du personnage qu'on y a pose.
    if distances[0] < 1:
        return False
    # Contigue : un anneau plein entre portee mini et maxi. Une zone de placement est un
    # bloc a l'ecart, dont les distances au joueur ne forment pas une bande reguliere.
    return distances == list(range(distances[0], distances[-1] + 1))


# Part des cases surlignees qui doivent tomber dans le rayon des PM pour qu'on accepte d'y
# lire une zone de deplacement.
#
# SON EXIGENCE REELLE SAUTE AVEC LA TAILLE, et c'est ce qui la rend utile :
#
#     cases lues        2   3   4   5   6   8  10  20  30
#     exigees           2   3   4   4   5   7   8  16  24
#     parasites tolerees 0   0   0   1   1   1   2   4   6
#
# En dessous de cinq cases il faut donc 100 %. Cela ressemble a une garde plus etroite que
# sa promesse -- le defaut que ce projet corrige d'ordinaire -- et j'ai mesure avant d'y
# toucher. Sur 220355, 220420 et 221831, la lecture rend trois cases groupees plus une
# egaree ; tolerer un seul parasite y donne un centre unique, et DEPLACERAIT le personnage
# de 9, 9 et 18 cases. Ces trois cases sont a une ou deux cases d'un ENNEMI et a huit a
# dix-huit du joueur : c'est une portee de sort, pas une zone de deplacement.
#
# La strictesse aux petites tailles n'est donc pas un arrondi malheureux, c'est ce qui
# empeche de relire une zone hostile comme la sienne. Tenu par un test dans les deux sens :
# assouplir et resserrer font rougir.
MIN_ZONE_COVERAGE = 0.8


def movement_centre(board: BoardMap, highlighted: set[int],
                    mp: int, marked: set[int] | None = None) -> int | None:
    """Case sur laquelle la zone de deplacement est CENTREE, ou None si ce n'en est pas une.

    Le jeu dessine cette zone autour du personnage, dans un rayon egal aux PM restants.
    C'est donc un LOCALISATEUR du joueur entierement independant des marqueurs de couleur
    -- et un localisateur independant est ce qui manquait, parce qu'une position fausse ne
    se voit nulle part : les numeros de case restent plausibles et le plan reste credible,
    mais tout y est calcule autour d'un personnage qui n'est pas la.

    Mesure sur les six captures de combat, case detectee contre case impliquee :

        tacle.png           26 -> 26     ecart 0     rayon 1 pour 4 PM
        bugcarreblanc.png   83 -> 83     ecart 0     rayon 1 pour 4 PM
        combat1.png         36           rayon 3 pour 3 PM : coherent, rien a dire
        combat1920.png     119 -> 142    ecart 10    rayon 14 detecte, 4 implique
        bug4.png           138 ->  70    ecart 10    rayon 10 detecte, 1 implique
        bug.png            161           aucun centre a moins de 4 : pas une zone

    Les deux premieres confirment la methode, la derniere montre qu'elle sait se taire.

    -> None quand aucune case ne rassemble la zone dans le rayon des PM : le vert lu n'est
    alors pas une zone de deplacement (decor, ou lecture partielle), et on ne peut RIEN en
    conclure sur la position du joueur.
    """
    if not highlighted or mp <= 0:
        return None
    # COUVERTURE, et non plus rayon maximal. Le critere precedent -- la case qui minimise
    # la distance a la plus lointaine surlignee -- est ruine par UNE SEULE aberration.
    # Constate sur 24 captures de session : la zone reelle est correctement lue, mais trois
    # cases d'herbe a l'autre bout de l'ecran font passer le rayon de 4 a 9, et la mesure
    # renonce. Compter combien de surlignees tombent dans le rayon des PM ignore les
    # aberrations au lieu de s'y soumettre.
    scores = []
    for cell in range(len(board)):
        near = [grid_distance(board, cell, lit) for lit in highlighted]
        inside = [d for d in near if d <= mp]
        # Deux criteres, et chacun corrige le defaut de l'autre. La COUVERTURE ignore les
        # aberrations, mais elle egalise des que la zone est plus petite que les PM -- toutes
        # les cases alentour couvrent alors tout. Le RESSERREMENT departage ces cas, mais
        # seul il se laisse ruiner par une seule case egaree.
        scores.append((len(inside), -max(inside, default=0)))
    best = max(scores)
    if best[0] < MIN_ZONE_COVERAGE * len(highlighted):
        return None
    if scores.count(best) == 1:
        return scores.index(best)

    # EGALITE : la zone designe plusieurs cases aussi bien, et jusqu'ici on renoncait. Or
    # une egalite n'est pas une ignorance -- c'est une LISTE COURTE, et il existe un
    # deuxieme signal capable de choisir dedans sans etre capable de choisir tout seul.
    #
    # Le marqueur de couleur du joueur est trop faible pour localiser : mesure sur les huit
    # captures ou les deux signaux repondent, il donne la bonne case 3 fois sur 8, avec des
    # ecarts allant jusqu'a 16 cases, et son vote ne dit RIEN de sa justesse (0,479 se
    # trompe de 4 cases, 0,250 tombe juste). Mais departager deux cases proposees par la
    # geometrie est une question mille fois plus facile que d'en designer une parmi 200.
    #
    # L'INDEPENDANCE EST PRESERVEE LA OU ELLE COMPTE : l'ENSEMBLE des candidats vient de la
    # seule zone de deplacement, sans aucun apport des marqueurs. Ceux-ci n'interviennent
    # que si l'ensemble compte deja plusieurs cases, et seulement si UNE SEULE d'entre
    # elles porte un marqueur. Zero ou plusieurs, on renonce comme avant -- l'ambiguite
    # reelle reste une ambiguite.
    #
    # Mesure sur les trois egalites de `data/runs/` :
    #     221351   ex aequo 75 et 84,        marqueur sur 75 seul   -> 75
    #     220519   ex aequo 38 et 54,        marqueur sur 54 seul   -> 54
    #     222649   ex aequo 85, 86 et 94,    marqueurs sur 86 ET 94 -> None, a raison
    exaequo = [cell for cell, score in enumerate(scores) if score == best]
    porteuses = [cell for cell in exaequo if cell in (marked or set())]
    return porteuses[0] if len(porteuses) == 1 else None


def is_placement_phase(frame: NDArray[np.uint8], board: BoardMap,
                       player_cell: int | None = None) -> bool:
    """Vrai si l'ecran est celui du PLACEMENT et non d'un tour de jeu.

    Le seuil porte sur la PART DES CASES du plateau, pas sur des pixels : la zone de
    placement en couvre une fraction notable, alors qu'un tour ordinaire n'en teinte
    aucune.

    `player_cell` est ce qui rend le verdict fiable : sans lui le critere confond une
    portee de sort avec une zone de placement (cf. `looks_like_spell_range`). Il reste
    facultatif -- joueur non localise, on retombe sur le critere seul, et l'ignorance ne
    bloque pas.
    """
    if not len(board):
        return False
    zone = read_placement_zone(frame, board)
    if len(zone) / len(board) < MIN_PLACEMENT_SHARE:
        return False
    return not (player_cell is not None
                and looks_like_spell_range(board, zone, player_cell))


def blocked_from_highlight(board: BoardMap, highlighted: set[int], start: int) -> set[int]:
    """Cases a considerer comme infranchissables, deduites du surlignage.

    Tout ce qui n'est pas surligne est hors d'atteinte -- que ce soit un obstacle, un
    occupant, ou simplement trop loin. Un BFS lance avec ce blocage retrouve les memes
    cases que le jeu, ET leur cout exact en PM (que le surlignage seul ne donne pas).
    """
    return {cell for cell in range(len(board))
            if cell not in highlighted and cell != start}
