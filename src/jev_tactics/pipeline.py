"""Assemblage capture -> perception -> observation structuree.

Une Observation est ce que l'agent PERCOIT a un instant t, uniquement depuis les pixels :
la lecture d'UI (PV/PA/PM) et les entites presentes sur le plateau.

Deux rythmes distincts, et c'est volontaire :
  - la **geometrie du plateau** (BoardMap) est estimee UNE FOIS par combat (~150 ms) ;
  - l'**observation** est produite a chaque decision (~20 ms) en reutilisant ce plateau.
Re-estimer la grille a chaque frame couterait 8x plus cher pour un resultat identique
tant que la camera ne bouge pas.

Serialisable et horodatee : on doit pouvoir logger une Observation et rejouer une
decision hors-ligne (invariant de replay du projet).
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from jev_tactics.calibration.grid import BoardMap, estimate_grid
from jev_tactics.perception import UiReading, read_ui
from jev_tactics.perception.entities import (
    CURSEUR_REEL,
    detect_markers_on_board,
    hovered_cell,
    markers_to_entities,
    reconcile_with_timeline,
)
from jev_tactics.perception.highlight import (
    looks_like_spell_range,
    movement_centre,
    read_movement_range,
    read_placement_zone,
)
from jev_tactics.perception.timeline import TimelineEntry, read_timeline
from jev_tactics.perception.ui import UI_ZONES, read_hp_gauge
from jev_tactics.rules.movement import grid_distance
from jev_tactics.state import Entity, Team


class Observation(BaseModel):
    timestamp: float
    ui: UiReading
    entities: list[Entity] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    # Cases surlignees par le jeu comme atteignables (vide hors de notre tour).
    reachable: set[int] = Field(default_factory=set)

    def perception_gap(self) -> int:
        """Ecart entre le nombre d'ennemis vus sur le PLATEAU et sur la TIMELINE.

        Les deux perceptions sont independantes, donc chacune controle l'autre : un ecart
        non nul signale un marqueur manque (ou un portrait mal lu) sans qu'il faille une
        verite terrain. Mesure : l'ecart etait de 2 sur dofusscreen4, ce qui a revele des
        marqueurs non detectes.

        UN ECART NUL N'EST PAS UNE VALIDATION, et c'est la limite de ce controle. Deux
        raisons distinctes :

          - `reconcile_with_timeline` TRONQUE le plateau au compte de la timeline. Le sens
            « plateau > timeline » ne peut donc pas survivre jusqu'ici, et son absence
            n'apprend rien. Seul « plateau < timeline » reste informatif.
          - les deux perceptions peuvent se tromper ENSEMBLE. Sur
            `data/runs/20260807-222511.png`, quatre monstres morts laissent une tombe sur
            le plateau et gardent leur portrait dans la timeline : les deux comptent 4, cet
            ecart vaut 0, et il n'y a aucun ennemi vivant. Les deux se corrigent ensemble
            quelques tours plus tard (cf. `tests/test_captures.py`).
        """
        from jev_tactics.state import Team

        on_board = sum(1 for e in self.entities if e.team is Team.ENEMY)
        on_timeline = sum(1 for e in self.timeline if e.team is Team.ENEMY)
        return abs(on_board - on_timeline)

    def enemies_on_timeline(self) -> int:
        """Nombre d'ennemis annonces par la TIMELINE, independamment du plateau.

        Sert a distinguer deux situations que `perception_gap` confond, parce qu'il rend
        une valeur absolue : « il en manque un » et « on n'en voit AUCUN alors qu'il y en
        a ». La seconde n'est pas une imprecision, c'est une perception ratee.
        """
        from jev_tactics.state import Team

        return sum(1 for e in self.timeline if e.team is Team.ENEMY)

    @property
    def in_combat(self) -> bool:
        """La timeline n'existe qu'en combat : c'est un gate gratuit et fiable.

        Verifie : les captures hors combat donnent 0 entree, les captures de combat en
        donnent autant que de combattants. L'orbe PV/PA/PM, lui, est visible hors combat
        et ne discrimine donc pas.
        """
        return bool(self.timeline)


def outside_ui(board: BoardMap) -> BoardMap:
    """Retire du plateau les cases qui tombent sous un panneau d'interface."""
    if not len(board):
        return board
    keep = []
    for index in range(len(board)):
        x, y = board.center(index)
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in UI_ZONES):
            continue
        keep.append(board.cells[index])
    if len(keep) == len(board):
        return board
    return BoardMap(cells=np.array(keep, dtype=np.int64), e_x=board.e_x,
                    e_y=board.e_y, origin=board.origin)


def build_board(image: NDArray[np.uint8]) -> BoardMap | None:
    """Estime la geometrie du plateau depuis une capture de combat (1x par combat).

    -> None si aucune grille exploitable (typiquement : on n'est pas en combat).
    """
    estimate = estimate_grid(image)
    if estimate is None or not len(estimate.board):
        return None
    board = outside_ui(BoardMap.from_estimate(estimate))
    return board if len(board) else None


def observe(
    image: NDArray[np.uint8],
    board: BoardMap | None = None,
    timestamp: float | None = None,
    *,
    cursor: object = CURSEUR_REEL,
) -> Observation:
    """Frame BGR -> Observation. Perception seule (aucune decision ici).

    `board` : geometrie du plateau (cf. `build_board`). Sans elle, `entities` reste vide
    -- la lecture d'UI, elle, ne depend pas de la grille.

    `cursor` : POSITION DE LA SOURIS, ET C'EST UNE ENTREE DE LA PERCEPTION. Par defaut
    elle est lue sur le systeme, ce qui est correct quand `image` vient de l'ecran. Sur une
    image STOCKEE il faut passer `cursor=None` : sinon le resultat depend de l'endroit ou
    se trouve la souris de l'utilisateur au moment du rejeu, sans rapport avec l'image.
    Voir `hovered_cell`, qui detaille les semaines d'echecs intermittents que ce defaut a
    values au projet.
    """
    timeline = read_timeline(image)
    # La timeline arbitre le NOMBRE de combattants avant qu'on en fasse des entites : le
    # plateau seul produit des faux positifs (mesure sur le client reel : 5 marqueurs
    # pour 3 combattants). Filtrer ici plutot qu'en aval evite que le desaccord se
    # propage jusqu'au planificateur, qui viserait des cases vides.
    marquees: set[int] = set()
    if board is not None:
        markers = detect_markers_on_board(image, board)
        # AVANT reconciliation : la reconciliation jette pres de la moitie des marqueurs
        # (71 sur 148 mesures) et se trompe d'equipe, mais « il y a quelque chose ici »
        # reste vrai pour ceux qu'elle ecarte. C'est tout ce qu'on demande a ce signal --
        # departager deux cases que la zone de deplacement propose a egalite.
        marquees = {m.cell for m in markers}
        # Le curseur du bot cree un contour blanc identique au marqueur du joueur :
        # retirer la case survolee evite qu'il se prenne lui-meme pour son personnage.
        hovered = hovered_cell(board, cursor)
        if hovered is not None:
            markers = [m for m in markers if m.cell != hovered]
        entities = markers_to_entities(reconcile_with_timeline(markers, timeline))
    else:
        entities = []
    ui = read_ui(image)

    # Les PV du joueur sont connus EXACTEMENT par l'OCR : on les injecte dans l'entite
    # marquee `is_self` plutot que d'utiliser le ratio approximatif de la timeline
    # (mesure : la hauteur de barre donne le bon ratio a ~0.03 pres, mais s'ecarte
    # jusqu'a 0.10 sur certaines captures).
    # LES PA ET LES PM NE DEPENDENT PAS DU MAXIMUM DE PV, et ils en dependaient. Toute
    # cette injection etait gardee par `ui.pv_max`, si bien que sur l'ATH ou le maximum
    # n'est pas affiche -- celui du personnage reellement joue -- le personnage gardait
    # ses valeurs par defaut. Mesure sur `20260819-115025`, un vrai combat : l'OCR lit
    # « pa=6 pm=3 » et l'etat construit portait « 0 PA, 0 PM ». Le solveur ne pouvait donc
    # ni bouger ni lancer un sort, et rendait « passer le tour » sans que rien n'echoue.
    # C'est la meme faute que pour `--min-hp` : une donnee absente en condamnait d'autres,
    # parfaitement lisibles.
    for entity in entities:
        if not entity.is_self:
            continue
        if ui.pv is not None and ui.pv_max:
            entity.hp, entity.hp_max = ui.pv, ui.pv_max
            entity.hp_known = True   # mesure exacte, contrairement aux ennemis
        elif ui.pv is not None:
            # Maximum non affiche : la JAUGE donne la proportion, donc le maximum s'en
            # deduit. A 1,6 point pres sur la part (cf. `read_hp_gauge`), soit ~2 % sur le
            # maximum -- largement assez pour `danger_factor`, qui compare des ordres de
            # grandeur. Sans ca `hp_known` restait faux et le solveur jouait a l'aveugle
            # sur sa propre survie.
            part = read_hp_gauge(image)
            if part:
                entity.hp = ui.pv
                entity.hp_max = max(ui.pv, round(ui.pv / part))
                entity.hp_known = True
        if ui.pa is not None:
            entity.ap = ui.pa
        if ui.pm is not None:
            entity.mp = ui.pm

    # Une SEULE lecture de la zone de deplacement pour les deux fonctions qui la
    # consomment. Elles l'appelaient chacune de leur cote, avec la meme image et le meme
    # plateau : 222 ms rendus deux fois a l'identique.
    highlighted = read_movement_range(image, board) if board is not None else set()
    _relocate_from_movement(image, board, entities, ui, highlighted, marquees)

    return Observation(
        timestamp=time.time() if timestamp is None else timestamp,
        ui=ui,
        entities=entities,
        timeline=timeline,
        reachable=_readable_movement(image, board, entities, highlighted),
    )


def _relocate_from_movement(image, board, entities, ui, highlighted=None,
                            marquees=None) -> None:
    """Remet le joueur ou le JEU le place, quand les marqueurs se sont trompes.

    Une position de joueur fausse ne se voit nulle part : les numeros de case restent
    plausibles et le plan reste credible, mais tout y est calcule autour d'un personnage
    qui n'est pas la. Mesure sur combat1920.png -- deux cases votent ALLIE, et c'est la
    FAUSSE qui vote le plus fort :

        case 142 (le vrai personnage)   ALLY 0,271
        case 119 (une ombre de trou)    ALLY 0,313   <- retenue, car la plus forte

    Le vote de couleur n'a aucun moyen de trancher. La zone de deplacement, elle, en a un :
    le jeu la dessine AUTOUR du personnage. C'est le meme principe que `reachable_hint` --
    adopter la reponse du jeu plutot que la notre -- applique cette fois a la position.

    DEUX CONDITIONS, et chacune ecarte un cas mesure :

      - la position detectee doit CONTREDIRE la zone (rayon > PM). Sur combat1.png et
        tacle.png elle est coherente : on ne touche a rien, meme si le centre calcule
        differe. Corriger une detection qui marche est le seul vrai risque ici ;
      - le centre doit etre UNIQUE (cf. `movement_centre`). Une zone tronquee peut avoir
        plusieurs cases aussi centrales.

    Resultat sur les six captures de combat : deux corrections de dix cases
    (combat1920 119->142, bug4 138->70, verifiees a l'oeil sur les captures annotees),
    une confirmation (bugcarreblanc), trois abstentions.

    La zone est lue ICI SANS le garde-fou des portees de sort : une zone recouverte par une
    portee est tronquee, mais elle reste CENTREE sur le personnage -- ce qui est tout ce
    qu'on lui demande. bug4.png et bugcarreblanc.png ont un sort selectionne et donnent
    quand meme le bon centre.
    """
    mp = ui.pm
    if board is None or not mp:
        return
    # `highlighted` est fourni par `observe`, qui le calcule UNE fois pour les deux
    # fonctions qui en ont besoin. Mesure : 222 ms l'appel, soit un cinquieme du cout
    # d'une frame gaspille a refaire le meme travail sur la meme image. Le parametre reste
    # optionnel pour que cette fonction s'appelle encore seule.
    if highlighted is None:
        highlighted = read_movement_range(image, board)
    if not highlighted:
        return
    me = next((e for e in entities if e.is_self), None)
    if me is None:
        # AUCUN marqueur de joueur. Mesure sur 24 captures d'une vraie session : c'est le
        # cas DIX FOIS, sur des plateaux pourtant sains de 175 a 242 cases -- et la session
        # refuse alors de jouer, faute de savoir ou elle est. Le contour de la case du
        # personnage y est BLEU la ou le detecteur attend du blanc peu sature.
        #
        # La zone de deplacement, elle, est bien lue. Verifie a l'oeil sur l'une d'elles :
        # la case deduite porte le libelle « MOI 160 » -- qui vient de l'overlay du
        # BOT, incruste dans ces captures, et non du jeu : la verification tient
        # (l'overlay place ce libelle sur la case qu'il a lue) mais elle n'est pas
        # independante, et l'ecrire comme si le jeu l'affichait le laissait croire. Creer l'entite manquante y rend sept captures sur neuf jouables.
        centre = movement_centre(board, highlighted, mp, marquees)
        if centre is None or any(e.cell == centre for e in entities):
            return
        # Les PV viennent de l'OCR, exactement comme pour une entite detectee : sans eux
        # `danger_factor` resterait neutre et le solveur jouerait a l'aveugle sur sa propre
        # survie -- le terme le plus lourd du score apres la mise a mort.
        entities.append(Entity(
            entity_id="me", team=Team.ALLY, cell=centre,
            hp=ui.pv or 1, hp_max=ui.pv_max or 1, hp_known=ui.pv is not None,
            ap=ui.pa or 6, mp=mp, is_self=True, cell_confirmed=True))
        return
    if max(grid_distance(board, me.cell, lit) for lit in highlighted) <= mp:
        # Coherent : la detection tient, on n'y touche pas -- mais on le DIT. C'est une
        # confirmation par un signal independant des marqueurs, et sans ce drapeau elle
        # etait indistinguable d'une case jamais verifiee.
        me.cell_confirmed = True
        return                      # coherent : la detection tient, on n'y touche pas
    centre = movement_centre(board, highlighted, mp, marquees)
    if centre is None:
        return
    # LE CENTRE QUI TOMBE SUR LA CASE DU JOUEUR EST UNE CONFIRMATION, PAS UNE COLLISION.
    # La garde ci-dessous refuse d'empiler deux combattants -- mais `entities` contient le
    # joueur lui-meme, donc elle attrapait aussi le cas ou la zone CONFIRME la position.
    # Invisible tant que rien n'enregistrait la confirmation ; mesure une fois le drapeau
    # pose : 222511 et 222604 ont une couverture de 97 % et un centre egal a leur case, et
    # se declaraient pourtant non confirmees.
    if centre == me.cell:
        me.cell_confirmed = True
        return
    # Une case deja occupee par une AUTRE entite serait une contradiction de plus, pas une
    # correction : on s'abstient plutot que d'empiler deux combattants au meme endroit.
    if any(e.cell == centre for e in entities):
        return
    me.cell = centre
    me.cell_confirmed = True


def _readable_movement(image, board, entities, highlighted=None) -> set[int]:
    """Zone de deplacement du jeu -- SAUF quand un sort selectionne la masque.

    Deuxieme consequence du meme fait que le garde-fou de placement : un sort selectionne
    fait peindre sa portee par-dessus le plateau. Les cases de cette portee ne sont donc
    plus vertes, qu'elles soient atteignables ou non -- leur statut de deplacement est
    CACHE, pas negatif.

    Ce que cela donnait, mesure sur les captures ou un sort est selectionne :

        bugcarreblanc.png   4 PM -> 3 cases vertes, toutes a distance 1
        tacle.png           4 PM -> 2 cases vertes, toutes a distance 1
        combat1.png (rien de selectionne)   3 PM -> 10 cases, distance max 3

    Et cette lecture tronquee faisait AUTORITE : `_plausible_hint` ne demande qu'une seule
    case voisine surlignee, or il y en avait trois. `blocked_from_highlight` declarait
    ensuite infranchissable tout le reste du plateau. Le bot avait 4 PM et une case pour
    les depenser, donc il ne pouvait plus s'approcher de personne.

    Rendre l'ensemble VIDE n'est pas perdre l'information : c'est dire « je ne sais pas »,
    et la chaine sait deja quoi en faire -- `assemble` met alors `reachable_hint` a None et
    le solveur retombe sur son BFS, qui ne se trompe que sur les obstacles.
    """
    if board is None:
        return set()
    if highlighted is None:
        highlighted = read_movement_range(image, board)
    me = next((e for e in entities if e.is_self), None)
    if me is None:
        return highlighted
    spell_range = read_placement_zone(image, board)
    if looks_like_spell_range(board, spell_range, me.cell):
        return set()

    # Une case plus loin que les PM restants est INATTEIGNABLE. C'est une regle du jeu, pas
    # un seuil : rien a mesurer, rien a regler. Le vert du surlignage attrape pourtant de
    # l'herbe -- hors combat, ou le jeu n'affiche AUCUNE zone, le lecteur en trouve 12 et 18
    # selon la capture -- et en combat le plateau la recouvre presque partout. Presque :
    # bug.png en garde quatre, dont une a VINGT-SEPT cases du personnage.
    #
    # Ce filtre vient APRES `_relocate_from_movement`, et l'ordre compte. Le recentrage a
    # besoin du surlignage BRUT : c'est en constatant qu'aucune case detectee n'explique la
    # zone qu'il repere une position fausse. Filtrer avant lui reviendrait a effacer la
    # preuve pour faire disparaitre le symptome.
    if me.mp:
        return {cell for cell in highlighted
                if grid_distance(board, me.cell, cell) <= me.mp}
    return highlighted
