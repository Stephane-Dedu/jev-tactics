"""Zone de deplacement surlignee : lecture, et primaute sur notre propre pathfinding.

C'est de la perception ACTIVE : le jeu a deja calcule la reponse (obstacles compris),
on la lit au lieu de la recalculer."""

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception.highlight import (
    INNER_SCALE,
    _cell_polygon,
    blocked_from_highlight,
    is_placement_phase,
    looks_like_spell_range,
    movement_centre,
    read_movement_range,
    read_placement_zone,
)
from jev_tactics.pipeline import build_board, observe
from jev_tactics.planner import Move, TurnState, legal_actions
from jev_tactics.planner.legal import movement_options
from jev_tactics.rules.movement import grid_distance
from jev_tactics.state import CombatState, Entity, Team

BOARD = BoardMap(
    cells=np.array([(i, j) for j in range(5) for i in range(5)], dtype=np.int64),
    e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
    origin=np.array([600.0, 400.0]),
)
CENTRE = BOARD.index_at(2, 2)
GREEN_BGR = (60, 200, 60)


def _frame():
    return np.zeros((800, 1200, 3), dtype=np.uint8)


def _paint(frame, cells):
    for index in cells:
        centre = BOARD.center(index)
        polygon = np.array([
            centre + (BOARD.e_x - BOARD.e_y) * INNER_SCALE,
            centre + (BOARD.e_x + BOARD.e_y) * INNER_SCALE,
            centre - (BOARD.e_x - BOARD.e_y) * INNER_SCALE,
            centre - (BOARD.e_x + BOARD.e_y) * INNER_SCALE,
        ], dtype=np.int32)
        cv2.fillPoly(frame, [polygon], GREEN_BGR)


def _state(reachable_hint=None, mp=3, obstacles=frozenset()):
    return CombatState(
        turn=1,
        entities=[Entity(entity_id="me", team=Team.ALLY, cell=CENTRE, hp=100,
                         hp_max=100, ap=6, mp=mp, is_self=True)],
        obstacles=set(obstacles),
        reachable_hint=reachable_hint,
    )


class TestReadHighlight:
    def test_reads_painted_cells(self):
        expected = {BOARD.index_at(2, 1), BOARD.index_at(1, 2), BOARD.index_at(3, 2)}
        frame = _frame()
        _paint(frame, expected)
        assert read_movement_range(frame, BOARD) == expected

    def test_empty_when_nothing_highlighted(self):
        assert read_movement_range(_frame(), BOARD) == set()

    def test_ignores_non_green_fill(self):
        frame = _frame()
        centre = BOARD.center(CENTRE)
        cv2.circle(frame, (int(centre[0]), int(centre[1])), 20, (200, 60, 60), -1)
        assert read_movement_range(frame, BOARD) == set()

    def test_blocked_is_the_complement(self):
        highlighted = {BOARD.index_at(2, 1), BOARD.index_at(1, 2)}
        blocked = blocked_from_highlight(BOARD, highlighted, CENTRE)
        assert not (blocked & highlighted)
        assert CENTRE not in blocked           # on est deja dessus
        assert BOARD.index_at(0, 0) in blocked


class TestHintDrivesMovement:
    def test_hint_restricts_destinations(self):
        """Le jeu dit 2 cases atteignables : le planificateur n'en propose pas d'autres."""
        allowed = {BOARD.index_at(2, 1), BOARD.index_at(1, 2)}
        options = movement_options(BOARD, _state(reachable_hint=allowed),
                                   TurnState(cell=CENTRE, ap=6, mp=3))
        assert set(options) - {CENTRE} == allowed

    def test_hint_reveals_obstacles_bfs_would_miss(self):
        """Sans indice, le BFS croit le plateau vide et propose plus que le jeu."""
        allowed = {BOARD.index_at(2, 1)}
        free = movement_options(BOARD, _state(), TurnState(cell=CENTRE, ap=6, mp=3))
        hinted = movement_options(BOARD, _state(reachable_hint=allowed),
                                  TurnState(cell=CENTRE, ap=6, mp=3))
        assert len(hinted) < len(free)

    def test_costs_still_computed_within_hint(self):
        """Le surlignage donne les cases mais pas leur cout : le BFS le retrouve."""
        near, far = BOARD.index_at(2, 1), BOARD.index_at(2, 0)
        options = movement_options(BOARD, _state(reachable_hint={near, far}),
                                   TurnState(cell=CENTRE, ap=6, mp=3))
        assert options[near] == 1 and options[far] == 2

    def test_falls_back_to_bfs_without_hint(self):
        options = movement_options(BOARD, _state(), TurnState(cell=CENTRE, ap=6, mp=1))
        assert len(options) == 5          # la case + ses 4 voisins

    def test_hint_ignored_when_we_are_OUTSIDE_the_zone(self):
        """Ce test disait « le surlignage decrit la position de DEPART ; ailleurs il ne
        s'applique plus ». C'etait la mauvaise raison, et elle coutait cher (voir
        TestTheZoneSurvivesAMove). La vraie regle est plus etroite : on ne suit la zone
        que si l'on s'y trouve.

        L'etat ci-dessous est incoherent -- le personnage est en CENTRE, le tour en (0,0),
        qui n'est pas dans la zone. La suivre declarerait infranchissable la case sous nos
        pieds ; le repli sur le BFS reste donc justifie ICI."""
        allowed = {BOARD.index_at(2, 1)}
        moved = TurnState(cell=BOARD.index_at(0, 0), ap=6, mp=2)
        options = movement_options(BOARD, _state(reachable_hint=allowed), moved)
        assert BOARD.index_at(1, 0) in options   # voisin du nouveau point, hors indice

    def test_legal_actions_respect_the_hint(self):
        allowed = {BOARD.index_at(2, 1)}
        actions = legal_actions(BOARD, _state(reachable_hint=allowed), [],
                                TurnState(cell=CENTRE, ap=6, mp=3))
        assert {a.cell for a in actions if isinstance(a, Move)} == allowed


class TestPlacementPhase:
    """Detection de la phase de placement.

    Defaut constate en jeu : avant le debut du combat, Dofus colore les zones de
    placement et laisse quelques secondes pour se positionner. La timeline existe deja et
    l'UI affiche PV/PA/PM, donc le bot prenait cet ecran pour un tour ordinaire -- il
    planifiait puis cliquait une case a plusieurs cases de la. Vu du joueur : « le monstre
    etait colle a gauche et le bot est parti en bas a droite »."""

    def _paint(self, frame, cells, colour):
        for index in cells:
            cv2.fillPoly(frame, [_cell_polygon(BOARD, index)], colour)

    def test_red_zone_is_placement(self):
        frame = _frame()
        self._paint(frame, range(len(BOARD)), (40, 40, 130))   # rouge sombre
        assert is_placement_phase(frame, BOARD)

    def test_a_normal_turn_is_not_placement(self):
        """Garde-fou contre l'inverse : refuser de jouer un vrai tour ferait perdre le
        combat, alors que jouer un tour de placement ne coute qu'un positionnement."""
        frame = _frame()
        self._paint(frame, range(len(BOARD)), (60, 180, 60))   # vert
        assert not is_placement_phase(frame, BOARD)

    def test_a_few_red_cells_are_not_placement(self):
        """Du decor rouge, ou un sort qui teinte quelques cases, ne doit pas suffire."""
        frame = _frame()
        self._paint(frame, [0], (40, 40, 130))
        assert not is_placement_phase(frame, BOARD)

    def test_empty_board_is_not_placement(self):
        board = BoardMap(cells=np.empty((0, 2), dtype=np.int64),
                         e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
                         origin=np.array([300.0, 300.0]))
        assert not is_placement_phase(_frame(), board)

    def test_the_zone_lists_the_tinted_cells(self):
        frame = _frame()
        self._paint(frame, [0, 1, 2], (40, 40, 130))
        assert read_placement_zone(frame, BOARD) == {0, 1, 2}


class TestImplausibleHighlight:
    """Surlignage partiel : ignorer plutot que subir.

    Le surlignage fait autorite quand il est lu correctement -- c'est le pathfinding du
    jeu. Mais `blocked_from_highlight` declare infranchissable tout ce qui n'y figure
    pas : une lecture PARTIELLE ne degrade pas la qualite, elle PARALYSE. Constate en
    jeu : « 0 cases atteignables (zone du jeu) » avec 4 PM, tour passe pour rien --
    alors que le meme bot, surlignage vide et BFS de secours, trouvait 17 cases."""

    def _state(self, hint):
        return CombatState(turn=1, is_our_turn=True, reachable_hint=hint, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=CENTRE, hp=100, hp_max=100,
                   ap=6, mp=3, is_self=True)])

    def test_a_detached_highlight_is_ignored(self):
        """Des cases surlignees loin du personnage ne decrivent pas SA zone."""
        far = {BOARD.index_at(0, 0), BOARD.index_at(0, 1)}
        turn = TurnState.from_combat(self._state(far))
        options = movement_options(BOARD, self._state(far), turn)
        assert len(options) > 1, "le BFS de secours doit reprendre la main"

    def test_a_contiguous_highlight_is_trusted(self):
        """Le cas nominal ne doit pas changer : le jeu fait autorite."""
        near = {BOARD.index_at(2, 3), BOARD.index_at(3, 2)}
        state = self._state(near)
        options = movement_options(BOARD, state, TurnState.from_combat(state))
        assert set(options) - {CENTRE} == near

    def test_an_empty_highlight_falls_back(self):
        """Deja le comportement d'avant : un ensemble vide devient None en amont."""
        state = self._state(None)
        assert len(movement_options(BOARD, state, TurnState.from_combat(state))) > 1

class TestTheZoneSurvivesAMove:
    """La zone du jeu reste valide APRES un deplacement, et le solveur l'abandonnait.

    C'est demontrable : si X est atteignable depuis A avec les PM restants, alors le
    chemin depart -> A -> X coute au plus le total de PM. X etait donc deja atteignable au
    depart, donc X est dans la zone. La zone est un MAJORANT qui survit au deplacement.

    Ce que coutait l'abandon, sur un personnage tacle a qui le jeu n'accorde qu'un pas :
    au depart 2 cases, correct ; apres ce pas, le BFS en proposait 12, dont 11 hors zone.
    Le solveur planifiait une traversee de plateau apres le premier pas -- et la table de
    tir, construite sur les seules cases de depart, ne repondait rien pour celles-la :
    les lancers y etaient perdus EN SILENCE."""

    def _tackled(self):
        """Le jeu n'accorde qu'un pas, alors que 4 PM restent affiches."""
        start, step = CENTRE, BOARD.index_at(2, 1)
        state = _state(reachable_hint={start, step})
        return state, start, step

    def test_after_one_step_it_stays_in_the_zone(self):
        state, start, step = self._tackled()
        moved = TurnState(cell=step, ap=6, mp=3)
        assert set(movement_options(BOARD, state, moved)) <= {start, step}

    def test_the_bfs_would_have_escaped_it(self):
        """Controle : sans la zone, le meme etat propose bien plus. Sans quoi le test
        precedent passerait pour de mauvaises raisons."""
        _, _, step = self._tackled()
        moved = TurnState(cell=step, ap=6, mp=3)
        assert len(movement_options(BOARD, _state(), moved)) > 5

    def test_the_start_cell_stays_reachable(self):
        """Revenir sur ses pas doit rester possible : c'est un coup legitime."""
        state, start, step = self._tackled()
        assert start in movement_options(BOARD, state, TurnState(cell=step, ap=6, mp=3))


class TestSpellRangeIsNotPlacement:
    """Le critere « assez de rouge, donc placement » pris en flagrant delit.

    bugcarreblanc.png affiche « FIN DE TOUR » en toutes lettres — c'est un tour ordinaire.
    Il donnait pourtant 17,9 % de cases rouges, au-dessus du seuil de 15 %. Ce rouge est
    la PORTEE D'UN SORT, que le jeu affiche des qu'un sort est selectionne :

        rouge aux distances {2: 6, 3: 8, 4: 10}     un anneau contigu de 2 a 4
        vert  a  la distance {1: 3}                 le deplacement, lui, colle au joueur

    La boucle se refermait sur elle-meme : le sort selectionne declenchait la fausse
    detection, la session repondait PLACEMENT, et le runner envoyait la touche « pret » —
    qui est celle de fin de tour. Le bot terminait son tour PARCE QU'IL avait choisi un
    sort.
    """

    def test_an_annulus_around_the_player_is_a_spell_range(self):
        ring = {i for i in range(len(BOARD))
                if 2 <= grid_distance(BOARD, 12, i) <= 4}
        assert looks_like_spell_range(BOARD, ring, 12)

    def test_a_zone_containing_the_player_is_not(self):
        """Une portee de sort ne contient jamais la case du lanceur ; une zone de
        placement contient celle du personnage qu'on y a pose."""
        blob = {12} | {i for i in range(len(BOARD)) if grid_distance(BOARD, 12, i) <= 2}
        assert not looks_like_spell_range(BOARD, blob, 12)

    def test_a_gap_in_the_distances_is_not_a_range(self):
        """Contigue ou rien : deux blocs separes ne sont pas un anneau de portee."""
        split = {i for i in range(len(BOARD))
                 if grid_distance(BOARD, 12, i) in (1, 4)}
        assert not looks_like_spell_range(BOARD, split, 12)

    def test_the_real_capture_is_no_longer_placement(self):
        path = Path(__file__).resolve().parents[1] / "bugcarreblanc.png"
        if not path.exists():
            pytest.skip("bugcarreblanc.png absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        me = next((e for e in observe(frame, board, cursor=None).entities
                   if e.team is Team.ALLY), None)
        assert me is not None, "joueur non localise"
        assert is_placement_phase(frame, board), "le critere seul doit encore se tromper"
        assert not is_placement_phase(frame, board, player_cell=me.cell)

    def test_without_the_player_the_old_answer_stands(self):
        """L'ignorance ne bloque pas : joueur non localise, on retombe sur le critere
        seul. Un garde-fou qui EXIGE une lecture qui peut manquer se desactive tout seul
        au pire moment."""
        frame = _frame()
        for index in range(len(BOARD)):
            cv2.fillPoly(frame, [_cell_polygon(BOARD, index)], (40, 40, 130))
        assert is_placement_phase(frame, BOARD) is is_placement_phase(
            frame, BOARD, player_cell=None)


class TestASelectedSpellMasksTheMovementRange:
    """Deuxieme consequence du meme fait, et plus insidieuse que la premiere.

    Un sort selectionne fait peindre sa portee PAR-DESSUS le plateau : les cases de cette
    portee ne sont plus vertes, qu'elles soient atteignables ou non. Leur statut de
    deplacement est CACHE, pas negatif.

    Or cette lecture tronquee faisait autorite. `_plausible_hint` ne demande qu'une seule
    case voisine surlignee -- il y en avait trois -- puis `blocked_from_highlight`
    declarait infranchissable tout le reste du plateau :

        bugcarreblanc.png   4 PM ->  4 cases atteignables    apres :  30
        tacle.png           4 PM ->  3 cases atteignables    apres :  24

    Le bot avait quatre points de mouvement et une case pour les depenser. Il ne pouvait
    donc plus s'approcher de personne -- exactement le symptome rapporte.
    """

    def _reachable(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        return observe(frame, board, cursor=None).reachable

    def test_a_masked_range_is_reported_unknown_not_empty(self):
        """Vide veut dire « je ne sais pas », et la chaine sait deja quoi en faire :
        `assemble` met `reachable_hint` a None et le solveur retombe sur son BFS."""
        assert self._reachable("bugcarreblanc.png") == set()
        assert self._reachable("tacle.png") == set()

    def test_an_unmasked_turn_keeps_the_game_answer(self):
        """Sans sort selectionne, le surlignage reste l'autorite -- c'est le pathfinding
        du jeu, obstacles compris, et le perdre serait une regression."""
        assert len(self._reachable("combat1.png")) == 10


class TestMovementZoneLocatesThePlayer:
    """Un localisateur INDEPENDANT des marqueurs de couleur, et c'est ce qui manquait.

    Une position fausse ne se voit nulle part : les numeros de case restent plausibles,
    le plan reste credible, mais tout y est calcule autour d'un personnage qui n'est pas
    la. Le jeu, lui, centre la zone de deplacement sur le personnage, dans le rayon des PM.

    Mesure sur les six captures de combat, case detectee -> case impliquee :

        tacle.png           26 ->  26    ecart 0
        bugcarreblanc.png   83 ->  83    ecart 0
        combat1.png         36           rayon 3 pour 3 PM : coherent
        combat1920.png     119 -> 142    ecart 10, rayon 14 detecte contre 4 implique
        bug.png            161           aucun centre dans le rayon : ce n'est pas une zone
    """

    def test_a_disc_gives_its_centre(self):
        disc = {i for i in range(len(BOARD)) if grid_distance(BOARD, CENTRE, i) <= 2}
        assert movement_centre(BOARD, disc, mp=3) == CENTRE

    def test_scattered_cells_give_nothing(self):
        """Elle sait se taire : si aucune case ne rassemble le vert dans le rayon des PM,
        ce vert n'est pas une zone de deplacement et on n'en conclut RIEN."""
        corners = {0, len(BOARD) - 1}
        assert movement_centre(BOARD, corners, mp=1) is None

    def test_no_movement_points_gives_nothing(self):
        assert movement_centre(BOARD, {CENTRE}, mp=0) is None

    def test_it_confirms_a_detection_that_was_already_right(self):
        """bugcarreblanc.png : centre unique 83, exactement la case detectee. Un temoin
        qui ne dirait que « c'est faux » ne vaudrait rien -- il doit aussi confirmer."""
        path = Path(__file__).resolve().parents[1] / "bugcarreblanc.png"
        if not path.exists():
            pytest.skip("bugcarreblanc.png absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        obs = observe(frame, board, cursor=None)
        me = next(e for e in obs.entities if e.team is Team.ALLY)
        assert movement_centre(board, read_movement_range(frame, board),
                               obs.ui.pm or 0) == me.cell

    @pytest.mark.parametrize("name,expected", [("combat1920.png", 142),
                                               ("bug4.png", 70)])
    def test_a_wrong_detection_is_put_back_in_place(self, name, expected):
        """Les deux captures ou les marqueurs se trompaient de DIX cases. La case attendue
        a ete verifiee a l'oeil sur la capture annotee : le personnage y est."""
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        me = next(e for e in observe(frame, board, cursor=None).entities if e.team is Team.ALLY)
        assert me.cell == expected

    def test_a_consistent_detection_is_left_alone(self):
        """combat1.png : la case detectee (36) est a portee de toute la zone, donc rien ne
        la contredit. Le centre calcule y differe pourtant (24) -- corriger sur cette base
        casserait une detection qui marche, et c'est le seul vrai risque de ce temoin."""
        path = Path(__file__).resolve().parents[1] / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        me = next(e for e in observe(frame, board, cursor=None).entities if e.team is Team.ALLY)
        assert me.cell == 36

    def test_ties_are_refused(self):
        """Une zone tronquee -- obstacle, bord de plateau, portee de sort par-dessus --
        peut avoir plusieurs cases aussi centrales. Departager au hasard donnerait une
        position fausse la ou l'on croit en gagner une. Mesure : tacle.png a deux centres
        ex-aequo {26, 38}, et la case detectee (26) en fait partie -- se taire n'y perd
        donc rien."""
        path = Path(__file__).resolve().parents[1] / "tacle.png"
        if not path.exists():
            pytest.skip("tacle.png absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        assert movement_centre(board, read_movement_range(frame, board), 4) is None


class TestATieIsAShortListNotAnIgnorance:
    """UNE EGALITE N'EST PAS UNE IGNORANCE. Quand la zone de deplacement designe deux cases
    aussi bien l'une que l'autre, `movement_centre` renoncait -- alors qu'il restait un
    deuxieme signal capable de choisir DANS une liste de deux, sans etre capable de choisir
    tout seul parmi deux cents.

    Le marqueur de couleur du joueur est ce signal faible : mesure sur les huit captures de
    `data/runs/` ou les deux repondent, il donne la bonne case 3 fois sur 8, avec des ecarts
    jusqu'a 16 cases, et son vote ne dit rien de sa justesse (0,479 se trompe de 4 cases,
    0,250 tombe juste). Inutilisable seul, decisif entre deux.

    L'INDEPENDANCE EST PRESERVEE LA OU ELLE COMPTE : l'ensemble des candidats vient de la
    seule zone. Les marqueurs n'entrent qu'apres, et seulement si UN SEUL candidat en porte.
    """

    def _egalite(self):
        """Deux cases equivalentes : un segment de zone symetrique entre elles.

        Construit et non tire d'une capture, pour que le test dise ce qu'il teste. La
        verification que ces cases sont bien a egalite est faite par le premier test.
        """
        a = CENTRE
        voisins = sorted(i for i in range(len(BOARD)) if grid_distance(BOARD, a, i) == 1)
        b = voisins[0]
        zone = {i for i in range(len(BOARD))
                if grid_distance(BOARD, a, i) <= 1 and grid_distance(BOARD, b, i) <= 1}
        return a, b, zone

    def test_without_a_marker_it_still_refuses(self):
        """LA BASE DE COMPARAISON : sans marqueur, l'egalite reste une egalite."""
        a, b, zone = self._egalite()
        assert movement_centre(BOARD, zone, mp=1) is None, (
            "ces deux cases doivent etre a egalite, sinon les tests suivants ne portent "
            "sur rien")
        assert a != b

    def test_a_single_marked_candidate_breaks_it(self):
        a, b, zone = self._egalite()
        assert movement_centre(BOARD, zone, mp=1, marked={a}) == a
        assert movement_centre(BOARD, zone, mp=1, marked={b}) == b

    def test_two_marked_candidates_keep_the_ambiguity(self):
        """CONTRE-EPREUVE : le marqueur departage, il ne vote pas. Deux candidats marques,
        c'est une ambiguite REELLE -- en trancher une au hasard reintroduirait exactement
        la position fausse que ce localisateur existe pour eviter."""
        a, b, zone = self._egalite()
        assert movement_centre(BOARD, zone, mp=1, marked={a, b}) is None

    def test_a_marker_elsewhere_changes_nothing(self):
        """Un marqueur hors de la liste courte n'a pas voix au chapitre : c'est la zone qui
        decide QUI est candidat."""
        a, b, zone = self._egalite()
        ailleurs = next(i for i in range(len(BOARD)) if i not in (a, b))
        assert movement_centre(BOARD, zone, mp=1, marked={ailleurs}) is None

    def test_the_pipeline_actually_hands_the_markers_over(self, tmp_path):
        """CABLAGE, et il manquait : les cinq tests ci-dessus passent tous alors que
        `observe` ne transmet AUCUN marqueur -- ils appellent `movement_centre` en direct.
        Un departage qui fonctionne et n'est branche nulle part se comporte comme un
        departage absent, et rien ne le signale.

        20260807-220519.png est la capture qui l'exerce : la zone y designe 38 et 54 a
        egalite, un seul marqueur brut tombe sur 54, et la voie par marqueurs seule plaçait
        le personnage en 93. Verifie a l'oeil sur la capture annotee : le libelle « MOI »
        de l'overlay est bien sur 54."""
        chemin = Path(__file__).resolve().parents[1] / "data" / "runs" /             "20260807-220519.png"
        if not chemin.exists():
            pytest.skip("capture absente")
        frame = cv2.imread(str(chemin))
        board = build_board(frame)
        if board is None:
            pytest.skip("plateau non estimable")
        me = next((e for e in observe(frame, board, cursor=None).entities if e.is_self),
                  None)
        assert me is not None and me.cell == 54, (
            "sans les marqueurs transmis, l'egalite reste non tranchee et le personnage "
            "garde la position fausse issue des seuls marqueurs")

    def test_a_clear_winner_ignores_the_markers(self):
        """CONTRE-EPREUVE DE PORTEE, et c'est celle qui compte : le marqueur ne doit
        JAMAIS peser quand la zone tranche deja. Sinon un signal a 3/8 de justesse
        deplacerait une reponse geometrique sure."""
        disc = {i for i in range(len(BOARD)) if grid_distance(BOARD, CENTRE, i) <= 2}
        ailleurs = next(i for i in range(len(BOARD)) if i != CENTRE)
        assert movement_centre(BOARD, disc, mp=3, marked={ailleurs}) == CENTRE
        assert movement_centre(BOARD, disc, mp=3, marked=set()) == CENTRE

class TestUnreachableGreenIsDropped:
    """Le vert du surlignage attrape aussi de l'HERBE, et je ne l'avais jamais chiffre.

    Hors combat, ou le jeu n'affiche AUCUNE zone de deplacement, le lecteur en trouve
    pourtant 12 et 18 selon la capture : 100 % de faux. En combat le plateau recouvre
    presque toute l'herbe -- cinq captures sur six n'ont aucun faux positif -- mais
    bug.png en garde quatre, dont une a VINGT-SEPT cases du personnage pour 4 PM.

    Le filtre n'est pas un seuil : une case plus loin que les PM restants est
    inatteignable, c'est une regle du jeu. Rien a mesurer, rien a regler.
    """

    def _reachable(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        return observe(frame, build_board(frame), cursor=None).reachable

    def test_grass_beyond_the_movement_points_is_dropped(self):
        assert self._reachable("bug.png") == set()

    @pytest.mark.parametrize("name,expected", [("combat1.png", 10),
                                               ("combat1920.png", 25)])
    def test_a_genuine_zone_is_untouched(self, name, expected):
        """Le pendant indispensable : un filtre qui rognerait la vraie zone rendrait le
        surlignage inutilisable, et le jeu sait mieux que nous ou l'on peut aller."""
        assert len(self._reachable(name)) == expected

    def test_the_relocation_still_sees_the_raw_zone(self):
        """L'ORDRE compte. Le recentrage du joueur a besoin du surlignage BRUT : c'est en
        constatant qu'aucune case detectee n'explique la zone qu'il repere une position
        fausse. Filtrer avant lui effacerait la preuve pour faire disparaitre le
        symptome -- et combat1920.png retomberait a une position fausse de dix cases."""
        path = Path(__file__).resolve().parents[1] / "combat1920.png"
        if not path.exists():
            pytest.skip("combat1920.png absent")
        frame = cv2.imread(str(path))
        me = next(e for e in observe(frame, build_board(frame), cursor=None).entities
                  if e.team is Team.ALLY)
        assert me.cell == 142


class TestTheZoneCanAlsoCreateTheMissingPlayer:
    """Vingt-quatre captures d'une VRAIE session ont ete fournies. Le doctor y donne :

        coordonnees 24/24   grille 24/24   phase 24/24   barre de sorts 24/24
        entites : 5 ok, 10 avertissements, 9 echecs

    Les echecs sont tous « joueur non localise », sur des plateaux pourtant sains de 175 a
    242 cases. Verifie a l'oeil : le contour de la case du personnage y est BLEU la ou le
    detecteur attend du blanc peu sature.

    La zone de deplacement, elle, est bien lue -- mais l'ancien critere y renoncait, parce
    que TROIS cases d'herbe a l'autre bout de l'ecran faisaient passer le rayon de 4 a 9.
    Un critere qui minimise le maximum est ruine par une seule aberration.

    Le nouveau combine COUVERTURE (ignore les aberrations) et RESSERREMENT (departage
    quand la zone est plus petite que les PM). Chacun corrige le defaut de l'autre :
    seul, le premier egalise, seul, le second se laisse ruiner.

    Resultat : joueur localise sur 22 captures de session sur 24, contre 15 avant.
    """

    def _self(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        return next((e for e in observe(frame, build_board(frame), cursor=None).entities
                     if e.team is Team.ALLY), None)

    @pytest.mark.parametrize("name", ["data/runs/20260807-222431.png",
                                      "data/runs/20260807-222627.png",
                                      "data/runs/20260807-222542.png"])
    def test_a_missing_player_is_created_from_the_zone(self, name):
        me = self._self(name)
        assert me is not None and me.is_self

    def test_the_verified_one_lands_where_the_run_s_overlay_put_the_player(self):
        """20260807-222431.png : la case deduite (190) porte le libelle « MOI 160 ».

        CE LIBELLE VIENT DE L'OVERLAY DU BOT, pas du jeu -- `overlay.py` ecrit « MOI » et
        « ENNEMI », et les captures de `data/runs/` ont ete enregistrees annotees. Le dire
        « ecrit par le jeu », comme cette methode le faisait, prete au controle une force
        qu'il n'a pas : ce n'est pas le temoignage du client, c'est celui d'une version
        anterieure du code.

        CE QU'IL VAUT QUAND MEME, et ce n'est pas rien : le numero du libelle (160) et la
        case deduite (190) DIFFERENT, parce que l'indexation depend de la grille estimee et
        change d'une session a l'autre. Ce qui coincide est la POSITION a l'ecran. Le
        localisateur par zone de deplacement retombe donc a l'endroit ou la voie par
        marqueurs avait place le personnage lors de l'enregistrement -- deux chemins
        differents, meme endroit."""
        me = self._self("data/runs/20260807-222431.png")
        assert me is not None and me.cell == 190

    def test_health_comes_from_the_gauges(self):
        """Une entite creee sans PV laisserait `danger_factor` neutre, et le solveur
        jouerait a l'aveugle sur sa propre survie -- le terme le plus lourd du score apres
        la mise a mort."""
        me = self._self("data/runs/20260807-222431.png")
        assert me is not None and me.hp_known and me.hp_max > 1

    @pytest.mark.parametrize("name,expected", [("combat1920.png", 142),
                                               ("bug4.png", 70)])
    def test_existing_corrections_still_hold(self, name, expected):
        assert self._self(name).cell == expected


class TestTheSelfLocationRateAcrossTheWholeSession:
    """« joueur localise sur 22 captures sur 24, contre 15 avant » : ce chiffre justifie a
    lui seul le critere couverture + resserrement, et il n'etait ecrit que dans le
    docstring voisin. Trois captures nommees etaient verifiees ; les vingt et une autres
    ne l'etaient pas.

    Trois captures choisies ne disent rien du taux : un critere qui les reussirait en
    perdant huit autres passerait pour intact. C'est le meme trou que pour le comptage
    d'ennemis, et le meme remede -- asservir la LISTE des echecs plutot que leur nombre,
    pour qu'une permutation se voie.

    Reproduction de la mesure : 22 sur 24, les deux echecs etant 220329 et 221808.
    """

    ECHECS: ClassVar = {"20260807-220329.png", "20260807-221808.png"}

    _CACHE: ClassVar[dict] = {}

    def _echecs(self):
        if "echecs" in self._CACHE:
            return self._CACHE["echecs"]
        racine = Path(__file__).resolve().parents[1]
        captures = sorted((racine / "data" / "runs").glob("*.png"))
        if len(captures) < 20:
            pytest.skip("captures de session absentes")
        perdus = set()
        for chemin in captures:
            frame = cv2.imread(str(chemin))
            board = build_board(frame)
            if board is None:
                perdus.add(chemin.name)
                continue
            if not any(e.is_self for e in observe(frame, board, cursor=None).entities):
                perdus.add(chemin.name)
        self._CACHE["echecs"] = (perdus, len(captures))
        return self._CACHE["echecs"]

    def test_the_failing_captures_are_exactly_the_known_ones(self):
        perdus, _total = self._echecs()
        assert perdus == self.ECHECS, f"la liste des echecs a change : {sorted(perdus)}"

    def test_the_rate_has_not_fallen_back(self):
        """Le taux chiffre, pour que la chute se lise directement. 15 etait la valeur
        d'avant le critere actuel : y revenir doit rougir bruyamment."""
        perdus, total = self._echecs()
        assert total - len(perdus) >= 22, (
            f"{total - len(perdus)} joueurs localises sur {total}, mesure a 22")


class TestCellsAtTheFrameEdge:
    """Le decoupage au bord est le point risque du passage au rectangle englobant.

    Avec un masque plein ecran, une case qui depassait de l'image etait tronquee
    IMPLICITEMENT : fillPoly ecrivait hors du cadre, personne ne s'en apercevait, et
    l'ensemble lu ne portait que la partie visible. Le rectangle englobant, lui, doit
    ramener ses bornes a la main -- et une erreur de signe y produirait soit une exception,
    soit un ratio calcule sur une surface tronquee, donc un faux positif.

    Ces cas sont rares dans les captures du depot, d'ou ce test construit : un plateau
    volontairement pose a cheval sur les quatre bords.
    """

    def _board(self, origin):
        return BoardMap(
            cells=np.array([(i, j) for j in range(5) for i in range(5)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array(origin, dtype=float))

    @pytest.mark.parametrize("origin", [(0.0, 0.0), (5.0, 5.0),
                                        (795.0, 595.0), (-30.0, 300.0)])
    def test_a_board_hanging_off_the_edge_does_not_raise(self, origin):
        frame = np.zeros((600, 800, 3), np.uint8)
        board = self._board(origin)
        assert read_movement_range(frame, board) == set()
        assert read_placement_zone(frame, board) == set()

    def test_a_fully_lit_frame_lights_the_cells_that_are_visible(self):
        """Contre-epreuve : sans elle, le test precedent serait satisfait par une fonction
        qui ne rend JAMAIS rien."""
        frame = np.zeros((600, 800, 3), np.uint8)
        frame[:] = (60, 200, 60)                    # vert franc, dans GREEN_BAND
        board = self._board((400.0, 300.0))
        allumees = read_movement_range(frame, board)
        assert allumees, "aucune case allumee sur une image entierement verte"


class TestTheSmallZoneRefusalIsProtectiveNotAnOversight:
    """LE SEUIL DE COUVERTURE PARAIT AVOIR UN DEFAUT D'ARRONDI, ET C'EST SA QUALITE.

    `MIN_ZONE_COVERAGE` vaut 80 %, mais l'exigence REELLE saute avec la taille de la zone :

        cases lues     2   3   4   5   6   8  10  20  30
        exigees        2   3   4   4   5   7   8  16  24
        parasites
        tolerees       0   0   0   1   1   1   2   4   6

    En dessous de cinq cases, il faut donc 100 % : une seule case d'herbe egaree fait
    renoncer. Cela ressemble a une garde plus etroite que sa promesse -- le defaut que ce
    projet corrige d'habitude. J'ai mesure avant de corriger.

    LES TROIS CAPTURES QUE CETTE STRICTESSE SAUVE. Sur 220355, 220420 et 221831, la lecture
    rend quatre cases : trois groupees et une egaree. Assouplir le seuil a « tolerer au
    moins un parasite » donne un centre unique a chaque fois -- et deplacerait le
    personnage de 9, 9 et 18 cases.

    Ou sont ces trois cases ? A UNE OU DEUX CASES D'UN ENNEMI, et a huit a dix-huit du
    joueur. Ce n'est pas une zone de deplacement mal lue, c'est une portee de sort ou une
    zone hostile. Le seuil ne se trompe pas d'arrondi : il refuse une zone qui n'est pas
    celle qu'on croit lire.

    `looks_like_spell_range` ne les attrape pas -- elle cherche un ANNEAU centre sur le
    joueur, pas un paquet colle a un ennemi. Il n'existe donc rien d'autre ici que ce
    seuil, et c'est pour cela qu'il est tenu.
    """

    CAPTURES = ("20260807-220355.png", "20260807-220420.png", "20260807-221831.png")

    def _lecture(self, nom):
        chemin = Path(__file__).resolve().parents[1] / "data" / "runs" / nom
        if not chemin.exists():
            pytest.skip(f"{nom} absent")
        frame = cv2.imread(str(chemin))
        board = build_board(frame)
        if board is None:
            pytest.skip("plateau non estimable")
        obs = observe(frame, board, cursor=None)
        moi = next((e.cell for e in obs.entities if e.is_self), None)
        ennemis = [e.cell for e in obs.entities if not e.is_self]
        if moi is None or not ennemis:
            pytest.skip("joueur ou ennemis non localises")
        return board, read_movement_range(frame, board), obs.ui.pm, moi, ennemis

    @pytest.mark.parametrize("nom", CAPTURES)
    def test_the_zone_is_refused(self, nom):
        board, zone, pm, _moi, _ennemis = self._lecture(nom)
        assert movement_centre(board, zone, pm or 0) is None

    @pytest.mark.parametrize("nom", CAPTURES)
    def test_and_the_reason_is_that_it_hugs_an_enemy(self, nom):
        """CE QUI DONNE SON SENS AU TEST PRECEDENT. Un refus peut etre une prudence
        stupide ; celui-ci porte sur une zone qui n'est pas au joueur. Sans cette
        assertion, quelqu'un lirait le refus comme un reglage trop severe -- ce que j'ai
        moi-meme cru avant de mesurer."""
        board, zone, _pm, moi, ennemis = self._lecture(nom)
        groupe = sorted(zone, key=lambda c: min(grid_distance(board, e, c)
                                                for e in ennemis))[:3]
        pres_ennemi = max(min(grid_distance(board, e, c) for e in ennemis)
                          for c in groupe)
        loin_joueur = min(grid_distance(board, moi, c) for c in groupe)
        assert pres_ennemi <= 2 < loin_joueur, (
            f"les cases lues sont a {pres_ennemi} d'un ennemi et {loin_joueur} du joueur : "
            f"si elles etaient pres du joueur, le refus serait effectivement trop severe")


class TestAConfirmedPositionIsDistinguishableFromAGuessedOne:
    """`cell_confirmed` EST LE PENDANT DE `hp_known`, POUR LA POSITION.

    Sans lui, deux situations tres differentes rendaient la meme entite : une case validee
    par la zone de deplacement du jeu, et une case issue des seuls marqueurs de couleur.
    Or la mesure separe ces deux cas nettement -- sur les captures ou les deux signaux
    repondent, la case du marqueur couvre 0 % de la zone HUIT FOIS SUR QUATORZE, avec des
    ecarts jusqu'a seize cases. Ce n'est pas une imprecision, c'est un autre endroit.

    Une position fausse ne se voit nulle part : les numeros restent plausibles et le plan
    reste credible. Ce booleen est le seul endroit ou la difference peut se lire.
    """

    def _me(self, nom):
        chemin = Path(__file__).resolve().parents[1] / "data" / "runs" / nom
        if not chemin.exists():
            pytest.skip(f"{nom} absent")
        frame = cv2.imread(str(chemin))
        board = build_board(frame)
        if board is None:
            pytest.skip("plateau non estimable")
        return next((e for e in observe(frame, board, cursor=None).entities
                     if e.is_self), None)

    def test_a_zone_backed_position_is_marked_confirmed(self):
        """20260807-220519.png : la zone tranche l'egalite et donne la case 54."""
        me = self._me("20260807-220519.png")
        assert me is not None and me.cell == 54 and me.cell_confirmed

    def test_a_marker_only_position_is_not(self):
        """CONTRE-EPREUVE, et c'est la seule qui donne du sens a l'autre : un drapeau
        toujours vrai ne distinguerait rien.

        20260807-224524.png : la lecture de la zone de deplacement n'y rend que DEUX cases,
        d'ou aucun centre deductible, et la case du joueur vient donc des marqueurs seuls.

        J'avais d'abord ecrit ici « c'est une phase de placement ». C'est FAUX, et la
        verification tient en un crop : le bouton affiche « FIN DE TOUR » avec 6 s au
        compteur, la ou un placement afficherait « PRET ». `is_placement_phase` rend
        d'ailleurs False sur les 28 captures du depot, celle-ci comprise. Ce que j'avais
        pris pour un marqueur de placement etait le personnage rendu petit, et les cases
        vertes et rouges etaient la zone de deplacement et la zone hostile.
        """
        me = self._me("20260807-224524.png")
        assert me is not None and not me.cell_confirmed

    def test_a_centre_landing_on_the_player_confirms_instead_of_colliding(self):
        """LE PIEGE QUE LE DRAPEAU A REVELE. La garde anti-collision refuse un centre deja
        occupe -- mais `entities` contient le JOUEUR, donc elle attrapait aussi le cas ou
        la zone tombe exactement sur sa case, c'est-a-dire une CONFIRMATION.

        Invisible tant que rien n'enregistrait la confirmation. Une fois le drapeau pose :
        222511 et 222604 ont 97 % de couverture, un centre egal a leur case, et se
        declaraient non verifiees."""
        for nom, case in (("20260807-222511.png", 38), ("20260807-222604.png", 105)):
            me = self._me(nom)
            assert me is not None and me.cell == case and me.cell_confirmed, (
                f"{nom} : la zone confirme la case {case} et le drapeau reste faux")

    def test_a_position_already_consistent_is_confirmed_too(self):
        """L'AUTRE CHEMIN, et il ne passe pas par une correction. Quand la case detectee
        explique deja toute la zone, la fonction sort par le haut sans rien changer -- une
        sortie precoce qui oubliait de dire qu'elle venait de VERIFIER quelque chose.
        221351 y passe : rayon maximal 4 pour 4 PM."""
        me = self._me("20260807-221351.png")
        assert me is not None and me.cell_confirmed

    def test_the_two_cases_both_occur_in_the_session(self):
        """Un drapeau qui ne prendrait qu'une seule valeur sur toute une session ne
        porterait aucune information. Mesure : 15 confirmees sur 22."""
        dossier = Path(__file__).resolve().parents[1] / "data" / "runs"
        captures = sorted(dossier.glob("*.png"))
        if not captures:
            pytest.skip("captures absentes")
        etats = []
        for chemin in captures:
            frame = cv2.imread(str(chemin))
            board = build_board(frame)
            if board is None:
                continue
            me = next((e for e in observe(frame, board, cursor=None).entities
                       if e.is_self), None)
            if me is not None:
                etats.append(me.cell_confirmed)
        assert etats.count(True) >= 15 and etats.count(False) >= 3, (
            f"{etats.count(True)} confirmees / {etats.count(False)} non : un drapeau "
            f"quasi constant ne distinguerait rien")

    def test_the_default_is_unconfirmed(self):
        """Le defaut doit etre « non confirmee ». Une entite construite ailleurs -- test,
        simulateur, etat rejoue -- ne doit pas se declarer verifiee par une zone qu'elle
        n'a jamais vue. C'est la meme regle que pour `left`/`top` des candidats de chasse :
        une valeur plausible par defaut est pire qu'une absence."""
        from jev_tactics.state import Entity, Team
        e = Entity(entity_id="x", team=Team.ALLY, cell=3, hp=10, hp_max=10, ap=6, mp=3)
        assert not e.cell_confirmed
