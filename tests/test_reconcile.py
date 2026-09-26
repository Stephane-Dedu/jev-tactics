"""Reconciliation plateau / timeline, et unicite du personnage joue.

Les deux defauts corriges ici ont ete reveles par le CLIENT REEL, pas par un test : le
doctor a rapporte « 3 joueur, 2 ennemis, ecart timeline 2 » sur un combat a 3
combattants. Deux consequences, dont la seconde est bien plus grave que la premiere.
"""


import sys

import numpy as np
import pytest

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception.entities import (
    CellMarker,
    hovered_cell,
    markers_to_entities,
    reconcile_with_timeline,
)
from jev_tactics.perception.timeline import TimelineEntry
from jev_tactics.state import CombatState, Team


def _portrait(slot, team):
    return TimelineEntry(slot=slot, x=1252 + 77 * slot, hp_ratio=1.0, team=team)


def _marker(cell, team, vote=0.5):
    return CellMarker(cell=cell, team=team, vote_ratio=vote)


class TestSingleSelf:
    """LE defaut grave. `is_self` etait pose sur TOUT marqueur allie ; le client reel en
    produit trois. `self_entity()` en choisissait alors un par ordre de CASE, et toute la
    geometrie du tour -- deplacements, portees, lignes de vue -- se calculait depuis une
    position ou le personnage n'est pas. Sans qu'aucune erreur ne soit levee."""

    def test_exactly_one_self_among_several_allies(self):
        entities = markers_to_entities([
            _marker(10, Team.ALLY, 0.63),
            _marker(44, Team.ALLY, 0.22),
            _marker(77, Team.ALLY, 0.25),
        ])
        assert sum(1 for e in entities if e.is_self) == 1

    def test_self_is_the_most_confident_not_the_lowest_cell(self):
        """Le cas qui piegeait : le mieux vote n'est pas la case la plus basse."""
        entities = markers_to_entities([
            _marker(3, Team.ALLY, 0.21),      # faux positif, case basse
            _marker(50, Team.ALLY, 0.68),     # le vrai personnage
        ])
        assert next(e for e in entities if e.is_self).cell == 50

    def test_other_allies_are_kept_as_allies(self):
        """En combat de groupe ils sont reels : les effacer priverait le planificateur
        des cases qu'ils occupent."""
        entities = markers_to_entities([
            _marker(10, Team.ALLY, 0.7), _marker(20, Team.ALLY, 0.3)])
        assert len(entities) == 2
        assert all(e.team is Team.ALLY for e in entities)

    def test_self_entity_is_unambiguous(self):
        """Ce que ca garantit en aval : `self_entity()` ne depend plus de l'ordre."""
        entities = markers_to_entities([
            _marker(3, Team.ALLY, 0.2), _marker(50, Team.ALLY, 0.9)])
        assert CombatState(turn=1, entities=entities).self_entity().cell == 50

    def test_no_self_without_any_ally_marker(self):
        entities = markers_to_entities([_marker(5, Team.ENEMY, 0.6)])
        assert not any(e.is_self for e in entities)


class TestReconciliation:
    def test_trims_the_least_confident_phantom(self):
        markers = [_marker(1, Team.ENEMY, 0.7), _marker(2, Team.ENEMY, 0.2)]
        kept = reconcile_with_timeline(markers, [_portrait(0, Team.ENEMY)])
        assert [m.cell for m in kept] == [1]

    def test_keeps_everything_when_counts_agree(self):
        markers = [_marker(1, Team.ENEMY, 0.7), _marker(2, Team.ENEMY, 0.6)]
        kept = reconcile_with_timeline(
            markers, [_portrait(0, Team.ENEMY), _portrait(1, Team.ENEMY)])
        assert len(kept) == 2

    def test_does_not_invent_when_the_board_sees_fewer(self):
        """Asymetrie voulue : un marqueur manquant laisse un plan VALIDE mais incomplet,
        alors qu'un fantome fait gaspiller des PA. Inventer une position serait pire."""
        markers = [_marker(1, Team.ENEMY, 0.7)]
        kept = reconcile_with_timeline(
            markers, [_portrait(0, Team.ENEMY), _portrait(1, Team.ENEMY)])
        assert [m.cell for m in kept] == [1]

    def test_teams_are_trimmed_independently(self):
        """Le cas mesure sur le client : trop d'allies, le bon nombre d'ennemis."""
        markers = [_marker(1, Team.ALLY, 0.6), _marker(2, Team.ALLY, 0.3),
                   _marker(3, Team.ALLY, 0.2), _marker(9, Team.ENEMY, 0.5)]
        kept = reconcile_with_timeline(
            markers, [_portrait(0, Team.ALLY), _portrait(1, Team.ENEMY)])
        assert sorted(m.team for m in kept) == sorted([Team.ALLY, Team.ENEMY])

    def test_empty_timeline_changes_nothing(self):
        """Hors combat, ou lecture de timeline ratee : elle ne fait autorite que
        lorsqu'elle a elle-meme repondu. Sinon on effacerait tout."""
        markers = [_marker(1, Team.ENEMY, 0.7), _marker(2, Team.ENEMY, 0.6)]
        assert reconcile_with_timeline(markers, []) == markers

    def test_team_absent_from_the_timeline_is_left_alone(self):
        """Une equipe dont aucun portrait n'a ete lu ne doit pas etre effacee du plateau :
        l'absence d'information n'est pas l'information. Le TOTAL reste une borne."""
        markers = [_marker(1, Team.ALLY, 0.7), _marker(2, Team.ALLY, 0.6)]
        timeline = [_portrait(0, Team.ENEMY), _portrait(1, None), _portrait(2, None)]
        assert len(reconcile_with_timeline(markers, timeline)) == 2

    def test_total_bounds_the_count_when_no_team_is_readable(self):
        """LE trou constate en jeu. Le filtrage par equipe se desactive entierement quand
        aucun portrait n'est identifie (« equipe ? » partout) -- resultat : 12 ennemis
        detectes pour 3 combattants, et un tour depense sur une case vide.

        Or compter les barres ne demande pas de lire leur couleur : le total reste fiable
        quand les equipes ne le sont plus."""
        markers = [_marker(i, Team.ENEMY, 0.9 - i / 100) for i in range(12)]
        timeline = [_portrait(i, None) for i in range(3)]
        kept = reconcile_with_timeline(markers, timeline)
        assert len(kept) == 3

    def test_the_total_bound_keeps_the_best_voted(self):
        markers = [_marker(1, Team.ENEMY, 0.2), _marker(2, Team.ENEMY, 0.9)]
        kept = reconcile_with_timeline(markers, [_portrait(0, None)])
        assert [m.cell for m in kept] == [2]

    def test_output_is_ordered_by_cell(self):
        markers = [_marker(9, Team.ENEMY, 0.9), _marker(2, Team.ENEMY, 0.8)]
        kept = reconcile_with_timeline(
            markers, [_portrait(0, Team.ENEMY), _portrait(1, Team.ENEMY)])
        assert [m.cell for m in kept] == [2, 9]

    def test_the_real_client_case_end_to_end(self):
        """Reproduction exacte du constat doctor : 3 marqueurs joueur + 2 ennemis pour
        une timeline de 1 allie + 2 ennemis."""
        markers = [_marker(10, Team.ALLY, 0.63), _marker(44, Team.ALLY, 0.22),
                   _marker(77, Team.ALLY, 0.25), _marker(30, Team.ENEMY, 0.55),
                   _marker(31, Team.ENEMY, 0.51)]
        timeline = [_portrait(0, Team.ALLY), _portrait(1, Team.ENEMY),
                    _portrait(2, Team.ENEMY)]
        entities = markers_to_entities(reconcile_with_timeline(markers, timeline))
        assert len(entities) == 3
        assert sum(1 for e in entities if e.is_self) == 1
        assert next(e for e in entities if e.is_self).cell == 10


class TestHoverArtefact:
    """Le curseur du bot cree un contour blanc identique au marqueur du personnage.

    Boucle de retroaction constatee en jeu, et visible seulement sur le calque : apres un
    clic, le curseur du bot reste sur le plateau, le jeu y dessine un contour blanc, et a
    la frame suivante le bot PREND SON PROPRE SURVOL POUR SON PERSONNAGE. Toute la
    geometrie du tour se calcule alors depuis une case vide -- et la zone de deplacement
    du jeu, centree sur le vrai personnage, ne la touche plus : « 0 cases atteignables ».

    Ce defaut n'existe pas sur une capture figee : seule l'observation du bot EN ACTION
    pouvait le reveler."""

    def _board(self):
        return BoardMap(
            cells=np.array([(i, j) for j in range(5) for i in range(5)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([600.0, 400.0]))

    # Position du curseur SIMULEE. L'assertion d'origine etait
    # `assert hovered_cell(board) is None or True`, c'est-a-dire toujours vraie -- elle
    # dependait de la souris reelle, donc ne pouvait rien affirmer. Constate en
    # neutralisant la fonction : la suite entiere restait verte, alors que sans elle le
    # bot prend le contour blanc de son PROPRE curseur pour son personnage, et calcule
    # tout le tour depuis une case ou il n'est pas.
    @staticmethod
    def _cursor_at(monkeypatch, x, y):
        import ctypes

        class _Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        def get(pointeur):
            contenu = ctypes.cast(pointeur, ctypes.POINTER(_Point)).contents
            contenu.x, contenu.y = int(x), int(y)
            return 1

        monkeypatch.setattr(ctypes.windll.user32, "GetCursorPos", get, raising=False)

    @pytest.mark.skipif(sys.platform != "win32", reason="curseur lu via l'API Windows")
    def test_the_cell_under_the_cursor_is_the_one_returned(self, monkeypatch):
        board = self._board()
        for cell in (board.index_at(2, 2), board.index_at(0, 4)):
            x, y = board.center(cell)
            self._cursor_at(monkeypatch, x, y)
            assert hovered_cell(board) == cell

    @pytest.mark.skipif(sys.platform != "win32", reason="curseur lu via l'API Windows")
    def test_a_cursor_far_from_the_board_hovers_nothing(self, monkeypatch):
        """`nearest` rend TOUJOURS une case : sans ce controle, on ecarterait une case
        legitime chaque fois que la souris est ailleurs a l'ecran."""
        board = self._board()
        x, y = board.center(board.index_at(2, 2))
        self._cursor_at(monkeypatch, x + 300, y + 300)
        assert hovered_cell(board) is None

    def test_the_hovered_cell_is_excluded_from_entities(self, monkeypatch):
        import jev_tactics.pipeline as pipeline_module

        board = self._board()
        centre = board.index_at(2, 2)
        monkeypatch.setattr(pipeline_module, "detect_markers_on_board",
                            lambda _f, _b: [_marker(centre, Team.ALLY, 0.6)])
        monkeypatch.setattr(pipeline_module, "read_timeline", lambda _f: [])
        # Deux parametres : `observe` transmet desormais la position du curseur
        # qu'on lui donne, au lieu d'aller lire la souris du systeme.
        monkeypatch.setattr(pipeline_module, "hovered_cell",
                            lambda _b, _c=None: centre)
        monkeypatch.setattr(pipeline_module, "read_ui",
                            lambda _f: pipeline_module.UiReading())
        monkeypatch.setattr(pipeline_module, "read_movement_range", lambda _f, _b: set())
        obs = pipeline_module.observe(np.zeros((700, 900, 3), np.uint8), board=board)
        assert obs.entities == []

    def test_other_cells_survive(self, monkeypatch):
        import jev_tactics.pipeline as pipeline_module

        board = self._board()
        monkeypatch.setattr(pipeline_module, "detect_markers_on_board",
                            lambda _f, _b: [_marker(5, Team.ALLY, 0.6)])
        monkeypatch.setattr(pipeline_module, "read_timeline", lambda _f: [])
        monkeypatch.setattr(pipeline_module, "hovered_cell", lambda _b, _c=None: 9)
        monkeypatch.setattr(pipeline_module, "read_ui",
                            lambda _f: pipeline_module.UiReading())
        monkeypatch.setattr(pipeline_module, "read_movement_range", lambda _f, _b: set())
        obs = pipeline_module.observe(np.zeros((700, 900, 3), np.uint8), board=board)
        assert [e.cell for e in obs.entities] == [5]
