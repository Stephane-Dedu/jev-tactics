"""L'assemblage d'Observation est hermetique (OCR simule) et serialisable (replay)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics import pipeline
from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception import UiReading
from jev_tactics.perception.entities import CellMarker
from jev_tactics.perception.timeline import TimelineEntry
from jev_tactics.pipeline import (
    Observation,
    _readable_movement,
    build_board,
    observe,
    outside_ui,
)
from jev_tactics.state import Entity, Team

ROOT = Path(__file__).resolve().parents[1]

BOARD = BoardMap(
    cells=np.array([(i, j) for i in range(-1, 2) for j in range(-1, 2)], dtype=np.int64),
    e_x=np.array([46.0, 23.0]),
    e_y=np.array([-46.0, 23.0]),
    origin=np.array([300.0, 300.0]),
)


@pytest.fixture
def fake_ui(monkeypatch):
    """L'OCR est simule : la suite ne doit pas dependre du binaire Tesseract."""
    monkeypatch.setattr(
        pipeline, "read_ui", lambda _img: UiReading(pv=1597, pv_max=1597, pa=10, pm=4)
    )


def test_observe_reads_ui(fake_ui):
    obs = observe(np.zeros((1080, 1920, 3), dtype=np.uint8), timestamp=123.0)
    assert obs.timestamp == 123.0
    assert (obs.ui.pv, obs.ui.pa, obs.ui.pm) == (1597, 10, 4)


def test_entities_empty_without_board(fake_ui):
    """Sans plateau, la lecture d'UI marche quand meme (elle ne depend pas de la grille)."""
    obs = observe(np.zeros((1080, 1920, 3), dtype=np.uint8), board=None)
    assert obs.entities == []
    assert obs.ui.pv == 1597


def test_entities_detected_with_board(fake_ui, monkeypatch):
    monkeypatch.setattr(
        pipeline, "detect_markers_on_board",
        lambda _img, _board: [CellMarker(cell=0, team=Team.ENEMY, vote_ratio=0.6)],
    )
    obs = observe(np.zeros((1080, 1920, 3), dtype=np.uint8), board=BOARD)
    assert [e.cell for e in obs.entities] == [0]


def test_timeline_trims_phantom_markers(fake_ui, monkeypatch):
    """Bout en bout : le desaccord plateau/timeline est resolu AVANT que les entites
    n'atteignent le planificateur, qui viserait sinon des cases vides."""
    monkeypatch.setattr(
        pipeline, "detect_markers_on_board",
        lambda _img, _board: [
            CellMarker(cell=1, team=Team.ENEMY, vote_ratio=0.7),
            CellMarker(cell=2, team=Team.ENEMY, vote_ratio=0.2),   # fantome
        ],
    )
    monkeypatch.setattr(
        pipeline, "read_timeline",
        lambda _img: [TimelineEntry(slot=0, x=1252, hp_ratio=1.0, team=Team.ENEMY)],
    )
    obs = observe(np.zeros((1080, 1920, 3), dtype=np.uint8), board=BOARD)
    assert [e.cell for e in obs.entities] == [1]
    assert obs.perception_gap() == 0


def test_build_board_returns_none_without_grid():
    """Une frame sans reseau (bruit) ne doit pas produire de plateau fantome."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(600, 900, 3)).astype(np.uint8)
    assert build_board(noise) is None


class TestPerceptionGap:
    """Plateau et timeline sont deux perceptions INDEPENDANTES : leur desaccord revele
    un defaut sans qu'il faille une verite terrain."""

    def _obs(self, board_enemies, timeline_enemies):
        return Observation(
            timestamp=0.0,
            ui=UiReading(pv=50, pv_max=100),
            entities=[
                Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=i, hp=1, hp_max=1,
                       ap=0, mp=0) for i in range(board_enemies)
            ],
            timeline=[
                TimelineEntry(slot=i, x=1252 + 77 * i, hp_ratio=1.0, team=Team.ENEMY)
                for i in range(timeline_enemies)
            ],
        )

    def test_zero_when_sources_agree(self):
        assert self._obs(3, 3).perception_gap() == 0

    def test_flags_missing_board_marker(self):
        assert self._obs(3, 5).perception_gap() == 2

    def test_symmetric(self):
        assert self._obs(5, 3).perception_gap() == 2


def test_observation_roundtrip_json():
    obs = Observation(
        timestamp=1.0,
        ui=UiReading(pv=800, pv_max=1000, pa=6, pm=3),
        entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=4, hp=800, hp_max=1000,
                   ap=6, mp=3, is_self=True)
        ],
    )
    assert Observation.model_validate_json(obs.model_dump_json()) == obs


class TestUiExclusion:
    """Cases du plateau tombant sous l'interface.

    Defaut constate en jeu : un « ennemi » detecte a (937, 919), soit en plein sur la
    barre de sorts -- une icone orange prise pour un marqueur. Le controle
    plateau/timeline ne l'avait pas vu parce que le COMPTE tombait juste : un ennemi reel
    manque, un fantome ajoute. Un comptage ne detecte pas un echange."""

    def _board_at(self, x, y, cells=4):
        return BoardMap(
            cells=np.array([(i, 0) for i in range(cells)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([float(x), float(y)]))

    def test_cells_under_the_spell_bar_are_dropped(self):
        board = self._board_at(700, 950)      # en plein dans la barre de sorts
        assert len(outside_ui(board)) < len(board)

    def test_cells_on_the_play_area_are_kept(self):
        board = self._board_at(700, 400)
        assert len(outside_ui(board)) == len(board)

    def test_the_geometry_is_preserved(self):
        """Retirer des cases ne doit pas deplacer les autres : la base et l'origine du
        reseau restent celles estimees."""
        board = self._board_at(700, 400)
        cleaned = outside_ui(board)
        assert np.array_equal(cleaned.e_x, board.e_x)
        assert np.array_equal(cleaned.origin, board.origin)

    def test_an_empty_board_is_handled(self):
        empty = BoardMap(cells=np.empty((0, 2), dtype=np.int64),
                         e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
                         origin=np.array([300.0, 300.0]))
        assert len(outside_ui(empty)) == 0


class TestTheMovementZoneIsReadOnce:
    """`_relocate_from_movement` et `_readable_movement` appelaient chacune
    `read_movement_range(image, board)` -- meme image, meme plateau, meme resultat.

    Mesure : 222 ms l'appel, sur une frame qui en coute 836 au total. Un quart du budget
    de perception rendu deux fois a l'identique, sur CHAQUE frame que le bot observe, pas
    seulement en test. Apres factorisation : 603 ms, et resultats identiques sur les 30
    captures (entites et cases atteignables comparees une a une).

    Ce test compte les appels plutot que de chronometrer : une mesure de duree rougirait
    au gre de la charge de la machine, la sur-consommation, elle, est un fait exact.
    """

    def test_observe_calls_it_exactly_once(self, monkeypatch):
        import jev_tactics.pipeline as module

        chemin = ROOT / "combat1920.png"
        if not chemin.exists():
            pytest.skip("combat1920.png absent")
        frame = cv2.imread(str(chemin))
        board = build_board(frame)
        vrai, appels = module.read_movement_range, []

        def compte(image, plateau):
            appels.append(1)
            return vrai(image, plateau)

        monkeypatch.setattr(module, "read_movement_range", compte)
        module.observe(frame, board)
        assert len(appels) == 1, f"{len(appels)} lectures de la zone de deplacement"

    def test_each_function_still_stands_alone(self):
        """Le parametre reste optionnel : appelees seules, elles calculent la zone
        elles-memes. Sans quoi la factorisation deplacerait le probleme sur leurs
        appelants futurs."""
        chemin = ROOT / "combat1920.png"
        if not chemin.exists():
            pytest.skip("combat1920.png absent")
        frame = cv2.imread(str(chemin))
        board = build_board(frame)
        entities = list(observe(frame, board).entities)
        assert _readable_movement(frame, board, entities) is not None
