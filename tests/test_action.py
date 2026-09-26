"""Couche d'action : delais realistes, trajectoires courbes, execution d'un plan.

Tout passe par DryRunBackend : aucun test ne touche la souris, donc la suite reste
executable en CI sur une machine sans jeu ni ecran."""

import math

import numpy as np
import pytest

from jev_tactics.action import (
    Delay,
    DelaySampler,
    DryRunBackend,
    Executor,
    bezier_path,
    move_along_curve,
)
from jev_tactics.action.timing import MAX_DELAY, MIN_DELAY, PROFILES
from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner import Cast, EndTurn, Move

BOARD = BoardMap(
    cells=np.array([(i, j) for j in range(5) for i in range(5)], dtype=np.int64),
    e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
    origin=np.array([900.0, 500.0]),
)
KEYS = {"trait": "2", "souffle": "3"}


class TestDelays:
    def test_within_bounds(self):
        sampler = DelaySampler(seed=1)
        for kind in Delay:
            for _ in range(200):
                assert MIN_DELAY <= sampler.sample(kind) <= MAX_DELAY

    def test_median_matches_profile(self):
        sampler = DelaySampler(seed=2)
        samples = sorted(sampler.sample(Delay.CLICK) for _ in range(2000))
        median = samples[len(samples) // 2]
        assert median == pytest.approx(PROFILES[Delay.CLICK][0], rel=0.15)

    def test_distribution_is_asymmetric(self):
        """Une log-normale a une queue a DROITE : moyenne > mediane. Une gaussienne,
        signature machine, aurait les deux confondues."""
        sampler = DelaySampler(seed=3)
        samples = [sampler.sample(Delay.BETWEEN_SPELLS) for _ in range(3000)]
        median = sorted(samples)[len(samples) // 2]
        assert sum(samples) / len(samples) > median

    def test_no_two_identical_delays(self):
        """Un sleep constant est la signature la plus facile a reperer."""
        sampler = DelaySampler(seed=4)
        values = [sampler.sample(Delay.CLICK) for _ in range(50)]
        assert len(set(values)) == len(values)

    def test_seed_makes_it_reproducible(self):
        a = [DelaySampler(seed=7).sample(Delay.CLICK) for _ in range(5)]
        b = [DelaySampler(seed=7).sample(Delay.CLICK) for _ in range(5)]
        assert a == b

    def test_profiles_ordered_by_deliberation(self):
        """On hesite plus avant d'ouvrir un tour qu'entre deux clics."""
        assert PROFILES[Delay.TURN_START][0] > PROFILES[Delay.CLICK][0]


class TestBezier:
    def test_ends_exactly_on_target(self):
        assert bezier_path((0, 0), (400, 300))[-1] == (400, 300)

    def test_path_is_not_a_straight_line(self):
        """Le trajet doit s'ecarter de la droite : un segment parfait est artificiel."""
        path = bezier_path((0, 0), (400, 0), rng=__import__("random").Random(5))
        assert max(abs(y) for _, y in path) > 5

    def test_speed_is_not_uniform(self):
        """Lissage en cloche : les pas du milieu sont plus longs que ceux des bords."""
        path = bezier_path((0, 0), (600, 0), steps=30,
                           jitter=0.0, rng=__import__("random").Random(6))
        steps = [math.dist(a, b) for a, b in zip(path, path[1:])]
        assert max(steps) > 2 * min(steps)

    def test_short_distance_is_a_single_hop(self):
        assert len(bezier_path((10, 10), (10, 10))) == 1

    def test_all_points_between_endpoints_roughly(self):
        path = bezier_path((0, 0), (200, 200), rng=__import__("random").Random(8))
        assert all(-80 <= x <= 280 and -80 <= y <= 280 for x, y in path)

    def test_move_along_curve_emits_moves(self):
        backend = DryRunBackend()
        move_along_curve(backend, (300.0, 200.0), start=(0.0, 0.0))
        assert len(backend.moves()) > 5 and backend.moves()[-1] == (300, 200)


class TestExecutor:
    def _executor(self):
        return Executor(board=BOARD, backend=DryRunBackend(), spell_keys=KEYS,
                        delays=DelaySampler(seed=11), seed=11)

    def test_move_clicks_the_cell_centre(self):
        executor = self._executor()
        executor.move_to_cell(7)
        expected = tuple(int(round(v)) for v in BOARD.center(7))
        assert executor.backend.clicks()[-1] == expected

    def test_cast_presses_key_then_clicks_target(self):
        executor = self._executor()
        assert executor.cast("trait", 9) is True
        assert executor.backend.keys() == ["2"]
        assert executor.backend.clicks()[-1] == tuple(
            int(round(v)) for v in BOARD.center(9))

    def test_unknown_spell_is_skipped_not_guessed(self):
        """Sans raccourci connu, on saute l'action plutot que de cliquer au hasard."""
        executor = self._executor()
        assert executor.cast("inconnu", 3) is False
        assert executor.skipped == ["inconnu"] and executor.backend.clicks() == []

    def test_executes_full_plan_in_order(self):
        executor = self._executor()
        executor.execute([Move(cell=6, cost=1), Cast(spell="trait", target=9, cost=3)])
        kinds = [e[0] for e in executor.backend.events]
        assert kinds.count("click") == 2
        assert executor.backend.keys() == ["2", "f1"]   # sort puis fin de tour

    def test_end_turn_action_stops_execution(self):
        executor = self._executor()
        executor.execute([Cast(spell="trait", target=9, cost=3), EndTurn(),
                          Move(cell=1, cost=1)])
        assert executor.backend.clicks() == [tuple(int(round(v))
                                                   for v in BOARD.center(9))]

    def test_delays_are_interleaved(self):
        executor = self._executor()
        executor.execute([Move(cell=6, cost=1), Cast(spell="trait", target=9, cost=3)])
        assert [e for e in executor.backend.events if e[0] == "sleep"]

    def test_dry_run_touches_nothing_real(self):
        """Invariant de securite : le backend par defaut n'agit pas sur la machine."""
        executor = Executor(board=BOARD)
        executor.execute([Move(cell=6, cost=1)])
        assert isinstance(executor.backend, DryRunBackend)

    def test_empty_plan_still_ends_the_turn(self):
        executor = self._executor()
        executor.execute([])
        assert executor.backend.keys() == ["f1"]

    def test_can_skip_ending_the_turn(self):
        executor = self._executor()
        executor.execute([Move(cell=6, cost=1)], finish_turn=False)
        assert executor.backend.keys() == []


class TestModifierKeys:
    """Raccourcis a modificateurs. Les deuxieme et troisieme lignes de la barre sont
    liees a Ctrl+chiffre et Ctrl+Maj+chiffre : sans ce support, deux tiers des sorts d'un
    personnage restent injouables -- et « ctrl+1 » envoye tel quel appuierait sur une
    touche inexistante, donc ne lancerait rien, EN SILENCE."""

    class _Recorder:
        def __init__(self):
            self.events = []

        def press(self, key):
            self.events.append(("press", key))

        def keyDown(self, key):
            self.events.append(("down", key))

        def keyUp(self, key):
            self.events.append(("up", key))

    def _backend(self):
        from jev_tactics.action.mouse import DirectInputBackend

        backend = DirectInputBackend.__new__(DirectInputBackend)
        backend._backend = self._Recorder()
        return backend

    def test_plain_key_needs_no_modifier(self):
        backend = self._backend()
        backend.press_key("3")
        assert backend._backend.events == [("press", "3")]

    def test_ctrl_is_held_around_the_press(self):
        backend = self._backend()
        backend.press_key("ctrl+4")
        assert backend._backend.events == [
            ("down", "ctrl"), ("press", "4"), ("up", "ctrl")]

    def test_several_modifiers_are_released_in_reverse(self):
        """Relacher dans l'ordre inverse de l'appui : c'est ce que fait une main, et
        certains pilotes s'en accommodent mal autrement."""
        backend = self._backend()
        backend.press_key("ctrl+shift+2")
        assert backend._backend.events == [
            ("down", "ctrl"), ("down", "shift"), ("press", "2"),
            ("up", "shift"), ("up", "ctrl")]

    def test_modifiers_are_released_even_if_the_press_fails(self):
        """LE cas qui compte : un modificateur reste enfonce transformerait tous les
        clics suivants en clics modifies -- le bot deviendrait incontrolable."""
        backend = self._backend()
        backend._backend.press = lambda key: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            backend.press_key("ctrl+shift+1")
        assert ("up", "ctrl") in backend._backend.events
        assert ("up", "shift") in backend._backend.events

    def test_case_and_spaces_are_tolerated(self):
        backend = self._backend()
        backend.press_key("CTRL + Shift + 5")
        assert ("press", "5") in backend._backend.events
