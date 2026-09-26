"""La boucle de combat, jouee sans le jeu.

Ce que ces tests gardent n'est pas « le combat se deroule ». C'est la COMPTABILITE : un
plan reduit a « fin de tour » s'execute sans erreur et marque le tour joue. Trente tours
muets se lisaient donc comme trente tours joues, et le travail associe a « limite de tours
atteinte » aurait ete de lever la borne -- soit soixante tours muets au lieu de trente.
"""

from __future__ import annotations

import numpy as np

from jev_tactics.action.actions import Executor
from jev_tactics.action.mouse import DryRunBackend
from jev_tactics.bot.fight import Outcome, play_fight
from jev_tactics.decision import Decision
from jev_tactics.planner.legal import Cast
from jev_tactics.planner.search import Plan
from jev_tactics.sim import square_board
from jev_tactics.state import CombatState, Entity, Team

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def a_state(our_turn: bool = True) -> CombatState:
    return CombatState(turn=1, is_our_turn=our_turn, entities=[
        Entity(entity_id="me", team=Team.ALLY, cell=0, hp=100, hp_max=100,
               ap=6, mp=3, is_self=True),
        Entity(entity_id="e0", team=Team.ENEMY, cell=5, hp=60, hp_max=60, ap=6, mp=3),
    ])


class Decider:
    """Rend un plan fixe. `mute=True` reproduit le plan « fin de tour »."""

    def __init__(self, mute: bool = False, spell: str = "trait", fallback: bool = False):
        self.mute = mute
        self.spell = spell
        self.fallback = fallback

    def decide(self, board, state, spells) -> Decision:
        actions = [] if self.mute else [Cast(spell=self.spell, target=5, cost=3)]
        return Decision(plan=Plan(actions=actions, score=1.0),
                        source="fallback:test" if self.fallback else "search",
                        reason="test")


class Reader:
    """Rend une suite d'etats scriptee, puis None."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def __call__(self, frame, turn, previous):
        self.calls += 1
        return self.states.pop(0) if self.states else None


def executor_for(board, keys=None) -> Executor:
    return Executor(board=board, backend=DryRunBackend(),
                    spell_keys=keys if keys is not None else {"trait": "1"}, seed=0)


def board():
    return square_board(5)


class TestMuteTurnsAreCountedApart:
    """LE DEFAUT QUE CE MODULE EXISTE POUR RENDRE VISIBLE."""

    def test_a_fight_with_no_spell_at_all_says_so(self):
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 5),
                            Decider(mute=True), executor_for(b), max_turns=5)
        assert report.turns == 5
        assert report.mute_turns == 5
        assert report.spells_cast == 0
        assert "AUCUN SORT LANCE" in report.describe()

    def test_it_is_not_reported_as_a_turn_limit(self):
        """« limite de tours atteinte » ferait lever la borne : deux fois plus de tours
        muets au lieu d'une barre de sorts verifiee."""
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 5),
                            Decider(mute=True), executor_for(b), max_turns=5)
        assert "limite" not in report.describe().lower()

    def test_a_normal_fight_counts_its_spells(self):
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 3),
                            Decider(), executor_for(b), max_turns=3)
        assert report.spells_cast == 3
        assert report.mute_turns == 0
        assert "AUCUN SORT" not in report.describe()

    def test_a_single_mute_turn_is_only_an_observation(self):
        """Rien a portee, tout en rechargement : c'est normal et ne doit pas alerter."""
        b = board()
        mute_then_active = [a_state()] * 4
        deciders = [Decider(mute=True)] + [Decider()] * 3

        class Alternating:
            def __init__(self): self.i = 0
            def decide(self, *a):
                d = deciders[min(self.i, len(deciders) - 1)].decide(*a)
                self.i += 1
                return d

        report = play_fight(b, [], lambda: FRAME, Reader(mute_then_active),
                            Alternating(), executor_for(b), max_turns=4)
        assert report.mute_turns == 1
        assert "beaucoup" not in report.describe()
        assert "AUCUN SORT" not in report.describe()

    def test_the_ratio_warns_above_a_half(self):
        b = board()
        deciders = [Decider(mute=True)] * 3 + [Decider()]

        class Mostly:
            def __init__(self): self.i = 0
            def decide(self, *a):
                d = deciders[min(self.i, len(deciders) - 1)].decide(*a)
                self.i += 1
                return d

        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 4),
                            Mostly(), executor_for(b), max_turns=4)
        assert report.mute_ratio > 0.5
        assert "beaucoup" in report.describe()


class TestSkippedSpellsSurface:
    def test_a_spell_without_a_shortcut_is_reported(self):
        """Un slot errone retire le sort en silence -- le defaut le plus difficile a
        diagnostiquer du projet. L'executeur le sait ; encore faut-il le remonter."""
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 2),
                            Decider(spell="inconnu"), executor_for(b, keys={}),
                            max_turns=2)
        assert "inconnu" in report.skipped_spells
        assert "sans raccourci" in report.describe() or report.mute_turns == 0


class TestReadingFailures:
    def test_one_unreadable_frame_does_not_end_the_fight(self):
        """Une frame de transition est illisible sans que le combat soit fini. La declarer
        terminee renvoyait le bot sur son circuit en pleine bagarre."""
        b = board()
        reader = Reader([a_state(), None, a_state(), a_state()])
        report = play_fight(b, [], lambda: FRAME, reader, Decider(),
                            executor_for(b), max_turns=6, blind_limit=3)
        assert report.turns >= 2, "un hoquet de lecture a interrompu le combat"

    def test_repeated_blindness_ends_it(self):
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()]), Decider(),
                            executor_for(b), max_turns=10, blind_limit=3)
        assert report.outcome is Outcome.ENDED
        assert report.turns == 1

    def test_blind_from_the_start_is_a_perception_failure_not_a_finished_fight(self):
        """Zero tour joue : il n'y a pas eu de combat, il y a eu une panne de lecture.
        Les confondre ferait passer un bot aveugle pour un bot qui a gagne vite."""
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([]), Decider(),
                            executor_for(b), max_turns=10, blind_limit=3)
        assert report.outcome is Outcome.BLIND
        assert report.turns == 0


class TestOtherBookkeeping:
    def test_it_waits_instead_of_playing_out_of_turn(self):
        b = board()
        reader = Reader([a_state(our_turn=False), a_state(our_turn=False), a_state()])
        report = play_fight(b, [], lambda: FRAME, reader, Decider(),
                            executor_for(b), max_turns=5)
        assert report.turns == 1, "le bot a joue hors de son tour"

    def test_fallbacks_are_counted(self):
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 3),
                            Decider(fallback=True), executor_for(b), max_turns=3)
        assert report.fallbacks == 3
        assert "repli" in report.describe()

    def test_the_turn_limit_is_reported_as_such(self):
        b = board()
        report = play_fight(b, [], lambda: FRAME, Reader([a_state()] * 20),
                            Decider(), executor_for(b), max_turns=4)
        assert report.outcome is Outcome.EXHAUSTED
        assert report.turns == 4
