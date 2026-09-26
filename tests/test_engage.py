"""L'engagement, verifie sans le jeu.

Le defaut que ces tests gardent n'est pas « le clic rate ». C'est le VERDICT TROP TOT :
conclure a l'echeance qu'un groupe n'en etait pas un, alors que le combat demarrait juste
apres. Il ecrivait deux traces fausses -- un faux negatif au journal, un endroit condamne
pour la session -- et les deux portaient sur les groupes LOINTAINS et GROS, soit ceux
qu'il fallait apprendre a preferer.
"""

from __future__ import annotations

import numpy as np

from jev_tactics.action.mouse import DryRunBackend
from jev_tactics.perception.monsters import MonsterGroup
from jev_tactics.world.engage import (
    MAX_FAILED_SPOTS_PER_MAP,
    Engagement,
    Engager,
    FailedSpots,
)

FRAME = np.zeros((200, 320, 3), dtype=np.uint8)


def group(x: int = 100, y: int = 80) -> MonsterGroup:
    return MonsterGroup(x=x, y=y, area=400, width=20, height=20)


class FakeCombat:
    """La timeline apparait apres `appears_after` lectures. Jamais si None."""

    def __init__(self, appears_after: int | None = 0):
        self.appears_after = appears_after
        self.reads = 0

    def read_timeline(self, frame):
        self.reads += 1
        if self.appears_after is None:
            return []
        return ["combatant"] if self.reads > self.appears_after else []


class SpyBackend(DryRunBackend):
    def __init__(self):
        super().__init__()
        self.slept = 0.0

    def sleep(self, seconds: float) -> None:
        self.slept += seconds


def engager_for(combat: FakeCombat, timeout: float = 8.0):
    backend = SpyBackend()
    return Engager(lambda: FRAME, backend,
                   read_timeline=combat.read_timeline, timeout=timeout), backend


class TestEngaging:
    def test_timeline_appearing_means_the_fight_started(self):
        engager, _ = engager_for(FakeCombat(appears_after=0))
        result = engager.engage(group())
        assert result.outcome is Engagement.STARTED
        assert result.started

    def test_a_click_is_actually_sent_at_the_group(self):
        """La detection par mouvement attrape aussi les autres joueurs : il faut bien
        cliquer la cible pour que la timeline puisse trancher."""
        engager, backend = engager_for(FakeCombat(appears_after=0))
        engager.engage(group(x=123, y=45))
        assert any(e[1:3] == (123, 45) for e in backend.events if len(e) >= 3), backend.events

    def test_a_slow_engagement_still_succeeds(self):
        """Marcher jusqu'a un groupe lointain PLUS l'ecran de placement. Une attente fixe
        de 2,5 s declarait l'echec ; on scrute jusqu'a l'echeance."""
        engager, backend = engager_for(FakeCombat(appears_after=8))
        result = engager.engage(group())
        assert result.outcome is Engagement.STARTED
        assert backend.slept > 0

    def test_a_fast_engagement_pays_no_wait(self):
        engager, backend = engager_for(FakeCombat(appears_after=0))
        engager.engage(group())
        assert backend.slept == 0

    def test_the_waited_delay_is_returned_not_thrown_away(self):
        """`ENGAGE_TIMEOUT` vaut 8 s par ANALOGIE, jamais par mesure. Ce chiffre est
        produit a chaque engagement et n'allait nulle part."""
        engager, _ = engager_for(FakeCombat(appears_after=5))
        assert engager.engage(group()).waited > 0

    def test_the_deadline_bounds_the_wait(self):
        engager, backend = engager_for(FakeCombat(appears_after=None), timeout=2.0)
        result = engager.engage(group())
        assert result.outcome is Engagement.PENDING
        assert backend.slept <= 2.0 + 0.4


class TestTheDeferredVerdict:
    """LE COEUR DU MODULE."""

    def test_a_timeout_is_pending_never_missed(self):
        """Conclure ici condamnait un vrai groupe pour toute la session."""
        engager, _ = engager_for(FakeCombat(appears_after=None), timeout=1.0)
        result = engager.engage(group())
        assert result.outcome is Engagement.PENDING
        assert result.outcome is not Engagement.MISSED
        assert engager.pending is not None

    def test_in_combat_next_cycle_is_late_not_started(self):
        """Rien ne prouve que ce combat vienne de ce clic : un groupe qui passe peut
        agresser le personnage, et il marchait justement vers la cible."""
        engager, _ = engager_for(FakeCombat(appears_after=None), timeout=1.0)
        engager.engage(group())
        settled = engager.settle(in_combat=True)
        assert settled is not None
        assert settled.outcome is Engagement.LATE
        assert settled.outcome is not Engagement.STARTED

    def test_not_in_combat_next_cycle_confirms_the_miss(self):
        engager, _ = engager_for(FakeCombat(appears_after=None), timeout=1.0)
        engager.engage(group())
        settled = engager.settle(in_combat=False)
        assert settled is not None and settled.outcome is Engagement.MISSED

    def test_settling_clears_the_pending_group(self):
        engager, _ = engager_for(FakeCombat(appears_after=None), timeout=1.0)
        engager.engage(group())
        engager.settle(in_combat=False)
        assert engager.pending is None
        assert engager.settle(in_combat=False) is None

    def test_a_success_leaves_nothing_pending(self):
        engager, _ = engager_for(FakeCombat(appears_after=0))
        engager.engage(group())
        assert engager.pending is None


class TestFailedSpots:
    def test_a_remembered_spot_is_recognised_nearby(self):
        spots = FailedSpots()
        spots.remember("map_5_5", 100, 100)
        assert spots.is_known_bad("map_5_5", 105, 98)

    def test_a_distant_spot_is_not_confused_with_it(self):
        spots = FailedSpots()
        spots.remember("map_5_5", 100, 100)
        assert not spots.is_known_bad("map_5_5", 400, 400)

    def test_spots_do_not_leak_between_maps(self):
        spots = FailedSpots()
        spots.remember("map_5_5", 100, 100)
        assert not spots.is_known_bad("map_6_5", 100, 100)

    def test_the_list_is_capped_and_drops_the_oldest(self):
        """Sans plafond la memoire finit par AVEUGLER la carte : 14,9 % de l'ecran en onze
        passages, sans saturation."""
        spots = FailedSpots()
        for i in range(MAX_FAILED_SPOTS_PER_MAP + 5):
            spots.remember("map_5_5", i * 100, 0)
        assert len(spots.per_map["map_5_5"]) == MAX_FAILED_SPOTS_PER_MAP
        # le tout premier est parti, le dernier est la
        assert not spots.is_known_bad("map_5_5", 0, 0)
        assert spots.is_known_bad("map_5_5", (MAX_FAILED_SPOTS_PER_MAP + 4) * 100, 0)

    def test_changing_map_can_clear_what_was_learnt_there(self):
        spots = FailedSpots()
        spots.remember("map_5_5", 100, 100)
        spots.forget_map("map_5_5")
        assert not spots.is_known_bad("map_5_5", 100, 100)
