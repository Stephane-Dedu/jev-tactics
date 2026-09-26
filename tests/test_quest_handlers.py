"""Les gestionnaires reels : ce qui relie le directeur aux couches qui agissent.

Les deux proprietes gardees ici sont des refus, pas des succes :

  - un deplacement ne fait avancer AUCUN objectif, meme reussi ;
  - un combat entierement muet n'est pas un combat gagne, quoi qu'en dise la timeline.

Les deux feraient progresser une quete sur une panne.
"""

from __future__ import annotations

from jev_tactics.bot.fight import FightReport, Outcome
from jev_tactics.perception.coordinates import MapPosition
from jev_tactics.perception.monsters import MonsterGroup
from jev_tactics.quest.director import Act, Intent
from jev_tactics.quest.handlers import fight_handler, travel_handler
from jev_tactics.world.engage import Engagement, EngageResult
from jev_tactics.world.navigation import Direction
from jev_tactics.world.travel import Move, Verdict
from jev_tactics.world.walker import WalkReport


class FakeWalker:
    def __init__(self, arrived=True, drifts=0):
        self.arrived = arrived
        self.drifts = drifts
        self.asked: list = []

    def travel_to(self, goal, max_steps=30):
        self.asked.append(goal)
        verdicts = []
        for _ in range(self.drifts):
            verdicts.append(Verdict(Move.DRIFTED, Direction.RIGHT,
                                    MapPosition(x=0, y=0), MapPosition(x=0, y=1), (1, 0)))
        return WalkReport(verdicts=verdicts, arrived=self.arrived)


class FakeEngager:
    def __init__(self, started=True):
        self.started = started

    def engage(self, group):
        outcome = Engagement.STARTED if self.started else Engagement.PENDING
        return EngageResult(outcome, group, waited=1.2)

    def settle(self, in_combat):
        return None


def group() -> MonsterGroup:
    return MonsterGroup(x=100, y=80, area=400, width=20, height=20)


def travel_intent() -> Intent:
    return Intent(Act.TRAVEL, where=(7, 3))


def fight_intent() -> Intent:
    return Intent(Act.FIGHT, target="Bouftou", count=2)


class TestTravel:
    def test_arriving_succeeds_but_advances_nothing(self):
        """Un deplacement rend le prochain objectif JOUABLE, il ne l'accomplit pas.
        Le compter ferait passer un aller-retour sans fin pour du progres."""
        attempt = travel_handler(FakeWalker())(travel_intent())
        assert attempt.ok
        assert attempt.progressed == 0

    def test_not_arriving_fails(self):
        attempt = travel_handler(FakeWalker(arrived=False))(travel_intent())
        assert not attempt.ok
        assert attempt.progressed == 0

    def test_drifts_are_surfaced(self):
        """La derive est le seul signe que le graphe et le jeu divergent."""
        attempt = travel_handler(FakeWalker(drifts=2))(travel_intent())
        assert "derive" in attempt.note

    def test_an_intent_without_a_map_is_refused_not_attempted(self):
        walker = FakeWalker()
        attempt = travel_handler(walker)(Intent(Act.TRAVEL, where=None))
        assert not attempt.ok
        assert walker.asked == [], "un trajet a ete tente sans destination"


class TestFight:
    @staticmethod
    def report(turns=3, mute=0, outcome=Outcome.ENDED) -> FightReport:
        return FightReport(outcome=outcome, turns=turns, mute_turns=mute,
                           spells_cast=0 if mute == turns else turns)

    def test_a_won_fight_advances(self):
        handler = fight_handler(FakeEngager(), group, lambda: self.report())
        attempt = handler(fight_intent())
        assert attempt.ok and attempt.progressed == 1

    def test_no_group_is_a_failure_not_a_crash(self):
        handler = fight_handler(FakeEngager(), lambda: None, lambda: self.report())
        attempt = handler(fight_intent())
        assert not attempt.ok
        assert "aucun groupe" in attempt.note

    def test_an_unconfirmed_engagement_does_not_advance(self):
        """PENDING, pas MISSED : le combat peut demarrer juste apres l'echeance, et
        l'endroit ne doit surtout pas etre condamne ici."""
        handler = fight_handler(FakeEngager(started=False), group,
                                lambda: self.report())
        attempt = handler(fight_intent())
        assert not attempt.ok
        assert "non confirme" in attempt.note

    def test_a_completely_mute_fight_is_not_a_win(self):
        """LE REFUS QUI COMPTE. Sans lui, une barre de sorts mal decrite ferait
        progresser la quete a chaque combat « gagne » sans un seul sort lance."""
        handler = fight_handler(FakeEngager(), group,
                                lambda: self.report(turns=4, mute=4))
        attempt = handler(fight_intent())
        assert not attempt.ok
        assert attempt.progressed == 0
        assert "AUCUN SORT" in attempt.note

    def test_a_partly_mute_fight_still_counts(self):
        """Un tour muet isole est normal : rien a portee, tout en rechargement."""
        handler = fight_handler(FakeEngager(), group,
                                lambda: self.report(turns=4, mute=1))
        assert handler(fight_intent()).ok

    def test_a_blind_fight_does_not_advance(self):
        handler = fight_handler(FakeEngager(), group,
                                lambda: self.report(turns=0, outcome=Outcome.BLIND))
        assert not handler(fight_intent()).ok

    def test_the_kill_count_is_configurable_and_conservative(self):
        """Sous-estimer refait un combat de trop ; surestimer envoie rendre une quete
        non finie. La premiere erreur coute des minutes, la seconde casse la quete."""
        default = fight_handler(FakeEngager(), group, lambda: self.report())
        assert default(fight_intent()).progressed == 1
        generous = fight_handler(FakeEngager(), group, lambda: self.report(),
                                 kills_per_fight=3)
        assert generous(fight_intent()).progressed == 3
