"""La boucle d'execution d'une quete, et surtout ce qu'elle fait quand rien n'avance.

Enchainer des succes est trivial. Ce que ces tests gardent, c'est le cas ou une intention
REUSSIT sans faire avancer l'objectif -- on se deplace vers la bonne carte, on y tue le
mauvais monstre. Un compteur d'echecs ne le voit pas, puisque rien n'a echoue, et la
boucle tourne indefiniment en se declarant saine.
"""

from __future__ import annotations

from jev_tactics.quest.director import Act, Intent
from jev_tactics.quest.model import Objective, Progress, Quest, Step
from jev_tactics.quest.runner import Attempt, QuestRunner, accepted_handler

GIVER = (5, 5)
FIELD = (7, 3)


def simple_quest(count: int = 2) -> Quest:
    return Quest(name="q", giver="Aventurier", giver_at=GIVER,
                 objectives=(Objective(Step.KILL, "Bouftou", count=count, where=FIELD),))


class World:
    """Un monde minuscule : une position, qu'un deplacement peut changer."""

    def __init__(self, here=GIVER, travel_works=True):
        self.here = here
        self.travel_works = travel_works
        self.fights = 0

    def locate(self):
        return self.here

    def travel(self, intent: Intent) -> Attempt:
        if not self.travel_works:
            return Attempt(intent, ok=False, note="deplacement sans effet")
        self.here = intent.where
        return Attempt(intent, ok=True)

    def fight(self, intent: Intent) -> Attempt:
        self.fights += 1
        return Attempt(intent, ok=True, progressed=1)

    def fight_wrong_monster(self, intent: Intent) -> Attempt:
        """LE CAS VICIEUX : le combat se gagne, l'objectif ne bouge pas."""
        self.fights += 1
        return Attempt(intent, ok=True, progressed=0, note="ce n'etait pas un Bouftou")


def runner_for(world: World, progress: Progress, **extra) -> QuestRunner:
    handlers = {
        Act.TRAVEL: world.travel,
        Act.FIGHT: world.fight,
        Act.TALK: accepted_handler(progress),
    }
    handlers.update(extra)
    return QuestRunner(handlers, world.locate)


class TestTheHappyPath:
    def test_a_quest_runs_to_completion(self):
        progress = Progress(simple_quest())
        world = World(here=(0, 0))
        report = runner_for(world, progress).run(progress)
        assert report.finished, report.describe()
        assert progress.handed_in
        assert world.fights == 2

    def test_the_report_counts_progress_not_just_steps(self):
        progress = Progress(simple_quest(count=3))
        world = World()
        report = runner_for(world, progress).run(progress)
        assert report.progressed == 3
        assert "quete rendue" in report.describe()


class TestStallDetection:
    """LE COEUR DU MODULE."""

    def test_success_without_progress_is_still_a_stall(self):
        """Le combat se gagne a chaque fois et l'objectif ne bouge jamais. Rien
        n'« echoue », donc un compteur d'echecs laisserait tourner la boucle."""
        progress = Progress(simple_quest())
        world = World()
        runner = runner_for(world, progress, **{Act.FIGHT: world.fight_wrong_monster})
        report = runner.run(progress, max_steps=30)
        assert not report.finished or progress.notes, "aucun pietinement detecte"
        assert any("sans avancement" in n for n in progress.notes)

    def test_the_stall_limit_is_respected(self):
        progress = Progress(simple_quest())
        world = World()
        runner = QuestRunner(
            {Act.TRAVEL: world.travel, Act.FIGHT: world.fight_wrong_monster,
             Act.TALK: accepted_handler(progress)},
            world.locate, stall_limit=3)
        runner.run(progress, max_steps=30)
        # trois tentatives sur l'etape, puis abandon de l'etape
        assert world.fights == 3, f"{world.fights} tentatives au lieu de 3"

    def test_a_failing_travel_does_not_loop_forever(self):
        """Trouve un vrai defaut : `skip()` fait avancer l'index D'OBJECTIF, ce qui ne
        veut rien dire tant que la quete n'est pas acceptee. La boucle sautait des
        objectifs qu'elle n'avait pas le droit de commencer, et redemandait le meme
        trajet impossible a chaque tour."""
        progress = Progress(simple_quest())
        world = World(here=(0, 0), travel_works=False)
        report = runner_for(world, progress).run(progress, max_steps=30)
        assert report.stalled, "le pietinement n'a pas arrete la boucle"
        assert report.steps < 30, "la boucle a consomme toutes ses etapes"
        assert any("impossible d'aller l'accepter" in n for n in progress.notes)

    def test_a_stall_before_acceptance_skips_nothing(self):
        """On ne « saute » pas l'acceptation d'une quete : sans elle, le jeu ne suit
        aucun objectif."""
        progress = Progress(simple_quest())
        world = World(here=(0, 0), travel_works=False)
        runner_for(world, progress).run(progress, max_steps=30)
        assert progress.index == 0, "des objectifs ont ete sautes avant l'acceptation"
        assert not progress.accepted

    def test_a_stall_at_hand_in_abandons_rather_than_skips(self):
        progress = Progress(simple_quest(count=1), accepted=True)
        progress.advance(1)
        world = World(here=(0, 0), travel_works=False)
        report = runner_for(world, progress).run(progress, max_steps=30)
        assert report.stalled
        assert any("impossible d'aller la rendre" in n for n in progress.notes)

    def test_progress_resets_the_stall_counter(self):
        """Un echec isole ne doit pas condamner une etape qui avance par ailleurs."""
        progress = Progress(simple_quest(count=3))
        world = World()
        calls = {"n": 0}

        def flaky(intent):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return Attempt(intent, ok=False, note="rate")
            return Attempt(intent, ok=True, progressed=1)

        report = QuestRunner({Act.TRAVEL: world.travel, Act.FIGHT: flaky,
                              Act.TALK: accepted_handler(progress)},
                             world.locate).run(progress, max_steps=40)
        assert report.finished, report.describe()


class TestMissingCapabilities:
    def test_an_unhandled_act_skips_the_step_out_loud(self):
        """Parler a un PNJ n'est pas executable. On passe EN LE DISANT : un saut
        silencieux se lirait comme un objectif accompli."""
        progress = Progress(simple_quest(), accepted=True)
        world = World(here=FIELD)
        runner = QuestRunner({Act.TRAVEL: world.travel, Act.FIGHT: world.fight},
                             world.locate)
        runner.run(progress, max_steps=20)
        assert any("aucun gestionnaire" in n for n in progress.notes)

    def test_a_blocked_intent_skips_with_the_directors_reason(self):
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.FETCH, "Laine"),))
        progress = Progress(quest, accepted=True)
        world = World()
        QuestRunner({Act.TRAVEL: world.travel}, world.locate).run(progress, max_steps=10)
        assert any("table objet" in n for n in progress.notes)


class TestPerceptionFailureIsNotQuestFailure:
    def test_an_unreadable_position_stops_the_run_distinctly(self):
        """Position illisible : panne de PERCEPTION. Recalibrer et reessayer sont des
        gestes opposes, et les confondre rendait les deux indiscernables."""
        progress = Progress(simple_quest())
        runner = QuestRunner({}, lambda: None)
        report = runner.run(progress)
        assert report.lost
        assert not report.finished
        assert not report.stalled
        assert "illisible" in report.describe()
