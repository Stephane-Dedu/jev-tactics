"""Le deroule d'une quete, verifie sans le jeu.

C'est tout l'interet d'un directeur PUR : une quete entiere -- aller chercher, voyager,
tuer, revenir, rendre -- se joue ici en quelques millisecondes, et une erreur d'ordre se
voit dans un test au lieu de se decouvrir apres quarante minutes de session.
"""

from __future__ import annotations

import pytest

from jev_tactics.quest.director import EXECUTABLE, Act, next_intent, plan_ahead
from jev_tactics.quest.model import Objective, Progress, Quest, Step

GIVER = (5, 5)
FIELD = (7, 3)
ELDER = (2, 8)


def fetch_quest() -> Quest:
    """Le cas complet : un donneur, un objectif ailleurs, et un AUTRE PNJ pour rendre."""
    return Quest(
        name="Les bouftous du champ",
        giver="Aventurier",
        giver_at=GIVER,
        objectives=(
            Objective(Step.KILL, "Bouftou", count=3, where=FIELD),
        ),
        turn_in="Ancien",
        turn_in_at=ELDER,
    )


class TestTheQuestMustBeAcceptedFirst:
    """Remplir les objectifs sans avoir parle au donneur ne compte pas : le jeu ne suit
    rien tant que le dialogue n'a pas eu lieu. L'erreur ne se voit qu'a la fin, quand le
    PNJ ne propose pas de rendre."""

    def test_it_travels_to_the_giver_first(self):
        intent = next_intent(Progress(fetch_quest()), here=(0, 0))
        assert intent.act is Act.TRAVEL
        assert intent.where == GIVER

    def test_on_the_spot_it_talks(self):
        intent = next_intent(Progress(fetch_quest()), here=GIVER)
        assert intent.act is Act.TALK
        assert intent.target == "Aventurier"

    def test_objectives_are_not_attempted_before_acceptance(self):
        """Meme debout sur la carte de l'objectif, il faut d'abord accepter."""
        intent = next_intent(Progress(fetch_quest()), here=FIELD)
        assert intent.act is Act.TRAVEL
        assert intent.where == GIVER


class TestObjectives:
    def test_it_travels_to_where_the_objective_is(self):
        p = Progress(fetch_quest(), accepted=True)
        intent = next_intent(p, here=GIVER)
        assert intent.act is Act.TRAVEL and intent.where == FIELD

    def test_on_the_spot_it_fights(self):
        p = Progress(fetch_quest(), accepted=True)
        intent = next_intent(p, here=FIELD)
        assert intent.act is Act.FIGHT
        assert intent.target == "Bouftou"
        assert intent.count == 3

    def test_the_count_is_what_remains_not_the_total(self):
        p = Progress(fetch_quest(), accepted=True)
        p.advance(2)
        assert next_intent(p, here=FIELD).count == 1

    def test_an_objective_without_a_map_is_attempted_here(self):
        """`where=None` veut dire « n'importe ou », pas « on ne sait pas ou ». Confondre
        les deux ferait soit errer le bot, soit l'arreter sur une etape jouable ici."""
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.KILL, "Pious", count=2),))
        intent = next_intent(Progress(quest, accepted=True), here=(99, 99))
        assert intent.act is Act.FIGHT


class TestHandingIn:
    def test_it_returns_to_the_right_npc_not_the_giver(self):
        """« finir en parlant au BON PNJ » : c'est le cas courant, pas l'exception."""
        p = Progress(fetch_quest(), accepted=True)
        p.advance(3)
        intent = next_intent(p, here=FIELD)
        assert intent.act is Act.TRAVEL
        assert intent.where == ELDER, "retour au donneur au lieu du PNJ de rendu"

    def test_it_talks_to_the_turn_in_npc_on_arrival(self):
        p = Progress(fetch_quest(), accepted=True)
        p.advance(3)
        intent = next_intent(p, here=ELDER)
        assert intent.act is Act.TALK and intent.target == "Ancien"

    def test_a_quest_without_a_turn_in_returns_to_the_giver(self):
        quest = Quest(name="q", giver="Aventurier", giver_at=GIVER,
                      objectives=(Objective(Step.KILL, "Pious"),))
        p = Progress(quest, accepted=True)
        p.advance(1)
        assert next_intent(p, here=GIVER).target == "Aventurier"

    def test_a_satisfied_goto_is_step_done_not_done(self):
        """Deux sens pour un nom : un appelant ecrivant `if act is DONE: stop()` arretait
        la quete au milieu, sur une etape de deplacement satisfaite."""
        quest = Quest(name="q", giver="g", giver_at=GIVER, objectives=(
            Objective(Step.GOTO, where=FIELD),
            Objective(Step.KILL, "Bouftou"),
        ))
        intent = next_intent(Progress(quest, accepted=True), here=FIELD)
        assert intent.act is Act.STEP_DONE
        assert intent.act is not Act.DONE

    def test_once_handed_in_there_is_nothing_left(self):
        p = Progress(fetch_quest(), accepted=True, handed_in=True)
        assert next_intent(p, here=ELDER).act is Act.DONE


class TestProgressCounting:
    def test_overkill_does_not_spill_into_the_next_objective(self):
        """Tuer trois monstres d'un sort de zone quand il en fallait deux ne doit pas
        faire sauter l'etape suivante : le surplus est perdu, comme dans le jeu."""
        quest = Quest(name="q", giver="g", giver_at=GIVER, objectives=(
            Objective(Step.KILL, "A", count=2),
            Objective(Step.KILL, "B", count=2),
        ))
        p = Progress(quest, accepted=True)
        p.advance(5)
        assert p.index == 1
        assert p.counted == 0
        assert p.current is not None and p.current.target == "B"

    def test_advancing_past_the_end_is_harmless(self):
        p = Progress(fetch_quest(), accepted=True)
        p.advance(99)
        p.advance(99)
        assert p.objectives_done

    def test_skipping_records_why(self):
        """Un saut silencieux se lirait comme un objectif accompli."""
        p = Progress(fetch_quest(), accepted=True)
        p.skip("PNJ introuvable")
        assert p.objectives_done
        assert any("PNJ introuvable" in n for n in p.notes)


class TestWhatIsNotExecutableSaysSo:
    """Le manque est declare DANS LE CODE, pas dans un document."""

    def test_talking_is_not_executable_yet(self):
        intent = next_intent(Progress(fetch_quest()), here=GIVER)
        assert intent.act is Act.TALK
        assert not intent.implemented
        assert "perception manquante" in intent.describe()

    def test_travel_and_fight_are(self):
        p = Progress(fetch_quest(), accepted=True)
        assert next_intent(p, here=GIVER).implemented      # travel
        assert next_intent(p, here=FIELD).implemented      # fight

    def test_the_executable_set_matches_the_layers_that_exist(self):
        """Echouera le jour ou la perception des PNJ arrivera sans que le directeur soit
        mis a jour -- exactement le moment ou il faut y penser."""
        assert Act.TALK not in EXECUTABLE, (
            "TALK est declare executable : la perception des PNJ existe-t-elle vraiment ? "
            "Si oui, retirer cette assertion et brancher le dialogue.")
        assert {Act.TRAVEL, Act.FIGHT}.issubset(EXECUTABLE)

    def test_a_fetch_objective_is_blocked_with_a_reason(self):
        """Un objet s'obtient en tuant ou en recoltant ; sans table objet -> source, on le
        DIT plutot que de deviner et d'envoyer le bot au mauvais endroit."""
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.FETCH, "Laine", count=4),))
        intent = next_intent(Progress(quest, accepted=True), here=GIVER)
        assert intent.act is Act.BLOCKED
        assert "table objet" in intent.why

    def test_a_goto_without_a_map_is_blocked(self):
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.GOTO),))
        intent = next_intent(Progress(quest, accepted=True), here=GIVER)
        assert intent.act is Act.BLOCKED


class TestThePlanReadsEndToEnd:
    def test_the_whole_quest_unrolls_in_the_right_order(self):
        """LE TEST « BOUT EN BOUT » : prendre, voyager, tuer, revenir, rendre."""
        trace = plan_ahead(Progress(fetch_quest()), here=(0, 0))
        acts = [i.act for i in trace]
        assert acts == [
            Act.TRAVEL,   # vers le donneur
            Act.TALK,     # accepter
            Act.TRAVEL,   # vers le champ
            Act.FIGHT,    # les bouftous
            Act.TRAVEL,   # vers l'Ancien
            Act.TALK,     # rendre
            Act.DONE,
        ], [i.describe() for i in trace]

    def test_a_blocked_step_stops_the_reading(self):
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.FETCH, "Laine"),))
        p = Progress(quest, accepted=True)
        trace = plan_ahead(p, here=GIVER)
        assert trace[-1].act is Act.BLOCKED

    def test_reading_never_loops_forever(self):
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.KILL, "A", count=99),))
        assert len(plan_ahead(Progress(quest, accepted=True), here=GIVER, limit=5)) <= 5


class TestTheModelRefusesNonsense:
    def test_a_quest_without_objectives_is_rejected(self):
        with pytest.raises(ValueError, match="sans objectif"):
            Quest(name="vide", giver="g", giver_at=GIVER, objectives=())

    def test_a_turn_in_npc_without_a_map_is_caught(self):
        quest = Quest(name="q", giver="g", giver_at=GIVER,
                      objectives=(Objective(Step.KILL, "A"),),
                      turn_in="Ailleurs", turn_in_at=None)
        with pytest.raises(AssertionError, match="carte est inconnue"):
            quest.handed_to()
