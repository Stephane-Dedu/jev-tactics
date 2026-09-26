"""Les postures : un plan par intention, et surtout des plans DIFFERENTS.

Ce module existe a cause d'une mesure. `top_sequences` rendait les K meilleurs plans ;
sur `sacrieur.json`, 16 options sur 24 etaient redondantes -- memes sorts, memes cibles,
degats 93 contre 93, seule la case d'arrivee differait. Les tests ci-dessous gardent la
propriete qui manquait : deux options proposees ne doivent jamais se decrire pareil.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jev_tactics.planner.candidates import Candidate
from jev_tactics.planner.postures import (
    POSTURES,
    REFERENCE,
    Posture,
    distinct_enough,
    posture_plans,
)
from jev_tactics.planner.scoring import DEFAULT_WEIGHTS, Weights
from jev_tactics.planner.search import Plan, best_sequence
from jev_tactics.rules.spells import load_spells
from jev_tactics.sim import square_board
from jev_tactics.sim.arena import random_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spells():
    return load_spells(ROOT / "configs" / "spells" / "sacrieur.json")


@pytest.fixture(scope="module")
def board():
    return square_board(9)


def constrained(board, seed: int):
    """Le regime ou les postures ont un sens : peu de PA, peu de PV.

    A 10 PA et 200 PV le personnage ecrase tout et le meilleur plan est le meme sous
    toutes les intentions -- MESURE : 17 tours sur 40 n'offrent qu'une option. Le dilemme
    apparait quand les ressources manquent, ce qui est aussi quand la decision compte.
    """
    return random_scenario(board, random.Random(seed), enemies=2, hp=60, ap=6, mp=3)


class TestTheReferenceIsTheFallback:
    """L'invariant du repli : sans lui, retomber sur le solveur ne vaudrait plus les
    60 % mesures."""

    def test_the_reference_comes_first(self, board, spells):
        for seed in range(8):
            plans = posture_plans(board, constrained(board, seed), spells)
            if plans:
                assert plans[0][0].name == "reference"

    def test_the_reference_is_the_solver_plan(self, board, spells):
        """Renoter sous DEFAULT_WEIGHTS doit redonner l'argmax du solveur."""
        agree = 0
        for seed in range(12):
            state = constrained(board, seed)
            plans = posture_plans(board, state, spells)
            if not plans:
                continue
            solver = best_sequence(board, state, spells)
            mine = plans[0][1].plan
            if _shape(solver) == _shape(mine):
                agree += 1
        assert agree >= 11, f"la reference s'ecarte du solveur ({agree}/12)"


class TestOptionsAreDistinguishable:
    """LE DEFAUT CORRIGE. C'est la seule propriete qui compte : un modele ne peut pas
    trancher entre deux descriptions identiques."""

    def test_no_two_postures_share_a_signature(self, board, spells):
        for seed in range(15):
            plans = posture_plans(board, constrained(board, seed), spells)
            sigs = [c.signature() for _, c in plans]
            assert len(sigs) == len(set(sigs)), f"doublon a la graine {seed}"

    def test_a_constrained_turn_offers_a_real_choice(self, board, spells):
        """Dans le regime contraint, presque tout tour doit offrir au moins deux
        intentions. MESURE : 0 tour sur 40 n'en offrait qu'une."""
        single = sum(1 for seed in range(20)
                     if len(posture_plans(board, constrained(board, seed), spells)) < 2)
        assert single <= 4, f"{single}/20 tours sans choix : les postures ne separent plus"

    def test_at_most_one_plan_per_posture(self, board, spells):
        plans = posture_plans(board, constrained(board, 1), spells)
        names = [p.name for p, _ in plans]
        assert len(names) == len(set(names))
        assert len(plans) <= len(POSTURES)


class TestTheWeightsActuallySteer:
    """Si changer la ponderation ne changeait pas le plan, les postures seraient un
    habillage -- exactement le defaut d'origine sous un autre nom."""

    def test_pure_damage_ignores_distance(self, board, spells):
        """`degats` neutralise prudence, prime de mise a mort et penalites : son plan doit
        infliger au moins autant de degats que la reference."""
        wins = 0
        for seed in range(12):
            plans = dict((p.name, c) for p, c in
                         posture_plans(board, constrained(board, seed), spells))
            if "degats" in plans and "reference" in plans:
                if plans["degats"].total_damage >= plans["reference"].total_damage:
                    wins += 1
        assert wins >= 1, "la posture degats n'a jamais frappe plus fort"

    def test_shelter_ends_further_away(self, board, spells):
        """`abri` doit, au moins parfois, finir plus loin que la reference. Sinon le
        terme de prudence ne pese rien et la posture est decorative."""
        further = 0
        for seed in range(15):
            state = constrained(board, seed)
            plans = dict((p.name, c) for p, c in posture_plans(board, state, spells))
            if "abri" not in plans or "reference" not in plans:
                continue
            if plans["abri"].signature()[0] != plans["reference"].signature()[0]:
                further += 1
        assert further >= 1, "abri n'a jamais choisi une autre case que la reference"

    def test_every_posture_has_a_distinct_weighting(self):
        seen = {p.weights for p in POSTURES}
        assert len(seen) == len(POSTURES), "deux postures partagent leur ponderation"

    def test_the_reference_keeps_the_default_weighting(self):
        """Elle DOIT rester le defaut : c'est ce qui garde la mesure comparable."""
        assert REFERENCE.weights == DEFAULT_WEIGHTS


class TestSkippingTheCall:
    def test_one_option_is_not_enough_to_ask(self):
        assert not distinct_enough([(REFERENCE, _fake())])

    def test_none_is_not_enough_either(self):
        assert not distinct_enough([])

    def test_two_options_are(self):
        other = Posture("autre", "x", Weights(safety=9.0))
        assert distinct_enough([(REFERENCE, _fake()), (other, _fake())])


class TestRescoringWithoutATurn:
    def test_a_candidate_without_turn_keeps_its_score(self, board, spells):
        """Le repli d'approche est fabrique hors recherche et n'a pas de TurnState.
        L'ecarter rendrait une liste vide la ou c'est le seul plan disponible."""
        from jev_tactics.planner.postures import _rescore

        cand = _fake(score=12.5)
        state = constrained(board, 0)
        assert _rescore(board, state, spells, cand, Weights(safety=99.0)) == 12.5


def _fake(score: float = 1.0) -> Candidate:
    return Candidate(plan=Plan(actions=[], score=score), damage={}, turn=None)


def _shape(plan: Plan):
    return [(type(a).__name__, getattr(a, "spell", None),
             getattr(a, "cell", getattr(a, "target", None))) for a in plan.actions]
