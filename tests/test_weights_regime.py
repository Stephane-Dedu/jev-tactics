"""Quels poids agissent REELLEMENT, dans chaque regime de PV.

Ce fichier consigne une mesure inattendue : **deux des six poids ne font rien dans le
regime que le bot joue.**

`evaluate` se scinde en deux branches selon `entity.hp_known`. Toute la partie riche --
plafonnement des degats utiles, prime de mise a mort -- vit dans la branche « PV connus »,
et le `continue` de l'autre branche la saute entierement. Or sur 32 captures reelles,
72 ennemis detectes et **zero** avec des PV lisibles : la branche riche n'est jamais prise
en jeu.

Consequence : `KILL_BONUS`, le poids le plus documente du depot -- « le terme qui, seul,
fait passer le taux de victoire de 0,6 % a 81 % » -- est **inerte en production**. Idem
pour `WOUNDED_WEIGHT`, dont le facteur `(1 - hp/hp_max)` vaut zero tant que les PV sont un
remplissage.

Ce n'est pas un defaut a corriger ici : c'est un fait a NE PAS OUBLIER. Regler `KILL_BONUS`
sur des mesures d'arene, puis s'etonner que le bot joue mal, est le piege que ces tests
ferment. Ils echoueront le jour ou la lecture des PV ennemis arrivera -- et ce jour-la,
tous les poids seront a remesurer.
"""

from __future__ import annotations

import pytest

from jev_tactics.planner.legal import TurnState
from jev_tactics.planner.scoring import Weights, evaluate
from jev_tactics.sim import square_board
from jev_tactics.state import CombatState, Entity, Team


@pytest.fixture(scope="module")
def board():
    return square_board(9)


def state(board, hp_known: bool) -> CombatState:
    return CombatState(turn=1, entities=[
        Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(4, 4),
               hp=60, hp_max=60, ap=6, mp=3, is_self=True),
        Entity(entity_id="a", team=Team.ENEMY, cell=board.index_at(4, 1),
               hp=60, hp_max=60, ap=6, mp=3, hp_known=hp_known),
    ])


def turn(board) -> TurnState:
    return TurnState(cell=board.index_at(4, 3), ap=0, mp=0)


LETHAL = {"a": 60.0}   # de quoi tuer la cible


class TestWhatIsDeadInTheRealRegime:
    """PV ennemis inconnus : le seul regime que le bot rencontre."""

    def test_the_kill_bonus_does_nothing(self, board):
        """LE POIDS LE PLUS DOCUMENTE DU DEPOT, ET IL EST INERTE EN JEU.

        « le terme qui, seul, fait passer le taux de victoire de 0,6 % a 81 % » -- mesure
        en arene, a PV connus. En jeu, la branche qui l'applique n'est jamais atteinte.
        """
        s = state(board, hp_known=False)
        zero = evaluate(board, s, turn(board), LETHAL, weights=Weights(kill_bonus=0.0))
        huge = evaluate(board, s, turn(board), LETHAL, weights=Weights(kill_bonus=600.0))
        assert zero == huge, (
            "KILL_BONUS agit a PV inconnus : la branche a change, tous les poids sont "
            "a remesurer")

    def test_the_wounded_bonus_does_nothing_either(self, board):
        """Son facteur `(1 - hp/hp_max)` vaut zero tant que les PV sont un remplissage."""
        s = state(board, hp_known=False)
        zero = evaluate(board, s, turn(board), LETHAL, weights=Weights(wounded=0.0))
        huge = evaluate(board, s, turn(board), LETHAL, weights=Weights(wounded=400.0))
        assert zero == huge

    def test_safety_is_the_one_that_bites(self, board):
        """Le seul differenciateur vivant -- ce qui explique pourquoi une ponderation qui
        l'annule (`degats`) bat celle reglee a la main dans ce regime."""
        s = state(board, hp_known=False)
        low = evaluate(board, s, turn(board), LETHAL, weights=Weights(safety=0.0))
        high = evaluate(board, s, turn(board), LETHAL, weights=Weights(safety=12.0))
        assert high > low


class TestWhatIsAliveInTheArena:
    """PV connus : le regime ou tous les reglages ont ete faits."""

    def test_the_kill_bonus_is_alive_here(self, board):
        s = state(board, hp_known=True)
        zero = evaluate(board, s, turn(board), LETHAL, weights=Weights(kill_bonus=0.0))
        huge = evaluate(board, s, turn(board), LETHAL, weights=Weights(kill_bonus=600.0))
        assert huge > zero, "meme a PV connus la prime ne fait rien : verifier evaluate"

    def test_the_two_regimes_do_not_agree(self, board):
        """La meme ponderation, le meme coup, deux scores : c'est l'ecart qui rend les
        mesures d'arene non transposables."""
        w = Weights()
        known = evaluate(board, state(board, True), turn(board), LETHAL, weights=w)
        unknown = evaluate(board, state(board, False), turn(board), LETHAL, weights=w)
        assert known != unknown


class TestTheDocumentationMatchesTheCode:
    def test_kill_bonus_carries_a_warning_about_its_regime(self):
        """Le poids le plus documente doit dire OU il agit, sinon son commentaire
        continuera de faire regler un terme mort."""
        from pathlib import Path

        from jev_tactics.planner import scoring

        source = Path(scoring.__file__).read_text(encoding="utf-8")
        bloc = next(b for b in source.split("\n\n") if "\nKILL_BONUS = " in "\n" + b)
        assert "inconnu" in bloc.lower() or "hp_known" in bloc, (
            "le bloc de KILL_BONUS ne dit pas qu'il est inerte a PV inconnus")
