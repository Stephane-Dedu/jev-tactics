"""Arene : mesurer une politique plutot que l'affirmer.

Sert deux fois : garde-fou de non-regression du solveur, et reference que le RL devra
battre. Une politique apprise qui ne depasse pas la recherche 1 tour est cassee -- sans
ce chiffre on ne peut meme pas le dire."""

import random
from pathlib import Path

import pytest

from jev_tactics.planner import best_sequence
from jev_tactics.rules.spells import Spell
from jev_tactics.sim import (
    BenchmarkResult,
    FightResult,
    benchmark,
    random_scenario,
    run_fight,
    square_board,
)
from jev_tactics.state import Team

ROOT = Path(__file__).resolve().parents[1]

BOARD = square_board(7)
BOLT = Spell(name="trait", ap_cost=3, range_min=1, range_max=6,
             needs_line_of_sight=False, damage_min=20, damage_max=20,
             max_casts_per_turn=2)
SPELLS = [BOLT]


def solver(board, state, spells):
    return best_sequence(board, state, spells).actions


def passive(board, state, spells):
    """Ne fait jamais rien : doit perdre."""
    return []


class TestScenario:
    def test_entities_on_distinct_cells(self):
        state = random_scenario(BOARD, random.Random(0), enemies=3)
        cells = [e.cell for e in state.entities]
        assert len(cells) == len(set(cells)) == 4

    def test_exactly_one_self(self):
        state = random_scenario(BOARD, random.Random(1), enemies=2)
        assert sum(1 for e in state.entities if e.is_self) == 1

    def test_same_seed_same_scenario(self):
        a = random_scenario(BOARD, random.Random(7), enemies=2)
        b = random_scenario(BOARD, random.Random(7), enemies=2)
        assert [e.cell for e in a.entities] == [e.cell for e in b.entities]


class TestTheArenaCanExpressTheRealMatchup:
    """`enemy_hp`, `ap` et `mp` existaient dans `random_scenario` et n'etaient PAS
    transmis par `benchmark` : l'arene ne savait mesurer qu'un seul appariement, 100 PV /
    6 PA / 3 PM contre des ennemis a 60 PV.

    Or le personnage reel, mesure sur la capture de combat, porte 1632 PV, 10 PA et 4 PM,
    et une zone a larves oppose des monstres faibles. L'appariement que le bot va jouer
    des heures durant etait le seul que l'instrument ne pouvait pas exprimer."""

    def _ap_seen(self):
        """Enregistre les PA que la politique voit a chaque tour."""
        vus = []

        def policy(board, state, spells):
            vus.append(state.self_entity().ap)
            return passive(board, state, spells)

        return vus, policy

    def test_action_points_survive_the_first_turn(self):
        """LE defaut derriere l'omission. `run_fight` recharge les combattants au debut
        de chaque tour avec SES PROPRES defauts (6/3) : un scenario construit a 10 PA
        jouait son premier tour a 10 et tous les suivants a 6, en silence. Un instrument
        qui modifie le scenario apres coup rend un chiffre credible et faux."""
        vus, policy = self._ap_seen()
        state = random_scenario(BOARD, random.Random(3), enemies=1,
                                enemy_hp=100000, ap=10, mp=4)
        run_fight(BOARD, state, SPELLS, policy, max_turns=3, ap=10, mp=4)
        assert vus[:3] == [10, 10, 10], f"les PA sont retombes : {vus}"

    def test_the_defaults_are_unchanged(self):
        """Les mesures deja consignees doivent rester comparables."""
        vus, policy = self._ap_seen()
        state = random_scenario(BOARD, random.Random(3), enemies=1, enemy_hp=100000)
        run_fight(BOARD, state, SPELLS, policy, max_turns=2)
        assert vus[:2] == [6, 6]

    def test_benchmark_forwards_the_matchup(self):
        """Le bout en bout : ce que `benchmark` construit doit etre ce qu'on a demande."""
        vus, policy = self._ap_seen()
        benchmark(policy, SPELLS, fights=1, seed=0, board=BOARD, enemies=1,
                  hp=500, enemy_hp=100000, ap=11, mp=5)
        assert vus and set(vus) == {11}, f"PA vus : {sorted(set(vus))}"

    def test_a_weak_enemy_makes_the_fight_easy(self):
        """Ce que l'arene doit pouvoir montrer : contre des monstres faibles, le solveur
        gagne. C'est la question que pose une zone a larves, et elle etait inexprimable."""
        result = benchmark(solver, SPELLS, fights=6, seed=1, board=BOARD,
                           enemies=2, hp=1600, enemy_hp=25, ap=10, mp=4,
                           enemy_hp_known=False)
        assert result.win_rate == 1.0, result.describe()


class TestRunFight:
    def test_solver_beats_a_lone_weak_enemy(self):
        state = random_scenario(BOARD, random.Random(3), enemies=1, enemy_hp=20)
        assert run_fight(BOARD, state, SPELLS, solver).winner is Team.ALLY

    def test_passive_policy_loses(self):
        """Verification que l'adversaire de reference exerce bien une pression."""
        state = random_scenario(BOARD, random.Random(4), enemies=2, hp=40)
        assert run_fight(BOARD, state, SPELLS, passive).winner is Team.ENEMY

    def test_turn_limit_produces_a_draw_not_a_hang(self):
        """Un combat qui ne se conclut pas doit s'arreter, pas boucler."""
        state = random_scenario(BOARD, random.Random(5), enemies=1, enemy_hp=100000)
        result = run_fight(BOARD, state, SPELLS, passive, max_turns=3)
        assert result.turns <= 3

    def test_reports_damage_and_hp(self):
        state = random_scenario(BOARD, random.Random(6), enemies=1, enemy_hp=20)
        result = run_fight(BOARD, state, SPELLS, solver)
        assert result.damage_dealt > 0 and result.hp_left >= 0


class TestBenchmark:
    def test_counts_add_up(self):
        result = benchmark(solver, SPELLS, fights=6, seed=0, board=BOARD, enemies=1)
        assert result.wins + result.losses + result.draws == result.fights == 6

    def test_same_seed_reproduces_the_result(self):
        """Deux politiques doivent etre comparees sur LES MEMES combats, sinon l'ecart
        mesure serait celui des scenarios."""
        a = benchmark(solver, SPELLS, fights=5, seed=11, board=BOARD, enemies=1)
        b = benchmark(solver, SPELLS, fights=5, seed=11, board=BOARD, enemies=1)
        assert (a.wins, a.turns, a.hp_left) == (b.wins, b.turns, b.hp_left)

    def test_solver_beats_a_passive_policy(self):
        """Le test qui donne son sens a l'arene : une politique meilleure doit gagner
        davantage, sur les memes scenarios."""
        kwargs = {"fights": 8, "seed": 2, "board": BOARD, "enemies": 1}
        assert (benchmark(solver, SPELLS, **kwargs).wins
                > benchmark(passive, SPELLS, **kwargs).wins)

    def test_win_rate_is_a_fraction(self):
        result = benchmark(solver, SPELLS, fights=4, seed=3, board=BOARD, enemies=1)
        assert 0.0 <= result.win_rate <= 1.0

    def test_empty_benchmark_is_safe(self):
        empty = BenchmarkResult()
        assert empty.win_rate == 0.0 and empty.median_turns == 0.0

    def test_defeat_profile_is_empty_without_defeats(self):
        result = BenchmarkResult()
        assert "aucune defaite" in result.defeat_profile()

    def test_defeat_profile_counts_defeats_only(self):
        """Melanger victoires et defaites dans une moyenne ne dit rien : ce qu'on veut
        savoir, c'est comment on PERD."""
        result = benchmark(passive, SPELLS, fights=4, seed=8, board=BOARD, enemies=1)
        assert len(result.defeats) == result.losses

    def test_defeat_profile_distinguishes_a_wipe_from_a_near_miss(self):
        """Perdre sans avoir tue personne et perdre a un ennemi pres appellent des
        corrections opposees (agressivite contre survie) : les confondre rendrait la
        mesure inutilisable pour decider."""
        wipe = FightResult(winner=Team.ENEMY, turns=2, hp_left=0, damage_dealt=0.0,
                           enemies_left=2, enemies_start=2)
        near = FightResult(winner=Team.ENEMY, turns=4, hp_left=0, damage_dealt=80.0,
                           enemies_left=1, enemies_start=2)
        text = BenchmarkResult(defeats=[wipe, near]).defeat_profile()
        assert "1 sans aucune victime" in text and "1 a un ennemi pres" in text

    def test_kills_derives_from_the_two_counts(self):
        assert FightResult(winner=None, turns=1, hp_left=5, damage_dealt=0.0,
                           enemies_left=1, enemies_start=3).kills == 2

    def test_damage_taken_is_recorded(self):
        state = random_scenario(BOARD, random.Random(4), enemies=2, hp=40)
        assert run_fight(BOARD, state, SPELLS, passive).damage_taken > 0

    def test_describe_is_readable(self):
        result = benchmark(solver, SPELLS, fights=3, seed=4, board=BOARD, enemies=1)
        assert "victoires" in result.describe() and "tours" in result.describe()


class TestTheArenaSaturatesWithoutMoreHealth:
    """L'instrument ne discrimine QU'A DEUX ENNEMIS avec ses reglages par defaut.

        1 ennemi  100%    2 ennemis  82%    3 ennemis  2,5%    4 ennemis  0%

    A trois, le resultat est le meme quelle que soit la ponderation testee -- et ce n'est
    pas une faiblesse du solveur : trois adversaires infligent ~162 degats par tour a un
    joueur de 100 PV qui en rend ~54. Le plafond est ARITHMETIQUE.

    Tout ce qui a ete regle dans ce projet l'a donc ete dans la seule fenetre ou l'arene a
    de la resolution. Le parametre `hp` existe pour en ouvrir d'autres : a 200 PV, trois
    ennemis redeviennent jouables, et le classement des poids y CHANGE.
    """

    def _rate(self, enemies, hp):
        board = square_board(11)
        policy = lambda b, s, sp: best_sequence(b, s, sp).actions
        return benchmark(policy, SPELLS, fights=12, seed=0, board=board,
                         enemies=enemies, hp=hp).win_rate

    def test_three_enemies_are_hopeless_at_the_default_health(self):
        assert self._rate(enemies=3, hp=100) < 0.2

    def test_more_health_lifts_the_ceiling(self):
        """On teste la DIRECTION, pas une bande : le point exact ou l'arene discrimine
        depend du jeu de sorts. Avec les trois sorts de la config d'exemple il tombe vers
        200 PV ; avec le sort unique de ces tests, vers 300. Figer une bande ici la ferait
        casser au premier sort ajoute, pour une raison sans rapport avec ce qu'on verifie.

            BOLT seul, 3 ennemis :  100 PV -> 0%   200 -> 8%   400 -> 100%
        """
        assert self._rate(enemies=3, hp=400) > self._rate(enemies=3, hp=100)

    def test_health_reaches_the_scenario(self):
        """Le parametre doit traverser jusqu'au generateur : ajoute mais non branche, il
        aurait donne des mesures identiques et rassurantes."""
        rng = random.Random(0)
        state = random_scenario(square_board(9), rng, enemies=2, hp=250)
        assert state.find_self().hp_max == 250


class TestTheArenaAssumedKnownEnemyHealth:
    """L'ecart le plus important entre cette arene et le jeu, et il etait invisible.

    Mesure sur les 32 captures disponibles : 72 ennemis detectes, ZERO avec des PV connus.
    Rien ne les lit -- ni l'OCR, reserve au joueur, ni la timeline, qui ne donne qu'un
    ratio sans maximum.

    Sans PV, `evaluate` ne peut ni plafonner les degats ni accorder la PRIME DE MISE A
    MORT -- le terme qui, seul, fait passer le taux de victoire de 0,6 % a 81 %.

        PV ennemis connus (ce que l'arene mesurait)     85,8%      70,0% a 3 ennemis
        PV ennemis inconnus (ce que le bot fait)        36,7%      50,0%

    Quarante-neuf points. Tous les poids regles jusqu'ici l'ont donc ete dans un regime que
    le bot ne connait pas.
    """

    def _rate(self, known):
        policy = lambda b, s, sp: best_sequence(b, s, sp).actions
        return benchmark(policy, SPELLS, fights=12, seed=0, board=square_board(11),
                         enemies=2, enemy_hp_known=known).win_rate

    def test_hiding_enemy_health_costs_a_lot(self):
        """L'ecart mesure ici est plus petit que les 49 points annonces ci-dessus, et
        c'est attendu : ces tests n'ont qu'UN sort de 20 degats, quand la config d'exemple
        en a trois jusqu'a 30. Moins on frappe fort, moins la prime de mise a mort a
        d'occasions de se declencher, donc moins sa perte coute. Mesure avec ce sort
        unique : 25 % contre 17 %."""
        assert self._rate(True) - self._rate(False) > 0.05

    def test_the_option_reaches_the_entities(self):
        """Ajoute mais non branche, il aurait rendu deux chiffres identiques et
        rassurants -- le pire des resultats."""
        assert self._rate(False) < self._rate(True)


class TestWhereTheInstrumentSaturates:
    """Le point de saturation du banc, jamais ecrit alors que tout se lit a travers lui.

    `hp=100` existe « PARCE QUE L'INSTRUMENT SATURE » : au-dela, le solveur gagne tout et
    la mesure ne separe plus rien. Mais le seuil n'etait chiffre nulle part, si bien qu'on
    lit « 20 % de victoires a 3 ennemis » comme une prediction de ce qui se passera en jeu.
    C'en est une mauvaise.

    Mesure avec la vraie configuration de sorts, 3 ennemis, 25 combats :

        100 PV :  20 %      <- le reglage du banc, choisi pour discriminer
        150 PV :  72 %
        200 PV : 100 %
        300 PV : 100 %

    Le personnage reel a 1672 PV et farme des larves bleues, donc tres au-dessus du point
    de saturation. J'ai failli regler les poids de survie pour redresser ces 20 % : le
    balayage n'a rien donne dans aucune direction -- et pour cause, il n'y avait rien a
    redresser.
    """

    SORTS = ROOT / "configs" / "spells" / "sacrieur.json"

    def _taux(self, pv):
        from jev_tactics.rules.spells import load_spells

        if not self.SORTS.exists():
            pytest.skip("config du personnage absente")
        sorts = load_spells(self.SORTS)
        return benchmark(lambda b, s, sp: best_sequence(b, s, sp).actions, sorts,
                         fights=15, seed=1, enemies=3, hp=pv).win_rate

    def test_the_bench_setting_is_discriminating(self):
        """A 100 PV le banc doit encore PERDRE : un instrument qui gagne tout ne mesure
        plus rien, et c'est justement ce qu'on lui demande d'eviter."""
        assert self._taux(100) < 0.6

    def test_a_realistic_character_wins_every_fight(self):
        """Ce que le chiffre du banc ne dit pas, et qu'il faut savoir avant de conclure
        que le combat est le goulot."""
        assert self._taux(200) == 1.0

    def test_the_saturation_sits_between_the_two(self):
        """Sans cet encadrement, les deux tests precedents seraient compatibles avec un
        instrument qui bascule n'importe ou."""
        assert self._taux(100) < self._taux(150) < 1.0
