"""Environnement d'entrainement.

Ce qu'on verifie ici n'est pas « ca tourne » mais que le SIGNAL est juste : une
recompense mal cablee produit une politique qui optimise autre chose que gagner, et rien
dans les courbes ne le dit. On verifie donc le sens de chaque terme, et surtout que
l'environnement ne se laisse pas exploiter (coup illegal, immobilisme)."""

import numpy as np

from jev_tactics.rl.env import (
    LOSS_PENALTY,
    STEP_COST,
    WIN_REWARD,
    CombatEnv,
    encode,
    observation_size,
)
from jev_tactics.rules.spells import Spell
from jev_tactics.sim import square_board
from jev_tactics.state import CombatState, Entity, Team

BOARD = square_board(7)
CENTRE = BOARD.index_at(3, 3)
FOE = BOARD.index_at(3, 5)
BOLT = Spell(name="trait", ap_cost=3, range_min=1, range_max=6,
             needs_line_of_sight=False, damage_min=20, damage_max=20,
             max_casts_per_turn=2)
SPELLS = [BOLT]


def _env(**kwargs):
    return CombatEnv(spells=SPELLS, board=BOARD, **kwargs)


def _state(me=CENTRE, enemies=(FOE,), hp=100, enemy_hp=100,
           obstacles=frozenset()):
    entities = [Entity(entity_id="me", team=Team.ALLY, cell=me, hp=hp, hp_max=100,
                       ap=6, mp=3, is_self=True)]
    entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=c, hp=enemy_hp,
                        hp_max=100, ap=6, mp=3) for i, c in enumerate(enemies)]
    return CombatState(turn=1, entities=entities, obstacles=set(obstacles))


class TestEncoding:
    def test_size_is_announced_correctly(self):
        assert encode(BOARD, _state()).shape == (observation_size(BOARD),)

    def test_self_and_enemy_land_on_distinct_channels(self):
        vector = encode(BOARD, _state())
        cells = len(BOARD)
        assert vector[CENTRE] == 1.0                              # canal « moi »
        assert vector[2 * cells + BOARD.index_at(3, 5)] == 1.0     # canal « ennemi »

    def test_obstacles_are_encoded(self):
        cells = len(BOARD)
        vector = encode(BOARD, _state(obstacles={7}))
        assert vector[3 * cells + 7] == 1.0

    def test_hp_encoded_as_ratio_not_absolute(self):
        """Une politique entrainee sur des monstres a 60 PV doit rester valable face a
        des monstres a 600 : c'est le ratio qui porte l'information."""
        small = Entity(entity_id="e", team=Team.ENEMY, cell=5, hp=30, hp_max=60,
                       ap=6, mp=3)
        big = small.model_copy(update={"hp": 300, "hp_max": 600})
        a = encode(BOARD, CombatState(turn=1, entities=[_state().entities[0], small]))
        b = encode(BOARD, CombatState(turn=1, entities=[_state().entities[0], big]))
        assert np.array_equal(a, b)

    def test_survives_a_dead_player(self):
        """La mort du joueur est une issue NORMALE en entrainement : l'encodage ne doit
        pas lever, sinon l'episode terminal fait tomber le run."""
        only_enemy = CombatState(turn=1, entities=[
            Entity(entity_id="e0", team=Team.ENEMY, cell=5, hp=50, hp_max=50,
                   ap=6, mp=3)])
        assert encode(BOARD, only_enemy).shape == (observation_size(BOARD),)


class TestReset:
    def test_same_seed_same_start(self):
        a, _ = _env().reset(seed=3)
        b, _ = _env().reset(seed=3)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        a, _ = _env().reset(seed=1)
        b, _ = _env().reset(seed=2)
        assert not np.array_equal(a, b)

    def test_mask_offered_from_the_start(self):
        _obs, mask = _env().reset(seed=0)
        assert mask.any()


class TestIllegalActions:
    def test_unmasked_index_is_penalised_not_raised(self):
        """Une politique qui ignore le masque doit produire un signal negatif, jamais
        faire tomber un entrainement de plusieurs heures."""
        env = _env()
        env.reset(seed=0)
        blocked = int(np.flatnonzero(~env.action_mask())[0])
        result = env.step(blocked)
        assert result.reward < 0 and "illegal" in result.info

    def test_unmasked_index_leaves_the_state_untouched(self):
        env = _env()
        before, _ = env.reset(seed=0)
        blocked = int(np.flatnonzero(~env.action_mask())[0])
        assert np.array_equal(env.step(blocked).observation, before)


class TestReward:
    def _fixed(self, state, **kwargs):
        env = _env(**kwargs)
        env.reset(seed=0)
        env.state = state
        from jev_tactics.planner.legal import TurnState
        env.turn = TurnState.from_combat(state)
        return env

    def test_damage_is_rewarded(self):
        env = self._fixed(_state())
        index = env.space.cast_index("trait", BOARD.index_at(3, 5))
        assert env.step(index).reward > 0

    def test_killing_the_last_enemy_wins(self):
        env = self._fixed(_state(enemy_hp=20))
        result = env.step(env.space.cast_index("trait", BOARD.index_at(3, 5)))
        assert result.terminated and result.reward > WIN_REWARD

    def test_dying_is_penalised(self):
        """Le joueur a 1 PV finit son tour : l'adversaire frappe et le combat se conclut
        en sa defaveur."""
        env = self._fixed(_state(hp=1))
        result = env.step(0)                      # fin de tour
        assert result.terminated and result.reward < -LOSS_PENALTY / 2

    def test_doing_nothing_costs_something(self):
        """Sans cout par pas, tourner en rond vaut autant que gagner et la politique
        apprend l'immobilisme."""
        env = self._fixed(_state(enemies=()))
        env.state = _state(enemies=(BOARD.index_at(0, 0),))
        result = env.step(0)
        assert result.reward < 0 or result.info["turns"] == 1
        assert STEP_COST > 0


class TestEpisode:
    def test_turn_limit_truncates_instead_of_hanging(self):
        env = _env(max_turns=2)
        env.reset(seed=0)
        for _ in range(50):
            result = env.step(0)                  # fin de tour en boucle
            if result.terminated or result.truncated:
                break
        assert result.terminated or result.truncated

    def test_mask_and_simulator_never_disagree_over_many_episodes(self):
        """Version integrale de l'accord masque/simulateur : sur des centaines de coups
        tires au hasard, aucun coup masque comme jouable ne doit etre refuse."""
        env = _env()
        rng = np.random.default_rng(0)
        for episode in range(12):
            _obs, mask = env.reset(seed=episode)
            for _ in range(120):
                result = env.step(int(rng.choice(np.flatnonzero(mask))))
                assert "desaccord" not in result.info, result.info
                assert "illegal" not in result.info, result.info
                mask = result.mask
                if result.terminated or result.truncated:
                    break

    def test_a_random_masked_policy_does_not_win_easily(self):
        """Reference de plancher : si le hasard gagnait souvent, l'environnement serait
        trop facile pour mesurer quoi que ce soit -- et les 71 % du solveur ne voudraient
        rien dire."""
        env = _env()
        rng = np.random.default_rng(1)
        wins = 0
        for episode in range(10):
            _obs, mask = env.reset(seed=100 + episode)
            for _ in range(200):
                result = env.step(int(rng.choice(np.flatnonzero(mask))))
                mask = result.mask
                if result.terminated or result.truncated:
                    break
            wins += result.terminated and not env.state.enemies()
        assert wins <= 3
