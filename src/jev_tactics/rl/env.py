"""Environnement d'entrainement : un combat, une action a la fois.

Volontairement SANS dependance a gymnasium. L'encodage, le masque et la recompense sont
la vraie substance, et les enfermer derriere une bibliotheque les rendrait testables
seulement avec elle. L'adaptateur gymnasium se resume ensuite a reexposer `reset`/`step`,
qui ont deja la bonne forme.

Le pas est UNE ACTION, pas un tour. Un tour entier par pas obligerait la politique a
produire une sequence complete d'un coup -- exactement le probleme combinatoire (~10^8
sequences) que le solveur doit elaguer. Action par action, la politique construit sa
sequence progressivement, et le masque la guide a chaque etape.

Deux garde-fous, memes raisons que dans le simulateur :

  - **une action non masquee ne leve jamais d'exception.** Une politique qui ignore le
    masque (bug d'integration, exploration forcee) doit produire un signal negatif, pas
    faire tomber un entrainement de plusieurs heures.
  - **la recompense ne recompense que ce qui est arrive.** Elle est calculee depuis
    l'etat rendu par le simulateur, jamais depuis l'action demandee.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import EndTurn, TurnState, apply_action
from jev_tactics.rl.space import ActionSpace, action_mask, legal_table
from jev_tactics.rules.spells import Spell
from jev_tactics.sim.arena import random_scenario, square_board
from jev_tactics.sim.combat import apply_actions, enemy_policy, winner
from jev_tactics.state import CombatState, Team

# Canaux par case : moi, allie, ennemi, obstacle, PV de l'occupant.
CHANNELS = 5
# Grandeurs globales : PA, PM, mes PV, nombre d'ennemis restants.
SCALARS = 4

# Ponderations de la recompense. Les degats sont ramenes a l'echelle des PV pour que le
# signal ne depende pas des valeurs absolues choisies dans le scenario.
KILL_REWARD = 1.0
WIN_REWARD = 5.0
LOSS_PENALTY = 5.0
# Cout constant par action : sans lui, tourner en rond a somme nulle est aussi bon que
# gagner, et la politique apprend a ne rien faire jusqu'a la limite de tours.
STEP_COST = 0.01
ILLEGAL_PENALTY = 0.5


def encode(board: BoardMap, state: CombatState) -> NDArray[np.float32]:
    """Etat -> vecteur de taille fixe (CHANNELS par case + SCALARS).

    Les PV sont encodes en RATIO, pas en valeur absolue : une politique entrainee sur des
    monstres a 60 PV doit rester valable face a des monstres a 600. C'est la meme
    precaution que pour les degats.
    """
    cells = len(board)
    grid = np.zeros((CHANNELS, cells), dtype=np.float32)

    for entity in state.entities:
        if not 0 <= entity.cell < cells:
            continue
        if entity.is_self:
            grid[0, entity.cell] = 1.0
        elif entity.team is Team.ALLY:
            grid[1, entity.cell] = 1.0
        else:
            grid[2, entity.cell] = 1.0
        grid[4, entity.cell] = entity.hp / entity.hp_max if entity.hp_max else 0.0

    for cell in state.obstacles:
        if 0 <= cell < cells:
            grid[3, cell] = 1.0

    me = state.find_self()
    scalars = np.array([
        me.ap / 12.0 if me else 0.0,
        me.mp / 6.0 if me else 0.0,
        (me.hp / me.hp_max if me and me.hp_max else 0.0),
        len(state.enemies()) / 4.0,
    ], dtype=np.float32)

    return np.concatenate([grid.reshape(-1), scalars])


def observation_size(board: BoardMap) -> int:
    return CHANNELS * len(board) + SCALARS


@dataclass
class StepResult:
    observation: NDArray[np.float32]
    reward: float
    terminated: bool         # le combat s'est conclu
    truncated: bool          # limite de tours atteinte, issue inconnue
    mask: NDArray[np.bool_]
    info: dict = field(default_factory=dict)


@dataclass
class CombatEnv:
    """Un combat contre l'adversaire de reference, joue action par action."""

    spells: list[Spell]
    board: BoardMap = field(default_factory=square_board)
    enemies: int = 2
    max_turns: int = 25
    ap: int = 6
    mp: int = 3

    state: CombatState = field(init=False)
    turn: TurnState = field(init=False)
    space: ActionSpace = field(init=False)
    turns_played: int = field(init=False, default=0)
    _rng: random.Random = field(init=False, default_factory=random.Random)

    def __post_init__(self) -> None:
        self.space = ActionSpace.build(self.board, self.spells)
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32],
                                                      NDArray[np.bool_]]:
        if seed is not None:
            self._rng = random.Random(seed)
        self.state = random_scenario(self.board, self._rng, enemies=self.enemies,
                                     ap=self.ap, mp=self.mp)
        self.turn = TurnState.from_combat(self.state)
        self.turns_played = 0
        return encode(self.board, self.state), self.action_mask()

    def _table(self) -> dict:
        return legal_table(self.space, self.board, self.state, self.spells, self.turn)

    def action_mask(self) -> NDArray[np.bool_]:
        return action_mask(self.space, self._table())

    def step(self, index: int) -> StepResult:
        table = self._table()
        action = table.get(index)

        if action is None:
            # Coup non masque : on ne l'applique pas, on ne plante pas, et on le dit.
            return self._result(-ILLEGAL_PENALTY,
                                info={"illegal": self.space.describe(index)})

        if isinstance(action, EndTurn):
            return self._end_turn()

        before = self._hp_totals()
        outcome = apply_actions(self.board, self.state, self.spells, [action])
        self.state = outcome.state
        self.turn = apply_action(self.turn, action)

        reward = self._damage_reward(before) + KILL_REWARD * outcome.kills - STEP_COST
        if outcome.illegal:
            # Le masque et le simulateur ont diverge : c'est un bug, pas un coup joue.
            # L'entrainement continue, mais la trace doit rester dans `info`.
            reward -= ILLEGAL_PENALTY
            return self._result(reward, info={"desaccord": self.space.describe(index)})
        return self._result(reward)

    def _end_turn(self) -> StepResult:
        """Fin de tour : l'adversaire joue, puis les ressources se rechargent."""
        before = self._hp_totals()
        for enemy in [e for e in self.state.entities if e.team is Team.ENEMY]:
            actions = enemy_policy(self.board, self.state, self.spells, enemy.entity_id)
            self.state = apply_actions(self.board, self.state, self.spells, actions,
                                       actor_id=enemy.entity_id).state

        self.turns_played += 1
        self.state = self.state.model_copy(update={
            "entities": [e.model_copy(update={"ap": self.ap, "mp": self.mp})
                         for e in self.state.entities],
            "turn": self.state.turn + 1,
        })
        self.turn = (TurnState.from_combat(self.state)
                     if self.state.find_self() else TurnState(cell=0, ap=0, mp=0))
        return self._result(self._damage_reward(before) - STEP_COST)

    def _hp_totals(self) -> tuple[float, float]:
        mine = sum(e.hp for e in self.state.entities if e.is_self)
        theirs = sum(e.hp for e in self.state.entities if e.team is Team.ENEMY)
        return mine, theirs

    def _damage_reward(self, before: tuple[float, float]) -> float:
        """Degats infliges moins degats subis, normalises par les PV de depart."""
        mine, theirs = self._hp_totals()
        me = self.state.find_self()
        scale = float(me.hp_max) if me and me.hp_max else 100.0
        return ((before[1] - theirs) - (before[0] - mine)) / scale

    def _result(self, reward: float, info: dict | None = None) -> StepResult:
        outcome = winner(self.state)
        terminated = outcome is not None
        truncated = not terminated and self.turns_played >= self.max_turns

        if outcome is Team.ALLY:
            reward += WIN_REWARD
        elif outcome is Team.ENEMY:
            reward -= LOSS_PENALTY

        return StepResult(
            observation=encode(self.board, self.state),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            mask=self.action_mask(),
            info={"turns": self.turns_played, **(info or {})},
        )
