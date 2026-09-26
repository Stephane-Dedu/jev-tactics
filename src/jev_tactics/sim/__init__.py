"""Simulateur de combat : l'investissement qui conditionne le RL.

Le point critique n'est pas l'algorithme d'apprentissage mais la FIDELITE du simulateur.
Une politique entrainee sur des regles fausses joue faux, et rien dans les courbes
d'apprentissage ne le signale. D'ou la regle : le simulateur consomme `rules/`, le meme
code que le solveur -- pas une reimplementation.
"""

from jev_tactics.sim.arena import (
    BenchmarkResult,
    FightResult,
    benchmark,
    random_scenario,
    run_fight,
    square_board,
)
from jev_tactics.sim.combat import TurnOutcome, apply_actions, enemy_policy, winner

__all__ = [
    "BenchmarkResult", "FightResult", "TurnOutcome", "apply_actions", "benchmark",
    "enemy_policy", "random_scenario", "run_fight", "square_board", "winner",
]
