"""Planification : des actions legales vers la meilleure sequence d'un tour.

`legal_actions` est la brique centrale du projet : elle sert A LA FOIS de generateur au
solveur exhaustif et de masque d'actions au futur PPO. Une seule implementation, donc
aucun risque que la politique apprenne sur des regles differentes de celles du solveur.
"""

from jev_tactics.planner.legal import (
    Action,
    Cast,
    EndTurn,
    Move,
    TurnState,
    apply_action,
    legal_actions,
)
from jev_tactics.planner.prune import useful_actions
from jev_tactics.planner.scoring import evaluate
from jev_tactics.planner.search import Plan, best_sequence

__all__ = [
    "Action", "Cast", "EndTurn", "Move", "Plan", "TurnState",
    "apply_action", "best_sequence", "evaluate", "legal_actions", "useful_actions",
]
