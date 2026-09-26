"""Le decideur de REFERENCE : le solveur existant, derriere l'interface commune.

Il n'apporte aucune capacite nouvelle -- c'est le but. Tant que ce decideur reproduit
exactement `best_sequence`, le chiffre de l'arene (79 % de victoires) reste comparable, et
tout ecart mesure ensuite s'impute a Jev et non au portage.

Il sert aussi de REPLI. Quand Jev est indisponible, lent ou peu sur de lui, c'est ici que
la decision revient : un solveur deterministe et mesure est un bien meilleur defaut que la
fin de tour, qui s'executerait sans erreur en donnant l'illusion d'un bot prudent.
"""

from __future__ import annotations

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.decision import Decision
from jev_tactics.planner.search import DEFAULT_MAX_NODES, best_sequence
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState


class SearchDecider:
    """Argmax de `evaluate` par recherche elaguee. Deterministe, hors ligne, explicable."""

    name = "search"

    def __init__(self, max_nodes: int = DEFAULT_MAX_NODES):
        self.max_nodes = max_nodes

    def decide(
        self,
        board: BoardMap,
        state: CombatState,
        spells: list[Spell],
    ) -> Decision:
        plan = best_sequence(board, state, spells, max_nodes=self.max_nodes)
        reason = plan.describe()
        if plan.truncated:
            # La troncature n'est pas une erreur, mais elle n'est pas neutre : a 2 000
            # noeuds, un tiers des decisions amputees coutait huit points de victoire.
            # Elle doit se lire dans le journal, pas seulement dans un compteur.
            reason += " [recherche tronquee]"
        return Decision(plan=plan, source="search", reason=reason)
