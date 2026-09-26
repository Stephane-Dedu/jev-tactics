"""Regles du combat : geometrie, deplacement, portees.

Source de verite UNIQUE des regles. Le solveur (planner/) et le futur simulateur RL
consomment ces memes fonctions -- dupliquer les regles entre les deux serait la faute la
plus couteuse du projet (deux implementations qui divergent silencieusement, et une
politique entrainee sur des regles fausses).
"""

from jev_tactics.rules.movement import (
    grid_distance,
    is_walkable,
    path_between,
    reachable_cells,
)

__all__ = ["grid_distance", "is_walkable", "path_between", "reachable_cells"]
