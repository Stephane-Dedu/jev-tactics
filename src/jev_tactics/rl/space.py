"""Espace d'actions discret et masque de legalite.

Le masquage d'actions n'est pas une optimisation, c'est une condition d'existence. A
chaque etat, l'immense majorite des actions sont illegales (PA/PM insuffisants, hors
portee, case occupee, plafond de lancers atteint) : sans masque, PPO passe son temps a
echantillonner des coups impossibles et n'apprend rien.

**Le masque n'est pas une reimplementation des regles.** Il est construit A PARTIR de
`legal_actions`, le meme generateur que le solveur : on enumere les actions legales, on
les range chacune dans son emplacement, et le masque est exactement « il y a une action
ici ». Deux consequences :

  - le masque ne peut pas diverger des regles -- il n'y a pas de second jeu de regles ;
  - le decodage est une lecture de table, pas une reconstruction. Un indice masque n'a
    tout simplement pas d'entree, donc aucune action mal reconstituee ne peut fuir.

C'est la raison d'etre de la brique partagee annoncee dans `planner/legal.py`.

Disposition des indices, fixe pour un plateau et une liste de sorts donnes :

    0                     fin de tour
    1 .. N                deplacement vers la case (indice - 1)
    1+N + s*N + c         lancer du sort s sur la case c
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import Action, Cast, Move, TurnState, legal_actions
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState

END_TURN = 0


@dataclass(frozen=True)
class ActionSpace:
    """Numerotation stable des actions. La taille ne depend que du plateau et des sorts.

    Elle doit rester CONSTANTE pendant tout l'entrainement : un reseau a une tete de
    taille fixe. D'ou l'enumeration exhaustive case x sort, y compris les combinaisons
    qui ne seront jamais legales -- le masque s'en charge a l'execution.
    """

    cells: int
    spells: tuple[str, ...]

    @classmethod
    def build(cls, board: BoardMap, spells: list[Spell]) -> ActionSpace:
        return cls(cells=len(board), spells=tuple(s.name for s in spells))

    @property
    def size(self) -> int:
        return 1 + self.cells * (1 + len(self.spells))

    def move_index(self, cell: int) -> int:
        return 1 + cell

    def cast_index(self, spell: str, target: int) -> int:
        return 1 + self.cells * (1 + self.spells.index(spell)) + target

    def index_of(self, action: Action) -> int:
        if isinstance(action, Move):
            return self.move_index(action.cell)
        if isinstance(action, Cast):
            return self.cast_index(action.spell, action.target)
        return END_TURN

    def describe(self, index: int) -> str:
        if index == END_TURN:
            return "fin de tour"
        offset, cell = divmod(index - 1, self.cells)
        if offset == 0:
            return f"deplacement -> {cell}"
        return f"{self.spells[offset - 1]} -> {cell}"


def legal_table(
    space: ActionSpace,
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    turn: TurnState,
) -> dict[int, Action]:
    """Indice -> action legale, construit depuis `legal_actions`.

    Un sort absent de l'espace (config changee en cours de route) est ignore plutot que
    de faire echouer l'episode : mieux vaut un coup de moins qu'un entrainement qui
    s'arrete au bout de trois heures.
    """
    table: dict[int, Action] = {}
    for action in legal_actions(board, state, spells, turn):
        if isinstance(action, Cast) and action.spell not in space.spells:
            continue
        table[space.index_of(action)] = action
    return table


def action_mask(space: ActionSpace, table: dict[int, Action]) -> NDArray[np.bool_]:
    """Masque booleen de meme taille que l'espace : True = jouable maintenant.

    Toujours au moins une entree vraie (`legal_actions` rend toujours EndTurn). Un
    masque entierement faux bloquerait l'echantillonnage de la politique -- c'est le
    genre de panne qui se manifeste par un NaN trois epoques plus tard.
    """
    mask = np.zeros(space.size, dtype=bool)
    if table:
        mask[list(table)] = True
    return mask
