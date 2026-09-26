"""Changer de carte : ou cliquer, et comment savoir si l'on est arrive.

Extrait de `farming/runner.py`, ou ces deux questions etaient melees a la comptabilite de
la recolte -- rapport de session, graphe de rendement, oubli des gisements, renoncement
de chasse. Rien de tout cela ne concerne le DEPLACEMENT, et c'est ce melange qui rendait
le bot incapable de se deplacer pour une autre raison que recolter.

La separation passe ici : ce module decide **ou cliquer** et **ce qui s'est passe**, il
n'enregistre rien. L'appelant tire les consequences -- ce sont les siennes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.coordinates import MapPosition
from jev_tactics.perception.ui import UI_ZONES, hidden_by_ui, ui_mask
from jev_tactics.world.navigation import Direction

# Marge depuis le bord de l'ecran. Le bord nominal est deja sur la carte a gauche, a
# droite et en haut ; en bas il faut rentrer davantage, l'ATH y mord.
EDGE_MARGIN = 6
# Jusqu'ou chercher un point libre en rentrant vers l'interieur, et par quel pas.
EDGE_SEARCH_LIMIT = 300
EDGE_SEARCH_STEP = 6


def on_interface(panels: NDArray[np.uint8] | None, x: int, y: int) -> bool:
    """Les deux exclusions ensemble : les panneaux OUVERTS, reconnus a leur teinte, et
    les rectangles fixes que leur clarte rend invisibles au masque."""
    return hidden_by_ui(panels, x, y) or any(
        x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in UI_ZONES)


def edge_point(frame: NDArray[np.uint8], direction: Direction) -> tuple[int, int]:
    """Ou cliquer pour sortir par ce bord, en evitant l'interface.

    On part du bord nominal et l'on RENTRE VERS L'INTERIEUR, jamais sur le cote. Le bord
    inferieur a bien des plages libres, mais la plus large est la MINICARTE : son
    interieur est un rendu de carte, clair et sature, que `ui_mask` ne reconnait pas comme
    interface. Chercher lateralement finirait par cliquer dedans. Rentrer ne peut, au
    pire, que cliquer la carte.

    Si rien n'est libre dans la limite, rend le point nominal : le deplacement echouera et
    sera COMPTE, ce qui vaut mieux qu'un clic au milieu du plateau.
    """
    height, width = frame.shape[:2]
    start, step = {
        Direction.LEFT: ((EDGE_MARGIN, height // 2), (1, 0)),
        Direction.RIGHT: ((width - EDGE_MARGIN, height // 2), (-1, 0)),
        Direction.TOP: ((width // 2, EDGE_MARGIN), (0, 1)),
        Direction.BOTTOM: ((width // 2, height - EDGE_MARGIN), (0, -1)),
    }[direction]
    panels = ui_mask(frame)
    x, y = start
    for offset in range(0, EDGE_SEARCH_LIMIT, EDGE_SEARCH_STEP):
        point = (x + step[0] * offset, y + step[1] * offset)
        if not on_interface(panels, *point):
            return point
    return start


class Move(StrEnum):
    """Ce qui s'est reellement passe apres le clic."""

    ARRIVED = "arrived"    # on est sur la carte voulue
    DRIFTED = "drifted"    # on a change de carte, mais pas pour celle-la
    STUCK = "stuck"        # le personnage n'a pas bouge


@dataclass(frozen=True)
class Verdict:
    """Le jugement d'un deplacement, sans aucune consequence tiree.

    `drift` n'est pas un detail de journal. Cliquer un bord fait MARCHER le personnage, et
    un obstacle sur le trajet peut le faire sortir par un autre cote -- on voulait la
    droite, on monte. Une simple comparaison d'images valide ce deplacement-la, puisqu'il
    change bien tout l'ecran ; le circuit repart alors d'une carte decalee, et comme il ne
    connait que des directions relatives, **le decalage ne se rattrape jamais**.

    C'est le mode de panne silencieux que seules les coordonnees rendent visible, et la
    raison pour laquelle ce module rend trois issues et non un booleen.
    """

    move: Move
    direction: Direction
    before: MapPosition
    after: MapPosition
    expected: tuple[int, int]

    @property
    def changed_map(self) -> bool:
        return self.move is not Move.STUCK

    def describe(self) -> str:
        if self.move is Move.ARRIVED:
            return f"{self.before.as_tuple()} -> {self.after.as_tuple()}"
        if self.move is Move.STUCK:
            return f"deplacement {self.direction.value} sans effet"
        return (f"derive {self.direction.value} : {self.before.as_tuple()} -> "
                f"{self.after.as_tuple()}, voulu {self.expected}")


def judge_move(
    direction: Direction,
    before: MapPosition,
    after: MapPosition,
) -> Verdict:
    """Comparer la carte ATTEINTE a celle VOULUE.

    Verifier qu'on a change de carte ne suffit pas : il faut verifier qu'on a change pour
    la BONNE. Fonction pure -- c'est ce qui la rend testable sans le jeu, alors que la
    version d'origine ne pouvait s'executer qu'au milieu d'une session de recolte.
    """
    dx, dy = direction.delta
    expected = (before.x + dx, before.y + dy)

    if after.as_tuple() == expected:
        return Verdict(Move.ARRIVED, direction, before, after, expected)
    if after.as_tuple() == before.as_tuple():
        return Verdict(Move.STUCK, direction, before, after, expected)
    return Verdict(Move.DRIFTED, direction, before, after, expected)
