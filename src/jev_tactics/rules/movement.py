"""Deplacement sur le plateau : distances, cases atteignables, chemins.

Le plateau est un graphe : les cases sont les noeuds, l'adjacence est celle du reseau
iso (4-voisinage sur les coordonnees diagonales (i, j) -- en vue iso ce sont bien les
quatre cases qui touchent la case courante par une arete).

Deux notions de distance, a ne pas confondre :
  - `grid_distance` : distance de Manhattan sur (i, j), a vol d'oiseau. C'est la portee
    des sorts en Dofus, et elle ignore les obstacles.
  - `reachable_cells` : distance de PARCOURS (BFS), qui contourne les obstacles. C'est
    ce que consomment les PM.
Un ennemi peut etre a portee de sort (grid_distance 2) tout en etant inatteignable a
pied (mur entre les deux) : le planificateur a besoin des deux.
"""

from __future__ import annotations

from collections import deque

from jev_tactics.calibration.grid import BoardMap


def grid_distance(board: BoardMap, a: int, b: int) -> int:
    """Distance de Manhattan entre deux cases, sur les coordonnees du reseau.

    C'est la distance « de portee » : a vol d'oiseau, sans tenir compte des obstacles.
    """
    (ia, ja), (ib, jb) = board.cells[a], board.cells[b]
    return int(abs(int(ia) - int(ib)) + abs(int(ja) - int(jb)))


def is_walkable(cell: int, blocked: set[int]) -> bool:
    return cell not in blocked


def reachable_cells(
    board: BoardMap, start: int, movement: int, blocked: set[int]
) -> dict[int, int]:
    """Cases atteignables en <= `movement` pas -> {case: cout en PM}.

    BFS sur le graphe du plateau. La case de depart est incluse (cout 0) meme si elle
    figure dans `blocked` : on s'y trouve deja, ce qui n'empeche pas d'en partir.
    """
    if movement <= 0:
        return {start: 0}

    costs = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        cost = costs[cell]
        if cost >= movement:
            continue
        for neighbour in board.neighbours(cell):
            if neighbour in costs or not is_walkable(neighbour, blocked):
                continue
            costs[neighbour] = cost + 1
            queue.append(neighbour)
    return costs


def path_between(
    board: BoardMap, start: int, target: int, blocked: set[int]
) -> list[int] | None:
    """Plus court chemin `start` -> `target` (exclut le depart), ou None si inatteignable.

    Necessaire pour AGIR : cliquer une case lointaine suppose de savoir par ou l'on
    passe, ne serait-ce que pour verifier que le trajet reste dans les PM.
    """
    if start == target:
        return []

    previous: dict[int, int] = {start: start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbour in board.neighbours(cell):
            if neighbour in previous or not is_walkable(neighbour, blocked):
                continue
            previous[neighbour] = cell
            if neighbour == target:
                path = [target]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return path[-2::-1]  # retire le depart, remet dans l'ordre
            queue.append(neighbour)
    return None
