"""Solveur 1 tour : meilleure sequence d'actions, par recherche en profondeur elaguee.

Role double, comme prevu au plan :
  - **planificateur** utilisable tel quel (deterministe, explicable, sans entrainement) ;
  - **oracle** pour initialiser la politique RL par imitation -- on ne part pas de zero.

L'arbre brut etant intractable (mesure : ~10^8 sequences), la recherche s'appuie sur :
  - l'elagage de `useful_actions` (cibles utiles, deplacements dominants) ;
  - la **deduplication par etat** : deux ordres d'actions menant au meme TurnState avec
    les memes degats sont equivalents, on n'explore la suite qu'une fois. C'est ce qui
    absorbe l'explosion combinatoire due aux permutations de sorts ;
  - un **plafond de noeuds** qui garantit une reponse en temps borne. Depasse, la
    recherche renvoie le meilleur trouve jusque-la : degrader la qualite est acceptable,
    bloquer le bot ne l'est pas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import (
    Action,
    Cast,
    EndTurn,
    Move,
    TurnState,
    apply_action,
    movement_options,
)
from jev_tactics.planner.prune import build_shot_table, useful_actions
from jev_tactics.planner.scoring import evaluate
from jev_tactics.rules.movement import grid_distance
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState, Team

# Plafond de noeuds. MESURE en arene avec la vraie configuration (20 sorts, plateau de
# 81 cases, 2 ennemis) :
#
#     plafond   60 000   20 000   6 000   2 000
#     victoires  58,3%    58,3%   58,3%   50,0%
#     tronques    0/33     0/33    1/33   12/35
#     ms/combat    187      187     170     135
#
# Deux enseignements. Le plafond actuel est DIX FOIS plus haut que necessaire -- il n'est
# jamais atteint, et une decision coute ~190 ms, pas les secondes que je supposais. Et
# c'est bien la TRONCATURE qui coute des victoires : a 2 000, un tiers des decisions sont
# amputees et le taux chute de huit points.
#
# On le laisse haut : puisqu'il n'est pas atteint, il ne coute rien, et il protege des
# etats pathologiques (beaucoup d'ennemis, beaucoup de cases atteignables) que l'arene ne
# produit pas. Le baisser n'apporterait qu'un risque de troncature.
DEFAULT_MAX_NODES = 60_000


@dataclass
class Plan:
    """Resultat du solveur : la sequence retenue et de quoi la justifier."""

    actions: list[Action] = field(default_factory=list)
    score: float = float("-inf")
    nodes: int = 0
    truncated: bool = False       # plafond de noeuds atteint

    def describe(self) -> str:
        if not self.actions:
            return "passer le tour"
        parts = []
        for action in self.actions:
            if isinstance(action, Move):
                parts.append(f"aller case {action.cell} (-{action.cost} PM)")
            elif isinstance(action, Cast):
                parts.append(f"{action.spell} sur {action.target} (-{action.cost} PA)")
        return " puis ".join(parts) if parts else "passer le tour"


def _apply_damage(
    board: BoardMap,
    state: CombatState,
    spell: Spell,
    target: int,
    damage: dict[str, float],
) -> dict[str, float]:
    """Repartit les degats d'un lancer sur TOUT ce qui est touche (zone incluse).

    Les allies aussi, et c'est le point. Ce filtre ne gardait que les ennemis, si bien que
    les degats infliges a son propre camp etaient jetes ICI, en amont : l'evaluation ne
    pouvait pas les penaliser meme si elle l'avait voulu.

    Mesure de l'aveuglement, sur un sort de zone a 40 degats touchant un ennemi et un
    allie : score 47,2 avec l'allie touche, 47,2 sans. Exactement zero. Entre deux cibles
    a egalite -- l'une epargnant l'allie, l'autre non -- le solveur tranchait donc au
    hasard de l'ordre d'enumeration.

    `_killed_so_far` filtre deja sur `Team.ENEMY` : ajouter des cles d'allies ne peut pas
    lui faire prendre un ami pour un cadavre.
    """
    hit = set(spell.area_cells(board, target))
    updated = dict(damage)
    for entity in state.entities:
        if entity.cell in hit:
            updated[entity.entity_id] = (
                updated.get(entity.entity_id, 0.0) + spell.average_damage
            )
    return updated


def best_sequence(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    max_nodes: int = DEFAULT_MAX_NODES,
) -> Plan:
    """Meilleure sequence d'actions pour le tour courant."""
    by_name = {spell.name: spell for spell in spells}
    plan = Plan()
    seen: set[tuple] = set()
    nodes = 0

    # Toutes les positions ou l'on peut se trouver ce tour-ci : portees et lignes de vue
    # y sont precalculees une fois pour toutes (cf. build_shot_table).
    # `movement_options` et non un BFS local. Le solveur tirait ses positions candidates
    # d'un calcul, et ses deplacements LEGAUX d'un autre : les Move produits pendant la
    # recherche passent par `legal_actions`, qui honore deja la zone surlignee par le jeu.
    # Deux sources pour une meme question, donc deux reponses possibles.
    #
    # Ecart MESURE entre les deux, sur les captures reelles :
    #
    #     combat1.png   12 cases calculees contre 10 surlignees
    #     tacle.png     20 cases calculees contre  3 surlignees
    #     bug4.png      19 / 19        combat1920.png  34 / 34
    #
    # `tacle.png` porte ce nom parce que le personnage y est TACLE : il perd ses PM, le
    # jeu le sait, notre BFS l'ignore. La zone du jeu FAIT AUTORITE -- c'est le resultat
    # de son propre pathfinding, obstacles et etats compris. Ce raisonnement etait deja
    # ecrit dans `legal.py` et applique la-bas.
    #
    # Le dict (case -> cout en PM) est conserve : le repli d'approche a besoin des
    # couts pour annoncer un deplacement honnete.
    origins = movement_options(board, state, TurnState.from_combat(state))
    shot_table = build_shot_table(board, state, spells, list(origins))

    def visit(turn: TurnState, damage: dict[str, float], path: list[Action]) -> None:
        nonlocal nodes, plan

        score = evaluate(board, state, turn, damage, spells)
        if score > plan.score:
            plan = Plan(actions=list(path), score=score, nodes=nodes)

        if nodes >= max_nodes:
            plan.truncated = True
            return

        last_was_move = bool(path) and isinstance(path[-1], Move)
        # `damage` transmis : sans lui, l'elagage ne peut pas savoir qui la sequence
        # en cours a deja tue, et le solveur continue de viser un cadavre.
        for action in useful_actions(board, state, spells, turn, shot_table, damage):
            # Le plafond se verifie AUSSI dans la boucle : sinon un seul noeud tres
            # ramifie le depasserait largement avant le prochain controle.
            if nodes >= max_nodes:
                plan.truncated = True
                return
            if isinstance(action, EndTurn):
                continue  # deja evalue : le noeud courant EST la fin de tour
            # Deux deplacements consecutifs sont domines par le trajet direct, que le
            # BFS trouve deja : les enchainer ne fait qu'allonger l'arbre.
            if last_was_move and isinstance(action, Move):
                continue
            nodes += 1

            next_turn = apply_action(turn, action)
            next_damage = damage
            if isinstance(action, Cast):
                next_damage = _apply_damage(
                    board, state, by_name[action.spell], action.target, damage)

            # Deux ordres menant au meme etat + memes degats sont equivalents.
            key = (next_turn.cell, next_turn.ap, next_turn.mp,
                   tuple(sorted(next_turn.casts)), tuple(sorted(next_damage.items())))
            if key in seen:
                continue
            seen.add(key)

            path.append(action)
            visit(next_turn, next_damage, path)
            path.pop()

    visit(TurnState.from_combat(state), {}, [])
    plan.nodes = nodes

    if not any(isinstance(a, Cast) for a in plan.actions):
        approach = approach_move(board, state, spells, origins)
        if approach is not None:
            plan = Plan(actions=[approach], score=plan.score, nodes=nodes)
    return plan


def approach_move(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    origins: dict[int, int] | list[int],
) -> Move | None:
    """Deplacement qui rapproche de l'ennemi le plus proche, ou None si inutile.

    Repli EXPLICITE, hors evaluation, pour un cas que l'horizon d'un seul tour ne peut
    pas traiter. Quand aucun ennemi n'est a portee depuis aucune case atteignable, le
    tour ne rapporte rien quoi qu'on fasse ; le terme de securite recompense alors la
    distance, et rester sur place bat tout le reste. Constate en jeu : trois ennemis,
    10 PA, et « passer le tour » -- indefiniment, puisque la situation se reproduit au
    tour suivant. **Avec des sorts de portee 1 a 3, le bot ne pouvait jamais engager.**

    Corriger l'evaluation aurait demande de lui faire valoir un gain FUTUR (etre a portee
    au tour prochain), ce qu'un horizon d'un tour ne represente pas. Et la toucher aurait
    remis en cause les 81 % mesures. Le repli est donc pose a cote : il ne s'applique que
    lorsque le plan ne contient aucun lancer, c'est-a-dire quand il n'y avait rien a
    perdre.
    """
    enemies = [e.cell for e in state.entities if e.team is Team.ENEMY]
    if not enemies:
        return None

    costs = origins if isinstance(origins, dict) else {c: 0 for c in origins}
    me = state.self_entity()
    here = min(grid_distance(board, me.cell, c) for c in enemies)

    # On ne se colle pas : il suffit d'etre a portee de tir DU TOUR PROCHAIN, c'est-a-dire
    # a `portee + PM` de l'ennemi. Aller plus pres n'ouvre aucun coup supplementaire et
    # ne fait qu'avancer le moment ou l'on encaisse.
    #
    # Ce que la mesure en dit, et ses limites. L'approche coute 4 points en arene
    # (81 % -> 77 %), et le palier n'en recupere qu'une partie. Mais **l'arene ne peut
    # pas juger ce changement** : son adversaire de reference se contente de reduire la
    # distance, donc y camper est toujours gratuit. En jeu reel, ne jamais engager ne
    # coute pas 4 points -- ca empeche le combat d'avoir lieu. Reserve deja consignee au
    # §6.1.b du doc d'archi, et voici le cas qu'elle annoncait.
    # Portee des sorts OFFENSIFS seulement. Compter les utilitaires fausse le palier :
    # Attirance porte a 10 sans infliger de degats, ce qui donnait un palier de 14 cases
    # -- exactement la distance ou le bot se trouvait, donc plus aucune approche. Il
    # restait plante, avec 11 PA, a une distance ou aucun sort ne pouvait rien.
    reach = max((s.range_max for s in spells if s.damage_max > 0), default=0)
    floor = max(1, reach + me.mp) if reach else 1
    if here <= floor:
        return None

    best, best_distance, best_cost = None, here, 0
    for cell, cost in costs.items():
        if cell == me.cell:
            continue
        distance = max(floor, min(grid_distance(board, cell, c) for c in enemies))
        # A egalite de rapprochement, le moins de PM depenses : garder des PM ouvre
        # davantage d'options au tour suivant.
        if distance < best_distance or (distance == best_distance and best is not None
                                        and cost < best_cost):
            best, best_distance, best_cost = cell, distance, cost
    if best is None:
        return None
    return Move(cell=best, cost=best_cost)
