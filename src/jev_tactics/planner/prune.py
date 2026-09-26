"""Elagage : reduire les actions legales a celles qui peuvent changer l'issue du tour.

Distinction volontaire avec `legal_actions` :
  - `legal_actions` renvoie l'ensemble COMPLET des coups jouables. C'est ce qu'il faut
    au masque d'actions du RL : la politique doit pouvoir explorer tout le jouable, y
    compris des coups que nos heuristiques jugent inutiles a tort.
  - `useful_actions` en renvoie un SOUS-ENSEMBLE, elague par des regles de dominance.
    C'est ce qu'il faut au solveur, dont l'arbre est sinon intractable.

Mesure qui motive l'elagage (etats reels, cf. commit precedent) : 94 a 107 actions
legales, dont 70 a 79 lancers pour seulement 4 ou 5 ennemis -- l'ecrasante majorite vise
des cases vides.

Deux regles, celles du doc d'archi §6.1 :

1. **Utilite de la cible.** Un sort offensif dont la zone d'effet ne touche personne ne
   change rien a l'issue. On ne garde que les lancers qui atteignent au moins une entite.

2. **Dominance des deplacements.** Deux cases d'arrivee qui ouvrent exactement les memes
   (sort, cible) sont interchangeables du point de vue de ce tour : on garde la moins
   couteuse en PM. Reste toujours au moins une case par groupe, donc aucune option
   reellement distincte n'est perdue.
"""

from __future__ import annotations

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import (
    Action,
    Cast,
    EndTurn,
    Move,
    TurnState,
    legal_actions,
    movement_options,
)
from jev_tactics.rules.movement import grid_distance
from jev_tactics.rules.spells import Spell, castable_targets
from jev_tactics.state import CombatState, Team


def offensive(spells: list[Spell]) -> list[Spell]:
    """Sorts que le solveur peut effectivement choisir : ceux qui infligent des degats.

    L'evaluation ne note que les degats (aucun soin ni buff n'est modelise) : un sort a
    0 degat ne peut donc JAMAIS ameliorer un score. L'explorer quand meme multiplie
    l'arbre sans rien changer au resultat.

    Mesure sur la configuration reelle du Sacrieur -- 19 sorts, dont 9 utilitaires :
    la recherche saturait le plafond de 60 000 noeuds en 3,1 s, et rendait donc un plan
    TRONQUE. Ecarter les sorts sans degat rend la recherche exhaustive a nouveau.

    Ce filtrage est a REVOIR le jour ou soins et buffs seront modelises -- c'est alors
    l'evaluation qu'il faudra etendre, pas ce filtre qu'il faudra contourner.
    """
    return [spell for spell in spells if spell.damage_max > 0]


def _hits_someone(
    board: BoardMap, spell: Spell, target: int, occupied: set[int]
) -> bool:
    return any(cell in occupied for cell in spell.area_cells(board, target))


def _enabled_shots(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    origin: int,
    enemy_cells: set[int],
) -> frozenset[tuple[str, int]]:
    """(sort, cible) que l'on pourrait lancer depuis `origin` en touchant un ennemi.

    C'est la signature qui sert a comparer deux cases d'arrivee : deux positions offrant
    la meme signature sont equivalentes pour ce tour.
    """
    shots: set[tuple[str, int]] = set()
    blocked = state.blocked_cells()
    for spell in spells:
        for target in castable_targets(board, spell, origin, blocked):
            if _hits_someone(board, spell, target, enemy_cells):
                shots.add((spell.name, target))
    return frozenset(shots)


def useful_targets(
    board: BoardMap, spells: list[Spell], targets: set[int]
) -> set[int]:
    """Cases dont viser touche au moins une entite de `targets` (zones d'effet comprises).

    Restreindre les cibles a cet ensemble est le premier gain : sur les etats reels, les
    ennemis occupent 4 a 5 cases sur 75 a 118, et seules celles-la (plus leur voisinage
    pour les sorts de zone) peuvent produire un effet.
    """
    radius = max((spell.area_radius for spell in spells), default=0)
    if radius == 0:
        return set(targets)
    return {
        cell for cell in range(len(board))
        if any(grid_distance(board, cell, t) <= radius for t in targets)
    }


def build_shot_table(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    origins: list[int],
) -> dict[int, list[Cast]]:
    """Precalcule, pour chaque position possible, les lancers qui touchent un ennemi.

    Les lignes de vue ne dependent que de (origine, cible) et des obstacles : rien de
    tout cela ne change pendant le tour. Les recalculer a chaque noeud etait le goulot du
    solveur (mesure : 66 s par decision, dont l'essentiel en lignes de vue redondantes).
    """
    spells = offensive(spells)
    enemy_cells = {e.cell for e in state.entities if e.team is Team.ENEMY}
    candidates = sorted(useful_targets(board, spells, enemy_cells))
    # Hors des boucles : `blocked_cells()` reconstruit un ensemble a chaque appel, et il
    # est invariant sur tout le tour. L'appeler ici coutait |origines| x |sorts|
    # reconstructions -- environ un millier par decision.
    blocked = state.blocked_cells()

    table: dict[int, list[Cast]] = {}
    for origin in origins:
        shots: list[Cast] = []
        for spell in spells:
            for target in castable_targets(
                board, spell, origin, blocked, candidates
            ):
                # LES CASES ENNEMIES, ET NON TOUTES LES CASES OCCUPEES. Le nom de cette
                # fonction dit « qui touchent un ennemi » ; elle acceptait un lancer des
                # qu'il tombait sur un combattant QUELCONQUE, allie compris. Or
                # `_enabled_shots` -- l'autre source de la meme information, utilisee quand
                # il n'y a pas de table -- ne compte que les ennemis. Deux definitions
                # d'une seule notion, et c'est la table qui sert au vrai solveur.
                #
                # CE QUE CA PRODUISAIT, verifie sur un plan retenu : un ennemi en case 3,
                # un allie colle en case 4, un sort plafonne a un lancer par cible. Le
                # solveur tire sur l'ennemi, puis -- le plafond l'empechant de recommencer
                # -- tire sur la case de l'ALLIE, qui est une cible distincte offerte par
                # la table. `evaluate` ignore les allies, donc ce lancer ne coute rien au
                # score et evite 0,3 de penalite de PA non depense. Trois PA perdus et
                # trente degats sur son propre camp, pour un gain de 0,3.
                if _hits_someone(board, spell, target, enemy_cells):
                    shots.append(
                        Cast(spell=spell.name, target=target, cost=spell.ap_cost))
        table[origin] = shots
    return table


def _killed_so_far(state: CombatState, damage: dict[str, float] | None) -> set[int]:
    """Cases des ennemis que le plan en cours a DEJA tues.

    PV inconnus : jamais consideres comme morts. Supposer une mise a mort qu'on n'a pas
    mesuree ferait abandonner une cible bien vivante -- et le meme principe vaut deja
    dans l'evaluation, qui refuse la prime de mise a mort sans PV connus.
    """
    if not damage:
        return set()
    return {e.cell for e in state.entities
            if e.team is Team.ENEMY and e.hp_known
            and damage.get(e.entity_id, 0.0) >= e.hp}


def useful_actions(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    turn: TurnState,
    shot_table: dict[int, list[Cast]] | None = None,
    damage: dict[str, float] | None = None,
) -> list[Action]:
    """Sous-ensemble elague de `legal_actions`. Toujours au moins EndTurn.

    `shot_table` (cf. `build_shot_table`) evite de recalculer les portees et lignes de
    vue a chaque appel ; sans elle, tout est recalcule -- correct mais lent.

    `damage` : degats deja infliges PAR LE PLAN EN COURS. Sans lui, le solveur continue
    de viser un ennemi que sa propre sequence a deja tue. Signale en jeu sous deux formes
    qui n'en font qu'une : « il frappe un ennemi deja mort », et « il tape une case morte
    (terrain) » -- c'est la meme case, une fois le cadavre disparu.

    Ce n'est pas seulement inutile, c'est ATTIRANT pour le solveur : les degats sont
    plafonnes aux PV restants, donc un lancer sur un mort rapporte 0, mais il depense des
    PA et la penalite de PA non depense (2,0/PA) recule d'autant. Mesure sur un ennemi a
    20 PV face a un sort a 50 degats et 9 PA :

        0 lancer  -> -16,5      2 lancers -> 72,5
        1 lancer  ->  66,5      3 lancers -> 78,5

    Le score MONTE de 6 par lancer sans effet. Le solveur ne se trompait donc pas : il
    optimisait ce qu'on lui avait demande.
    """
    spells = offensive(spells)
    by_name = {spell.name: spell for spell in spells}
    corpses = _killed_so_far(state, damage)
    enemy_cells = {e.cell for e in state.entities
                   if e.team is Team.ENEMY and e.cell not in corpses}

    if turn.ended:
        return []

    # --- Lancers -------------------------------------------------------------------
    kept: list[Action] = []
    if shot_table is not None:
        candidates = shot_table.get(turn.cell, [])
    else:
        occupied = {e.cell for e in state.entities if e.cell != turn.cell}
        candidates = [
            a for a in legal_actions(board, state, spells, turn)
            if isinstance(a, Cast)
            and _hits_someone(board, by_name[a.spell], a.target, occupied)
        ]
    for cast in candidates:
        spell = by_name[cast.spell]
        if spell.ap_cost > turn.ap:
            continue
        # Ne viser que ce qui est encore VIVANT dans ce plan. La zone d'effet compte :
        # un sort qui touche un cadavre ET un survivant reste utile.
        if corpses and not (set(spell.area_cells(board, cast.target)) & enemy_cells):
            continue
        if turn.casts_of(spell.name) >= spell.max_casts_per_turn:
            continue
        if turn.casts_on(spell.name, cast.target) >= spell.max_casts_per_target:
            continue
        kept.append(cast)

    # --- Deplacements : un representant (le moins cher) par signature de tirs --------
    moves = [
        Move(cell=cell, cost=cost)
        for cell, cost in movement_options(board, state, turn).items()
        if cell != turn.cell
    ]

    def signature(origin: int) -> frozenset[tuple[str, int]]:
        if shot_table is not None:
            return frozenset((c.spell, c.target) for c in shot_table.get(origin, []))
        return _enabled_shots(board, state, spells, origin, enemy_cells)

    best_by_signature: dict[frozenset[tuple[str, int]], Move] = {}
    current = signature(turn.cell)
    for move in moves:
        offered = signature(move.cell)
        # Se deplacer sans changer ce qu'on peut viser est domine par rester sur place.
        if offered == current and offered:
            continue
        previous = best_by_signature.get(offered)
        if previous is None or move.cost < previous.cost:
            best_by_signature[offered] = move

    kept.extend(best_by_signature.values())
    kept.append(EndTurn())
    return kept
