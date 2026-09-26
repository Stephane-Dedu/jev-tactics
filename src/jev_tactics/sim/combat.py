"""Simulateur de combat : appliquer un tour et faire avancer l'etat.

C'est l'investissement qui conditionne le RL. Le point critique n'est PAS l'algorithme
d'apprentissage mais la fidelite du simulateur : une politique entrainee sur des regles
fausses joue faux, et rien dans les courbes d'apprentissage ne le signale.

D'ou la regle posee des le depart du projet : **le simulateur consomme `rules/`**, le
meme code que le solveur et que le generateur d'actions legales. Reimplementer les
portees ou les deplacements ici creerait deux verites qui divergeraient en silence.

Ce que le simulateur modelise, et ce qu'il ne modelise pas -- a savoir, car c'est la
mesure de sa fidelite :
  - modelise : deplacements (PM), lancers (PA), portees, lignes de vue, zones d'effet,
    plafonds de lancer, degats moyens, morts, fin de combat ;
  - ne modelise PAS : resistances, coups critiques, etats et buffs, invocations,
    poussees, initiative variable. Les degats sont pris a leur MOYENNE, donc deterministes.
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import Action, Cast, EndTurn, Move, TurnState, apply_action
from jev_tactics.rules.movement import grid_distance, reachable_cells
from jev_tactics.rules.spells import Spell, castable_targets
from jev_tactics.state import CombatState, Entity, Team


@dataclass(frozen=True)
class TurnOutcome:
    """Resultat de l'application d'un tour."""

    state: CombatState
    damage_dealt: float = 0.0
    kills: int = 0
    illegal: list[Action] = None  # actions refusees (utile pour deboguer une politique)

    def __post_init__(self) -> None:
        if self.illegal is None:
            object.__setattr__(self, "illegal", [])


def _hit_entities(
    board: BoardMap, state: CombatState, spell: Spell, target: int
) -> list[Entity]:
    cells = set(spell.area_cells(board, target))
    return [e for e in state.entities if e.cell in cells]


def apply_actions(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    actions: list[Action],
    actor_id: str | None = None,
) -> TurnOutcome:
    """Applique une sequence d'actions et renvoie l'etat resultant.

    Les actions ILLEGALES sont ignorees et rapportees plutot que levees : pendant
    l'entrainement, une politique en proposera forcement, et faire echouer l'episode
    rendrait le signal d'apprentissage inutilisable. On veut savoir combien, pas planter.
    """
    by_name = {spell.name: spell for spell in spells}
    entities = {e.entity_id: e.model_copy(deep=True) for e in state.entities}
    actor = (actor_id or next(e.entity_id for e in state.entities if e.is_self))
    if actor not in entities:
        return TurnOutcome(state=state)

    me = entities[actor]
    turn = TurnState(cell=me.cell, ap=me.ap, mp=me.mp)
    damage_total = 0.0
    kills = 0
    refused: list[Action] = []

    for action in actions:
        if isinstance(action, EndTurn):
            break

        if isinstance(action, Move):
            occupied = {e.cell for e in entities.values() if e.entity_id != actor}
            blocked = state.obstacles | occupied
            reachable = reachable_cells(board, turn.cell, turn.mp, blocked)
            if action.cell not in reachable:
                refused.append(action)
                continue
            turn = apply_action(turn, Move(cell=action.cell, cost=reachable[action.cell]))
            entities[actor].cell = turn.cell
            continue

        if isinstance(action, Cast):
            spell = by_name.get(action.spell)
            if spell is None or spell.ap_cost > turn.ap:
                refused.append(action)
                continue
            if turn.casts_of(spell.name) >= spell.max_casts_per_turn:
                refused.append(action)
                continue
            if turn.casts_on(spell.name, action.target) >= spell.max_casts_per_target:
                refused.append(action)
                continue
            if action.target not in castable_targets(
                board, spell, turn.cell, state.obstacles, [action.target]
            ):
                refused.append(action)
                continue

            turn = apply_action(turn, action)
            # Instantane des entites AVANT degats : une entite qui meurt sur ce lancer
            # doit quand meme avoir ete touchee.
            snapshot = CombatState(turn=state.turn, entities=list(entities.values()),
                                   obstacles=state.obstacles)
            for victim in _hit_entities(board, snapshot, spell, action.target):
                if victim.entity_id == actor:
                    continue
                target_entity = entities[victim.entity_id]
                dealt = min(spell.average_damage, float(target_entity.hp))
                target_entity.hp = max(0, target_entity.hp - int(round(dealt)))
                if victim.team is Team.ENEMY:
                    damage_total += dealt
                    if target_entity.hp == 0:
                        kills += 1

    survivors = [e for e in entities.values() if e.hp > 0]
    for entity in survivors:
        if entity.entity_id == actor:
            entity.cell, entity.ap, entity.mp = turn.cell, turn.ap, turn.mp

    return TurnOutcome(
        state=state.model_copy(update={"entities": survivors}),
        damage_dealt=damage_total, kills=kills, illegal=refused,
    )


def enemy_policy(
    board: BoardMap, state: CombatState, spells: list[Spell], enemy_id: str
) -> list[Action]:
    """Adversaire de reference : avancer vers le joueur et frapper si possible.

    Volontairement simple et LISIBLE. Son role n'est pas de bien jouer mais de fournir
    une pression reproductible : si une politique apprise ne bat pas cet adversaire-la,
    le probleme est ailleurs que dans sa subtilite.
    """
    enemy = next((e for e in state.entities if e.entity_id == enemy_id), None)
    target = next((e for e in state.entities if e.is_self), None)
    if enemy is None or target is None:
        return []

    actions: list[Action] = []
    turn = TurnState(cell=enemy.cell, ap=enemy.ap, mp=enemy.mp)
    occupied = {e.cell for e in state.entities if e.entity_id != enemy_id}
    blocked = state.obstacles | occupied

    # Se rapprocher : la case atteignable la plus proche de la cible.
    reachable = reachable_cells(board, turn.cell, turn.mp, blocked)
    best = min(reachable, key=lambda c: grid_distance(board, c, target.cell))
    if best != turn.cell:
        actions.append(Move(cell=best, cost=reachable[best]))
        turn = apply_action(turn, actions[-1])

    # Puis frapper tant que les PA le permettent.
    for spell in sorted(spells, key=lambda s: -s.average_damage):
        while (spell.ap_cost <= turn.ap
               and turn.casts_of(spell.name) < spell.max_casts_per_turn
               and target.cell in castable_targets(
                   board, spell, turn.cell, state.obstacles, [target.cell])):
            cast = Cast(spell=spell.name, target=target.cell, cost=spell.ap_cost)
            actions.append(cast)
            turn = apply_action(turn, cast)
    return actions


def winner(state: CombatState) -> Team | None:
    """Equipe gagnante, ou None si le combat continue."""
    allies = [e for e in state.entities if e.team is Team.ALLY]
    enemies = [e for e in state.entities if e.team is Team.ENEMY]
    if not enemies:
        return Team.ALLY
    if not allies:
        return Team.ENEMY
    return None
