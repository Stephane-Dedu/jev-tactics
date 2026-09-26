"""Actions legales d'un tour, et leur application.

Brique partagee : le solveur exhaustif l'utilise comme GENERATEUR, le futur PPO comme
MASQUE d'actions. Sans masquage, l'immense majorite des actions d'un etat sont illegales
(PA/PM insuffisants, hors portee, case occupee) et PPO gaspille son exploration ; avec,
il n'echantillonne que du jouable. Les deux consomment le meme code, donc la politique
ne peut pas apprendre des regles differentes de celles du solveur.

`TurnState` est immuable et hachable : le solveur explore en empilant des etats sans
copie defensive, et peut dedupliquer les sequences qui aboutissent au meme etat.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic import BaseModel

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception.highlight import blocked_from_highlight
from jev_tactics.rules.movement import reachable_cells
from jev_tactics.rules.spells import Spell, castable_targets
from jev_tactics.state import CombatState


class Move(BaseModel):
    """Deplacement vers une case atteignable. `cost` est le nombre de PM consommes."""

    kind: str = "move"
    cell: int
    cost: int


class Cast(BaseModel):
    """Lancer de sort sur une case. `cost` est le nombre de PA consommes."""

    kind: str = "cast"
    spell: str
    target: int
    cost: int


class EndTurn(BaseModel):
    kind: str = "end_turn"


Action = Move | Cast | EndTurn


@dataclass(frozen=True)
class TurnState:
    """Etat du tour en cours : position et ressources du personnage actif.

    `casts` conserve l'historique (sort, cible) du tour, seul moyen d'appliquer les
    plafonds par tour et par cible. Le tuple le rend hachable, donc l'etat entier est
    utilisable comme cle de deduplication dans la recherche.
    """

    cell: int
    ap: int
    mp: int
    casts: tuple[tuple[str, int], ...] = ()
    ended: bool = False

    def casts_of(self, spell: str) -> int:
        return sum(1 for name, _ in self.casts if name == spell)

    def casts_on(self, spell: str, target: int) -> int:
        return sum(1 for name, cell in self.casts if name == spell and cell == target)

    @classmethod
    def from_combat(cls, state: CombatState) -> TurnState:
        me = state.self_entity()
        return cls(cell=me.cell, ap=me.ap, mp=me.mp)


def movement_options(
    board: BoardMap, state: CombatState, turn: TurnState
) -> dict[int, int]:
    """Destinations atteignables -> cout en PM.

    Si le jeu a surligne sa zone de deplacement (`reachable_hint`), elle fait autorite :
    c'est le resultat de SON pathfinding, obstacles compris, alors que le notre ne
    connait que les obstacles qu'on a su detecter. On lance quand meme un BFS a
    l'interieur de cette zone, car le surlignage donne les cases mais pas leur cout.

    LA ZONE RESTE VALIDE APRES UN DEPLACEMENT, et cette version-ci abandonnait la zone
    des qu'on quittait la case de depart, au motif qu'elle « ne decrit plus la
    situation ». C'est faux, et demontrable : si X est atteignable depuis A avec les PM
    restants, alors le chemin depart -> A -> X coute au plus le total de PM, donc X etait
    deja atteignable au depart -- donc X est dans la zone. La zone est un MAJORANT qui
    survit au deplacement.

    Ce que coutait l'abandon, mesure sur un personnage tacle a qui le jeu n'accorde qu'un
    pas : au depart 2 cases, correct ; apres ce pas, le BFS en propose 12, dont 11 que le
    jeu avait declarees hors d'atteinte. Le solveur planifiait donc une traversee de
    plateau apres le premier pas -- et la table de tir, construite sur les seules cases de
    depart, ne repondait rien pour ces cases-la : les lancers y etaient perdus EN SILENCE
    (`shot_table.get(cell, [])`).

    La plausibilite se juge sur la case de DEPART, parce que c'est elle que le
    surlignage decrit -- « la zone touche-t-elle le personnage ? » n'a de sens que la.
    Aucune consequence mesuree apres un seul pas, la case de depart restant voisine ;
    c'est de la coherence, pas un correctif.
    """
    others = {e.cell for e in state.entities if e.cell != turn.cell}
    hint = state.reachable_hint
    me = next((e for e in state.entities if e.is_self), None)
    start = me.cell if me is not None else turn.cell

    # `turn.cell in hint or turn.cell == start` : on ne suit la zone que si l'on s'y
    # trouve. Une recherche normale n'en sort jamais -- elle ne se deplace que vers les
    # cases que cette fonction propose -- mais une position incoherente (etat mal
    # assemble, zone lue apres coup) rendrait `blocked_from_highlight` paralysant, en
    # declarant infranchissable jusqu'a la case sous nos pieds.
    inside = turn.cell == start or turn.cell in (hint or ())
    if hint is not None and inside and _plausible_hint(board, hint, start):
        blocked = blocked_from_highlight(board, hint, start)
    else:
        blocked = state.obstacles | others

    return reachable_cells(board, turn.cell, turn.mp, blocked)


def _plausible_hint(board: BoardMap, hint: set[int], start: int) -> bool:
    """La zone surlignee touche-t-elle le personnage ?

    Le surlignage fait autorite QUAND il est lu correctement -- c'est le pathfinding du
    jeu, obstacles compris. Mais `blocked_from_highlight` declare infranchissable tout ce
    qui n'y figure pas : une lecture PARTIELLE ne degrade donc pas la qualite, elle
    paralyse. Constate en jeu : « 0 cases atteignables (zone du jeu) » avec 4 PM, tour
    passe pour rien -- alors que le meme bot, quand le surlignage sort VIDE et qu'on
    retombe sur le BFS, trouve 17 cases et joue correctement.

    Une vraie zone de deplacement est contigue et part du personnage. Si aucune case
    voisine n'y figure, ce n'est pas une zone : mieux vaut l'ignorer et retomber sur le
    BFS, qui est le comportement d'avant et qui marche.
    """
    return any(cell in hint for cell in board.neighbours(start))


def legal_actions(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    turn: TurnState,
) -> list[Action]:
    """Toutes les actions jouables depuis `turn`. Toujours au moins EndTurn.

    Les cases occupees par d'autres combattants bloquent le deplacement mais restent
    ciblables : c'est justement sur elles qu'on veut lancer des sorts.
    """
    if turn.ended:
        return []

    actions: list[Action] = []

    for cell, cost in movement_options(board, state, turn).items():
        if cell != turn.cell:
            actions.append(Move(cell=cell, cost=cost))

    for spell in spells:
        if spell.ap_cost > turn.ap:
            continue
        if turn.casts_of(spell.name) >= spell.max_casts_per_turn:
            continue
        # Les autres combattants BLOQUENT la visee -- c'est une regle du jeu, et
        # l'ignorer faisait planifier des tirs a travers les monstres. `blocked_cells`
        # ajoute les occupants au decor ; `has_line_of_sight` exclut deja le lanceur et
        # la cible, donc viser un ennemi reste possible.
        for target in castable_targets(board, spell, turn.cell, state.blocked_cells()):
            if turn.casts_on(spell.name, target) >= spell.max_casts_per_target:
                continue
            actions.append(Cast(spell=spell.name, target=target, cost=spell.ap_cost))

    actions.append(EndTurn())
    return actions


def apply_action(turn: TurnState, action: Action) -> TurnState:
    """Applique une action au tour en cours -> nouvel etat (l'ancien est inchange)."""
    if isinstance(action, Move):
        return replace(turn, cell=action.cell, mp=turn.mp - action.cost)
    if isinstance(action, Cast):
        return replace(
            turn,
            ap=turn.ap - action.cost,
            casts=turn.casts + ((action.spell, action.target),),
        )
    return replace(turn, ended=True)
