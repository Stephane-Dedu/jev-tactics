"""Generateur de plans CANDIDATS : les K meilleures sequences, pas seulement la premiere.

`best_sequence` resout un argmax : il rend UN plan, celui qui maximise `evaluate`. C'est
ce qu'il faut quand la fonction de score fait autorite. Ici elle ne fait plus autorite --
c'est justement ce qu'on veut confier a Jev -- donc il faut lui presenter un CHOIX.

Pourquoi un module separe plutot qu'un parametre `k` sur `best_sequence` :

  - les deux fonctions n'ont pas le meme objectif. L'argmax veut le maximum ; le
    generateur veut un eventail *lisible et distinct*. Un top-K naif rend K variantes du
    meme plan a un PM pres, sur lesquelles aucun juge -- humain ou modele -- ne peut se
    prononcer utilement. La deduplication par signature ci-dessous n'a aucun sens pour
    l'argmax, qui se moque des ex aequo ;
  - `best_sequence` est couvert par `test_search.py` et sert de REFERENCE mesuree
    (79 % de victoires en arene). Le plan de travail est de comparer Jev a ce chiffre :
    une refonte du solveur au moment ou l'on change de decideur rendrait l'ecart
    ininterpretable. On ne touche pas a la regle pendant qu'on change le joueur.

Le cout assume est la duplication du parcours en profondeur. Les briques -- elagage,
application, evaluation -- restent partagees ; seule la boucle de collecte differe.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from jev_tactics.planner.search import (
    DEFAULT_MAX_NODES,
    Plan,
    _apply_damage,
    approach_move,
)
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState

# Nombre de candidats par defaut. L'API accepte beaucoup plus (255 options annoncees),
# mais le facteur limitant n'est pas la : c'est la place occupee par les descriptions
# dans `criteria`, et surtout la capacite a DIFFERENCIER. Huit plans distincts couvrent
# deja l'eventail utile d'un tour (frapper / approcher / se replier / temporiser) ;
# au-dela on paie des tokens pour des nuances que rien ne permet de departager.
DEFAULT_K = 8


@dataclass(frozen=True)
class Candidate:
    """Un plan propose au decideur, avec de quoi le decrire sans recalculer."""

    plan: Plan
    damage: dict[str, float]
    # L'ETAT DE FIN DE TOUR, garde pour pouvoir renoter le plan sans relancer la
    # recherche. C'est ce qui rend les postures abordables : une seule exploration, puis
    # une reevaluation par objectif -- au lieu de cinq parcours complets.
    turn: TurnState | None = None

    @property
    def total_damage(self) -> float:
        return sum(self.damage.values())

    def signature(self) -> tuple:
        """Identite d'un plan POUR LE CHOIX, pas pour l'execution.

        Deux sequences qui finissent au meme endroit, lancent les memes sorts sur les
        memes cibles et infligent les memes degats sont le meme plan du point de vue de
        qui doit trancher -- meme si l'ordre des actions differe. Les presenter toutes
        les deux gaspille une option et brouille la distribution de probabilites.
        """
        casts = tuple(sorted(
            (a.spell, a.target) for a in self.plan.actions if isinstance(a, Cast)))
        cell = next((a.cell for a in reversed(self.plan.actions)
                     if isinstance(a, Move)), None)
        return (cell, casts)


def top_sequences(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    k: int = DEFAULT_K,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> list[Candidate]:
    """Les `k` meilleurs plans DISTINCTS du tour, du meilleur au moins bon selon `evaluate`.

    `evaluate` sert ici d'ELAGAGE, pas de verdict : il ordonne les candidats et decide
    lesquels survivent, le choix final revenant au decideur. C'est une nuance qui a son
    importance -- un plan que `evaluate` classe cinquieme reste presentable, un plan
    qu'il n'a jamais produit est invisible. Le biais du scoring n'est donc pas supprime,
    il est REDUIT a un role de filtre. Le mesurer est le travail de l'arene.

    Rend toujours au moins un candidat : la fin de tour est un plan valide (liste
    d'actions vide), et un decideur sans option ne peut rien rendre du tout.
    """
    by_name = {spell.name: spell for spell in spells}
    best: dict[tuple, Candidate] = {}
    seen: set[tuple] = set()
    nodes = 0
    truncated = False

    origins = movement_options(board, state, TurnState.from_combat(state))
    shot_table = build_shot_table(board, state, spells, list(origins))

    def keep(path: list[Action], score: float, damage: dict[str, float],
             turn_state: TurnState) -> None:
        cand = Candidate(
            plan=Plan(actions=list(path), score=score, nodes=nodes, truncated=truncated),
            damage=dict(damage),
            turn=turn_state,
        )
        sig = cand.signature()
        # A signature egale on garde le mieux note : c'est le meme plan, joue mieux.
        previous = best.get(sig)
        if previous is None or score > previous.plan.score:
            best[sig] = cand

    def visit(turn: TurnState, damage: dict[str, float], path: list[Action]) -> None:
        nonlocal nodes, truncated

        keep(path, evaluate(board, state, turn, damage, spells), damage, turn)

        if nodes >= max_nodes:
            truncated = True
            return

        last_was_move = bool(path) and isinstance(path[-1], Move)
        for action in useful_actions(board, state, spells, turn, shot_table, damage):
            if nodes >= max_nodes:
                truncated = True
                return
            if isinstance(action, EndTurn):
                continue
            if last_was_move and isinstance(action, Move):
                continue
            nodes += 1

            next_turn = apply_action(turn, action)
            next_damage = damage
            if isinstance(action, Cast):
                next_damage = _apply_damage(
                    board, state, by_name[action.spell], action.target, damage)

            key = (next_turn.cell, next_turn.ap, next_turn.mp,
                   tuple(sorted(next_turn.casts)), tuple(sorted(next_damage.items())))
            if key in seen:
                continue
            seen.add(key)

            path.append(action)
            visit(next_turn, next_damage, path)
            path.pop()

    visit(TurnState.from_combat(state), {}, [])

    ranked = sorted(best.values(), key=lambda c: c.plan.score, reverse=True)[:k]

    # LE REPLI D'APPROCHE, que `best_sequence` applique et qui manquait ici.
    #
    # Quand aucun ennemi n'est a portee depuis aucune case atteignable, tous les plans
    # engendres ci-dessus se valent a zero degat, et `evaluate` -- dont le terme de
    # securite recompense la distance -- classe « rester sur place » en tete. Le solveur
    # corrige ce cas hors evaluation en substituant un deplacement d'approche ; sans la
    # meme correction, le premier candidat propose de ne rien faire.
    #
    # Ce n'est pas une divergence theorique. Mesure sur 60 scenarios d'arene : sur l'un
    # d'eux, `candidates[0]` ne lance rien la ou `best_sequence` engage. Un tour sur
    # soixante passe a ne pas avancer, indefiniment puisque la situation se reproduit --
    # et un bot qui n'attaque pas ressemble a un bot prudent.
    #
    # Le candidat d'approche est place EN TETE : il doit etre ce sur quoi l'on retombe
    # quand Jev est indisponible ou peu sur de lui, exactement comme le solveur.
    if ranked and not any(isinstance(a, Cast) for a in ranked[0].plan.actions):
        approach = approach_move(board, state, spells, origins)
        if approach is not None:
            ranked.insert(0, Candidate(
                plan=Plan(actions=[approach], score=ranked[0].plan.score,
                          nodes=nodes, truncated=truncated),
                damage={},
                turn=apply_action(TurnState.from_combat(state), approach),
            ))
            ranked = ranked[:k]
    # La troncature n'est connue qu'a la fin du parcours, alors que les Plan ont ete
    # construits en cours de route : sans cette reprise, un plan retenu tot porterait
    # `truncated=False` dans une recherche qui a bel et bien ete amputee. La session
    # compte ces decisions -- elle compterait faux.
    if truncated:
        ranked = [
            Candidate(
                plan=Plan(actions=c.plan.actions, score=c.plan.score,
                          nodes=nodes, truncated=True),
                damage=c.damage,
                turn=c.turn,
            )
            for c in ranked
        ]
    return ranked
