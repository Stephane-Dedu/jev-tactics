"""Projection : de `CombatState` vers l'etat compact envoye a Jev.

Deux contraintes, opposees.

**La taille.** L'API plafonne `state` a 8 000 caracteres serialises. Un `CombatState`
brut ne tient pas : le seul ensemble `obstacles` compte plusieurs centaines d'entiers sur
un plateau de 218 cases, et ne dit rien d'utile a qui doit choisir entre huit plans deja
calcules -- la geometrie a DEJA ete consommee par le solveur, qui n'a retenu que des
coups legaux. La reexpedier reviendrait a payer des tokens pour une contrainte deja
appliquee.

**La suffisance.** Ce qu'on retire ici, Jev ne peut pas en tenir compte. C'est le vrai
travail de ce module et il n'est pas neutre : la projection DEFINIT ce sur quoi le modele
raisonne. Une omission ne produit pas une erreur, elle produit un choix moins bon, ce qui
ne se voit que dans le taux de victoire.

Regle de contenu, qui est aussi une regle de securite : **rien de ce qui figure ici n'est
controle par un tiers.** Les `entity_id` sont synthetiques (`"me"`, `"enemy_42"`, `"e3"`,
cf. `perception/entities.py`), jamais des pseudonymes ; les noms de sorts viennent de la
configuration de l'operateur. Aucun texte d'un autre joueur -- pseudo, message -- n'entre
dans `state`. C'est ce qui ferme la surface d'injection decrite par Check Point, ou des
donnees d'apparence legitime glissees dans l'entree font basculer 59 % des verdicts.
`assert_untrusted_free` en fait une verification executable plutot qu'une intention.
"""

from __future__ import annotations

import json
import re
from typing import Any

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.candidates import Candidate
from jev_tactics.planner.legal import Cast, Move
from jev_tactics.rules.movement import grid_distance
from jev_tactics.state import CombatState, Team

# Plafond API. On vise nettement en dessous : la marge absorbe les combats denses
# (beaucoup d'ennemis) sans avoir a tronquer au dernier moment.
STATE_LIMIT = 8_000
STATE_BUDGET = 6_000

# Identifiants acceptables dans un etat projete. Deliberement etroit : si la perception
# se met un jour a nommer les entites d'apres le jeu, ce motif doit echouer.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,48}$")


def project_state(
    board: BoardMap,
    state: CombatState,
    candidates: list[Candidate],
) -> dict[str, Any]:
    """L'etat tactique tel que Jev le voit : des nombres, des distances, pas de pixels.

    Les positions sont traduites en DISTANCES a soi plutot qu'en numeros de case. Un
    indice de cellule n'a aucun sens hors du plateau courant -- il change d'un combat a
    l'autre -- alors qu'une distance se compare. C'est la difference entre une donnee et
    une coordonnee.
    """
    me = state.self_entity()
    enemies = [e for e in state.entities if e.team is Team.ENEMY]
    allies = [e for e in state.entities if e.team is Team.ALLY and not e.is_self]

    return {
        "turn": state.turn,
        "me": {
            "hp": me.hp,
            "hp_max": me.hp_max,
            "hp_pct": round(100 * me.hp / me.hp_max),
            "ap": me.ap,
            "mp": me.mp,
            # Une position non confirmee rend TOUTES les distances suspectes. Le taire
            # serait presenter une incertitude comme une mesure.
            "position_confirmed": me.cell_confirmed,
        },
        "enemies": [
            {
                "id": e.entity_id,
                "hp_pct": round(100 * e.hp / e.hp_max) if e.hp_known else None,
                "hp_known": e.hp_known,
                "distance": grid_distance(board, me.cell, e.cell),
            }
            for e in enemies
        ],
        "allies": [
            {"id": a.entity_id, "distance": grid_distance(board, me.cell, a.cell)}
            for a in allies
        ],
        "enemy_count": len(enemies),
        "plans": [_describe(board, state, c, i) for i, c in enumerate(candidates)],
    }


def _describe(
    board: BoardMap,
    state: CombatState,
    cand: Candidate,
    index: int,
) -> dict[str, Any]:
    """Un plan, decrit par ses CONSEQUENCES et non par ses actions.

    « aller case 122 puis Absorption sur 53 » est illisible pour qui ne connait pas le
    plateau. « se rapprocher a 2 cases, 82 degats sur 1 ennemi, finit a 3 PA » se juge.
    """
    me = state.self_entity()
    moves = [a for a in cand.plan.actions if isinstance(a, Move)]
    casts = [a for a in cand.plan.actions if isinstance(a, Cast)]
    final_cell = moves[-1].cell if moves else me.cell

    enemies = [e for e in state.entities if e.team is Team.ENEMY]
    distances = [grid_distance(board, final_cell, e.cell) for e in enemies]

    # Les degats sont repartis par entite : on distingue « 80 sur un ennemi » de
    # « 40 sur deux », que le total confondrait alors qu'ils ne valent pas la meme chose.
    enemy_ids = {e.entity_id for e in enemies}
    on_enemies = {k: v for k, v in cand.damage.items() if k in enemy_ids}
    friendly_fire = sum(v for k, v in cand.damage.items() if k not in enemy_ids)

    return {
        "id": plan_id(index),
        "spells": [a.spell for a in casts],
        "damage_total": round(sum(on_enemies.values())),
        "damage_per_enemy": {k: round(v) for k, v in sorted(on_enemies.items())},
        "enemies_hit": len(on_enemies),
        # Les degats a son propre camp sont un critere de rejet, pas un detail : le
        # solveur les a longtemps jetes en amont et arbitrait au hasard entre deux cibles
        # a egalite dont l'une epargnait l'allie (cf. `search._apply_damage`).
        "friendly_fire": round(friendly_fire),
        "moves": len(moves),
        "ends_at_distance": min(distances) if distances else None,
        "ap_left": _remaining(me.ap, casts),
        "heuristic_score": round(cand.plan.score, 1),
    }


def _remaining(ap: int, casts: list[Cast]) -> int:
    """PA restants apres les lancers du plan.

    Sans casts la somme vaut zero et la fonction rend `ap` : le cas « plan vide » n'a
    donc pas besoin d'etre garde a l'appel. Il l'etait, par un `a and b or c` qui
    retombait sur `c` des que `b` valait 0 -- c'est-a-dire precisement quand le plan
    depensait TOUS les PA. Chaque plan s'annoncait donc avec ses PA intacts. Rien ne
    plantait : un nombre plausible, simplement faux, comme le veut le mode de panne
    habituel de ce depot.
    """
    return max(0, ap - sum(c.cost for c in casts))


def plan_id(index: int) -> str:
    """Identifiant positionnel, conserve pour la PROJECTION seule.

    Les options soumises a Jev portent desormais des noms d'INTENTION (`achever`,
    `abri`...), et ce n'est pas un indice glisse dans l'enonce : l'intention EST la
    question. On ne demande plus « lequel de ces huit plans » -- question a laquelle huit
    descriptions identiques ne permettaient pas de repondre -- mais « que veux-tu faire ».

    Cet identifiant ne sert plus qu'a reperer les plans dans l'etat projete.
    """
    return f"plan_{index}"


def serialise(projected: dict[str, Any]) -> str:
    """JSON compact, et la garantie de tenir dans la limite de l'API.

    Degrade par ETAGES plutot que de tronquer la chaine : un JSON coupe en deux est
    refuse par l'API (422) alors qu'un JSON allege reste une question valide. On sacrifie
    d'abord le detail par ennemi, puis les allies lointains -- jamais les plans, qui sont
    l'objet meme de la question.
    """
    payload = json.loads(json.dumps(projected))  # copie, on va l'amputer
    text = json.dumps(payload, separators=(",", ":"))
    if len(text) <= STATE_BUDGET:
        return text

    for plan in payload.get("plans", []):
        plan.pop("damage_per_enemy", None)
    text = json.dumps(payload, separators=(",", ":"))
    if len(text) <= STATE_BUDGET:
        return text

    payload["allies"] = payload.get("allies", [])[:3]
    payload["enemies"] = sorted(
        payload.get("enemies", []), key=lambda e: e.get("distance") or 99)[:8]
    text = json.dumps(payload, separators=(",", ":"))
    if len(text) <= STATE_LIMIT:
        return text

    raise ValueError(
        f"etat projete irreductible : {len(text)} caracteres pour un plafond de "
        f"{STATE_LIMIT}. Reduire k dans top_sequences.")


def assert_untrusted_free(projected: dict[str, Any]) -> None:
    """Verifie qu'aucun texte hors de notre controle n'a atteint l'etat projete.

    Leve plutot que de filtrer. Un filtrage silencieux ferait passer pour normale une
    perception qui s'est mise a produire des noms venus du jeu -- exactement le genre de
    changement qui doit arreter la boucle et non la faire continuer autrement.
    """
    for entity in [*projected.get("enemies", []), *projected.get("allies", [])]:
        ident = entity.get("id", "")
        if not _SAFE_ID.match(str(ident)):
            raise ValueError(
                f"identifiant non synthetique dans l'etat projete : {ident!r}. "
                "Les entity_id doivent rester generes (cf. perception/entities.py) ; "
                "un texte venu du jeu ouvrirait une injection sur l'entree de Jev.")
