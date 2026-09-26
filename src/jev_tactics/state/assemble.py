"""Observation (pixels) -> CombatState (entree du planificateur).

Deux problemes que la perception brute ne resout pas, et qui se traitent ici :

1. **Est-ce notre tour ?** La timeline designe le portrait actif, mais pas a quelle
   entite du plateau il correspond. On identifie l'emplacement du JOUEUR par la COULEUR
   de son portrait ; le tour est a nous si cet emplacement est l'actif.

   L'appariement se faisait auparavant par ratio de PV, seul chiffre connu exactement
   (OCR). Il fonctionnait sur les captures de reference :

     dofusscreen   emplacement 0 (0.34) vs OCR 0.37  -> ecart 0.03, non actif
     dofusscreen4  emplacement 0 (0.50) vs OCR 0.51  -> ecart 0.01, actif
     dofusscreen5  emplacement 0 (0.51) vs OCR 0.41  -> ecart 0.10, actif

   ... mais ces trois captures ont un point commun invisible : le combat y est ENGAGE,
   donc les PV differents. En debut de combat tout le monde est a 100 % et le ratio ne
   distingue plus rien -- ce que le jeu reel a montre. Le jeu d'eval partageait un biais
   avec lui-meme ; c'est le genre de trou qu'aucun nombre de captures similaires ne
   comble. La couleur d'equipe, elle, ne depend d'aucun degat subi (cf.
   `identify_self_slot`).

2. **Suivi d'identite entre tours.** Les entites sont nommees d'apres leur case, qui
   change des qu'elles bougent. On reapparie donc chaque entite a la plus proche du tour
   precedent (meme equipe), ce qui conserve un `entity_id` stable -- necessaire pour
   parler d'une cible d'un tour a l'autre.
"""

from __future__ import annotations

from collections.abc import Callable

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception.timeline import TimelineEntry
from jev_tactics.perception.ui import UiReading
from jev_tactics.rules import grid_distance
from jev_tactics.state import CombatState, Entity, Team

# Au-dela de cette distance (en cases), on considere que c'est une AUTRE entite plutot
# qu'un deplacement : evite de recycler l'identite d'un mort sur une invocation lointaine.
MAX_TRACKING_DISTANCE = 8


def identify_self_slot(timeline: list[TimelineEntry], ui: UiReading) -> int | None:
    """Emplacement de timeline correspondant au joueur.

    L'EQUIPE PRIME SUR LES PV, et c'est un defaut constate en jeu qui l'impose. La
    version precedente appariait par ratio de PV seul ; or **en debut de combat tout le
    monde est a 100 %** -- joueur 1662/1662 et monstres 1000/1000 donnent tous un ratio
    de 1,0. `min()` rendait alors le premier emplacement venu, souvent un monstre, et le
    bot en concluait « tour adverse » pendant son propre tour : il attendait
    indefiniment un tour qu'il avait deja. C'est le cas le PLUS FREQUENT, pas un cas
    limite -- tout combat commence ainsi.

    La couleur du portrait, elle, ne depend d'aucun degat subi. Elle est donc consultee
    d'abord ; les PV ne servent plus qu'a departager entre allies (combat de groupe).
    """
    if not timeline:
        return None

    allies = [e for e in timeline if e.team is Team.ALLY]
    if len(allies) == 1:
        return allies[0].slot

    # Plusieurs allies, ou aucune equipe lue : les PV departagent, faute de mieux. Sans
    # lecture d'UI il ne reste aucune evidence -- rendre un emplacement reviendrait a
    # deviner, et un « c'est notre tour » invente fait cliquer hors tour.
    if ui.pv is None or not ui.pv_max:
        return None
    target = ui.pv / ui.pv_max
    return min(allies or timeline, key=lambda e: abs(e.hp_ratio - target)).slot


def is_our_turn(timeline: list[TimelineEntry], ui: UiReading) -> bool:
    """Vrai si l'emplacement du joueur est le portrait actif."""
    slot = identify_self_slot(timeline, ui)
    if slot is None:
        return False
    return any(e.slot == slot and e.is_active for e in timeline)


def _distance_fn(board: BoardMap | None) -> Callable[[int, int], int]:
    """Fonction de distance pour l'appariement.

    Avec un plateau, c'est la vraie distance de grille. Sans lui, on retombe sur l'ecart
    d'indices : ce n'est PAS une distance geometrique (les indices sont seulement
    ordonnes), mais deux entites proches sur le plateau ont des indices voisins, ce qui
    suffit a un appariement de secours.
    """
    if board is None:
        return lambda a, b: abs(a - b)
    return lambda a, b: grid_distance(board, a, b)


def track_identities(
    previous: list[Entity], current: list[Entity], board: BoardMap | None = None
) -> list[Entity]:
    """Reattribue aux entites courantes l'`entity_id` de leur correspondante precedente.

    Appariement glouton par proximite, au sein d'une meme equipe. Une entite sans
    correspondance garde son identifiant derive de sa case (nouvelle entite, ou invocation).
    """
    if not previous:
        return current

    distance = _distance_fn(board)
    unused = list(previous)
    tracked: list[Entity] = []
    for entity in current:
        candidates = [
            p for p in unused
            if p.team is entity.team
            and distance(p.cell, entity.cell) <= MAX_TRACKING_DISTANCE
        ]
        if candidates:
            match = min(candidates, key=lambda p: distance(p.cell, entity.cell))
            unused.remove(match)
            tracked.append(entity.model_copy(update={"entity_id": match.entity_id}))
        else:
            tracked.append(entity)
    return tracked


def assemble_combat_state(
    observation,  # jev_tactics.pipeline.Observation (import differe : evite un cycle)
    turn: int = 0,
    previous: CombatState | None = None,
    board: BoardMap | None = None,
) -> CombatState | None:
    """Observation -> CombatState, ou None si l'etat n'est pas exploitable.

    Renvoie None hors combat, ou si le personnage controle n'a pas ete localise : mieux
    vaut pas d'etat qu'un etat dont le planificateur tirerait des decisions fausses.
    """
    if not observation.timeline:
        return None  # hors combat

    entities = observation.entities
    if not any(e.is_self for e in entities):
        return None  # joueur non localise : etat inexploitable

    if previous is not None:
        entities = track_identities(previous.entities, entities, board)

    our_turn = is_our_turn(observation.timeline, observation.ui)
    self_entity = next(e for e in entities if e.is_self)

    # Le surlignage du jeu ne vaut que pendant notre tour ; ailleurs il est absent ou
    # decrit le deplacement de quelqu'un d'autre.
    hint = getattr(observation, "reachable", None)
    reachable_hint = set(hint) if (our_turn and hint) else None

    return CombatState(
        turn=turn,
        active_entity_id=self_entity.entity_id if our_turn else None,
        is_our_turn=our_turn,
        entities=entities,
        reachable_hint=reachable_hint,
    )
