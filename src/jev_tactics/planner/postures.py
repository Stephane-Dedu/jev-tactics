"""Des POSTURES, pas des variantes : un plan par intention tactique.

Le defaut que ce module corrige est mesure. `top_sequences` rendait les K plans les mieux
notes ; avec la configuration reelle (`sacrieur.json`, 20 sorts), **16 options sur 24
etaient redondantes** -- memes sorts sur les memes cibles, degats 93 contre 93, seule la
case d'arrivee differait. Demander a Jev de trancher entre quatre descriptions identiques
est un appel paye pour rien.

Et le choix compte : forcer le premier plan donne 62 % de victoires, forcer le dernier
35 %. Vingt-sept points separaient deux options que rien ne distinguait dans l'enonce.

LA CORRECTION N'EST PAS DE MIEUX DEDUPLIQUER. Deux plans peuvent etre reellement
differents et rester indescriptibles -- « 93 degats en finissant a 3 cases » contre
« 93 degats en finissant a 4 cases » ne se juge pas. Ce qu'il faut presenter, ce sont des
INTENTIONS : frapper fort, achever, se mettre a l'abri, se placer. Chacune est le plan
optimal sous un objectif different, et les quatre se lisent d'un coup d'oeil.

Le solveur ne change pas. Ce qui change est l'objectif qu'on lui donne -- et c'est
exactement la question qu'un modele de decision typee sait trancher, la ou enumerer des
sequences ne l'est pas.

COUT : une seule exploration. Les plans sont collectes une fois, puis RENOTES sous chaque
ponderation (`Candidate.turn` existe pour cela). Cinq parcours complets coutaient cinq
fois le temps de decision ; une renotation coute un appel a `evaluate` par plan.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.candidates import Candidate, top_sequences
from jev_tactics.planner.scoring import DEFAULT_WEIGHTS, Weights, evaluate
from jev_tactics.planner.search import DEFAULT_MAX_NODES
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState

# Taille du vivier renote. Le vivier est classe par la ponderation PAR DEFAUT ; le
# tronquer revient donc a laisser l'heuristique decider quels plans les autres postures
# ont le droit de considerer -- exactement le biais qu'on cherche a retirer. Un plan que
# le defaut classe six-centieme peut etre le meilleur sous « se mettre a l'abri ».
#
# MESURE : 612 plans distincts sur un tour de `sacrieur.json` en regime contraint, contre
# une borne fixee a 400. Le vivier etait donc bel et bien coupe, et rien ne le disait.
# La borne existe encore pour empecher un cas pathologique de faire exploser la
# renotation, mais elle est posee tres au-dessus de ce qui a ete observe -- et
# `posture_plans` previent quand elle mord.
POOL = 5000


@dataclass(frozen=True)
class Posture:
    """Une intention tactique, et la ponderation qui l'exprime."""

    name: str
    intent: str          # une ligne, telle qu'elle sera soumise au decideur
    weights: Weights


# La REFERENCE vient en tete et n'est pas negociable : c'est le plan qu'aurait choisi le
# solveur seul, donc celui sur lequel on retombe quand Jev est indisponible ou peu sur.
# Sans elle, un repli ne vaudrait plus les 60 % mesures.
REFERENCE = Posture(
    "reference",
    "le meilleur compromis selon l'heuristique du solveur",
    DEFAULT_WEIGHTS,
)

POSTURES: tuple[Posture, ...] = (
    REFERENCE,
    Posture(
        "degats",
        "infliger le maximum de degats, sans se soucier de la position",
        # Tout sauf les degats est neutralise : ni prime de mise a mort, ni prudence, ni
        # penalite de ressources. Le score DEVIENT la somme des degats utiles.
        Weights(kill_bonus=0.0, safety=0.0, wounded=0.0,
                wasted_ap=0.0, wasted_mp=0.0),
    ),
    Posture(
        "achever",
        "tuer un ennemi ce tour, meme pour moins de degats totaux",
        # La prime domine tout le reste : un ennemi mort vaut plus que n'importe quel
        # total de degats repartis. C'est la posture qui repond a « il en reste un a 12 PV ».
        Weights(kill_bonus=240.0, wounded=40.0, safety=0.0),
    ),
    Posture(
        "abri",
        "s'eloigner le plus possible, quitte a frapper moins",
        # La prudence l'emporte sur les degats. Utile quand on est bas en PV -- et c'est
        # precisement le regime ou `danger_factor` amplifie deja ce terme.
        Weights(safety=12.0, kill_bonus=0.0, wounded=0.0),
    ),
    Posture(
        "position",
        "se placer pour le tour suivant, en frappant si possible",
        # Penalise les PM NON depenses : on veut bouger. La prudence reste moderee pour
        # que le deplacement serve a quelque chose plutot qu'a fuir.
        Weights(safety=5.0, wasted_mp=1.0, wasted_ap=0.0),
    ),
)


def _rescore(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    cand: Candidate,
    weights: Weights,
) -> float:
    """Note un plan deja construit sous une autre ponderation.

    Un candidat sans `turn` n'est pas renotable -- cas du repli d'approche, fabrique hors
    recherche. On lui rend son score d'origine plutot que de l'ecarter : c'est souvent le
    seul plan disponible, et l'ecarter rendrait une liste vide.
    """
    if cand.turn is None:
        return cand.plan.score
    return evaluate(board, state, cand.turn, cand.damage, spells, weights=weights)


def posture_plans(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    max_nodes: int = DEFAULT_MAX_NODES,
    postures: tuple[Posture, ...] = POSTURES,
) -> list[tuple[Posture, Candidate]]:
    """Le meilleur plan sous chaque intention, sans doublon.

    La REFERENCE est toujours en premiere position quand elle existe : c'est l'invariant
    du repli. Les postures qui retombent sur un plan deja retenu sont ecartees -- mieux
    vaut trois options distinctes que cinq dont deux se repetent, puisque c'est
    exactement le defaut qu'on corrige.
    """
    pool = top_sequences(board, state, spells, k=POOL, max_nodes=max_nodes)
    if not pool:
        return []
    if len(pool) >= POOL:
        # Le vivier a ete coupe : les postures choisissent parmi un sous-ensemble filtre
        # par l'heuristique par defaut. Ce n'est pas fatal, mais c'est un biais silencieux
        # -- et un biais silencieux sur le choix est precisement ce que ce module corrige.
        warnings.warn(
            f"vivier tronque a {POOL} plans : les postures ne voient qu'un sous-ensemble "
            "classe par la ponderation par defaut. Relever POOL.",
            RuntimeWarning, stacklevel=2)

    chosen: list[tuple[Posture, Candidate]] = []
    seen: set[tuple[object, ...]] = set()
    for posture in postures:
        best = max(pool, key=lambda c: _rescore(board, state, spells, c, posture.weights))
        sig = best.signature()
        if sig in seen:
            continue
        seen.add(sig)
        chosen.append((posture, best))
    return chosen


def distinct_enough(plans: list[tuple[Posture, Candidate]]) -> bool:
    """Y a-t-il de quoi poser une question ?

    En dessous de deux options reellement differentes, l'appel n'a pas d'objet : on paie
    une latence et des tokens pour un choix force. L'appelant s'en sert pour SAUTER
    l'appel -- ce qui arrive legitimement en fin de combat, a court de PA, quand le tour
    se reduit a « frapper » ou « passer ».
    """
    return len(plans) >= 2
