"""Decideur Jev : le solveur propose, le modele dispose.

Repartition des roles, et la raison de chacune :

  - **le code enumere.** `top_sequences` construit des plans a partir de `legal_actions`.
    Un plan illegal n'est pas rejete, il n'est pas construit.
  - **le code elague.** `evaluate` classe et coupe a K. On ne demande pas a Jev de
    parcourir 10^8 sequences ; on lui en soumet huit, deja distinctes.
  - **Jev tranche.** Une question `choice` sur les identifiants de plans. La reponse est
    un indice, jamais un coup.

Pourquoi ce partage et pas « Jev choisit l'action » : l'espace d'actions de ce jeu vaut
`1 + cases x (1 + sorts)`, soit 5 233 indices avec la configuration reelle (218 cases,
23 sorts). Une question `choice` en accepte 255 au plus. L'ecart n'est pas un reglage,
c'est un ordre de grandeur -- et de toute facon un tour n'est pas une action mais une
SEQUENCE, que la primitive ne sait pas exprimer.

Ce que Jev remplace reellement, c'est `scoring.py` : 14 ko de poids regles a la main pour
departager des plans. C'est exactement une tache de jugement typee, et c'est la que le
modele a une chance d'etre meilleur que nous.
"""

from __future__ import annotations

from typing import Any

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.decision import Decision
from jev_tactics.decision.projection import (
    assert_untrusted_free,
    plan_id,
    project_state,
    serialise,
)
from jev_tactics.decision.search import SearchDecider
from jev_tactics.decision.transport import Transport
from jev_tactics.planner.candidates import DEFAULT_K, top_sequences
from jev_tactics.planner.search import DEFAULT_MAX_NODES
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState

# En deca de ce seuil, la distribution de Jev est trop plate pour valoir mieux que
# l'heuristique : on garde le plan le mieux classe par `evaluate`. Le motif est celui du
# « confidence-gated routing » de la documentation Jev. La valeur est un POINT DE DEPART,
# pas une mesure : la regler demande de tracer le taux de victoire en arene contre le
# seuil, ce que ce depot n'a pas encore fait.
DEFAULT_MIN_CONFIDENCE = 0.35

INSTRUCTIONS = (
    "Tu choisis le plan a jouer pour ce tour dans un combat tactique au tour par tour. "
    "L'etat donne ta situation (points de vie, PA, PM, ennemis et leur distance) puis la "
    "liste des plans deja verifies comme jouables. Chaque plan indique les degats "
    "infliges, le nombre d'ennemis touches, les degats a ton propre camp, et la distance "
    "a l'ennemi le plus proche a la fin du tour. "
    "Privilegie les degats concentres qui achevent un ennemi plutot que repartis ; "
    "evite les degats a ton propre camp ; ne finis pas au contact quand tes points de vie "
    "sont bas. Reponds uniquement par l'identifiant du plan."
)


class JevDecider:
    """Choisit parmi les plans candidats via une question `choice` a Jev.

    Ne leve jamais du fait du reseau : toute panne -- delai, quota, reponse inattendue --
    se resout en repli sur le solveur, trace dans `Decision.source`.
    """

    name = "jev"

    def __init__(
        self,
        transport: Transport,
        k: int = DEFAULT_K,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_nodes: int = DEFAULT_MAX_NODES,
    ):
        self.transport = transport
        self.k = k
        self.min_confidence = min_confidence
        self.max_nodes = max_nodes
        self._fallback = SearchDecider(max_nodes=max_nodes)

    def decide(
        self,
        board: BoardMap,
        state: CombatState,
        spells: list[Spell],
    ) -> Decision:
        candidates = top_sequences(
            board, state, spells, k=self.k, max_nodes=self.max_nodes)

        # Un seul candidat : il n'y a rien a trancher. On s'epargne l'appel, sa latence
        # et son cout. Ce cas n'est pas rare -- en fin de combat, a court de PA, le tour
        # se reduit souvent a « frapper » ou « passer ».
        if len(candidates) <= 1:
            best = candidates[0] if candidates else None
            if best is None:
                return self._fallback.decide(board, state, spells)
            return Decision(
                plan=best.plan, source="jev", candidates=1,
                reason=f"{best.plan.describe()} [seul plan possible, sans appel]")

        projected = project_state(board, state, candidates)
        assert_untrusted_free(projected)

        questions = {
            "plan": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {
                    plan_id(i): _criterion(projected["plans"][i])
                    for i in range(len(candidates))
                },
            }
        }

        try:
            payload = serialise(projected)
            answer = self.transport(payload, questions)["answers"]["plan"]
            chosen = answer["choice"]
            confidence = float(answer.get("confidence") or 0.0)
        except Exception as exc:  # repli volontairement large : aucune panne ne doit
            # arreter le combat, et les modes d'echec d'un appel distant sont ouverts
            # (reseau, quota, schema, cle absente, cassette manquante).
            decision = self._fallback.decide(board, state, spells)
            return Decision(
                plan=decision.plan,
                source=f"fallback:{type(exc).__name__}",
                reason=f"{decision.reason} [repli : {exc}]",
                candidates=len(candidates),
            )

        index = _index_of(chosen, len(candidates))
        if index is None:
            # Le modele a rendu une etiquette hors de l'ensemble propose. Impossible en
            # principe, donc a tracer et non a rattraper en silence.
            decision = self._fallback.decide(board, state, spells)
            return Decision(
                plan=decision.plan,
                source="fallback:unknown_choice",
                reason=f"{decision.reason} [choix hors ensemble : {chosen!r}]",
                candidates=len(candidates),
            )

        if confidence < self.min_confidence:
            best = candidates[0]
            return Decision(
                plan=best.plan,
                source="fallback:low_confidence",
                reason=(f"{best.plan.describe()} [confiance {confidence:.2f} < "
                        f"{self.min_confidence:.2f}, heuristique conservee]"),
                confidence=confidence,
                candidates=len(candidates),
            )

        picked = candidates[index]
        # L'ecart au classement heuristique est la SEULE mesure interessante du journal :
        # un decideur qui choisit toujours plan_0 n'apporte rien et coute un appel.
        return Decision(
            plan=picked.plan,
            source="jev",
            reason=(f"{picked.plan.describe()} "
                    f"[{chosen}, confiance {confidence:.2f}, rang heuristique {index}]"),
            confidence=confidence,
            candidates=len(candidates),
        )


def _criterion(plan: dict[str, Any]) -> str:
    """Description d'une option, en une ligne.

    `criteria` est plafonne (2 000 caracteres serialises par question) : avec huit plans
    cela laisse ~240 caracteres chacun, largement de quoi tenir cette phrase. La
    documentation insiste sur le fait que les descriptions doivent SEPARER les options ;
    on n'y met donc que ce qui varie d'un plan a l'autre.
    """
    bits = [f"{plan['damage_total']} degats"]
    if plan["enemies_hit"] > 1:
        bits.append(f"sur {plan['enemies_hit']} ennemis")
    if plan["friendly_fire"]:
        bits.append(f"dont {plan['friendly_fire']} sur allie")
    if plan["spells"]:
        bits.append("sorts " + "+".join(plan["spells"]))
    else:
        bits.append("aucun sort")
    if plan["ends_at_distance"] is not None:
        bits.append(f"finit a {plan['ends_at_distance']} cases")
    return ", ".join(bits)


def _index_of(choice: str, count: int) -> int | None:
    """Traduit l'etiquette rendue en indice, ou None si elle n'est pas des notres.

    Le controle de bornes est la garantie centrale de cette couche : quoi que rende le
    modele, il ne peut designer qu'un plan que nous avons construit.
    """
    for i in range(count):
        if choice == plan_id(i):
            return i
    return None
