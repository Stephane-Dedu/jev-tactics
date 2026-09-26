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
    project_state,
    serialise,
)
from jev_tactics.decision.search import SearchDecider
from jev_tactics.decision.transport import Transport
from jev_tactics.planner.postures import distinct_enough, posture_plans
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
    "Tu choisis l'INTENTION tactique du tour dans un combat au tour par tour. "
    "L'etat donne ta situation (points de vie, PA, PM, ennemis et leur distance), puis "
    "une option par intention -- chacune est le meilleur plan jouable sous cette "
    "intention, deja verifie comme legal. "
    "Choisis d'achever quand un ennemi peut tomber ce tour ; de te mettre a l'abri quand "
    "tes points de vie sont bas ou que plusieurs ennemis sont au contact ; de frapper "
    "fort quand tu es en securite ; de te placer quand rien n'est a portee utile. "
    "Reponds uniquement par le nom de l'intention."
)


class JevDecider:
    """Choisit une INTENTION tactique via une question `choice` a Jev.

    Il n'y a plus de `k` : le nombre d'options n'est pas un reglage, c'est le nombre
    d'intentions qui donnent des plans reellement differents dans CETTE position. Il vaut
    parfois un -- et l'appel est alors saute, faute de dilemme a trancher.

    Ne leve jamais du fait du reseau : toute panne -- delai, quota, reponse inattendue --
    se resout en repli sur le solveur, trace dans `Decision.source`.
    """

    name = "jev"

    def __init__(
        self,
        transport: Transport,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_nodes: int = DEFAULT_MAX_NODES,
    ):
        self.transport = transport
        self.min_confidence = min_confidence
        self.max_nodes = max_nodes
        self._fallback = SearchDecider(max_nodes=max_nodes)

    def decide(
        self,
        board: BoardMap,
        state: CombatState,
        spells: list[Spell],
    ) -> Decision:
        plans = posture_plans(
            board, state, spells, max_nodes=self.max_nodes)

        # Rien a trancher : une seule intention optimale, ou aucune. On s'epargne l'appel,
        # sa latence et son cout. Ce n'est pas rare et ce n'est pas un echec -- c'est
        # meme l'information la plus honnete que la position puisse donner. MESURE sur
        # `sacrieur.json` : a 10 PA et 200 PV, 17 tours sur 40 n'offrent qu'une intention,
        # parce qu'un personnage surpuissant n'a pas de dilemme. A 6 PA et 60 PV, AUCUN
        # tour n'est dans ce cas. Les postures comptent quand le bot est en difficulte,
        # c'est-a-dire quand la decision compte.
        if not distinct_enough(plans):
            if not plans:
                return self._fallback.decide(board, state, spells)
            posture, best = plans[0]
            return Decision(
                plan=best.plan, source="jev", candidates=1,
                reason=f"{best.plan.describe()} [seule intention : {posture.name}, "
                       f"sans appel]")

        candidates = [cand for _, cand in plans]
        projected = project_state(board, state, candidates)
        assert_untrusted_free(projected)

        questions = {
            "plan": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {
                    posture.name: _criterion(posture, projected["plans"][i])
                    for i, (posture, _) in enumerate(plans)
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
                candidates=len(plans),
            )

        index = next((i for i, (p, _) in enumerate(plans) if p.name == chosen), None)
        if index is None:
            # Une intention hors de l'ensemble propose. Impossible en principe, donc a
            # tracer et non a rattraper en silence.
            decision = self._fallback.decide(board, state, spells)
            return Decision(
                plan=decision.plan,
                source="fallback:unknown_choice",
                reason=f"{decision.reason} [intention inconnue : {chosen!r}]",
                candidates=len(plans),
            )

        if confidence < self.min_confidence:
            # plans[0] est la REFERENCE : le plan qu'aurait choisi le solveur seul.
            reference, best = plans[0]
            return Decision(
                plan=best.plan,
                source="fallback:low_confidence",
                reason=(f"{best.plan.describe()} [confiance {confidence:.2f} < "
                        f"{self.min_confidence:.2f}, {reference.name} conservee]"),
                confidence=confidence,
                candidates=len(plans),
            )

        posture, picked = plans[index]
        # L'ECART A LA REFERENCE est la seule mesure interessante du journal : un decideur
        # qui choisit toujours la reference n'apporte rien et coute un appel.
        return Decision(
            plan=picked.plan,
            source="jev",
            reason=(f"{picked.plan.describe()} "
                    f"[{posture.name}, confiance {confidence:.2f}"
                    f"{'' if index else ', = reference'}]"),
            confidence=confidence,
            candidates=len(plans),
        )


def _criterion(posture: Any, plan: dict[str, Any]) -> str:
    """Description d'une option, en une ligne.

    `criteria` est plafonne (2 000 caracteres serialises par question) : avec huit plans
    cela laisse ~240 caracteres chacun, largement de quoi tenir cette phrase. La
    documentation insiste sur le fait que les descriptions doivent SEPARER les options ;
    on n'y met donc que ce qui varie d'un plan a l'autre.
    """
    bits = [posture.intent, f"{plan['damage_total']} degats"]
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
