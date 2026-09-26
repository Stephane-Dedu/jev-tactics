"""Jouer un combat : percevoir, decider, executer, recommencer.

La boucle est simple. Ce qui ne l'est pas, c'est de savoir **si elle a reellement joué**.

Le mode de panne le plus cher du depot d'origine est ici, et il est documente dans son
README : un plan reduit a « fin de tour » s'execute **sans erreur** et marque le tour
comme joue. « 30 tours joues » se lisait donc exactement comme un combat mene au corps a
corps -- alors que le bot n'avait lance aucun sort de toute la partie. Et la cause la plus
frequente n'est pas tactique : un `slot` errone retire le sort d'`available_spells` en
silence, le bot cesse de le lancer, et **un bot qui n'attaque pas ressemble a un bot
prudent**.

D'ou la comptabilite de ce module. `FightReport` ne compte pas seulement les tours : il
compte les tours MUETS, les sorts effectivement lances, et les sorts que l'executeur a
sautes faute de raccourci connu. Un combat entier sans frapper ne s'annonce pas « limite
de tours atteinte » -- dont le travail associe serait « lever la borne », soit deux fois
plus de tours muets.

Tout est injecte : la boucle se joue en test sur des frames fabriquees, sans le jeu.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from jev_tactics.action.actions import Executor
from jev_tactics.calibration.grid import BoardMap
from jev_tactics.decision import Decision
from jev_tactics.planner.legal import Cast
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState

# Au-dela, on considere que le combat ne se conclura pas. La borne protege d'une boucle
# infinie ; elle ne dit RIEN de la qualite du jeu -- c'est `mute_turns` qui le dit.
MAX_TURNS = 30
# Attente entre deux lectures quand ce n'est pas notre tour.
POLL = 0.5

class Decides(Protocol):
    """Ce que la boucle attend d'un decideur -- la meme signature que `Decider`.

    Redeclare ici plutot qu'importe pour eviter un cycle : `decision` importe `planner`,
    qui n'a aucune raison de connaitre la boucle vivante.
    """

    def decide(self, board: BoardMap, state: CombatState,
               spells: list[Spell]) -> Decision: ...


Grab = Callable[[], NDArray[np.uint8]]
ReadState = Callable[[NDArray[np.uint8], int, CombatState | None], CombatState | None]


class Outcome(StrEnum):
    ENDED = "ended"          # la timeline a disparu : le combat est fini
    EXHAUSTED = "exhausted"  # borne de tours atteinte
    BLIND = "blind"          # etat illisible trop longtemps : panne de PERCEPTION


@dataclass
class FightReport:
    """Ce qu'un combat a donne. Les tours muets sont comptes A PART."""

    outcome: Outcome = Outcome.ENDED
    turns: int = 0
    mute_turns: int = 0                                  # tours sans le moindre sort
    spells_cast: int = 0
    skipped_spells: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def fallbacks(self) -> int:
        """Combien de decisions ont du retomber sur le solveur."""
        return sum(1 for d in self.decisions if d.is_fallback)

    @property
    def mute_ratio(self) -> float:
        return self.mute_turns / self.turns if self.turns else 0.0

    def describe(self) -> str:
        """Le bilan affiche la PROPORTION, pas le compte brut.

        Un tour muet isole est normal -- rien a portee, tout en rechargement -- et reste
        une simple observation. Au-dela de la moitie, c'est un avertissement, et la
        premiere chose a verifier est la barre de sorts.
        """
        base = (f"{self.outcome.value} en {self.turns} tour(s), "
                f"{self.spells_cast} sort(s) lance(s)")
        if self.fallbacks:
            base += f", {self.fallbacks} repli(s)"
        if not self.turns:
            return base
        if self.mute_turns == self.turns:
            return (base + " — AUCUN SORT LANCE de tout le combat. Verifier la barre : "
                    "un slot errone retire un sort en silence")
        if self.mute_ratio > 0.5:
            return (base + f" — {self.mute_ratio:.0%} de tours muets, "
                    "c'est beaucoup : verifier la barre de sorts")
        if self.skipped_spells:
            return base + f" — sorts sans raccourci : {sorted(set(self.skipped_spells))}"
        return base


def play_fight(
    board: BoardMap,
    spells: list[Spell],
    grab: Grab,
    read_state: ReadState,
    decider: Decides,
    executor: Executor,
    max_turns: int = MAX_TURNS,
    blind_limit: int = 3,
) -> FightReport:
    """Joue jusqu'a la fin du combat. -> le bilan.

    `read_state` rend None hors combat OU quand l'etat est inexploitable, et les deux
    demandent des reponses opposees : s'arreter dans un cas, reessayer dans l'autre.
    Impossible de les distinguer sur une seule lecture -- une frame de transition est
    illisible sans que le combat soit fini. On tranche donc par la REPETITION :
    `blind_limit` lectures vides d'affilee valent une fin de combat, une seule vaut un
    hoquet. Sans ce compteur, une frame perdue au milieu d'un combat le declarait termine
    et le bot repartait sur son circuit en pleine bagarre.
    """
    report = FightReport()
    previous: CombatState | None = None
    blind = 0

    for _ in range(max_turns):
        state = read_state(grab(), report.turns, previous)

        if state is None:
            blind += 1
            if blind >= blind_limit:
                report.outcome = (
                    Outcome.ENDED if report.turns else Outcome.BLIND)
                return report
            executor.backend.sleep(POLL)
            continue
        blind = 0
        previous = state

        if not state.is_our_turn:
            executor.backend.sleep(POLL)
            continue

        decision = decider.decide(board, state, spells)
        report.decisions.append(decision)

        before = len(executor.skipped)
        executor.execute(decision.plan.actions)
        report.skipped_spells.extend(executor.skipped[before:])

        casts = sum(1 for a in decision.plan.actions if isinstance(a, Cast))
        report.spells_cast += casts
        report.turns += 1
        # LE COMPTE QUI MANQUAIT. Un plan sans le moindre sort s'execute sans erreur et
        # marque le tour joue ; sans cette ligne, trente tours muets se lisent comme
        # trente tours joues.
        if casts == 0:
            report.mute_turns += 1

    report.outcome = Outcome.EXHAUSTED
    return report
