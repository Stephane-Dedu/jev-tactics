"""La couche de DECISION : une seule interface, plusieurs decideurs interchangeables.

C'est le point unique ou l'on choisit quoi jouer. Tout le reste du depot -- perception,
legalite, execution -- ignore QUI decide. Sans cette contrainte, brancher Jev signifierait
le cabler dans chaque appelant, et l'on ne pourrait plus le comparer au solveur : or la
comparaison est tout l'objet du projet (reference mesuree : 79 % de victoires en arene).

Invariant, non negociable : **un decideur CHOISIT parmi des plans engendres ici**, il n'en
formule jamais. `top_sequences` construit les candidats a partir de `legal_actions`, la
meme brique qui alimente le masque d'actions du RL. Un decideur ne peut donc pas produire
un coup illegal -- non pas parce qu'on le valide apres coup, mais parce qu'il n'a jamais
la main sur autre chose qu'un INDICE dans une liste.

Cette distinction est la seule defense qui tienne face a un modele : Check Point a montre
(septembre 2026) qu'une sortie typee n'est pas une frontiere de securite, et qu'un modele
de decision se manipule par son ENTREE. Un indice hors bornes est rejete par le code ;
un plan mal forme n'existe pas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.candidates import Candidate
from jev_tactics.planner.search import Plan
from jev_tactics.rules.spells import Spell
from jev_tactics.state import CombatState

__all__ = ["Candidate", "Decider", "Decision"]


@dataclass(frozen=True)
class Decision:
    """Le plan retenu, et de quoi rendre des comptes.

    `source` et `reason` ne sont pas du confort de journalisation. Le mode de panne
    redoute de ce depot est documente dans le README : un bot qui n'attaque pas ressemble
    a un bot prudent. Un repli silencieux sur le solveur -- ou sur la fin de tour -- se
    lirait exactement comme une decision deliberee. Ces deux champs sont ce qui permet a
    la session de compter les replis au lieu de les subir.
    """

    plan: Plan
    source: str                      # "search" | "jev" | "fallback:<cause>"
    reason: str
    confidence: float | None = None  # renseignee par les decideurs qui en exposent une
    candidates: int = 1              # taille de l'eventail soumis

    @property
    def is_fallback(self) -> bool:
        return self.source.startswith("fallback")


class Decider(Protocol):
    """Tout ce qu'un decideur doit savoir faire.

    Volontairement nu : meme signature que `best_sequence`, pour qu'un decideur puisse
    remplacer le solveur partout ou il est appele sans adapter l'appelant.
    """

    name: str

    def decide(
        self,
        board: BoardMap,
        state: CombatState,
        spells: list[Spell],
    ) -> Decision: ...
