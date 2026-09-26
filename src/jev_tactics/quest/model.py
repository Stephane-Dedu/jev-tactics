"""Le modele : une quete, ses etapes, et l'avancement d'un personnage dedans.

DEUX CHOSES SEPAREES, et c'est la decision structurante de ce module.

`Quest` est une DEFINITION : elle decrit la quete telle qu'elle existe dans le jeu, la
meme pour tout le monde, et ne change jamais. `Progress` est l'avancement d'UN personnage
dans cette quete, qui change a chaque objectif rempli.

Les melanger -- un seul objet portant `objectives` et `done` -- rendrait impossible de
charger les quetes depuis un fichier partage, de rejouer un avancement hors-ligne, ou
simplement de savoir si un ecart vient de la definition ou de l'etat. C'est la meme
raison qui fait de `CombatState` un objet serialisable a part : on doit pouvoir rejouer
une decision sans le jeu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from jev_tactics.world.navigation import Coord


class Step(StrEnum):
    """Ce qu'une etape demande. Volontairement peu de formes.

    Chacune correspond a une capacite que le bot possede DEJA, ou qu'il lui manque de
    facon identifiee -- pas a une taxonomie des quetes du jeu. Ajouter une forme que rien
    ne sait executer donnerait un directeur qui rend des intentions mortes.
    """

    TALK = "talk"          # parler a un PNJ            -> manque la perception des PNJ
    KILL = "kill"          # tuer N monstres            -> world.engage + decision
    HARVEST = "harvest"    # recolter N ressources      -> perception.resources
    GOTO = "goto"          # se rendre sur une carte    -> world.walker
    FETCH = "fetch"        # obtenir N objets           -> consequence de KILL/HARVEST


@dataclass(frozen=True)
class Objective:
    """Une etape. `where` est la carte ou elle peut etre accomplie, si elle est fixee.

    `where=None` signifie « n'importe ou » -- tuer des bouftous se fait sur beaucoup de
    cartes. Ne pas confondre avec « on ne sait pas ou » : une etape dont le lieu est
    inconnu n'est pas jouable, et le directeur le dit plutot que de faire errer le bot.
    """

    kind: Step
    target: str = ""
    count: int = 1
    where: Coord | None = None

    def describe(self) -> str:
        lieu = f" en {self.where}" if self.where else ""
        if self.kind is Step.GOTO:
            return f"se rendre en {self.where}"
        if self.count > 1:
            return f"{self.kind.value} {self.count}x {self.target}{lieu}"
        return f"{self.kind.value} {self.target}{lieu}"


@dataclass(frozen=True)
class Quest:
    """La definition. Immuable, partageable, chargeable depuis un fichier."""

    name: str
    giver: str
    giver_at: Coord
    objectives: tuple[Objective, ...]
    # Rendre la quete a QUELQU'UN D'AUTRE est le cas courant, pas l'exception : c'est
    # precisement ce que « finir en parlant au bon PNJ » demande de modeliser.
    turn_in: str = ""
    turn_in_at: Coord | None = None

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError(f"quete « {self.name} » sans objectif : rien a accomplir")

    @property
    def returns_to_giver(self) -> bool:
        return not self.turn_in or self.turn_in == self.giver

    def handed_to(self) -> tuple[str, Coord]:
        """Le PNJ a qui rendre, et ou. Par defaut le donneur."""
        if self.returns_to_giver:
            return self.giver, self.giver_at
        assert self.turn_in_at is not None, (
            f"« {self.name} » se rend a {self.turn_in} mais sa carte est inconnue")
        return self.turn_in, self.turn_in_at


@dataclass
class Progress:
    """L'avancement d'un personnage. Le seul objet mutable du module.

    `counted` porte l'avancement de l'etape COURANTE seulement. Un compteur par etape
    serait plus riche, mais rien ne sait encore le relire depuis le jeu : le journal de
    quetes n'est pas percu. Poser une structure que la perception ne peut pas remplir
    donnerait un etat qui derive en silence -- on s'en tient a ce qui est verifiable.
    """

    quest: Quest
    index: int = 0
    counted: int = 0
    accepted: bool = False
    handed_in: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def current(self) -> Objective | None:
        """L'etape en cours, ou None si toutes sont faites."""
        if self.index >= len(self.quest.objectives):
            return None
        return self.quest.objectives[self.index]

    @property
    def objectives_done(self) -> bool:
        return self.index >= len(self.quest.objectives)

    @property
    def complete(self) -> bool:
        return self.handed_in

    def advance(self, by: int = 1) -> None:
        """Compte `by` unites sur l'etape courante, et passe a la suivante si elle est pleine.

        Ne deborde JAMAIS sur l'etape suivante. Tuer trois bouftous d'un sort de zone
        quand il en fallait deux ne doit pas faire sauter l'etape d'apres : le surplus est
        perdu, ce que le jeu fait aussi, et surtout on ne veut pas d'un avancement qui
        depende de l'ordre des evenements.
        """
        objective = self.current
        if objective is None:
            return
        self.counted = min(objective.count, self.counted + by)
        if self.counted >= objective.count:
            self.index += 1
            self.counted = 0

    def skip(self, why: str) -> None:
        """Abandonne l'etape courante, en disant pourquoi.

        Une etape infaisable -- PNJ introuvable, carte inconnue -- doit pouvoir etre
        passee, sinon une seule quete bloquee arrete le bot pour la session. La RAISON est
        obligatoire : un saut silencieux se lirait comme un objectif accompli.
        """
        self.notes.append(f"etape {self.index} passee : {why}")
        self.index += 1
        self.counted = 0
