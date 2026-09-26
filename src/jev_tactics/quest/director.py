"""Le directeur : etant donne une quete en cours et ou l'on se trouve, que faire ensuite.

Fonction PURE. Elle ne regarde rien, ne clique rien, n'attend rien -- elle rend une
INTENTION que les couches deja ecrites savent executer. C'est ce qui permet de rejouer
tout le deroule d'une quete hors-ligne, sans le jeu, et de voir le bot « se tromper » dans
un test plutot qu'au bout de quarante minutes de session.

C'est aussi la couche qui relie enfin les trois briques faites : se deplacer
(`world.walker`), engager (`world.engage`), gagner (`decision`). Une quete n'est rien
d'autre qu'une SEQUENCE de ces trois choses, plus des dialogues.

CE QUI N'EST PAS ENCORE EXECUTABLE EST DIT ICI, dans le code, pas dans un document :
`Intent.implemented` vaut faux pour tout ce qui demande la perception des PNJ et des
dialogues, qui n'existe pas. Un test le verifie, et echouera le jour ou cette perception
arrivera sans que le directeur soit mis a jour -- ce qui est exactement le moment ou il
faut y penser.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jev_tactics.quest.model import Objective, Progress, Step
from jev_tactics.world.navigation import Coord


class Act(StrEnum):
    """Ce que le bot doit faire maintenant."""

    TRAVEL = "travel"      # rejoindre une carte        -> world.walker.travel_to
    TALK = "talk"          # parler a un PNJ            -> MANQUE
    FIGHT = "fight"        # engager puis gagner        -> world.engage + decision
    HARVEST = "harvest"    # recolter                   -> perception.resources
    STEP_DONE = "step_done"  # l'etape courante est satisfaite -- CONTINUER
    DONE = "done"            # la quete entiere est finie -- s'arreter
    BLOCKED = "blocked"      # rien de jouable, et on dit pourquoi


# Ce que le bot sait reellement faire aujourd'hui. Tenu a jour ICI plutot que dans un
# document : `test_director.py` compare cet ensemble aux capacites reelles.
EXECUTABLE: frozenset[Act] = frozenset(
    {Act.TRAVEL, Act.FIGHT, Act.HARVEST, Act.STEP_DONE, Act.DONE})


@dataclass(frozen=True)
class Intent:
    """Une action a mener, et de quoi la mener."""

    act: Act
    target: str = ""
    where: Coord | None = None
    count: int = 1
    why: str = ""

    @property
    def implemented(self) -> bool:
        """Le bot sait-il executer cette intention ?

        Rendre une intention inexecutable n'est pas un defaut : c'est la reponse juste,
        et la seule qui permette a l'appelant de passer a autre chose plutot que de
        s'arreter. Une intention manquante deguisee en BLOCKED perdrait l'information de
        ce qu'il FAUDRAIT faire.
        """
        return self.act in EXECUTABLE

    def describe(self) -> str:
        lieu = f" en {self.where}" if self.where else ""
        quoi = f" {self.target}" if self.target else ""
        nombre = f" x{self.count}" if self.count > 1 else ""
        manque = "" if self.implemented else "  [non executable : perception manquante]"
        return f"{self.act.value}{quoi}{nombre}{lieu}{manque}"


def next_intent(progress: Progress, here: Coord) -> Intent:
    """L'action suivante pour cette quete, depuis cette carte.

    L'ordre des tests est la logique de la quete, et il se lit de haut en bas :
    finie ? -> pas encore acceptee ? -> objectifs finis ? -> etape courante.
    """
    quest = progress.quest

    if progress.handed_in:
        return Intent(Act.DONE, why=f"« {quest.name} » rendue")

    # 1. L'ACCEPTER D'ABORD. Une quete non acceptee dont on remplirait les objectifs ne
    #    compte pas : le jeu ne suit rien tant que le dialogue n'a pas eu lieu. C'est le
    #    genre d'erreur qui ne se voit qu'a la fin, quand le PNJ ne propose pas de rendre.
    if not progress.accepted:
        if here != quest.giver_at:
            return Intent(Act.TRAVEL, where=quest.giver_at,
                          why=f"aller voir {quest.giver} pour prendre « {quest.name} »")
        return Intent(Act.TALK, target=quest.giver, where=quest.giver_at,
                      why=f"accepter « {quest.name} »")

    # 2. Tous les objectifs faits : rendre, au bon PNJ, qui n'est pas toujours le donneur.
    if progress.objectives_done:
        who, at = quest.handed_to()
        if here != at:
            return Intent(Act.TRAVEL, where=at,
                          why=f"aller rendre « {quest.name} » a {who}")
        return Intent(Act.TALK, target=who, where=at,
                      why=f"rendre « {quest.name} »")

    objective = progress.current
    assert objective is not None  # garanti par objectives_done
    return _for_objective(objective, progress, here)


def _for_objective(objective: Objective, progress: Progress, here: Coord) -> Intent:
    """L'intention d'une etape, en se deplacant d'abord si elle est ailleurs."""
    remaining = max(1, objective.count - progress.counted)

    if objective.kind is Step.GOTO:
        if objective.where is None:
            return Intent(Act.BLOCKED,
                          why=f"« {objective.describe()} » sans carte cible")
        if here != objective.where:
            return Intent(Act.TRAVEL, where=objective.where, why=objective.describe())
        # Deja sur place : l'etape est accomplie par le seul fait d'y etre.
        #
        # STEP_DONE ET NON DONE. Les deux se lisaient « done » et un appelant ecrivant
        # `if intent.act is Act.DONE: stop()` arretait la quete au milieu, sur une simple
        # etape de deplacement satisfaite. Deux sens pour un nom, dans une valeur de
        # retour dont tout le pilotage depend.
        return Intent(Act.STEP_DONE, where=objective.where,
                      why=f"arrive en {objective.where}")

    # `where=None` veut dire « n'importe ou », pas « on ne sait pas ou » : on tente sur
    # place. La distinction compte -- confondre les deux ferait soit errer le bot, soit
    # l'arreter sur une etape parfaitement jouable ici.
    if objective.where is not None and here != objective.where:
        return Intent(Act.TRAVEL, where=objective.where, why=objective.describe())

    if objective.kind is Step.TALK:
        return Intent(Act.TALK, target=objective.target, where=objective.where,
                      why=objective.describe())
    if objective.kind is Step.KILL:
        return Intent(Act.FIGHT, target=objective.target, count=remaining,
                      where=objective.where, why=objective.describe())
    if objective.kind is Step.HARVEST:
        return Intent(Act.HARVEST, target=objective.target, count=remaining,
                      where=objective.where, why=objective.describe())
    if objective.kind is Step.FETCH:
        # FETCH n'a pas d'action propre : un objet s'obtient en tuant ou en recoltant.
        # Le resoudre demande une table objet -> source que le depot n'a pas. On le dit,
        # plutot que de deviner une source et d'envoyer le bot au mauvais endroit.
        return Intent(Act.BLOCKED, target=objective.target, count=remaining,
                      why=f"« {objective.target} » : aucune table objet -> source")

    return Intent(Act.BLOCKED, why=f"etape inconnue : {objective.kind}")


def plan_ahead(progress: Progress, here: Coord, limit: int = 12) -> list[Intent]:
    """Deroule les intentions a venir, en supposant que chacune reussit.

    Sert a LIRE une quete avant de la lancer -- « ou va-t-il aller, dans quel ordre » --
    et a tester le directeur sur une quete entiere plutot qu'etape par etape. Ce n'est pas
    un plan d'execution : la premiere intention qui echoue invalide la suite, et le bot
    redemande de toute facon `next_intent` a chaque cycle.

    `limit` borne la boucle : une etape bloquee rendrait la meme intention indefiniment.
    """
    from copy import deepcopy

    trace: list[Intent] = []
    sim = deepcopy(progress)
    position = here
    for _ in range(limit):
        intent = next_intent(sim, position)
        trace.append(intent)
        if intent.act is Act.BLOCKED or sim.handed_in:
            break
        if intent.act is Act.TRAVEL and intent.where is not None:
            position = intent.where
            continue
        if intent.act is Act.TALK:
            if not sim.accepted:
                sim.accepted = True
            elif sim.objectives_done:
                sim.handed_in = True
            else:
                sim.advance(sim.current.count if sim.current else 1)
            continue
        if intent.act is Act.STEP_DONE:
            sim.advance(sim.current.count if sim.current else 1)
            continue
        if intent.act is Act.DONE:
            break
        sim.advance(intent.count)
    return trace
