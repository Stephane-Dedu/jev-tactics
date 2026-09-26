"""Executer une quete : demander l'intention, la jouer, recommencer.

Le directeur dit QUOI faire ; ce module le fait faire, et surtout **decide quoi faire
quand ca ne marche pas**. C'est la seule partie difficile : enchainer des succes est
trivial, et un bot ne passe pas sa vie a reussir.

Les trois pannes qui comptent, et ce qu'on en fait :

  - **l'intention echoue** (deplacement sans effet, groupe introuvable) : on reessaie,
    mais pas indefiniment ;
  - **l'intention reussit sans faire avancer** : le cas vicieux. Le deplacement aboutit,
    le combat se gagne, et l'objectif ne bouge pas -- mauvais monstre, mauvaise carte.
    Un compteur d'echecs classique ne le voit pas, puisque rien n'a echoue ;
  - **l'intention n'est pas executable** (parler a un PNJ) : on passe l'etape EN LE
    DISANT. Un saut silencieux se lirait comme un objectif accompli.

La boucle est donc gardee par un compteur de PIETINEMENT, pas d'echec : ce qu'on surveille
est l'absence d'avancement, quelle qu'en soit la cause. C'est la lecon du depot d'origine,
ou « 30 tours joues » se lisait exactement comme un combat mene au corps a corps alors que
le bot n'avait lance aucun sort.

Tous les collaborateurs sont injectes : la boucle se teste sans le jeu, sans capture et
sans reseau.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from jev_tactics.quest.director import Act, Intent, next_intent
from jev_tactics.quest.model import Progress
from jev_tactics.world.navigation import Coord

# Combien de fois de suite la MEME intention peut etre tentee sans que l'avancement bouge
# avant qu'on renonce a l'etape. Trois : assez pour absorber un alea (un deplacement rate,
# un groupe qui disparait), trop peu pour qu'un blocage reel consomme la session.
STALL_LIMIT = 3


@dataclass(frozen=True)
class Attempt:
    """Ce qu'une intention a donne. `progressed` est ce qui compte, pas `ok`."""

    intent: Intent
    ok: bool
    progressed: int = 0
    note: str = ""

    def describe(self) -> str:
        verdict = "ok" if self.ok else "echec"
        avance = f" +{self.progressed}" if self.progressed else ""
        detail = f" — {self.note}" if self.note else ""
        return f"{self.intent.describe()} -> {verdict}{avance}{detail}"


Handler = Callable[[Intent], Attempt]
Locate = Callable[[], Coord | None]


@dataclass
class RunReport:
    attempts: list[Attempt] = field(default_factory=list)
    finished: bool = False
    stalled: bool = False
    lost: bool = False          # position illisible : panne de PERCEPTION, pas de quete

    @property
    def steps(self) -> int:
        return len(self.attempts)

    @property
    def progressed(self) -> int:
        return sum(a.progressed for a in self.attempts)

    def describe(self) -> str:
        if self.lost:
            etat = "INTERROMPU : position illisible"
        elif self.finished:
            etat = "quete rendue"
        elif self.stalled:
            etat = "INTERROMPU : pietinement"
        else:
            etat = "limite d'etapes atteinte"
        return (f"{etat} — {self.steps} intention(s), {self.progressed} avancement(s)")


class QuestRunner:
    """Joue une quete jusqu'au bout, ou jusqu'a ce qu'elle n'avance plus.

    `handlers` associe une action a ce qui sait la faire. Une action sans gestionnaire
    n'est pas une erreur : c'est l'etat actuel du bot pour `TALK`, et l'etape est passee
    avec sa raison. Lever ici arreterait la session entiere pour une capacite manquante
    connue.
    """

    def __init__(
        self,
        handlers: Mapping[Act, Handler],
        locate: Locate,
        stall_limit: int = STALL_LIMIT,
    ):
        self.handlers = dict(handlers)
        self.locate = locate
        self.stall_limit = stall_limit

    def run(self, progress: Progress, max_steps: int = 60) -> RunReport:
        report = RunReport()
        last_key: tuple[object, ...] | None = None
        repeats = 0

        for _ in range(max_steps):
            here = self.locate()
            if here is None:
                # PANNE DE PERCEPTION, pas de quete. Les deux appellent des gestes
                # opposes -- recalibrer d'un cote, reessayer de l'autre -- et les
                # confondre rendait « coordonnees illisibles » indiscernable de
                # « deplacement rate » dans le depot d'origine.
                report.lost = True
                return report

            intent = next_intent(progress, here)

            if intent.act is Act.DONE:
                report.finished = True
                return report

            before = (progress.index, progress.counted, progress.accepted,
                      progress.handed_in)
            attempt = self._play(intent, progress)
            report.attempts.append(attempt)
            after = (progress.index, progress.counted, progress.accepted,
                     progress.handed_in)

            # LE PIETINEMENT SE MESURE SUR L'AVANCEMENT, pas sur `ok`. Une intention peut
            # reussir parfaitement et ne rien faire avancer -- on se deplace vers la bonne
            # carte, on y tue le mauvais monstre. Compter les echecs laisserait tourner
            # cette boucle-la indefiniment, en la declarant saine.
            key = (intent.act, intent.target, intent.where)
            if after == before:
                repeats = repeats + 1 if key == last_key else 1
                if repeats >= self.stall_limit:
                    if not self._give_up(progress, intent):
                        report.stalled = True
                        return report
                    repeats = 0
                    last_key = None
                    continue
            else:
                repeats = 0
            last_key = key

        return report

    def _give_up(self, progress: Progress, intent: Intent) -> bool:
        """Renoncer a ce qui pietine. -> False si la quete entiere est perdue.

        LA REPONSE DEPEND DE LA PHASE, et c'est un test qui l'a montre : `skip()` fait
        avancer l'INDEX D'OBJECTIF, ce qui ne veut rien dire tant que la quete n'est pas
        acceptee. Un deplacement impossible vers le donneur faisait donc tourner la boucle
        jusqu'a la derniere etape -- on sautait des objectifs qu'on n'avait meme pas le
        droit de commencer, et `next_intent` redemandait le meme trajet a chaque tour.

        Trois phases, trois reponses :

          - avant l'acceptation : rien n'est sautable, la quete ne peut pas commencer ;
          - sur un objectif : on le passe, les suivants restent jouables ;
          - au rendu : rien n'est sautable non plus, la quete ne peut pas se finir.
        """
        if not progress.accepted:
            progress.notes.append(
                f"quete abandonnee : impossible d'aller l'accepter ({intent.describe()})")
            return False
        if progress.objectives_done:
            progress.notes.append(
                f"quete abandonnee : impossible d'aller la rendre ({intent.describe()})")
            return False
        progress.skip(
            f"{self.stall_limit} tentatives sans avancement ({intent.describe()})")
        return True

    def _play(self, intent: Intent, progress: Progress) -> Attempt:
        """Joue une intention, en passant l'etape si rien ne sait la jouer."""
        handler = self.handlers.get(intent.act)

        if handler is None:
            if intent.act is Act.BLOCKED:
                progress.skip(intent.why)
                return Attempt(intent, ok=False, note=f"etape passee : {intent.why}")
            # Capacite manquante CONNUE (parler a un PNJ). On passe en le disant : un
            # saut silencieux se lirait comme un objectif accompli, et c'est le mode de
            # panne le plus cher de ce depot.
            progress.skip(f"aucun gestionnaire pour {intent.act.value}")
            return Attempt(intent, ok=False,
                           note=f"non executable : {intent.act.value}")

        attempt = handler(intent)
        if attempt.progressed:
            progress.advance(attempt.progressed)
        return attempt


def accepted_handler(progress: Progress) -> Handler:
    """Gestionnaire minimal de `TALK`, pour les quetes sans dialogue reel.

    N'existe que pour pouvoir dérouler une quete de bout en bout en test, et pour le jour
    ou l'acceptation se fera autrement qu'en lisant une fenetre de dialogue. Il ne LIT
    rien : il suppose que parler suffit, ce qui est faux dans le jeu -- d'ou son absence
    des gestionnaires par defaut.
    """
    def handle(intent: Intent) -> Attempt:
        if not progress.accepted:
            progress.accepted = True
            return Attempt(intent, ok=True, note="quete acceptee")
        if progress.objectives_done:
            progress.handed_in = True
            return Attempt(intent, ok=True, note="quete rendue")
        return Attempt(intent, ok=False, note="rien a dire a ce stade")
    return handle
