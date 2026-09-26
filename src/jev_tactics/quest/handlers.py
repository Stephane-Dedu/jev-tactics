"""Brancher le directeur sur ce qui sait agir.

`QuestRunner` ne connait que des intentions et des gestionnaires. Ce module fabrique les
gestionnaires reels a partir des couches deja ecrites -- `world.Walker` pour se deplacer,
`world.Engager` plus `bot.play_fight` pour combattre.

C'est la derniere piece du squelette : a partir d'ici, une quete se joue de bout en bout
avec de vrais clics, et ce qui manque pour la finir n'est plus de la plomberie mais de la
PERCEPTION (les PNJ, les dialogues).
"""

from __future__ import annotations

from collections.abc import Callable

from jev_tactics.bot.fight import FightReport, Outcome
from jev_tactics.perception.monsters import MonsterGroup
from jev_tactics.quest.director import Intent
from jev_tactics.quest.runner import Attempt
from jev_tactics.world.engage import Engager
from jev_tactics.world.travel import Move
from jev_tactics.world.walker import Walker, WalkReport

FindGroup = Callable[[], MonsterGroup | None]
PlayFight = Callable[[], FightReport]


def travel_handler(walker: Walker) -> Callable[[Intent], Attempt]:
    """Se rendre sur la carte demandee.

    Rend toujours `progressed=0`, et ce n'est pas un oubli : un deplacement ne fait
    avancer aucun objectif, il rend seulement le suivant jouable. C'est le directeur qui,
    depuis la nouvelle position, cesse de demander le trajet. Compter le deplacement comme
    un avancement masquerait un aller-retour sans fin entre deux cartes -- exactement le
    pietinement que le runner surveille.
    """
    def handle(intent: Intent) -> Attempt:
        if intent.where is None:
            return Attempt(intent, ok=False, note="aucune carte cible")
        report = walker.travel_to(intent.where)
        if report.arrived:
            note = f"{report.steps} pas"
            if report.drifts:
                note += f", {report.drifts} derive(s)"
            return Attempt(intent, ok=True, note=note)
        return Attempt(intent, ok=False, note=report.describe())
    return handle


def fight_handler(
    engager: Engager,
    find_group: FindGroup,
    play: PlayFight,
    kills_per_fight: int = 1,
) -> Callable[[Intent], Attempt]:
    """Trouver un groupe, l'engager, jouer le combat.

    `kills_per_fight` VAUT UN PAR DEFAUT, ET C'EST UNE SOUS-ESTIMATION ASSUMEE. Un groupe
    compte souvent plusieurs monstres, mais rien ici ne sait combien : le nombre d'ennemis
    d'un combat se lit dans la timeline, que `play` consomme sans le rendre, et le
    rapporter demanderait de savoir en plus lesquels sont MORTS -- or les PV ennemis ne
    sont jamais lisibles (72 ennemis sur 32 captures, zero `hp_known`).

    Sous-estimer fait refaire un combat de trop ; surestimer fait croire un objectif
    rempli et envoie le bot rendre une quete qui ne l'est pas. Entre les deux erreurs, la
    premiere coute des minutes et la seconde casse la quete.
    """
    def handle(intent: Intent) -> Attempt:
        group = find_group()
        if group is None:
            return Attempt(intent, ok=False, note="aucun groupe repere sur cette carte")

        engaged = engager.engage(group)
        if not engaged.started:
            # PENDING, pas MISSED : le combat peut demarrer juste apres l'echeance. Le
            # verdict est differe au cycle suivant (cf. `Engager.settle`), et l'appelant
            # ne doit surtout pas condamner l'endroit ici.
            return Attempt(intent, ok=False,
                           note=f"engagement non confirme ({engaged.outcome.value}, "
                                f"{engaged.waited:.1f}s)")

        report = play()
        note = report.describe()

        # Un combat sans le moindre sort n'est PAS un combat gagne, quoi qu'en dise la
        # timeline : c'est le signe d'une barre de sorts mal decrite, et le compter comme
        # un avancement ferait progresser la quete sur une panne.
        if report.turns and report.mute_turns == report.turns:
            return Attempt(intent, ok=False, note=note)
        if report.outcome is Outcome.BLIND:
            return Attempt(intent, ok=False, note=note)

        return Attempt(intent, ok=True, progressed=kills_per_fight, note=note)
    return handle


def settle_after(engager: Engager, in_combat: bool) -> str | None:
    """Trancher un engagement en attente, au cycle suivant. -> la trace, ou None.

    A appeler une fois par cycle, avec la lecture de timeline que la boucle fait de toute
    facon. Sans cela, un engagement lent reste `PENDING` pour toujours et son verdict
    n'est jamais ecrit.
    """
    settled = engager.settle(in_combat)
    if settled is None:
        return None
    return f"engagement {settled.outcome.value} sur ({settled.group.x},{settled.group.y})"


def drift_note(walker_report: WalkReport) -> str | None:
    """Resume les derives d'un trajet, s'il y en a eu.

    Une derive n'est pas un echec mais elle n'est pas neutre : c'est la que l'adjacence
    supposee du graphe se trompe. La taire priverait l'operateur du seul signe que sa
    carte mentale et celle du jeu divergent.
    """
    drifted = [v for v in walker_report.verdicts if v.move is Move.DRIFTED]
    if not drifted:
        return None
    return " | ".join(v.describe() for v in drifted)
