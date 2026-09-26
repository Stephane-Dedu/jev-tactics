"""Engager un combat : cliquer un groupe, puis verifier qu'il a demarre.

La verification n'est pas un confort. La detection par mouvement attrape aussi **les
autres joueurs**, et rien de visuel ne les distingue d'un groupe de monstres : cliquer
dessus ne declenche rien. Sans controle, le bot resterait a cliquer un joueur qui passe,
indefiniment, en croyant engager. La timeline tranche -- elle n'existe qu'en combat.

Extrait de `farming/runner.py`, ou l'engagement etait mele au journal de detection, a la
liste noire par carte et au compteur de series inengageables. Ici : cliquer, attendre,
constater. Les consequences appartiennent a l'appelant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from jev_tactics.action.mouse import InputBackend, move_along_curve
from jev_tactics.perception.monsters import MonsterGroup
from jev_tactics.perception.timeline import TimelineEntry, read_timeline

# L'echeance BORNE l'attente, elle ne la fixe pas : un succes rend la main des que la
# timeline parait. Elle vaut 8 s par analogie avec le trajet de recolte -- « le meme
# trajet, plus l'ecran de placement » -- et jamais par mesure. D'ou `EngageResult.waited`,
# qui rend le delai reel au lieu de le jeter : c'est la grandeur qui permettra de poser
# l'echeance juste au-dessus du maximum observe au lieu de la deviner.
#
# Elle n'est payee QUE sur les echecs. A ~2,3 essais perdus par combat engage, chaque
# seconde de trop coute 2,3 s par combat.
ENGAGE_TIMEOUT = 8.0
ENGAGE_POLL = 0.4

# Plafond de la liste noire, par carte. Sans plafond la memoire grandit d'un passage a
# l'autre et finit par AVEUGLER la carte : mesure a 14,9 % de l'ecran en onze passages,
# sans saturation.
MAX_FAILED_SPOTS_PER_MAP = 16
FAILED_SPOT_RADIUS = 30

Grab = Callable[[], NDArray[np.uint8]]
ReadTimeline = Callable[[NDArray[np.uint8]], list[TimelineEntry]]


class Engagement(StrEnum):
    """L'issue d'un clic sur un groupe."""

    STARTED = "started"    # la timeline est apparue : le combat a bien demarre
    PENDING = "pending"    # l'echeance est passee sans timeline -- VERDICT DIFFERE
    LATE = "late"          # on etait en combat au cycle suivant : indecidable
    MISSED = "missed"      # confirme : ce n'etait pas un groupe engageable


@dataclass(frozen=True)
class EngageResult:
    outcome: Engagement
    group: MonsterGroup
    waited: float

    @property
    def started(self) -> bool:
        return self.outcome is Engagement.STARTED


@dataclass
class FailedSpots:
    """Les endroits ou un clic n'a rien donne, par carte.

    Plafonnee, et **les plus anciennes partent**. Une liste noire sans bornes finit par
    couvrir la carte, et un endroit condamne l'est pour toute la session : c'est la
    structure la plus chere a laisser grandir.
    """

    per_map: dict[object, list[tuple[int, int]]] = field(default_factory=dict)
    cap: int = MAX_FAILED_SPOTS_PER_MAP
    radius: int = FAILED_SPOT_RADIUS

    def remember(self, map_key: object, x: int, y: int) -> None:
        known = self.per_map.setdefault(map_key, [])
        known.append((x, y))
        del known[:-self.cap]

    def is_known_bad(self, map_key: object, x: int, y: int) -> bool:
        return any(abs(x - bx) <= self.radius and abs(y - by) <= self.radius
                   for bx, by in self.per_map.get(map_key, []))

    def forget_map(self, map_key: object) -> None:
        self.per_map.pop(map_key, None)


class Engager:
    """Clique un groupe et constate. Ne consigne rien, ne condamne rien."""

    def __init__(
        self,
        grab: Grab,
        backend: InputBackend,
        read_timeline: ReadTimeline = read_timeline,
        timeout: float = ENGAGE_TIMEOUT,
    ):
        self.grab = grab
        self.backend = backend
        self.read_timeline = read_timeline
        self.timeout = timeout
        self._pending: MonsterGroup | None = None

    @property
    def pending(self) -> MonsterGroup | None:
        """Le groupe dont le verdict attend le cycle suivant."""
        return self._pending

    def engage(self, group: MonsterGroup) -> EngageResult:
        """Cliquer, puis scruter la timeline jusqu'a l'echeance.

        Un depassement d'echeance rend **PENDING**, jamais MISSED. `_wait_for_combat`
        borne l'attente ; un groupe lointain demande une longue marche PLUS l'ecran de
        placement, et le combat peut demarrer juste apres. Conclure tout de suite ecrivait
        deux traces fausses : au journal « ce candidat n'etait pas un monstre » sur un vrai
        groupe, et a la liste noire un endroit condamne pour toute la session sur une carte
        ou l'on vient de prouver qu'il y a un groupe.

        Le biais n'etait meme pas tire au hasard : ce sont les groupes LOINTAINS et les GROS
        qui depassent l'echeance -- exactement ceux qu'il fallait apprendre a preferer.
        """
        self._click(group.x, group.y)
        started, waited = self._wait_for_combat()
        if started:
            self._pending = None
            return EngageResult(Engagement.STARTED, group, waited)
        self._pending = group
        return EngageResult(Engagement.PENDING, group, waited)

    def settle(self, in_combat: bool) -> EngageResult | None:
        """Trancher l'engagement en attente, une fois le cycle suivant observe.

        `in_combat` vient de la seule lecture qui puisse le dire, et la boucle la fait deja
        en tete de cycle : la timeline. Aucune capture n'est ajoutee.

        VRAI -> **LATE, pas STARTED**. Rien ne prouve que ce combat vienne de ce clic : un
        groupe qui passe peut agresser le personnage, et il marchait justement vers la
        cible. Dans le doute on ne s'attribue pas le succes -- et on ne condamne pas
        l'endroit non plus.
        """
        group, self._pending = self._pending, None
        if group is None:
            return None
        outcome = Engagement.LATE if in_combat else Engagement.MISSED
        return EngageResult(outcome, group, 0.0)

    def _click(self, x: int, y: int) -> None:
        move_along_curve(self.backend, (float(x), float(y)))
        self.backend.click(x, y)

    def _wait_for_combat(self) -> tuple[bool, float]:
        """-> (la timeline est apparue, secondes attendues).

        SCRUTER plutot qu'attendre supprime l'arbitrage. L'original attendait 2,5 s fixes
        pour marcher jusqu'a un groupe, contre 3,5 s pour marcher jusqu'a une ressource --
        le meme geste, sur des distances du meme ordre, et l'engagement demande EN PLUS
        l'ecran de placement. Le budget le plus court etait accorde a l'action la plus
        longue.

        Un engagement declare rate alors qu'il aboutissait ne coute pas qu'un chiffre : le
        bot reprend son circuit, clique un bord, et se retrouve a jouer un combat qu'il
        croit ne pas avoir.

        Le delai rendu est la SOMME DES ATTENTES, pas l'horloge murale : les captures s'y
        ajoutent en temps reel mais pas a ce compteur, et melanger les deux ferait regler
        l'echeance sur une grandeur qu'elle ne controle pas.
        """
        waited = 0.0
        while True:
            if self.read_timeline(self.grab()):
                return True, waited
            if waited >= self.timeout:
                return False, waited
            self.backend.sleep(ENGAGE_POLL)
            waited += ENGAGE_POLL
