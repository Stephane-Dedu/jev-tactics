"""Marcher : enchainer clic, attente, confirmation, correction.

`travel.py` dit ou cliquer et ce qui s'est passe ; ce module le FAIT, en boucle, jusqu'a
une carte donnee. Il ne connait ni recolte, ni combat, ni session : on lui donne de quoi
voir (`grab`), de quoi agir (`backend`), et il rend des verdicts.

Les deux dependances qui rendaient l'original intestable -- la capture d'ecran et la
lecture des coordonnees -- sont injectees. Un test n'a donc besoin ni du jeu, ni d'une
capture : c'est l'invariant du depot (« rejouer une decision hors-ligne »), applique au
deplacement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jev_tactics.action.mouse import InputBackend, move_along_curve
from jev_tactics.perception.coordinates import MapPosition, diagnose_position
from jev_tactics.world.navigation import Coord, Direction, MapGraph, Route, map_changed
from jev_tactics.world.travel import Move, Verdict, edge_point, judge_move

# Plafond d'attente d'un chargement de carte, et pas de scrutation.
#
# CE DELAI COUVRE DEUX CHOSES de duree tres variable : la MARCHE du personnage jusqu'au
# bord, puis le chargement. Un clic pris a l'autre bout de la carte demande une longue
# traversee. L'original attendait 2,0 s FIXES, et ce qui arrive quand elles ne suffisent
# pas est le pire mode de panne du circuit : la verification tombe sur l'ANCIENNE carte,
# conclut « sans effet », rembobine la route -- et la carte finit par charger. Le circuit
# est alors decale d'un cran, definitivement, puisqu'il ne connait que des directions
# relatives. La derive silencieuse contre laquelle le rembobinage existe, provoquee par
# le rembobinage lui-meme.
#
# On scrute donc, au lieu d'attendre : le signal d'arret ne coute rien, ce sont les
# COORDONNEES, qu'il faut lire de toute facon pour juger le deplacement.
MAP_LOAD_TIMEOUT = 8.0
MAP_POLL = 0.4

Grab = Callable[[], NDArray[np.uint8]]
ReadPosition = Callable[[NDArray[np.uint8]], MapPosition | None]


def _read_position(frame: NDArray[np.uint8]) -> MapPosition | None:
    return diagnose_position(frame).position


@dataclass
class WalkReport:
    """Ce qu'un trajet a produit. Les derives sont comptees A PART des echecs.

    Une derive n'est pas un echec : le personnage a bouge, on a appris une sortie, et le
    graphe est plus riche qu'avant. Les confondre ferait passer un trajet instructif pour
    un trajet rate -- et inversement masquerait le seul evenement qui decale un circuit.
    """

    verdicts: list[Verdict] = field(default_factory=list)
    arrived: bool = False

    @property
    def steps(self) -> int:
        return len(self.verdicts)

    @property
    def drifts(self) -> int:
        return sum(1 for v in self.verdicts if v.move is Move.DRIFTED)

    @property
    def stuck(self) -> int:
        return sum(1 for v in self.verdicts if v.move is Move.STUCK)

    def describe(self) -> str:
        etat = "arrive" if self.arrived else "non arrive"
        return (f"{etat} en {self.steps} pas "
                f"({self.drifts} derive(s), {self.stuck} sans effet)")


class Walker:
    """Deplace le personnage de carte en carte.

    `graph` est facultatif mais recommande : sans lui, une derive ne peut pas etre
    rattrapee autrement qu'en reessayant le pas voulu -- ce qui retombe dans le meme
    piege, puisque la sortie qu'on vient d'emprunter est precisement celle qui trompe.
    """

    def __init__(
        self,
        grab: Grab,
        backend: InputBackend,
        graph: MapGraph | None = None,
        read_position: ReadPosition = _read_position,
        map_load_timeout: float = MAP_LOAD_TIMEOUT,
    ):
        self.grab = grab
        self.backend = backend
        self.graph = graph
        self.read_position = read_position
        self.map_load_timeout = map_load_timeout

    # -- un pas ---------------------------------------------------------------

    def step(self, direction: Direction) -> Verdict | None:
        """Sortir par ce bord. -> le verdict, ou None si la position est illisible.

        None n'est pas un echec de deplacement : c'est un echec de PERCEPTION, et les
        deux appellent des gestes opposes -- reessayer dans un cas, recalibrer dans
        l'autre. Les confondre en un booleen est ce qui rendait « 3 passages sans
        coordonnees lisibles » indiscernable de « 3 deplacements rates ».
        """
        frame = self.grab()
        before = self.read_position(frame)
        if before is None:
            return None

        self._click(*edge_point(frame, direction))
        after_frame = self._wait_for_map(frame, before)
        after = self.read_position(after_frame)
        if after is None:
            return None

        verdict = judge_move(direction, before, after)

        # Une derive est une observation aussi valable qu'un deplacement conforme, et plus
        # precieuse : c'est precisement la que l'adjacence supposee se trompe. Un « sans
        # effet » n'en est pas une -- le personnage n'a pas bouge, on n'a rien appris.
        if self.graph is not None and verdict.move is not Move.STUCK:
            self.graph.observe(before.as_tuple(), direction, after.as_tuple())

        return verdict

    def _click(self, x: int, y: int) -> None:
        move_along_curve(self.backend, (float(x), float(y)))
        self.backend.click(x, y)

    def _wait_for_map(
        self,
        before_frame: NDArray[np.uint8],
        before: MapPosition,
    ) -> NDArray[np.uint8]:
        """Scruter jusqu'a ce que la carte change, ou jusqu'a l'echeance.

        Ni les coordonnees ni la comparaison d'images ne peuvent affirmer qu'un
        deplacement a ECHOUE -- seulement qu'il a reussi. On rend donc la main a
        l'echeance sans conclure : le jugement reste a `judge_move`, qui sait distinguer
        « pas bouge » de « mauvaise carte ».
        """
        waited, after = 0.0, self.grab()
        while True:
            position = self.read_position(after)
            if position is not None:
                if position.as_tuple() != before.as_tuple():
                    return after
            elif map_changed(before_frame, after):
                # Coordonnees illisibles : on retombe sur la comparaison d'images plutot
                # que de bloquer, avec ses limites connues (cf. `map_changed`).
                return after
            if waited >= self.map_load_timeout:
                return after
            self.backend.sleep(MAP_POLL)
            waited += MAP_POLL
            after = self.grab()

    # -- un trajet ------------------------------------------------------------

    def follow(self, route: Route, steps: int) -> WalkReport:
        """Avancer de `steps` pas sur un circuit, en rembobinant les pas sans effet.

        Le rembobinage est ce qui empeche un circuit de se decaler : un pas qui n'a pas eu
        lieu ne doit pas etre consomme. Il n'est PAS applique a une derive -- on a bien
        change de carte, et refaire le pas voulu depuis la nouvelle reprendrait la sortie
        qui vient de tromper.
        """
        report = WalkReport()
        for _ in range(steps):
            direction = route.advance()
            if direction is None:
                break
            verdict = self.step(direction)
            if verdict is None:
                route.rewind()
                break
            report.verdicts.append(verdict)
            if verdict.move is Move.STUCK:
                route.rewind()
        report.arrived = bool(report.verdicts) and report.stuck == 0
        return report

    def travel_to(self, goal: Coord, max_steps: int = 30) -> WalkReport:
        """Rejoindre une carte precise, en recalculant apres chaque derive.

        Le trajet est RECALCULE a chaque pas plutot que suivi en aveugle. C'est ce qui
        rend une derive rattrapable : la sortie trompeuse vient d'etre enregistree dans le
        graphe, donc le trajet suivant ne la reprendra pas -- la ou refaire le pas voulu
        retomberait dans le meme piege. C'est aussi la seule reponse a un SAUT de
        plusieurs cartes (zaap, rappel), qu'aucun pas relatif ne rattrape.
        """
        report = WalkReport()
        if self.graph is None:
            raise ValueError(
                "travel_to demande un MapGraph : sans lui, aucun trajet ne peut etre "
                "calcule vers une carte qui n'est pas adjacente.")

        for _ in range(max_steps):
            frame = self.grab()
            here = self.read_position(frame)
            if here is None:
                break
            if here.as_tuple() == goal:
                report.arrived = True
                return report

            path = self.graph.route(here.as_tuple(), goal)
            if not path:
                break
            verdict = self.step(path[0])
            if verdict is None:
                break
            report.verdicts.append(verdict)

        # Un dernier controle : le pas precedent a pu nous y poser.
        frame = self.grab()
        here = self.read_position(frame)
        report.arrived = here is not None and here.as_tuple() == goal
        return report
