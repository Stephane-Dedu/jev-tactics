"""La boucle de deplacement, verifiee sans le jeu ni une seule capture.

La capture d'ecran et la lecture des coordonnees sont injectees : c'est ce qui permet de
rejouer ici le mode de panne le plus couteux du circuit d'origine -- une carte qui met
plus longtemps que prevu a charger -- alors qu'il demandait auparavant une session reelle
et de la chance.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev_tactics.action.mouse import DryRunBackend
from jev_tactics.perception.coordinates import MapPosition
from jev_tactics.world.navigation import Direction, MapGraph, Route
from jev_tactics.world.travel import Move
from jev_tactics.world.walker import Walker

FRAME = np.zeros((200, 320, 3), dtype=np.uint8)


class FakeWorld:
    """Un monde minuscule : une position, et ce qu'un clic lui fait.

    `loads_after` modelise le chargement de carte : la position ne change qu'apres N
    lectures. C'est la seule facon de tester l'attente adaptative -- avec un changement
    instantane, une attente FIXE de 2 s passerait tous les tests et echouerait en jeu.
    """

    def __init__(self, start=(5, 5), outcome="arrive", loads_after=0):
        self.position = start
        self.outcome = outcome
        self.loads_after = loads_after
        self.reads = 0
        self.pending: tuple[int, int] | None = None

    def grab(self):
        return FRAME

    def read_position(self, frame) -> MapPosition | None:
        if self.outcome == "unreadable":
            return None
        self.reads += 1
        if self.pending is not None and self.reads > self.loads_after:
            self.position, self.pending = self.pending, None
        return MapPosition(x=self.position[0], y=self.position[1])

    def click(self, direction: Direction) -> None:
        dx, dy = direction.delta
        if self.outcome == "arrive":
            self.pending = (self.position[0] + dx, self.position[1] + dy)
        elif self.outcome == "drift":
            # On demande la droite, un obstacle fait sortir par le haut.
            self.pending = (self.position[0], self.position[1] - 1)
        # "stuck" : rien ne bouge.


class SpyBackend(DryRunBackend):
    """DryRunBackend, plus le monde qu'un clic doit faire bouger."""

    def __init__(self, world: FakeWorld, direction: Direction):
        super().__init__()
        self.world = world
        self.direction = direction
        self.slept = 0.0

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self.world.click(self.direction)

    def sleep(self, seconds: float) -> None:
        self.slept += seconds


def walker_for(world: FakeWorld, direction: Direction, graph: MapGraph | None = None,
               timeout: float = 8.0) -> tuple[Walker, SpyBackend]:
    backend = SpyBackend(world, direction)
    return Walker(world.grab, backend, graph=graph,
                  read_position=world.read_position,
                  map_load_timeout=timeout), backend


class TestOneStep:
    def test_arriving_where_intended(self):
        world = FakeWorld(outcome="arrive")
        walker, _ = walker_for(world, Direction.RIGHT)
        verdict = walker.step(Direction.RIGHT)
        assert verdict is not None
        assert verdict.move is Move.ARRIVED
        assert world.position == (6, 5)

    def test_not_moving_is_stuck(self):
        world = FakeWorld(outcome="stuck")
        walker, _ = walker_for(world, Direction.RIGHT, timeout=1.0)
        verdict = walker.step(Direction.RIGHT)
        assert verdict is not None and verdict.move is Move.STUCK

    def test_leaving_by_the_wrong_side_is_drift(self):
        world = FakeWorld(outcome="drift")
        walker, _ = walker_for(world, Direction.RIGHT)
        verdict = walker.step(Direction.RIGHT)
        assert verdict is not None and verdict.move is Move.DRIFTED

    def test_unreadable_position_is_not_a_failed_move(self):
        """None, pas STUCK. Les deux appellent des gestes opposes -- reessayer dans un
        cas, recalibrer dans l'autre -- et les confondre rendait « coordonnees illisibles »
        indiscernable de « deplacement rate »."""
        world = FakeWorld(outcome="unreadable")
        walker, _ = walker_for(world, Direction.RIGHT)
        assert walker.step(Direction.RIGHT) is None


class TestWaitingForTheMap:
    """LE MODE DE PANNE QUI JUSTIFIE L'ATTENTE ADAPTATIVE."""

    def test_a_slow_map_is_still_a_success(self):
        """Une carte qui met plusieurs scrutations a charger doit rendre ARRIVED.

        Avec une attente FIXE trop courte, la verification tombait sur l'ANCIENNE carte,
        concluait « sans effet », rembobinait la route -- et la carte finissait par
        charger. Le circuit restait decale d'un cran, definitivement.
        """
        world = FakeWorld(outcome="arrive", loads_after=6)
        walker, backend = walker_for(world, Direction.RIGHT)
        verdict = walker.step(Direction.RIGHT)
        assert verdict is not None
        assert verdict.move is Move.ARRIVED, "une carte lente est passee pour un echec"
        assert backend.slept > 0, "aucune scrutation : l'attente n'est pas adaptative"

    def test_a_fast_map_does_not_wait(self):
        """Symetrique du precedent : ne pas payer l'attente quand elle est inutile."""
        world = FakeWorld(outcome="arrive", loads_after=0)
        walker, backend = walker_for(world, Direction.RIGHT)
        walker.step(Direction.RIGHT)
        assert backend.slept == 0

    def test_the_wait_gives_up_at_the_deadline(self):
        world = FakeWorld(outcome="stuck")
        walker, backend = walker_for(world, Direction.RIGHT, timeout=2.0)
        walker.step(Direction.RIGHT)
        assert backend.slept <= 2.0 + 0.4


class TestGraphLearning:
    def test_an_arrival_is_recorded(self):
        graph = MapGraph()
        world = FakeWorld(outcome="arrive")
        walker, _ = walker_for(world, Direction.RIGHT, graph=graph)
        walker.step(Direction.RIGHT)
        assert graph.neighbours((5, 5))

    def test_a_drift_is_recorded_too(self):
        """Plus precieuse qu'une arrivee : c'est la que l'adjacence supposee se trompe,
        et c'est ce qui permettra de l'eviter au trajet suivant."""
        graph = MapGraph()
        world = FakeWorld(outcome="drift")
        walker, _ = walker_for(world, Direction.RIGHT, graph=graph)
        walker.step(Direction.RIGHT)
        assert graph.neighbours((5, 5))

    def test_a_stuck_move_teaches_nothing(self):
        """Le personnage n'a pas bouge : enregistrer une sortie serait inventer une
        adjacence."""
        graph = MapGraph()
        world = FakeWorld(outcome="stuck")
        walker, _ = walker_for(world, Direction.RIGHT, graph=graph, timeout=1.0)
        walker.step(Direction.RIGHT)
        assert not graph.neighbours((5, 5))


class TestFollowingARoute:
    def test_a_stuck_step_is_rewound(self):
        """Un pas qui n'a pas eu lieu ne doit pas etre consomme, sinon le circuit se
        decale a jamais."""
        route = Route(steps=[Direction.RIGHT, Direction.BOTTOM])
        world = FakeWorld(outcome="stuck")
        walker, _ = walker_for(world, Direction.RIGHT, timeout=1.0)
        walker.follow(route, steps=1)
        assert route.peek() is Direction.RIGHT, "le pas rate a ete consomme"

    def test_a_drift_is_not_rewound(self):
        """On a bien change de carte : refaire le pas voulu reprendrait la sortie qui
        vient de tromper."""
        route = Route(steps=[Direction.RIGHT, Direction.BOTTOM])
        world = FakeWorld(outcome="drift")
        walker, _ = walker_for(world, Direction.RIGHT)
        report = walker.follow(route, steps=1)
        assert route.peek() is Direction.BOTTOM
        assert report.drifts == 1

    def test_report_separates_drifts_from_failures(self):
        route = Route(steps=[Direction.RIGHT])
        world = FakeWorld(outcome="drift")
        walker, _ = walker_for(world, Direction.RIGHT)
        report = walker.follow(route, steps=1)
        assert report.drifts == 1
        assert report.stuck == 0
        assert "derive" in report.describe()


class TestTravelTo:
    def test_reaching_an_adjacent_map(self):
        graph = MapGraph()
        graph.observe((5, 5), Direction.RIGHT, (6, 5))
        world = FakeWorld(start=(5, 5), outcome="arrive")
        walker, _ = walker_for(world, Direction.RIGHT, graph=graph)
        report = walker.travel_to((6, 5))
        assert report.arrived, report.describe()

    def test_already_there_costs_no_step(self):
        graph = MapGraph()
        world = FakeWorld(start=(5, 5))
        walker, _ = walker_for(world, Direction.RIGHT, graph=graph)
        report = walker.travel_to((5, 5))
        assert report.arrived and report.steps == 0

    def test_without_a_graph_it_refuses_rather_than_guesses(self):
        world = FakeWorld()
        walker, _ = walker_for(world, Direction.RIGHT, graph=None)
        with pytest.raises(ValueError, match="MapGraph"):
            walker.travel_to((9, 9))
