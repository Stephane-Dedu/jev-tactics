"""Le jugement d'un deplacement, verifie sans le jeu.

Ces trois issues n'existent que parce que la deuxieme -- la DERIVE -- est invisible
autrement. Un test qui ne verifierait que « a-t-on change de carte » repasserait le
defaut d'origine : le circuit reprend sur une carte decalee et ne se rattrape jamais.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev_tactics.perception.coordinates import MapPosition
from jev_tactics.world.navigation import Direction
from jev_tactics.world.travel import (
    EDGE_MARGIN,
    Move,
    edge_point,
    judge_move,
    on_interface,
)


def pos(x: int, y: int) -> MapPosition:
    return MapPosition(x=x, y=y)


class TestJudgeMove:
    """La fonction pure : trois issues, pas un booleen."""

    @pytest.mark.parametrize("direction,delta", [
        (Direction.LEFT, (-1, 0)),
        (Direction.RIGHT, (1, 0)),
        (Direction.TOP, (0, -1)),
        (Direction.BOTTOM, (0, 1)),
    ])
    def test_arriving_where_intended(self, direction, delta):
        before = pos(5, 5)
        after = pos(5 + delta[0], 5 + delta[1])
        verdict = judge_move(direction, before, after)
        assert verdict.move is Move.ARRIVED
        assert verdict.changed_map

    def test_not_moving_at_all_is_stuck(self):
        verdict = judge_move(Direction.RIGHT, pos(5, 5), pos(5, 5))
        assert verdict.move is Move.STUCK
        assert not verdict.changed_map
        assert "sans effet" in verdict.describe()

    def test_changing_map_but_not_the_intended_one_is_drift(self):
        """LE CAS QUI JUSTIFIE TOUT LE MODULE.

        On demande la droite, un obstacle fait sortir par le haut. La carte a bien change
        -- une comparaison d'images dirait « reussi » -- mais on n'est pas ou l'on croit.
        """
        verdict = judge_move(Direction.RIGHT, pos(5, 5), pos(5, 4))
        assert verdict.move is Move.DRIFTED
        assert verdict.changed_map          # on a bouge...
        assert verdict.expected == (6, 5)   # ...mais pas la
        assert "derive" in verdict.describe()

    def test_drift_is_not_confused_with_arrival_on_negative_coordinates(self):
        """Les coordonnees Dofus sont signees ; un `-` mal lu ferait passer une derive
        pour une arrivee."""
        verdict = judge_move(Direction.LEFT, pos(-3, -7), pos(-4, -7))
        assert verdict.move is Move.ARRIVED
        assert judge_move(Direction.LEFT, pos(-3, -7), pos(-2, -7)).move is Move.DRIFTED


class TestEdgePoint:
    """Ou cliquer. Le plein ecran sans interface est le cas simple ; ce qui compte est
    qu'un panneau ouvert repousse le point VERS L'INTERIEUR, jamais sur le cote."""

    @staticmethod
    def blank(width: int = 1920, height: int = 1080):
        # Gris moyen uniforme : ni sature ni sombre, donc pas pris pour de l'interface.
        return np.full((height, width, 3), 120, dtype=np.uint8)

    @pytest.mark.parametrize("direction", list(Direction))
    def test_point_lands_inside_the_frame(self, direction):
        frame = self.blank()
        x, y = edge_point(frame, direction)
        assert 0 <= x < frame.shape[1]
        assert 0 <= y < frame.shape[0]

    def test_horizontal_edges_stay_on_the_middle_row(self):
        frame = self.blank()
        for direction in (Direction.LEFT, Direction.RIGHT):
            _, y = edge_point(frame, direction)
            assert y == frame.shape[0] // 2

    def test_vertical_edges_stay_on_the_middle_column(self):
        frame = self.blank()
        for direction in (Direction.TOP, Direction.BOTTOM):
            x, _ = edge_point(frame, direction)
            assert x == frame.shape[1] // 2

    def test_search_moves_inward_not_sideways(self):
        """Rentrer ne peut, au pire, que cliquer la carte. Chercher lateralement en bas
        finirait dans la MINICARTE, dont l'interieur est un rendu de carte que le masque
        ne reconnait pas comme interface."""
        frame = self.blank()
        left = edge_point(frame, Direction.LEFT)
        right = edge_point(frame, Direction.RIGHT)
        # Le point gauche reste dans la moitie gauche, le droit dans la moitie droite :
        # la recherche n'a pas derive vers le centre au point de changer de camp.
        assert left[0] < frame.shape[1] // 2 < right[0]

    def test_nominal_point_is_returned_when_everything_is_interface(self):
        """Aucun point libre : on rend le bord nominal plutot qu'un clic arbitraire. Le
        deplacement echouera et sera COMPTE, ce qui vaut mieux qu'un clic au hasard."""
        # Tout blanc sature -> reconnu comme interface sur toute la bande exploree.
        frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        x, y = edge_point(frame, Direction.LEFT)
        assert (x, y) == (EDGE_MARGIN, 1080 // 2)


class TestOnInterface:
    def test_fixed_zones_count_even_without_a_mask(self):
        """Les rectangles fixes sont exclus meme quand le masque ne voit rien : leur
        clarte les rend precisement invisibles a `ui_mask`."""
        from jev_tactics.perception.ui import UI_ZONES

        assert UI_ZONES, "UI_ZONES vide : l'exclusion fixe ne protege plus rien"
        x0, y0, x1, y1 = UI_ZONES[0]
        inside = ((x0 + x1) // 2, (y0 + y1) // 2)
        assert on_interface(None, *inside)
