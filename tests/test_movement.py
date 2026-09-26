"""Deplacement : distance de portee vs distance de parcours, PM, contournement."""

import numpy as np
import pytest

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.rules import grid_distance, path_between, reachable_cells
from jev_tactics.rules.spells import Spell, castable_targets, has_line_of_sight
from jev_tactics.sim import square_board

E_X = np.array([46.0, 23.0])
E_Y = np.array([-46.0, 23.0])


def _board(cells) -> BoardMap:
    return BoardMap(cells=np.array(cells, dtype=np.int64), e_x=E_X, e_y=E_Y,
                    origin=np.array([500.0, 300.0]))


# Plateau plein 5x5, indices 0..24 (tri par j puis i, cf. BoardMap.from_estimate).
FULL = _board([(i, j) for j in range(5) for i in range(5)])
CENTRE = FULL.index_at(2, 2)


class TestGridDistance:
    def test_zero_to_itself(self):
        assert grid_distance(FULL, CENTRE, CENTRE) == 0

    def test_manhattan_on_lattice(self):
        assert grid_distance(FULL, FULL.index_at(0, 0), FULL.index_at(2, 3)) == 5

    def test_symmetric(self):
        a, b = FULL.index_at(1, 0), FULL.index_at(4, 3)
        assert grid_distance(FULL, a, b) == grid_distance(FULL, b, a)

    def test_ignores_obstacles(self):
        """La portee est a vol d'oiseau : un mur ne l'allonge pas."""
        assert grid_distance(FULL, FULL.index_at(0, 0), FULL.index_at(0, 2)) == 2


class TestReachable:
    def test_start_always_included_at_zero_cost(self):
        assert reachable_cells(FULL, CENTRE, 3, set())[CENTRE] == 0

    @pytest.mark.parametrize("mp, expected", [(0, 1), (1, 5), (2, 13)])
    def test_diamond_grows_with_movement(self, mp, expected):
        """Sur un plateau plein, la zone atteignable est un losange : 1, 5, 13 cases."""
        assert len(reachable_cells(FULL, CENTRE, mp, set())) == expected

    def test_costs_equal_step_counts(self):
        costs = reachable_cells(FULL, CENTRE, 2, set())
        for cell, cost in costs.items():
            assert cost == grid_distance(FULL, CENTRE, cell)

    def test_obstacles_are_excluded(self):
        blocked = {FULL.index_at(3, 2)}
        assert FULL.index_at(3, 2) not in reachable_cells(FULL, CENTRE, 2, blocked)

    def test_walls_off_makes_cells_unreachable(self):
        """Case enfermee par ses 4 voisins : hors d'atteinte malgre des PM suffisants."""
        target = FULL.index_at(0, 0)
        blocked = {FULL.index_at(1, 0), FULL.index_at(0, 1)}
        assert target not in reachable_cells(FULL, CENTRE, 9, blocked)

    def test_detour_costs_more_than_straight_line(self):
        blocked = {FULL.index_at(2, 1), FULL.index_at(1, 1), FULL.index_at(3, 1)}
        costs = reachable_cells(FULL, CENTRE, 9, blocked)
        target = FULL.index_at(2, 0)
        assert costs[target] > grid_distance(FULL, CENTRE, target)

    def test_zero_movement_stays_put(self):
        assert reachable_cells(FULL, CENTRE, 0, set()) == {CENTRE: 0}

    def test_never_leaves_the_board(self):
        """Un plateau troue ne doit pas laisser fuir le BFS hors des cases existantes."""
        sparse = _board([(0, 0), (1, 0), (2, 0)])
        assert set(reachable_cells(sparse, 0, 9, set())) == {0, 1, 2}


class TestPath:
    def test_empty_path_to_itself(self):
        assert path_between(FULL, CENTRE, CENTRE, set()) == []

    def test_path_excludes_start_and_ends_on_target(self):
        target = FULL.index_at(4, 2)
        path = path_between(FULL, CENTRE, target, set())
        assert path[0] != CENTRE and path[-1] == target

    def test_path_length_matches_bfs_cost(self):
        target = FULL.index_at(4, 4)
        path = path_between(FULL, CENTRE, target, set())
        assert len(path) == reachable_cells(FULL, CENTRE, 9, set())[target]

    def test_each_step_is_adjacent(self):
        path = path_between(FULL, CENTRE, FULL.index_at(0, 4), set())
        for previous, current in zip([CENTRE, *path], path):
            assert grid_distance(FULL, previous, current) == 1

    def test_path_avoids_blocked_cells(self):
        blocked = {FULL.index_at(2, 1), FULL.index_at(1, 1), FULL.index_at(3, 1)}
        path = path_between(FULL, CENTRE, FULL.index_at(2, 0), set(blocked))
        assert not (set(path) & blocked)

    def test_none_when_unreachable(self):
        blocked = {FULL.index_at(1, 0), FULL.index_at(0, 1)}
        assert path_between(FULL, CENTRE, FULL.index_at(0, 0), blocked) is None


class TestSpellGeometry:
    """Contraintes d'alignement et formes de zone.

    Les deux viennent de defauts constates en jeu. « Ravage en ligne droite mais tente
    sur un ennemi pas dans la ligne » : le coup etait planifie, le clic parti, et le jeu
    le refusait -- tour perdu sans que rien ne le signale."""

    BOARD = square_board(9)

    def _spell(self, **kwargs):
        base = dict(name="s", ap_cost=3, range_min=1, range_max=5,
                    needs_line_of_sight=False, damage_min=10, damage_max=10)
        return Spell(**{**base, **kwargs})

    def test_a_line_spell_refuses_an_off_line_target(self):
        b = self.BOARD
        caster = b.index_at(4, 4)
        assert not self._spell(cast_in_line=True).in_range(b, caster, b.index_at(5, 5))

    def test_a_line_spell_accepts_a_target_on_the_line(self):
        b = self.BOARD
        caster = b.index_at(4, 4)
        spell = self._spell(cast_in_line=True)
        assert spell.in_range(b, caster, b.index_at(4, 6))
        assert spell.in_range(b, caster, b.index_at(6, 4))

    def test_an_unconstrained_spell_accepts_anything_in_range(self):
        b = self.BOARD
        assert self._spell().in_range(b, b.index_at(4, 4), b.index_at(5, 5))

    def test_a_cross_area_hits_fewer_cells_than_a_circle(self):
        """Confondre les deux fait surestimer un sort de groupe : le solveur SOMME les
        degats sur les cases touchees."""
        b = self.BOARD
        target = b.index_at(4, 4)
        circle = self._spell(area_radius=2, area_shape="circle").area_cells(b, target)
        cross = self._spell(area_radius=2, area_shape="cross").area_cells(b, target)
        assert len(cross) < len(circle)

    def test_cross_and_circle_coincide_at_radius_one(self):
        """Constate en implementant : la distance de grille est DEJA en croix, donc les
        deux formes ne divergent qu'a partir de 2. Le savoir evite de croire a un bug."""
        b = self.BOARD
        target = b.index_at(4, 4)
        circle = self._spell(area_radius=1, area_shape="circle").area_cells(b, target)
        cross = self._spell(area_radius=1, area_shape="cross").area_cells(b, target)
        assert set(circle) == set(cross)

    def test_a_diagonal_area_reaches_the_diagonal_neighbours(self):
        """Les cases diagonales sont a DEUX pas de grille : mesurer la zone en distance
        de grille n'en toucherait aucune a rayon 1."""
        b = self.BOARD
        target = b.index_at(4, 4)
        cells = self._spell(area_radius=1, area_shape="diagonal").area_cells(b, target)
        assert b.index_at(5, 5) in cells and b.index_at(3, 3) in cells

    def test_a_point_shape_hits_only_the_target(self):
        """'P' porte param1=1 dans les donnees du jeu : le lire comme un rayon ferait
        croire a une zone sur la majorite des sorts."""
        b = self.BOARD
        target = b.index_at(4, 4)
        assert self._spell(area_radius=1, area_shape="point").area_cells(b, target) == [target]

    def test_an_area_always_includes_its_target(self):
        b = self.BOARD
        target = b.index_at(4, 4)
        for shape in ("circle", "cross", "diagonal"):
            cells = self._spell(area_radius=2, area_shape=shape).area_cells(b, target)
            assert target in cells, shape


class TestLineOfSight:
    """Ce qui BLOQUE la visee. Deux defauts constates en jeu : « on ne peut pas taper un
    ennemi derriere un mur epais, ou un monstre » -- le bot le tentait, le jeu refusait,
    et le tour etait perdu sans que rien ne le signale."""

    BOARD = square_board(9)

    def _spell(self, **kwargs):
        base = dict(name="s", ap_cost=3, range_min=1, range_max=8,
                    needs_line_of_sight=True, damage_min=10, damage_max=10)
        return Spell(**{**base, **kwargs})

    def test_a_clear_line_is_visible(self):
        b = self.BOARD
        assert has_line_of_sight(b, b.index_at(4, 2), b.index_at(4, 6), set())

    def test_an_entity_between_blocks(self):
        """Regle du jeu : un combattant coupe la ligne de vue."""
        b = self.BOARD
        blocker = b.index_at(4, 4)
        assert not has_line_of_sight(b, b.index_at(4, 2), b.index_at(4, 6), {blocker})

    def test_the_target_itself_never_blocks(self):
        """Sinon aucun ennemi ne serait jamais ciblable."""
        b = self.BOARD
        target = b.index_at(4, 6)
        assert has_line_of_sight(b, b.index_at(4, 2), target, {target})

    def test_the_caster_never_blocks_itself(self):
        b = self.BOARD
        caster = b.index_at(4, 2)
        assert has_line_of_sight(b, caster, b.index_at(4, 6), {caster})

    def test_a_gap_in_the_board_blocks(self):
        """LE defaut le plus surprenant : une position ABSENTE du plateau ne bloquait
        pas. Or ce qui n'est pas une case jouable est un mur, un trou ou du
        hors-plateau -- rien de tout cela ne se traverse."""
        full = self.BOARD
        hole = full.index_at(4, 4)
        pierced = BoardMap(
            cells=np.array([c for i, c in enumerate(full.cells) if i != hole],
                           dtype=np.int64),
            e_x=full.e_x, e_y=full.e_y, origin=full.origin)
        a, b_ = pierced.index_at(4, 2), pierced.index_at(4, 6)
        assert not has_line_of_sight(pierced, a, b_, set())

    def test_a_spell_ignoring_los_reaches_through(self):
        """Certains sorts ne testent pas la ligne de vue : la contrainte doit rester
        attachee au sort, pas au plateau."""
        b = self.BOARD
        blocker = {b.index_at(4, 4)}
        spell = self._spell(needs_line_of_sight=False)
        assert b.index_at(4, 6) in castable_targets(b, spell, b.index_at(4, 2), blocker)
