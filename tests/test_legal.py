"""Actions legales : portees, ligne de vue, plafonds de lancer, budgets PA/PM.

Ces tests protegent la brique la plus partagee du projet -- le solveur ET le futur
masque d'actions PPO en dependent."""

from pathlib import Path

import numpy as np

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner import Cast, EndTurn, Move, TurnState, apply_action, legal_actions
from jev_tactics.rules.spells import (
    Spell,
    castable_targets,
    has_line_of_sight,
    load_spells,
)
from jev_tactics.state import CombatState, Entity, Team

BOARD = BoardMap(
    cells=np.array([(i, j) for j in range(7) for i in range(7)], dtype=np.int64),
    e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
    origin=np.array([500.0, 300.0]),
)
CENTRE = BOARD.index_at(3, 3)

STRIKE = Spell(name="frappe", ap_cost=3, range_min=1, range_max=3,
               damage_min=20, damage_max=30, max_casts_per_turn=2)
BOLT = Spell(name="trait", ap_cost=2, range_min=1, range_max=6,
             needs_line_of_sight=False, damage_min=10, damage_max=14,
             max_casts_per_target=1)


def _state(me_cell=CENTRE, enemy_cells=(), ap=6, mp=3, obstacles=frozenset()):
    entities = [Entity(entity_id="me", team=Team.ALLY, cell=me_cell, hp=100,
                       hp_max=100, ap=ap, mp=mp, is_self=True)]
    entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=c, hp=50,
                        hp_max=50, ap=6, mp=3) for i, c in enumerate(enemy_cells)]
    return CombatState(turn=1, entities=entities, obstacles=set(obstacles))


def _turn(state):
    return TurnState.from_combat(state)


def _kinds(actions):
    return {a.kind for a in actions}


class TestLineOfSight:
    def test_clear_line(self):
        assert has_line_of_sight(BOARD, BOARD.index_at(1, 3), BOARD.index_at(5, 3), set())

    def test_blocked_by_obstacle_between(self):
        wall = {BOARD.index_at(3, 3)}
        assert not has_line_of_sight(BOARD, BOARD.index_at(1, 3), BOARD.index_at(5, 3), wall)

    def test_endpoints_do_not_block(self):
        """La cible n'obstrue pas sa propre visee, ni le lanceur la sienne."""
        a, b = BOARD.index_at(1, 3), BOARD.index_at(5, 3)
        assert has_line_of_sight(BOARD, a, b, {a, b})

    def test_same_cell(self):
        assert has_line_of_sight(BOARD, CENTRE, CENTRE, set())


class TestSpellRange:
    def test_respects_min_and_max(self):
        near = BOARD.index_at(3, 4)   # distance 1
        far = BOARD.index_at(3, 3)    # distance 0 -> sous range_min
        assert STRIKE.in_range(BOARD, CENTRE, near)
        assert not STRIKE.in_range(BOARD, CENTRE, far)

    def test_beyond_max_range_rejected(self):
        assert not STRIKE.in_range(BOARD, BOARD.index_at(0, 0), BOARD.index_at(6, 6))

    def test_no_los_spell_ignores_walls(self):
        wall = {BOARD.index_at(3, 3)}
        targets = castable_targets(BOARD, BOLT, BOARD.index_at(1, 3), wall)
        assert BOARD.index_at(5, 3) in targets

    def test_los_spell_respects_walls(self):
        wall = {BOARD.index_at(2, 3)}
        targets = castable_targets(BOARD, STRIKE, BOARD.index_at(1, 3), wall)
        assert BOARD.index_at(4, 3) not in targets

    def test_area_cells_cover_radius(self):
        blast = Spell(name="zone", ap_cost=4, range_max=5, area_radius=1)
        cells = blast.area_cells(BOARD, CENTRE)
        assert CENTRE in cells and len(cells) == 5   # la cible + ses 4 voisins

    def test_single_target_area_is_just_the_target(self):
        assert STRIKE.area_cells(BOARD, CENTRE) == [CENTRE]


class TestLegalActions:
    def test_end_turn_always_available(self):
        actions = legal_actions(BOARD, _state(), [STRIKE], _turn(_state()))
        assert any(isinstance(a, EndTurn) for a in actions)

    def test_no_actions_once_ended(self):
        turn = TurnState(cell=CENTRE, ap=6, mp=3, ended=True)
        assert legal_actions(BOARD, _state(), [STRIKE], turn) == []

    def test_moves_limited_by_mp(self):
        state = _state(mp=1)
        moves = [a for a in legal_actions(BOARD, state, [], _turn(state)) if isinstance(a, Move)]
        assert len(moves) == 4 and all(m.cost == 1 for m in moves)

    def test_no_moves_without_mp(self):
        state = _state(mp=0)
        assert not [a for a in legal_actions(BOARD, state, [], _turn(state)) if isinstance(a, Move)]

    def test_occupied_cells_block_movement_but_stay_targetable(self):
        enemy = BOARD.index_at(3, 4)
        state = _state(enemy_cells=(enemy,), mp=1)
        actions = legal_actions(BOARD, state, [STRIKE], _turn(state))
        assert enemy not in [a.cell for a in actions if isinstance(a, Move)]
        assert enemy in [a.target for a in actions if isinstance(a, Cast)]

    def test_spell_too_expensive_is_excluded(self):
        state = _state(ap=2)
        casts = [a for a in legal_actions(BOARD, state, [STRIKE], _turn(state))
                 if isinstance(a, Cast)]
        assert casts == []

    def test_per_turn_cap_enforced(self):
        state = _state()
        turn = TurnState(cell=CENTRE, ap=9, mp=0,
                         casts=(("frappe", 1), ("frappe", 2)))   # cap = 2
        casts = [a for a in legal_actions(BOARD, state, [STRIKE], turn) if isinstance(a, Cast)]
        assert casts == []

    def test_per_target_cap_enforced(self):
        target = BOARD.index_at(3, 4)
        state = _state()
        turn = TurnState(cell=CENTRE, ap=6, mp=0, casts=(("trait", target),))
        targets = [a.target for a in legal_actions(BOARD, state, [BOLT], turn)
                   if isinstance(a, Cast)]
        assert target not in targets          # cap par cible = 1
        assert targets                        # ... mais les autres cases restent ouvertes

    def test_actions_are_serialisable(self):
        state = _state(enemy_cells=(BOARD.index_at(3, 4),))
        for action in legal_actions(BOARD, state, [STRIKE], _turn(state)):
            assert type(action).model_validate_json(action.model_dump_json()) == action


class TestSpellConfig:
    """Les sorts sont de la CONFIG, pas du code : le solveur doit tourner sur le jeu de
    sorts du personnage sans recompilation."""

    CONFIG = Path(__file__).resolve().parents[1] / "configs" / "spells" / "example.json"

    def test_loads_from_json(self):
        spells = load_spells(self.CONFIG)
        assert {s.name for s in spells} == {"frappe", "trait", "souffle"}

    def test_defaults_applied_when_absent(self):
        spells = {s.name: s for s in load_spells(self.CONFIG)}
        assert spells["frappe"].max_casts_per_target == 99   # non precise -> illimite
        assert spells["souffle"].area_radius == 1

    def test_average_damage(self):
        spells = {s.name: s for s in load_spells(self.CONFIG)}
        assert spells["frappe"].average_damage == 27.0

    def test_loaded_spells_drive_legal_actions(self):
        state = _state(enemy_cells=(BOARD.index_at(3, 5),))
        actions = legal_actions(BOARD, state, load_spells(self.CONFIG), _turn(state))
        assert {a.spell for a in actions if isinstance(a, Cast)} <= {
            "frappe", "trait", "souffle"}


class TestApplyAction:
    def test_move_spends_mp_and_relocates(self):
        turn = apply_action(TurnState(cell=CENTRE, ap=6, mp=3), Move(cell=10, cost=2))
        assert (turn.cell, turn.mp, turn.ap) == (10, 1, 6)

    def test_cast_spends_ap_and_records_history(self):
        turn = apply_action(TurnState(cell=CENTRE, ap=6, mp=3),
                            Cast(spell="frappe", target=7, cost=3))
        assert turn.ap == 3 and turn.casts == (("frappe", 7),)
        assert turn.casts_of("frappe") == 1 and turn.casts_on("frappe", 7) == 1

    def test_end_turn_marks_ended(self):
        assert apply_action(TurnState(cell=CENTRE, ap=6, mp=3), EndTurn()).ended

    def test_original_state_untouched(self):
        turn = TurnState(cell=CENTRE, ap=6, mp=3)
        apply_action(turn, Move(cell=10, cost=2))
        assert (turn.cell, turn.mp) == (CENTRE, 3)

    def test_turn_state_is_hashable_for_dedup(self):
        """Le solveur deduplique les sequences aboutissant au meme etat."""
        a = TurnState(cell=5, ap=3, mp=1, casts=(("frappe", 7),))
        b = TurnState(cell=5, ap=3, mp=1, casts=(("frappe", 7),))
        assert len({a, b}) == 1

    def test_sequence_consumes_both_budgets(self):
        turn = TurnState(cell=CENTRE, ap=6, mp=3)
        turn = apply_action(turn, Move(cell=BOARD.index_at(3, 4), cost=1))
        turn = apply_action(turn, Cast(spell="frappe", target=CENTRE, cost=3))
        assert (turn.ap, turn.mp) == (3, 2)
