"""Simulateur de combat.

La fidelite du simulateur conditionne tout le RL a venir : une politique entrainee sur
des regles fausses joue faux, sans que rien ne le signale. Ces tests verifient surtout
que le simulateur applique LES MEMES REGLES que le solveur -- il consomme `rules/`, il
ne les reimplemente pas."""

import numpy as np

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner import Cast, EndTurn, Move, best_sequence
from jev_tactics.rules.spells import Spell
from jev_tactics.sim import apply_actions, enemy_policy, winner
from jev_tactics.state import CombatState, Entity, Team

BOARD = BoardMap(
    cells=np.array([(i, j) for j in range(7) for i in range(7)], dtype=np.int64),
    e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
    origin=np.array([500.0, 300.0]),
)
CENTRE = BOARD.index_at(3, 3)

BOLT = Spell(name="trait", ap_cost=3, range_min=1, range_max=6,
             needs_line_of_sight=False, damage_min=20, damage_max=20,
             max_casts_per_turn=2)
BLAST = Spell(name="souffle", ap_cost=4, range_min=1, range_max=5, area_radius=1,
              needs_line_of_sight=False, damage_min=10, damage_max=10)
SPELLS = [BOLT, BLAST]


def _state(me=CENTRE, enemies=(), ap=6, mp=3, enemy_hp=100, me_hp=100):
    entities = [Entity(entity_id="me", team=Team.ALLY, cell=me, hp=me_hp, hp_max=100,
                       ap=ap, mp=mp, is_self=True)]
    entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=c, hp=enemy_hp,
                        hp_max=100, ap=6, mp=3) for i, c in enumerate(enemies)]
    return CombatState(turn=1, entities=entities)


def _entity(state, entity_id):
    return next((e for e in state.entities if e.entity_id == entity_id), None)


class TestMovement:
    def test_move_relocates_and_spends_mp(self):
        state = _state()
        target = BOARD.index_at(3, 4)
        out = apply_actions(BOARD, state, SPELLS, [Move(cell=target, cost=1)])
        me = _entity(out.state, "me")
        assert me.cell == target and me.mp == 2

    def test_move_beyond_mp_is_refused(self):
        state = _state(mp=1)
        far = BOARD.index_at(3, 6)
        out = apply_actions(BOARD, state, SPELLS, [Move(cell=far, cost=1)])
        assert _entity(out.state, "me").cell == CENTRE and out.illegal

    def test_cost_recomputed_not_trusted(self):
        """Le cout annonce par l'action n'est pas cru : le simulateur le recalcule.
        Une politique pourrait sinon se deplacer loin en declarant un cout de 1."""
        state = _state(mp=3)
        far = BOARD.index_at(3, 6)          # a 3 cases
        out = apply_actions(BOARD, state, SPELLS, [Move(cell=far, cost=1)])
        assert _entity(out.state, "me").mp == 0


class TestCasting:
    def test_damage_applied_and_ap_spent(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        out = apply_actions(BOARD, state, SPELLS,
                            [Cast(spell="trait", target=BOARD.index_at(3, 5), cost=3)])
        assert _entity(out.state, "e0").hp == 80
        assert _entity(out.state, "me").ap == 3
        assert out.damage_dealt == 20.0

    def test_kill_removes_the_entity(self):
        state = _state(enemies=(BOARD.index_at(3, 5),), enemy_hp=20)
        out = apply_actions(BOARD, state, SPELLS,
                            [Cast(spell="trait", target=BOARD.index_at(3, 5), cost=3)])
        assert _entity(out.state, "e0") is None and out.kills == 1

    def test_area_spell_hits_neighbours(self):
        a, b = BOARD.index_at(3, 5), BOARD.index_at(3, 4)
        state = _state(enemies=(a, b))
        out = apply_actions(BOARD, state, SPELLS,
                            [Cast(spell="souffle", target=a, cost=4)])
        assert _entity(out.state, "e0").hp == 90 and _entity(out.state, "e1").hp == 90

    def test_area_spell_can_hit_the_caster(self):
        """Se toucher soi-meme ne compte pas dans les degats infliges, mais ne doit pas
        non plus etre ignore en silence : le caster est exclu explicitement."""
        state = _state(enemies=(BOARD.index_at(3, 4),))
        out = apply_actions(BOARD, state, SPELLS,
                            [Cast(spell="souffle", target=BOARD.index_at(3, 4), cost=4)])
        assert _entity(out.state, "me").hp == 100

    def test_out_of_ap_is_refused(self):
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=2)
        out = apply_actions(BOARD, state, SPELLS,
                            [Cast(spell="trait", target=BOARD.index_at(3, 5), cost=3)])
        assert _entity(out.state, "e0").hp == 100 and out.illegal

    def test_per_turn_cap_enforced(self):
        target = BOARD.index_at(3, 5)
        state = _state(enemies=(target,), ap=12)
        casts = [Cast(spell="trait", target=target, cost=3)] * 3   # plafond = 2
        out = apply_actions(BOARD, state, SPELLS, casts)
        assert _entity(out.state, "e0").hp == 60 and len(out.illegal) == 1

    def test_out_of_range_is_refused(self):
        close = Spell(name="corps", ap_cost=3, range_min=1, range_max=1,
                      needs_line_of_sight=False, damage_min=30, damage_max=30)
        far = BOARD.index_at(3, 6)
        state = _state(enemies=(far,))
        out = apply_actions(BOARD, state, [close], [Cast(spell="corps", target=far, cost=3)])
        assert _entity(out.state, "e0").hp == 100 and out.illegal

    def test_unknown_spell_is_refused(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        out = apply_actions(BOARD, state, SPELLS,
                            [Cast(spell="inexistant", target=BOARD.index_at(3, 5), cost=3)])
        assert out.illegal and _entity(out.state, "e0").hp == 100


class TestRobustness:
    def test_illegal_actions_are_reported_not_raised(self):
        """Pendant l'entrainement une politique proposera forcement des coups illegaux.
        Faire echouer l'episode rendrait le signal d'apprentissage inutilisable."""
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=0, mp=0)
        out = apply_actions(BOARD, state, SPELLS,
                            [Move(cell=0, cost=1),
                             Cast(spell="trait", target=BOARD.index_at(3, 5), cost=3)])
        assert len(out.illegal) == 2 and out.state.entities

    def test_end_turn_stops_processing(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        out = apply_actions(BOARD, state, SPELLS,
                            [EndTurn(),
                             Cast(spell="trait", target=BOARD.index_at(3, 5), cost=3)])
        assert _entity(out.state, "e0").hp == 100

    def test_original_state_untouched(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        apply_actions(BOARD, state, SPELLS,
                      [Cast(spell="trait", target=BOARD.index_at(3, 5), cost=3)])
        assert state.entities[1].hp == 100

    def test_empty_action_list_is_a_pass(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        out = apply_actions(BOARD, state, SPELLS, [])
        assert out.damage_dealt == 0.0 and len(out.state.entities) == 2


class TestSolverAgreement:
    """Le simulateur et le solveur doivent appliquer LES MEMES regles -- c'est tout
    l'interet de les faire consommer `rules/`."""

    def test_every_planned_action_is_accepted(self):
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(1, 2)), ap=9, mp=3)
        plan = best_sequence(BOARD, state, SPELLS)
        out = apply_actions(BOARD, state, SPELLS, plan.actions)
        assert out.illegal == [], f"le simulateur refuse un coup planifie : {out.illegal}"

    def test_planned_damage_is_actually_dealt(self):
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=6, mp=0)
        plan = best_sequence(BOARD, state, SPELLS)
        out = apply_actions(BOARD, state, SPELLS, plan.actions)
        assert out.damage_dealt > 0


class TestEnemyPolicy:
    def test_closes_distance_when_out_of_range(self):
        close = Spell(name="corps", ap_cost=3, range_min=1, range_max=1,
                      needs_line_of_sight=False, damage_min=30, damage_max=30)
        state = _state(me=BOARD.index_at(3, 3), enemies=(BOARD.index_at(3, 6),))
        actions = enemy_policy(BOARD, state, [close], "e0")
        assert any(isinstance(a, Move) for a in actions)

    def test_attacks_when_in_range(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        actions = enemy_policy(BOARD, state, SPELLS, "e0")
        assert any(isinstance(a, Cast) for a in actions)

    def test_its_actions_are_accepted_by_the_simulator(self):
        """Coherence croisee : l'adversaire de reference ne doit produire que des coups
        que le simulateur accepte."""
        state = _state(enemies=(BOARD.index_at(3, 5),))
        actions = enemy_policy(BOARD, state, SPELLS, "e0")
        out = apply_actions(BOARD, state, SPELLS, actions, actor_id="e0")
        assert out.illegal == []

    def test_damages_the_player(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        actions = enemy_policy(BOARD, state, SPELLS, "e0")
        out = apply_actions(BOARD, state, SPELLS, actions, actor_id="e0")
        assert _entity(out.state, "me").hp < 100

    def test_no_actions_without_a_target(self):
        state = CombatState(turn=1, entities=[
            Entity(entity_id="e0", team=Team.ENEMY, cell=CENTRE, hp=50, hp_max=50,
                   ap=6, mp=3)])
        assert enemy_policy(BOARD, state, SPELLS, "e0") == []


class TestWinner:
    def test_none_while_both_sides_stand(self):
        assert winner(_state(enemies=(BOARD.index_at(3, 5),))) is None

    def test_ally_wins_without_enemies(self):
        assert winner(_state()) is Team.ALLY

    def test_enemy_wins_without_allies(self):
        state = CombatState(turn=1, entities=[
            Entity(entity_id="e0", team=Team.ENEMY, cell=CENTRE, hp=50, hp_max=50,
                   ap=6, mp=3)])
        assert winner(state) is Team.ENEMY


class TestFullFight:
    def test_a_fight_reaches_a_conclusion(self):
        """Boucle complete joueur/adversaire : le simulateur doit converger, pas tourner
        indefiniment."""
        state = _state(enemies=(BOARD.index_at(3, 5),), enemy_hp=60, ap=6, mp=3)
        for _ in range(20):
            if winner(state) is not None:
                break
            plan = best_sequence(BOARD, state, SPELLS)
            state = apply_actions(BOARD, state, SPELLS, plan.actions).state
            for enemy in [e for e in state.entities if e.team is Team.ENEMY]:
                actions = enemy_policy(BOARD, state, SPELLS, enemy.entity_id)
                state = apply_actions(BOARD, state, SPELLS, actions,
                                      actor_id=enemy.entity_id).state
            # Les ressources se rechargent au tour suivant.
            state = state.model_copy(update={"entities": [
                e.model_copy(update={"ap": 6, "mp": 3}) for e in state.entities]})
        assert winner(state) is Team.ALLY
