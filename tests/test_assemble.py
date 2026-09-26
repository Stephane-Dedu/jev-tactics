"""Assemblage Observation -> CombatState : detection du tour, suivi d'identite,
et refus explicite d'assembler un etat inexploitable."""


from jev_tactics.perception.timeline import TimelineEntry
from jev_tactics.perception.ui import UiReading
from jev_tactics.pipeline import Observation
from jev_tactics.state import CombatState, Entity, Team
from jev_tactics.state.assemble import (
    assemble_combat_state,
    identify_self_slot,
    is_our_turn,
    track_identities,
)


def _entity(cell, team=Team.ENEMY, is_self=False, entity_id=None, hp=100):
    return Entity(
        entity_id=entity_id or f"{team.value}_{cell}",
        team=team, cell=cell, hp=hp, hp_max=100, ap=6, mp=3, is_self=is_self,
    )


def _timeline(*ratios, active_slot=0):
    return [
        TimelineEntry(slot=i, x=1252 + 77 * i, hp_ratio=r, is_active=(i == active_slot))
        for i, r in enumerate(ratios)
    ]


def _observation(entities, timeline, ui=None):
    return Observation(
        timestamp=0.0,
        ui=ui or UiReading(pv=50, pv_max=100, pa=6, pm=3),
        entities=entities,
        timeline=timeline,
    )


class TestSelfSlot:
    def test_matches_slot_by_hp_ratio(self):
        # Joueur a 50 % : c'est l'emplacement 1 qui correspond.
        timeline = _timeline(1.0, 0.5, 1.0)
        assert identify_self_slot(timeline, UiReading(pv=50, pv_max=100)) == 1

    def test_tolerates_bar_imprecision(self):
        """Ecart mesure jusqu'a 0.10 : l'appariement doit tenir quand meme."""
        timeline = _timeline(0.51, 1.0, 1.0)
        assert identify_self_slot(timeline, UiReading(pv=41, pv_max=100)) == 0

    def test_none_without_ui_reading(self):
        assert identify_self_slot(_timeline(1.0), UiReading()) is None

    def test_none_without_timeline(self):
        assert identify_self_slot([], UiReading(pv=50, pv_max=100)) is None


class TestSelfSlotAtFullHealth:
    """LE cas que le ratio de PV ne peut pas trancher, et c'est le plus frequent : en
    debut de combat tout le monde est a 100 %. Constate en jeu -- joueur 1662/1662 et
    monstres 1000/1000 donnaient tous 1,0, `min()` rendait un monstre, et le bot
    concluait « tour adverse » pendant son propre tour. Il attendait indefiniment un
    tour qu'il avait deja."""

    def _mixed(self, *teams, ratio=1.0):
        return [TimelineEntry(slot=i, x=1252 + 77 * i, hp_ratio=ratio, team=t)
                for i, t in enumerate(teams)]

    def test_team_decides_when_every_ratio_is_identical(self):
        timeline = self._mixed(Team.ENEMY, Team.ALLY, Team.ENEMY)
        assert identify_self_slot(timeline, UiReading(pv=1662, pv_max=1662)) == 1

    def test_the_real_case_from_the_client(self):
        """Reproduction exacte : le joueur n'est pas en premiere position."""
        timeline = self._mixed(Team.ALLY, Team.ENEMY, Team.ENEMY)
        assert identify_self_slot(timeline, UiReading(pv=1662, pv_max=1662)) == 0

    def test_our_turn_is_seen_at_full_health(self):
        """Ce que le defaut coutait vraiment : le tour n'etait jamais reconnu."""
        timeline = [
            TimelineEntry(slot=0, x=1252, hp_ratio=1.0, team=Team.ENEMY),
            TimelineEntry(slot=1, x=1329, hp_ratio=1.0, is_active=True, team=Team.ALLY),
        ]
        assert is_our_turn(timeline, UiReading(pv=1662, pv_max=1662))

    def test_team_beats_a_closer_hp_ratio(self):
        """L'equipe prime meme quand un ennemi colle mieux aux PV : elle ne depend
        d'aucun degat subi, le ratio si."""
        timeline = [
            TimelineEntry(slot=0, x=1252, hp_ratio=0.50, team=Team.ENEMY),
            TimelineEntry(slot=1, x=1329, hp_ratio=1.00, team=Team.ALLY),
        ]
        assert identify_self_slot(timeline, UiReading(pv=50, pv_max=100)) == 1

    def test_hp_ratio_still_separates_two_allies(self):
        """Combat de groupe : plusieurs allies, les PV redeviennent le discriminant."""
        timeline = [
            TimelineEntry(slot=0, x=1252, hp_ratio=1.0, team=Team.ALLY),
            TimelineEntry(slot=1, x=1329, hp_ratio=0.4, team=Team.ALLY),
            TimelineEntry(slot=2, x=1406, hp_ratio=1.0, team=Team.ENEMY),
        ]
        assert identify_self_slot(timeline, UiReading(pv=40, pv_max=100)) == 1

    def test_no_ally_portrait_falls_back_to_hp(self):
        """Equipes illisibles : on retombe sur l'ancien comportement plutot que rien."""
        timeline = [
            TimelineEntry(slot=0, x=1252, hp_ratio=1.0),
            TimelineEntry(slot=1, x=1329, hp_ratio=0.5),
        ]
        assert identify_self_slot(timeline, UiReading(pv=50, pv_max=100)) == 1


class TestOurTurn:
    def test_true_when_self_slot_is_active(self):
        timeline = _timeline(0.5, 1.0, active_slot=0)
        assert is_our_turn(timeline, UiReading(pv=50, pv_max=100)) is True

    def test_false_when_another_slot_is_active(self):
        timeline = _timeline(0.34, 0.5, active_slot=1)
        assert is_our_turn(timeline, UiReading(pv=37, pv_max=100)) is False


class TestTracking:
    def test_identity_follows_movement(self):
        previous = [_entity(10, entity_id="orig")]
        current = [_entity(12)]
        assert track_identities(previous, current)[0].entity_id == "orig"

    def test_distant_entity_keeps_new_identity(self):
        previous = [_entity(10, entity_id="orig")]
        current = [_entity(90)]
        assert track_identities(previous, current)[0].entity_id != "orig"

    def test_teams_are_not_confused(self):
        previous = [_entity(10, team=Team.ALLY, is_self=True, entity_id="me")]
        current = [_entity(11, team=Team.ENEMY)]
        assert track_identities(previous, current)[0].entity_id != "me"

    def test_one_previous_entity_matched_once(self):
        previous = [_entity(10, entity_id="orig")]
        current = [_entity(10), _entity(11)]
        ids = [e.entity_id for e in track_identities(previous, current)]
        assert ids.count("orig") == 1

    def test_no_previous_state_is_passthrough(self):
        current = [_entity(10)]
        assert track_identities([], current) == current

    def test_uses_grid_distance_when_board_given(self):
        """Avec un plateau, l'appariement suit la geometrie et non l'ecart d'indices :
        des cases d'indices eloignes peuvent etre voisines sur le terrain."""
        import numpy as np

        from jev_tactics.calibration.grid import BoardMap

        board = BoardMap(
            cells=np.array([(i, j) for j in range(5) for i in range(5)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([500.0, 300.0]),
        )
        # Indices 2 et 7 sont eloignes de 5 en indice, mais adjacents sur la grille.
        previous = [_entity(2, entity_id="orig")]
        tracked = track_identities(previous, [_entity(7)], board)
        assert tracked[0].entity_id == "orig"


class TestAssemble:
    def test_builds_state_on_our_turn(self):
        obs = _observation(
            [_entity(5, team=Team.ALLY, is_self=True), _entity(20)],
            _timeline(0.5, 1.0, active_slot=0),
        )
        state = assemble_combat_state(obs, turn=3)
        assert state.turn == 3
        assert state.is_our_turn is True
        assert state.active_entity_id == state.self_entity().entity_id
        assert len(state.enemies()) == 1

    def test_not_our_turn_leaves_active_unset(self):
        obs = _observation(
            [_entity(5, team=Team.ALLY, is_self=True), _entity(20)],
            _timeline(0.5, 1.0, active_slot=1),
        )
        state = assemble_combat_state(obs)
        assert state.is_our_turn is False
        assert state.active_entity_id is None

    def test_none_out_of_combat(self):
        obs = _observation([_entity(5, team=Team.ALLY, is_self=True)], [])
        assert assemble_combat_state(obs) is None

    def test_none_when_self_not_located(self):
        """Mieux vaut pas d'etat qu'un etat dont le planificateur tirerait du faux."""
        obs = _observation([_entity(20)], _timeline(1.0))
        assert assemble_combat_state(obs) is None

    def test_identities_carried_across_turns(self):
        first = assemble_combat_state(
            _observation([_entity(5, team=Team.ALLY, is_self=True), _entity(20)],
                         _timeline(0.5, 1.0)), turn=1)
        second = assemble_combat_state(
            _observation([_entity(5, team=Team.ALLY, is_self=True), _entity(22)],
                         _timeline(0.5, 1.0)), turn=2, previous=first)
        assert ({e.entity_id for e in second.entities}
                == {e.entity_id for e in first.entities})

    def test_blocked_cells_union_of_obstacles_and_occupants(self):
        state = CombatState(turn=0, entities=[_entity(5, team=Team.ALLY, is_self=True),
                                              _entity(20)], obstacles={7})
        assert state.blocked_cells() == {5, 20, 7}

    def test_state_roundtrip_json(self):
        obs = _observation(
            [_entity(5, team=Team.ALLY, is_self=True), _entity(20)], _timeline(0.5, 1.0))
        state = assemble_combat_state(obs, turn=2)
        assert CombatState.model_validate_json(state.model_dump_json()) == state
