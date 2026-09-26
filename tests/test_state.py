"""Le schema d'etat doit survivre a un aller-retour JSON (garantie de replay)."""

from jev_tactics.state import CombatState, Entity, Team


def _sample() -> CombatState:
    return CombatState(
        turn=3,
        active_entity_id="me",
        entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=225, hp=850, hp_max=1000,
                   ap=6, mp=3, is_self=True),
            Entity(entity_id="mob1", team=Team.ENEMY, cell=337, hp=400, hp_max=400,
                   ap=6, mp=3),
        ],
        obstacles={100, 101, 102},
    )


def test_roundtrip_json_stable():
    state = _sample()
    restored = CombatState.model_validate_json(state.model_dump_json())
    assert restored == state


def test_occupied_cells_and_self():
    state = _sample()
    assert state.occupied_cells() == {225, 337}
    assert state.self_entity().entity_id == "me"
