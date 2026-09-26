"""Espace d'actions et masque de legalite.

Les tests qui comptent sont ceux d'ACCORD : le masque doit dire exactement ce que le
simulateur accepte. Un masque trop large fait apprendre a la politique des coups que le
jeu refusera ; trop etroit, il lui cache des coups gagnants. Dans les deux cas rien dans
les courbes d'apprentissage ne le signale -- c'est pourquoi on le verifie ici."""

import numpy as np

from jev_tactics.planner.legal import Cast, EndTurn, Move, TurnState
from jev_tactics.rl import ActionSpace, action_mask, legal_table
from jev_tactics.rules.spells import Spell
from jev_tactics.sim import apply_actions, square_board
from jev_tactics.state import CombatState, Entity, Team

BOARD = square_board(7)
CENTRE = BOARD.index_at(3, 3)
FOE = BOARD.index_at(3, 5)

BOLT = Spell(name="trait", ap_cost=3, range_min=1, range_max=3,
             needs_line_of_sight=False, damage_min=20, damage_max=20,
             max_casts_per_turn=2)
BLAST = Spell(name="souffle", ap_cost=4, range_min=1, range_max=2, area_radius=1,
              needs_line_of_sight=False, damage_min=10, damage_max=10)
SPELLS = [BOLT, BLAST]
SPACE = ActionSpace.build(BOARD, SPELLS)


def _state(me=CENTRE, enemies=(FOE,), ap=6, mp=3):
    entities = [Entity(entity_id="me", team=Team.ALLY, cell=me, hp=100, hp_max=100,
                       ap=ap, mp=mp, is_self=True)]
    entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=c, hp=100, hp_max=100,
                        ap=6, mp=3) for i, c in enumerate(enemies)]
    return CombatState(turn=1, entities=entities)


def _table(state, turn=None):
    turn = turn or TurnState.from_combat(state)
    return legal_table(SPACE, BOARD, state, SPELLS, turn)


class TestLayout:
    def test_size_covers_every_cell_and_spell(self):
        assert SPACE.size == 1 + len(BOARD) * (1 + len(SPELLS))

    def test_size_is_constant_whatever_the_state(self):
        """Un reseau a une tete de taille fixe : l'espace ne peut pas dependre de
        l'etat. C'est le masque qui varie, jamais la numerotation."""
        sizes = {action_mask(SPACE, _table(state)).size
                 for state in (_state(), _state(ap=0, mp=0), _state(enemies=()),
                               _state(me=0), _state(enemies=(1, 2, 3)))}
        assert sizes == {SPACE.size}

    def test_indices_are_distinct(self):
        indices = ([0]
                   + [SPACE.move_index(c) for c in range(len(BOARD))]
                   + [SPACE.cast_index(s.name, c)
                      for s in SPELLS for c in range(len(BOARD))])
        assert len(indices) == len(set(indices)) == SPACE.size

    def test_last_index_is_exactly_the_last_slot(self):
        """Le dernier lancer du dernier sort doit tomber sur le dernier emplacement :
        un decalage laisserait des indices inatteignables ou deborderait du masque."""
        assert SPACE.cast_index(SPELLS[-1].name, len(BOARD) - 1) == SPACE.size - 1

    def test_round_trip_through_index_of(self):
        for action in (EndTurn(), Move(cell=12, cost=1),
                       Cast(spell="souffle", target=30, cost=4)):
            index = SPACE.index_of(action)
            assert 0 <= index < SPACE.size
        assert SPACE.index_of(Move(cell=12, cost=1)) == SPACE.move_index(12)

    def test_describe_names_the_action(self):
        assert "fin de tour" in SPACE.describe(0)
        assert "deplacement" in SPACE.describe(SPACE.move_index(5))
        assert "souffle" in SPACE.describe(SPACE.cast_index("souffle", 5))


class TestMask:
    def test_shape_matches_the_space(self):
        assert action_mask(SPACE, _table(_state())).shape == (SPACE.size,)

    def test_end_turn_is_always_available(self):
        """Un masque entierement faux bloquerait l'echantillonnage de la politique --
        panne qui se manifeste par un NaN plusieurs epoques plus tard."""
        for state in (_state(), _state(ap=0, mp=0), _state(enemies=())):
            assert action_mask(SPACE, _table(state))[0]

    def test_never_entirely_false(self):
        assert action_mask(SPACE, _table(_state(ap=0, mp=0))).any()

    def test_casts_disappear_without_ap(self):
        rich = action_mask(SPACE, _table(_state(ap=6)))
        poor = action_mask(SPACE, _table(_state(ap=2)))
        assert poor.sum() < rich.sum()

    def test_moves_disappear_without_mp(self):
        moves = slice(1, 1 + len(BOARD))
        assert not action_mask(SPACE, _table(_state(mp=0)))[moves].any()

    def test_out_of_range_target_is_masked_out(self):
        far = BOARD.index_at(3, 6)          # a 3 cases, hors portee de souffle (max 2)
        assert not action_mask(SPACE, _table(_state()))[SPACE.cast_index("souffle", far)]

    def test_ended_turn_offers_nothing(self):
        turn = TurnState(cell=CENTRE, ap=6, mp=3, ended=True)
        assert not action_mask(SPACE, _table(_state(), turn)).any()


class TestSimulatorAgreement:
    """LE test du module : masque et simulateur doivent dire la meme chose."""

    def test_every_masked_in_action_is_accepted(self):
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(2, 2)))
        table = _table(state)
        assert table, "aucune action legale : le test ne verifierait rien"
        for index, action in table.items():
            outcome = apply_actions(BOARD, state, SPELLS, [action])
            assert outcome.illegal == [], (
                f"{SPACE.describe(index)} est masquee comme jouable mais refusee "
                f"par le simulateur")

    def test_a_masked_out_cast_is_refused(self):
        """Reciproque : ce que le masque interdit, le simulateur le refuse aussi."""
        state = _state()
        far = BOARD.index_at(3, 6)
        mask = action_mask(SPACE, _table(state))
        assert not mask[SPACE.cast_index("souffle", far)]
        outcome = apply_actions(BOARD, state, SPELLS,
                                [Cast(spell="souffle", target=far, cost=4)])
        assert outcome.illegal

    def test_masked_in_moves_cost_what_they_claim(self):
        """Le simulateur recalcule les couts : si le masque annoncait un cout faux, la
        politique apprendrait une economie de PM qui n'existe pas."""
        state = _state()
        moves = [a for a in _table(state).values() if isinstance(a, Move)]
        assert moves, "aucun deplacement legal : le test ne verifierait rien"
        for action in moves:
            outcome = apply_actions(BOARD, state, SPELLS, [action])
            me = next(e for e in outcome.state.entities if e.is_self)
            assert me.mp == state.self_entity().mp - action.cost


class TestRobustness:
    def test_unknown_spell_is_dropped_not_fatal(self):
        """Une config changee en cours d'entrainement ne doit pas tuer l'episode : un
        coup de moins vaut mieux qu'un run perdu au bout de trois heures."""
        narrow = ActionSpace.build(BOARD, [BOLT])
        state = _state()
        table = legal_table(narrow, BOARD, state, SPELLS, TurnState.from_combat(state))
        assert table and all(not isinstance(a, Cast) or a.spell == "trait"
                             for a in table.values())

    def test_mask_dtype_is_boolean(self):
        assert action_mask(SPACE, _table(_state())).dtype == np.bool_
