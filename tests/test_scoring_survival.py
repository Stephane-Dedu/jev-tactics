"""Terme de survie de la fonction d'evaluation.

Ne vient pas d'une intuition mais d'une mesure. L'arene disait : 40 des 44 defaites se
jouent « a un ennemi pres » -- le solveur en tue un puis meurt sur l'autre -- et meme ses
victoires finissent a 6 PV en mediane. Il courait donc une course sans jamais tenir
compte de SA propre survie : la prudence valait autant a 100 PV qu'a 5.

Une tentative precedente d'amelioration, elle fondee sur une intuition, avait coute 6
points (cf. §6.1.b du doc d'archi). D'ou ces tests : ils fixent le SENS du terme, ce
qu'un taux de victoire global ne verifie jamais."""


from jev_tactics.planner.legal import TurnState
from jev_tactics.planner.scoring import SURVIVAL_WEIGHT, danger_factor, evaluate
from jev_tactics.sim import square_board
from jev_tactics.state import CombatState, Entity, Team

BOARD = square_board(7)
CENTRE = BOARD.index_at(3, 3)
FOE = BOARD.index_at(3, 5)


def _state(hp=100, hp_max=100, foe=FOE, hp_known=True, with_self=True):
    entities = []
    if with_self:
        entities.append(Entity(entity_id="me", team=Team.ALLY, cell=CENTRE, hp=hp,
                               hp_max=hp_max, ap=6, mp=3, is_self=True))
    entities.append(Entity(entity_id="e0", team=Team.ENEMY, cell=foe, hp=100,
                           hp_max=100, ap=6, mp=3, hp_known=hp_known))
    return CombatState(turn=1, entities=entities)


class TestDangerFactor:
    def test_neutral_at_full_health(self):
        assert danger_factor(_state(hp=100)) == 1.0

    def test_maximal_when_nearly_dead(self):
        assert danger_factor(_state(hp=0)) == 1.0 + SURVIVAL_WEIGHT

    def test_increases_as_health_drops(self):
        factors = [danger_factor(_state(hp=hp)) for hp in (100, 75, 50, 25, 1)]
        assert factors == sorted(factors)

    def test_scales_on_the_ratio_not_the_absolute(self):
        """Un personnage a 500/1000 PV est aussi en danger qu'un a 50/100 : c'est la
        proportion qui compte, sinon le terme ne vaudrait que pour un seul niveau."""
        assert danger_factor(_state(hp=50, hp_max=100)) == danger_factor(
            _state(hp=500, hp_max=1000))

    def test_neutral_without_a_readable_self(self):
        """Supposer le pire rendrait le bot craintif sur une simple panne de lecture --
        et un bot qui fuit sans raison ressemble a un bot prudent."""
        assert danger_factor(_state(with_self=False)) == 1.0


class TestEffectOnScoring:
    def _score(self, hp, cell):
        state = _state(hp=hp)
        turn = TurnState(cell=cell, ap=0, mp=0)
        return evaluate(BOARD, state, turn, {})

    def test_retreating_is_worth_more_when_wounded(self):
        """Le coeur du terme : la meme case de repli doit valoir davantage a 10 PV
        qu'a 100."""
        near, far = BOARD.index_at(3, 4), BOARD.index_at(0, 0)
        healthy = self._score(100, far) - self._score(100, near)
        wounded = self._score(10, far) - self._score(10, near)
        assert wounded > healthy

    def test_distance_still_counts_at_full_health(self):
        """Le terme module la prudence, il ne l'annule pas : intact, s'eloigner doit
        rester legerement preferable."""
        assert self._score(100, BOARD.index_at(0, 0)) > self._score(100, FOE)

    def test_applies_to_enemies_with_unknown_hp_too(self):
        """Les PV adverses inconnus sont le cas reel (la timeline ne s'apparie pas au
        plateau) : le terme doit y valoir aussi, sinon il ne servirait qu'en simulation."""
        turn_near = TurnState(cell=BOARD.index_at(3, 4), ap=0, mp=0)
        turn_far = TurnState(cell=BOARD.index_at(0, 0), ap=0, mp=0)
        gap = lambda hp: (evaluate(BOARD, _state(hp=hp, hp_known=False), turn_far, {})
                          - evaluate(BOARD, _state(hp=hp, hp_known=False), turn_near, {}))
        assert gap(10) > gap(100)

    def test_a_kill_still_outweighs_fleeing(self):
        """Garde-fou contre la derive inverse : rendre la prudence trop chere ferait
        fuir un solveur qui pouvait achever. Une mise a mort doit rester meilleure."""
        state = _state(hp=5)
        finish = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=0, mp=0),
                          {"e0": 100.0})
        flee = evaluate(BOARD, state, TurnState(cell=BOARD.index_at(0, 0), ap=0, mp=0),
                        {})
        assert finish > flee
