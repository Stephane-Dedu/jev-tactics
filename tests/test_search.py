"""Elagage, evaluation et solveur 1 tour.

Les tests de decision sont ecrits comme des SCENARIOS : on construit une situation dont
le bon coup est evident pour un humain, et on verifie que le solveur le trouve."""


import random
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner import (
    Cast,
    EndTurn,
    Move,
    TurnState,
    best_sequence,
    evaluate,
    legal_actions,
    useful_actions,
)
from jev_tactics.planner.prune import build_shot_table, offensive, useful_targets
from jev_tactics.planner.scoring import expected_damage
from jev_tactics.planner.search import approach_move
from jev_tactics.rules.movement import grid_distance
from jev_tactics.rules.roles import parse_assignment, spells_from_roles
from jev_tactics.rules.spells import Spell, load_spells
from jev_tactics.sim import apply_actions, square_board
from jev_tactics.sim.arena import random_scenario
from jev_tactics.state import CombatState, Entity, Team

BOARD = BoardMap(
    cells=np.array([(i, j) for j in range(7) for i in range(7)], dtype=np.int64),
    e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
    origin=np.array([500.0, 300.0]),
)
CENTRE = BOARD.index_at(3, 3)

BOLT = Spell(name="trait", ap_cost=3, range_min=1, range_max=6,
             needs_line_of_sight=False, damage_min=20, damage_max=20)
BLAST = Spell(name="souffle", ap_cost=4, range_min=1, range_max=6, area_radius=1,
              needs_line_of_sight=False, damage_min=10, damage_max=10)


def _state(me=CENTRE, enemies=(), ap=6, mp=3, enemy_hp=100, hp_known=True):
    entities = [Entity(entity_id="me", team=Team.ALLY, cell=me, hp=100, hp_max=100,
                       ap=ap, mp=mp, is_self=True)]
    entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=c, hp=enemy_hp,
                        hp_max=100, ap=6, mp=3, hp_known=hp_known)
                 for i, c in enumerate(enemies)]
    return CombatState(turn=1, entities=entities)


class TestPruning:
    def test_casts_on_empty_cells_are_dropped(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        turn = TurnState.from_combat(state)
        legal = [a for a in legal_actions(BOARD, state, [BOLT], turn) if isinstance(a, Cast)]
        pruned = [a for a in useful_actions(BOARD, state, [BOLT], turn) if isinstance(a, Cast)]
        assert len(pruned) < len(legal)
        assert {a.target for a in pruned} == {BOARD.index_at(3, 5)}

    def test_pruned_actions_are_a_subset_of_legal(self):
        """Invariant : l'elagage ne doit jamais inventer un coup illegal."""
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(1, 2)))
        turn = TurnState.from_combat(state)
        legal = {a.model_dump_json() for a in legal_actions(BOARD, state, [BOLT], turn)}
        for action in useful_actions(BOARD, state, [BOLT], turn):
            assert action.model_dump_json() in legal

    def test_the_subset_holds_on_random_states_where_the_caps_bind(self):
        """LE TEST CI-DESSUS NE POUVAIT PAS VOIR LA DIVERGENCE LA PLUS PROBABLE.

        Il porte sur UN etat fabrique, UN sort, et un tour VIERGE. Or les plafonds de
        lancers -- par tour et par cible -- sont implementes DEUX FOIS, dans `legal.py` et
        dans `prune.py`, et deux ecritures d'une meme regle finissent par diverger. Sur un
        tour sans aucun lancer, ces plafonds ne mordent jamais : la partie du code la plus
        exposee n'etait pas exercee.

        Ici : soixante scenarios d'arene, trois etats de tour PARTIELS chacun -- des
        lancers et des PM deja consommes. La mesure confirme que la regle mord vraiment,
        sur 70 des 180 etats pour le plafond par tour et 52 pour celui par cible, et le
        troisieme test de cette serie l'exige plutot que de l'esperer.

        Resultat : 0 action elaguee hors des legales. Les deux ecritures s'accordent."""
        board = square_board(11)
        spells = spells_from_roles(parse_assignment("1=cac,2=dps_longue,3=zone"))
        tirage = random.Random(7)
        hors = []
        for graine in range(60):
            state = random_scenario(board, rng=random.Random(graine),
                                    enemies=tirage.choice([1, 2, 3]))
            for turn in self._tours_partiels(state, spells, tirage):
                legaux = {a.model_dump_json()
                          for a in legal_actions(board, state, spells, turn)}
                hors += [a for a in useful_actions(board, state, spells, turn)
                         if a.model_dump_json() not in legaux]
        assert not hors, f"{len(hors)} action(s) elaguee(s) illegale(s), ex. {hors[:2]}"

    @staticmethod
    def _tours_partiels(state, spells, tirage, combien=3):
        """Etats de tour ou des lancers ont DEJA ete faits -- le regime que le test
        d'origine n'atteignait pas."""
        base = TurnState.from_combat(state)
        for _ in range(combien):
            reste = base.ap
            faits = []
            for _ in range(tirage.randint(0, 4)):
                sort = tirage.choice(spells)
                if reste < sort.ap_cost:
                    continue
                faits.append((sort.name, tirage.randrange(64)))
                reste -= sort.ap_cost
            yield TurnState(cell=base.cell, ap=reste,
                            mp=max(0, base.mp - tirage.randint(0, 2)),
                            casts=tuple(faits))

    def test_those_random_states_really_make_the_caps_bite(self):
        """CONTRE-EPREUVE INDISPENSABLE : un tirage qui ne saturerait jamais les plafonds
        donnerait le meme « 0 divergence » que le test d'origine, en paraissant plus fort.
        C'est le defaut qu'on vient de corriger, il ne faut pas le reintroduire ici."""
        board = square_board(11)
        spells = spells_from_roles(parse_assignment("1=cac,2=dps_longue,3=zone"))
        tirage = random.Random(7)
        par_tour = par_cible = total = 0
        for graine in range(60):
            state = random_scenario(board, rng=random.Random(graine),
                                    enemies=tirage.choice([1, 2, 3]))
            for turn in self._tours_partiels(state, spells, tirage):
                total += 1
                par_tour += any(turn.casts_of(s.name) >= s.max_casts_per_turn
                                for s in spells)
                par_cible += any(turn.casts_on(s.name, cible) >= s.max_casts_per_target
                                 for s in spells for _n, cible in turn.casts)
        assert par_tour > total // 5 and par_cible > total // 5, (
            f"plafond par tour mordant sur {par_tour}/{total}, par cible {par_cible}/"
            f"{total} : l'echantillon n'exerce plus la regle qu'il pretend verifier")

    def test_area_spell_keeps_splash_targets(self):
        """Un sort de zone peut viser a cote et toucher quand meme."""
        enemy = BOARD.index_at(3, 5)
        state = _state(enemies=(enemy,))
        turn = TurnState.from_combat(state)
        targets = {a.target for a in useful_actions(BOARD, state, [BLAST], turn)
                   if isinstance(a, Cast)}
        assert len(targets) > 1 and enemy in targets

    def test_useful_targets_covers_area_radius(self):
        cells = useful_targets(BOARD, [BLAST], {CENTRE})
        assert len(cells) == 5           # la case + ses 4 voisins
        assert useful_targets(BOARD, [BOLT], {CENTRE}) == {CENTRE}

    def test_end_turn_always_survives_pruning(self):
        state = _state()
        actions = useful_actions(BOARD, state, [BOLT], TurnState.from_combat(state))
        assert any(isinstance(a, EndTurn) for a in actions)

    def test_shot_table_matches_direct_computation(self):
        """Le precalcul doit donner exactement le meme resultat que le calcul direct --
        c'est ce qui autorise l'optimisation (x400 a x800 mesures)."""
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(1, 1)))
        turn = TurnState.from_combat(state)
        table = build_shot_table(BOARD, state, [BOLT, BLAST], [CENTRE])
        with_table = {(a.spell, a.target)
                      for a in useful_actions(BOARD, state, [BOLT, BLAST], turn, table)
                      if isinstance(a, Cast)}
        without = {(a.spell, a.target)
                   for a in useful_actions(BOARD, state, [BOLT, BLAST], turn)
                   if isinstance(a, Cast)}
        assert with_table == without


class TestEvaluate:
    def test_damage_capped_at_remaining_hp(self):
        assert expected_damage(80.0, 30) == 30.0
        assert expected_damage(20.0, 30) == 20.0

    def test_kill_scores_above_equal_damage_spread(self):
        """Achever un ennemi vaut mieux qu'egratigner deux : c'est le role du bonus."""
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(1, 1)), enemy_hp=20)
        turn = TurnState(cell=CENTRE, ap=0, mp=0)
        kill = evaluate(BOARD, state, turn, {"e0": 20.0})
        spread = evaluate(BOARD, state, turn, {"e0": 10.0, "e1": 10.0})
        assert kill > spread

    def test_unspent_resources_penalised(self):
        state = _state(enemies=(BOARD.index_at(3, 5),))
        spent = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=0, mp=0), {})
        idle = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=6, mp=3), {})
        assert spent > idle

    def test_no_kill_bonus_when_hp_unknown(self):
        """PV inconnus : pas de prime de MISE A MORT -- elle supposerait de savoir ce qui
        reste. Le placeholder valait 1 PV, ce qui faisait passer chaque attaque pour un
        achevement.

        Ce qui subsiste est la prime d'ENTAME (WOUNDED_WEIGHT), qui n'a besoin que du
        RATIO : elle rattrape la majeure partie de l'ecart mesure en arene, 35,0 % contre
        85,8 %. Sur un ennemi entame a 80 %, elle depasse KILL_BONUS -- d'ou un ecart
        NEGATIF ici, et c'est le comportement voulu."""
        turn = TurnState(cell=CENTRE, ap=0, mp=0)
        known = _state(enemies=(BOARD.index_at(3, 5),), enemy_hp=20)
        unknown = _state(enemies=(BOARD.index_at(3, 5),), enemy_hp=20, hp_known=False)
        gap = evaluate(BOARD, known, turn, {"e0": 20.0}) - evaluate(
            BOARD, unknown, turn, {"e0": 20.0})
        assert gap < 0

    def test_unknown_hp_still_values_damage(self):
        """On ignore les seuils, pas les degats : frapper reste mieux que ne rien faire."""
        turn = TurnState(cell=CENTRE, ap=0, mp=0)
        state = _state(enemies=(BOARD.index_at(3, 5),), hp_known=False)
        assert evaluate(BOARD, state, turn, {"e0": 20.0}) > evaluate(BOARD, state, turn, {})

    def test_distance_from_survivors_rewarded(self):
        state = _state(enemies=(BOARD.index_at(6, 6),))
        near = evaluate(BOARD, state, TurnState(cell=BOARD.index_at(5, 6), ap=0, mp=0), {})
        far = evaluate(BOARD, state, TurnState(cell=BOARD.index_at(0, 0), ap=0, mp=0), {})
        assert far > near


class TestSolver:
    def test_finds_the_available_kill(self):
        """Ennemi a 20 PV a portee : le solveur doit tirer dessus."""
        enemy = BOARD.index_at(3, 5)
        state = _state(enemies=(enemy,), ap=3, mp=0, enemy_hp=20)
        plan = best_sequence(BOARD, state, [BOLT])
        assert any(isinstance(a, Cast) and a.target == enemy for a in plan.actions)

    def test_moves_into_range_when_out_of_reach(self):
        close = Spell(name="corps", ap_cost=3, range_min=1, range_max=1,
                      needs_line_of_sight=False, damage_min=30, damage_max=30)
        enemy = BOARD.index_at(3, 6)
        state = _state(me=BOARD.index_at(3, 3), enemies=(enemy,), ap=3, mp=3)
        plan = best_sequence(BOARD, state, [close])
        assert any(isinstance(a, Move) for a in plan.actions)
        assert any(isinstance(a, Cast) for a in plan.actions)

    def test_passes_when_nothing_useful(self):
        state = _state(ap=0, mp=0)
        assert best_sequence(BOARD, state, [BOLT]).actions == []

    def test_respects_ap_budget(self):
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=5, mp=0)
        plan = best_sequence(BOARD, state, [BOLT])   # 3 PA le lancer -> un seul tient
        assert sum(a.cost for a in plan.actions if isinstance(a, Cast)) <= 5

    def test_respects_mp_budget(self):
        state = _state(enemies=(BOARD.index_at(3, 6),), ap=0, mp=2)
        plan = best_sequence(BOARD, state, [BOLT])
        assert sum(a.cost for a in plan.actions if isinstance(a, Move)) <= 2

    def test_never_two_moves_in_a_row(self):
        """Deux deplacements consecutifs sont domines par le trajet direct."""
        state = _state(enemies=(BOARD.index_at(0, 0),), ap=6, mp=4)
        actions = best_sequence(BOARD, state, [BOLT]).actions
        for previous, current in zip(actions, actions[1:]):
            assert not (isinstance(previous, Move) and isinstance(current, Move))

    def test_plan_is_replayable_from_legal_actions(self):
        """Invariant central : chaque action du plan doit etre legale a son tour de jeu.
        Sans lui, le bot tenterait des coups que le jeu refuserait."""
        from jev_tactics.planner import apply_action

        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(1, 1)), ap=9, mp=3)
        plan = best_sequence(BOARD, state, [BOLT, BLAST])
        turn = TurnState.from_combat(state)
        for action in plan.actions:
            available = legal_actions(BOARD, state, [BOLT, BLAST], turn)
            assert any(a.model_dump() == action.model_dump() for a in available)
            turn = apply_action(turn, action)

    def test_node_cap_bounds_the_search(self):
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(1, 1)), ap=12, mp=4)
        plan = best_sequence(BOARD, state, [BOLT, BLAST], max_nodes=20)
        assert plan.nodes <= 20 and plan.truncated

    def test_truncated_search_still_returns_a_plan(self):
        """Degrader la qualite est acceptable ; bloquer le bot ne l'est pas."""
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=12, mp=4)
        plan = best_sequence(BOARD, state, [BOLT, BLAST], max_nodes=5)
        assert plan.score > float("-inf")

    def test_describe_is_human_readable(self):
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=3, mp=0)
        assert "trait" in best_sequence(BOARD, state, [BOLT]).describe()


class TestApproach:
    """Repli d'approche. Defaut constate en jeu : trois ennemis, 10 PA, et « passer le
    tour » -- indefiniment, puisque la situation se reproduit au tour suivant. Avec des
    sorts de portee 1 a 3, le bot ne pouvait JAMAIS engager le combat."""

    def _far(self, size=15, mp=4):
        board = square_board(size)
        me = board.index_at(2, 2)
        foes = [board.index_at(12, 12), board.index_at(11, 13)]
        entities = [Entity(entity_id="me", team=Team.ALLY, cell=me, hp=990, hp_max=1642,
                           ap=10, mp=mp, is_self=True)]
        entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=c, hp=1000,
                            hp_max=1000, ap=6, mp=3, hp_known=False)
                     for i, c in enumerate(foes)]
        return board, CombatState(turn=1, entities=entities), foes

    def test_closes_distance_instead_of_passing(self):
        board, state, foes = self._far()
        spells = [Spell(name="courte", ap_cost=3, range_min=1, range_max=3,
                        needs_line_of_sight=False, damage_min=20, damage_max=20)]
        plan = best_sequence(board, state, spells)
        assert plan.actions, "le bot passe le tour au lieu d'engager"
        me = state.self_entity().cell
        before = min(grid_distance(board, me, f) for f in foes)
        after = min(grid_distance(board, plan.actions[0].cell, f) for f in foes)
        assert after < before

    def test_does_not_glue_itself_to_the_enemy(self):
        """Il suffit d'etre a portee du tour PROCHAIN : aller plus pres n'ouvre aucun
        coup et avance le moment ou l'on encaisse."""
        board, state, foes = self._far()
        spells = [Spell(name="longue", ap_cost=3, range_min=1, range_max=6,
                        needs_line_of_sight=False, damage_min=20, damage_max=20)]
        plan = best_sequence(board, state, spells)
        assert plan.actions
        after = min(grid_distance(board, plan.actions[0].cell, f) for f in foes)
        assert after >= 6, "trop pres : le palier de portee n'a pas joue"

    def test_never_overrides_a_plan_that_attacks(self):
        """Le repli ne s'applique que lorsqu'il n'y avait rien a perdre."""
        board = square_board(9)
        me, foe = board.index_at(4, 4), board.index_at(4, 6)
        state = CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=me, hp=100, hp_max=100,
                   ap=6, mp=3, is_self=True),
            Entity(entity_id="e0", team=Team.ENEMY, cell=foe, hp=100, hp_max=100,
                   ap=6, mp=3)])
        spells = [Spell(name="tir", ap_cost=3, range_min=1, range_max=4,
                        needs_line_of_sight=False, damage_min=20, damage_max=20)]
        plan = best_sequence(board, state, spells)
        assert any(isinstance(a, Cast) for a in plan.actions)

    def test_no_approach_without_enemies(self):
        board = square_board(9)
        state = CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(4, 4), hp=100,
                   hp_max=100, ap=6, mp=3, is_self=True)])
        assert approach_move(board, state, [], {}) is None

    def test_no_approach_when_already_in_reach(self):
        """Deja a portee du tour prochain : bouger ne servirait qu'a s'exposer."""
        board = square_board(9)
        me, foe = board.index_at(4, 4), board.index_at(4, 6)
        state = CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=me, hp=100, hp_max=100,
                   ap=6, mp=3, is_self=True),
            Entity(entity_id="e0", team=Team.ENEMY, cell=foe, hp=100, hp_max=100,
                   ap=6, mp=3)])
        spells = [Spell(name="tir", ap_cost=3, range_min=1, range_max=4,
                        needs_line_of_sight=False, damage_min=20, damage_max=20)]
        assert approach_move(board, state, spells, {me: 0}) is None

    def test_the_approach_is_accepted_by_the_simulator(self):
        """Coherence croisee : un repli qui produirait un coup illegal serait pire que
        de passer le tour."""
        board, state, _ = self._far()
        spells = [Spell(name="courte", ap_cost=3, range_min=1, range_max=3,
                        needs_line_of_sight=False, damage_min=20, damage_max=20)]
        plan = best_sequence(board, state, spells)
        assert apply_actions(board, state, spells, plan.actions).illegal == []


class TestOffensiveFilter:
    """Sorts sans degat exclus de la recherche.

    L'evaluation ne note que les degats : un sort a 0 degat ne peut jamais ameliorer un
    score, mais multiplie l'arbre. Mesure sur la config reelle du Sacrieur (19 sorts dont
    9 utilitaires) : a plafond de noeuds egal, le score du plan passe de 96,2 a 102,2 --
    la recherche explore plus profondement ce qui compte."""

    def test_zero_damage_spells_are_dropped(self):
        spells = [
            Spell(name="frappe", ap_cost=3, range_max=4, damage_min=10, damage_max=12),
            Spell(name="buff", ap_cost=2, range_max=6, damage_min=0, damage_max=0),
        ]
        assert [s.name for s in offensive(spells)] == ["frappe"]

    def test_an_all_utility_kit_leaves_nothing(self):
        spells = [Spell(name="buff", ap_cost=2, range_max=6)]
        assert offensive(spells) == []

    def test_the_solver_survives_a_kit_without_damage(self):
        """Il ne doit pas planter, seulement n'avoir rien a lancer."""
        board = square_board(7)
        state = CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(3, 3), hp=100,
                   hp_max=100, ap=6, mp=3, is_self=True),
            Entity(entity_id="e0", team=Team.ENEMY, cell=board.index_at(3, 5), hp=100,
                   hp_max=100, ap=6, mp=3)])
        plan = best_sequence(board, state, [Spell(name="buff", ap_cost=2, range_max=6)])
        assert not any(isinstance(a, Cast) for a in plan.actions)

class TestTheSolverObeysTheGameZone:
    """Le solveur recalculait sa propre zone de deplacement et ignorait celle du jeu.

    `legal.py` explique pourtant pourquoi le surlignage FAIT AUTORITE : « c'est le
    resultat de SON pathfinding, obstacles compris, alors que le notre ne connait que les
    obstacles qu'on a su detecter ». Ce raisonnement etait ecrit, applique la-bas, et
    ignore ici -- dans le code qui planifie vraiment.

    Ecart MESURE sur les captures reelles : combat1 12 cases calculees contre 10
    surlignees, et surtout tacle.png 20 contre 3. Cette capture porte ce nom parce que le
    personnage y est TACLE : il perd ses PM, le jeu le sait, nous l'ignorions.

    Honnetete : sur ces captures, le plan RETENU ne changeait pas -- le solveur n'avait
    pas choisi l'une des cases hors zone. Le defaut est donc latent, pas une erreur
    demontree en jeu. Le scenario ci-dessous, lui, le rend certain."""

    def _far_enemy_state(self, hint=None):
        """Ennemi hors de portee depuis le depart : il FAUT bouger pour le toucher."""
        state = _state(me=BOARD.index_at(0, 0), enemies=(BOARD.index_at(6, 6),), mp=6)
        state.reachable_hint = hint
        return state

    def test_without_a_hint_it_moves_freely(self):
        """Controle : sans surlignage, le BFS interne fait foi et le solveur bouge."""
        plan = best_sequence(BOARD, self._far_enemy_state(), [BOLT])
        assert any(isinstance(a, Move) for a in plan.actions)

    def test_a_tackled_character_barely_moves(self):
        """Le cas de tacle.png : le jeu n'accorde qu'un pas, la ou notre BFS en voyait
        six. Le solveur doit s'en tenir a ce que le jeu autorise.

        La zone doit TOUCHER le personnage pour etre retenue -- `_plausible_hint` ecarte
        une lecture partielle plutot que de se laisser paralyser par elle. On surligne
        donc le depart ET un voisin, ce qui est le vrai profil d'un tacle."""
        start, step = BOARD.index_at(0, 0), BOARD.index_at(1, 0)
        plan = best_sequence(BOARD, self._far_enemy_state(hint={start, step}), [BOLT])
        moves = [a.cell for a in plan.actions if isinstance(a, Move)]
        assert set(moves) <= {start, step}
        assert len(moves) <= 1

    def test_an_unreadable_hint_falls_back_instead_of_paralysing(self):
        """Garde-fou existant qu'il ne faut pas casser : une zone qui ne touche pas le
        personnage est une lecture ratee. La prendre au mot donnait « 0 case atteignable
        avec 4 PM », tour passe pour rien -- constate en jeu."""
        plan = best_sequence(BOARD, self._far_enemy_state(hint={BOARD.index_at(6, 0)}),
                             [BOLT])
        assert any(isinstance(a, Move) for a in plan.actions)

    def test_it_never_plans_a_move_outside_the_zone(self):
        """Le point qui compte : un deplacement hors zone ne rate pas un coup. Le clic ne
        mene nulle part, et le sort qui suit part de la MAUVAISE case."""
        allowed = {BOARD.index_at(0, 0), BOARD.index_at(1, 0), BOARD.index_at(0, 1)}
        plan = best_sequence(BOARD, self._far_enemy_state(hint=allowed), [BOLT])
        moves = [a.cell for a in plan.actions if isinstance(a, Move)]
        assert set(moves) <= allowed, f"deplacement hors de la zone du jeu : {moves}"

    def test_the_approach_fallback_obeys_it_too(self):
        """LA consequence concrete, et la seule ou l'ancien code produisait vraiment un
        mauvais coup.

        Les Move de la recherche passent par `legal_actions`, qui honorait deja la zone :
        les origines fausses n'y ajoutaient que du calcul inutile. Le repli d'approche,
        lui, choisit DIRECTEMENT une case dans `origins` -- il rattrapait donc par la
        fenetre le deplacement interdit par la porte.

        Mise en scene : sort de portee 2, ennemi a 12 cases. Aucun tir possible, donc le
        repli se declenche a coup sur."""
        short = Spell(name="dague", ap_cost=3, range_min=1, range_max=2,
                      needs_line_of_sight=False, damage_min=20, damage_max=20)
        allowed = {BOARD.index_at(0, 0), BOARD.index_at(1, 0)}
        plan = best_sequence(BOARD, self._far_enemy_state(hint=allowed), [short])
        moves = [a.cell for a in plan.actions if isinstance(a, Move)]
        assert moves, "le repli d'approche ne s'est pas declenche"
        assert set(moves) <= allowed, f"deplacement hors de la zone du jeu : {moves}"

class TestItStopsHittingCorpses:
    """Signale en jeu sous deux formes qui n'en font qu'une : « il frappe un ennemi deja
    mort » et « il tape une case morte (terrain) ». C'est la MEME case -- une fois le
    cadavre disparu, il ne reste que le decor.

    Le solveur ne se trompait pas, il optimisait ce qu'on lui avait demande : les degats
    sont plafonnes aux PV restants, donc un lancer sur un mort rapporte 0 -- mais il
    depense 3 PA, et la penalite de PA non depense (2,0/PA) recule d'autant. Mesure sur
    un ennemi a 20 PV, sort a 50 degats, 9 PA :

        0 lancer -> -16,5 | 1 -> 66,5 | 2 -> 72,5 | 3 -> 78,5

    Le score MONTE de 6 par lancer sans effet."""

    KILLER = Spell(name="foudre", ap_cost=3, range_min=1, range_max=6,
                   needs_line_of_sight=False, damage_min=50, damage_max=50)

    def test_one_cast_is_enough_for_a_dying_enemy(self):
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=9, enemy_hp=20)
        casts = [a for a in best_sequence(BOARD, state, [self.KILLER]).actions
                 if isinstance(a, Cast)]
        assert len(casts) == 1, "des lancers partent sur un cadavre"

    def test_the_spare_ap_goes_to_ANOTHER_enemy(self):
        """Ce qu'on veut vraiment : changer de cible, pas economiser les PA."""
        state = _state(enemies=(BOARD.index_at(3, 5), BOARD.index_at(4, 3)),
                       ap=9, enemy_hp=20)
        targets = {a.target for a in best_sequence(BOARD, state, [self.KILLER]).actions
                   if isinstance(a, Cast)}
        assert len(targets) == 2

    def test_a_tough_enemy_still_takes_every_cast(self):
        """Garde-fou inverse : sans lui, on aurait « corrige » en cessant d'insister sur
        une cible qui encaisse -- ce qui perdrait des combats."""
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=9, enemy_hp=200)
        casts = [a for a in best_sequence(BOARD, state, [self.KILLER]).actions
                 if isinstance(a, Cast)]
        assert len(casts) == 3

    def test_unknown_hp_is_never_taken_for_dead(self):
        """Supposer une mise a mort qu'on n'a pas mesuree ferait abandonner une cible bien
        vivante. Meme principe que l'evaluation, qui refuse la prime sans PV connus."""
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=9, enemy_hp=20,
                       hp_known=False)
        casts = [a for a in best_sequence(BOARD, state, [self.KILLER]).actions
                 if isinstance(a, Cast)]
        assert len(casts) == 3

    def test_an_area_spell_still_fires_if_it_reaches_a_survivor(self):
        """Une zone qui couvre un cadavre ET un vivant reste utile : filtrer sur la seule
        case visee ferait perdre des coups corrects."""
        dying, alive = BOARD.index_at(3, 5), BOARD.index_at(3, 4)
        state = _state(enemies=(dying, alive), ap=9, enemy_hp=20)
        area = Spell(name="souffle", ap_cost=4, range_min=1, range_max=6, area_radius=1,
                     needs_line_of_sight=False, damage_min=50, damage_max=50)
        plan = best_sequence(BOARD, state, [area])
        assert any(isinstance(a, Cast) for a in plan.actions)

class TestTheTiebreakersCannotDecideAlone:
    """Les penalites de ressources non depensees sont documentees comme un DEPARTAGE.
    A 2,0 par PA elles decidaient : un lancer rapportait 6 points quel que soit son effet,
    donc frapper un cadavre battait ne rien faire.

    Baisser ces poids a ete MESURE en arene avant d'etre fait : 80,3% de victoires a
    PA=2,0 PM=0,50 contre 80,7% a PA=0,1 PM=0,05, sur 5 graines et 300 combats. Aucun cout
    -- l'arene a toujours un ennemi vivant a frapper, donc le departage n'y decide jamais.
    C'est exactement pourquoi le baisser est sans risque la ou elle mesure, et utile la ou
    elle ne mesure pas."""

    def test_a_cast_with_no_effect_gains_less_than_one_point(self):
        """Le seuil qui compte : un coup sans effet ne doit jamais valoir un seul point
        de degat reel."""
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=9, enemy_hp=20)
        dead = {"e0": 50.0}
        spent = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=3, mp=3), dead)
        kept = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=6, mp=3), dead)
        assert 0 < spent - kept < 1.0

    def test_a_step_with_no_enemy_gains_almost_nothing(self):
        """Il valait +0,50, ce qui suffisait a faire marcher le bot au hasard quand la
        perception ne voyait aucun ennemi."""
        state = _state()
        moved = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=6, mp=2), {})
        still = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=6, mp=3), {})
        assert 0 < moved - still <= 0.1

    def test_it_still_breaks_a_genuine_tie(self):
        """Le departage doit SURVIVRE : a degats egaux, depenser vaut mieux que garder."""
        state = _state(enemies=(BOARD.index_at(3, 5),))
        damage = {"e0": 10.0}
        spent = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=0, mp=0), damage)
        kept = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=6, mp=3), damage)
        assert spent > kept

    def test_real_damage_still_dominates(self):
        """Un seul point de degat doit battre toutes les ressources economisees."""
        state = _state(enemies=(BOARD.index_at(3, 5),), ap=9, mp=6)
        hurt = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=0, mp=0), {"e0": 1.0})
        idle = evaluate(BOARD, state, TurnState(cell=CENTRE, ap=9, mp=6), {})
        assert hurt > idle

class TestSurroundedUsesTheAreaSpell:
    """« Entoure par des ennemis il n'a pas fait bain de sang et a rate le tour. »

    Enquete : le solveur l'utilise BIEN des que deux ennemis sont colles. Le sort n'etait
    donc pas en cause -- « a rate le tour » designe une perception qui a echoue, ce que
    les notes de session confirment (« joueur non localise », « grille non detectee »).

    Ce test fixe le comportement de PLANIFICATION pour que la question ne se repose pas :
    si un jour le bot cesse d'utiliser le sort de zone en etant entoure, c'est ici que ca
    se verra, et non dans un rapport de session ou tout se confond."""

    @pytest.fixture(scope="class")
    @classmethod
    def sacrieur(cls):
        """Chargee UNE fois pour la classe : relire et reponderer vingt sorts a chaque
        test coutait plus que la recherche elle-meme sur les cas courts."""
        from pathlib import Path

        from jev_tactics.rules.spells import apply_element_focus, load_spells

        path = Path(__file__).resolve().parents[1] / "configs" / "spells" / "sacrieur.json"
        if not path.exists():
            pytest.skip("configs/spells/sacrieur.json absent")
        return apply_element_focus(load_spells(path), "terre")

    # Plateau et PA REDUITS au minimum qui preserve le comportement. Vingt sorts font
    # exploser l'arbre : mesure sur les quatre scenarios, 9,1 s a 10 PA contre 3,8 s a 7,
    # pour des plans identiques. A 6 PA en revanche le sort de zone disparait du cas a
    # quatre ennemis -- 7 est donc le plancher, pas un chiffre arrondi.
    SMALL = BoardMap(
        cells=np.array([(i, j) for j in range(5) for i in range(5)], dtype=np.int64),
        e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
        origin=np.array([600.0, 400.0]))

    def _surrounded(self, count):
        centre = self.SMALL.index_at(2, 2)
        neighbours = self.SMALL.neighbours(centre)[:count]
        entities = [Entity(entity_id="me", team=Team.ALLY, cell=centre, hp=1600,
                           hp_max=1600, ap=7, mp=3, is_self=True)]
        entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=cell, hp=400,
                            hp_max=400, ap=6, mp=3)
                     for i, cell in enumerate(neighbours)]
        return CombatState(turn=1, entities=entities)

    def test_two_adjacent_enemies_trigger_the_area_spell(self, sacrieur):
        plan = best_sequence(self.SMALL, self._surrounded(2), sacrieur)
        assert any(getattr(a, "spell", "") == "Bain de Sang" for a in plan.actions)

    def test_a_single_enemy_does_NOT(self, sacrieur):
        """Garde-fou inverse : sur une cible seule, le sort de zone coute 4 PA pour moins
        de degats qu'un sort simple a 3. L'utiliser quand meme serait le defaut oppose."""
        plan = best_sequence(self.SMALL, self._surrounded(1), sacrieur)
        assert not any(getattr(a, "spell", "") == "Bain de Sang" for a in plan.actions)

    def test_it_always_finds_something_to_cast_when_surrounded(self, sacrieur):
        """Le vrai grief -- « il a rate le tour » -- est ici impossible : entoure, le plan
        contient toujours des lancers."""
        for count in (1, 2, 3, 4):
            plan = best_sequence(self.SMALL, self._surrounded(count),
                                 sacrieur)
            assert any(isinstance(a, Cast) for a in plan.actions), f"{count} ennemis"



class TestTheSolverNeverOverspendsOnAnyScenario:
    """`test_respects_ap_budget` et son jumeau posaient la bonne question sur DEUX etats
    choisis a la main, avec UN seul sort. C'est la meme faiblesse que trois captures
    choisies pour juger d'un taux : un solveur qui les respecterait en depassant ailleurs
    passerait pour correct.

    Ces regles ne sont pas des seuils, ce sont des regles du JEU : on ne depense pas ce
    qu'on n'a pas, on ne lance pas hors de portee, un sort mono-cible ne vise pas le vide.
    Les violer ne produit aucune erreur -- le bot envoie des touches qui ne font rien et
    croit avoir joue. C'est le mode de panne le plus silencieux du combat.

    Mesure sur les scenarios de l'arene, avec le jeu de sorts complet :
    545 lancers sur 300 scenarios, zero violation des trois regles.
    """

    SCENARIOS = 200
    # Les quatre tests posent la meme sweep sous quatre angles ; la recalculer quatre fois
    # coutait 22 s au lieu de 6.
    _CACHE: ClassVar[list] = []

    def _spells(self):
        chemin = Path(__file__).resolve().parents[1] / "configs" / "spells" / "example.json"
        return {s.name: s for s in load_spells(chemin)}

    def _plans(self):
        if self._CACHE:
            return self._CACHE
        board = square_board()
        sorts = self._spells()
        for graine in range(self.SCENARIOS):
            state = random_scenario(board, random.Random(graine), enemies=2)
            self._CACHE.append(
                (board, sorts, state, best_sequence(board, state, list(sorts.values()))))
        return self._CACHE

    def test_no_plan_spends_more_than_it_has(self):
        for board, _sorts, state, plan in self._plans():
            moi = next(e for e in state.entities if e.is_self)
            pa = sum(a.cost for a in plan.actions if isinstance(a, Cast))
            pm = sum(a.cost for a in plan.actions if isinstance(a, Move))
            assert pa <= moi.ap, f"{pa} PA depenses pour {moi.ap} disponibles"
            assert pm <= moi.mp, f"{pm} PM depenses pour {moi.mp} disponibles"

    def test_no_cast_is_out_of_range_from_where_it_is_thrown(self):
        """La portee se juge depuis la case ou l'on se trouve AU MOMENT du lancer, pas
        depuis la case de depart : un plan deplace puis lance."""
        for board, sorts, state, plan in self._plans():
            case = next(e for e in state.entities if e.is_self).cell
            for action in plan.actions:
                if isinstance(action, Move):
                    case = action.cell
                elif isinstance(action, Cast):
                    assert sorts[action.spell].in_range(board, case, action.target), (
                        f"{action.spell} depuis {case} vers {action.target}")

    def test_no_single_target_spell_aims_at_an_empty_cell(self):
        """Un sort de zone peut viser le vide -- c'est meme sa raison d'etre. Un sort
        mono-cible qui le fait ne touche rien et gaspille le tour."""
        for board, sorts, state, plan in self._plans():
            occupees = {e.cell for e in state.entities}
            for action in plan.actions:
                if isinstance(action, Cast) and sorts[action.spell].area_radius == 0:
                    assert action.target in occupees, (
                        f"{action.spell} vise la case vide {action.target}")

    def test_the_sweep_actually_produces_casts(self):
        """Contre-epreuve : trois invariants sur des plans vides seraient trois tests
        vides. Mesure : 545 lancers sur 300 scenarios."""
        lancers = sum(1 for _b, _s, _e, plan in self._plans()
                      for a in plan.actions if isinstance(a, Cast))
        assert lancers >= self.SCENARIOS, f"seulement {lancers} lancers"


class TestThePerTargetCapSurvivesTheFastPath:
    """LE PLAFOND PAR CIBLE N'ETAIT TENU PAR RIEN SUR LE CHEMIN QU'EMPRUNTE LA RECHERCHE.

    `useful_actions` a deux sources de candidats. Sans table de tirs, elle part de
    `legal_actions`, qui applique deja les deux plafonds -- les re-verifier y est
    redondant. Avec table de tirs -- le chemin rapide, celui que `best_sequence` utilise --
    les candidats viennent d'une table PRECALCULEE depuis la case de depart, qui ne peut
    rien savoir des lancers faits plus tard DANS le plan. Les deux verifications de
    `prune.py` y sont alors la seule barriere.

    Celle par tour etait couverte (`test_every_planned_action_is_accepted`). Celle par
    cible ne l'etait pas : la retirer laissait toute la suite verte.

    Ce que ca coute, mesure sur la scene ci-dessous -- 12 PA, un sort a 3 PA plafonne a UN
    lancer par cible, un seul ennemi assez robuste pour survivre :

        avec le plafond   1 lancer sur la cible
        sans              4 lancers sur la MEME cible

    Le jeu refuse les trois derniers. Trois quarts du tour perdus, sans que rien ne le dise
    -- le plan s'execute « sans erreur » et le bilan compte un tour joue.

    POURQUOI MON PREMIER TEST NE L'A PAS VU. J'avais ajoute un controle du sous-ensemble
    sur des etats de tour aleatoires, en placant les lancers deja faits sur des cases
    TIREES AU HASARD. Le plafond par cible ne mord que si la case rejouee est aussi une
    cible candidate : il ne mordait donc presque jamais. Un echantillon plus large ne
    remplace pas une scene qui vise la regle.
    """

    def _scene(self):
        board = square_board(9)
        spell = Spell(name="unique", ap_cost=3, range_min=1, range_max=6,
                      needs_line_of_sight=False, damage_min=20, damage_max=20,
                      max_casts_per_turn=5, max_casts_per_target=1)
        moi = Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(0, 0),
                     hp=100, hp_max=100, ap=12, mp=0, is_self=True)
        ennemi = Entity(entity_id="e", team=Team.ENEMY, cell=board.index_at(2, 0),
                        hp=500, hp_max=500, ap=6, mp=3)
        return board, spell, CombatState(turn=1, entities=[moi, ennemi])

    def _lancers(self):
        board, spell, state = self._scene()
        return [a for a in best_sequence(board, state, [spell]).actions
                if isinstance(a, Cast)]

    def test_no_target_is_hit_more_than_its_cap(self):
        from collections import Counter
        _board, spell, _state = self._scene()
        compte = Counter((a.spell, a.target) for a in self._lancers())
        assert compte, "aucun lancer : la scene ne teste plus rien"
        assert max(compte.values()) <= spell.max_casts_per_target, (
            f"{max(compte.values())} lancers sur la meme cible pour un plafond de "
            f"{spell.max_casts_per_target} : le jeu refuserait les suivants")

    def test_the_budget_would_otherwise_allow_four(self):
        """CONTRE-EPREUVE : si les PA ne permettaient qu'un lancer, le plafond ne serait
        pour rien dans le resultat et le test du dessus serait vide. Douze PA pour un sort
        a trois : le solveur en tenterait quatre."""
        _board, spell, state = self._scene()
        moi = state.self_entity()
        assert moi.ap // spell.ap_cost >= 4
        assert spell.max_casts_per_turn >= 4, (
            "le plafond PAR TOUR doit laisser passer quatre lancers, sinon c'est LUI qui "
            "borne et non celui par cible")

    def test_the_cap_is_per_target_and_not_per_turn_in_disguise(self):
        """CE QUE LA SCENE A UNE SEULE CIBLE NE PEUT PAS VOIR. Avec un ennemi unique, un
        plafond « une fois par cible » et un plafond « une fois par tour » produisent le
        MEME plan -- un lancer. Confondre les deux dans le code passait donc inapercu.

        Deux ennemis, un sort a deux lancers par tour mais un seul par cible : le bon
        comportement en tire DEUX, un sur chacun. Confondu avec le plafond par tour, il
        s'arreterait a un."""
        from collections import Counter
        board = square_board(9)
        spell = Spell(name="unique", ap_cost=3, range_min=1, range_max=6,
                      needs_line_of_sight=False, damage_min=20, damage_max=20,
                      max_casts_per_turn=2, max_casts_per_target=1)
        moi = Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(0, 0),
                     hp=100, hp_max=100, ap=12, mp=0, is_self=True)
        etat = CombatState(turn=1, entities=[
            moi,
            Entity(entity_id="a", team=Team.ENEMY, cell=board.index_at(2, 0),
                   hp=500, hp_max=500, ap=6, mp=3),
            Entity(entity_id="b", team=Team.ENEMY, cell=board.index_at(0, 2),
                   hp=500, hp_max=500, ap=6, mp=3)])
        lancers = [a for a in best_sequence(board, etat, [spell]).actions
                   if isinstance(a, Cast)]
        compte = Counter(a.target for a in lancers)
        assert len(lancers) == 2, (
            f"{len(lancers)} lancer(s) : le plafond par cible doit laisser frapper les "
            f"DEUX ennemis, celui par tour en autorise deux")
        assert max(compte.values()) == 1

    def test_the_fast_path_is_the_one_being_exercised(self):
        """CE QUI FAIT LA VALEUR DU TEST : `best_sequence` construit une table de tirs, donc
        il passe par la branche ou `legal_actions` n'intervient plus. Le verifier evite de
        croire couvrir le chemin rapide en couvrant l'autre."""
        board, spell, state = self._scene()
        depart = state.self_entity().cell
        table = build_shot_table(board, state, [spell], [depart])
        assert table.get(depart), (
            "aucun tir precalcule depuis la case de depart : la scene emprunterait "
            "l'autre branche, celle ou `legal_actions` applique deja les plafonds")


class TestTheSolverNeverFiresAtItsOwnSide:
    """« TOUCHER QUELQU'UN » AVAIT DEUX DEFINITIONS, ET C'EST LA MAUVAISE QUI SERVAIT.

    `build_shot_table` -- dont le docstring annonce « les lancers qui touchent un
    ENNEMI » -- acceptait un lancer des qu'il tombait sur un combattant quelconque, allie
    compris. `_enabled_shots`, l'autre source de la meme information, ne compte que les
    ennemis. Et c'est la table qui sert au vrai solveur : `best_sequence` la construit.

    LE PLAN QUE CELA PRODUISAIT, sur la scene ci-dessous -- ennemi en case 3, allie colle
    en case 4, sort a 30 degats plafonne a un lancer par cible, 9 PA :

        mono -> case 3   touche un ennemi
        mono -> case 4   NE TOUCHE PERSONNE D'AUTRE QUE L'ALLIE
        zone -> case 2   touche un ennemi

    Le second lancer est choisi parce que le plafond par cible interdit de recommencer sur
    l'ennemi, que la case de l'allie est une cible DISTINCTE offerte par la table, et que
    `evaluate` ignore les allies : ce tir ne coute rien au score et evite 0,3 de penalite
    de PA non depense. Trois PA perdus et trente degats sur son propre camp, pour 0,3.

    POURQUOI LES DEUX SCENES A DEUX COMBATTANTS NE LE VOYAIENT PAS. Sans allie a portee,
    les deux definitions coincident : tout ce qui est occupe est ennemi. Il fallait la
    conjonction d'un allie ADJACENT a un ennemi -- sinon sa case n'est meme pas une cible
    candidate -- et d'un plafond qui force le solveur a chercher ailleurs.
    """

    def _scene(self):
        board = square_board(11)
        zone = Spell(name="zone", ap_cost=3, range_min=1, range_max=5,
                     needs_line_of_sight=False, damage_min=10, damage_max=10,
                     area_radius=1, max_casts_per_turn=1, max_casts_per_target=1)
        mono = Spell(name="mono", ap_cost=3, range_min=1, range_max=5,
                     needs_line_of_sight=False, damage_min=30, damage_max=30,
                     max_casts_per_turn=9, max_casts_per_target=1)
        moi = Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(0, 0),
                     hp=100, hp_max=100, ap=9, mp=0, is_self=True)
        ennemi = Entity(entity_id="e", team=Team.ENEMY, cell=board.index_at(3, 0),
                        hp=500, hp_max=500, ap=6, mp=3)
        ami = Entity(entity_id="ami", team=Team.ALLY, cell=board.index_at(4, 0),
                     hp=100, hp_max=100, ap=6, mp=3)
        return board, [mono, zone], CombatState(turn=1, entities=[moi, ennemi, ami])

    def test_every_cast_in_the_plan_reaches_an_enemy(self):
        board, spells, state = self._scene()
        par_nom = {s.name: s for s in spells}
        ennemis = {e.cell for e in state.entities if e.team is Team.ENEMY}
        lancers = [a for a in best_sequence(board, state, spells).actions
                   if isinstance(a, Cast)]
        assert lancers, "aucun lancer : la scene ne teste plus rien"
        steriles = [a for a in lancers
                    if not set(par_nom[a.spell].area_cells(board, a.target)) & ennemis]
        assert not steriles, (
            f"{steriles} : lancer(s) ne touchant aucun ennemi. Sur cette scene, la cible "
            f"est la case d'un allie")

    def test_the_shot_table_offers_only_enemy_reaching_casts(self):
        """LA CAUSE, et non son effet. Un test qui ne regarde que le plan resterait vert
        si le solveur cessait d'ELIRE ce lancer pour une autre raison -- une ponderation
        modifiee, par exemple -- alors que la table continuerait de le proposer."""
        board, spells, state = self._scene()
        par_nom = {s.name: s for s in spells}
        ennemis = {e.cell for e in state.entities if e.team is Team.ENEMY}
        depart = state.self_entity().cell
        table = build_shot_table(board, state, spells, [depart])
        offerts = table.get(depart, [])
        assert offerts, "table vide : le test ne porte sur rien"
        steriles = [c for c in offerts
                    if not set(par_nom[c.spell].area_cells(board, c.target)) & ennemis]
        assert not steriles, f"la table propose {steriles}, qui ne touchent aucun ennemi"

    def test_the_other_source_of_the_same_notion_agrees(self):
        """L'AUTRE MOITIE DE LA PAIRE. `_enabled_shots` repond a la meme question que la
        table, pour le chemin sans precalcul et pour la signature des deplacements. Ne
        tester que la table laisserait la seconde definition deriver librement -- et c'est
        exactement la situation qu'on vient de corriger, dans l'autre sens.

        Les deux doivent proposer LE MEME ensemble depuis la case de depart."""
        from jev_tactics.planner.prune import _enabled_shots
        board, spells, state = self._scene()
        ennemis = {e.cell for e in state.entities if e.team is Team.ENEMY}
        depart = state.self_entity().cell
        directe = _enabled_shots(board, state, spells, depart, ennemis)
        par_table = frozenset((c.spell, c.target)
                              for c in build_shot_table(board, state, spells,
                                                        [depart]).get(depart, []))
        assert directe == par_table, (
            f"les deux sources divergent : {sorted(directe ^ par_table)}")
        par_nom = {s.name: s for s in spells}
        assert all(set(par_nom[nom].area_cells(board, cible)) & ennemis
                   for nom, cible in directe), (
            "la source directe propose un lancer qui ne touche aucun ennemi")

    def test_the_ally_cell_is_a_candidate_target_at_all(self):
        """CONTRE-EPREUVE : si la case de l'allie n'etait de toute facon pas une cible
        envisageable, les deux tests ci-dessus seraient verts sans rien garder. Elle l'est
        parce qu'elle est ADJACENTE a l'ennemi, donc dans le rayon du sort de zone."""
        from jev_tactics.planner.prune import useful_targets
        board, spells, state = self._scene()
        ami = next(e for e in state.entities
                   if e.team is Team.ALLY and not e.is_self)
        ennemis = {e.cell for e in state.entities if e.team is Team.ENEMY}
        assert ami.cell in useful_targets(board, spells, ennemis), (
            "la case de l'allie n'est pas une cible candidate : deplacer l'allie plus "
            "pres de l'ennemi pour que cette classe teste ce qu'elle annonce")


class TestFriendlyFireIsNoLongerFree:
    """LES DEGATS SUR SON PROPRE CAMP VALAIENT EXACTEMENT ZERO.

    `evaluate` sautait tout combattant non ennemi, et `_apply_damage` ne les enregistrait
    meme pas : l'information etait jetee en amont. Mesure de l'aveuglement, sur un sort de
    zone a 40 degats -- score 47,2 en touchant l'allie, 47,2 sans. Entre deux cibles a
    egalite, l'une epargnant l'allie et l'autre non, le solveur tranchait au hasard de
    l'ordre d'enumeration.

    CE QUE CE TERME EST, ET CE QU'IL N'EST PAS. Un DEPARTAGE, borne par construction a
    `FRIENDLY_FIRE_PENALTY` par combattant touche -- un seul allie coute donc au plus 0,5,
    sous un point de degat reel. Le signe n'est pas discutable ; la valeur, elle, n'est pas
    mesurable ici : l'arene ne place AUCUN allie, donc ce terme est inerte dans toutes les
    mesures deja faites -- ce qui rend son ajout sans risque pour l'acquis, et empeche de
    le regler. Modeliser des allies dans l'arene est le prealable.
    """

    def _scene(self):
        board = square_board(11)
        moi = Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(0, 0),
                     hp=100, hp_max=100, ap=6, mp=0, is_self=True)
        ennemi = Entity(entity_id="e1", team=Team.ENEMY, cell=board.index_at(3, 0),
                        hp=500, hp_max=500, ap=6, mp=3)
        ami = Entity(entity_id="ami", team=Team.ALLY, cell=board.index_at(4, 0),
                     hp=100, hp_max=100, ap=6, mp=3)
        etat = CombatState(turn=1, entities=[moi, ennemi, ami])
        return board, etat, TurnState(cell=moi.cell, ap=3, mp=0, casts=(("zone", 2),))

    def test_hitting_an_ally_costs_something(self):
        board, etat, turn = self._scene()
        epargne = evaluate(board, etat, turn, {"e1": 40.0})
        touche = evaluate(board, etat, turn, {"e1": 40.0, "ami": 40.0})
        assert touche < epargne, (
            f"{touche} contre {epargne} : blesser son allie ne coute toujours rien")

    def test_it_stays_a_tie_breaker(self):
        """CE QUI BORNE LE TERME. Un point de degat reel doit rester plus fort qu'un allie
        entierement pris dans la zone -- sinon le solveur renoncerait a frapper pour
        proteger, ce qu'aucune mesure ne justifie ici."""
        board, etat, turn = self._scene()
        pire = evaluate(board, etat, turn, {"e1": 40.0, "ami": 999.0})
        un_degat_de_plus = evaluate(board, etat, turn, {"e1": 41.0})
        assert un_degat_de_plus > evaluate(board, etat, turn, {"e1": 40.0}) > pire
        assert (evaluate(board, etat, turn, {"e1": 40.0}) - pire) < 1.0

    def test_the_damage_actually_reaches_the_scorer(self):
        """LE CABLAGE, et c'etait le vrai trou : `_apply_damage` ne repartissait les degats
        que sur les ENNEMIS. Le terme de score aurait pu etre juste et rester lettre morte,
        faute de recevoir la moindre valeur pour un allie."""
        from jev_tactics.planner.search import _apply_damage
        board, etat, _turn = self._scene()
        ami = next(e for e in etat.entities if e.entity_id == "ami")
        spell = Spell(name="zone", ap_cost=3, range_min=1, range_max=6,
                      needs_line_of_sight=False, damage_min=40, damage_max=40,
                      area_radius=1, max_casts_per_turn=9, max_casts_per_target=9)
        subis = _apply_damage(board, etat, spell, ami.cell, {})
        assert subis.get("ami", 0.0) > 0, (
            "les degats infliges a l'allie ne sont pas enregistres : le terme de score "
            "ne recevra jamais rien")

    def test_the_solver_prefers_the_target_that_spares_the_ally(self):
        """L'EFFET, bout en bout. Deux cases touchent le meme ennemi ; une seule epargne
        l'allie. Sans le terme, le choix etait arbitraire."""
        board, etat, _turn = self._scene()
        spell = Spell(name="zone", ap_cost=3, range_min=1, range_max=6,
                      needs_line_of_sight=False, damage_min=40, damage_max=40,
                      area_radius=1, max_casts_per_turn=1, max_casts_per_target=9)
        ami = next(e for e in etat.entities if e.entity_id == "ami")
        for action in best_sequence(board, etat, [spell]).actions:
            if isinstance(action, Cast):
                assert ami.cell not in set(spell.area_cells(board, action.target)), (
                    f"la case {action.target} eclabousse l'allie alors qu'une autre "
                    f"touche le meme ennemi sans le faire")
