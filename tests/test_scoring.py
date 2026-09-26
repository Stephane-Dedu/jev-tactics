"""Fonction d'evaluation : ce que le solveur cherche a maximiser.

Les ponderations sont le seul endroit ou l'on encode « ce qu'est un bon tour ». Chacune
doit donc porter sa mesure -- l'arene la donne, ces tests en fixent la consequence."""

from pathlib import Path

import numpy as np

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner import scoring
from jev_tactics.planner.legal import TurnState
from jev_tactics.planner.scoring import evaluate
from jev_tactics.rules.movement import grid_distance
from jev_tactics.state import CombatState, Entity, Team


class TestTheKillBonusIsOnAPlateau:
    """Le terme etait justifie, sa valeur ne l'etait pas. Balayage en arene, 60 combats par
    graine, 3 graines :

        KILL_BONUS      0     15     30     60    120    300
        victoires     0,6%  26,1%  77,8%  81,1%  81,1%  81,1%

    Sans lui, le bot gagne 0,6 % des combats : c'est de loin le terme le plus important de
    la fonction d'evaluation. Un solveur qui maximise les degats BRUTS repartit ses coups
    et ne tue personne ; les ennemis frappent alors tous, tous les tours.

    A partir de 60 les resultats sont IDENTIQUES graine par graine -- pas des moyennes
    voisines, les memes plans. 60 est le debut d'un plateau, pas un point regle.

    Ces tests-ci ne rejouent pas l'arene (35 s) : ils fixent la CONSEQUENCE mesurable en
    quelques millisecondes -- achever vaut mieux qu'eparpiller.
    """

    def _board_and_state(self):
        board = BoardMap(
            cells=np.array([(i, j) for j in range(7) for i in range(7)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([600.0, 400.0]))
        state = CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(3, 3),
                   hp=100, hp_max=100, ap=6, mp=3, is_self=True),
            Entity(entity_id="a", team=Team.ENEMY, cell=board.index_at(3, 1),
                   hp=20, hp_max=100, ap=6, mp=3),
            Entity(entity_id="b", team=Team.ENEMY, cell=board.index_at(1, 3),
                   hp=20, hp_max=100, ap=6, mp=3),
        ])
        return board, state

    def _turn(self, board, state):
        return TurnState(cell=state.find_self().cell, ap=0, mp=0)

    def test_finishing_one_beats_splitting_the_same_damage(self):
        board, state = self._board_and_state()
        turn = self._turn(board, state)
        acheve = evaluate(board, state, turn, {"a": 20.0})
        eparpille = evaluate(board, state, turn, {"a": 10.0, "b": 10.0})
        assert acheve > eparpille

    def test_the_preference_survives_at_the_measured_plateau_floor(self, monkeypatch):
        """A 30 l'arene donne deja 77,8 % : la preference doit tenir la."""
        monkeypatch.setattr(scoring, "KILL_BONUS", 30.0)
        board, state = self._board_and_state()
        turn = self._turn(board, state)
        assert (evaluate(board, state, turn, {"a": 20.0})
                > evaluate(board, state, turn, {"a": 10.0, "b": 10.0}))

    def test_without_the_bonus_the_preference_disappears(self, monkeypatch):
        """A zero, l'arene tombe a 0,6 %. La cause se voit ici : plus rien ne distingue
        achever d'eparpiller, puisque les degats utiles sont les memes."""
        monkeypatch.setattr(scoring, "KILL_BONUS", 0.0)
        board, state = self._board_and_state()
        turn = self._turn(board, state)
        assert (evaluate(board, state, turn, {"a": 20.0})
                <= evaluate(board, state, turn, {"a": 10.0, "b": 10.0}))


class TestThePrudenceWeightsSitBelowACliff:
    """Les deux poids de prudence ont une FALAISE, et c'est ce que les balayages apprennent.

        SAFETY_WEIGHT     0,0    0,5    1,5    3,0    6,0
        victoires        40,0%  72,8%  81,1%  81,1%   0,0%

        SURVIVAL_WEIGHT   0,0    1,5    3,0    6,0    10,0
        victoires        72,8%  80,0%  80,8%  81,7%  39,2%

    A 6,0 de prudence le bot ne gagne plus AUCUN combat : l'eloignement rapporte alors plus
    qu'un point de degat, donc fuir domine frapper. Le terme n'est pas « une prudence qu'on
    peut monter si l'on veut » -- au-dela d'un seuil il change la nature du joueur.

    Ces tests fixent la CAUSE en quelques millisecondes, la ou l'arene demande des minutes.
    """

    def _setup(self, hp=100):
        board = BoardMap(
            cells=np.array([(i, j) for j in range(9) for i in range(9)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([600.0, 400.0]))
        state = CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(4, 4),
                   hp=hp, hp_max=100, ap=6, mp=3, is_self=True),
            Entity(entity_id="a", team=Team.ENEMY, cell=board.index_at(4, 1),
                   hp=100, hp_max=100, ap=6, mp=3),
        ])
        return board, state

    def _frapper_vs_fuir(self, board, state):
        proche = TurnState(cell=board.index_at(4, 3), ap=0, mp=0)
        loin = TurnState(cell=board.index_at(4, 8), ap=0, mp=0)
        return (evaluate(board, state, proche, {"a": 20.0}),
                evaluate(board, state, loin, {}))

    def test_at_the_current_weight_striking_beats_fleeing(self):
        board, state = self._setup()
        frapper, fuir = self._frapper_vs_fuir(board, state)
        assert frapper > fuir

    def test_at_the_cliff_fleeing_wins_and_that_is_the_zero_percent(self, monkeypatch):
        """La cause du 0 % mesure en arene : a 6,0, s'eloigner sans rien faire rapporte
        plus que s'approcher et frapper."""
        monkeypatch.setattr(scoring, "SAFETY_WEIGHT", 6.0)
        board, state = self._setup()
        frapper, fuir = self._frapper_vs_fuir(board, state)
        assert fuir > frapper

    def test_the_survival_term_only_bites_when_hurt(self):
        """C'est tout le principe : a pleine sante l'eloignement compte peu, moribond il
        compte beaucoup. Sans cela il faudrait un terme de plus, et un poids de plus."""
        intact = scoring.danger_factor(self._setup(hp=100)[1])
        agonie = scoring.danger_factor(self._setup(hp=1)[1])
        assert intact == 1.0
        assert agonie > intact

    def test_unknown_health_stays_neutral(self):
        """PV illisibles : facteur neutre. Supposer le pire rendrait le bot craintif sur
        une simple panne de lecture -- et un bot qui fuit sans raison ressemble a un bot
        prudent."""
        _board, state = self._setup()
        state.find_self().hp_max = 0
        assert scoring.danger_factor(state) == 1.0


class TestPrudencePaysMostWhenSurrounded:
    """Le cas dont l'utilisateur se plaint -- « entoure d'ennemis » -- est celui ou la
    prudence compte le plus, et il n'avait jamais ete mesure : tous les balayages
    portaient sur DEUX ennemis.

        2 ennemis (6 graines)    1,5 -> 83,3%   2,5 -> 86,2%   3,0 -> 80,0%
        3 ennemis, 200 PV        1,5 -> 60,0%   2,5 -> 70,0%   3,0 -> 70,8%

    L'ecart entre 1,5 et 2,5 passe de 3 points a DIX quand on est encercle.

    A 3 ennemis, l'arene STANDARD est ingagnable -- 2,5 % quelle que soit la prudence,
    parce que trois adversaires infligent ~162 degats par tour a un joueur de 100 PV. Ce
    plafond est arithmetique, pas tactique : c'est l'instrument qui sature, pas le solveur
    qui echoue. Il fallait le savoir avant de lire « 2,5 % » comme un resultat.
    """

    def _state(self, enemies, board):
        entities = [Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(4, 4),
                           hp=100, hp_max=100, ap=6, mp=3, is_self=True)]
        entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY,
                            cell=board.index_at(4 - i, 2), hp=60, hp_max=60, ap=6, mp=3)
                     for i in range(enemies)]
        return CombatState(turn=1, entities=entities)

    def _board(self):
        return BoardMap(
            cells=np.array([(i, j) for j in range(9) for i in range(9)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([600.0, 400.0]))

    def test_retreating_is_worth_more_against_more_enemies(self):
        """La raison pour laquelle le meme poids ne convient pas aux deux regimes : chaque
        ennemi vivant ajoute son propre terme d'eloignement, donc la prudence pese
        mecaniquement plus lourd quand ils sont nombreux."""
        board = self._board()
        loin = TurnState(cell=board.index_at(4, 7), ap=0, mp=0)
        proche = TurnState(cell=board.index_at(4, 3), ap=0, mp=0)
        gains = []
        for n in (1, 3):
            state = self._state(n, board)
            gains.append(evaluate(board, state, loin, {})
                         - evaluate(board, state, proche, {}))
        assert gains[1] > gains[0]

    def test_the_chosen_weight_still_prefers_striking(self):
        """Le garde-fou du changement : monter la prudence de 1,5 a 2,5 ne doit pas faire
        basculer le bot du cote de la fuite, meme encercle. La falaise mesuree est a 4,0 a
        trois ennemis ; 2,5 en reste a 1,6x."""
        board = self._board()
        state = self._state(3, board)
        proche = TurnState(cell=board.index_at(4, 3), ap=0, mp=0)
        loin = TurnState(cell=board.index_at(4, 7), ap=0, mp=0)
        assert (evaluate(board, state, proche, {"e0": 60.0})
                > evaluate(board, state, loin, {}))


class TestEveryWeightCarriesTwoRegimes:
    """Garde-fou de methode, pas de comportement.

    Deux fois dans cette serie, un classement mesure a deux ennemis s'est INVERSE a trois :
    SURVIVAL_WEIGHT=6,0 passait de +0,9 point a -50,8, et le « plateau » de KILL_BONUS
    s'est revele un pic. Une mesure faite dans un seul regime ne dit rien des autres.

    Ce test verifie que chaque poids porte bien SES DEUX regimes dans le fichier. Il
    echouera si quelqu'un ajoute ou remplace un poids sans le mesurer encercle -- ce qui
    est exactement le moment ou il faut y penser.
    """

    def test_each_weight_documents_a_three_enemy_measurement(self):
        source = Path(scoring.__file__).read_text(encoding="utf-8")
        blocs = source.split("\n\n")
        for nom in ("KILL_BONUS", "SAFETY_WEIGHT", "SURVIVAL_WEIGHT", "WASTED_AP_PENALTY"):
            bloc = next(b for b in blocs if f"\n{nom} = " in "\n" + b)
            assert "3 ennemis" in bloc or "trois ennemis" in bloc.lower(), nom


class TestTheWoundedBonusWorksWithoutAbsoluteHealth:
    """En jeu reel, AUCUN ennemi n'a de PV connus -- 72 sur 32 captures, zero `hp_known`.
    Sans eux, `evaluate` ne peut ni plafonner les degats ni accorder KILL_BONUS, et le
    solveur n'a plus aucune raison d'achever un blesse : il repartit ses coups.

        PV ennemis connus       85,8 %
        PV ennemis inconnus     35,0 %

    Ce terme rattrape la majeure partie de l'ecart avec la seule ENTAME, qui ne demande pas
    les PV absolus :

        poids       0     x2     x6    x10    x20
        victoires  35,0%  66,2%  69,6%  75,4%  75,4%

    x10 et x20 donnent des resultats IDENTIQUES graine par graine : plateau, pas point
    regle.
    """

    def _board(self):
        return BoardMap(
            cells=np.array([(i, j) for j in range(9) for i in range(9)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([600.0, 400.0]))

    def _state(self, board, entame):
        return CombatState(turn=1, entities=[
            Entity(entity_id="me", team=Team.ALLY, cell=board.index_at(4, 4),
                   hp=100, hp_max=100, ap=6, mp=3, is_self=True),
            Entity(entity_id="a", team=Team.ENEMY, cell=board.index_at(4, 2),
                   hp=int(100 * (1 - entame)), hp_max=100, ap=6, mp=3, hp_known=False),
        ])

    def test_a_wounded_target_is_worth_more(self):
        board = self._board()
        turn = TurnState(cell=board.index_at(4, 4), ap=0, mp=0)
        intact = evaluate(board, self._state(board, 0.0), turn, {"a": 20.0})
        entame = evaluate(board, self._state(board, 0.8), turn, {"a": 20.0})
        assert entame > intact

    def test_it_is_inert_while_health_is_a_placeholder(self):
        """Le garde-fou qui rend ce terme livrable AUJOURD'HUI : tant que PV et PV max
        valent le meme remplissage, l'entame est nulle et le score ne bouge pas. En jeu,
        l'entame n'est encore attribuable a AUCUN ennemi precis -- la timeline donne un
        ratio par portrait, mais rien ne relie un portrait a une case."""
        board = self._board()
        turn = TurnState(cell=board.index_at(4, 4), ap=0, mp=0)
        state = self._state(board, 0.0)
        for e in state.entities:
            if e.team is Team.ENEMY:
                e.hp = e.hp_max = 1000
        sans_terme = 20.0 + scoring.SAFETY_WEIGHT * grid_distance(
            board, turn.cell, state.entities[1].cell)
        assert abs(evaluate(board, state, turn, {"a": 20.0}) - sans_terme) < 1e-6
