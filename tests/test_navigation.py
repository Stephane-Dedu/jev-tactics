"""Navigation : circuits de recolte et trajets vers une carte precise.

Deux outils pour deux besoins : le circuit ne demande AUCUNE connaissance de la position
(donc rien ne peut le faire derailler), le graphe en exige une mais permet de rejoindre
un point precis."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.world.navigation import (
    Direction,
    MapGraph,
    Route,
    direction_of,
    map_changed,
)

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "configs" / "routes" / "example.json"


class TestDirection:
    def test_deltas_follow_dofus_convention(self):
        """Droite = x+1, bas = y+1."""
        assert Direction.RIGHT.delta == (1, 0)
        assert Direction.BOTTOM.delta == (0, 1)

    def test_opposites_cancel_out(self):
        for direction in Direction:
            dx, dy = direction.delta
            ox, oy = direction.opposite.delta
            assert (dx + ox, dy + oy) == (0, 0)


# Zone 3x3 avec un trou en (1, 1) : oblige a contourner.
DONUT = MapGraph(walkable={(x, y) for x in range(3) for y in range(3)} - {(1, 1)})


class TestMapGraph:
    def test_neighbours_are_adjacent_and_walkable(self):
        found = dict(DONUT.neighbours((0, 0)))
        assert set(found) == {Direction.RIGHT, Direction.BOTTOM}
        assert found[Direction.RIGHT] == (1, 0)

    def test_hole_is_not_a_neighbour(self):
        assert (1, 1) not in dict(DONUT.neighbours((1, 0))).values()

    def test_direct_route(self):
        assert DONUT.route((0, 0), (2, 0)) == [Direction.RIGHT, Direction.RIGHT]

    def test_route_goes_around_the_hole(self):
        steps = DONUT.route((1, 0), (1, 2))
        assert len(steps) == 4          # 2 en ligne droite serait le trou

    def test_route_length_is_minimal(self):
        assert len(DONUT.route((0, 0), (2, 2))) == 4

    def test_empty_route_to_itself(self):
        assert DONUT.route((0, 0), (0, 0)) == []

    def test_none_when_unreachable(self):
        island = MapGraph(walkable={(0, 0), (5, 5)})
        assert island.route((0, 0), (5, 5)) is None

    def test_none_when_outside_the_zone(self):
        assert DONUT.route((0, 0), (9, 9)) is None

    def test_route_actually_walks_to_the_goal(self):
        """Verification structurelle : suivre les directions doit mener au but."""
        start, goal = (0, 0), (2, 2)
        position = start
        for step in DONUT.route(start, goal):
            dx, dy = step.delta
            position = (position[0] + dx, position[1] + dy)
            assert position in DONUT
        assert position == goal

    def test_loads_from_json(self, tmp_path):
        path = tmp_path / "zone.json"
        path.write_text(json.dumps([[0, 0], [1, 0]]), encoding="utf-8")
        assert MapGraph.from_json(path).walkable == {(0, 0), (1, 0)}


class TestRoute:
    def test_advances_through_the_circuit(self):
        route = Route(steps=[Direction.RIGHT, Direction.BOTTOM])
        assert [route.advance(), route.advance()] == [Direction.RIGHT, Direction.BOTTOM]

    def test_loops_back_and_counts_laps(self):
        route = Route(steps=[Direction.RIGHT, Direction.LEFT])
        route.advance()
        route.advance()
        assert route.laps == 1 and route.advance() is Direction.RIGHT

    def test_peek_does_not_consume(self):
        route = Route(steps=[Direction.TOP])
        assert route.peek() is Direction.TOP and route.index == 0

    def test_rewind_after_a_failed_move(self):
        """Le changement de carte n'a pas eu lieu : sans ce retour en arriere, tout le
        circuit se decalerait definitivement."""
        route = Route(steps=[Direction.RIGHT, Direction.BOTTOM])
        route.advance()
        route.rewind()
        assert route.peek() is Direction.RIGHT

    def test_rewind_wraps_backwards(self):
        route = Route(steps=[Direction.RIGHT, Direction.BOTTOM])
        route.advance()
        route.advance()                       # boucle -> index 0, laps 1
        route.rewind()
        assert route.peek() is Direction.BOTTOM and route.laps == 0

    def test_empty_route_is_harmless(self):
        route = Route(steps=[])
        assert route.advance() is None and route.peek() is None
        route.rewind()                        # ne doit pas lever

    def test_loads_from_json(self, tmp_path):
        path = tmp_path / "circuit.json"
        path.write_text(json.dumps(["right", "bottom"]), encoding="utf-8")
        assert Route.from_json(path).steps == [Direction.RIGHT, Direction.BOTTOM]


class TestDetour:
    """Rattraper une derive sans perdre sa place dans le circuit.

    Le circuit ne connait que des directions RELATIVES : arrive sur la mauvaise carte, il
    continue comme si de rien n'etait et le decalage devient permanent. Les coordonnees
    rendent la derive visible ; le detour la rattrape."""

    def _route(self):
        return Route(steps=[Direction.RIGHT, Direction.BOTTOM])

    def test_the_detour_is_played_first(self):
        route = self._route()
        route.divert(Direction.TOP)
        assert route.advance() is Direction.TOP

    def test_the_circuit_resumes_exactly_where_it_was(self):
        route = self._route()
        route.divert(Direction.TOP)
        route.advance()
        assert route.advance() is Direction.RIGHT
        assert route.index == 1 and route.laps == 0

    def test_a_detour_does_not_count_a_lap(self):
        """Un tour de circuit sert a mesurer le rendement d'une zone : compter des pas
        subis y melerait du bruit."""
        route = self._route()
        route.divert(Direction.TOP)
        for _ in range(3):
            route.advance()
        assert route.laps == 1        # 2 pas de circuit joues, pas 3

    def test_several_steps_keep_their_order(self):
        """Un saut se rattrape par un TRAJET, pas par un pas. `appendleft` empilant,
        l'insertion se fait a l'envers -- inverser l'ordre ferait parcourir le trajet a
        rebours, ce qui menerait n'importe ou."""
        route = self._route()
        route.divert(Direction.LEFT, Direction.TOP, Direction.LEFT)
        assert [route.advance() for _ in range(3)] == [
            Direction.LEFT, Direction.TOP, Direction.LEFT]

    def test_the_circuit_follows_a_multi_step_detour(self):
        route = self._route()
        route.divert(Direction.LEFT, Direction.TOP)
        for _ in range(2):
            route.advance()
        assert route.advance() is Direction.RIGHT and route.index == 1

    def test_peek_shows_the_detour(self):
        route = self._route()
        route.divert(Direction.TOP)
        assert route.peek() is Direction.TOP

    def test_a_failed_detour_is_not_lost(self):
        """Rembobiner apres un detour rate doit le remettre a faire. Le perdre rendrait
        justement permanent le decalage qu'il devait corriger."""
        route = self._route()
        route.divert(Direction.TOP)
        route.advance()
        route.rewind()
        assert route.peek() is Direction.TOP
        assert route.index == 0       # le circuit n'a pas recule

    def test_correcting_a_failed_detour_keeps_the_order(self):
        """Ordre voulu : nouvelle correction, PUIS le pas encore a refaire. L'inverse
        rejouerait le detour rate avant de l'avoir corrige."""
        route = self._route()
        route.divert(Direction.TOP)
        route.advance()
        route.rewind()
        route.divert(Direction.LEFT)
        assert [route.advance(), route.advance()] == [Direction.LEFT, Direction.TOP]


class TestDirectionOf:
    def test_names_a_single_step(self):
        assert direction_of((1, 0)) is Direction.RIGHT
        assert direction_of((0, -1)) is Direction.TOP

    def test_standing_still_has_no_direction(self):
        assert direction_of((0, 0)) is None

    def test_a_jump_has_no_direction(self):
        """Zaap ou rappel : aucun pas unitaire ne le decrit, et en inventer un ferait
        prendre au bot une correction pour un retour a la normale."""
        assert direction_of((-28, 32)) is None

    def test_a_diagonal_has_no_direction(self):
        assert direction_of((1, 1)) is None


class TestCircuitShape:
    """Un circuit de recolte doit REVENIR. Rien ne le verifiait.

    Ecrire une route a la main avec un pas en trop est l'erreur la plus facile a commettre
    -- et la plus difficile a constater : le bot avance d'une carte par tour, quitte sa
    zone en quelques minutes, ne trouve plus rien, et le bilan continue d'annoncer
    « N cartes parcourues » comme si tout allait bien."""

    def test_a_closed_circuit_returns_to_its_start(self):
        route = Route(steps=[Direction.RIGHT, Direction.RIGHT, Direction.BOTTOM,
                             Direction.LEFT, Direction.LEFT, Direction.TOP])
        assert route.is_closed and route.net_displacement() == (0, 0)

    def test_a_missing_step_makes_it_drift(self):
        """Un « left » retire au circuit precedent."""
        route = Route(steps=[Direction.RIGHT, Direction.RIGHT, Direction.BOTTOM,
                             Direction.LEFT, Direction.TOP])
        assert not route.is_closed and route.net_displacement() == (1, 0)

    def test_a_two_map_ping_pong_is_closed(self):
        """Aller-retour : degenere, mais parfaitement valide -- on recolte deux cartes en
        alternance. Le signaler serait crier a tort."""
        assert Route(steps=[Direction.RIGHT, Direction.LEFT]).is_closed

    def test_it_counts_the_maps_not_the_steps(self):
        """Un circuit qui repasse quelque part visite moins de cartes que de pas, et
        c'est ce nombre-la qui dit la taille reelle de la zone farmee."""
        route = Route(steps=[Direction.RIGHT, Direction.LEFT,
                             Direction.RIGHT, Direction.LEFT])
        assert "2 carte(s)" in route.describe_shape()

    def test_the_drift_is_quantified_not_just_flagged(self):
        """Un verdict sans mesure n'aide pas a corriger : il faut savoir de combien, et
        dans quel sens, pour retrouver le pas manquant."""
        route = Route(steps=[Direction.RIGHT, Direction.RIGHT, Direction.TOP])
        assert "(+2, -1)" in route.describe_shape()

    def test_an_empty_circuit_says_so(self):
        assert Route(steps=[]).describe_shape() == "circuit vide"

    def test_the_shipped_example_is_closed(self):
        """Le circuit livre sert de modele : le laisser derailler apprendrait l'erreur."""
        path = ROOT / "configs" / "routes" / "example.json"
        if not path.exists():
            pytest.skip("circuit d'exemple absent")
        assert Route.from_json(path).is_closed


class TestObservedTransitions:
    """Les faits l'emportent sur l'adjacence supposee.

    Le graphe ne demandait jusqu'ici qu'une liste ecrite a la main que personne n'avait,
    ce qui le rendait inutilisable. Il se remplit desormais en jouant -- et ce qu'il
    apprend est de meilleure qualite que ce qu'il supposait : la detection de derive a
    montre qu'un clic de bord fait MARCHER le personnage, et qu'un obstacle peut le faire
    sortir par un autre cote."""

    def test_observing_makes_both_maps_walkable(self):
        graph = MapGraph()
        graph.observe((3, 8), Direction.RIGHT, (4, 8))
        assert (3, 8) in graph and (4, 8) in graph

    def test_an_observed_link_beats_adjacency(self):
        """La sortie droite de (3,8) mene en (3,7), pas en (4,8). C'est constate."""
        graph = MapGraph(walkable={(3, 8), (4, 8), (3, 7)})
        graph.observe((3, 8), Direction.RIGHT, (3, 7))
        assert dict(graph.neighbours((3, 8)))[Direction.RIGHT] == (3, 7)

    def test_adjacency_still_fills_the_gaps(self):
        """Une direction jamais essayee reste une supposition utilisable : sans elle le
        graphe ne servirait a rien avant d'avoir tout parcouru."""
        graph = MapGraph(walkable={(0, 0), (1, 0)})
        assert dict(graph.neighbours((0, 0)))[Direction.RIGHT] == (1, 0)

    def test_a_newer_observation_wins(self):
        """Le monde change : obstacle retire, contournement different. Garder la premiere
        valeur a jamais serait s'attacher a un fait perime."""
        graph = MapGraph()
        graph.observe((0, 0), Direction.RIGHT, (0, -1))
        graph.observe((0, 0), Direction.RIGHT, (1, 0))
        assert dict(graph.neighbours((0, 0)))[Direction.RIGHT] == (1, 0)

    def test_routing_follows_the_observed_link(self):
        """L'interet reel : un trajet calcule sur l'adjacence supposee enverrait le bot
        vers une carte qu'il n'atteindra pas."""
        graph = MapGraph()
        graph.observe((0, 0), Direction.RIGHT, (0, 1))
        graph.observe((0, 1), Direction.RIGHT, (5, 5))
        assert graph.route((0, 0), (5, 5)) == [Direction.RIGHT, Direction.RIGHT]


class TestGraphPersistence:
    """Le graphe n'a d'interet qu'accumule : une session ne couvre qu'une poignee de
    cartes."""

    def test_round_trip_keeps_maps_and_links(self, tmp_path):
        graph = MapGraph()
        graph.observe((3, 8), Direction.RIGHT, (3, 7))
        graph.observe((3, 7), Direction.BOTTOM, (3, 8))
        path = tmp_path / "sub" / "maps.json"
        graph.to_json(path)
        reloaded = MapGraph.from_json(path)
        assert reloaded.walkable == graph.walkable
        assert reloaded.links == graph.links

    def test_an_empty_graph_round_trips(self, tmp_path):
        """Premiere session : le fichier est ecrit avant toute observation. Le relire ne
        doit pas faire tomber la session suivante."""
        path = tmp_path / "maps.json"
        MapGraph().to_json(path)
        assert MapGraph.from_json(path).walkable == set()

    def test_the_file_stays_readable_by_hand(self, tmp_path):
        """On veut pouvoir l'ouvrir et le corriger : `indent=` eclaterait chaque
        coordonnee sur trois lignes."""
        graph = MapGraph()
        graph.observe((3, 8), Direction.RIGHT, (3, 7))
        path = tmp_path / "maps.json"
        graph.to_json(path)
        assert "[3, 7]" in path.read_text(encoding="utf-8")

    def test_the_hand_written_list_format_still_loads(self, tmp_path):
        """Les circuits deja rediges a la main doivent continuer de fonctionner."""
        path = tmp_path / "maps.json"
        path.write_text(json.dumps([[0, 0], [1, 0]]), encoding="utf-8")
        graph = MapGraph.from_json(path)
        assert graph.walkable == {(0, 0), (1, 0)} and graph.links == {}


class TestMapChanged:
    def _scene(self, seed):
        return np.random.default_rng(seed).integers(0, 255, size=(400, 600, 3),
                                                    dtype=np.uint8)

    def test_same_frame_is_no_change(self):
        frame = self._scene(0)
        assert not map_changed(frame, frame)

    def test_different_scene_is_a_change(self):
        assert map_changed(self._scene(0), self._scene(1))

    def test_small_motion_is_not_a_change(self):
        """Un personnage qui bouge ne doit pas passer pour un changement de carte :
        sinon le circuit avancerait sans qu'on ait quitte la carte."""
        before = self._scene(2)
        after = before.copy()
        cv2.circle(after, (300, 200), 18, (255, 255, 255), -1)
        assert not map_changed(before, after)

    def test_resized_frame_counts_as_change(self):
        """Taille differente : on ne peut rien comparer, on suppose le changement
        plutot que d'affirmer l'immobilite."""
        assert map_changed(self._scene(0), np.zeros((10, 10, 3), dtype=np.uint8))


class TestMapChangeOnRealCaptures:
    """La limite « seuil jamais valide, faute de capture de transition » est levee autant
    qu'elle peut l'etre : les huit captures forment 28 paires, et leurs coordonnees, lues
    puis verifiees a l'oeil, donnent la verite terrain.

        meme carte (1 paire)     49,4
        cartes differentes       12,0 a 50,0, mediane 33,0

    Les deux classes se CHEVAUCHENT. La seule paire « meme carte » oppose une capture hors
    combat a une capture EN combat : la vignette mesure surtout l'interface et l'etat de
    combat, pas le terrain.

    Cela ne condamne pas le seuil, parce que 27 de ces paires ne sont pas la situation ou
    la fonction sert -- entre deux frames encadrant un clic de bord, HORS COMBAT.
    """

    def _frame(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        return cv2.imread(str(path))

    def test_the_only_representative_pair_is_detected(self):
        """Deux cartes differentes, toutes deux HORS COMBAT : la seule paire qui
        reproduise l'usage reel. Ecart 29,4 pour un seuil a 18."""
        assert map_changed(self._frame("capture.png"), self._frame("hors_combat.png"))

    def test_a_frame_against_itself_is_no_change(self):
        frame = self._frame("capture.png")
        assert not map_changed(frame, frame.copy())

    def test_two_similar_combat_maps_are_missed(self):
        """EPINGLE le defaut. bug4 (0,8) et tacle (3,8) sont deux cartes differentes, et
        l'ecart n'est que de 12,0 : la fonction dirait « pas de changement ». Le circuit
        rembobinerait alors un deplacement REUSSI et se decalerait pour de bon.

        Si ce test se met a echouer, c'est une bonne nouvelle -- verifier alors que la
        paire representative ci-dessus passe toujours."""
        assert not map_changed(self._frame("bug4.png"), self._frame("tacle.png"))

    def test_a_different_size_is_always_a_change(self):
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        assert map_changed(small, np.zeros((120, 100, 3), dtype=np.uint8))


class TestTheCircuitIsNotPerfectlyPeriodic:
    """Une trajectoire exactement periodique est la signature la plus simple qui existe.
    Les travaux sur la detection de bots (F-score 99,4 a 99,7 %) exploitent justement les
    trajectoires de deplacement -- et les delais log-normaux et courbes de Bezier deja en
    place ne servent a rien si le CHEMIN, lui, se repete a l'identique indefiniment.

    Le remede est GRATUIT, et c'est ce qui le rend evident une fois vu : la fermeture d'un
    circuit ne depend que du MULTI-ENSEMBLE de ses pas, pas de leur ordre. Le deplacement
    net est une somme, et une somme est commutative.
    """

    def _route(self, graine=0):
        return Route(steps=[Direction.LEFT, Direction.TOP,
                            Direction.RIGHT, Direction.BOTTOM],
                     shuffle=random.Random(graine))

    def _tour(self, route):
        return tuple(route.advance() for _ in range(len(route)))

    def test_two_laps_differ(self):
        route = self._route()
        assert self._tour(route) != self._tour(route)

    def test_closure_survives_every_lap(self):
        """L'invariant que ce projet protege depuis le debut. Une route non fermee fait
        avancer le bot en ligne droite pour toujours, a raison d'une carte par tour."""
        route = self._route()
        for _ in range(20):
            self._tour(route)
            assert route.is_closed and route.net_displacement() == (0, 0)

    def test_the_multiset_is_preserved(self):
        route = self._route()
        avant = sorted(s.value for s in route.steps)
        for _ in range(10):
            self._tour(route)
        assert sorted(s.value for s in route.steps) == avant

    def test_without_a_shuffler_the_order_is_kept(self):
        """Rejouer une session a l'identique doit rester possible."""
        route = Route(steps=[Direction.LEFT, Direction.TOP, Direction.RIGHT,
                             Direction.BOTTOM])
        assert self._tour(route) == self._tour(route)


class TestTheGraphProposesACircuit:
    """Le depot mesurait DEJA quelles cartes rapportent (`yields`, cumule d'une session a
    l'autre) et quelles sorties menent ou (`links`, constatees en jouant). Rien ne
    joignait les deux : composer un circuit restait un travail a la main sur une zone
    qu'on ne connait pas encore -- alors que c'est precisement la question qu'une
    premiere session devrait resoudre elle-meme."""

    def _grille(self, largeur=3, hauteur=3):
        """Une grille pleine, toutes transitions constatees."""
        graph = MapGraph()
        for x in range(largeur):
            for y in range(hauteur):
                graph.walkable.add((x, y))
        for x in range(largeur):
            for y in range(hauteur):
                for direction in Direction:
                    dx, dy = direction.delta
                    voisin = (x + dx, y + dy)
                    if voisin in graph.walkable:
                        graph.observe((x, y), direction, voisin)
        return graph

    def _combats(self, graph, position, combats, passages=2):
        for _ in range(passages):
            graph.record_visit(position, 0, combats)

    def test_it_closes_the_loop(self):
        """Un circuit qui ne revient pas a son point de depart derive d'une carte par
        tour -- la panne la plus difficile a constater du farming."""
        graph = self._grille()
        self._combats(graph, (1, 0), 5)
        self._combats(graph, (2, 2), 4)
        pas = graph.suggest_route((0, 0))
        assert pas, "aucun circuit propose sur un graphe pourtant complet"
        route = Route(steps=list(pas))
        assert route.is_closed, f"circuit ouvert : {route.net_displacement()}"

    def test_it_passes_through_the_best_maps(self):
        graph = self._grille()
        self._combats(graph, (2, 0), 5)
        self._combats(graph, (0, 2), 4)
        visitees = set(Route(steps=list(graph.suggest_route((0, 0)))).offsets())
        assert (2, 0) in visitees and (0, 2) in visitees

    def test_a_single_visit_proposes_nothing(self):
        """Une suggestion batie sur un seul passage serait du bruit presente comme un
        conseil -- le meme piege que les seuils regles sur une capture."""
        graph = self._grille()
        graph.record_visit((1, 1), 0, 9)
        assert graph.suggest_route((0, 0)) is None

    def test_an_empty_graph_proposes_nothing(self):
        assert MapGraph().suggest_route((0, 0)) is None

    def test_an_unreachable_target_refuses_rather_than_drifts(self):
        """Rendre un circuit PARTIEL serait pire que rien : il ne boucle pas."""
        graph = self._grille()
        graph.walkable.add((50, 50))           # ilot, aucune transition vers lui
        self._combats(graph, (50, 50), 9)
        assert graph.suggest_route((0, 0)) is None

    def test_the_ranking_criterion_is_honoured(self):
        """En chasse c'est le combat qui compte ; en recolte, la ressource. Le circuit
        propose doit suivre celui qu'on lui demande."""
        graph = self._grille()
        for _ in range(2):
            graph.record_visit((2, 0), 0, 9)   # que des combats
            graph.record_visit((0, 2), 7, 0)   # que des recoltes
        chasse = Route(steps=list(graph.suggest_route((0, 0), keep=1, by="fights")))
        recolte = Route(steps=list(graph.suggest_route((0, 0), keep=1, by="harvests")))
        assert (2, 0) in set(chasse.offsets())
        assert (0, 2) in set(recolte.offsets())


class TestTheSuggestedCircuitCanBeSaved:
    """Le dernier pas MANUEL d'une chaine ou tout le reste se mesure seul : lire six
    directions dans un bilan et les recopier dans un JSON. Un pas a la main au bout d'une
    mesure automatique est exactement ce qui fait qu'on ne s'en sert pas."""

    def _graphe(self, connu=True):
        graph = MapGraph()
        for x in range(2):
            for y in range(2):
                graph.walkable.add((x, y))
        for x in range(2):
            for y in range(2):
                for direction in Direction:
                    dx, dy = direction.delta
                    if (x + dx, y + dy) in graph.walkable:
                        graph.observe((x, y), direction, (x + dx, y + dy))
        if connu:
            for _ in range(2):
                graph.record_visit((1, 1), 0, 4)
        return graph

    def test_it_writes_a_route_the_runner_can_read(self, tmp_path):
        chemin = tmp_path / "circuit.json"
        message = self._graphe().save_suggested_route((0, 0), chemin)
        assert "ecrit" in message
        # Le vrai controle : ce que `from_json` en relit doit etre un circuit FERME.
        relu = Route.from_json(chemin)
        assert len(relu) and relu.is_closed

    def test_an_existing_route_is_not_clobbered(self, tmp_path):
        """Un circuit existant a souvent ete corrige a la main, sur des heures de jeu.
        Le remplacer par une suggestion tiree de la derniere session serait un echange
        perdant, et silencieux."""
        chemin = tmp_path / "circuit.json"
        chemin.write_text('["top"]', encoding="utf-8")
        message = self._graphe().save_suggested_route((0, 0), chemin)
        assert "non ecrase" in message
        assert chemin.read_text(encoding="utf-8") == '["top"]'
        # Le message doit tout de meme DONNER la suggestion, sinon refuser la fait perdre.
        assert "[" in message.split("suggere est")[-1]

    def test_overwrite_is_explicit(self, tmp_path):
        chemin = tmp_path / "circuit.json"
        chemin.write_text('["top"]', encoding="utf-8")
        message = self._graphe().save_suggested_route((0, 0), chemin, overwrite=True)
        assert "ecrit" in message
        assert Route.from_json(chemin).is_closed

    def test_nothing_to_suggest_writes_nothing(self, tmp_path):
        chemin = tmp_path / "circuit.json"
        message = self._graphe(connu=False).save_suggested_route((0, 0), chemin)
        assert "aucun circuit" in message
        assert not chemin.exists(), "un fichier vide vaut moins que pas de fichier"

    def test_a_route_survives_a_round_trip(self, tmp_path):
        chemin = tmp_path / "circuit.json"
        route = Route(steps=[Direction.RIGHT, Direction.BOTTOM,
                             Direction.LEFT, Direction.TOP])
        route.to_json(chemin)
        assert Route.from_json(chemin).steps == route.steps


class TestYieldsSurviveTheSession:
    """Le rendement par carte existait deja — et mourait avec la session :
    `FarmingSession.yields()` l'affichait puis le perdait.

    Or c'est la seule decision que la recolte demande : un circuit de dix cartes dont trois
    portent tout le rendement se parcourt mieux en trois cartes. Cette decision se prend sur
    des HEURES, pas sur une session, et chacune repartait de zero. Le graphe etait deja
    persistant ; y loger le rendement ne coute rien de plus.
    """

    def test_visits_accumulate(self):
        graph = MapGraph()
        graph.record_visit((5, 8), 3)
        graph.record_visit((5, 8), 1)
        assert graph.yields[(5, 8)] == (4, 0, 2)

    def test_a_single_visit_does_not_win_the_ranking(self):
        """Une carte vue UNE fois et ayant rapporte trois ressources afficherait 3,0 par
        passage et dominerait le classement sur un seul echantillon. C'est le meme piege
        que les seuils regles sur une capture, et il a deja coute assez a ce projet."""
        graph = MapGraph()
        graph.record_visit((1, 9), 3)
        for _ in range(2):
            graph.record_visit((5, 8), 2)
        classement = graph.best_maps(minimum_visits=2)
        assert [position for position, _ in classement] == [(5, 8)]

    def test_fights_accumulate_too(self):
        """Sur un circuit de CHASSE le combat EST le rendement : une zone a larves
        n'offre rien a recolter, et le classement persistant valait donc zero partout --
        precisement dans le mode ou l'on s'en sert pour tailler le circuit."""
        graph = MapGraph()
        graph.record_visit((4, -19), 0, 2)
        graph.record_visit((4, -19), 0, 1)
        assert graph.yields[(4, -19)] == (0, 3, 2)

    def test_the_ranking_can_be_asked_for_fights(self):
        graph = MapGraph()
        for _ in range(2):
            graph.record_visit((4, -19), 0, 3)
            graph.record_visit((5, -19), 0, 1)
        classement = graph.best_maps(minimum_visits=2, by="fights")
        assert [position for position, _ in classement] == [(4, -19), (5, -19)]
        assert classement[0][1] == 3.0

    def test_the_harvest_ranking_ignores_fights(self):
        """Les deux ne se comparent pas : l'appelant choisit, on n'additionne pas."""
        graph = MapGraph()
        for _ in range(2):
            graph.record_visit((4, -19), 0, 9)     # que des combats
            graph.record_visit((5, -19), 2, 0)     # que des recoltes
        classement = graph.best_maps(minimum_visits=2, by="harvests")
        assert classement[0][0] == (5, -19)

    def test_a_fight_round_trip_keeps_them(self, tmp_path):
        graph = MapGraph()
        graph.record_visit((4, -19), 1, 4)
        path = tmp_path / "maps.json"
        graph.to_json(path)
        assert MapGraph.from_json(path).yields == {(4, -19): (1, 4, 1)}

    def test_a_file_written_before_fights_existed_still_loads(self, tmp_path):
        """Refuser les anciennes lignes jetterait un rendement accumule sur des HEURES
        pour un champ ajoute apres coup — c'est ce que la persistance sert a eviter."""
        path = tmp_path / "maps.json"
        path.write_text('{"walkable": [], "links": [], "yields": [[[5, 8], 7, 3]]}',
                        encoding="utf-8")
        assert MapGraph.from_json(path).yields == {(5, 8): (7, 0, 3)}

    def test_a_round_trip_keeps_them(self, tmp_path):
        graph = MapGraph()
        graph.observe((5, 8), Direction.LEFT, (4, 8))
        graph.record_visit((5, 8), 3)
        path = tmp_path / "maps.json"
        graph.to_json(path)
        relu = MapGraph.from_json(path)
        assert relu.yields == {(5, 8): (3, 0, 1)}
        assert relu.links == graph.links

    def test_an_old_file_without_yields_still_loads(self):
        """Les graphes deja ecrits — et les circuits rediges a la main — doivent continuer
        de se charger. Un format qui casse ses propres anciens fichiers punit l'usage."""
        import json

        path = Path(__file__).parent / "_graphe_ancien.json"
        path.write_text(json.dumps({"walkable": [[1, 2]], "links": []}), encoding="utf-8")
        try:
            assert MapGraph.from_json(path).yields == {}
        finally:
            path.unlink()


class TestBarrenMapsOnTheCircuit:
    """La premiere chose de ce projet qui touche le RENDEMENT et non la fiabilite.

    Le circuit ne connait que des directions RELATIVES — c'est ce qui le rend increvable,
    aucune lecture ratee ne peut le derailler. La position de depart, elle, est ABSOLUE.
    Leur composition dit quelles cartes il va parcourir, et le graphe dit ce qu'elles ont
    rapporte.

    Un circuit de six cartes dont trois sont steriles fait perdre la moitie du temps en
    deplacements. Le bot ne REECRIT pas le circuit — une carte peut etre traversee pour en
    atteindre une autre, et lui seul ne peut pas le savoir — mais il le DIT, chiffres a
    l'appui.
    """

    def _route(self):
        return Route(steps=[Direction.RIGHT, Direction.TOP,
                            Direction.LEFT, Direction.BOTTOM])

    def test_it_names_the_barren_ones(self):
        graph = MapGraph()
        for position, taken in (((5, 8), 0), ((5, 8), 0), ((6, 8), 3), ((6, 8), 2),
                                ((6, 7), 0), ((6, 7), 0)):
            graph.record_visit(position, taken)
        assert graph.barren_on(self._route(), (5, 8)) == [(5, 8), (6, 7)]

    def test_a_single_visit_proves_nothing(self):
        """Declarer sterile une carte sur UN passage est le meme piege que regler un seuil
        sur une capture — et une carte peut simplement ne pas avoir eu le temps de
        repousser."""
        graph = MapGraph()
        graph.record_visit((5, 8), 0)
        assert graph.barren_on(self._route(), (5, 8)) == []

    def test_an_unknown_circuit_says_nothing(self):
        assert MapGraph().barren_on(self._route(), (5, 8)) == []

    def test_the_closing_step_is_not_counted_twice(self):
        """Un circuit ferme revient a son point de depart : la derniere position repete la
        premiere, et la compter deux fois ferait apparaitre un doublon dans la liste."""
        graph = MapGraph()
        for _ in range(2):
            graph.record_visit((5, 8), 0)
        assert graph.barren_on(self._route(), (5, 8)).count((5, 8)) == 1

    def test_a_map_that_only_gives_fights_is_not_barren(self):
        """LE conseil inverse. Une carte a larves ne rapporte AUCUNE ressource par
        construction : jugee sur les seules recoltes, elle etait declaree sterile — et le
        bilan conseillait de retirer du circuit ses meilleures cartes."""
        graph = MapGraph()
        for _ in range(3):
            graph.record_visit((5, 8), 0, 2)
        assert graph.barren_on(self._route(), (5, 8)) == []

    def test_a_map_giving_neither_is_still_barren(self):
        """Le pendant : sterile veut dire « rien du tout », pas « rien a recolter »."""
        graph = MapGraph()
        for _ in range(3):
            graph.record_visit((5, 8), 0, 0)
        assert graph.barren_on(self._route(), (5, 8)) == [(5, 8)]


class TestTheCircuitNeverDesyncs:
    """L'invariant dont la violation ne se rattrape JAMAIS.

    Le circuit ne connait que des directions RELATIVES. Si un pas est consomme alors que
    le deplacement n'a pas eu lieu, tout ce qui suit est decale d'un cran -- et comme rien
    dans le circuit n'est absolu, aucun passage ulterieur ne remet les choses en place. Le
    bot parcourt alors une boucle voisine de celle qu'on lui a decrite, indefiniment, sans
    qu'aucun compteur ne s'en plaigne.

    `rewind` et `divert` etaient verifies sur des cas UNIQUES, choisis a la main. Ce qui
    manquait est une suite quelconque : c'est l'enchainement qui decale, pas le pas isole.

    Mesure : 800 suites aleatoires, zero violation.
    """

    SUITES = 200

    def _circuit(self):
        return [step for step in Route.from_json(str(ROUTE)).steps]

    def test_failures_never_shift_the_circuit(self):
        """Un pas rembobine ne doit RIEN consommer : les directions effectivement suivies
        doivent redonner le circuit, dans l'ordre, indefiniment."""
        base = self._circuit()
        for graine in range(self.SUITES):
            rng = random.Random(graine)
            route = Route.from_json(str(ROUTE))
            suivies = []
            for _ in range(24):
                direction = route.advance()
                if direction is None:
                    break
                if rng.random() < 0.35:
                    route.rewind()          # le deplacement n'a pas eu lieu
                else:
                    suivies.append(direction)
            attendu = [base[i % len(base)] for i in range(len(suivies))]
            assert suivies == attendu, f"circuit decale (graine {graine})"

    def test_detours_do_not_consume_circuit_steps(self):
        """Un detour corrige une derive : il s'intercale SANS avancer le circuit. Le
        confondre avec un pas ordinaire decalerait tout ce qui suit -- et un detour
        rembobine doit rester a faire, sinon la correction est perdue et la derive
        devient permanente."""
        base = self._circuit()
        for graine in range(self.SUITES):
            rng = random.Random(graine)
            route = Route.from_json(str(ROUTE))
            suivies, restants = [], 0
            for _ in range(30):
                if restants == 0 and rng.random() < 0.15:
                    combien = rng.randint(1, 2)
                    route.divert(*[rng.choice(list(Direction)) for _ in range(combien)])
                    restants = combien
                direction = route.advance()
                if direction is None:
                    break
                if rng.random() < 0.3:
                    route.rewind()
                    continue
                if restants:
                    restants -= 1
                else:
                    suivies.append(direction)
            attendu = [base[i % len(base)] for i in range(len(suivies))]
            assert suivies == attendu, f"circuit decale par un detour (graine {graine})"

    def test_the_sweep_really_walks(self):
        """Contre-epreuve : deux invariants sur des suites vides seraient deux tests
        vides."""
        route = Route.from_json(str(ROUTE))
        suivies = [route.advance() for _ in range(12)]
        assert all(d is not None for d in suivies)

    def test_a_failure_on_the_wrap_does_not_break_a_SHUFFLED_lap(self):
        """LES DEUX SONDES CI-DESSUS CONSTRUISENT LEURS ROUTES SANS REBATTAGE, et c'est
        precisement la ou le defaut vivait : il ne nait que de la RENCONTRE des deux.

        `advance()` rebat les pas au moment ou il reboucle. Si ce pas-la echoue, l'ancien
        `rewind()` ramenait l'indice a `len - 1` -- mais dans la liste NEUVE, ou cet indice
        ne designe plus le pas rate, et ou il place le curseur a la FIN du tour. Le
        prochain `advance` consommait donc ce dernier pas, rebouclait aussitot, et les
        len-1 premiers pas du tour n'etaient JAMAIS joues. Un « tour » d'un seul pas ne
        vaut pas (0, 0) : le circuit cesse d'etre ferme et le bot s'eloigne de sa zone,
        exactement le mode de panne que `is_closed` existe pour interdire.

        MESURE, avant correction, sur le circuit d'exemple : 199 suites sur 200. Apres :
        0 sur 200, et a 0 %, 10 %, 35 % comme a 60 % d'echecs.

        L'INVARIANT N'EST PAS L'ORDRE mais le MULTI-ENSEMBLE : le rebattage change l'ordre
        par construction (c'est ce qu'on lui demande), et la fermeture d'un circuit ne
        depend que de la somme de ses pas. On decoupe donc les directions SUIVIES en
        tranches de la longueur du circuit, et chaque tranche doit etre une permutation.
        """
        base = self._circuit()
        n = len(base)
        for taux in (0.1, 0.35, 0.6):
            for graine in range(self.SUITES):
                rng = random.Random(graine)
                route = Route.from_json(str(ROUTE))
                route.shuffle = random.Random(graine)
                suivies = []
                for _ in range(8 * n):
                    direction = route.advance()
                    if direction is None:
                        break
                    if rng.random() < taux:
                        route.rewind()      # le deplacement n'a pas eu lieu
                    else:
                        suivies.append(direction)
                for depart in range(0, len(suivies) // n * n, n):
                    tranche = suivies[depart:depart + n]
                    assert sorted(tranche, key=lambda d: d.value) == sorted(
                        base, key=lambda d: d.value), (
                        f"tour incomplet a {taux:.0%} d'echecs (graine {graine}, "
                        f"pas {depart}) : {[d.value for d in tranche]}")

    def test_a_shuffled_lap_really_reorders(self):
        """CONTRE-EPREUVE du precedent : un invariant sur le multi-ensemble serait vide si
        le rebattage ne rebattait rien. Il doit produire un ordre AUTRE au moins une fois
        sur quelques tours, sinon le test ci-dessus ne parle que du cas non rebattu."""
        base = self._circuit()
        route = Route.from_json(str(ROUTE))
        route.shuffle = random.Random(0)
        suivies = [route.advance() for _ in range(4 * len(base))]
        tranches = [suivies[i:i + len(base)]
                    for i in range(0, len(suivies), len(base))]
        assert any(tranche != base for tranche in tranches), "aucun tour rebattu"


class TestTheGraphSaysWhenItLearnedNothing:
    """Constate en repetition generale : une session de deux alternances ecrit un
    maps.json entierement vide, et le bilan affiche « 6 cartes » puis « 0 transitions »
    sur deux lignes voisines. Cette contradiction apparente se lit comme un graphe casse.

    C'est l'inverse. Une liaison ne s'enregistre que si les coordonnees sont lues AVANT et
    APRES le changement ; sinon `_travel` retombe sur la comparaison d'images, qui
    confirme le changement sans savoir vers ou. Le circuit avance, le graphe reste
    aveugle -- et il ne pourra donc ni corriger une derive ni conseiller un circuit.

    La cause est unique et nommable : autant que le bot la nomme lui-meme.
    """

    def test_maps_walked_without_a_single_link_is_named(self):
        message = MapGraph().describe_learning(6)
        assert message is not None and "coordonnees" in message

    def test_a_graph_that_learned_says_nothing(self):
        graph = MapGraph()
        graph.observe((5, 8), Direction.RIGHT, (6, 8))
        assert graph.describe_learning(6) is None

    def test_a_session_that_never_moved_says_nothing(self):
        """Zero carte parcourue n'est pas un defaut de lecture : il n'y avait rien a
        apprendre. Signaler la serait un faux positif sur la session la plus courte."""
        assert MapGraph().describe_learning(0) is None


class TestLapsAreCounted:
    """Compteur incremente depuis toujours et lu nulle part. « 6 cartes parcourues » ne
    dit pas si le circuit a BOUCLE une fois ou pas du tout -- or c'est ce qui distingue un
    bot qui tourne d'un bot qui derive."""

    def test_a_full_circuit_counts_one_lap(self):
        route = Route.from_json(str(ROUTE))
        for _ in range(len(route)):
            route.advance()
        assert route.laps == 1

    def test_a_partial_circuit_counts_none(self):
        route = Route.from_json(str(ROUTE))
        for _ in range(len(route) - 1):
            route.advance()
        assert route.laps == 0

    def test_rewinding_across_the_wrap_undoes_the_lap(self):
        """Sinon un pas rate juste apres la boucle compterait un tour de trop, et le
        chiffre grossirait tout seul sur un circuit bloque."""
        route = Route.from_json(str(ROUTE))
        for _ in range(len(route)):
            route.advance()
        route.rewind()
        assert route.laps == 0
