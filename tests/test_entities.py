"""Detecteur v1 (marqueurs de cellule) : l'echantillonnage par la grille identifie la
BONNE case et la BONNE equipe. Frames synthetiques -- la geometrie testee ici ne depend
pas des bandes HSV reelles, relevees separement sur captures."""

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.perception.entities import (
    GRID_CORNER_INSET,
    TEAM_BANDS,
    detect_entities,
    detect_markers_on_board,
    enemy_candidates_outside,
)
from jev_tactics.perception.highlight import looks_like_spell_range
from jev_tactics.perception.timeline import read_timeline
from jev_tactics.pipeline import build_board, observe
from jev_tactics.rules.movement import grid_distance
from jev_tactics.state import Team

E_X = np.array([46.0, 23.0])
E_Y = np.array([-46.0, 23.0])
ORIGIN = np.array([700.0, 400.0])
CANVAS = (900, 1500, 3)

ENEMY_BGR = (0, 90, 230)     # orange-rouge -> H~14, dans la bande ENEMY
ALLY_BGR = (245, 245, 245)   # blanc casse -> S bas / V haut, dans la bande ALLY
#                              (le marqueur joueur est un contour BLANC, cf. entities.py)


def _board(cells: list[tuple[int, int]]) -> BoardMap:
    return BoardMap(cells=np.array(cells, dtype=np.int64),
                    e_x=E_X, e_y=E_Y, origin=ORIGIN)


# Petit plateau de cases bien separees, toutes dans le canvas.
CELLS = [(-3, -3), (-1, -3), (1, -3), (-3, -1), (-1, -1), (1, -1),
         (-3, 1), (-1, 1), (1, 1)]
BOARD = _board(CELLS)


def _draw_marker(frame, index: int, color, thickness: int = 4) -> None:
    """Trace le marqueur d'une case, au rayon ou le jeu le dessine (GRID_CORNER_INSET)."""
    pts = BOARD.outline(index, per_edge=2, inset=GRID_CORNER_INSET)
    cv2.polylines(frame, [pts[::2].astype(np.int32)], True, color, thickness)


def test_board_cells_are_inside_canvas():
    for k in range(len(BOARD)):
        pts = BOARD.outline(k)
        assert pts[:, 0].min() > 0 and pts[:, 0].max() < CANVAS[1]
        assert pts[:, 1].min() > 0 and pts[:, 1].max() < CANVAS[0]


def test_marker_maps_to_its_own_cell():
    frame = np.zeros(CANVAS, dtype=np.uint8)
    _draw_marker(frame, 4, ENEMY_BGR)
    assert [(e.cell, e.team) for e in detect_entities(frame, BOARD)] == [(4, Team.ENEMY)]


def test_multiple_markers_and_teams():
    frame = np.zeros(CANVAS, dtype=np.uint8)
    _draw_marker(frame, 0, ENEMY_BGR)
    _draw_marker(frame, 4, ENEMY_BGR)
    _draw_marker(frame, 8, ALLY_BGR)

    found = {e.cell: e.team for e in detect_entities(frame, BOARD)}
    assert found == {0: Team.ENEMY, 4: Team.ENEMY, 8: Team.ALLY}


def test_cell_ids_are_valid_and_unique():
    frame = np.zeros(CANVAS, dtype=np.uint8)
    _draw_marker(frame, 2, ENEMY_BGR)
    cells = [e.cell for e in detect_entities(frame, BOARD)]
    assert len(cells) == len(set(cells))
    assert all(0 <= c < len(BOARD) for c in cells)


def test_empty_frame_detects_nothing():
    assert detect_entities(np.zeros(CANVAS, dtype=np.uint8), BOARD) == []


def test_blob_at_cell_centre_is_not_a_marker():
    """Le vote porte sur le contour : une tache au centre ne doit rien declencher."""
    frame = np.zeros(CANVAS, dtype=np.uint8)
    cx, cy = BOARD.center(4)
    cv2.circle(frame, (int(cx), int(cy)), 8, ENEMY_BGR, -1)
    assert detect_markers_on_board(frame, BOARD) == []


class TestBoardMap:
    def test_center_matches_lattice_arithmetic(self):
        i, j = BOARD.cells[3]
        assert np.allclose(BOARD.center(3), ORIGIN + i * E_X + j * E_Y)

    def test_index_at_roundtrip(self):
        for k in range(len(BOARD)):
            i, j = BOARD.cells[k]
            assert BOARD.index_at(int(i), int(j)) == k

    def test_index_at_off_board(self):
        assert BOARD.index_at(99, 99) is None

    def test_nearest_finds_own_cell(self):
        for k in range(len(BOARD)):
            cx, cy = BOARD.center(k)
            assert BOARD.nearest(cx + 3.0, cy - 2.0) == k

    def test_neighbours_are_adjacent_and_on_board(self):
        # BOARD espace ses cases de 2 : il faut un plateau CONTIGU pour l'adjacence.
        dense = _board([(i, j) for i in range(-1, 2) for j in range(-1, 2)])
        centre = dense.index_at(0, 0)
        neighbours = dense.neighbours(centre)
        assert len(neighbours) == 4, "la case centrale a 4 voisins sur un plateau plein"
        for n in neighbours:
            di, dj = dense.cells[n] - dense.cells[centre]
            assert abs(int(di)) + abs(int(dj)) == 1

    def test_neighbours_clipped_at_board_edge(self):
        dense = _board([(i, j) for i in range(-1, 2) for j in range(-1, 2)])
        corner = dense.index_at(-1, -1)
        assert len(dense.neighbours(corner)) == 2  # un coin n'a que 2 voisins

    def test_ordering_is_deterministic(self):
        shuffled = np.array(CELLS[::-1], dtype=np.int64)
        from jev_tactics.calibration.grid import GridEstimate

        a = BoardMap.from_estimate(GridEstimate(E_X, E_Y, ORIGIN, 1.0, shuffled))
        b = BoardMap.from_estimate(
            GridEstimate(E_X, E_Y, ORIGIN, 1.0, np.array(CELLS, dtype=np.int64)))
        assert np.array_equal(a.cells, b.cells)


class TestTintedCells:
    """Cases TEINTEES prises pour des marqueurs.

    Defaut constate en jeu, et diagnostique par l'utilisateur : quand le personnage est
    TACLE, Dofus colore sa zone de deplacement en ROUGE -- la teinte meme que porte le
    marqueur ennemi. Le bot y a lu 12 ennemis pour 3 combattants, tous groupes dans la
    meme direction, et a depense un tour entier sur une case vide.

    Le discriminant est geometrique, pas colorimetrique : un marqueur est une LIGNE sur le
    pourtour, une teinte remplit la case."""

    def _board(self):
        return BoardMap(
            cells=np.array([(i, j) for j in range(4) for i in range(4)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([400.0, 300.0]))

    def _frame(self):
        return np.zeros((700, 900, 3), dtype=np.uint8)

    def _fill(self, frame, board, cell, inset, colour=(30, 60, 220)):
        pts = board.outline(cell, per_edge=40, inset=inset).astype(np.int32)
        cv2.fillPoly(frame, [pts], colour)

    def test_a_ring_is_a_marker(self):
        """Anneau : rouge au pourtour, terrain au centre."""
        board, frame = self._board(), self._frame()
        self._fill(frame, board, 5, 0.5)
        self._fill(frame, board, 5, 0.30, colour=(0, 0, 0))   # recreuser le coeur
        assert any(m.cell == 5 for m in detect_markers_on_board(frame, board))

    def test_a_filled_cell_is_not_a_marker(self):
        """Case pleine : c'est un surlignage, pas un combattant."""
        board, frame = self._board(), self._frame()
        self._fill(frame, board, 5, 0.5)
        assert not any(m.cell == 5 for m in detect_markers_on_board(frame, board))

    def test_a_whole_tinted_zone_produces_nothing(self):
        """Le cas reel : toute une zone teintee. Sans ce controle, chaque case comptait
        pour un ennemi."""
        board, frame = self._board(), self._frame()
        for cell in range(len(board)):
            self._fill(frame, board, cell, 0.5)
        assert detect_markers_on_board(frame, board) == []


class TestEnemiesOutsideTheDetectedBoard:
    """Reformulation d'une limitation connue, appuyee sur trois mesures NEGATIVES.

    Le plateau de combat1.png s'arrete avant deux monstres : « 1 joueur, 0 ennemis, ecart
    timeline 2 », et la session refuse de jouer sans dire ou chercher. Or les deux monstres
    portent un marqueur d'equipe parfaitement ordinaire, pose sur le RESEAU. Verifie a
    l'oeil en zoomant : un Scarafeuille blanc en (431, 293), un bleu en (569, 317).

    Ce n'est donc pas la detection de plateau qu'il faut reprendre. En elargissant le
    reseau on ramene quatre candidats pour deux monstres -- les deux vrais, un arbre
    d'automne et le panneau lateral -- et aucune des trois mesures du module ne les separe :

        vote de couleur    blanc 0,292 | ARBRE 0,292 | bleu 0,271 | panneau 0,250
        reponse de grille  ARBRE 19,4  | bleu 17,2   | blanc 16,0 | panneau 11,6
        anneau vs tache    blanc 0,000 | bleu 0,000  | ARBRE 0,021 | panneau 0,146

    D'ou une fonction qui SIGNALE sans CHOISIR : le jour ou un discriminant existera, il
    aura ces candidats en entree.
    """

    def _board(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        return frame, build_board(frame)

    def test_the_two_missing_monsters_are_among_the_candidates(self):
        frame, board = self._board("combat1.png")
        found = enemy_candidates_outside(frame, board)
        for expected in ((431, 294), (569, 317)):
            assert any(abs(x - expected[0]) <= 2 and abs(y - expected[1]) <= 2
                       for x, y, _ in found), f"{expected} absent de {found}"

    def test_scenery_is_among_them_too_and_is_not_filtered(self):
        """Le test dirait le contraire si je pretendais avoir resolu le probleme.
        L'arbre en (201, 410) est la, avec le meme vote que le Scarafeuille blanc."""
        frame, board = self._board("combat1.png")
        found = enemy_candidates_outside(frame, board)
        assert any(abs(x - 201) <= 2 and abs(y - 410) <= 2 for x, y, _ in found)

    def test_an_empty_board_yields_nothing(self):
        board = BoardMap(cells=np.empty((0, 2), dtype=np.int64),
                         e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
                         origin=np.array([300.0, 300.0]))
        assert enemy_candidates_outside(np.zeros((80, 80, 3), np.uint8), board) == []

    def test_results_are_ranked_by_vote(self):
        frame, board = self._board("combat1.png")
        votes = [vote for _x, _y, vote in enemy_candidates_outside(frame, board)]
        assert votes == sorted(votes, reverse=True)


class TestTheEnemyRingIsPureRed:
    """Le defaut le plus couteux de la perception, et il tenait a UN chiffre.

    Sur 24 captures d'une vraie session, douze manquaient des ennemis et cinq n'en voyaient
    AUCUN. J'ai d'abord accuse le garde-fou anti-teinte : les cases rouges PLEINES y sont
    rejetees parce que leur coeur est rempli a 96-100 %. C'etait faux, et la verification a
    l'oeil l'a montre -- ces cases sont une teinte de zone, pas des ennemis. Le garde-fou
    avait raison.

    Les vrais ennemis etaient ailleurs. Le mode d'affichage de cette session ecrit
    « ENNEMI 85 » au-dessus de chaque case ennemie ; en detectant ce texte rouge vif et en
    le rabattant de 40 px, on tombe a 4 px des cases 60 et 64 -- exactement les deux
    monstres. Et LA, la mesure du contour donne :

        contour des cases ennemies   teinte 0, S=255, V=237 a 255
        bande attendue               teinte 3 a 17

    La borne basse excluait le rouge PUR. Il n'en restait qu'une fraction dans la bande, et
    le vote tombait a 0,12 et 0,19 pour un seuil a 0,20. Juste en dessous.

        bande     comptages justes   manquent   en trop      (30 captures)
        3..17            16             13          1
        0..17            27              2          1
    """

    def _frame(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        return cv2.imread(str(path))

    def test_the_two_labelled_enemies_are_seen(self):
        """20260807-221351.png : deux libelles « ENNEMI 85 », deux ennemis sur la timeline,
        et zero detecte avant ce correctif."""
        frame = self._frame("data/runs/20260807-221351.png")
        board = build_board(frame)
        enemies = [m for m in detect_markers_on_board(frame, board)
                   if m.team is Team.ENEMY]
        assert len(enemies) >= 2

    def test_pure_red_is_inside_the_band(self):
        """La borne basse EST le correctif : a 3, le rouge pur du contour tombait dehors."""
        low, _high = TEAM_BANDS[Team.ENEMY][0]
        assert low[0] == 0

    def test_the_timeline_agrees(self):
        frame = self._frame("data/runs/20260807-221351.png")
        board = build_board(frame)
        vus = sum(1 for e in observe(frame, board, cursor=None).entities if e.team is Team.ENEMY)
        annonces = sum(1 for e in read_timeline(frame) if e.team is Team.ENEMY)
        assert vus == annonces

class TestTheMissingDiscriminator:
    """La circularite decrite plus haut EST levee -- `movement_centre` donne le joueur sans
    passer par les marqueurs. Ce n'est donc plus l'obstacle. Ce qui manque est un test de
    « portee de sort » assez fin, et deux formes ont ete mesurees et refusees :

        contiguite des distances   repond oui des qu'on lui donne six cases
        remplissage de l'anneau    portees 0,75 0,74 0,40 0,33 | ennemis 0,28 0,24 0,10

    Cinq centiemes separent les deux classes, contre 36 % pour le crane du chat et 56 %
    pour le mot « Epuise ». Ce test epingle la faiblesse du premier, pour qu'on ne le
    reutilise pas ailleurs en le croyant discriminant.
    """

    def test_contiguity_alone_accepts_a_handful_of_scattered_cells(self):
        board = BoardMap(
            cells=np.array([(i, j) for j in range(9) for i in range(9)], dtype=np.int64),
            e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
            origin=np.array([600.0, 400.0]))
        centre = board.index_at(4, 4)
        eparses = {c for c in range(len(board))
                   if grid_distance(board, centre, c) in (2, 3, 4)}
        six = set(sorted(eparses)[:6])
        assert looks_like_spell_range(board, six, centre), (
            "le test accepte six cases eparses : c'est la faiblesse mesuree")


class TestTheAllyVoteIsNarrow:
    """Marge mesuree avec le libelle « MOI » comme ORACLE : il designe la case du
    personnage, on mesure ce que le detecteur y voit.

        4 captures sur 6    vote 0,25 a 0,27      seuil 0,20
        2 captures sur 6    vote 0,00             contour vert sature (H=36-43, S=170-180)

    Un quart au-dessus du seuil quand ca marche, rien du tout sinon. C'est ce qui rend le
    secours par `movement_centre` decisif -- il a fait passer la localisation de 15 a 22
    captures sur 24.

    Non corrige : elargir la bande vers le vert sature attraperait la ZONE DE DEPLACEMENT,
    qui couvre justement les cases autour du joueur. Le remede serait pire que le mal.
    """

    def test_the_ally_band_stays_desaturated(self):
        """Le garde-fou du non-correctif : si quelqu'un elargit la saturation pour
        rattraper les deux captures ratees, il attrapera le vert du deplacement."""
        _low, high = TEAM_BANDS[Team.ALLY][0]
        assert high[1] <= 70, "bande ALLY etendue vers le sature : cf. la note ci-dessus"

    def test_the_rescue_is_what_covers_the_gap(self):
        """Les deux captures ou le vote vaut zero sont localisees quand meme, par la zone
        de deplacement. C'est le filet, et il doit rester en place."""
        for name in ("data/runs/20260807-220501.png", "data/runs/20260807-220519.png"):
            path = Path(__file__).resolve().parents[1] / name
            if not path.exists():
                pytest.skip(f"{name} absent")
            frame = cv2.imread(str(path))
            me = next((e for e in observe(frame, build_board(frame), cursor=None).entities
                       if e.is_self), None)
            assert me is not None, name


class TestTheLastTwoGapsAreBoardTruncation:
    """Apres l'elargissement de la bande de teinte, il reste DEUX comptages faux sur trente
    -- et ils ont la meme cause, deja documentee.

        combat1.png                  2 ennemis annonces, 0 vus
        data/runs/20260807-224524    1 ennemi annonce,  0 vu

    Sur la seconde, l'infobulle de survol affiche « Bouftou 139/186 (74%) Niv. 24 » : le
    monstre est bien la, a x~460. Le plateau detecte, lui, commence a x=525. Le marqueur
    n'est pas manque, il n'est jamais ECHANTILLONNE.

    Ce n'est donc pas un probleme de couleur ni de seuil, et aucun reglage de detection ne
    le corrigera. `enemy_candidates_outside` les signale deja, faute de pouvoir les
    distinguer du decor (cinq discriminants mesures et ecartes, cf. son docstring).
    """

    def _board_and_frame(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        return frame, build_board(frame)

    @pytest.mark.parametrize("name", ["combat1.png",
                                      "data/runs/20260807-224524.png"])
    def test_the_missing_enemy_is_outside_the_detected_board(self, name):
        frame, board = self._board_and_frame(name)
        vus = sum(1 for e in observe(frame, board, cursor=None).entities if e.team is Team.ENEMY)
        annonces = sum(1 for e in read_timeline(frame) if e.team is Team.ENEMY)
        assert vus < annonces, "plus de troncature : verifier les cinq discriminants"
        assert enemy_candidates_outside(frame, board), (
            "aucun candidat hors plateau : la cause n'est plus la troncature")


class TestTheEnemyCountAcrossEverySessionCapture:
    """Le chiffre 27/30 etait ECRIT dans le docstring voisin, jamais verifie. Un seul
    cliche l'etait (221351), si bien qu'une regression ramenant le compte a 18 serait
    passee inapercue tant qu'elle epargnait cette capture-la.

    Le temoin porte sur l'ENSEMBLE des captures en desaccord, pas sur leur nombre : une
    modification qui en repare une et en casse une autre laisse le total a 27 et change la
    liste. C'est ce changement-la qu'on veut voir.

    Les trois exceptions sont connues et deja epinglees ailleurs : deux plateaux tronques
    qui coupent leurs ennemis, et une capture qui voit un marqueur de trop.

    COUT : une trentaine de secondes, soit pres de la moitie de la suite complete. Le prix
    est assume et non reduit -- restreindre a un echantillon rendrait au temoin le defaut
    qu'il corrige, celui de ne surveiller qu'une partie du terrain.
    """

    DESACCORDS: ClassVar = {"20260807-222701.png", "20260807-224524.png", "combat1.png"}
    ATTENDUS_JUSTES = 27

    def _captures(self):
        racine = Path(__file__).resolve().parents[1]
        fichiers = sorted((racine / "data" / "runs").glob("*.png"))
        fichiers += [racine / nom for nom in ("combat1920.png", "combat1.png", "bug.png",
                                              "bug4.png", "tacle.png", "bugcarreblanc.png")]
        presents = [f for f in fichiers if f.exists()]
        if len(presents) < 20:
            pytest.skip("captures de session absentes")
        return presents

    _CACHE: ClassVar[dict] = {}

    def _desaccords(self):
        # Trente plateaux a detecter : une trentaine de secondes. Les deux tests de cette
        # classe posent la MEME mesure sous deux angles ; la recalculer doublerait la
        # facture sans rien apprendre.
        if "ecarts" in self._CACHE:
            return self._CACHE["ecarts"]
        ecarts = {}
        for chemin in self._captures():
            frame = cv2.imread(str(chemin))
            board = build_board(frame)
            if board is None:
                ecarts[chemin.name] = ("pas de plateau", None)
                continue
            vus = sum(1 for e in observe(frame, board, cursor=None).entities if e.team is Team.ENEMY)
            annonces = sum(1 for e in read_timeline(frame) if e.team is Team.ENEMY)
            if vus != annonces:
                ecarts[chemin.name] = (annonces, vus)
        self._CACHE["ecarts"] = ecarts
        return ecarts

    def test_the_disagreeing_captures_are_exactly_the_known_ones(self):
        ecarts = self._desaccords()
        assert set(ecarts) == self.DESACCORDS, (
            f"la liste des desaccords a change : {ecarts}")

    def test_the_measured_level_is_held(self):
        """Le chiffre lui-meme, pour que la baisse se lise sans decoder une liste."""
        captures = self._captures()
        justes = len(captures) - len(self._desaccords())
        assert justes >= self.ATTENDUS_JUSTES, (
            f"{justes} comptages justes sur {len(captures)}, mesure a "
            f"{self.ATTENDUS_JUSTES}")
