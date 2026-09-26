"""Exclusion des panneaux d'interface : ce que les deux detecteurs de terrain ne
doivent PAS prendre pour du mouvement."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.ui import UI_ZONES, hidden_by_ui, ui_mask
from jev_tactics.pipeline import build_board


class TestOpenPanelsAreMasked:
    """`UI_ZONES` a ete dessine sur une capture reelle, et le resultat etait accablant :
    trois rectangles, dont un couvrant du TERRAIN JOUABLE en haut a droite, tandis que le
    chat qui defile, la carte des quetes, le panneau d'attitudes et la minicarte restaient
    grand ouverts. Quatre sources d'animation permanente, dans un projet dont les deux
    detecteurs de terrain reposent justement sur le mouvement.

    Des rectangles de plus ne repareraient rien durablement : ces panneaux s'ouvrent et se
    deplacent. On les reconnait donc a ce qu'ils SONT. Mesure sur capture.png, part de
    pixels sombres :

        chat 0,82 | attitudes 0,84 | quetes 0,58   |   herbe 0,012 | chemin 0,017
    """

    def _mask(self, name="capture.png"):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        return ui_mask(cv2.imread(str(path)))

    @pytest.mark.parametrize("point", [(180, 900), (150, 200), (1700, 470)])
    def test_panels_absent_from_ui_zones_are_covered(self, point):
        assert hidden_by_ui(self._mask(), *point)
        assert not any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                       for x0, y0, x1, y1 in UI_ZONES), "deja couvert par UI_ZONES"

    @pytest.mark.parametrize("point", [(1080, 505), (700, 400)])
    def test_the_map_stays_visible(self, point):
        """Le troupeau de Bouftous et l'herbe : masquer la carte serait pire que le mal."""
        assert not hidden_by_ui(self._mask(), *point)

    def test_a_mostly_dark_frame_is_refused(self):
        """Une frame majoritairement sombre n'est pas une interface -- chargement, scene
        de nuit, image de test. Masquer 90 % de l'ecran rendrait les deux detecteurs de
        terrain aveugles SANS RIEN DIRE, ce qui est le pire des echecs de ce projet."""
        assert not ui_mask(np.zeros((400, 600, 3), dtype=np.uint8)).any()

    def test_no_mask_blocks_nothing(self):
        assert not hidden_by_ui(None, 10, 10)

    def test_a_point_outside_the_frame_is_not_hidden(self):
        assert not hidden_by_ui(self._mask(), 99_999, 99_999)


class TestTopBarsAreExcluded:
    """Les barres d'icones du bandeau superieur sont le seul element d'ATH que `ui_mask`
    ne peut PAS rattraper : leurs icones sont CLAIRES (0,023 de pixels sombres, contre
    0,82 pour le chat). Elles etaient absentes de `UI_ZONES`.

    Relevees colonne par colonne sur cinq captures, en combat et hors combat : deux plages
    sortent sur les CINQ, (0..81) et (1850..1920), et la barre centrale droite sur quatre.
    """

    @pytest.mark.parametrize("point", [(40, 20), (1600, 20), (1880, 20)])
    def test_the_bars_are_covered(self, point):
        assert any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                   for x0, y0, x1, y1 in UI_ZONES)

    @pytest.mark.parametrize("point", [(940, 20), (700, 300)])
    def test_the_map_between_them_stays_visible(self, point):
        """Le centre du bandeau est de la CARTE : y placer un rectangle par prudence
        aveuglerait le bot sur une bande de terrain."""
        assert not any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                       for x0, y0, x1, y1 in UI_ZONES)

    def test_they_cover_no_board_cell(self):
        """Le controle qui a autorise l'ajout : sur les huit captures ces rectangles ne
        recouvrent aucune case de plateau, sauf 3 sur 294 hors combat -- sous la barre,
        donc ni visibles ni cliquables."""
        path = Path(__file__).resolve().parents[1] / "combat1920.png"
        if not path.exists():
            pytest.skip("combat1920.png absent")
        board = build_board(cv2.imread(str(path)))
        top = [z for z in UI_ZONES if z[3] <= 45]
        hits = [i for i in range(len(board))
                for x0, y0, x1, y1 in top
                if x0 <= board.center(i)[0] <= x1 and y0 <= board.center(i)[1] <= y1]
        assert hits == []


class TestPanelsAreNotJustDarkSpeckle:
    """Le masque declarait CACHEES des cases en plein milieu du plateau.

    Cause : la fermeture morphologique soude en une seule composante des pixels sombres
    EPARS -- bordures de cases, ombre d'un sprite, feuillage -- et l'enveloppe obtenue
    depasse l'aire minimale. Elle n'a pourtant rien d'un panneau. Mesure de la densite de
    pixels vraiment sombres a l'interieur des composantes retenues, sur quatre captures :

        panneaux reels                   0,64 a 0,89
        amas soudes (plateau, decor)     0,14 a 0,50

    Sans ce controle, jusqu'a QUATRE cases de combat par capture etaient masquees a tort :
    le planificateur pouvait les viser, et le clic serait tombe sur ce que le masque
    croyait etre une fenetre. Le defaut etait donc de meme nature que celui qu'il causait.
    """

    @pytest.mark.parametrize("name", ["bugcarreblanc.png", "combat1.png", "tacle.png",
                                      "bug.png", "bug4.png"])
    def test_no_combat_cell_is_masked(self, name):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        frame = cv2.imread(str(path))
        board = build_board(frame)
        mask = ui_mask(frame)
        hidden = [i for i in range(len(board))
                  if hidden_by_ui(mask, *board.center(i))]
        assert hidden == [], f"{len(hidden)} case(s) masquee(s) a tort"

    def test_the_real_panels_are_still_masked(self):
        """Le pendant : un garde-fou qui vide le masque ne vaudrait rien."""
        path = Path(__file__).resolve().parents[1] / "bugcarreblanc.png"
        if not path.exists():
            pytest.skip("bugcarreblanc.png absent")
        mask = ui_mask(cv2.imread(str(path)))
        assert hidden_by_ui(mask, 200, 850)      # chat
        assert hidden_by_ui(mask, 1700, 500)     # attitudes
        assert not hidden_by_ui(mask, 300, 300)  # herbe
