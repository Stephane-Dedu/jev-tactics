"""Infobulle de survol : savoir AVANT de cliquer qu'une ressource ne rapportera rien."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.tooltip import MIN_SCORE, depleted_score, is_depleted

ROOT = Path(__file__).resolve().parents[1]


def _frame(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} absent")
    return cv2.imread(str(path))


class TestDepletedResources:
    """Deux captures fournies par l'utilisateur montrent les DEUX raisons d'un clic sans
    effet :

        data/epuise.png      Frene         +0% XP    « Epuise »
        data/pasniveau.png   Chataignier   +2% XP    « Bucheron Niv.20 »

    La ressource est bien la, bien detectee, bien surlignee -- et cliquer ne rapporte
    rien. Le bot le decouvrait apres 3,5 s d'attente, six fois de suite avant d'abandonner
    la carte.

        data/epuise.png                              1,000
        data/pasniveau.png                           0,560
        les huit captures de jeu (combat et carte)   0,599 a 0,642

    L'intervalle entre 0,642 et 1,000 est vide.
    """

    def test_the_depleted_tooltip_is_recognised(self):
        assert is_depleted(_frame("data/epuise.png"))

    def test_a_level_locked_resource_is_not_called_depleted(self):
        """Elle N'EST PAS epuisee : elle rapporterait, a un autre personnage. La confondre
        ferait retirer du jeu des ressources parfaitement bonnes des que le metier monte.

        Cette ligne-la n'est volontairement PAS lue : le metier et le niveau varient, et
        le bot ne connait de toute facon pas ses propres niveaux. Une ressource hors
        niveau reste donc cliquee, puis retiree par le garde-fou des recoltes sans effet.
        """
        assert not is_depleted(_frame("data/pasniveau.png"))

    @pytest.mark.parametrize("name", ["capture.png", "hors_combat.png", "combat1.png",
                                      "bug.png", "tacle.png"])
    def test_ordinary_frames_are_not_depleted(self, name):
        assert not is_depleted(_frame(name))

    def test_the_threshold_sits_in_an_empty_interval(self):
        assert 0.65 < MIN_SCORE < 1.0

    def test_a_frame_smaller_than_the_glyph_is_safe(self):
        assert depleted_score(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0


class TestWhatIsNotKnown:
    """Trois detecteurs d'infobulle mesures et REFUSES, consignes pour que la prochaine
    tentative ne les repete pas :

        fenetre centree sur l'infobulle       0,70 / 0,74  contre  0,18   separe bien,
                                              mais suppose de savoir ou elle est
        maximum glissant autour du curseur    0,74         contre  0,90   detruit
        idem, panneaux ecartes                0,65         contre  0,57   12 % de marge

    A comparer aux marges acceptees ailleurs : 36 % pour le crane du chat, 56 % pour le
    mot « Epuise ». Douze pour cent ne decident rien.
    """

    def test_a_false_answer_does_not_mean_a_tooltip_was_seen(self):
        """Le trou assume : `is_depleted` rend False aussi bien quand l'infobulle dit
        « c'est bon » que quand elle n'est jamais apparue. Le runner le comble par
        recoupement, pas par un detecteur."""
        assert not is_depleted(np.zeros((300, 300, 3), dtype=np.uint8))
        assert not is_depleted(_frame("data/pasniveau.png"))

    def test_the_margin_of_the_accepted_reader_is_wide(self):
        """Ce qui distingue le lecteur retenu des trois refuses."""
        vu = depleted_score(_frame("data/epuise.png"))
        pas_vu = max(depleted_score(_frame(n))
                     for n in ("data/pasniveau.png", "capture.png", "bug.png"))
        assert vu - pas_vu > 0.3


class TestTheRedHypothesisIsRefuted:
    """Hypothese mesuree puis rejetee, consignee pour qu'on ne la retente pas : un
    prerequis non rempli s'affiche en rouge dans le jeu, donc la part de rouge vif
    pourrait distinguer « hors niveau » d'« epuise ».

    Elle separe tres bien les deux infobulles ENTRE ELLES -- rapport de trente. Elle ne
    separe pas de la CARTE, et c'est ce qui compte : le critere s'appliquerait a une frame
    entiere, ou les captures de jeu tombent pile entre les deux valeurs.

    Ce test n'ajoute pas un detecteur : il fige la mesure qui l'a refute, pour qu'une
    relecture optimiste du chiffre « rapport de trente » ne relance pas la piste.
    """

    @staticmethod
    def _red_share(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0, 120, 90), np.uint8),
                           np.array((10, 255, 255), np.uint8))
        mask |= cv2.inRange(hsv, np.array((170, 120, 90), np.uint8),
                            np.array((179, 255, 255), np.uint8))
        return float((mask > 0).mean())

    def _read(self, name):
        path = ROOT / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        return cv2.imread(str(path))

    def test_the_two_tooltips_are_far_apart(self):
        epuise = self._red_share(self._read("data/epuise.png"))
        niveau = self._red_share(self._read("data/pasniveau.png"))
        assert niveau > 10 * epuise

    def test_but_the_game_falls_between_them(self):
        """Le fait qui refute : sur une frame entiere, le critere ne decide rien."""
        epuise = self._red_share(self._read("data/epuise.png"))
        niveau = self._red_share(self._read("data/pasniveau.png"))
        for nom in ("capture.png", "hors_combat.png", "combat1920.png"):
            part = self._red_share(self._read(nom))
            assert epuise < part < niveau, f"{nom} : {part:.4f} hors de l'intervalle"
