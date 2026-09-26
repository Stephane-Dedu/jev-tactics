"""Classification d'ecran : la question que le bot ne se posait jamais."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.screen import ScreenState, classify_screen

ROOT = Path(__file__).resolve().parents[1]
SESSION = sorted((ROOT / "data" / "runs").glob("*.png"))
REFERENCE = sorted(ROOT.glob("*.png"))


class TestEveryKnownCaptureIsRecognised:
    """Mesure sur les 32 captures du depot : 30 COMBAT, 2 CARTE, ZERO inconnu.

    Les deux discriminants sont ceux dont la fiabilite est deja etablie, et non de
    nouveaux seuils : la timeline n'existe qu'en combat, et les coordonnees se lisent
    24/24 en session comme 8/8 en reference.
    """

    def test_no_known_capture_is_unknown(self):
        if not SESSION and not REFERENCE:
            pytest.skip("aucune capture")
        unknown = [p.name for p in SESSION + REFERENCE
                   if classify_screen(cv2.imread(str(p))) is ScreenState.UNKNOWN]
        assert unknown == []

    def test_session_captures_are_combat(self):
        if not SESSION:
            pytest.skip("captures de session absentes")
        states = {classify_screen(cv2.imread(str(p))) for p in SESSION}
        assert states == {ScreenState.COMBAT}

    @pytest.mark.parametrize("name", ["capture.png", "hors_combat.png"])
    def test_the_two_out_of_combat_captures_are_maps(self, name):
        path = ROOT / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        assert classify_screen(cv2.imread(str(path))) is ScreenState.MAP


class TestUnknownIsAState:
    """`INCONNU` doit se declencher, sinon il ne sert a rien. Le verifier demande quelque
    chose qui N'EST PAS un ecran de jeu -- les deux recadrages d'infobulle fournis par
    l'utilisateur en sont, et ils tombent bien dedans."""

    @pytest.mark.parametrize("name", ["data/epuise.png", "data/pasniveau.png"])
    def test_a_crop_is_not_a_screen(self, name):
        path = ROOT / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        assert classify_screen(cv2.imread(str(path))) is ScreenState.UNKNOWN

    def test_a_blank_frame_is_unknown(self):
        assert classify_screen(np.zeros((1080, 1920, 3), np.uint8)) is ScreenState.UNKNOWN

    def test_a_tiny_frame_does_not_raise(self):
        """Jamais d'exception : un classificateur qui leve sur une frame inattendue est
        exactement ce qu'il est cense remplacer."""
        assert classify_screen(np.zeros((4, 4, 3), np.uint8)) is ScreenState.UNKNOWN


class TestCombatWinsOverMap:
    """L'ORDRE des deux tests compte, et se mesure. Les coordonnees sont lisibles EN
    COMBAT AUSSI -- cinq captures de combat sur six les donnent -- donc les tester en
    premier classerait un combat comme une carte, et le bot recolterait pendant qu'on le
    frappe."""

    def test_a_combat_capture_reads_its_coordinates_too(self):
        from jev_tactics.perception.coordinates import diagnose_position

        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        frame = cv2.imread(str(path))
        assert diagnose_position(frame).position is not None
        assert classify_screen(frame) is ScreenState.COMBAT
