"""Lecture des morts annoncees dans le chat.

Le chat est la derniere grande source d'information que le bot n'exploitait pas, et celle
que l'utilisateur avait designee en premier : « on voit les degats et les ennemis morts »."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.deaths import (
    MIN_SCORE,
    death_marks,
    load_glyph,
    new_death_marks,
)

ROOT = Path(__file__).resolve().parents[1]


def _frame(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} absent")
    return cv2.imread(str(path))


class TestTheGlyphIsTheInvariant:
    """Deux captures portent une mort, et elles ne la formulent PAS pareil :

        capture.png    « Timongouste : -212 PV  (☠ mort). »
        bug.png        « ☠ Flammeche Eau est mort ! »

    Le libelle change, le GLYPHE non. Viser le crane plutot que le mot evite d'avoir a
    couvrir toutes les tournures du jeu, et resiste au fait qu'un nom de monstre peut
    contenir n'importe quoi.
    """

    def test_both_phrasings_are_found(self):
        assert len(death_marks(_frame("capture.png"))) == 2
        assert len(death_marks(_frame("bug.png"))) == 1

    def test_the_positions_match_what_the_eye_reads(self):
        marks = death_marks(_frame("capture.png"))
        for expected in ((269, 774), (262, 824)):
            assert any(abs(x - expected[0]) <= 3 and abs(y - expected[1]) <= 3
                       for x, y in marks), f"{expected} absent de {marks}"

    @pytest.mark.parametrize("name", ["bug4.png", "bugcarreblanc.png", "combat1.png",
                                      "combat1920.png", "hors_combat.png", "tacle.png"])
    def test_captures_without_a_death_report_none(self, name):
        """Le pendant : un detecteur qui voit des morts partout ne vaudrait rien. Ces six
        captures plafonnent a 0,710 de correlation, pour un seuil a 0,85."""
        assert death_marks(_frame(name)) == []

    def test_the_threshold_sits_in_an_empty_interval(self):
        """0,710 (le plus haut sans mort) contre 0,965 (la plus basse avec). Le seuil s'y
        pose sans arbitrage -- c'est la seule justification acceptable pour un nombre."""
        assert 0.71 < MIN_SCORE < 0.965


class TestRefusals:
    def test_a_frame_smaller_than_the_glyph_is_safe(self):
        assert death_marks(np.zeros((4, 4, 3), dtype=np.uint8)) == []

    def test_a_blank_frame_finds_nothing(self):
        assert death_marks(np.zeros((300, 300, 3), dtype=np.uint8)) == []

    def test_the_glyph_is_small_enough_to_scan_a_full_frame(self):
        """Cherche sur TOUTE la frame plutot que dans une boite : le panneau de chat se
        deplace au gre du joueur, et un rectangle fixe aurait la fragilite de ceux qu'on
        vient de retirer ailleurs."""
        assert load_glyph().shape == (15, 16)


class TestOnlyNewDeathsCount:
    """Une marque VISIBLE ne dit pas qu'une mort vient d'avoir lieu.

    Le chat garde les anciennes a l'ecran : compter les cranes ferait croire a une
    hecatombe permanente, et le bot changerait de cible a chaque tour sur la foi d'une mort
    d'il y a cinq minutes. Ce qui compte est l'APPARITION.

    Le chat monte d'une ligne par message -- hauteur mesuree sur capture.png : 17 px, les
    ecarts releves allant de 13 a 23 selon les majuscules et les accents. Une ancienne
    marque change donc de position sans etre nouvelle. On estime ce defilement par
    correlation : sur le chat reel defile artificiellement, 0, 17 et 34 px sont retrouves
    EXACTEMENT, avec une correlation de 1,0.
    """

    def test_an_unchanged_chat_has_no_new_death(self):
        frame = _frame("capture.png")
        assert new_death_marks(frame, frame) == []

    def test_a_scrolled_chat_has_no_new_death(self):
        """Le cas qui justifie tout le mecanisme : les deux marques ont BOUGE de 17 px, et
        aucune n'est nouvelle. Une comparaison de positions brutes en aurait vu deux."""
        frame = _frame("capture.png")
        assert new_death_marks(frame, np.roll(frame, -17, axis=0)) == []

    def test_a_fresh_death_is_reported_where_it_appeared(self):
        frame = _frame("capture.png")
        after = frame.copy()
        after[900:915, 261:277] = frame[767:782, 261:277]
        fresh = new_death_marks(frame, after)
        assert len(fresh) == 1
        assert abs(fresh[0][1] - 907) <= 3

    def test_everything_is_new_when_nothing_was_there_before(self):
        """Sans marque dans la frame precedente, aucune estimation n'est necessaire : tout
        ce qu'on voit vient d'arriver."""
        frame = _frame("capture.png")
        before = frame.copy()
        before[760:840, 250:290] = frame[700:780, 250:290]   # on efface les deux cranes
        assert len(new_death_marks(before, frame)) == 2

    def test_a_chat_without_any_death_stays_silent(self):
        frame = _frame("combat1.png")
        assert new_death_marks(frame, frame) == []
