"""Taux de remplissage de l'inventaire, affiche en haut a gauche.

Ce badge a longtemps ete tenu pour inexistant : `--pods-full` a meme ete RETIRE comme
inerte, faute de lecteur, parce qu'on croyait devoir ouvrir l'inventaire. Il est en fait
affiche en permanence hors combat. La difference est concrete : elle separe « arreter
apres six recoltes ratees » de « aller vider avant que ca coince »."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.digits import DigitTemplates
from jev_tactics.perception.pods import (
    BADGE_DARKNESS,
    PODS_ROI,
    blind_spots,
    diagnose_pods,
    load_pods_templates,
    read_pods_ratio,
)

ROOT = Path(__file__).resolve().parents[1]


def _capture(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} absent (capture locale, non versionnee)")
    return cv2.imread(str(path))


class TestOnTheRealCapture:
    def test_it_reads_the_percentage(self):
        assert diagnose_pods(_capture("hors_combat.png")).ratio == pytest.approx(0.59)

    def test_read_pods_ratio_keeps_the_short_contract(self):
        assert read_pods_ratio(_capture("hors_combat.png")) == pytest.approx(0.59)


class TestTheBadgeOnlyExistsOutOfCombat:
    """MESURE sur sept captures : la pastille n'apparait que sur la seule prise hors
    combat. Sans consequence -- la recolte, qui en a besoin, tourne hors combat -- mais il
    fallait le savoir avant de croire a une panne de lecture."""

    @pytest.mark.parametrize("name", ["combat1.png", "tacle.png", "bug4.png",
                                      "combat1920.png"])
    def test_in_combat_it_says_the_badge_is_ABSENT(self, name):
        """Le fond SOMBRE tranche avant toute lecture de chiffre. Sans lui, le texte
        « - Niveau 30 » qui passe dans la meme zone etait pris pour un taux puis rejete
        comme illisible : un refus juste, pour une raison fausse."""
        reading = diagnose_pods(_capture(name))
        assert reading.ratio is None and "pastille absente" in reading.reason

    def test_the_darkness_gap_is_wide(self):
        """0,50 quand elle est affichee, 0,01 a 0,02 sinon. Le seuil tombe dans un
        intervalle vide -- rare, et confortable."""
        x0, y0, x1, y1 = PODS_ROI
        shown = _capture("hors_combat.png")[y0:y1, x0:x1]
        hidden = _capture("combat1.png")[y0:y1, x0:x1]
        assert float((shown.max(axis=2) < 70).mean()) > BADGE_DARKNESS * 2
        assert float((hidden.max(axis=2) < 70).mean()) < BADGE_DARKNESS / 4


class TestRefusalIsTheContract:
    """Une lecture fausse ferait croire l'inventaire plein et arreterait la session, ou
    l'inverse -- laisser recolter dans le vide. Refuser vaut mieux dans les deux sens."""

    def test_uncovered_digits_are_refused_and_NAMED(self):
        covered = load_pods_templates().covered_digits()
        assert covered, "gabarits vides"
        assert covered <= set(range(10))

    def test_a_frame_too_small_is_safe(self):
        assert diagnose_pods(np.zeros((40, 40, 3), np.uint8)).ratio is None

    def test_a_blank_frame_reads_nothing(self):
        assert read_pods_ratio(np.zeros((1080, 1920, 3), np.uint8)) is None

    def test_an_impossible_percentage_is_rejected(self, monkeypatch):
        """Plus de 100% est une lecture fausse, pas un inventaire tres plein. La rendre
        ferait arreter la session pour rien."""
        from jev_tactics.perception import pods

        monkeypatch.setattr(pods, "_darkness", lambda _f: 1.0)
        monkeypatch.setattr(pods, "glyphs_of", lambda _f: [(0, 0, 1, 1, None)] * 3)
        monkeypatch.setattr(pods, "normalize_glyph", lambda _p: None)

        class _AlwaysNine:
            def classify(self, _glyph):
                return 9, 1.0

            def covered_digits(self):
                return set(range(10))

        reading = pods.diagnose_pods(np.zeros((1080, 1920, 3), np.uint8), _AlwaysNine())
        assert reading.ratio is None and "impossible" in reading.reason


class TestBlindSpots:
    """Ce que la lecture ne verra PAS -- et pourquoi la liste des chiffres manquants ne
    suffit pas a le dire.

    « Manquants : [0, 1, 3, 4, 7, 8] » est exact et sans portee. Ce qui compte est la
    BANDE DE DECISION : les taux au-dela du seuil, seuls capables de declencher l'arret.
    Avec les gabarits construits sur deux captures reelles et le seuil par defaut, quatre
    valeurs sur onze sont lisibles -- le lecteur est aveugle la ou il sert.
    """

    def test_the_default_band_is_mostly_blind(self):
        blind = blind_spots(0.90)
        assert len(blind) == 7, blind
        assert set(blind).isdisjoint({92, 95, 96, 99})

    def test_a_full_alphabet_sees_everything(self):
        complete = DigitTemplates(
            exemplars=np.zeros((10, 16, 8), dtype=np.float32),
            digits=np.arange(10, dtype=np.int64))
        assert blind_spots(0.0, templates=complete) == []

    def test_one_hundred_is_in_the_band(self):
        """100% est le taux le plus important de la bande et le plus facile a manquer :
        trois chiffres la ou tous les autres en ont deux, donc trois occasions de tomber
        sur un gabarit absent. Il est bien dans la bande, et aujourd'hui aveugle."""
        assert 100 in blind_spots(0.90)

    def test_raising_the_threshold_narrows_the_band(self):
        assert set(blind_spots(0.95)) <= set(blind_spots(0.90))
