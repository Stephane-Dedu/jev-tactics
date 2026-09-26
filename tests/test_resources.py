"""Detection de ressources par difference de surbrillance (touche Y).

Le jeu designe lui-meme les elements interactifs : on lit sa reponse au lieu
d'apprendre a reconnaitre chaque sprite. Ces tests verifient surtout la ROBUSTESSE au
bruit -- entre les deux captures, le decor bouge."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.resources import (
    DIFF_THRESHOLD,
    MIN_AREA,
    SUSPICIOUS_RESOURCE_COUNT,
    ResourceSpot,
    detect_resources,
    highlight_mask,
    nearest_resource,
)
from jev_tactics.perception.ui import UI_ZONES

ROOT = Path(__file__).resolve().parents[1]

SIZE = (600, 900, 3)


def _scene(seed: int = 0):
    """Fond texture : la detection ne doit pas dependre d'un fond uni."""
    rng = np.random.default_rng(seed)
    return rng.integers(40, 90, size=SIZE, dtype=np.uint8)


def _light_up(frame, x, y, radius=22, boost=90):
    """Simule la surbrillance d'un element interactif."""
    lit = frame.copy()
    cv2.circle(lit, (x, y), radius, (255, 255, 255), -1)
    blended = cv2.addWeighted(frame, 1.0, lit, 0.0, 0)
    cv2.circle(blended, (x, y), radius,
               tuple(int(min(255, c + boost)) for c in (90, 90, 90)), -1)
    return blended


class TestHighlightMask:
    def test_nothing_changed_gives_empty_mask(self):
        frame = _scene()
        assert highlight_mask(frame, frame).sum() == 0

    def test_lit_region_is_captured(self):
        before = _scene()
        after = _light_up(before, 300, 200)
        assert highlight_mask(before, after)[200, 300] > 0

    def test_only_brightening_counts(self):
        """La surbrillance AJOUTE de la lumiere : un assombrissement n'est pas une
        ressource. Ce choix ecarte d'emblee la moitie des changements parasites."""
        before = _scene()
        after = before.copy()
        cv2.circle(after, (300, 200), 22, (5, 5, 5), -1)      # nettement plus sombre
        assert highlight_mask(before, after).sum() == 0

    def test_mismatched_sizes_rejected(self):
        with pytest.raises(ValueError):
            highlight_mask(_scene(), np.zeros((10, 10, 3), dtype=np.uint8))

    def test_threshold_is_respected(self):
        before = _scene()
        after = np.clip(before.astype(np.int16) + DIFF_THRESHOLD // 3, 0, 255).astype(
            np.uint8)
        assert highlight_mask(before, after).sum() == 0      # variation trop faible


class TestDetectResources:
    def test_finds_each_lit_element(self):
        before = _scene()
        after = before
        for x, y in [(200, 150), (500, 300), (750, 450)]:
            after = _light_up(after, x, y)
        assert len(detect_resources(before, after)) == 3

    def test_positions_match_the_elements(self):
        before = _scene()
        after = _light_up(before, 400, 250)
        spot = detect_resources(before, after)[0]
        assert abs(spot.x - 400) <= 3 and abs(spot.y - 250) <= 3

    def test_nothing_when_no_highlight(self):
        frame = _scene()
        assert detect_resources(frame, frame) == []

    def test_tiny_changes_ignored(self):
        """Une animation de feuillage produit de petits changements : pas une ressource."""
        before = _scene()
        after = _light_up(before, 300, 200, radius=3)
        assert detect_resources(before, after) == []

    def test_thin_streak_ignored(self):
        """Une trainee fine (bordure qui s'eclaire) n'est pas un blob compact."""
        before = _scene()
        after = before.copy()
        after[200:206, 100:700] = 220
        assert detect_resources(before, after) == []

    def test_survives_background_motion(self):
        """Entre les deux captures le decor bouge : la ressource doit ressortir quand
        meme."""
        before = _scene()
        after = _scene(seed=1)                       # tout le fond a change
        after = _light_up(after, 400, 300, radius=26)
        spots = detect_resources(before, after)
        assert any(abs(s.x - 400) <= 6 and abs(s.y - 300) <= 6 for s in spots)

    def test_excluded_region_is_ignored(self):
        """L'UI change pour ses propres raisons : on doit pouvoir l'exclure."""
        before = _scene()
        after = _light_up(before, 120, 120)
        assert detect_resources(before, after, exclude=((0, 0, 300, 300),)) == []

    def test_sorted_by_area_descending(self):
        before = _scene()
        after = _light_up(_light_up(before, 250, 200, radius=16), 600, 350, radius=30)
        areas = [s.area for s in detect_resources(before, after)]
        assert areas == sorted(areas, reverse=True)

    def test_reported_area_is_plausible(self):
        before = _scene()
        after = _light_up(before, 400, 300, radius=25)
        spot = detect_resources(before, after)[0]
        assert spot.area >= MIN_AREA
        assert 30 <= spot.width <= 70 and 30 <= spot.height <= 70


class TestNearest:
    SPOTS = [ResourceSpot(x=100, y=100, area=500, width=20, height=20),
             ResourceSpot(x=800, y=500, area=900, width=30, height=30)]

    def test_picks_the_closest(self):
        assert nearest_resource(self.SPOTS, (150, 120)).x == 100
        assert nearest_resource(self.SPOTS, (700, 480)).x == 800

    def test_none_when_empty(self):
        assert nearest_resource([], (0, 0)) is None

    def test_ignores_area_when_choosing(self):
        """La plus proche, pas la plus grosse : marcher coute du temps."""
        assert nearest_resource(self.SPOTS, (110, 110)).area == 500


class TestOnRealCaptures:
    """Ce fichier ne contenait AUCUN imread : le detecteur de ressources -- le coeur de
    « trouver les ressources » -- n'avait jamais tourne sur une capture reelle. Ses tests
    construisaient des blobs synthetiques si francs qu'ils survivaient a n'importe quel
    seuil, ce qui explique que l'audit des constantes ait pu doubler DIFF_THRESHOLD et
    MIN_AREA sans faire rougir quoi que ce soit.

    Le depot ne contient pas de paire eteint/allume (deux captures de la MEME carte, l'une
    avant Y, l'autre apres) : la mesure qui compte vraiment reste donc a faire. Ce qui suit
    ancre les deux bouts accessibles sans elle.
    """

    def test_a_real_frame_against_itself_finds_nothing(self):
        """Le bout sur : sans changement, rien. Mesure a 0 sur les deux cartes du depot."""
        for nom in ("capture.png", "hors_combat.png"):
            chemin = ROOT / nom
            if not chemin.exists():
                continue
            frame = cv2.imread(str(chemin))
            assert detect_resources(frame, frame, exclude=UI_ZONES) == []

    def test_a_changed_scene_trips_the_implausibility_signal(self):
        """L'autre bout : le detecteur est un simple differentiel, donc TOUT changement de
        scene l'inonde. Mesure : 111 blobs entre deux cartes differentes, contre un seuil
        d'invraisemblance a 40.

        C'est la premiere fois que ce seuil est confronte a des donnees reelles, et cela
        confirme sa place : bien au-dessus d'un scan normal, bien en dessous d'un changement
        de scene. Cela confirme aussi que ce qui protege la recolte n'est pas ce seuil mais
        la CAUSALITE de la sonde -- deux bascules de la touche, et l'on ne garde que ce qui
        s'allume aux deux.
        """
        a, b = ROOT / "capture.png", ROOT / "hors_combat.png"
        if not (a.exists() and b.exists()):
            pytest.skip("captures absentes")
        trouves = detect_resources(cv2.imread(str(a)), cv2.imread(str(b)), exclude=UI_ZONES)
        assert len(trouves) > SUSPICIOUS_RESOURCE_COUNT
