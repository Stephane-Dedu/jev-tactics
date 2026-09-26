"""Lecture de la part de PV restants, et ce qu'elle sert a decider.

Un bot qui chasse a 10 % de PV meurt, se releve en etat de faiblesse, rechasse et
remeurt : la session ne produit rien et coute de l'energie. Encore faut-il que la lecture
soit sure -- un ratio faux dans un sens fait refuser tous les combats, dans l'autre les
fait tous accepter."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.ui import read_hp_ratio, read_ui

ROOT = Path(__file__).resolve().parents[1]

# Valeurs relevees a l'oeil sur les captures reelles. TOUTES EN COMBAT : aucune capture
# hors combat n'existe, et c'est pourtant hors combat que la question se pose puisque
# c'est la qu'on decide d'engager. La limite est consignee, pas contournee.
REAL = [
    ("combat1.png", 1.00),
    ("combat1920.png", 1.00),
    ("bug4.png", 1549 / 1656),
    ("bugcarreblanc.png", 1489 / 1646),
    ("tacle.png", 1146 / 1637),
    ("bug.png", 892 / 1645),
]


@pytest.mark.parametrize(("name", "expected"), REAL)
def test_reads_real_captures(name, expected):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} absent (capture locale, non versionnee)")
    assert read_hp_ratio(cv2.imread(str(path))) == pytest.approx(expected, abs=0.01)


class TestRefusalIsTheContract:
    """None quand la lecture echoue, JAMAIS une valeur de repli.

    Un repli a 1.0 ferait engager un personnage mourant ; un repli a 0.0 desactiverait la
    chasse pour toujours. Les deux sont pires que « on ne sait pas »."""

    def test_a_blank_frame_reads_nothing(self):
        assert read_hp_ratio(np.zeros((1080, 1920, 3), np.uint8)) is None

    def test_a_frame_too_small_is_safe(self):
        """ROI hors capture : la lecture doit rendre None, pas lever."""
        assert read_hp_ratio(np.zeros((40, 40, 3), np.uint8)) is None

    def test_pv_above_max_is_rejected(self, monkeypatch):
        """Deja rencontre en jeu. Un ratio superieur a 1 laisserait croire a une pleine
        sante alors que les DEUX lectures sont fausses."""
        from jev_tactics.perception import ui

        monkeypatch.setattr(ui, "read_ui",
                            lambda _f: ui.UiReading(pv=2000, pv_max=1600))
        assert ui.read_hp_ratio(np.zeros((1080, 1920, 3), np.uint8)) is None

    def test_a_zero_maximum_is_rejected(self, monkeypatch):
        """Garde contre la division par zero, mais surtout : un maximum nul est une
        lecture ratee, pas un personnage sans PV."""
        from jev_tactics.perception import ui

        monkeypatch.setattr(ui, "read_ui", lambda _f: ui.UiReading(pv=0, pv_max=0))
        assert ui.read_hp_ratio(np.zeros((10, 10, 3), np.uint8)) is None


class TestTheTwoHealthReadersAgree:
    """Le garde-fou « ne pas engager a bas PV » repose entierement sur ce lecteur, et sa
    panne serait SILENCIEUSE : s'il rendait None, `too_hurt` ne serait jamais vrai et le
    bot chasserait a 10 % de PV. Rien ne le surveillait sur l'ensemble des captures.

    CE QUE CE TEST PROUVE, ET CE QU'IL AFFIRMAIT A TORT. Il annoncait croiser DEUX
    LECTEURS INDEPENDANTS -- « celui-ci mesure la longueur de la jauge, `read_ui` lit les
    chiffres par gabarits » -- et concluait « ecart maximal 0,000 » sur 31 captures.

    Il n'y a plus qu'un lecteur. `read_hp_ratio` appelle `read_ui` et divise `pv` par
    `pv_max` ; ce test recalcule la meme division a cote. L'ecart nul n'est donc pas une
    concordance, c'est une identite : les deux membres sont la meme expression. Le lecteur
    de jauge a disparu du code sans que le docstring ni ce test ne le sachent.

    CE QUI RESTE GARDE, et ce n'est pas rien : que `read_hp_ratio` ne s'ecarte pas de
    `pv / pv_max` (une inversion, un facteur, un plafonnement mal place seraient vus), et
    que les deux repondent sur toutes les captures.

    CE QUI N'EST PLUS GARDE, et il faut le lire avant de faire confiance a ce garde-fou :
    aucune SECONDE source ne confirme les chiffres. Sur une disposition ou `pv_max` n'est
    pas affiche -- constate le 19/08, un ATH ou les PV tiennent dans un coeur, « 69 », sans
    maximum -- la lecture rend None, le garde-fou des PV est aveugle, et rien ne prend le
    relais. C'est exactement la panne SILENCIEUSE que l'en-tete de cette classe redoute.
    """

    TOLERANCE = 0.02

    def _captures(self):
        racine = Path(__file__).resolve().parents[1]
        fichiers = sorted((racine / "data" / "runs").glob("*.png"))
        fichiers += [racine / nom for nom in ("capture.png", "hors_combat.png",
                                              "combat1920.png", "combat1.png",
                                              "bug.png", "bug4.png", "tacle.png")]
        presents = [f for f in fichiers if f.exists()]
        if len(presents) < 20:
            pytest.skip("captures absentes")
        return presents

    def test_every_capture_is_readable_by_both(self):
        muettes = []
        for chemin in self._captures():
            frame = cv2.imread(str(chemin))
            lecture = read_ui(frame)
            if read_hp_ratio(frame) is None or not (lecture.pv and lecture.pv_max):
                muettes.append(chemin.name)
        assert muettes == [], f"PV illisibles : {muettes}"

    def test_the_bar_and_the_digits_never_contradict(self):
        for chemin in self._captures():
            frame = cv2.imread(str(chemin))
            lecture, barre = read_ui(frame), read_hp_ratio(frame)
            if not (lecture.pv and lecture.pv_max) or barre is None:
                continue
            assert abs(lecture.pv / lecture.pv_max - barre) <= self.TOLERANCE, (
                f"{chemin.name} : chiffres {lecture.pv}/{lecture.pv_max}, jauge {barre}")

    def test_a_wounded_character_reads_below_one(self):
        """Contre-epreuve indispensable : rendre 1,0 partout est aussi ce que ferait un
        lecteur casse, et toutes les captures de session montrent un personnage intact.
        tacle.png est la seule blessee du depot -- 1146/1637."""
        chemin = Path(__file__).resolve().parents[1] / "tacle.png"
        if not chemin.exists():
            pytest.skip("tacle.png absent")
        assert read_hp_ratio(cv2.imread(str(chemin))) == pytest.approx(0.70, abs=0.02)
