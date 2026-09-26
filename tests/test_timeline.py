"""Lecture de la timeline : PV proportionnels a la hauteur de barre, ordre d'initiative,
reperage du portrait actif, et rejet des elements rouges parasites de l'UI."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.timeline import (
    BAR_HEIGHT_MIN,
    FULL_BAR_HEIGHT,
    read_timeline,
)
from jev_tactics.state import Team

RED_BGR = (40, 40, 220)          # rouge sature -> dans RED_BANDS
SLOT_X = [1252, 1338, 1415, 1492, 1569]
BASELINE = 130                   # bas commun des portraits normaux
ACTIVE_BOTTOM = 121              # le portrait actif est agrandi


def _frame() -> np.ndarray:
    return np.zeros((1079, 1919, 3), dtype=np.uint8)


def _bar(frame, x: int, height: int, bottom: int = BASELINE, width: int = 7) -> None:
    frame[bottom - height:bottom, x:x + width] = RED_BGR


def test_reads_full_health_bars():
    frame = _frame()
    for x in SLOT_X[:3]:
        _bar(frame, x, int(FULL_BAR_HEIGHT))
    entries = read_timeline(frame)
    assert len(entries) == 3
    assert all(e.hp_ratio == pytest.approx(1.0) for e in entries)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_hp_ratio_tracks_bar_height(fraction):
    frame = _frame()
    _bar(frame, SLOT_X[0], int(FULL_BAR_HEIGHT * fraction))
    _bar(frame, SLOT_X[1], int(FULL_BAR_HEIGHT))
    entries = read_timeline(frame)
    assert entries[0].hp_ratio == pytest.approx(fraction, abs=0.03)


def test_slots_ordered_left_to_right():
    frame = _frame()
    for x in SLOT_X:
        _bar(frame, x, 40)
    entries = read_timeline(frame)
    assert [e.slot for e in entries] == [0, 1, 2, 3, 4]
    assert [e.x for e in entries] == SLOT_X


def test_active_portrait_is_the_offset_one():
    frame = _frame()
    _bar(frame, SLOT_X[0], 50, bottom=ACTIVE_BOTTOM)   # agrandi -> bas decale
    for x in SLOT_X[1:4]:
        _bar(frame, x, 50)
    entries = read_timeline(frame)
    assert [e.is_active for e in entries] == [True, False, False, False]


def test_wide_ui_decoration_rejected():
    """Les icones rouges de l'UI sont plus LARGES que les jauges (9-12 px vs 7)."""
    frame = _frame()
    for x in SLOT_X[:3]:
        _bar(frame, x, 50)
    frame[100:118, 1300:1311] = RED_BGR                # 11x18 : decoration
    assert len(read_timeline(frame)) == 3


def test_close_duplicate_keeps_taller_bar():
    """Deux barres a moins d'un emplacement d'ecart : la plus haute est la vraie."""
    frame = _frame()
    _bar(frame, SLOT_X[0], 50)
    _bar(frame, SLOT_X[0] + 38, 11)                    # parasite colle
    _bar(frame, SLOT_X[1], 50)
    entries = read_timeline(frame)
    assert [e.x for e in entries] == [SLOT_X[0], SLOT_X[1]]


def test_no_timeline_out_of_combat():
    """Hors combat la timeline est absente -> liste vide (signal de gate)."""
    assert read_timeline(_frame()) == []


def test_ignores_red_outside_timeline_roi():
    frame = _frame()
    frame[600:670, 400:407] = RED_BGR                  # une barre, mais sur le plateau
    assert read_timeline(frame) == []


class TestPortraitTeam:
    """L'equipe se lit sur la teinte du fond du portrait (mesure : joueur ~103,
    ennemis ~176). Elle sert au controle croise avec le plateau."""

    def _with_portrait(self, hue: int):
        frame = _frame()
        _bar(frame, SLOT_X[0], 50)
        bottom = BASELINE
        patch = np.full((70, 50, 3), 0, dtype=np.uint8)
        patch[:] = cv2.cvtColor(
            np.uint8([[[hue, 150, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
        frame[bottom - 75:bottom - 5, SLOT_X[0] + 10:SLOT_X[0] + 60] = patch
        return read_timeline(frame)[0]

    def test_ally_hue_detected(self):
        assert self._with_portrait(103).team is Team.ALLY

    def test_enemy_hue_detected(self):
        assert self._with_portrait(176).team is Team.ENEMY

    def test_ambiguous_hue_stays_unknown(self):
        """Mieux vaut None qu'une equipe devinee : le controle croise s'en apercevrait."""
        assert self._with_portrait(60).team is None


def test_hp_ratio_never_exceeds_one():
    frame = _frame()
    _bar(frame, SLOT_X[0], int(FULL_BAR_HEIGHT) + 15)  # barre plus haute que la reference
    assert read_timeline(frame)[0].hp_ratio == 1.0


class TestActivePortrait:
    """Reperage du portrait actif. Le defaut corrige ici inversait la reponse a DEUX
    combattants -- le cas le plus courant en solo -- et faisait attendre le bot pendant
    son propre tour, indefiniment."""

    def test_two_combatants_player_active(self):
        """LE cas constate en jeu : « slot 0 ally <- MOI / slot 1 enemy <- ACTIF » alors
        que le tour etait au joueur. A deux, il n'y a pas de bas majoritaire ; l'ancienne
        regle rendait arbitrairement la plus petite valeur, donc celle de l'actif, et
        l'inversion etait systematique."""
        frame = _frame()
        _bar(frame, SLOT_X[0], int(FULL_BAR_HEIGHT), bottom=ACTIVE_BOTTOM)
        _bar(frame, SLOT_X[1], int(FULL_BAR_HEIGHT))
        entries = read_timeline(frame)
        assert [e.is_active for e in entries] == [True, False]

    def test_two_combatants_enemy_active(self):
        frame = _frame()
        _bar(frame, SLOT_X[0], int(FULL_BAR_HEIGHT))
        _bar(frame, SLOT_X[1], int(FULL_BAR_HEIGHT), bottom=ACTIVE_BOTTOM)
        entries = read_timeline(frame)
        assert [e.is_active for e in entries] == [False, True]

    def test_at_most_one_active(self):
        """Deux portraits actifs n'ont aucun sens : `is_our_turn` en deduirait un tour
        partage. Le plus haut tranche."""
        frame = _frame()
        _bar(frame, SLOT_X[0], int(FULL_BAR_HEIGHT), bottom=ACTIVE_BOTTOM)
        _bar(frame, SLOT_X[1], int(FULL_BAR_HEIGHT), bottom=ACTIVE_BOTTOM - 4)
        _bar(frame, SLOT_X[2], int(FULL_BAR_HEIGHT))
        assert sum(e.is_active for e in read_timeline(frame)) == 1

    def test_none_active_when_all_aligned(self):
        """Entre deux tours, aucun portrait n'est agrandi : ne rien inventer."""
        frame = _frame()
        for x in SLOT_X[:3]:
            _bar(frame, x, int(FULL_BAR_HEIGHT))
        assert not any(e.is_active for e in read_timeline(frame))

    def test_single_combatant_is_not_active_by_default(self):
        """Un seul portrait ne fournit aucune reference : ne pas le declarer actif sur
        rien."""
        frame = _frame()
        _bar(frame, SLOT_X[0], int(FULL_BAR_HEIGHT))
        assert not any(e.is_active for e in read_timeline(frame))


class TestPortraitTeamOnRealCaptures:
    """Non-regression sur DONNEES REELLES.

    Le defaut corrige ici : l'illustration de la creature occupe le centre du portrait et
    dominait la mediane de teinte. Sur `tacle.png`, le Chef de Guerre Bouftou est
    blanc-creme, ce qui tirait son portrait a H=110 -- en plein dans la bande ALLIE. La
    timeline annoncait deux allies, la reconciliation avec le plateau se desactivait
    (elle compte par equipe), et l'identification du joueur devenait un tirage au sort.

    Ces captures sont locales (non versionnees) : le test se saute si elles manquent
    plutot que d'echouer chez quelqu'un d'autre."""

    @pytest.mark.parametrize(("name", "expected"), [
        ("tacle.png", ["ally", "enemy"]),
        ("combat1.png", ["ally", "enemy", "enemy"]),
        ("bug.png", ["ally", "enemy"]),
    ])
    def test_teams_are_read_correctly(self, name, expected):
        path = Path(__file__).resolve().parents[1] / name
        if not path.exists():
            pytest.skip(f"{name} absent (capture locale)")
        teams = [e.team.value if e.team else "?" for e in read_timeline(cv2.imread(str(path)))]
        assert teams == expected

    def test_a_pale_creature_does_not_read_as_an_ally(self):
        """Le cas precis, isole : une creature claire sur fond ennemi."""
        path = Path(__file__).resolve().parents[1] / "tacle.png"
        if not path.exists():
            pytest.skip("tacle.png absent (capture locale)")
        entries = read_timeline(cv2.imread(str(path)))
        assert sum(1 for e in entries if e.team is Team.ALLY) == 1


class TestTheBarHeightFloorIsDefended:
    """Une barre COURTE est un combattant presque mort, et le seuil l'exclut : sur
    20260807-222701.png la barre du monstre fait 4 px, soit ~6 % de vie. Le bot perd donc
    de vue un ennemi au moment precis ou il faudrait l'achever.

    J'allais baisser le seuil. Mesure sur 30 captures, d'abord :

        seuil    comptages justes   un seul actif   equipes lues
          10          27/30              29              28
           8          27/30              30               2
           6          24/30              29               2
           4          23/30              30               2

    La lecture des EQUIPES s'effondre de 28 a 2 des qu'on descend a 8 : les barres courtes
    admises sont majoritairement du decor et de l'ATH, et leur teinte brouille
    l'attribution d'equipe de TOUTE la timeline. Le remede coute cinq fois le defaut.

    La perte reste bornee : l'ennemi disparait de la TIMELINE, pas du plateau -- ses
    marqueurs de case restent detectes, et c'est eux que vise le solveur.
    """

    def test_the_floor_stays_where_it_was_measured(self):
        assert BAR_HEIGHT_MIN == 10

    def test_a_nearly_dead_bar_is_below_it(self):
        """Le defaut assume, chiffre : 4 px sur 70 pour une barre pleine, soit 6 % de vie."""
        assert 4 < BAR_HEIGHT_MIN
        assert 4 / FULL_BAR_HEIGHT < 0.10


class TestTheTimelineIsFoundOutsideItsHardcodedRectangle:
    """`TIMELINE_ROI` est un rectangle en dur, et il a rendu le bot aveugle en combat.

    Releve sur des combats a quatre et plus, il couvre x de 1150 a 1700. Or la timeline
    n'occupe que la largeur de ses combattants : a DEUX -- ce qui est la norme en zone de
    depart -- elle tombe a x=937..1023, soit 113 px avant le bord du rectangle, et sa
    premiere barre commence a y=27 quand le rectangle debute a y=30. Les deux axes
    coupaient.

    La panne n'etait pas une erreur mais un SILENCE : `read_timeline` rendait [], donc
    `classify_screen` rendait INCONNU, donc l'orchestrateur envoyait sa touche de dernier
    recours et s'arretait -- en plein combat, tour apres tour. C'est mot pour mot ce que
    le docstring de `locate_timeline` annoncait, sans que rien ne le branche.
    """

    COMBAT = "20260818-205954-ecran-inconnu.png"

    def _failure(self, nom):
        path = Path(__file__).resolve().parents[1] / "data" / "failures" / nom
        if not path.exists():
            pytest.skip(f"{nom} absent")
        return cv2.imread(str(path))

    def test_a_two_fighter_timeline_is_read(self):
        """La capture qui a fait tomber la session : deux combattants, hors rectangle."""
        entries = read_timeline(self._failure(self.COMBAT))
        assert [e.x for e in entries] == [937, 1023]
        assert [e.team for e in entries] == [Team.ALLY, Team.ENEMY]

    def test_the_active_portrait_is_still_identified(self):
        """Le repli ne doit pas rendre une timeline amputee de ce qui la rend utile.

        Le portrait actif se repere au decalage vertical (mesure ici : 9 px, 27 contre
        36). Une timeline lue sans lui ferait jouer le bot au tour d'un autre.
        """
        entries = read_timeline(self._failure(self.COMBAT))
        assert [e.is_active for e in entries] == [True, False]

    def test_the_bot_own_control_panel_is_not_a_fight(self):
        """CONTRE-EPREUVE, et le faux positif le plus dangereux du lot.

        Le panneau de controle du bot, pose par-dessus le jeu, offre un fond bleu qui
        donne deux barres de 10 px a y=217, classees « alliees » par la teinte. Chercher
        dans tout le bandeau les acceptait ; c'est ce qui impose de garder la fenetre
        VERTICALE de la timeline. Les prendre pour un combat ferait jouer un tour dans
        une fenetre d'interface.
        """
        assert read_timeline(self._failure("20260818-210621-ecran-inconnu.png")) == []

    def test_a_map_is_not_a_fight(self):
        """Une carte porte du decor rouge en haut : rien de tout cela n'est une timeline."""
        assert read_timeline(self._failure("20260818-210845-ecran-inconnu.png")) == []

    def test_the_team_filter_does_not_apply_to_the_trusted_rectangle(self):
        """Le filtre d'equipe est le prix du REPLI, pas une regle generale.

        `_portrait_team` rend None des que le fond n'est pas net, et cela arrive sur de
        VRAIES barres : sur cette capture, deux barres dans le rectangle et aucune equipe
        concluante. L'appliquer partout perdrait ce combat -- le rectangle est une preuve
        de position suffisante, la bande entiere ne l'est pas.
        """
        path = Path(__file__).resolve().parents[1] / "data" / "runs" / "20260807-222701.png"
        if not path.exists():
            pytest.skip("20260807-222701.png absent")
        entries = read_timeline(cv2.imread(str(path)))
        assert len(entries) == 2
        # Le second combattant n'a pas d'equipe concluante. Le filtre du repli l'aurait
        # ecarte, et avec lui la moitie de la timeline -- d'ou une lecture a un seul
        # combattant, sous le minimum, donc « hors combat » sur un combat en cours.
        assert entries[1].team is None, "capture temoin : le second n'a pas d'equipe nette"
