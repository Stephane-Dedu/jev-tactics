"""LE SECOND ATH, celui du personnage reellement joue -- et que rien ne lisait.

Toutes les captures de reference du depot (`combat1.png`, `combat1920.png`, `capture.png`,
`hors_combat.png`, `tacle.png`) montrent LE MEME personnage sur LE MEME ATH : un Sacrieur a
1632 PV, orbe affichant « courant SUR maximum », barre de sorts a trois rangees, grand
plateau. Les 2384 tests passaient donc au vert pendant que le bot etait aveugle en combat
reel, parce qu'aucun d'eux ne regardait autre chose.

Les captures du 19/08 -- conservees par le bot lui-meme quand il a renonce -- montrent
l'autre ATH : niveau 4 a Incarnam, 88 PV dans un orbe SANS maximum affiche, barre a UNE
rangee, petit plateau clair. Trois etages y echouaient d'un coup :

    lecture UI       pv=None pv_max=None          (l'orbe a un seul nombre)
    barre de sorts   « 14 prets, 16 en recharge » (24 emplacements lus dans le decor)
    grille           None                         (« aucune grille detectee — hors combat ? »)

Sans grille il n'y a ni plateau, ni entites, ni zone de deplacement, ni plan : le bot
declarait « combat termine sans un seul tour joue » et abandonnait des combats REELS, sur
lesquels c'etait son tour et le bouton FIN DE TOUR etait allume.

Ces tests fixent la lecture de ce second ATH. Ils sont ecrits sur les frames que le bot a
lui-meme conservees : c'est la vraie verite terrain du projet.
"""

import pathlib

import cv2
import pytest

from jev_tactics.calibration.grid import estimate_grid
from jev_tactics.perception.spellbar import SlotState, bar_layout, read_spell_slots
from jev_tactics.perception.timeline import read_timeline
from jev_tactics.perception.ui import read_hp_gauge, read_hp_ratio, read_ui

RACINE = pathlib.Path(__file__).resolve().parents[1]
ECHECS = RACINE / "data" / "failures"

# Les trois captures de COMBAT REEL du 19/08. La timeline y est pleine et le combat
# incontestable ; c'est la boucle de combat qui n'en voyait rien.
COMBATS = (
    "20260819-115025-combat-vide-recolte.png",
    "20260819-115106-combat-vide-recolte.png",
    "20260819-115112-combat-vide-recolte.png",
)
# Les captures de reference, pour que ces tests soient aussi une NON-REGRESSION : tout ce
# qui suit doit continuer de valoir sur l'ATH d'origine.
REFERENCES = ("combat1920.png", "combat1.png", "tacle.png")


def _lire(nom, dossier=ECHECS):
    chemin = dossier / nom
    image = cv2.imread(str(chemin))
    if image is None:
        pytest.skip(f"{nom} absente du depot")
    return image


class TestTheOrbWithoutAMaximumIsStillReadable:
    """L'ORBE A UN SEUL NOMBRE. `pv_max` n'est pas illisible : il n'est PAS AFFICHE."""

    @pytest.mark.parametrize("nom", COMBATS)
    def test_the_current_hp_number_is_read(self, nom):
        lecture = read_ui(_lire(nom))
        assert lecture.pv is not None, (
            f"{nom} : les PV courants sont affiches dans l'orbe et doivent se lire. "
            f"Ils tombaient a cheval sur les deux ROIs empilees, qui echouaient ensemble")
        assert 0 < lecture.pv <= 200, f"{nom} : {lecture.pv} PV pour un niveau 4"

    @pytest.mark.parametrize("nom", COMBATS)
    def test_action_and_movement_points_are_read(self, nom):
        """PA ET PM NE DEPENDENT PAS DU MAXIMUM DE PV, et ils en dependaient.

        `pipeline` gardait toute l'injection derriere `ui.pv_max` : sur cet ATH le
        personnage restait donc a 0 PA / 0 PM, le solveur ne pouvait ni bouger ni lancer
        un sort, et rendait « passer le tour » sans que rien n'echoue.
        """
        lecture = read_ui(_lire(nom))
        assert lecture.pa, f"{nom} : PA illisibles"
        assert lecture.pm is not None, f"{nom} : PM illisibles"

    @pytest.mark.parametrize("nom", COMBATS)
    def test_the_hp_share_is_known_without_a_maximum(self, nom):
        """LE GARDE-FOU `--min-hp` ETAIT MORT SILENCIEUSEMENT.

        `too_hurt = hp_ratio is not None and ...` : avec `hp_ratio` toujours None, il
        valait False a TOUS les PV. Le bot engageait donc a n'importe quelle sante.
        """
        part = read_hp_ratio(_lire(nom))
        assert part is not None, (
            f"{nom} : sans part de PV, `--min-hp` ne se declenche jamais")
        assert 0.0 < part <= 1.0

    def test_the_gauge_agrees_with_the_numbers_where_both_exist(self):
        """LE SECOND LECTEUR, INDEPENDANT DES CHIFFRES.

        `tacle.png` est la capture qui DISCRIMINE : lire 1,0 partout serait aussi le
        comportement d'une jauge cassee, et c'est la seule reference ou le personnage est
        entame (1146/1637 = 0,700).
        """
        image = _lire("tacle.png", RACINE)
        lecture = read_ui(image)
        chiffres = lecture.pv / lecture.pv_max
        jauge = read_hp_gauge(image)
        assert jauge is not None
        assert abs(jauge - chiffres) <= 0.02, (
            f"jauge {jauge:.3f} contre chiffres {chiffres:.3f}")

    def test_an_orb_hidden_by_a_panel_is_refused_rather_than_guessed(self):
        """UN ORBE ENTAME ET UN ORBE MASQUE SE RESSEMBLENT.

        Le panneau de chat deplie recouvre le haut de l'orbe ; le rouge commence alors
        plus bas, exactement comme s'il manquait des PV. La jauge annoncait 0,68 pour
        0,78 reels. Le lisere clair de l'orbe tranche -- absent sous un panneau.
        """
        assert read_hp_gauge(_lire("20260819-115106-combat-vide.png")) is None


class TestTheSingleRowSpellBar:
    """LA BARRE A UNE RANGEE. 24 emplacements inexistants etaient lus dans le decor."""

    @pytest.mark.parametrize("nom", COMBATS)
    def test_the_layout_is_measured_not_assumed(self, nom):
        disposition = bar_layout(_lire(nom))
        assert disposition is not None, f"{nom} : disposition de barre introuvable"
        assert disposition.rows == 1, (
            f"{nom} : barre relevee a {disposition.rows} rangee(s), or cette capture en "
            f"montre une seule")

    @pytest.mark.parametrize("nom", COMBATS)
    def test_the_bar_is_not_read_out_of_the_scenery(self, nom):
        """Le personnage a CINQ sorts. Le doctor en annoncait quatorze prets.

        Le decor d'Incarnam est clair, donc lu « pret » : les emplacements des rangees
        absentes offraient au solveur des sorts qui n'existent pas.
        """
        etats = read_spell_slots(_lire(nom))
        prets = sum(1 for e in etats if e is SlotState.READY)
        assert prets <= 6, (
            f"{nom} : {prets} sorts prets pour un personnage qui en a cinq — la lecture "
            f"retombe dans le decor des rangees inexistantes")
        assert all(e is SlotState.EMPTY for e in etats[12:]), (
            f"{nom} : les emplacements 12+ n'existent pas sur une barre a une rangee")

    @pytest.mark.parametrize("nom", REFERENCES)
    def test_the_three_row_bar_is_unchanged(self, nom):
        """NON-REGRESSION : l'ATH d'origine doit lire exactement comme avant."""
        image = _lire(nom, RACINE)
        disposition = bar_layout(image)
        assert disposition is not None and disposition.rows == 3
        prets = sum(1 for e in read_spell_slots(image) if e is SlotState.READY)
        assert prets >= 20, f"{nom} : {prets} sorts prets, la barre en montre une vingtaine"


class TestTheSmallBrightBoardIsDetected:
    """LA PANNE QUI EMPECHAIT DE JOUER : « aucune grille detectee — hors combat ? »."""

    @pytest.mark.parametrize("nom", COMBATS)
    def test_a_board_is_found_on_a_real_fight(self, nom):
        image = _lire(nom)
        assert read_timeline(image), f"{nom} : cette capture est bien un combat"
        estimation = estimate_grid(image)
        assert estimation is not None, (
            f"{nom} : sans plateau il n'y a ni entites, ni zone de deplacement, ni plan — "
            f"c'est la panne qui faisait abandonner des combats reels")
        assert len(estimation.board) >= 40

    @pytest.mark.parametrize("nom", COMBATS)
    def test_the_cell_width_is_the_expected_one(self, nom):
        """UN PLATEAU FAUX EST PIRE QU'AUCUN PLATEAU.

        Le repli ne doit pas rendre n'importe quel reseau : la cellule du jeu fait 92 a
        93 px de large a ce zoom, et le parasite qui l'emportait avant en faisait 36.
        """
        estimation = estimate_grid(_lire(nom))
        largeur = abs(estimation.e_x[0] - estimation.e_y[0])
        assert 85 <= largeur <= 100, f"{nom} : cellule de {largeur:.0f} px"

    @pytest.mark.parametrize("nom", REFERENCES)
    def test_the_reference_captures_are_untouched(self, nom):
        """LE REPLI NE PEUT QUE CONVERTIR UN ECHEC EN DETECTION.

        Il n'est appele qu'apres un None, donc aucune capture qui marchait ne change de
        resultat. Ce test fixe les valeurs d'avant.
        """
        attendu = {"combat1920.png": 239, "combat1.png": 113, "tacle.png": 155}
        estimation = estimate_grid(_lire(nom, RACINE))
        assert estimation is not None
        assert len(estimation.board) == attendu[nom], (
            f"{nom} : {len(estimation.board)} cases au lieu de {attendu[nom]} — le repli "
            f"a change une detection qui marchait")

    def test_a_screen_without_a_board_stays_refused(self):
        """Le repli ne doit pas INVENTER de plateau la ou il n'y en a pas."""
        for nom in ("hdv_vide.png", "cherche.png", "5.png", "padofre.png"):
            image = cv2.imread(str(RACINE / nom))
            if image is None:
                continue
            assert estimate_grid(image) is None, f"{nom} : plateau invente"


class TestTheDigitTemplatesCoverEveryDigit:
    """LES CHIFFRES 3 ET 8 MANQUAIENT, et personne ne pouvait s'en apercevoir.

    Un chiffre absent des exemplaires est REJETE puis relu par Tesseract -- c'est le
    comportement voulu, et il est silencieux. Consequence mesuree : `read_ui` coutait
    203 ms au lieu des 1,3 ms annoncees par le README, parce que « 1632 » contient un 3 et
    partait donc a chaque fois chez Tesseract. Sur un poste sans Tesseract installe, la
    meme lecture rendait simplement None.
    """

    def test_every_digit_has_an_exemplar(self):
        from jev_tactics.perception.digits import DigitTemplates

        manquants = sorted(set(range(10)) - DigitTemplates.load().covered_digits())
        assert not manquants, (
            f"chiffres sans exemplaire : {manquants} — toute valeur les contenant part "
            f"chez Tesseract, ou rend None s'il n'est pas installe")

    def test_the_reference_readings_are_unchanged(self):
        attendu = {
            "combat1920.png": (1632, 1632, 10, 4),
            "combat1.png": (1657, 1657, 10, 3),
            "hors_combat.png": (1667, 1667, 11, 4),
            "capture.png": (1672, 1672, 11, 4),
            "tacle.png": (1146, 1637, 10, 4),
        }
        for nom, (pv, pv_max, pa, pm) in attendu.items():
            lecture = read_ui(_lire(nom, RACINE))
            assert (lecture.pv, lecture.pv_max, lecture.pa, lecture.pm) == (pv, pv_max, pa, pm), nom
