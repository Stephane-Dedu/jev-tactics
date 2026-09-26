"""Barre de sorts : distinguer vide / en recharge / pret, et en tirer les sorts jouables.

Le point delicat est qu'un sort en recharge reste COLORE : c'est la luminosite qui
tranche, pas la saturation. Un test le verifie explicitement."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.spellbar import (
    COLUMNS,
    MAX_PLAUSIBLE_COOLING_SHARE,
    ORIGIN_Y,
    ROWS,
    SlotState,
    audit_bar,
    detect_bar_geometry,
    read_spell_slots,
    ready_slots,
    slot_index,
    slot_region,
)
from jev_tactics.rules.spells import Spell, available_spells


def _frame():
    return np.zeros((1079, 1919, 3), dtype=np.uint8)


def _paint(frame, index: int, hue: int, saturation: int, value: int) -> None:
    x0, y0, x1, y1 = slot_region(index)
    colour = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = colour


# Valeurs relevees sur les captures reelles.
READY = (15, 150, 200)
COOLING = (15, 150, 80)     # meme teinte et saturation, luminosite effondree
EMPTY = (15, 45, 58)


ROOT = Path(__file__).resolve().parents[1]


class TestSlotStates:
    def test_all_empty_on_a_blank_frame(self):
        assert set(read_spell_slots(_frame())) == {SlotState.EMPTY}

    def test_ready_slot_detected(self):
        frame = _frame()
        _paint(frame, 0, *READY)
        assert read_spell_slots(frame)[0] is SlotState.READY

    def test_cooling_slot_detected(self):
        frame = _frame()
        _paint(frame, 0, *COOLING)
        assert read_spell_slots(frame)[0] is SlotState.COOLING

    def test_cooling_keeps_colour_only_brightness_drops(self):
        """Le discriminant est la LUMINOSITE : meme teinte et saturation qu'un sort pret,
        et pourtant indisponible. Se fier a la saturation aurait rate le cas."""
        ready_frame, cooling_frame = _frame(), _frame()
        _paint(ready_frame, 3, *READY)
        _paint(cooling_frame, 3, READY[0], READY[1], COOLING[2])
        assert read_spell_slots(ready_frame)[3] is SlotState.READY
        assert read_spell_slots(cooling_frame)[3] is SlotState.COOLING

    def test_empty_distinguished_from_cooling(self):
        frame = _frame()
        _paint(frame, 1, *EMPTY)
        _paint(frame, 2, *COOLING)
        states = read_spell_slots(frame)
        assert states[1] is SlotState.EMPTY and states[2] is SlotState.COOLING

    def test_pale_but_bright_icon_is_not_empty(self):
        """Une icone claire et peu saturee (gris/blanc) reste un sort : il faut les DEUX
        criteres bas pour conclure au vide."""
        frame = _frame()
        _paint(frame, 4, 15, 28, 151)
        assert read_spell_slots(frame)[4] is not SlotState.EMPTY

    def test_one_state_per_slot(self):
        assert len(read_spell_slots(_frame())) == COLUMNS * ROWS

    def test_slot_index_is_row_major(self):
        assert slot_index(0, 0) == 0 and slot_index(1, 0) == COLUMNS
        assert slot_index(2, 3) == 2 * COLUMNS + 3

    def test_regions_do_not_overlap(self):
        a, b = slot_region(0), slot_region(1)
        assert a[2] <= b[0] or b[2] <= a[0]

    def test_ready_slots_returns_indices(self):
        frame = _frame()
        _paint(frame, 5, *READY)
        _paint(frame, 6, *COOLING)
        assert ready_slots(frame) == {5}


class TestAvailableSpells:
    BOLT = Spell(name="trait", ap_cost=3, range_max=6, slot=0)
    BLAST = Spell(name="souffle", ap_cost=4, range_max=4, slot=1)
    UNMAPPED = Spell(name="inconnu", ap_cost=2, range_max=3)

    def test_filters_out_unavailable_slots(self):
        assert available_spells([self.BOLT, self.BLAST], {0}) == [self.BOLT]

    def test_unmapped_spell_is_kept(self):
        """Sans emplacement connu on n'a aucune information : l'ecarter priverait le
        planificateur de coups sans raison."""
        assert self.UNMAPPED in available_spells([self.BOLT, self.UNMAPPED], {9})

    def test_no_reading_keeps_everything(self):
        spells = [self.BOLT, self.BLAST]
        assert available_spells(spells, None) == spells

    def test_empty_ready_set_filters_all_mapped(self):
        assert available_spells([self.BOLT, self.BLAST], set()) == []

class TestBarAudit:
    """La barre CONFIGUREE correspond-elle a celle affichee ?

    La config associe un sort a un emplacement et en DEDUIT son raccourci. Si la barre a
    bouge en jeu, le bot appuie sur la touche en croyant lancer un sort et lance ce qui
    s'y trouve -- sur la cible calculee pour l'autre. Rien ne le verifiait, et la barre de
    ce projet a deja ete remappee une fois : seule une relecture attentive de la config
    l'avait rattrape."""

    def _frame_with(self, occupied):
        """Frame ou seuls les emplacements donnes sont occupes."""
        frame = np.zeros((1080, 1920, 3), np.uint8)
        for index in occupied:
            x0, y0, x1, y1 = slot_region(index)
            frame[y0:y1, x0:x1] = (40, 200, 200)      # sature = emplacement rempli
        return frame

    def test_a_matching_bar_is_consistent(self):
        audit = audit_bar(self._frame_with([0, 1, 2]), [0, 1, 2])
        assert audit.consistent and audit.missing == ()

    def test_a_configured_slot_gone_from_the_bar_is_flagged(self):
        """Le cas qui compte : la config croit tenir un sort en 5, l'ecran dit non."""
        audit = audit_bar(self._frame_with([0, 1, 2]), [0, 1, 2, 5])
        assert not audit.consistent and audit.missing == (5,)

    def test_extra_occupied_slots_are_NOT_an_error(self):
        """La barre contient des objets, des consommables et des sorts qu'on a choisi de
        ne pas jouer. Mesure sur les captures reelles : 26 emplacements occupes pour 20
        sorts configures, et les 20 sont bien la. Crier ici serait crier a chaque tour."""
        audit = audit_bar(self._frame_with([0, 1, 2, 7]), [0, 1, 2])
        assert audit.consistent and audit.unmapped == (7,)

    def test_an_empty_bar_means_NOT_DISPLAYED(self):
        """Hors combat l'inventaire occupe la place de la barre. Signaler alors vingt
        sorts manquants serait une fausse alerte sur le premier ecran qu'on regarde --
        le meme piege que la lecture d'UI hors combat."""
        audit = audit_bar(np.zeros((1080, 1920, 3), np.uint8), [0, 1, 2])
        assert not audit.displayed and audit.consistent

    def test_the_message_names_the_slots(self):
        """Un verdict sans mesure n'aide pas : c'est le numero d'emplacement qui permet
        de corriger la config."""
        audit = audit_bar(self._frame_with([0]), [0, 9])
        assert "[9]" in audit.describe()


class TestAuditOnRealCaptures:
    """Les captures reelles doivent CONFIRMER la config livree, sans quoi le controle
    crierait des le premier tour et finirait desactive."""

    def _slots(self):
        from jev_tactics.rules.spells import load_spells

        path = ROOT / "configs" / "spells" / "sacrieur.json"
        if not path.exists():
            pytest.skip("config sacrieur absente")
        return [s.slot for s in load_spells(path) if s.slot is not None]

    @pytest.mark.parametrize("name", ["combat1.png", "tacle.png", "bug4.png"])
    def test_the_shipped_config_matches_the_screen(self, name):
        path = ROOT / name
        if not path.exists():
            pytest.skip(f"{name} absent")
        audit = audit_bar(cv2.imread(str(path)), self._slots())
        assert audit.consistent, audit.describe()



class TestAPanelOverTheBarIsNotTwentyTwoCooldowns:
    """LE BOT SE CREVE LES YEUX LUI-MEME.

    Un panneau du jeu peut se dessiner PAR-DESSUS la barre de sorts, et le bot en ouvre un
    de son propre chef : l'infobulle de monstre, qu'il affiche pour lire les PV ennemis
    (`enemy_hp_hovers`). Les emplacements couverts deviennent sombres, donc lus « en
    rechargement » -- pas « vides ».

    La garde de `CombatSession` ne couvrait que le cas TOUT VIDE. Sa surveillance etait donc
    plus etroite que sa promesse : sur une barre recouverte, `occupied` reste plein, la garde
    se tait, et le solveur planifie avec CINQ sorts sur vingt-sept.

    Mesure sur les 24 captures de session, part des occupes lus en rechargement :

        22 captures      4 % a 12 %      barre degagee
         2 captures     81 %             infobulle par-dessus la barre

    L'intervalle entre 12 % et 81 % est VIDE : le seuil s'y pose sans arbitrage, comme celui
    du glyphe de mort. Et physiquement, quatre sorts sur cinq en rechargement au meme
    instant n'arrive pas.

    CE QUE COUTAIT LE DEFAUT : au mieux un plan mediocre, au pire un tour passe -- le mode
    de panne le plus cher du projet, puisque le bilan rend alors « 30 tours joues »,
    strictement le meme qu'un combat gagne.
    """

    RECOUVERTES = ("20260807-221715.png", "20260807-224524.png")

    def _part(self, nom):
        import cv2
        chemin = Path(__file__).resolve().parents[1] / "data" / "runs" / nom
        if not chemin.exists():
            pytest.skip(f"{nom} absent")
        frame = cv2.imread(str(chemin))
        assert frame is not None, f"{nom} illisible"
        etats = read_spell_slots(frame)
        occupes = [e for e in etats if e is not SlotState.EMPTY]
        prets = [e for e in etats if e is SlotState.READY]
        assert occupes, f"{nom} : barre introuvable, ce n'est pas le cas teste ici"
        return (len(occupes) - len(prets)) / len(occupes)

    def _toutes(self):
        dossier = Path(__file__).resolve().parents[1] / "data" / "runs"
        noms = sorted(p.name for p in dossier.glob("*.png"))
        if len(noms) < 20:
            pytest.skip("captures de session absentes")
        return noms

    def test_the_covered_captures_exceed_the_threshold(self):
        for nom in self.RECOUVERTES:
            assert self._part(nom) > MAX_PLAUSIBLE_COOLING_SHARE, (
                f"{nom} : part {self._part(nom):.0%}, seuil "
                f"{MAX_PLAUSIBLE_COOLING_SHARE:.0%} — la barre recouverte n'est plus "
                f"reconnue comme illisible et le solveur y perdra vingt sorts")

    def test_every_other_capture_stays_far_below(self):
        """CONTRE-EPREUVE : un seuil qui declencherait partout ferait renoncer au filtrage
        sur toutes les frames, donc planifier des sorts en recharge a chaque tour."""
        parts = {nom: self._part(nom) for nom in self._toutes()
                 if nom not in self.RECOUVERTES}
        assert parts, "aucune capture degagee"
        assert max(parts.values()) < MAX_PLAUSIBLE_COOLING_SHARE / 2, (
            f"la plus haute part sur barre degagee vaut {max(parts.values()):.0%} : la "
            f"marge sous le seuil a fondu")

    def test_the_gap_between_the_two_regimes_is_empty(self):
        """CE QUI POSE LE SEUIL SANS ARBITRAGE. Entre la pire barre degagee et la meilleure
        barre recouverte, il n'y a rien -- et le seuil tombe au milieu."""
        degagees = max(self._part(n) for n in self._toutes()
                       if n not in self.RECOUVERTES)
        couvertes = min(self._part(n) for n in self.RECOUVERTES)
        assert degagees < MAX_PLAUSIBLE_COOLING_SHARE < couvertes
        assert couvertes - degagees > 0.4, (
            f"l'ecart entre les deux regimes est tombe a {couvertes - degagees:.0%} : le "
            f"seuil demande desormais un arbitrage, et il faut le mesurer")


class TestTheBarGeometryIsCheckedAgainstTheScreen:
    """LA BARRE DE SORTS SE CONFIGURE EN JEU, et la geometrie du code n'en sait rien.

    Elle peut montrer une, deux ou trois rangees, et les rangees s'empilent VERS LE HAUT
    depuis une rangee du bas qui, elle, ne bouge pas. Mesure sur les captures :

        reference (3 rangees)   rangees a y = 924, 969, 1011   bandeau y 888..1079
        session du 19/08 (1)    rangee  a y = 1006             bandeau y 951..1079

    `ORIGIN_Y` est ancre sur la rangee du HAUT : sur une barre a une rangee, tous les
    prelevements tombent 85 px trop haut, DANS LE DECOR. Celui d'Incarnam etant clair et
    sature, il se lit comme des emplacements occupes -- le bilan a annonce « 30
    emplacements occupes » pour un personnage de niveau 5 qui en a cinq, puis a conclu que
    la config etait coherente. Une lecture fausse qui se tait coute plus qu'une lecture
    absente : elle envoie corriger un fichier qui n'a rien.
    """

    SESSION = ROOT / "data" / "failures" / "20260819-115112-combat-vide-recolte.png"

    def test_a_one_row_bar_is_named(self):
        if not self.SESSION.exists():
            pytest.skip("frame de session absente (non versionnee)")
        geo = detect_bar_geometry(cv2.imread(str(self.SESSION)))
        assert geo is not None, "la barre n'est pas localisee du tout"
        assert geo.rows == 1, f"{geo.rows} rangee(s) detectee(s)"
        assert 960 <= geo.origin_y <= 990, geo

    @pytest.mark.parametrize("nom", ["combat1920.png", "combat1.png", "tacle.png",
                                     "bug4.png", "bugcarreblanc.png"])
    def test_the_reference_layout_is_not_accused(self, nom):
        """CONTRE-EPREUVE, et c'est elle qui compte : un garde-fou qui crie sur la
        disposition NORMALE serait retire au bout de deux sessions."""
        chemin = ROOT / nom
        if not chemin.exists():
            pytest.skip(f"{nom} absent")
        geo = detect_bar_geometry(cv2.imread(str(chemin)))
        assert geo is not None and geo.rows == 3, geo
        assert abs(geo.origin_y - ORIGIN_Y) <= 3, (
            f"la geometrie relevee {geo} s'ecarte de la constante posee a la main")

    def test_out_of_combat_says_nothing(self):
        """Hors combat la barre laisse place a l'inventaire : rien a confronter, et une
        accusation y serait permanente."""
        for nom in ("capture.png", "hors_combat.png"):
            chemin = ROOT / nom
            if chemin.exists():
                assert detect_bar_geometry(cv2.imread(str(chemin))) is None, nom

    def test_the_audit_puts_the_geometry_FIRST(self):
        """Si la geometrie tombe a cote, `missing` et `unmapped` ne parlent plus de la
        barre mais du decor -- et le font avec assurance. L'audit doit donc annoncer le
        decalage AVANT de conclure a une config perimee."""
        if not self.SESSION.exists():
            pytest.skip("frame de session absente")
        audit = audit_bar(cv2.imread(str(self.SESSION)), [0, 1, 2])
        assert audit.consistent, "les 3 premiers emplacements sont bien occupes"
        assert audit.layout is not None and "1 rangee" in audit.layout, audit.layout
        assert audit.describe().startswith(audit.layout)
