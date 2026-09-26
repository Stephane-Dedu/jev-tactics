"""Affectation des sorts aux emplacements reels de la barre.

L'appariement automatique par icone a ete essaye et mesure comme insuffisant (marge
mediane de 0,033, 17 noms distincts pour 26 cases). Ces tests portent donc sur la partie
qui reste automatisee : appliquer une affectation fournie par l'humain sans rien perdre
ni rien inventer."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from map_spells import apply_assignment, parse_assignment

from jev_tactics.rules.spells import load_spells

SPELLS = [
    {"name": "Absorption", "source": "dofusdb", "slot": 0, "ap_cost": 3,
     "range_min": 1, "range_max": 6, "damage_min": 20, "damage_max": 24},
    {"name": "Ravage", "source": "dofusdb", "slot": 1, "ap_cost": 3,
     "range_min": 1, "range_max": 6, "damage_min": 28, "damage_max": 32},
]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "sorts.json"
    path.write_text(json.dumps(SPELLS), encoding="utf-8")
    return path


class TestParsing:
    def test_reads_slot_name_pairs(self):
        assert parse_assignment("0=Absorption,4=Ravage") == {0: "Absorption", 4: "Ravage"}

    def test_tolerates_spaces_and_accents(self):
        assert parse_assignment(" 2 = Hemorragie ") == {2: "Hemorragie"}

    def test_malformed_pair_exits(self):
        with pytest.raises(SystemExit):
            parse_assignment("Absorption")

    def test_non_numeric_slot_exits(self):
        with pytest.raises(SystemExit):
            parse_assignment("premier=Absorption")


class TestApply:
    def test_assigns_the_given_slots(self, config):
        apply_assignment(config, {5: "Absorption", 9: "Ravage"})
        slots = {s.name: s.slot for s in load_spells(config)}
        assert slots == {"Absorption": 5, "Ravage": 9}

    def test_matching_is_case_insensitive(self, config):
        apply_assignment(config, {3: "absorption"})
        assert next(s for s in load_spells(config) if s.name == "Absorption").slot == 3

    def test_unassigned_spells_lose_their_slot(self, config):
        """LE point delicat : un sort non affecte ne doit PAS garder le numero herite de
        l'ordre de la classe. Ce numero est faux -- le bot appuierait sur la touche d'un
        autre sort et lancerait autre chose."""
        apply_assignment(config, {5: "Absorption"})
        assert next(s for s in load_spells(config) if s.name == "Ravage").slot is None

    def test_unknown_name_exits_instead_of_being_ignored(self, config):
        """Une faute de frappe silencieusement ignoree laisserait un sort muet, sans que
        rien ne le dise."""
        with pytest.raises(SystemExit):
            apply_assignment(config, {0: "Sort Qui N Existe Pas"})

    def test_nothing_is_written_when_a_name_is_unknown(self, config):
        """Echec ATOMIQUE : une affectation partiellement appliquee serait pire qu'un
        refus, car elle melangerait ancien et nouveau."""
        before = config.read_text(encoding="utf-8")
        with pytest.raises(SystemExit):
            apply_assignment(config, {0: "Absorption", 1: "Inexistant"})
        assert config.read_text(encoding="utf-8") == before

    def test_the_config_stays_loadable(self, config):
        apply_assignment(config, {7: "Absorption", 8: "Ravage"})
        assert len(load_spells(config)) == 2
