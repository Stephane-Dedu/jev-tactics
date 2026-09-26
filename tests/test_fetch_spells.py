"""Conversion des donnees dofusdb en sorts du bot.

Aucun test ne touche au reseau : ce sont les FONCTIONS DE CONVERSION qui peuvent se
tromper, pas la requete HTTP. Les donnees d'entree sont des extraits reels de l'API,
recopies tels quels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_spells import area_radius, area_shape, convert, damage_range

from jev_tactics.rules.spells import Spell

# Extrait reel : Absorption, niveau 3 (20-24 vol Feu).
ABSORPTION_EFFECTS = [
    {"effectId": 94, "diceNum": 20, "diceSide": 24, "rawZone": None},
    {"effectId": 186, "diceNum": 150, "diceSide": 0, "rawZone": None},
    {"effectId": 90, "diceNum": 10, "diceSide": 0, "rawZone": None},
]
ABSORPTION_LEVEL = {
    "apCost": 3, "range": 6, "minRange": 1, "maxCastPerTurn": 3,
    "castTestLos": True, "effects": ABSORPTION_EFFECTS,
}
ABSORPTION = {"id": 12734, "name": {"fr": "Absorption", "en": "Absorption"}}


class TestDamageRange:
    def test_reads_the_dice_range(self):
        assert damage_range(ABSORPTION_EFFECTS) == (20, 24)

    def test_ignores_non_damage_effects(self):
        """Un sort porte des effets de buff, de soin, d'etat : les compter comme degats
        gonflerait sa valeur et fausserait tous les arbitrages du solveur."""
        assert damage_range([{"effectId": 186, "diceNum": 150, "diceSide": 0}]) == (0, 0)

    def test_sums_multi_element_damage(self):
        effects = [{"effectId": 97, "diceNum": 10, "diceSide": 12},
                   {"effectId": 99, "diceNum": 5, "diceSide": 7}]
        assert damage_range(effects) == (15, 19)

    def test_fixed_value_has_no_upper_dice(self):
        """diceSide=0 signifie une valeur fixe, pas des degats maximaux nuls."""
        assert damage_range([{"effectId": 100, "diceNum": 30, "diceSide": 0}]) == (30, 30)

    def test_no_damage_effect_gives_zero(self):
        """Un sort utilitaire rend 0 : le solveur ne le choisira jamais, ce qui est
        correct tant qu'aucun soin ni buff n'est modelise. Inventer une valeur le ferait
        jouer pour de mauvaises raisons."""
        assert damage_range([]) == (0, 0)


def _zone(shape: str, size: int = 1) -> dict:
    return {"shape": ord(shape), "param1": size, "param2": 0}


class TestAreaRadius:
    """Extraits reels : `zoneDescr.shape` porte un code ASCII ('P' cible unique,
    'C' cercle, '+' croix, 'X' croix diagonale) et `param1` la taille."""

    def test_reads_the_radius_from_the_zone(self):
        assert area_radius([{"effectId": 100, "zoneDescr": _zone("C", 2)}]) == 2

    def test_point_shape_is_single_target(self):
        """'P' porte pourtant param1=1 : le prendre pour un rayon ferait croire a une
        zone sur sept sorts de la classe, et le solveur surestimerait chaque coup."""
        assert area_radius([{"effectId": 100, "zoneDescr": _zone("P", 1)}]) == 0

    def test_no_zone_is_single_target(self):
        assert area_radius(ABSORPTION_EFFECTS) == 0

    def test_zone_of_a_non_damage_effect_is_ignored(self):
        """La zone d'un buff n'est pas celle des degats."""
        assert area_radius([{"effectId": 186, "zoneDescr": _zone("C", 3)}]) == 0

    def test_radius_is_capped(self):
        """Une zone aberrante ferait exploser `area_cells`, qui parcourt le plateau."""
        assert area_radius([{"effectId": 100, "zoneDescr": _zone("C", 9)}]) <= 6


class TestAreaShape:
    def test_known_shapes_are_mapped(self):
        for letter, expected in (("P", "point"), ("C", "circle"),
                                 ("+", "diagonal"), ("X", "cross")):
            got = area_shape([{"effectId": 100, "zoneDescr": _zone(letter)}])
            assert got == expected, letter

    def test_the_letters_describe_the_SCREEN_not_the_lattice(self):
        """Piege geometrique constate en jeu (« Decimation tape dans le vide »).

        En vue isometrique, les voisins ADJACENTS du reseau apparaissent en X a l'ecran
        (haut-gauche, haut-droite, bas-gauche, bas-droite) et les voisins DIAGONAUX
        apparaissent en + (gauche, droite, haut, bas). Les lettres du jeu decrivent
        l'ecran : « + » porte donc sur les diagonales du reseau, « X » sur les voisines
        immediates. Je les avais interverties."""
        assert area_shape([{"effectId": 100, "zoneDescr": _zone("+")}]) == "diagonal"
        assert area_shape([{"effectId": 100, "zoneDescr": _zone("X")}]) == "cross"

    def test_unknown_shape_falls_back_on_circle(self):
        """Approximation ASSUMEE : un cercle surestime la zone. Le solveur y verra des
        cibles absentes -- decevant, mais moins nuisible que d'ignorer un coup groupe."""
        assert area_shape([{"effectId": 100, "zoneDescr": _zone("Z")}]) == "circle"


class TestConvert:
    def test_produces_a_loadable_spell(self):
        entry = convert(ABSORPTION, ABSORPTION_LEVEL, 0)
        assert Spell.model_validate(entry).name == "Absorption"

    def test_keeps_the_french_name(self):
        assert convert(ABSORPTION, ABSORPTION_LEVEL, 0)["name"] == "Absorption"

    def test_marks_the_source(self):
        """Ni « mesure » (rien n'a ete releve sur l'ecran) ni « saisie » (personne n'a
        recopie) : une reference verifiable, qui merite son propre nom."""
        assert convert(ABSORPTION, ABSORPTION_LEVEL, 0)["source"] == "dofusdb"

    def test_carries_ranges_and_caps(self):
        entry = convert(ABSORPTION, ABSORPTION_LEVEL, 0)
        assert (entry["range_min"], entry["range_max"]) == (1, 6)
        assert entry["max_casts_per_turn"] == 3

    def test_missing_cap_becomes_unlimited(self):
        """L'API rend 0 pour « pas de plafond ». Le prendre au pied de la lettre
        interdirait tout lancer."""
        level = {**ABSORPTION_LEVEL, "maxCastPerTurn": 0, "maxCastPerTarget": 0}
        entry = convert(ABSORPTION, level, 0)
        assert entry["max_casts_per_turn"] == 99 and entry["max_casts_per_target"] == 99

    def test_passive_spell_is_dropped(self):
        """Un sort a 0 PA n'est pas jouable : le garder polluerait la recherche."""
        assert convert(ABSORPTION, {**ABSORPTION_LEVEL, "apCost": 0}, 0) is None

    def test_unnamed_spell_falls_back_on_its_id(self):
        entry = convert({"id": 999, "name": {}}, ABSORPTION_LEVEL, 0)
        assert "999" in entry["name"]
