"""Roles de sorts.

L'enjeu n'est pas que les valeurs soient exactes -- elles ne le sont pas et ne le
pretendent pas -- mais que les archetypes soient COHERENTS ENTRE EUX : le solveur compare
des coups, donc ce sont les rapports qui gouvernent ses choix. Un role longue portee qui
coûterait moins cher et frapperait plus fort qu'un corps a corps rendrait le corps a
corps inutile, et le bot ne s'approcherait jamais."""

from pathlib import Path

import pytest

from jev_tactics.rules.roles import (
    ROLES,
    describe_roles,
    parse_assignment,
    spells_from_roles,
)
from jev_tactics.rules.spells import (
    DEFAULT_ROW_KEYS,
    Spell,
    apply_element_focus,
    audit_spell_config,
    default_spell_keys,
    load_spells,
)


class TestParsing:
    def test_reads_slot_role_pairs(self):
        assert parse_assignment("0=cac,2=dps_longue") == {0: "cac", 2: "dps_longue"}

    def test_tolerates_spaces(self):
        assert parse_assignment(" 1 = zone , 3 = cac ") == {1: "zone", 3: "cac"}

    def test_unknown_role_raises_and_lists_the_known_ones(self):
        """Ignorer une affectation en silence donnerait un personnage ampute d'un sort
        sans que rien ne le dise -- exactement la panne que ce module evite."""
        with pytest.raises(ValueError, match="role inconnu"):
            parse_assignment("0=archimage")

    def test_malformed_pair_raises(self):
        with pytest.raises(ValueError, match="illisible"):
            parse_assignment("cac")

    def test_non_numeric_slot_raises(self):
        with pytest.raises(ValueError, match="emplacement invalide"):
            parse_assignment("premier=cac")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="aucune affectation"):
            parse_assignment("  ,  ")


class TestConversion:
    def test_produces_loadable_spells(self):
        spells = spells_from_roles({0: "cac", 2: "dps_longue"})
        assert all(isinstance(s, Spell) for s in spells)

    def test_slots_are_preserved(self):
        assert [s.slot for s in spells_from_roles({3: "zone", 0: "cac"})] == [0, 3]

    def test_names_are_distinct(self):
        """Deux sorts de meme nom se confondraient dans les plafonds par tour."""
        spells = spells_from_roles({0: "cac", 1: "cac", 2: "cac"})
        assert len({s.name for s in spells}) == 3

    def test_area_role_keeps_its_radius(self):
        assert spells_from_roles({0: "zone"})[0].area_radius == 1


class TestCoherence:
    """Ce que les archetypes doivent respecter pour que le solveur arbitre sensement."""

    def test_range_costs_more_and_hits_less_than_melee(self):
        melee, ranged = ROLES["cac"], ROLES["dps_longue"]
        assert ranged.ap_cost > melee.ap_cost
        assert ranged.damage_max < melee.damage_max

    def test_area_hits_less_per_target_than_single_target(self):
        assert ROLES["zone"].damage_max < ROLES["dps_courte"].damage_max

    def test_heavy_hitter_is_capped_to_one_cast(self):
        assert ROLES["finition"].max_casts_per_turn == 1
        assert ROLES["finition"].damage_max > ROLES["cac"].damage_max

    def test_support_deals_no_damage(self):
        """Soins et buffs ne sont pas modelises : mieux vaut un role a 0 degat, que le
        solveur ignorera visiblement, qu'une valeur inventee qu'il prendrait au serieux."""
        assert ROLES["soutien"].damage_max == 0

    def test_every_role_is_a_valid_spell(self):
        for slot, name in enumerate(ROLES):
            assert ROLES[name].to_spell(slot).slot == slot

    def test_ranges_are_never_inverted(self):
        assert all(r.range_min <= r.range_max for r in ROLES.values())

    def test_description_lists_every_role(self):
        text = describe_roles()
        assert all(name in text for name in ROLES)


class TestDefaultKeys:
    """Le `slot` porte deja le lien vers la touche : le redemander a l'utilisateur
    l'obligeait a ressaisir une information deja donnee, avec une faute de frappe
    possible. Or un sort sans raccourci est SAUTE EN SILENCE -- constate en jeu :
    « sorts sans raccourci, sautes : ['dps_courte_1'] », 0 clic, aucun sort lance."""

    def test_slot_maps_to_the_row_key(self):
        keys = default_spell_keys(spells_from_roles({0: "cac", 2: "dps_longue"}))
        assert keys == {"cac_0": "1", "dps_longue_2": "3"}

    def test_covers_the_whole_first_row(self):
        keys = default_spell_keys(spells_from_roles({i: "cac" for i in range(10)}))
        assert sorted(keys.values()) == sorted(DEFAULT_ROW_KEYS)

    def test_beyond_the_first_row_proposes_nothing(self):
        """Au-dela, les raccourcis dependent de modificateurs dont la convention n'a pas
        ete verifiee. Une touche fausse lancerait LE MAUVAIS SORT -- pire que rien."""
        assert default_spell_keys(spells_from_roles({12: "cac"})) == {}

    def test_spell_without_slot_gets_no_key(self):
        assert default_spell_keys([Spell(name="x", ap_cost=3, range_max=4)]) == {}

    def test_every_generated_spell_is_playable(self):
        """Bout en bout : une config issue de roles ne doit laisser aucun sort muet."""
        spells = spells_from_roles({0: "cac", 1: "dps_courte", 3: "zone"})
        keys = default_spell_keys(spells)
        assert all(s.name in keys for s in spells)


class TestTheConfigAuditSaysWhatWillActuallyBeCast:
    """Ce que la config PROMET contre ce que le bot en fera.

    Mesure sur la vraie configuration du Sacrieur, 20 sorts : **9 sont utilitaires**,
    donc ecartes par `offensive_spells` pour garder la recherche exhaustive. Pres de la
    moitie d'une config obtenue par `fetch_spells.py` ne sert a rien, et rien ne le
    disait -- ce n'est pas une panne, c'est une attente a corriger avant la session."""

    def _spell(self, name, damage=20, **kwargs):
        return Spell(name=name, ap_cost=3, range_max=4,
                     damage_min=damage, damage_max=damage, **kwargs)

    def test_a_clean_config_says_nothing(self):
        """Le silence est le cas normal : une ligne toujours affichee cesse d'etre lue."""
        spells = [self._spell("a"), self._spell("b")]
        assert audit_spell_config(spells, {"a": "1", "b": "2"}) == []

    def test_utility_spells_are_reported_as_an_observation(self):
        spells = [self._spell("frappe"), self._spell("Attirance", damage=0)]
        ligne = audit_spell_config(spells, {"frappe": "1", "Attirance": "2"})[0]
        assert "Attirance" in ligne and "ignores" in ligne
        assert not ligne.startswith("/!" + chr(92)), (
            "l'exclusion est deliberee et mesuree, pas une panne")

    def test_an_offensive_spell_without_a_shortcut_is_a_warning(self):
        """LE defaut le plus couteux du projet : le solveur le planifie, l'executeur le
        SAUTE, et le tour part sans lui."""
        spells = [self._spell("frappe"), self._spell("Hemorragie")]
        ligne = audit_spell_config(spells, {"frappe": "1"})[0]
        assert ligne.startswith("/!" + chr(92))
        assert "Hemorragie" in ligne and "SAUTES" in ligne

    def test_a_utility_spell_without_a_shortcut_is_not_reported(self):
        """Il ne sera de toute facon jamais planifie : le signaler enverrait corriger un
        raccourci qui ne servirait a rien."""
        spells = [self._spell("frappe"), self._spell("Transfusion", damage=0)]
        lignes = audit_spell_config(spells, {"frappe": "1"})
        assert not any("SAUTES" in ligne for ligne in lignes)

    def test_a_shortcut_for_an_unknown_spell_is_a_warning(self):
        """Une faute de frappe dans --spell-keys ne fait RIEN : le sort vise garde son
        raccourci d'origine, et la correction qu'on croyait apporter n'a pas eu lieu."""
        lignes = audit_spell_config([self._spell("Ravage")],
                                    {"Ravage": "2", "Ravag": "3"})
        ligne = next(x for x in lignes if "absent de la config" in x)
        assert ligne.startswith("/!" + chr(92)) and "Ravag" in ligne

    def test_a_config_without_any_offensive_spell_is_named(self):
        """Certain, pas probable : le solveur ne pourra jamais frapper."""
        spells = [self._spell("Transfusion", damage=0)]
        ligne = audit_spell_config(spells, {"Transfusion": "1"})[0]
        assert ligne.startswith("/!" + chr(92)) and "aucun sort offensif" in ligne

    def test_the_real_sacrieur_config_is_audited(self):
        """Sur la vraie config : 11 offensifs, 9 utilitaires, aucun sort muet."""
        path = Path(__file__).resolve().parents[1] / "configs" / "spells" / "sacrieur.json"
        if not path.exists():
            pytest.skip("sacrieur.json absent")
        spells = load_spells(path)
        lignes = audit_spell_config(spells, default_spell_keys(spells))
        assert len(lignes) == 1, lignes
        assert "11 offensifs" in lignes[0] and "9 utilitaires" in lignes[0]
        assert not lignes[0].startswith("/!" + chr(92))


class TestExplicitKeys:
    """Raccourci declare. La deduction par `slot` ne couvre que la premiere ligne ; au
    dela, les barres sont liees a des COMBINAISONS que la position ne permet pas de
    deviner. Sans champ explicite, les deux tiers d'un arsenal restent injouables."""

    def _spell(self, **kwargs):
        base = dict(name="x", ap_cost=3, range_max=4, damage_min=10, damage_max=12)
        return Spell(**{**base, **kwargs})

    def test_declared_key_wins_over_the_slot(self):
        keys = default_spell_keys([self._spell(slot=0, key="ctrl+shift+1")])
        assert keys == {"x": "ctrl+shift+1"}

    def test_slot_still_serves_when_no_key_is_declared(self):
        assert default_spell_keys([self._spell(slot=2)]) == {"x": "3"}

    def test_a_spell_beyond_the_first_row_gets_a_key_only_if_declared(self):
        """Sans raccourci declare, un sort de la troisieme ligne reste muet -- ce qui est
        prefere a une touche devinee, qui lancerait LE MAUVAIS SORT."""
        assert default_spell_keys([self._spell(slot=25)]) == {}
        assert default_spell_keys([self._spell(slot=25, key="ctrl+shift+2")])


class TestElementFocus:
    """Renforcement de l'element principal.

    En Dofus les degats dependent de la caracteristique de l'element : une Force elevee
    ne renforce QUE les sorts Terre. Les degats de base ne sont donc pas comparables
    entre elements des que le build est oriente -- or le solveur ne fait que comparer.

    Effet mesure sur la config reelle (Sacrieur Terre) : sans element il choisissait
    Desolation (Air, 26-30) ; avec, il choisit Bain de Sang (Terre, 27-31)."""

    def _kit(self):
        return [
            Spell(name="terrestre", ap_cost=3, range_max=6, element="terre",
                  damage_min=20, damage_max=24),
            Spell(name="aerien", ap_cost=3, range_max=6, element="air",
                  damage_min=26, damage_max=30),
        ]

    def test_matching_element_is_boosted(self):
        scaled = {s.name: s.damage_max
                  for s in apply_element_focus(self._kit(), "terre", factor=3.0)}
        assert scaled["terrestre"] == 72 and scaled["aerien"] == 30

    def test_the_ranking_can_flip(self):
        """Le point de la fonction : sans elle, le solveur prefere un sort hors-element
        pour quelques points de degats de base qu'il n'infligera jamais."""
        base = self._kit()
        assert base[0].average_damage < base[1].average_damage
        focused = apply_element_focus(base, "terre")
        assert focused[0].average_damage > focused[1].average_damage

    def test_no_element_changes_nothing(self):
        """Ne rien supposer par defaut : un build inconnu garde les valeurs du jeu."""
        assert apply_element_focus(self._kit(), None) == self._kit()

    def test_case_is_ignored(self):
        assert apply_element_focus(self._kit(), "TERRE")[0].damage_max > 24

    def test_the_original_list_is_untouched(self):
        """Rend des copies : on doit pouvoir comparer avec les valeurs du jeu."""
        kit = self._kit()
        apply_element_focus(kit, "terre")
        assert kit[0].damage_max == 24

    def test_a_spell_without_element_is_left_alone(self):
        spells = [Spell(name="x", ap_cost=3, range_max=4, damage_min=10, damage_max=10)]
        assert apply_element_focus(spells, "terre")[0].damage_max == 10
