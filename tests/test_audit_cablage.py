"""La CHIRURGIE de l'audit de cablage, verifiee avant qu'on croie ses verdicts.

Cet audit retire une INSTRUCTION et rejoue la suite : vert = personne ne garde ce
branchement. Il comble le trou entre les deux autres audits -- une fonction peut etre
parfaitement gardee et n'etre APPELEE nulle part au bon moment, et c'est exactement la forme
des deux defauts qu'il a trouves.

Tout repose donc sur `mute`, et sur la FRAICHEUR de la table. Une cible perimee est signalee
a l'execution mais n'echoue pas : la table peut vieillir en silence, et un audit qui
n'audite plus rien passerait pour un audit vert. C'est deja arrive a `audit_fonctions.py`.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location(
        "audit_cablage", ROOT / "scripts" / "audit_cablage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = """\
def marche():
    etat = 1
    etat = 0
    return etat
"""


class TestTheSurgery:
    def test_a_statement_is_removed(self, audit):
        assert audit.mute(SOURCE, "    etat = 0\n", "", False) == (
            "def marche():\n    etat = 1\n    return etat\n")

    def test_only_the_first_by_default(self, audit):
        """Neutraliser TOUTES les occurrences ou une SEULE ne mesure pas la meme chose : la
        premiere dit « ce site precis est-il garde », la seconde « ce branchement l'est-il
        quelque part ». Les deux defauts trouves avaient trois sites, dont certains gardes
        et d'autres non."""
        sortie = audit.mute("a = 1\na = 1\n", "a = 1\n", "", False)
        assert sortie == "a = 1\n"

    def test_all_occurrences_when_asked(self, audit):
        assert audit.mute("a = 1\na = 1\n", "a = 1\n", "", True) == ""

    def test_an_absent_pattern_is_reported(self, audit):
        """None, et non une exception : l'audit doit dire « introuvable » et continuer."""
        assert audit.mute(SOURCE, "absent = 3\n", "", False) is None

# Modules restes dans le depot d'origine avec la recolte : ce depot ne porte que le
# combat. Une cible qui les vise n'est pas une cible PERIMEE -- elle surveille du code
# encore vivant, ailleurs. La distinction compte : on veut continuer d'echouer sur une
# cible morte de CE depot, et seulement taire celles que le portage a laissees derriere.
HORS_PERIMETRE = (
    "farming/", "bot/", "market/", "ui/", "diagnose", "overlay",
    # scripts restes en amont parce qu'ils importent ces modules-la
    "scripts/farm.py", "scripts/doctor.py", "scripts/panel.py", "scripts/hdv.py",
    "scripts/play_combat.py", "scripts/play_turn.py", "scripts/replay.py",
    "scripts/spells.py", "scripts/endurance.py", "scripts/study_engagements.py",
)


def _reste_en_amont(relatif: str) -> bool:
    return any(part in relatif.replace("\\", "/") for part in HORS_PERIMETRE)


class TestTheTargetsAreFresh:
    """Une cible introuvable est signalee mais n'echoue pas. Sans ce test, la table
    vieillit en silence et l'audit rend « 0 neutralisable » en n'ayant rien mute."""

    def test_every_target_is_found_in_its_file(self, audit):
        manquantes = []
        for nom, relatif, motif, remplacement, partout in audit.CIBLES:
            source = ROOT / relatif
            if not source.exists():
                if _reste_en_amont(relatif):
                    continue
                manquantes.append(f"{nom} ({relatif} absent)")
                continue
            if audit.mute(source.read_text(encoding="utf-8"),
                          motif, remplacement, partout) is None:
                manquantes.append(nom)
        assert not manquantes, f"cibles introuvables : {manquantes}"

    def test_every_mutation_still_compiles(self, audit):
        """Une mutation qui casse le fichier ferait rougir la suite entiere, et l'audit
        rendrait « gardee » -- le verdict le plus rassurant, pour la pire des raisons."""
        casses = []
        for nom, relatif, motif, remplacement, partout in audit.CIBLES:
            source = ROOT / relatif
            if not source.exists():
                continue
            modifie = audit.mute(source.read_text(encoding="utf-8"),
                                 motif, remplacement, partout)
            if modifie is None:
                continue
            try:
                compile(modifie, str(source), "exec")
            except SyntaxError as erreur:
                casses.append(f"{nom} — {erreur}")
        assert not casses, f"mutations invalides : {casses}"

    def test_a_mutation_actually_changes_the_file(self, audit):
        """CONTRE-EPREUVE des deux precedents : une mutation qui rendrait le fichier
        INCHANGE serait trouvee, compilerait, et ne mesurerait rien."""
        inertes = []
        for nom, relatif, motif, remplacement, partout in audit.CIBLES:
            source = ROOT / relatif
            if not source.exists():
                continue
            origine = source.read_text(encoding="utf-8")
            modifie = audit.mute(origine, motif, remplacement, partout)
            if modifie is not None and modifie == origine:
                inertes.append(nom)
        assert not inertes, f"mutations sans effet : {inertes}"
