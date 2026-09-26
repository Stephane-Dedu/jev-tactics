"""La CHIRURGIE de l'audit de fonctions, verifiee avant qu'on croie ses verdicts.

L'audit remplace le corps d'une fonction par un retour plausible et rejoue la suite : vert
= personne ne garde ce comportement. Tout repose donc sur `remplace` -- et un remplacement
qui echoue silencieusement rendrait « gardee » pour la mauvaise raison, ou pire, casserait
le fichier et rendrait « gardee » partout.

Le motif d'origine exigeait la COLONNE ZERO. Aucune methode n'etait donc auditable, alors
qu'une part croissante du paquet vit dans des classes -- la machine a etats de la recolte,
le graphe des cartes, le journal des engagements.
"""

import codecs
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location(
        "audit_fonctions", ROOT / "scripts" / "audit_fonctions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = '''\
def avant():
    return 1


def cible(a, b) -> int:
    """Un docstring."""
    calcul = a + b
    return calcul


def apres():
    return 2
'''

CLASSE = '''\
class Chose:
    """Une classe."""

    attribut: int = 3

    def cible(self, a) -> int:
        """Un docstring."""
        return a * 2

    def voisine(self):
        return "intacte"


def dehors():
    return 4
'''


class TestModuleLevelFunctions:
    def test_the_body_is_replaced(self, audit):
        sortie = audit.remplace(MODULE, "cible", "    return None")
        assert "calcul = a + b" not in sortie
        assert "    return None" in sortie

    def test_the_signature_survives(self, audit):
        """La remplacer changerait plus que le comportement : les appelants planteraient,
        et l'audit mesurerait un plantage au lieu d'une surveillance."""
        assert "def cible(a, b) -> int:" in audit.remplace(MODULE, "cible", "    return None")

    def test_the_docstring_survives(self, audit):
        assert '"""Un docstring."""' in audit.remplace(MODULE, "cible", "    return None")

    def test_the_neighbours_survive(self, audit):
        sortie = audit.remplace(MODULE, "cible", "    return None")
        assert "return 1" in sortie and "return 2" in sortie

    def test_an_unknown_name_is_reported(self, audit):
        """None, et non une exception : l'audit doit dire « introuvable » et continuer."""
        assert audit.remplace(MODULE, "absente", "    return None") is None


class TestMethods:
    """Ce que l'audit ne savait pas atteindre."""

    def test_a_method_body_is_replaced(self, audit):
        sortie = audit.remplace(CLASSE, "cible", "    return None")
        assert sortie is not None, "methode introuvable : le motif exige la colonne zero"
        assert "return a * 2" not in sortie

    def test_the_replacement_is_reindented(self, audit):
        """Un corps colle a la colonne quatre dans une classe est une IndentationError --
        le fichier ne s'importe plus, la suite echoue partout, et l'audit conclut
        « gardee » pour toutes les fonctions suivantes."""
        sortie = audit.remplace(CLASSE, "cible", "    return None")
        assert "        return None" in sortie
        compile(sortie, "<classe>", "exec")           # doit rester du Python valide

    def test_the_rest_of_the_class_survives(self, audit):
        """S'arreter au prochain `def` de colonne zero avalerait toute la fin de la
        classe -- et l'audit mesurerait la suppression de dix methodes, pas d'une."""
        sortie = audit.remplace(CLASSE, "cible", "    return None")
        assert "def voisine" in sortie and 'return "intacte"' in sortie
        assert "def dehors" in sortie and "return 4" in sortie

    def test_the_class_attributes_survive(self, audit):
        assert "attribut: int = 3" in audit.remplace(CLASSE, "cible", "    return None")


class TestNoSourceFileCarriesAByteOrderMark:
    """Un BOM UTF-8 est invisible et casse toute relecture en texte.

    Trois fichiers en portaient un -- `farming/runner.py`, `tests/test_farming.py`,
    `tests/test_integration.py` -- sans que rien ne le signale : le chargeur de Python
    decode les sources en utf-8-sig, donc les imports marchaient. C'est `compile()` sur le
    texte lu en utf-8 qui refuse, et c'est ce que fait l'audit de fonctions ci-dessous.

    Le test qui l'a trouve ne couvre que les fichiers inscrits dans `CIBLES` ; celui-ci
    couvre le depot. Un defaut qui ne se voit que par un outil de mesure se reintroduit sans
    bruit -- il faut le garder la ou il peut apparaitre, pas la ou il a ete vu.
    """

    def test_the_repository_is_free_of_them(self):
        portent = [f.relative_to(ROOT).as_posix()
                   for motif in ("src/**/*.py", "tests/**/*.py", "scripts/**/*.py")
                   for f in ROOT.glob(motif)
                   if f.read_bytes().startswith(codecs.BOM_UTF8)]
        assert not portent, f"BOM UTF-8 en tete de : {portent}"

    def test_the_check_would_see_one(self):
        """CONTRE-EPREUVE : une verification qui ne detecterait rien passerait toujours."""
        assert "# rien".encode("utf-8-sig").startswith(codecs.BOM_UTF8)

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


class TestTheTargetsExist:
    """Une cible introuvable est signalee a l'execution mais n'echoue pas : la table peut
    donc vieillir en silence, et un audit qui n'audite plus rien passerait pour un audit
    vert. C'est arrive -- la table datait d'avant tout le travail de chasse."""

    def test_every_target_is_found_in_its_module(self, audit):
        manquantes = []
        for relatif, nom, corps in audit.CIBLES:
            source = ROOT / "src" / "jev_tactics" / relatif
            if not source.exists():
                if _reste_en_amont(relatif):
                    continue
                manquantes.append(f"{relatif} (fichier absent)")
                continue
            if audit.remplace(source.read_text(encoding="utf-8"), nom, corps) is None:
                manquantes.append(f"{relatif}:{nom}")
        assert not manquantes, f"cibles introuvables : {manquantes}"

    def test_every_replacement_still_compiles(self, audit):
        """Un remplacement qui casse le fichier ferait rougir la suite entiere, et
        l'audit rendrait « gardee » -- le verdict le plus rassurant, pour la pire des
        raisons."""
        casses = []
        for relatif, nom, corps in audit.CIBLES:
            source = ROOT / "src" / "jev_tactics" / relatif
            if not source.exists():
                continue
            modifie = audit.remplace(source.read_text(encoding="utf-8"), nom, corps)
            if modifie is None:
                continue
            try:
                compile(modifie, str(source), "exec")
            except SyntaxError as erreur:
                casses.append(f"{relatif}:{nom} — {erreur}")
        assert not casses, f"remplacements invalides : {casses}"
