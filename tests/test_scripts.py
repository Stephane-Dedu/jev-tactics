"""Tout script du depot doit repondre a --help, et vite.

Trois y manquaient, et leurs echecs n'etaient pas du meme ordre :

  - `audit_constantes.py` et `audit_fonctions.py` n'avaient pas d'analyseur d'arguments,
    donc `--help` LANCAIT l'audit -- une vingtaine de minutes pour obtenir une aide ;
  - `panel.py` ouvrait la fenetre et bloquait dans sa boucle.

Aucun n'est un defaut de calcul, et c'est bien pour cela qu'aucune relecture ne les
attrape : il faut TAPER la commande. Ce test la tape pour les vingt-sept.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
# Large, mais fini : ce qu'on interdit est le script qui ne rend JAMAIS la main, pas le
# demarrage un peu lent d'un interpreteur qui importe OpenCV.
DELAI = 60.0


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_answers_help(script):
    if not SCRIPTS:                          # pragma: no cover - depot sans scripts
        pytest.skip("aucun script")
    try:
        issue = subprocess.run([sys.executable, str(script), "--help"],
                               capture_output=True, text=True, timeout=DELAI,
                               check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script.name} n'a pas rendu la main en {DELAI:.0f} s")
    sortie = (issue.stdout or "") + (issue.stderr or "")
    detail = sortie[:400]
    assert issue.returncode == 0, f"{script.name} : code {issue.returncode} — {detail}"
    assert "usage" in sortie.lower(), f"{script.name} n'affiche pas d'aide"


class TestTheRunbookDocumentsRealOptions:
    """Le runbook est ce qu'on suit pour un premier lancement. Une option qui n'existe
    pas y coute plus qu'une option absente : elle envoie taper une commande qui echoue,
    au moment ou l'on ne sait pas encore distinguer sa propre erreur d'un defaut du bot.

    La derive n'est pas theorique -- ce document annoncait « ~497 tests, ~3 s » alors que
    la suite en compte 1438 et dure pres de trois minutes, et affirmait qu'aucun test ne
    demande d'ecran alors que deux en demandent un depuis peu.
    """

    RUNBOOK = ROOT / "docs" / "RUNBOOK.md"

    def _cites(self):
        import re

        if not self.RUNBOOK.exists():
            pytest.skip("runbook absent")
        texte = self.RUNBOOK.read_text(encoding="utf-8")
        cites: dict[str, set[str]] = {}
        for m in re.finditer(r"python scripts/(\w+)\.py([^\n`]*)", texte):
            cites.setdefault(m.group(1), set()).update(re.findall(r"--[\w-]+", m.group(2)))
        return cites

    def test_every_cited_script_exists(self):
        absents = [nom for nom in self._cites() if not (ROOT / "scripts" / f"{nom}.py").exists()]
        assert absents == [], f"scripts cites mais absents : {absents}"

    def test_every_cited_option_exists(self):
        inconnues = []
        for nom, options in sorted(self._cites().items()):
            chemin = ROOT / "scripts" / f"{nom}.py"
            if not chemin.exists():
                continue
            issue = subprocess.run([sys.executable, str(chemin), "--help"],
                                   capture_output=True, text=True, timeout=DELAI,
                                   check=False)
            aide = (issue.stdout or "") + (issue.stderr or "")
            inconnues += [f"{nom}.py {opt}" for opt in sorted(options) if opt not in aide]
        assert inconnues == [], f"options documentees mais inconnues : {inconnues}"
