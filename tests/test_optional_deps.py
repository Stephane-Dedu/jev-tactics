"""Le paquet doit s'importer avec les SEULES dependances obligatoires.

Ce test existe parce que l'inverse est arrive et ne s'est vu nulle part en local.
`perception/ui.py` importait `pytesseract` en tete de fichier ; `perception/__init__`
importe `ui` ; donc tout le paquet exigeait un extra declare comme facultatif. Le depot
s'installait selon son propre README -- `pip install -e ".[dev]"` -- et la suite entiere
tombait en erreur de COLLECTE : 31 fichiers, pas un seul test execute.

Rien ne le signalait sur la machine d'origine, dont le venv avait Tesseract par habitude.
La premiere CI l'a trouve en une minute, en n'installant que ce qui est annonce.

On teste ici les extras ENSEMBLE plutot que le seul coupable : le defaut n'est pas
« pytesseract etait mal importe », c'est « un extra peut redevenir obligatoire sans que
personne le remarque ».
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Tout ce que `pyproject.toml` declare hors de `dependencies`.
OPTIONAL = ("pytesseract", "pydirectinput", "typesafe_sdk", "torch", "ultralytics")

_PROBE = textwrap.dedent(
    """
    import importlib
    import pkgutil
    import sys

    BLOCKED = {blocked!r}


    class Blocker:
        \"\"\"Fait disparaitre les extras, comme sur une installation propre.\"\"\"

        def find_module(self, name, path=None):
            root = name.split(".")[0]
            return self if root in BLOCKED else None

        def load_module(self, name):
            raise ImportError(f"{{name}} indisponible (simule)")


    sys.meta_path.insert(0, Blocker())

    import jev_tactics

    failed = []
    for info in pkgutil.walk_packages(jev_tactics.__path__, "jev_tactics."):
        try:
            importlib.import_module(info.name)
        except ImportError as exc:
            failed.append(f"{{info.name}} -> {{exc}}")

    print("FAILED:" + "|".join(failed) if failed else "OK")
    """
)


def _probe(blocked: tuple[str, ...]) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(blocked=blocked)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"la sonde elle-meme a echoue :\n{result.stderr[-2000:]}")
    return result.stdout.strip().splitlines()[-1]


def test_every_module_imports_without_any_optional_extra():
    """LE TEST QUI MANQUAIT. Un sous-processus, parce qu'un extra deja importe par la
    suite courante ne peut pas etre retire du processus en cours."""
    verdict = _probe(OPTIONAL)
    assert verdict == "OK", (
        "des modules exigent un extra facultatif :\n  "
        + "\n  ".join(verdict.removeprefix("FAILED:").split("|"))
    )


@pytest.mark.parametrize("extra", OPTIONAL)
def test_no_single_extra_is_secretly_required(extra):
    """Un par un : dit LEQUEL redevient obligatoire, la ou le test groupe dit seulement
    que quelque chose l'est."""
    verdict = _probe((extra,))
    assert verdict == "OK", (
        f"{extra} est devenu obligatoire :\n  "
        + "\n  ".join(verdict.removeprefix("FAILED:").split("|"))
    )
