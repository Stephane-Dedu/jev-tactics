"""Quelles constantes puis-je changer sans qu'aucun test ne s'en apercoive ?

Chacune est un endroit ou une modification future casserait le bot EN SILENCE. Le script
perturbe chaque constante numerique de module (+50 %, ou +2 pour les petits entiers),
rejoue les tests, puis restaure la valeur.

    vert  = personne ne garde cette valeur
    rouge = un test la tient

Premier passage : 80 « libres » sur 120. Ce chiffre est TROP FLATTEUR et le script le
dit -- il ne rejoue que les tests du module concerne, jusqu'a quatre fichiers, pour tenir
en quelques minutes. Une garde ecrite ailleurs lui echappe : `test_integration.py` ne
contient pas le mot « orchestrator », donc la garde de UNKNOWN_CONFIRMATIONS n'etait meme
pas executee.

Contre-verification de 14 constantes contre la suite COMPLETE : 13 libres quand meme,
seule SAFETY_WEIGHT est tenue (par le banc d'arene). Le resultat cible est donc une borne
BASSE du probleme, pas une exageration.

Ce n'est pas une liste de defauts : beaucoup de ces valeurs sont des soupapes, ou desserrer
ne casse rien de visible. C'est une liste d'endroits ou une mesure a ete faite puis laissee
sans temoin -- et cette session a montre ce que cela coute (CONFIRM_TOLERANCE, abaissee de
24 a 6 sur mesure, que rien n'empechait de remonter).

    python scripts/audit_constantes.py
"""

import argparse
import pathlib
import re
import subprocess
import sys

RACINE = pathlib.Path("C:/Users/Piral/dofus")
PYTHON = str(RACINE / ".venv/Scripts/python.exe")
MOTIF = re.compile(r"^([A-Z][A-Z0-9_]{3,})\s*(?::\s*[\w\[\], ]+)?=\s*(-?\d+\.?\d*)\s*(?:#.*)?$")

# Constantes dont la valeur est un fait de format, pas un reglage : les perturber
# n'apprend rien.
IGNOREES = {"SCHEMA_VERSION", "GRID_CELLS", "MAP_WIDTH", "MAP_HEIGHT", "END_TURN",
            "CHANNELS", "SCALARS", "GLYPH_SIZE"}


def tests_pour(chemin):
    """Fichiers de test a rejouer pour ce module source."""
    nom = pathlib.Path(chemin).stem
    candidats = [RACINE / "tests" / f"test_{nom}.py"]
    paquet = pathlib.Path(chemin).parent.name
    candidats.append(RACINE / "tests" / f"test_{paquet}.py")
    trouves = [str(c) for c in candidats if c.exists()]
    # Le filet : les fichiers qui nomment explicitement le module.
    for t in sorted((RACINE / "tests").glob("test_*.py")):
        if str(t) in trouves:
            continue
        if nom in t.read_text(encoding="utf-8"):
            trouves.append(str(t))
    return trouves[:4]


def perturbe(valeur):
    if "." in valeur:
        return f"{float(valeur) * 1.5 + 0.1:.4f}"
    entier = int(valeur)
    return str(entier + 2 if abs(entier) < 8 else int(entier * 1.5) + 1)


def main():
    resultats = []
    sources = sorted((RACINE / "src" / "jev_tactics").rglob("*.py"))
    for source in sources:
        texte = source.read_text(encoding="utf-8")
        lignes = texte.splitlines()
        for i, ligne in enumerate(lignes):
            m = MOTIF.match(ligne.strip())
            if not m or m.group(1) in IGNOREES:
                continue
            nom, valeur = m.group(1), m.group(2)
            cibles = tests_pour(source)
            if not cibles:
                resultats.append((source, nom, valeur, "AUCUN TEST"))
                continue
            modifiees = list(lignes)
            modifiees[i] = ligne.replace(f"= {valeur}", f"= {perturbe(valeur)}", 1)
            source.write_text("\n".join(modifiees) + "\n", encoding="utf-8")
            try:
                issue = subprocess.run(
                    [PYTHON, "-m", "pytest", "-x", "-q", "--no-header", *cibles],
                    cwd=RACINE, capture_output=True, timeout=300,
                    check=False)   # l'echec EST le resultat cherche
                verdict = "libre" if issue.returncode == 0 else "gardee"
            except subprocess.TimeoutExpired:
                verdict = "timeout"
            finally:
                source.write_text(texte, encoding="utf-8")
            resultats.append((source, nom, valeur, verdict))
            print(f"{verdict:9s} {source.name}:{nom} = {valeur}", flush=True)

    libres = [r for r in resultats if r[3] != "gardee"]
    print(f"\n== {len(libres)} constante(s) sans garde sur {len(resultats)}")
    for source, nom, valeur, verdict in libres:
        print(f"   [{verdict}] {source.as_posix().split('jev_tactics/')[1]} : {nom} = {valeur}")


def _arguments() -> None:
    """Repond a --help sans rien lancer.

    Sans analyseur, `--help` DEMARRAIT l'audit -- une vingtaine de minutes pour obtenir
    une aide. Constate en interrogeant les 27 scripts du depot : ces deux-la etaient les
    seuls a ne pas repondre.
    """
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()


if __name__ == "__main__":
    _arguments()
    sys.exit(main())
