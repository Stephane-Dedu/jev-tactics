"""Lire un fichier `.env`, sans dependance et sans surprise.

Le depot n'avait aucun moyen de recevoir une cle API autrement qu'en exportant une
variable a chaque session. Un `.env` etait deja ignore par `.gitignore` -- par precaution,
avant que quoi que ce soit ne sache le lire -- ce qui est le pire etat : on croit la cle
prise en compte, et rien ne la lit.

`python-dotenv` ferait l'affaire, mais ajouter une dependance d'execution pour une
variable serait disproportionne : le format utile tient en quinze lignes.

TROIS REGLES, et chacune evite une panne connue de ce genre de fichier :

  - **l'environnement reel gagne toujours.** Un `.env` qui ecraserait `TYPESAFE_API_KEY`
    deja exportee rendrait impossible de tester avec une autre cle sans editer un fichier,
    et surtout la CI ne pourrait plus imposer la sienne ;
  - **la valeur n'est jamais journalisee.** Les fonctions d'ici rendent des booleens et des
    noms de variables, jamais des valeurs. Une cle apparue une fois dans une trace est une
    cle a revoquer ;
  - **un fichier absent n'est pas une erreur.** La grande majorite des usages -- toute la
    suite de tests -- n'a besoin d'aucune cle.
"""

from __future__ import annotations

import os
from pathlib import Path

# Racine du depot : ce fichier est dans src/jev_tactics/, donc deux niveaux au-dessus.
ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

_loaded = False


def load_env(path: Path | None = None, override: bool = False) -> list[str]:
    """Charge `.env` dans l'environnement. -> les noms des variables POSEES.

    Rend les noms et non les valeurs, pour qu'un appelant puisse dire « j'ai lu la cle »
    sans pouvoir l'afficher par accident.

    Idempotent : plusieurs appels ne relisent le fichier qu'une fois, sauf `path` explicite
    (les tests en ont besoin).
    """
    global _loaded
    target = path or ENV_FILE
    if path is None:
        if _loaded:
            return []
        _loaded = True

    if not target.exists():
        return []

    posees: list[str] = []
    for ligne in target.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        nom, _, valeur = ligne.partition("=")
        nom = nom.strip()
        # `export FOO=bar` est ce que les gens copient depuis une documentation shell.
        nom = nom.removeprefix("export ").strip()
        valeur = valeur.strip().strip('"').strip("'")
        if not nom:
            continue
        if nom in os.environ and not override:
            continue        # l'environnement reel fait autorite
        os.environ[nom] = valeur
        posees.append(nom)
    return posees


def api_key(name: str = "TYPESAFE_API_KEY") -> str | None:
    """La cle, depuis l'environnement ou depuis `.env`. None si absente.

    C'est le seul point d'entree a utiliser : il garantit que `.env` a ete tente avant de
    conclure a une absence. Chercher `os.environ` directement donnerait « cle absente »
    sur une machine ou le fichier existe.
    """
    load_env()
    return os.environ.get(name) or None


def describe_key(name: str = "TYPESAFE_API_KEY") -> str:
    """Une ligne de diagnostic qui ne divulgue rien.

    Ni la valeur, ni sa longueur exacte -- seulement sa presence et sa provenance. Un
    script peut donc dire ce qu'il a trouve sans qu'une copie d'ecran devienne une fuite.
    """
    load_env()
    if name not in os.environ or not os.environ[name]:
        emplacement = ENV_FILE if ENV_FILE.exists() else "aucun .env"
        return f"{name} absente ({emplacement})"
    origine = "environnement ou .env"
    return f"{name} presente ({origine})"
