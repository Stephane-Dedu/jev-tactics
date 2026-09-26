"""Quelles fonctions puis-je NEUTRALISER sans qu'aucun test ne rougisse ?

Analogue de `scripts/audit_constantes.py`, applique au CODE plutot qu'aux valeurs. Chaque
fonction est remplacee par un retour constant plausible -- ensemble vide, None, False --
puis la suite est rejouee. Vert = personne ne garde ce comportement.

Le retour est PLAUSIBLE et non une exception : une exception serait attrapee par n'importe
quel test qui passe par la, ce qui mesurerait la couverture et non la surveillance.

Passage sur 23 fonctions : UNE SEULE neutralisable, `hovered_cell`, et la cause valait le
detour. Son unique test direct affirmait

    assert hovered_cell(board) is None or True

c'est-a-dire toujours vrai -- il dependait de la souris reelle, donc ne pouvait rien
affirmer. Le second test la remplacait par un faux pour verifier le pipeline, ce qui est
legitime mais ne garde pas la vraie. Corrige en simulant GetCursorPos.

Le second lot -- decision, regles, plus proche voisin -- est integralement garde : 0 sur 12.
C'est le resultat attendu d'une suite qui mesure des COMPORTEMENTS, et il vaut d'etre
consigne, parce qu'il dit ou il est inutile de rajouter des tests.

TROISIEME LOT : la chasse et le pilotage sans surveillance, 14 fonctions et methodes.
**0 neutralisable sur 14.** Verifie sur une base verte (1675 tests) et rejoue apres coup :
1675 de nouveau, donc aucune mutation laissee en place.

    audit_spell_config · confirmations_for · _turn_limit_reason · _circuit_propose
    diagnose_hunting · suggest_route · save_suggested_route · repeated_spots
    unlocated_misses · record_fight · harvestable · GroupLimits.accepts
    record_visit · barren_on

Deux choses rendaient ce lot inauditable, et elles comptent plus que le resultat :

  - **la table avait vieilli.** Elle datait d'avant tout ce travail, et une cible absente
    est signalee mais n'echoue pas. Un audit qui n'audite plus rien passe donc pour vert.
    `tests/test_audit_fonctions.py` verifie desormais que CHAQUE cible est trouvee.
  - **les methodes etaient invisibles.** Le motif exigeait la colonne zero, si bien que
    tout ce qui vit dans une classe -- `FarmingSession`, `MapGraph`, `EngagementJournal`,
    `GroupLimits` -- echappait a la mesure. C'est la moitie de ce lot.

QUATRIEME LOT : ce que la chasse a gagne depuis. **0 neutralisable sur 4.**

    _mark_trails · beyond_recorded · reach_probability · _settle_engagement

Le marquage des lobes de depart, l'enveloppe des combats enregistres, le calcul de chance
d'atteinte et le verdict differe d'engagement. Aucune n'etait dans cette table : une
fonction ABSENTE est invisible a l'audit, et rien ne le signale -- c'est le meme angle mort
que « la table avait vieilli », a ceci pres qu'une cible jamais ecrite ne peut meme pas etre
signalee introuvable. La seule parade est d'inscrire chaque fonction en meme temps qu'on
l'ecrit.

CE QUE LEUR INSCRIPTION A FAIT SORTIR, et ce n'etait pas le sujet. `test_audit_fonctions.py`
verifie que chaque remplacement COMPILE ; celui de `_settle_engagement` echouait sur
« invalid non-printable character U+FEFF ». Trois fichiers du depot portaient un BOM UTF-8 :
`farming/runner.py`, `tests/test_farming.py`, `tests/test_integration.py`. Python les
importe sans broncher -- le chargeur decode en utf-8-sig -- mais `compile()` sur leur texte
lu en utf-8 refuse, et tout outil qui relit un fichier comme du texte tombe dessus. Retires.

    python scripts/audit_fonctions.py

Compter une vingtaine de minutes : chaque fonction non gardee coute une suite complete,
les autres echouent vite grace a `-x`.
"""

import argparse
import pathlib
import re
import subprocess
import sys

RACINE = pathlib.Path("C:/Users/Piral/dofus")
PYTHON = str(RACINE / ".venv/Scripts/python.exe")

# (module, fonction, corps de remplacement) -- un retour plausible, pas une exception :
# une exception serait attrapee par n'importe quel test qui passe par la, ce qui ne
# mesurerait que la couverture, pas la surveillance du COMPORTEMENT.
CIBLES = [
    ("perception/highlight.py", "read_movement_range", "    return set()"),
    ("perception/highlight.py", "read_placement_zone", "    return set()"),
    ("perception/highlight.py", "movement_centre", "    return None"),
    ("perception/highlight.py", "looks_like_spell_range", "    return False"),
    ("perception/entities.py", "hovered_cell", "    return None"),
    ("perception/deaths.py", "death_marks", "    return []"),
    ("perception/tooltip.py", "is_depleted", "    return False"),
    ("perception/ui.py", "hidden_by_ui", "    return False"),
    ("perception/screen.py", "classify_screen", "    return ScreenState.MAP"),
    ("perception/mobinfo.py", "read_mob_hp", "    return MobReading()"),
    ("farming/navigation.py", "map_changed", "    return True"),
    ("farming/navigation.py", "direction_of", "    return None"),
    # Second lot : decision et regles, la ou une neutralisation ne casse rien de visible
    # mais fait jouer n'importe quoi.
    ("planner/prune.py", "offensive", "    return list(spells)"),
    ("planner/prune.py", "useful_targets", "    return set()"),
    ("planner/search.py", "approach_move", "    return None"),
    ("planner/scoring.py", "expected_damage", "    return 0.0"),
    ("state/assemble.py", "assemble_combat_state", "    return None"),
    ("rules/spells.py", "apply_element_focus", "    return list(spells)"),
    ("rules/spells.py", "default_spell_keys", "    return {}"),
    ("farming/session.py", "_same_spot", "    return False"),
    ("perception/resources.py", "nearest_resource", "    return None"),
    ("perception/monsters.py", "best_group", "    return None"),
    ("perception/monsters.py", "_anchor", "    return round(cx), round(cy)"),
    ("perception/timeline.py", "read_timeline", "    return []"),
    # Troisieme lot : tout ce que la chasse et le pilotage sans surveillance ont ajoute.
    # Aucune de ces fonctions n'etait auditee -- la table datait d'avant elles, et la
    # moitie sont des METHODES, que le motif ne savait pas atteindre.
    ("rules/spells.py", "audit_spell_config", "    return []"),
    ("bot/orchestrator.py", "confirmations_for", "    return UNKNOWN_CONFIRMATIONS"),
    ("bot/runner.py", "_turn_limit_reason", "    return ''"),
    ("bot/diagnostics.py", "session_diagnostics", "    return []"),
    ("bot/diagnostics.py", "_circuit_propose", "    return []"),
    ("diagnose.py", "diagnose_hunting", "    return Diagnosis()"),
    ("farming/navigation.py", "suggest_route", "    return None"),
    ("farming/navigation.py", "save_suggested_route", "    return ''"),
    ("farming/navigation.py", "record_visit", "    return None"),
    ("farming/navigation.py", "barren_on", "    return []"),
    ("farming/journal.py", "repeated_spots", "    return []"),
    ("farming/journal.py", "unlocated_misses", "    return 0"),
    ("farming/session.py", "record_fight", "    return None"),
    ("farming/session.py", "harvestable", "    return list(resources)"),
    ("perception/monsters.py", "accepts", "    return True"),
    # QUATRIEME LOT : ce que la chasse a gagne depuis le troisieme audit -- le marquage des
    # lobes de depart, l'enveloppe des combats enregistres, le calcul de chance d'atteinte
    # et le verdict differe d'engagement. Aucune n'etait auditee, la table datant d'avant
    # elles : une fonction absente de cette liste est invisible a l'audit, et rien ne le
    # signale.
    ("perception/monsters.py", "_mark_trails", "    return groups"),
    ("perception/monsters.py", "beyond_recorded", "    return False"),
    ("perception/monsters.py", "reach_probability", "    return 1.0"),
    ("farming/runner.py", "_settle_engagement", "    return None"),
]


def remplace(texte, nom, corps):
    """Remplace le corps de `nom` par `corps`, en gardant sa signature.

    Accepte AUSSI les methodes, c'est-a-dire les `def` indentes. L'audit n'en voyait
    aucune : son motif exigeait la colonne zero, si bien que tout ce qui vit dans une
    classe -- la machine a etats de la recolte, le graphe des cartes, le journal des
    engagements -- echappait a la mesure. C'est une part croissante du paquet, et pas la
    moins decisive.

    Le corps fourni est ecrit pour une fonction de module (indente de quatre) ; on le
    REINDENTE sur le `def` trouve, de sorte que la table des cibles reste lisible et que
    fonctions et methodes s'y ecrivent pareil.
    """
    motif = re.compile(rf"^([ \t]*)def {nom}\((.*?)\)( -> [^:]+)?:\n",
                       re.MULTILINE | re.DOTALL)
    m = motif.search(texte)
    if not m:
        return None
    marge = m.group(1)
    if marge:
        corps = "\n".join(marge + ligne if ligne.strip() else ligne
                          for ligne in corps.split("\n"))
    debut = m.end()
    # La fin du corps est le prochain `def` DE MEME NIVEAU ou moins indente : pour une
    # methode, s'arreter au prochain `\ndef ` avalerait toute la fin de la classe.
    suivant = re.compile(rf"^[ \t]{{0,{len(marge)}}}(def |@|class )", re.MULTILINE)
    trouve = suivant.search(texte, debut)
    fin = trouve.start() if trouve else len(texte)
    suite = texte[debut:fin]
    # Conserver le docstring : le retirer changerait plus que le comportement.
    doc = ""
    if suite.lstrip().startswith(('"""', "'''")):
        marque = suite.lstrip()[:3]
        i = suite.find(marque)
        j = suite.find(marque, i + 3)
        if j > 0:
            doc = suite[: j + 3] + "\n"
    return texte[:debut] + doc + corps + "\n" + texte[fin:]


def main(filtre=None):
    libres = []
    cibles = [c for c in CIBLES if not filtre or filtre in f"{c[0]}:{c[1]}"]
    if not cibles:
        print(f"aucune cible ne correspond a « {filtre} »")
        return 1
    for relatif, nom, corps in cibles:
        source = RACINE / "src" / "jev_tactics" / relatif
        origine = source.read_text(encoding="utf-8")
        modifie = remplace(origine, nom, corps)
        if modifie is None:
            print(f"introuvable {relatif}:{nom}", flush=True)
            continue
        source.write_text(modifie, encoding="utf-8")
        try:
            issue = subprocess.run(
                [PYTHON, "-m", "pytest", "-x", "-q", "--no-header",
                 "-p", "no:cacheprovider", "tests/"],
                cwd=RACINE, capture_output=True, timeout=900, check=False)
            verdict = "LIBRE" if issue.returncode == 0 else "gardee"
        except subprocess.TimeoutExpired:
            verdict = "timeout"
        finally:
            source.write_text(origine, encoding="utf-8")
        if verdict != "gardee":
            libres.append(f"{relatif}:{nom}")
        print(f"{verdict:7s} {relatif}:{nom}", flush=True)

    print(f"\n== {len(libres)} fonction(s) neutralisables sans qu'un test ne rougisse")
    for nom in libres:
        print(f"   {nom}")


def _arguments():
    """Repond a --help sans rien lancer.

    Sans analyseur, `--help` DEMARRAIT l'audit -- une vingtaine de minutes pour obtenir
    une aide. Constate en interrogeant les 27 scripts du depot : ces deux-la etaient les
    seuls a ne pas repondre.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filtre", default=None,
                    help="n'auditer que les cibles dont « module:fonction » contient ce "
                         "motif. L'audit complet coute une vingtaine de minutes ; apres "
                         "une modification on ne veut relire qu'une poignee de fonctions")
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(main(_arguments().filtre))
