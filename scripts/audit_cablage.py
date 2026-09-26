"""Quelle INSTRUCTION puis-je neutraliser sans qu'aucun test ne rougisse ?

Troisieme audit du depot, et il comble un trou entre les deux autres :

    audit_constantes.py   perturbe une VALEUR          « ce nombre est-il tenu ? »
    audit_fonctions.py    vide une FONCTION entiere    « ce calcul est-il tenu ? »
    audit_cablage.py      retire UNE INSTRUCTION       « ce branchement est-il tenu ? »

Une fonction peut etre parfaitement gardee et n'etre APPELEE nulle part au bon moment. Les
deux defauts trouves par cette methode sont exactement de cette forme, et aucun des deux
autres audits ne pouvait les voir -- la fonction existait, sa valeur etait juste, seul
l'appel manquait.

CE QU'IL A TROUVE, sept cablages sur le chemin de chasse :

    self._hunt_given_up_here = False        x3   le renoncement ne retombait jamais
    self._forget_unlocated_spots()          x3   la liste noire devenait globale
    report.hunt_scans += farm.hunt_scans         « on n'a jamais regarde » indiscernable
    report.implausible_group_scans += ...        l'avertissement « decor » se taisait
    report.biggest_rejected = max(...)           « rien n'a bouge » indiscernable
    print(ligne) dans scripts/farm.py            le bilan n'etait plus imprime
    le refus de DISPOSITION dans setup.py        toutes les ROIs pointaient a cote

Les deux premiers se neutralisaient en laissant 2067 tests verts, et ils ont la meme cause :
les tests du sujet REIMPLEMENTENT dans un stub ce que le vrai code fait, si bien qu'ils
verifient leur propre copie. L'un des stubs va jusqu'a ecrire en commentaire ce que l'oubli
coute, puis simule l'appel au lieu de le garder. Les pannes sont silencieuses et durables :
apres une seule carte abandonnee le bot ne chasse plus, ou clique indefiniment a cote.

Les trois suivants traversent la FRONTIERE entre le bilan d'un tour et celui de la session
-- celle qui a deja produit deux doubles comptages. Un chiffre perdu y vaut zero sur les
sessions memes qui en produisent le plus, puisque l'orchestrateur relance `run()` apres
chaque combat.

Le dernier est au bout de la chaine : `scripts/farm.py` est l'entree principale de la chasse
(`--hunt --execute`), et son affichage n'etait garde par rien. Construire un diagnostic et
ne pas l'imprimer revient exactement au meme que ne pas le construire. Le bilan a ete
deplace dans le paquet (`bot/diagnostics.farm_summary`), ou une dizaine de tests l'exercent.

Le septieme est le plus ancien controle du demarrage : le refus de jouer sur une disposition
inattendue. Toutes les zones du projet sont des pixels absolus releves sur 1919x1079 ;
ailleurs elles pointent a cote et TOUTES les lectures echouent ensemble, ce qui ressemble a
une panne de detection. Le module l'explique longuement -- et rien ne gardait le refus.

CE QU'IL A CONFIRME comme tenu, et qui vaut d'etre consigne pour ne pas y revenir :

    session      _hurt_refusals remis a zero · _fights_here incremente
    runner       serie inengageable · cartes abandonnees rapportees · engagements tardifs
                 hunt_scans · trail_scans · biggest_rejected
    perception   masque des panneaux · exclusion des zones d'interface · tri par aire
                 seuil de difference · fermeture morphologique · ancrage · connectivite 8

    frontiere    traces · engagements tardifs · cartes abandonnees · compte plausible
    journal      entree memorisee · ecriture disque
    chasse       plafond de la liste noire · paire du scan · paire consommee
                 decalage mesure
    demarrage    origine de capture · espace de clic · disposition
    entree       bilan de farm imprime

    python scripts/audit_cablage.py --filtre chasse
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

RACINE = pathlib.Path(__file__).resolve().parents[1]
PYTHON = str(RACINE / ".venv/Scripts/python.exe")

# SES PROPRES TESTS SONT ECARTES DE LA SUITE REJOUEE, et sans cela l'audit ment.
#
# `test_every_target_is_found_in_its_file` verifie que chaque motif de la table existe dans
# son fichier. Or une mutation qui SUPPRIME une ligne fait disparaitre ce motif : la garde
# de fraicheur rougit, pytest rend un code non nul, et l'audit conclut « gardee ». Toujours.
#
# Constate dans la journee qui a suivi l'ecriture de cet audit : trois cablages de la
# frontiere orchestrateur etaient rendus « gardee » sur la suite complete alors qu'aucun
# test ne les tenait. C'est exactement le mode de panne que `test_every_mutation_still_
# compiles` surveille chez le voisin -- le verdict le plus rassurant pour la pire raison --
# et il s'appliquait a l'outil lui-meme.
PROPRE_TEST = "tests/test_audit_cablage.py"

# (nom, fichier, motif, remplacement, toutes_les_occurrences)
#
# Le remplacement doit rester du Python VALIDE et plausible : retirer une ligne, ou la
# remplacer par un equivalent inerte. Une exception serait attrapee par n'importe quel test
# qui passe par la, ce qui mesurerait la couverture et non la surveillance.
CIBLES: list[tuple[str, str, str, str, bool]] = [
    # --- chasse : ce que `_travel` doit effacer en changeant de carte. Les deux defauts
    # trouves par cet audit. Ils sont gardes desormais ; on les garde ICI aussi, parce
    # qu'une cible retiree de la table est une surveillance qui s'arrete sans bruit.
    ("chasse/renoncement-carte", "src/jev_tactics/farming/runner.py",
     "self._hunt_given_up_here = False", "pass  # neutralise", True),
    ("chasse/oubli-positions", "src/jev_tactics/farming/runner.py",
     "self._forget_unlocated_spots()", "pass  # neutralise", True),
    # --- chasse : les compteurs qui expliquent une session a zero combat.
    ("chasse/serie-inengageable", "src/jev_tactics/farming/runner.py",
     "self._unengageable_maps += 1", "pass  # neutralise", True),
    ("chasse/cartes-abandonnees", "src/jev_tactics/farming/runner.py",
     "report.unengageable_maps += 1", "pass  # neutralise", True),
    ("chasse/engagements-tardifs", "src/jev_tactics/farming/runner.py",
     "report.late_engagements += 1", "pass  # neutralise", True),
    ("chasse/scans-comptes", "src/jev_tactics/farming/runner.py",
     "report.hunt_scans += 1", "pass  # neutralise", True),
    ("chasse/traces-comptees", "src/jev_tactics/farming/runner.py",
     "report.trail_scans += any(g.stale for g in groups)", "pass  # neutralise", True),
    ("chasse/refus-releves", "src/jev_tactics/farming/runner.py",
     "self._biggest_rejected = max([self._biggest_rejected, *refuses])",
     "pass  # neutralise", True),
    ("chasse/plafond-liste-noire", "src/jev_tactics/farming/runner.py",
     "        del connus[:-MAX_FAILED_SPOTS_PER_MAP]", "        pass  # neutralise", True),
    # --- chasse : la mesure de deplacement, seule piste physique trouvee pour distinguer
    # un monstre du decor. Elle est ENREGISTREE et pas encore utilisee pour ordonner ; un
    # cablage rompu rendrait une colonne vide, decouverte au moment de l'analyse.
    ("chasse/paire-du-scan", "src/jev_tactics/farming/runner.py",
     "        self._last_pair = (before, apres)", "        pass  # neutralise", True),
    ("chasse/paire-consommee", "src/jev_tactics/farming/runner.py",
     "        paire, self._last_pair = self._last_pair, None",
     "        paire = self._last_pair", True),
    ("chasse/decalage-mesure", "src/jev_tactics/farming/runner.py",
     "        decalage = phase_shift(*paire, group.box) if paire is not None else None",
     "        decalage = None", True),
    # Le CLIC doit rester colle a la derniere capture : tout ce qui s'intercale est du temps
    # pendant lequel la cible se deplace.
    ("chasse/clic-avant-mesures", "src/jev_tactics/farming/runner.py",
     "        self._click(group.x, group.y)", "        pass  # neutralise", True),
    # --- budget de temps : TROIS lignes identiques dans trois fichiers, et aucune n'etait
    # gardee. Le budget est PARTAGE par les trois boucles, et chacune le demarrait en tete
    # de son `run()` ; comme `start()` remettait le compteur a zero, l'orchestrateur --
    # qui relance la recolte a chaque alternance et le combat a chaque combat -- le
    # rembobinait plusieurs fois par minute. `--max-minutes` ne bornait donc RIEN en
    # session orchestree, et seule la borne en ALTERNANCES arretait le bot.
    #
    # `start()` est desormais idempotent, ce qui rend les trois appels corrects ensemble.
    # On les surveille ici parce que la panne etait invisible autrement : chacun pris
    # separement avait l'air juste, et aucun test ne passait de budget non demarre.
    ("budget/demarrage-orchestrateur", "src/jev_tactics/bot/orchestrator.py",
     "        self.budget.start()", "        pass  # neutralise", False),
    ("budget/demarrage-recolte", "src/jev_tactics/farming/runner.py",
     "        self.budget.start()", "        pass  # neutralise", False),
    ("budget/demarrage-combat", "src/jev_tactics/bot/runner.py",
     "        self.budget.start()", "        pass  # neutralise", False),
    # --- machine a etats.
    ("session/refus-pv-remis-a-zero", "src/jev_tactics/farming/session.py",
     "self._hurt_refusals = 0", "pass  # neutralise", True),
    ("session/combats-par-carte", "src/jev_tactics/farming/session.py",
     "self._fights_here += 1", "pass  # neutralise", True),
    # --- perception : les instructions de `detect_groups`, que l'audit de fonctions ne
    # voit pas puisqu'il remplace la fonction ENTIERE.
    ("perception/masque-panneaux", "src/jev_tactics/perception/monsters.py",
     "        mask[ui_mask(after) > 0] = 0", "        pass  # neutralise", False),
    ("perception/exclusion-ath", "src/jev_tactics/perception/monsters.py",
     ("        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in exclude):\n"
      "            continue\n"), "", False),
    ("perception/tri-par-aire", "src/jev_tactics/perception/monsters.py",
     "    return sorted(_mark_trails(groups, boxes, before, after), key=lambda g: -g.area)",
     "    return _mark_trails(groups, boxes, before, after)", False),
    ("perception/seuil-difference", "src/jev_tactics/perception/monsters.py",
     "    _, mask = cv2.threshold(delta, MIN_DIFFERENCE, 255, cv2.THRESH_BINARY)",
     "    _, mask = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY)", False),
    ("perception/fermeture", "src/jev_tactics/perception/monsters.py",
     "    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)", "    return mask",
     False),
    ("perception/ancrage", "src/jev_tactics/perception/monsters.py",
     ("        x, y = _anchor(labels, index, (left, top, width, height), "
      "*centroids[index])"),
     ("        x, y = round(float(centroids[index][0])), "
      "round(float(centroids[index][1]))"), False),
    ("perception/connectivite", "src/jev_tactics/perception/monsters.py",
     "        mask, connectivity=8)", "        mask, connectivity=4)", False),
    # LA DISTINCTION SUR LAQUELLE REPOSE TOUT LE CLASSEMENT : il ORDONNE, il ne refuse pas.
    # Transformer la relegation en refus donne le meme premier clic sur toutes les cartes
    # encombrees -- donc les memes tests verts -- et rend le cliquet de `beyond_recorded`
    # definitivement gele, y compris sur les cartes calmes ou il se desserre aujourd'hui.
    ("perception/relegation-pas-refus", "src/jev_tactics/perception/monsters.py",
     "    ox, oy = origin\n",
     ("    groups = [g for g in groups if not beyond_recorded(g)] or list(groups)\n"
      "    ox, oy = origin\n"), False),
    # --- orchestrateur : la FRONTIERE entre le bilan d'un tour et celui de la session.
    # Elle a deja produit deux doubles comptages (`unengageable_maps`, `failed_harvests`),
    # et un chiffre perdu la vaut zero sur les sessions memes qui en produisent le plus --
    # l'orchestrateur relance `run()` apres chaque combat.
    ("frontiere/scans-de-chasse", "src/jev_tactics/bot/orchestrator.py",
     "            report.hunt_scans += farm.hunt_scans\n", "", True),
    ("frontiere/traces", "src/jev_tactics/bot/orchestrator.py",
     "            report.trail_scans += farm.trail_scans\n", "", True),
    ("frontiere/engagements-tardifs", "src/jev_tactics/bot/orchestrator.py",
     "            report.late_engagements += farm.late_engagements\n", "", True),
    ("frontiere/cartes-abandonnees", "src/jev_tactics/bot/orchestrator.py",
     "            report.unengageable_maps += farm.unengageable_maps\n", "", True),
    ("frontiere/scans-invraisemblables", "src/jev_tactics/bot/orchestrator.py",
     "            report.implausible_group_scans += farm.implausible_group_scans\n",
     "", True),
    ("frontiere/compte-plausible", "src/jev_tactics/bot/orchestrator.py",
     ("            report.busiest_plausible_scan = max(report.busiest_plausible_scan,\n"
      "                                                farm.busiest_plausible_scan)\n"),
     "", True),
    ("frontiere/plus-gros-refus", "src/jev_tactics/bot/orchestrator.py",
     ("            report.biggest_rejected = max(report.biggest_rejected,\n"
      "                                          farm.biggest_rejected)\n"), "", True),
    # --- journal : la seule verite terrain de la detection de monstres. Ce qui n'y entre
    # pas n'existe pas, et rien ne le signale.
    ("journal/entree-memorisee", "src/jev_tactics/farming/journal.py",
     "        self.entries.append(entry)\n", "", True),
    # --- demarrage : les trois hypotheses les plus basses de la chaine de clic. Chacune,
    # violee, rend toute la session muette et fait accuser le detecteur.
    ("demarrage/origine-capture", "src/jev_tactics/bot/setup.py",
     "        if origine != (0, 0):", "        if False:", True),
    ("demarrage/espace-de-clic", "src/jev_tactics/bot/setup.py",
     ("        if (abs(clic[0] - vue[0]) > RESOLUTION_TOLERANCE\n"
      "                or abs(clic[1] - vue[1]) > RESOLUTION_TOLERANCE):"),
     "        if False:", True),
    ("demarrage/disposition", "src/jev_tactics/bot/setup.py",
     "    if not verdict.matches and options.execute:", "    if False:", True),
    # --- entrees : construire un diagnostic et ne pas l'imprimer revient au meme que ne
    # pas le construire. `scripts/farm.py` est l'entree principale de la chasse.
    # Le CORPS de la boucle et non son en-tete : remplacer `farm_summary(` laisse ses
    # arguments orphelins sur les lignes suivantes, donc un fichier qui ne compile plus --
    # et l'audit rendrait « gardee » pour la pire des raisons. `test_every_mutation_still_
    # compiles` l'a refuse avant que l'essai ne tourne.
    ("entree/bilan-de-farm", "scripts/farm.py",
     "            print(ligne)", "            pass  # neutralise", True),
    ("journal/ecriture-disque", "src/jev_tactics/farming/journal.py",
     '                stream.write(json.dumps(entry.as_dict()) + "\\n")',
     "                pass  # neutralise", True),
]


def mute(texte: str, motif: str, remplacement: str, partout: bool) -> str | None:
    """-> le texte mute, ou None si le motif est introuvable.

    None et non une exception : une cible perimee doit etre SIGNALEE et l'audit doit
    continuer. Un test verifie par ailleurs que chaque cible est trouvee -- sans quoi la
    table vieillit en silence et l'audit passe pour vert en n'auditant plus rien.
    """
    if motif not in texte:
        return None
    return texte.replace(motif, remplacement) if partout else texte.replace(
        motif, remplacement, 1)


def main(filtre: str | None = None, tests: list[str] | None = None) -> int:
    cibles = [c for c in CIBLES if not filtre or filtre in c[0]]
    if not cibles:
        print(f"aucune cible ne correspond a « {filtre} »")
        return 1
    cibles_tests = tests or ["tests/"]
    libres = []
    for nom, relatif, motif, remplacement, partout in cibles:
        source = RACINE / relatif
        origine = source.read_text(encoding="utf-8")
        modifie = mute(origine, motif, remplacement, partout)
        if modifie is None:
            print(f"introuvable  {nom}", flush=True)
            continue
        source.write_text(modifie, encoding="utf-8")
        try:
            issue = subprocess.run(
                [PYTHON, "-m", "pytest", "-x", "-q", "--no-header",
                 "-p", "no:cacheprovider", f"--ignore={PROPRE_TEST}", *cibles_tests],
                cwd=RACINE, capture_output=True, timeout=1800, check=False)
            verdict = "LIBRE" if issue.returncode == 0 else "gardee"
        except subprocess.TimeoutExpired:
            verdict = "timeout"
        finally:
            source.write_text(origine, encoding="utf-8")
        if verdict != "gardee":
            libres.append(nom)
        print(f"{verdict:7s} {nom}", flush=True)

    print(f"\n== {len(libres)} instruction(s) neutralisables sans qu'un test ne rougisse")
    for nom in libres:
        print(f"   {nom}")
    return 0


def _arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filtre", default=None,
                    help="n'auditer que les cibles dont le nom contient ce motif")
    ap.add_argument("--tests", nargs="*", default=None,
                    help="fichiers de test a rejouer (defaut : toute la suite). Restreindre "
                         "rend l'audit rapide mais son resultat devient une BORNE BASSE : "
                         "une garde ecrite ailleurs lui echappe")
    return ap.parse_args()


if __name__ == "__main__":
    args = _arguments()
    sys.exit(main(args.filtre, args.tests))
