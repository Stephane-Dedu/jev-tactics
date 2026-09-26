"""Caracterise ce qui BOUGE sur la carte, pour caler la detection de groupes.

    python scripts/study_monsters.py --live                 # deux captures espacees
    python scripts/study_monsters.py --live --out zones.png # + calque de controle
    python scripts/study_monsters.py avant.png apres.png    # sur deux captures
    python scripts/study_monsters.py --runs                 # sur data/runs/, le fond du depot

Les seuils de `perception/monsters.py` (aire, compacite) sont des ESTIMATIONS : aucune
capture de carte avec des monstres n'etait disponible a l'ecriture. Ce script produit la
mesure qui permet de les remplacer.

Ce qu'il faut regarder, dans l'ordre :
  1. **combien** de candidats sortent. Beaucoup = le seuil d'aire est trop bas, ou le
     decor de cette carte s'anime (eau, feuillages) ;
  2. **ou** ils sont. Un calque le montre ; un groupe de monstres est un amas compact,
     une riviere une longue trainee ;
  3. **quelles aires** ils font. C'est le chiffre qui separe un groupe d'une touffe
     d'herbe, et il ne peut venir que d'ici.

Le meme travail que `study_markers.py` a fait pour les marqueurs de cellule -- et qui
avait alors montre que la piste suivie etait mauvaise. Mieux vaut l'apprendre d'un
histogramme que d'un bot qui attaque des buissons.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.perception import mobinfo, monsters
from jev_tactics.perception.coordinates import diagnose_position
from jev_tactics.pipeline import UI_ZONES


def capture_pair(delay: float) -> tuple[np.ndarray, np.ndarray]:
    from jev_tactics.capture import ScreenCapture, wait_for_game

    wait_for_game(on_wait=lambda: print(
        "bascule sur le jeu (Alt+Tab) — deux captures vont etre prises"))
    with ScreenCapture() as capture:
        before = capture.grab().image
        print(f"# premiere capture, seconde dans {delay:.1f} s")
        time.sleep(delay)
        return before, capture.grab().image


def describe(before: np.ndarray, after: np.ndarray, exclude) -> list:
    mask = monsters.motion_mask(before, after)
    moved = float((mask > 0).mean())
    print(f"\n{moved:.2%} de l'ecran a bouge")
    if moved == 0.0:
        print("  -> rien n'a bouge : les deux captures sont identiques (jeu en pause ?)")
    elif moved > 0.5:
        print("  -> plus de la moitie de l'ecran : changement de carte, ou camera qui "
              "se deplace")

    groups = monsters.detect_groups(before, after, exclude=exclude)
    print(f"\n{len(groups)} candidat(s) retenu(s) "
          f"[aire {monsters.MIN_AREA}-{monsters.MAX_AREA}, "
          f"compacite >= {monsters.MIN_EXTENT}]")
    for group in groups:
        extent = group.area / max(group.width * group.height, 1)
        print(f"   ({group.x:>4},{group.y:>4})  aire {group.area:>6}  "
              f"{group.width}x{group.height}  compacite {extent:.2f}")

    # Les composantes ECARTEES comptent autant : elles disent si le seuil est bien place.
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)),
                   reverse=True)
    if areas:
        print(f"\n{len(areas)} composantes au total ; les 12 plus grosses :")
        print("   " + ", ".join(str(a) for a in areas[:12]))
        rejected = [a for a in areas if a < monsters.MIN_AREA]
        print(f"   {len(rejected)} sous le seuil d'aire, "
              f"{sum(1 for a in areas if a > monsters.MAX_AREA)} au-dessus")
    return groups


def overlay(after: np.ndarray, groups, path: Path) -> None:
    canvas = after.copy()
    for x0, y0, x1, y1 in UI_ZONES:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (120, 120, 120), 1)
    for group in groups:
        half_w, half_h = group.width // 2, group.height // 2
        cv2.rectangle(canvas, (group.x - half_w, group.y - half_h),
                      (group.x + half_w, group.y + half_h), (0, 0, 255), 2)
        cv2.putText(canvas, str(group.area), (group.x - half_w, group.y - half_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(canvas, str(group.area), (group.x - half_w, group.y - half_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    print(f"\ncalque : {path}")


def find_tooltip(frame, near, span=(300, 160), minimum: int = 2) -> list:
    """Ou l'infobulle s'affiche-t-elle, AUTOUR DU CURSEUR ? -> bandes trouvees.

    Le chiffre qui manque, et le seul. Survoler un candidat avant de le cliquer coute
    0,35 s la ou un clic rate en coute 8, et la mesure dit que l'infobulle discrimine
    proprement : sur les 24 captures de `data/runs/`, 22 sans survol rendent ZERO nombre
    dans la ROI de combat et 2 avec survol en rendent deux. Mais cette ROI est celle du
    combat ; hors combat, l'infobulle d'un groupe s'affiche ailleurs.

    PREMIERE VERSION REFUTEE, et c'est la lecon. Balayer la frame ENTIERE par bandes
    rendait 71 a 78 bandes sur CHAQUE capture, survol ou pas : le chat, la barre de sorts,
    le panneau de quetes et les coordonnees sont tous du texte clair sur fond sombre. Un
    instrument qui repond la meme chose sur les positifs et les negatifs ne mesure rien --
    exactement ce que la sonde de mouvement fait deja, et qu'on cherche a corriger.

    Resserrer autour du curseur ameliore beaucoup sans SUFFIRE : verifie sur huit
    captures avec un curseur simule pres de l'infobulle connue, les deux positives rendent
    neuf bandes contigues a partir de y789, mais les six negatives en rendent encore 2 a 4
    (barre de sorts, chat). Ce chercheur NE TRANCHE DONC PAS : il RESSERRE.

    C'est pourquoi il ecrit aussi un DECOUPAGE annote. Une image ne peut pas se tromper de
    seuil : on y lit la ROI a l'oeil, une fois, et elle devient une constante mesuree.
    Deux passages -- un en survolant un groupe, un en survolant du vide -- suffisent, et
    la difference entre les deux images est plus sure que n'importe quel comptage.
    """
    hauteur, largeur = frame.shape[:2]
    cx, cy = near
    dx, dy = span
    x0, x1 = max(cx - dx, 0), min(cx + dx, largeur)
    y0, y1 = max(cy - 60, 0), min(cy + dy, hauteur)
    trouvees = []
    BANDE, PAS = 26, 13
    for haut in range(y0, max(y1 - BANDE, y0 + 1), PAS):
        groupes = mobinfo.glyph_groups(frame, (x0, haut, x1, haut + BANDE))
        if len(groupes) < minimum:
            continue
        gauches = [g[0][0] + x0 for g in groupes]
        trouvees.append((haut, min(gauches), max(gauches), len(groupes)))
    return trouvees


TRANCHES = ((0, 30), (30, 60), (60, 120), (120, 300))


def _seconds(stem: str) -> int | None:
    """« 20260807-215714 » -> un instant en secondes. None si le nom ne dit pas l'heure.

    Arithmetique plutot que `datetime` : seuls les ECARTS servent ici, et une date naive
    suffit pour les calculer. Passer par un instant date obligerait a lui inventer un
    fuseau, c'est-a-dire a repondre a une question que la mesure ne pose pas.
    """
    jour, _, heure = stem.partition("-")
    if len(jour) != 8 or len(heure) != 6 or not (jour + heure).isdigit():
        return None
    return (int(jour[6:]) * 86_400 + int(heure[:2]) * 3600
            + int(heure[2:4]) * 60 + int(heure[4:]))


def _mesures_espacement(dossier: Path) -> list[tuple[int, int]]:
    """(ecart en secondes, nombre de candidats) pour toutes les paires exploitables.

    Separe du VERDICT juste en dessous, et pas par gout du decoupage : la regle de lecture
    doit pouvoir etre verifiee sans dependre de ce que contient `data/runs/`, sinon un
    changement de fond ferait passer ou echouer un test qui parle d'autre chose.

    Toutes les paires de moins de cinq minutes sont formees, pas seulement les
    consecutives : c'est ce qui donne des ecarts varies a partir de 24 captures.
    """
    fichiers = sorted(dossier.glob("*.png"))
    images = {f: cv2.imread(str(f)) for f in fichiers}
    horo = {f: s for f in fichiers if (s := _seconds(f.stem)) is not None}

    mesures: list[tuple[int, int]] = []
    for avant, apres in itertools.combinations([f for f in fichiers if f in horo], 2):
        a, b = images[avant], images[apres]
        if a is None or b is None or a.shape != b.shape:
            continue
        ecart = horo[apres] - horo[avant]
        if not 0 < ecart <= TRANCHES[-1][1]:
            continue
        # Meme critere que `survey_runs` : au-dela, c'est un changement de carte et la
        # difference ne veut plus rien dire.
        if float((monsters.motion_mask(a, b) > 0).mean()) > 0.30:
            continue
        mesures.append((ecart, len(monsters.detect_groups(a, b, exclude=UI_ZONES))))
    return mesures


def survey_spacing(dossier: Path) -> list[tuple[int, int, int, int]]:
    """Le nombre de candidats depend-il de l'ECART entre les deux captures ?

    La question decide de ce que vaut tout le reste de ce script. Si le compte grandit
    avec l'ecart, alors les paires de `data/runs/` -- espacees de dizaines de secondes --
    majorent le vrai scan, qui prend les siennes a 0,35 s, et leur 20 a 49 candidats est
    un plafond rassurant. Si le compte est PLAT, cette consolation n'existe pas.

    Le facteur 1,5 qui separe les deux verdicts est un ARBITRAGE, pas une mesure. Il est
    place la ou une difference cesse d'etre du bruit d'echantillonnage sur une dizaine de
    paires, et il ne decide de rien dans le bot -- seulement du mot imprime.

    -> une ligne par tranche : (borne basse, borne haute, nombre de paires, mediane).
    """
    mesures = _mesures_espacement(dossier)

    lignes = []
    for bas, haut in TRANCHES:
        comptes = sorted(n for ecart, n in mesures if bas <= ecart < haut)
        if comptes:
            lignes.append((bas, haut, len(comptes), comptes[len(comptes) // 2]))

    print(f"\n== candidats selon l'ECART entre les deux captures "
          f"({len(mesures)} paires de meme carte)")
    for bas, haut, combien, median in lignes:
        print(f"   {bas:3}-{haut:<3} s : {combien:3} paires, mediane {median:3} candidats")
    if len(lignes) < 2:
        print("   pas assez d'ecarts differents pour conclure")
        return lignes
    medianes = [ligne[3] for ligne in lignes]
    if max(medianes) <= 1.5 * min(medianes):
        print("\n   PLAT. Le compte ne suit pas l'ecart, donc ce n'est pas la distance")
        print("   parcourue qui le produit -- c'est de l'animation permanente. Rien ne")
        print("   permet d'affirmer que le scan reel, a 0,35 s, en verrait moins.")
    else:
        print("\n   CROISSANT. Les paires longues majorent bien le vrai scan.")
    return lignes


def survey_runs(dossier: Path) -> int:
    """Le detecteur passe sur TOUTES les paires consecutives de `data/runs/`.

    Ces captures existaient depuis le premier jour et n'avaient jamais ete passees au
    detecteur de groupes -- le seul retour terrain etait un journal de quatre lignes,
    toutes fausses. Elles sont espacees de 12 a 40 s, soit cinquante a cent fois
    `MOTION_DELAY`.

    ON A LONGTEMPS ECRIT ICI QU'ELLES « MAJORENT LARGEMENT LE MOUVEMENT », et donc que le
    compte obtenu etait un plafond. C'ETAIT UNE DEDUCTION, JAMAIS UNE MESURE, et
    `--espacement` la refute : le nombre de candidats ne bouge pas avec l'ecart entre les
    deux captures. Ce n'est donc pas la distance parcourue qui le produit, et rien ne
    permet d'affirmer que le vrai scan, a 0,35 s, en verrait moins.

    Le plancher ci-dessous reste utile -- il dit ce que le detecteur rend sur du reel --
    mais il faut le lire pour ce qu'il est : une mesure DANS UN AUTRE REGIME que celui
    d'exploitation, dont le lien avec le regime reel n'est pas etabli.

    Les paires dont plus de 30 % de la surface a bouge sont ecartees : c'est un changement
    de carte, et la difference n'y veut plus rien dire.

    -> le plancher observe, c'est-a-dire le plus PETIT nombre de candidats par paire.

    CE QU'IL FAUT EN FAIRE A CHANGE. On lisait ici que `SUSPICIOUS_GROUP_COUNT` doit rester
    EN DESSOUS de ce plancher, « pour pouvoir signaler ». C'est vrai et insuffisant : sous le
    plancher, il signale TOUJOURS -- mesure a 12, il declarait 87 % des scans reels
    invraisemblables, et un avertissement permanent n'est plus lu. Le seuil doit tomber DANS
    la distribution, et les deux bornes se gardent
    (`tests/test_farming.py::TestTheHuntScanIsWatchedToo`).
    """
    fichiers = sorted(dossier.glob("*.png"))
    print(f"{len(fichiers)} captures dans {dossier}")
    comptes, aires = [], []
    for avant, apres in itertools.pairwise(fichiers):
        a, b = cv2.imread(str(avant)), cv2.imread(str(apres))
        if a is None or b is None or a.shape != b.shape:
            continue
        bouge = float((monsters.motion_mask(a, b) > 0).mean())
        if bouge > 0.30:
            print(f"  {avant.stem[-6:]} -> {apres.stem[-6:]}  "
                  f"{bouge:6.1%} ECARTEE (changement de carte)")
            continue
        groupes = monsters.detect_groups(a, b, exclude=UI_ZONES)
        comptes.append(len(groupes))
        aires.extend(g.area for g in groupes)
        print(f"  {avant.stem[-6:]} -> {apres.stem[-6:]}  {bouge:6.1%}  "
              f"{len(groupes):3d} candidats")

    if not comptes:
        print("\n" + "aucune paire exploitable")
        return 0
    aires.sort()
    print(f"\n== {len(comptes)} paires exploitables, {len(aires)} candidats")
    print(f"   candidats par paire : min {min(comptes)}  max {max(comptes)}")
    print(f"   aires : min {aires[0]}  median {aires[len(aires) // 2]}  max {aires[-1]}")
    seuil = monsters.SUSPICIOUS_GROUP_COUNT
    bavards = sum(1 for c in comptes if c > seuil)
    print(f"\n   candidats par paire : plancher {min(comptes)}, plafond {max(comptes)}.")
    print(f"   SUSPICIOUS_GROUP_COUNT = {seuil} — se declencherait sur "
          f"{bavards}/{len(comptes)} de ces paires ({bavards / len(comptes):.0%}).")
    # DEUX modes de panne, et ce script n'en signalait qu'un. Il annoncait « sous le
    # plancher, il peut signaler » comme si c'etait le bon cote -- alors que sous le
    # plancher il signale TOUJOURS, et un avertissement permanent n'est plus lu. Le seuil
    # doit tomber DANS la distribution.
    if seuil <= min(comptes):
        print("   SOUS LE PLANCHER : il se declenchera a chaque scan, donc plus personne "
              "ne le lira.")
    elif seuil >= max(comptes):
        print("   AU-DESSUS DU PLAFOND : il ne signalera jamais rien.")
    else:
        print("   Dans la distribution : muet sur les scans calmes, parlant sur les "
              "encombres.")
    return min(comptes)



def survey_ui_leak(dossier: Path, pas: int = 60) -> int:
    """Reste-t-il de l'INTERFACE hors de `UI_ZONES` ? -> nombre de cases suspectes.

    LA QUESTION. `UI_ZONES` refuse des candidats, et un element d'interface oublie produit
    des candidats a chaque frame -- toujours au meme endroit de l'ECRAN. Chaque clic dessus
    coute `ENGAGE_TIMEOUT` sans jamais engager, et la liste noire ne le retient pas
    puisqu'elle est rangee par carte.

    LA SIGNATURE CHERCHEE, et pourquoi elle demande DEUX mesures. Un element d'interface
    est (a) present sur toutes les cartes, et (b) PIXEL POUR PIXEL le meme d'une carte a
    l'autre. Le terrain peut satisfaire (a) -- un circuit repasse, les cartes d'une meme
    zone se ressemblent -- mais pas (b).

    UN TEMOIN UNIQUE NE SUFFIT PAS, et ma premiere version s'y est trompee. Elle comparait
    le coin suspect (0,92) a UNE case de terrain choisie a la main (0,75) et concluait que
    la mesure ne tranchait pas. C'etait la case qui etait atypique : sur la grille entiere,

        cases DEJA exclues par UI_ZONES     stabilite  min -0,16  med +0,76  max +1,00
        cases NON exclues                   stabilite  min -0,60  med +0,11  max +1,00

    Les deux populations se separent tres bien. La lecon est generale et vaut au-dela
    d'ici : un temoin unique n'est pas une distribution, et l'ecart entre deux points
    choisis ne dit rien de l'ecart entre deux populations.

    UNE STABILITE NE SUFFIT PAS NON PLUS, et c'est la seconde erreur de cette mesure. Sa
    premiere version demandait « cette case est-elle hors de `UI_ZONES` et stable ? » et
    designait 75 cases, dont un bloc entier a droite (x 1530-1830, y 390-570). Deux defauts
    dans cette question :

      - `detect_groups` exclut par DEUX mecanismes, le masque de teinte des panneaux ouverts
        ET les rectangles fixes. 33 des 75 etaient deja retirees par `ui_mask` ;
      - un panneau qui NE BOUGE PAS ne produit aucun candidat, donc ne coute rien. Le bloc
        de droite est de ceux-la : stable, et parfaitement silencieux.

    En demandant ce que `detect_groups` rend REELLEMENT, il reste QUATRE cases, dispersees
    a gauche, a 0,90-0,92 de stabilite sur 2 ou 3 cartes. Aucun bloc, rien qui ressemble a
    un panneau. Sur ces captures, il n'y a pas de fuite d'interface detectable.

    CE QUI TRANCHERAIT VRAIMENT : des captures d'une zone VISIBLEMENT differente. Les
    quatre cartes lues -- (0,8), (1,8), (1,9), (2,9) -- sont adjacentes et partagent leur
    tileset, si bien qu'un decor commun peut passer pour de l'interface. Cette fonction est
    ecrite pour ce jour-la ; en attendant, elle dit honnetement qu'elle ne trouve rien.
    """
    fichiers = sorted(dossier.glob("*.png"))
    par_carte: dict[tuple[int, int], list] = {}
    for chemin in fichiers:
        frame = cv2.imread(str(chemin))
        if frame is None:
            continue
        lue = diagnose_position(frame).position
        if lue is not None:
            par_carte.setdefault(lue.as_tuple(), []).append(frame)
    cartes = sorted(par_carte)
    print(f"{len(fichiers)} captures, {len(cartes)} cartes distinctes : {cartes}")
    if len(cartes) < 2:
        print("  moins de deux cartes : la comparaison n'a aucun sens")
        return 0

    # LES CASES QUE LE DETECTEUR RETIENT VRAIMENT, et non celles hors de `UI_ZONES`.
    # `detect_groups` exclut par DEUX mecanismes -- le masque de teinte des panneaux
    # ouverts ET les rectangles fixes -- et la premiere version de cette mesure ignorait le
    # premier : 33 de ses 75 « suspectes » etaient deja retirees par `ui_mask`, soit 44 %
    # de surestimation. Demander directement ce que `detect_groups` rend supprime la
    # question au lieu d'y repondre a moitie.
    vivantes: dict[tuple[int, int], set] = {}
    for carte, frames in par_carte.items():
        for avant, apres in itertools.pairwise(frames):
            if avant.shape != apres.shape:
                continue
            if float((monsters.motion_mask(avant, apres) > 0).mean()) > 0.30:
                continue
            for groupe in monsters.detect_groups(avant, apres, exclude=UI_ZONES):
                vivantes.setdefault((groupe.x // pas, groupe.y // pas), set()).add(carte)

    hauteur, largeur = par_carte[cartes[0]][0].shape[:2]
    dedans, dehors, suspectes = [], [], []
    for y in range(0, hauteur - pas, pas):
        for x in range(0, largeur - pas, pas):
            scores = []
            for i in range(len(cartes)):
                for j in range(i + 1, len(cartes)):
                    a = par_carte[cartes[i]][0][y:y + pas, x:x + pas].astype(np.float32)
                    b = par_carte[cartes[j]][0][y:y + pas, x:x + pas].astype(np.float32)
                    scores.append(float(cv2.matchTemplate(
                        a, b, cv2.TM_CCOEFF_NORMED)[0, 0]))
            stable = min(scores)
            centre = (x + pas // 2, y + pas // 2)
            exclue = any(x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1
                         for x0, y0, x1, y1 in UI_ZONES)
            (dedans if exclue else dehors).append(stable)
            # Suspecte = le detecteur y produit des candidats SUR PLUSIEURS CARTES, et les
            # pixels y sont aussi stables que de l'interface. Les deux conditions comptent :
            # la stabilite seule designe du decor partage, les candidats seuls du decor
            # anime.
            sur = vivantes.get((x // pas, y // pas), set())
            if len(sur) >= 2 and stable >= 0.90:
                suspectes.append((centre, stable, len(sur)))

    def bornes(valeurs):
        valeurs = sorted(valeurs)
        return (f"min {valeurs[0]:+.2f} med {valeurs[len(valeurs) // 2]:+.2f} "
                f"max {valeurs[-1]:+.2f}") if valeurs else "aucune case"

    print(f"\n  stabilite entre cartes, cases DEJA exclues : {bornes(dedans)}")
    print(f"  stabilite entre cartes, cases NON exclues   : {bornes(dehors)}")
    if dehors and sorted(dehors)[len(dehors) // 2] >= 0.80:
        print("\n  LES DEUX POPULATIONS NE SE SEPARENT PAS : le terrain de ces captures est")
        print("  aussi stable que de l'interface. Elles viennent d'une seule zone, et cette")
        print("  mesure ne tranchera rien tant qu'on n'aura pas des zones differentes.")
        return 0
    print(f"\n  {len(suspectes)} case(s) ou le detecteur produit des candidats sur "
          f"plusieurs cartes,")
    print("  et dont les pixels sont aussi stables que de l'interface :")
    for centre, stable, sur in sorted(suspectes, key=lambda c: -c[1])[:12]:
        print(f"     ({centre[0]:>4},{centre[1]:>4})  stabilite {stable:+.2f}  "
              f"{sur} cartes")
    return len(suspectes)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", metavar="AVANT APRES")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="ecart entre les deux captures (defaut 1 s)")
    ap.add_argument("--out", help="calque de controle")
    ap.add_argument("--all", action="store_true",
                    help="ne pas exclure les zones d'interface (pour les voir)")
    ap.add_argument("--tooltip-roi", action="store_true",
                    help="chercher OU s'affiche l'infobulle, en survolant un groupe hors "
                         "combat. C'est la seule donnee qui manque pour verifier un "
                         "candidat par survol (0,35 s) au lieu d'un clic rate (8 s)")
    ap.add_argument("--runs", action="store_true",
                    help="passer le detecteur sur toutes les paires de data/runs/ — "
                         "les seules captures de carte reelles du depot. Produit le "
                         "plancher qui retient SUSPICIOUS_GROUP_COUNT")
    ap.add_argument("--fuite-ui", action="store_true",
                    help="reste-t-il de l'INTERFACE hors de UI_ZONES ? Cherche les cases "
                         "d'ecran identiques d'une carte a l'autre. Resultat NEGATIF sur "
                         "les captures actuelles : quatre cartes d'une meme zone, le "
                         "terrain y est aussi stable que l'interface")
    ap.add_argument("--espacement", action="store_true",
                    help="le nombre de candidats depend-il de l'ecart entre les deux "
                         "captures ? Repond a « ces paires majorent-elles le vrai scan », "
                         "qui etait affirme sans mesure")
    args = ap.parse_args()

    if getattr(args, "tooltip_roi", False):
        from jev_tactics.capture import ScreenCapture, wait_for_game
        wait_for_game(on_wait=lambda: print(
            "bascule sur le jeu, SURVOLE un groupe de monstres et ne bouge plus"))
        time.sleep(args.delay)
        with ScreenCapture() as capture:
            frame = capture.grab().image
        from jev_tactics.perception.entities import cursor_position
        curseur = cursor_position()
        if curseur is None:
            print("position du curseur illisible — mesure impossible")
            return
        print(f"curseur en {curseur}")
        bandes = find_tooltip(frame, near=curseur)
        sortie = Path(args.out or "data/tooltip_autour_curseur.png")
        cx, cy = curseur
        x0, y0 = max(cx - 300, 0), max(cy - 60, 0)
        crop = frame[y0:min(cy + 160, frame.shape[0]), x0:min(cx + 300, frame.shape[1])]
        vue = crop.copy()
        for haut, bx0, bx1, _ in bandes:
            cv2.rectangle(vue, (bx0 - x0 - 6, haut - y0), (bx1 - x0 + 60, haut - y0 + 26),
                          (0, 0, 255), 1)
        cv2.circle(vue, (cx - x0, cy - y0), 4, (0, 255, 255), -1)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sortie), vue)
        print(f"{len(bandes)} bande(s) portant au moins 2 nombres pres du curseur :")
        for haut, bx0, bx1, combien in bandes:
            print(f"   y {haut:>4}-{haut + 26:<4}  x {bx0:>4}-{bx1:<4}  "
                  f"{combien} nombres")
        print(f"   decoupage annote : {sortie}")
        print("   ATTENTION : ce comptage NE TRANCHE PAS. Verifie sur huit captures, "
              "l'ATH")
        print("   (barre de sorts, chat) rend encore 2 a 4 bandes SANS aucun survol.")
        print("   Lire la ROI SUR L'IMAGE, et refaire un passage en survolant du VIDE :")
        print("   c'est la difference entre les deux images qui designe l'infobulle.")
        return

    if args.runs or args.espacement or args.fuite_ui:
        dossier = Path(__file__).resolve().parents[1] / "data" / "runs"
        if args.runs:
            survey_runs(dossier)
        if args.espacement:
            survey_spacing(dossier)
        if args.fuite_ui:
            survey_ui_leak(dossier)
        return

    if args.live:
        before, after = capture_pair(args.delay)
    elif len(args.images) == 2:
        before, after = (cv2.imread(p) for p in args.images)
        if before is None or after is None:
            sys.exit("image illisible")
    else:
        ap.error("preciser --live, ou deux images")

    groups = describe(before, after, () if args.all else UI_ZONES)
    if args.out:
        overlay(after, groups, Path(args.out))
    print("\nSeuils a ajuster dans src/jev_tactics/perception/monsters.py "
          "(MIN_AREA, MIN_EXTENT).")


if __name__ == "__main__":
    main()
