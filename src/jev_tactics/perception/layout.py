"""La disposition observee correspond-elle a celle sur laquelle tout a ete releve ?

Toutes les zones du projet -- ROIs d'UI, coordonnees de carte, timeline, grille de la
barre de sorts, zones d'exclusion de l'ATH -- sont des PIXELS ABSOLUS, releves sur une
capture 1919x1079. Sur une autre disposition elles pointent a cote, et rien ne leve
d'erreur : la lecture rend simplement None ou du bruit, partout, en meme temps.

`diagnose.py` le disait deja, et le disait bien. Le probleme etait qu'il fallait penser a
lancer le doctor : NI `farm.py` NI `play_turn.py` ne verifiaient quoi que ce soit avant
de jouer. On pouvait donc lancer une session complete en fenetre redimensionnee et
n'obtenir que des lectures vides, sans le moindre indice pointant vers la cause.

Ce module extrait le verdict pour qu'il serve aux deux endroits. La regle de tolerance
n'est pas negociee ici : un ecart de quelques pixels vient des outils de capture qui
perdent une bordure, et les ROIs ont plus de marge que cela. Confondre ce cas avec un
vrai changement de resolution ferait chercher la panne au mauvais endroit -- exactement
ce que ce controle doit eviter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.spellbar import COLUMNS, bar_layout

# Disposition de reference : celle sur laquelle toutes les constantes ont ete relevees.
REFERENCE_SIZE = (1919, 1079)
# Ecart tolere avec la reference, en pixels, avant de soupconner la disposition. La
# reference vient d'une capture qui a perdu un pixel sur chaque axe, artefact courant.
#
# 2 et non 4, et ces deux pixels ne sont pas cosmetiques. Mesure par deplacement
# synthetique, sur capture.png et sur une frame de combat, des QUATRE lecteurs a ROI
# absolue -- exprimee ici en ECART MESURE, c'est-a-dire dans l'unite de cette constante :
#
#   decalage du contenu (bord haut-gauche perdu)   retrait seul (bord bas-droite perdu)
#     coordonnees  perte a un ecart de 8            aucune perte
#     pods         aucune perte                     aucune perte
#     timeline     perte a un ecart de 3   <--      aucune perte
#     PV du mob    perte a un ecart de 3   <--      aucune perte
#
# A 4, `check_layout` BENISSAIT donc une frame dont la timeline ne comptait plus que 1
# combattant sur 2 -- et la timeline est le discriminant « suis-je en combat », lu par les
# deux boucles et par la classification d'ecran. Le decompte devenait faux sans que rien ne
# le signale : exactement le mode de panne que ce module existe pour rendre visible.
#
# Piege a ne pas repeter : l'ecart n'est PAS le decalage. La reference (1919x1079) est elle
# meme un pixel sous la taille reelle des captures (1920x1080), donc un contenu decale de d
# pixels rend un ecart de |d - 1|. La timeline lache a d = 4, ce qui fait un ecart de 3, et
# la tolerance doit donc etre 2. Avoir lu ces deux nombres comme un seul m'a fait poser 3
# une premiere fois ; c'est le test etendu aux quatre lecteurs qui l'a rattrape.
#
# Les deux sens ne sont pas symetriques, et c'est le retrait -- inoffensif d'un bout a
# l'autre de la mesure -- qui correspond a l'artefact de capture invoque ci-dessus. La
# tolerance est bornee par l'autre sens, qu'elle ne sait pas distinguer.
RESOLUTION_TOLERANCE = 2


@dataclass(frozen=True)
class LayoutVerdict:
    """Ce que la taille de la capture dit de la fiabilite des lectures."""

    width: int
    height: int
    matches: bool
    message: str

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


def describe_hud(frame: NDArray[np.uint8], declared_slots: Iterable[int] = ()) -> list[str]:
    """Ce que la DISPOSITION de l'ATH implique, au-dela de la seule resolution.

    POURQUOI CE N'EST PAS `check_layout`. Celui-ci ne compare que la TAILLE de la capture,
    alors que son message promet de couvrir « les ROIs d'UI, les coordonnees de carte, la
    timeline et la barre de sorts ». Le 19/08 la resolution etait bonne -- 1920x1080 -- et
    la session a tourne des heures sur un ATH que rien ne savait lire : orbe de PV a un
    seul nombre, barre de sorts a une rangee. Le garde-fou benissait la frame parce qu'il
    ne regardait que ses dimensions.

    Les lectures s'adaptent desormais aux deux ATH. Ce qui ne peut PAS s'adapter, c'est
    une config de sorts qui designe des emplacements inexistants : sur une barre a une
    rangee, un sort declare en case 24 ne sera jamais lance, et le bot ressemblera a un
    bot prudent. C'est le defaut le plus difficile a diagnostiquer du projet, et il se
    constate ici en une ligne.
    """
    notes: list[str] = []
    geometrie = bar_layout(frame)
    if geometrie is None:
        return notes
    visibles = COLUMNS * geometrie.rows
    notes.append(f"barre de sorts : {geometrie.rows} rangee(s) affichee(s), "
                 f"emplacements 0 a {visibles - 1}")
    hors = sorted(s for s in declared_slots if s >= visibles)
    if hors:
        notes.append(
            r"/!\ " + f"{len(hors)} sort(s) declares hors de la barre affichee "
            f"(emplacements {hors}) : ils ne seront JAMAIS lances, et un bot qui "
            f"n'attaque pas ressemble a un bot prudent. Afficher plus de rangees en jeu, "
            f"ou reprendre la config avec `python scripts/spells.py --live --spells ...`")
    return notes


def check_layout(frame: NDArray[np.uint8]) -> LayoutVerdict:
    """Verdict sur la taille d'une capture, sans rien afficher ni lever."""
    height, width = frame.shape[:2]
    reference = f"{REFERENCE_SIZE[0]}x{REFERENCE_SIZE[1]}"
    size = f"{width}x{height}"

    if (width, height) == REFERENCE_SIZE:
        return LayoutVerdict(width, height, True,
                             f"{size} (disposition de reference)")

    drift = max(abs(width - REFERENCE_SIZE[0]), abs(height - REFERENCE_SIZE[1]))
    if drift <= RESOLUTION_TOLERANCE:
        return LayoutVerdict(
            width, height, True,
            f"{size} (reference {reference}, ecart de {drift} px — sans consequence "
            f"sur les ROIs)")

    return LayoutVerdict(
        width, height, False,
        f"{size} au lieu de {reference} — les ROIs d'UI, les coordonnees de carte, la "
        f"timeline et la barre de sorts sont calees sur la reference et pointent a "
        f"cote. Toutes les lectures echoueront, EN SILENCE. Passer le jeu en "
        f"{reference} (ou plein ecran), puis relancer `python scripts/doctor.py --live`")
