"""A QUEL ECRAN a-t-on affaire ? La question que le bot ne se posait jamais.

Le controle actuel enchaine deux boucles -- recolte et combat -- et suppose que l'ecran est
forcement l'un des deux. Devant autre chose, rien de specifique ne se produit : les lectures
rendent None ou du bruit, les boucles tournent a vide, et le bilan devient une fiction.

C'est un mode de panne deja rencontre DEUX FOIS et corrige cas par cas -- `layout.py` pour
une resolution inattendue, `guard.py` pour la perte de focus. Meme classe de defaut, traitee
au coup par coup. Le jeu, lui, a bien d'autres ecrans : niveau de metier insuffisant
(data/pasniveau.png), ressource epuisee (data/epuise.png), mort, coffre plein, invitation,
deconnexion, chargement, modale d'evenement.

CE QUE CE MODULE APPORTE, ET QUI N'EXISTE PAS AILLEURS : `INCONNU` est un ETAT, pas une
absence de detection. Un bot qui s'arrete en disant « je ne reconnais pas cet ecran, voici
la capture » est plus utile qu'un bot qui tourne quarante minutes dans le vide -- et chaque
frame ainsi conservee est un etat de plus a coder. Le bot ecrit sa propre liste de taches.

Les deux discriminants sont ceux dont la fiabilite est deja mesuree, et non de nouveaux
seuils :

    timeline non vide           -> COMBAT     (30 captures sur 30, la timeline n'existe
                                               qu'en combat -- mesure de longue date)
    en-tete de l'HDV trouve     -> HDV        (9 captures sur 9 a 0,99-1,00 ; 0,13-0,27
                                               sur les 8 autres, cf. hdv.find_anchor)
    coordonnees lisibles        -> CARTE      (24/24 en session, 8/8 en reference)
    aucun des trois             -> INCONNU

Deliberement PAUVRE : trois etats, aucun seuil nouveau. Les etats fins -- chargement,
modale, mort -- se rajouteront un par un, chacun a partir d'une frame INCONNUE conservee.
Les inventer maintenant reviendrait a deviner a quoi ils ressemblent.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.coordinates import diagnose_position
from jev_tactics.perception.hdv import find_anchor
from jev_tactics.perception.timeline import read_timeline


class ScreenState(str, Enum):
    COMBAT = "combat"
    MAP = "carte"
    HDV = "hdv"
    UNKNOWN = "inconnu"


def classify_screen(frame: NDArray[np.uint8]) -> ScreenState:
    """Frame -> etat d'ecran. Jamais d'exception, jamais de supposition.

    L'ORDRE COMPTE, et pour deux raisons distinctes.

    La timeline d'abord : les coordonnees sont lisibles EN COMBAT AUSSI (mesure : cinq
    captures de combat sur six les donnent), donc les tester en premier classerait un
    combat comme une carte -- et le bot recolterait pendant qu'on le frappe.

    L'HDV ENSUITE, avant la carte : une fenetre d'Hotel de vente ouverte se trouve
    toujours SUR une carte. Classer cet ecran « carte » relancerait la boucle de recolte
    par-dessus une fenetre modale -- des clics de recolte envoyes dans une liste de
    prix. L'HDV est donc, comme le combat, un etat qui INTERDIT la recolte, et il doit
    etre reconnu avant elle.

    L'HDV est le premier etat ajoute par le flux que ce module prevoyait : neuf captures
    conservees parce qu'elles sortaient INCONNU, un discriminant mesure dessus, un etat
    de plus. Separation relevee sur les 17 captures du depot -- correlation de l'en-tete
    a 0,99-1,00 sur les neuf captures d'HDV, 0,13-0,27 sur les huit autres. Aucun seuil
    nouveau n'a ete invente : c'est celui de `hdv.find_anchor`, qui sert deja a ancrer
    les ROIs.
    """
    if read_timeline(frame):
        return ScreenState.COMBAT
    if find_anchor(frame).found:
        return ScreenState.HDV
    if diagnose_position(frame).position is not None:
        return ScreenState.MAP
    return ScreenState.UNKNOWN
