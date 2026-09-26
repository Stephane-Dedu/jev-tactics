"""Lecture de l'infobulle qui apparait au SURVOL d'une ressource.

Signalee par l'utilisateur, et elle repond a un compteur que le bot affichait sans pouvoir
l'expliquer : « N clics de recolte sans effet ». Deux captures montrent les deux raisons :

    data/epuise.png      Frene         +0% XP    « Epuise »
    data/pasniveau.png   Chataignier   +2% XP    « Bucheron Niv.20 »

La ressource est bien la, bien detectee, bien surlignee par la touche Y -- et cliquer
dessus ne rapporte rien. Le bot le decouvrait apres 3,5 secondes d'attente, six fois de
suite avant d'abandonner la carte. Le survol le dit en une fraction de seconde.

CE QU'ON LIT, ET CE QU'ON NE LIT PAS. Seul « Epuise » est reconnu. C'est un mot FIXE, donc
un gabarit suffit -- meme methode que le crane du chat, et pour la meme raison : la
troisieme ligne de l'infobulle change de libelle selon le cas, mais celui-la ne change pas.

La ligne « Bucheron Niv.20 » n'est PAS lue : le nom du metier varie (bucheron, mineur,
alchimiste, paysan...) et le niveau aussi, ce qui demanderait un lecteur de texte complet
pour une information que le bot ne saurait de toute facon pas exploiter -- il ne connait
pas ses propres niveaux de metier. Une ressource hors niveau reste donc detectee, cliquee,
et retiree par le garde-fou des recoltes sans effet, comme avant.

MESURE, correlation du gabarit sur les dix captures disponibles :

    data/epuise.png                              1,000
    data/pasniveau.png                           0,560
    les huit captures de jeu (combat et carte)   0,599 a 0,642

L'intervalle entre 0,642 et 1,000 est vide.

CE QU'ON NE SAIT PAS DIRE, et il vaut mieux l'ecrire que le decouvrir en session : quand
`is_depleted` rend False, on ignore si l'infobulle disait « c'est bon » ou si elle n'est
JAMAIS APPARUE. Un delai de survol trop court rendrait tout ce controle inerte, sans que
rien ne le signale.

Quatre detecteurs d'infobulle ont ete essayes pour combler ce trou, et mesures :

  - fenetre 90x90 CENTREE sur l'infobulle : 0,704 et 0,738 de pixels sombres, contre 0,176
    au pire sur 120 fenetres de carte. Excellent -- mais il faudrait deja savoir ou est
    l'infobulle, alors qu'on ne connait que le point survole ;
  - MAXIMUM GLISSANT autour du point survole (+-200 px) : la carte monte a 0,90 (panneaux,
    trous du plateau) contre 0,74 pour une vraie infobulle. Separation detruite ;
  - le meme, en ecartant d'abord les panneaux connus : la carte redescend, mais combat1.png
    tient encore 0,574 contre 0,651 -- 12 % de marge, la ou le crane du chat en a 36 % et
    le mot « Epuise » 56 %. Trop mince pour decider ;
  - PART DE ROUGE VIF, l'idee etant qu'un prerequis non rempli s'affiche en rouge dans le
    jeu. La mesure separe tres bien les DEUX infobulles entre elles -- 0,0010 pour
    « Epuise » contre 0,0314 pour « Bucheron Niv.20 », un rapport de trente. Mais elle ne
    separe pas de la CARTE : capture.png, hors_combat.png et combat1920.png donnent 0,0114
    a 0,0121, c'est-a-dire entre les deux. Sur une frame entiere le critere est donc
    inerte, et il faudrait a nouveau savoir ou est l'infobulle -- ce qui est precisement
    la question de depart.

La ligne « Niv. » a ete envisagee comme gabarit fixe, sur le modele d'« Epuise » : le nom
du metier varie mais « Niv. » non. Non tentee faute de pouvoir isoler ces quatre glyphes
sans les lire -- il faudrait un decoupage manuel dans la capture, donc un jugement a l'oeil
que ce projet s'interdit ailleurs. Si l'utilisateur fournit un jour un recadrage de cette
ligne seule, la methode d'« Epuise » s'applique telle quelle.

Le trou est donc comble autrement, par recoupement : le runner compte les ressources
ecartees au survol A COTE des recoltes sans effet. Beaucoup des secondes et zero des
premieres est la signature d'une infobulle qui n'apparait jamais.
"""

from __future__ import annotations

from importlib import resources

import cv2
import numpy as np
from numpy.typing import NDArray

# Mesure : 1,000 sur l'infobulle « Epuise », 0,642 au plus partout ailleurs.
MIN_SCORE = 0.85
_GLYPH_FILE = "depleted_glyph.npz"
_GLYPH: NDArray[np.uint8] | None = None


def load_glyph() -> NDArray[np.uint8]:
    global _GLYPH
    if _GLYPH is None:
        ref = resources.files("jev_tactics.perception") / "data" / _GLYPH_FILE
        with resources.as_file(ref) as path:
            _GLYPH = np.load(path)["glyph"]
    return _GLYPH


def depleted_score(frame: NDArray[np.uint8]) -> float:
    """Meilleure correlation du mot « Epuise » sur la frame. 0 si elle est trop petite."""
    glyph = load_glyph()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if grey.shape[0] < glyph.shape[0] or grey.shape[1] < glyph.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(grey, glyph, cv2.TM_CCOEFF_NORMED).max())


def is_depleted(frame: NDArray[np.uint8]) -> bool:
    """La ressource SURVOLEE est-elle epuisee ?

    Cherche sur toute la frame : l'infobulle suit le curseur, donc sa position n'a rien de
    fixe. Le mot est assez distinctif pour s'en passer -- 0,642 au plus sur huit captures
    de jeu qui n'en contiennent aucune.
    """
    return depleted_score(frame) >= MIN_SCORE
