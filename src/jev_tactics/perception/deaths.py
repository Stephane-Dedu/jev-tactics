"""Lecture des MORTS annoncees dans le chat.

Le chat est la derniere grande source d'information que le bot n'exploitait pas, et c'est
celle que l'utilisateur avait designee en premier : « on voit les degats et les ennemis
morts ». Il est parfaitement lisible -- texte net, fort contraste sur fond sombre -- et
porte ce qu'aucun autre etage ne donne : ce qui vient de SE PASSER, par opposition a ce qui
est actuellement affiche.

CE QU'ON LIT, ET POURQUOI CE N'EST PAS DU TEXTE. Deux captures reelles portent une mort, et
elles ne la formulent PAS pareil :

    capture.png    « Timongouste : -212 PV  (☠ mort). »
    bug.png        « ☠ Flammeche Eau est mort ! »

Le libelle change, le GLYPHE non. Viser le crane plutot que le mot evite d'avoir a couvrir
toutes les tournures du jeu, dans toutes les langues, et resiste au fait qu'un nom de
monstre peut contenir n'importe quoi. C'est la meme raison qui avait fait preferer la
geometrie du reseau a la lecture des bordures de cases.

MESURE, sur les huit captures, score de correlation maximal :

    capture.png     1,000 et 0,993     les deux morts relevees a l'oeil
    bug.png         0,965              « Flammeche Eau est mort ! »
    six autres      0,663 a 0,710      aucune mort dans leur chat

L'intervalle entre 0,710 et 0,965 est vide : le seuil s'y pose sans arbitrage.

CE QUE CE MODULE NE FAIT PAS. Il rend les marques VISIBLES, pas les morts RECENTES. Le chat
defile : une mort d'il y a trois tours reste affichee, et une mort ancienne peut sortir par
le haut. Distinguer « quelqu'un vient de mourir » demanderait de suivre le defilement d'une
frame a l'autre -- ce qu'une capture par situation ne permet pas de valider. Rendre un
compte serait donc une lecture fausse deguisee en mesure.
"""

from __future__ import annotations

from importlib import resources

import cv2
import numpy as np
from numpy.typing import NDArray

# Score au-dela duquel une correlation designe le glyphe. Mesure : 0,965 a 1,000 sur les
# trois morts reelles, 0,663 a 0,710 sur les captures qui n'en portent aucune.
MIN_SCORE = 0.85
# Deux marques plus proches que cela sont le meme glyphe vu deux fois.
MIN_SPACING = 8
_GLYPH_FILE = "death_glyph.npz"
_GLYPH: NDArray[np.uint8] | None = None


def load_glyph() -> NDArray[np.uint8]:
    global _GLYPH
    if _GLYPH is None:
        ref = resources.files("jev_tactics.perception") / "data" / _GLYPH_FILE
        with resources.as_file(ref) as path:
            _GLYPH = np.load(path)["glyph"]
    return _GLYPH


def death_marks(frame: NDArray[np.uint8]) -> list[tuple[int, int]]:
    """Positions (x, y) des cranes de mort visibles dans le chat, du plus sur au moins sur.

    Cherche sur TOUTE la frame, et non dans une boite : le panneau de chat se deplace et se
    redimensionne au gre du joueur, et un rectangle fixe aurait la meme fragilite que ceux
    qu'on vient de retirer ailleurs. Le glyphe est assez distinctif pour s'en passer --
    c'est la mesure qui le dit, pas une intuition.
    """
    glyph = load_glyph()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if grey.shape[0] < glyph.shape[0] or grey.shape[1] < glyph.shape[1]:
        return []
    scores = cv2.matchTemplate(grey, glyph, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(scores >= MIN_SCORE)
    taken = np.zeros(scores.shape, dtype=bool)
    marks: list[tuple[int, int]] = []
    half = (glyph.shape[1] // 2, glyph.shape[0] // 2)
    for y, x in sorted(zip(ys, xs, strict=True), key=lambda t: -scores[t]):
        window = taken[max(0, y - MIN_SPACING):y + MIN_SPACING,
                       max(0, x - MIN_SPACING):x + MIN_SPACING]
        if window.any():
            continue
        taken[y, x] = True
        marks.append((int(x) + half[0], int(y) + half[1]))
    return marks


# Amplitude de defilement exploree entre deux frames, en pixels. Le chat monte d'une ligne
# par message ; la hauteur de ligne mesuree sur capture.png est de 17 px (ecarts releves :
# 13 a 23 selon que la ligne porte des majuscules ou des accents).
MAX_SCROLL = 80
# Ecart tolere pour reconnaitre une meme marque apres defilement.
SAME_MARK = 6


def scroll_offset(before: NDArray[np.uint8], after: NDArray[np.uint8],
                  band: tuple[int, int, int, int]) -> int | None:
    """De combien de pixels le contenu de `band` a-t-il monte entre les deux frames ?

    -> None si la bande sort de l'image ou si rien ne correspond.

    Mesure de faisabilite, sur le chat de capture.png defile artificiellement : 0, 17 et
    34 px sont retrouves EXACTEMENT, avec une correlation de 1,0. Le texte du chat est un
    motif riche et non repetitif -- le cas favorable pour une correlation.
    """
    x0, y0, x1, y1 = band
    tall = after[max(0, y0 - MAX_SCROLL):y1 + MAX_SCROLL, x0:x1]
    patch = before[y0:y1, x0:x1]
    if patch.size == 0 or tall.shape[0] < patch.shape[0] or tall.shape[1] != patch.shape[1]:
        return None
    scores = cv2.matchTemplate(cv2.cvtColor(tall, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                               cv2.TM_CCOEFF_NORMED)
    top = max(0, y0 - MAX_SCROLL)
    return int(np.argmax(scores)) + top - y0


def new_death_marks(before: NDArray[np.uint8],
                    after: NDArray[np.uint8]) -> list[tuple[int, int]]:
    """Marques de mort presentes dans `after` et pas dans `before`, DEFILEMENT COMPRIS.

    C'est ce qui manquait pour que le lecteur serve a quelque chose. Une marque visible ne
    dit pas qu'une mort vient d'avoir lieu : le chat garde les anciennes a l'ecran, et
    compter les cranes ferait croire a une hecatombe permanente. Ce qui compte est
    l'APPARITION.

    Le chat monte d'une ligne a chaque message, donc une ancienne marque change de
    position sans etre nouvelle. On estime ce defilement par correlation, puis on rapproche
    chaque marque de son homologue deplacee. Ce qui n'a pas d'homologue vient d'arriver.

    Sans marque dans `before`, tout ce qu'on voit dans `after` est nouveau -- ce qui est
    juste, et ne demande aucune estimation.
    """
    fresh = death_marks(after)
    if not fresh:
        return []
    old = death_marks(before)
    if not old:
        return fresh

    xs = [x for x, _ in old] + [x for x, _ in fresh]
    ys = [y for _, y in old]
    band = (max(0, min(xs) - 60), max(0, min(ys) - 40),
            min(after.shape[1], max(xs) + 60), min(after.shape[0], max(ys) + 40))
    shift = scroll_offset(before, after, band)
    if shift is None:
        return fresh
    moved = [(x, y + shift) for x, y in old]
    return [(x, y) for x, y in fresh
            if not any(abs(x - mx) <= SAME_MARK and abs(y - my) <= SAME_MARK
                       for mx, my in moved)]
