"""Detection des ressources recoltables par DIFFERENCE DE SURBRILLANCE.

La touche Y met les elements interactifs en surbrillance. Plutot que d'apprendre a
reconnaitre chaque sprite de ressource -- qui change avec les saisons, les skins et les
niveaux -- on laisse le jeu les designer :

    capture A (normale) -> presser Y -> capture B (surlignee) -> |B - A| -> ressources

Ce que ca evite : un detecteur a entrainer, des templates a maintenir par type de
ressource, et les faux positifs du decor (le probleme exact qui avait fait echouer la
detection de marqueurs par couleur en combat, ou le terrain imitait la cible).

Ce que ca coute : deux captures et un appui touche, soit ~100 ms. Negligeable devant le
budget farming (< 2 s par cycle).

Robustesse : la difference capte aussi ce qui a bouge entre les deux prises (animations,
autres joueurs, feuillage). On filtre donc par TAILLE -- une ressource surlignee forme un
blob compact et net, une animation produit des changements diffus ou minuscules.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.ui import ui_mask

# Ecart d'intensite au-dela duquel un pixel est considere comme ayant change.
DIFF_THRESHOLD = 28
# Taille plausible d'une ressource surlignee, en pixels.
MIN_AREA = 180
MAX_AREA = 40_000
# Nombre de ressources au-dela duquel un scan est INVRAISEMBLABLE : une carte de Dofus
# n'en porte pas des dizaines. Estimation, pas mesure -- elle sert a SIGNALER, jamais a
# refuser. Une session temoin en a rapporte jusqu'a 45 sur une frame.
SUSPICIOUS_RESOURCE_COUNT = 40
# Une ressource est un blob compact : on ecarte les trainees fines (bordures, animations).
MIN_EXTENT = 0.18      # aire / aire de la boite englobante
MAX_ASPECT = 6.0       # rapport largeur/hauteur (ou son inverse)


@dataclass(frozen=True)
class ResourceSpot:
    """Une ressource recoltable, en coordonnees ecran (cible de clic directe)."""

    x: int
    y: int
    area: int
    width: int
    height: int

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)


def highlight_mask(
    before: NDArray[np.uint8], after: NDArray[np.uint8], threshold: int = DIFF_THRESHOLD
) -> NDArray[np.uint8]:
    """Masque des pixels qui se sont illumines entre les deux captures.

    On ne retient que les changements VERS LE CLAIR : la surbrillance ajoute de la
    lumiere. Ce choix ecarte d'emblee la moitie des changements parasites (ombres qui
    passent, sprites qui s'assombrissent).
    """
    if before.shape != after.shape:
        raise ValueError("les deux captures doivent avoir la meme taille")

    grey_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
    grey_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
    brighter = np.clip(grey_after - grey_before, 0, 255).astype(np.uint8)

    mask = (brighter >= threshold).astype(np.uint8) * 255
    # Fermeture : recoller un contour surligne que le seuil aurait morcele.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_resources(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    exclude: tuple[tuple[int, int, int, int], ...] = (),
) -> list[ResourceSpot]:
    """Ressources recoltables entre deux captures (avant / apres surbrillance).

    `exclude` : rectangles (x0, y0, x1, y1) a ignorer -- l'ATH, qui change pour ses
    propres raisons (chat qui defile, compteurs, infobulles) entre les deux captures.
    Ce n'est pas une precaution de confort : la position rendue ici est CLIQUEE telle
    quelle, donc une fausse ressource tombant sur la barre de sorts ne coute pas un cycle,
    elle lance un sort.

    Le masque est mis a zero AVANT l'etiquetage, la ou `detect_groups` filtre par
    centroide : ici on veut la garantie qu'aucun pixel d'ATH ne participe a un candidat,
    fut-il a cheval sur la bordure.

    Resultat trie par aire decroissante -- les plus grosses cibles sont les plus surement
    cliquables.
    """
    mask = highlight_mask(before, after)
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = 0
    # Les panneaux OUVERTS en plus des rectangles fixes : le chat defile, la carte des
    # quetes s'anime, et aucun des deux n'est dans `UI_ZONES`. Cf. `ui_mask`.
    if mask.shape == after.shape[:2]:      # tailles discordantes : masque neutre
        mask[ui_mask(after) > 0] = 0

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    spots: list[ResourceSpot] = []
    for index in range(1, count):
        x, y, width, height, area = (int(stats[index, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        if area / max(width * height, 1) < MIN_EXTENT:
            continue
        aspect = width / max(height, 1)
        if aspect > MAX_ASPECT or aspect < 1.0 / MAX_ASPECT:
            continue
        cx, cy = centroids[index]
        spots.append(ResourceSpot(x=int(round(cx)), y=int(round(cy)), area=area,
                                  width=width, height=height))
    return sorted(spots, key=lambda s: s.area, reverse=True)


def nearest_resource(
    spots: list[ResourceSpot], origin: tuple[int, int]
) -> ResourceSpot | None:
    """Ressource la plus proche d'un point ecran (le personnage, en general)."""
    if not spots:
        return None
    ox, oy = origin
    return min(spots, key=lambda s: (s.x - ox) ** 2 + (s.y - oy) ** 2)

def describe_scan_totals(scans: int, busiest: int, implausible: int) -> str | None:
    """Ce que les scans de recolte ont vu, quand cela demande une action. None sinon.

    `busiest_scan` etait calcule puis JETE — aucun affichage ne l'exposait. Or c'est la
    seule mesure capable de distinguer deux situations que le bilan confond, et qui
    appellent des gestes opposes :

        « 0 recoltes, 6 cartes »   la zone etait vide          -> changer de circuit
        « 0 recoltes, 6 cartes »   la touche n'est pas liee    -> lier la touche

    Constate en faisant tourner la chaine contre un faux jeu qui ne repond PAS a la touche
    Y : six cycles, zero ressource, et un bilan rigoureusement identique a celui d'une zone
    videe. C'est la panne la plus sournoise de la recolte, parce qu'elle ressemble a un
    resultat.

    FONCTION ET NON METHODE, et c'est la correction d'un defaut. La formulation vivait sur
    `FarmReport`, donc sur le bilan d'UN SEUL `run()`. L'orchestrateur l'appelait a chaque
    alternance et ecrasait le resultat :

        tour 1   3 scans invraisemblables, record 45   -> avertissement
        tour 2   aucun                                 -> None, qui EFFACE le precedent

        ce que le bilan de session retenait : rien
        ce qui s'etait passe                : 3 scans a plus de 40 ressources

    La phrase parle pourtant « de la session ». Elle ne peut donc etre juste que si on lui
    donne les totaux de la session -- ce que seule une fonction libre permet, sans
    dupliquer le texte a deux echelles. Les deux appelants restent : `FarmReport` pour le
    farm autonome, l'orchestrateur pour la session complete.

    `scans` a zero veut dire QUE PERSONNE N'A CHERCHE. « Aucune vue » ne veut alors rien
    dire, et l'avertissement enverrait lier une touche dont la session n'a pas besoin --
    un diagnostic juste dans son domaine, faux hors de lui.
    """
    if not scans:
        return None
    if busiest == 0:
        return ("aucune ressource vue sur AUCUN scan de la session — une zone videe "
                "donne le meme bilan qu'une touche de surbrillance non liee. "
                "Trancher : python scripts/doctor.py --farm")
    if implausible:
        return (f"{implausible} scan(s) a plus de {SUSPICIOUS_RESOURCE_COUNT} "
                f"ressources (record {busiest}) — une carte n'en porte pas autant, "
                f"la difference capte autre chose")
    return None
