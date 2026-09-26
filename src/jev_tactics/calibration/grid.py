"""Estimation de l'homographie depuis la GRILLE VISIBLE d'une capture de combat.

Pourquoi : le detecteur d'entites (perception/entities.py) echantillonne le contour des
cellules projetees ; il lui faut une homographie valide POUR LA CARTE COURANTE. La
calibration manuelle (scripts/calibrate.py) ne passe pas a l'echelle -- une carte, un clic.

Methode : **autocorrelation**, pas detection de droites.

    /!\\ Hough a ete essaye et MESURE : Canny sur la frame entiere sort 30 000 a 60 000
    droites (le decor, les sprites et l'UI produisent des contours partout) et des bases
    aberrantes (cellules ~25 px au lieu de ~93). Les lignes de grille sont une minorite
    noyee dans le bruit ; aucun reglage de seuil ne les isole.

La grille est en revanche le seul motif PERIODIQUE etendu de l'image. L'autocorrelation
2D (via FFT, theoreme de Wiener-Khintchine) fait donc ressortir le reseau : ses pics sont
exactement les vecteurs du reseau. Mesure sur 5 captures de combat : pics dominants
(+-46, +-23) -- les vecteurs primitifs -- accompagnes de leurs combinaisons (+-93, 0) et
(0, +-46). Robuste, sans seuil sensible.

Choix de la base parmi les pics : le reseau admet plusieurs bases equivalentes (p. ex.
{(0,46), (46,23)} engendre le meme reseau que {(46,23), (-46,23)}). On leve l'ambiguite
par la **convention iso Dofus** : e_x et e_y descendent tous deux, de part et d'autre de
la verticale, avec des normes egales.

Ancrage : l'origine obtenue est COHERENTE mais pas ABSOLUE (cf. §4.4 du doc d'archi) --
en coordonnees diagonales Ankama, distances et adjacences sont invariantes par
translation, ce qui suffit a toute la planification tactique. L'ancrage absolu (ids de
cellules du serveur) reste une etape distincte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.calibration.homography import CELL_COORDS

MIN_CELL_HALF_PX = 20.0    # demi-diagonale minimale plausible d'une cellule
MAX_CELL_HALF_PX = 140.0   # ... et maximale (garde-fous contre faux reseaux)
PEAK_THRESHOLD = 0.12      # part du pic central au-dela de laquelle un pic compte
MIRROR_TOLERANCE = 0.15    # ecart relatif tolere a la symetrie miroir (convention iso)
# Seuil bas de l'hysteresis, en fraction du seuil d'Otsu (cf. detect_board_cells).
# Mesure sur combat1.png : les cases occupees par un personnage scorent 16 a 25 pour un
# Otsu a 31,2, soit 0,51 a 0,79 fois le seuil. 0,5 les couvre toutes avec de la marge,
# tout en restant au-dessus du decor (score median 12,8).
LOW_THRESHOLD_RATIO = 0.5
# Au-dela, ce n'est plus un plateau de combat mais le decor : l'herbe environnante a le
# MEME pavage isometrique, donc un seuil trop permissif l'absorbe sans rien qui le
# signale. Mesure : un plateau reel fait 60 a 230 cases sur les captures disponibles ;
# une fuite en produit 500 a 2000 -- et un cas a 1988 a fait tomber tout le pipeline.
# On prefere alors la selection stricte (seuil d'Otsu seul), quitte a un plateau
# incomplet : un plateau trop petit prive de quelques cases, un plateau qui deborde
# fait viser du decor.
MAX_PLAUSIBLE_CELLS = 400
# Largeur de cellule plausible, en pixels. Mesure : 92 px sur TOUTES les captures reelles
# (le zoom de combat de Dofus 3.6 est fixe). La fourchette est large pour absorber un
# reglage d'interface, mais elle doit exclure les motifs periodiques de l'UI.
#
# Ce garde-fou vient d'une panne en jeu : sur une carte au plateau BEIGE, dont les lignes
# de grille sont bien plus faibles, l'autocorrelation a lache le plateau (force 0,28 au
# lieu de 0,61) et s'est rabattue sur un reseau de 36x18 px situe sur la BARRE DE SORTS.
# Le bot y a lu 18 « ennemis » -- les icones de sorts, orange comme les marqueurs -- et
# aucun joueur. Un plateau faux est pire qu'aucun plateau : il produit un etat complet et
# entierement fictif, sur lequel tout le reste raisonne serieusement.
CELL_WIDTH_RANGE = (60.0, 160.0)
MIN_STRENGTH = 0.35
STRENGTH_RATIO = 0.60      # force minimale d'une paire, relative a la meilleure
# Ecart tolere entre 2v et le pic le plus proche, pour tenir v pour un vecteur du reseau.
# Mesure sur les huit captures reelles :
#
#     vraie base (46,23)   2v tombe a 0,99 a 1,08 px d'un pic
#     parasite   (18,9)    2v tombe a 11,21 a 11,30 px d'un pic
#
# Separation totale, et le seuil se pose dans un intervalle vide de dix pixels.
HARMONIC_TOLERANCE = 4.0


@dataclass(frozen=True)
class GridEstimate:
    """Base du reseau estimee depuis l'image."""

    e_x: NDArray[np.float64]      # deplacement ecran (px) pour +1 en coordonnee i
    e_y: NDArray[np.float64]      # idem pour +1 en j
    origin: NDArray[np.float64]   # point ecran de la position de reseau (0, 0)
    strength: float               # force du pic retenu (0..1), indicateur de confiance
    board: NDArray[np.int64] = field(  # (M, 2) : positions (i, j) jouables du plateau
        default_factory=lambda: np.empty((0, 2), dtype=np.int64))

    @property
    def cell_size(self) -> tuple[float, float]:
        """(largeur, hauteur) du losange d'une cellule, en pixels."""
        return (abs(self.e_x[0] - self.e_y[0]), abs(self.e_x[1] + self.e_y[1]))

    def homography(self) -> NDArray[np.float64]:
        """Homographie affine reseau -> ecran (colonnes : e_x, e_y, origine)."""
        h = np.eye(3, dtype=np.float64)
        h[:2, 0] = self.e_x
        h[:2, 1] = self.e_y
        h[:2, 2] = self.origin
        return h


def _line_response(
    frame: NDArray[np.uint8], kernel_size: int = 9, mode: str = "blackhat"
) -> NDArray[np.float32]:
    """Rehausse les lignes de grille (structures FINES) sur toute la frame.

    Le chapeau haut-de-forme noir supprime le fond lentement variable (terrain, ombres)
    et ne laisse que les traits fins : c'est le signal sur lequel on cherche le reseau.

    LES LIGNES NE SONT PAS TOUJOURS SOMBRES. Sur les captures de reference le liseré des
    cases est plus fonce que le terrain, d'ou le chapeau NOIR. Sur le plateau clair
    d'Incarnam il est plus CLAIR que l'herbe : mesure sur `20260819-115112`, le chapeau
    noir ne trouve aucun reseau la ou le chapeau blanc en trouve un a 0,52. `mode` permet
    donc de demander l'autre polarite, ou les deux -- les valeurs par defaut sont celles
    d'origine, aucun appelant existant ne change de comportement.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if mode == "blackhat":
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    if mode == "tophat":
        return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return np.maximum(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
                      cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))


def _autocorrelation(response: NDArray[np.float32], crop_frac: float) -> NDArray[np.float64]:
    """Autocorrelation 2D normalisee de la zone centrale de la reponse de lignes."""
    height, width = response.shape
    cy, cx = height // 2, width // 2
    ch, cw = int(height * crop_frac / 2), int(width * crop_frac / 2)
    roi = response[cy - ch:cy + ch, cx - cw:cx + cw].copy()

    roi -= roi.mean()
    # Fenetrage de Hann : evite les pics parasites dus aux bords du crop.
    roi *= np.outer(np.hanning(roi.shape[0]), np.hanning(roi.shape[1]))

    spectrum = np.fft.rfft2(roi)
    ac = np.fft.irfft2(np.abs(spectrum) ** 2, s=roi.shape)
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    return ac / peak if peak > 0 else ac


def _refine(ac: NDArray[np.float64], y: int, x: int, radius: int = 2) -> NDArray[np.float64]:
    """Position sous-pixel d'un pic par centroide pondere sur un petit voisinage."""
    y0, y1 = max(0, y - radius), min(ac.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(ac.shape[1], x + radius + 1)
    patch = np.clip(ac[y0:y1, x0:x1], 0, None)
    total = patch.sum()
    if total <= 0:
        return np.array([float(x), float(y)])
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return np.array([float((patch * xs).sum() / total), float((patch * ys).sum() / total)])


def _lattice_vectors(ac: NDArray[np.float64]) -> list[tuple[NDArray[np.float64], float]]:
    """Pics de l'autocorrelation -> vecteurs du reseau (tries par force decroissante)."""
    height, width = ac.shape
    center = np.array([width // 2, height // 2], dtype=float)
    ys, xs = np.mgrid[0:height, 0:width]
    radius = np.hypot(ys - center[1], xs - center[0])

    valid = (radius >= MIN_CELL_HALF_PX) & (radius <= MAX_CELL_HALF_PX) & (ac > PEAK_THRESHOLD)
    coords = np.argwhere(valid)
    if not len(coords):
        return []
    strengths = ac[valid]

    vectors: list[tuple[NDArray[np.float64], float]] = []
    for idx in np.argsort(strengths)[::-1]:
        y, x = coords[idx]
        v = _refine(ac, int(y), int(x)) - center
        # Un pic et son oppose decrivent le meme vecteur : ne garder qu'un exemplaire,
        # et ecarter les doublons proches (largeur du pic).
        if any(min(np.linalg.norm(v - u), np.linalg.norm(v + u)) < 8.0 for u, _ in vectors):
            continue
        vectors.append((v, float(strengths[idx])))
        if len(vectors) >= 12:
            break
    return vectors


def _has_harmonic(vectors: list[tuple[NDArray[np.float64], float]],
                  v: NDArray[np.float64]) -> bool:
    """`v` a-t-il son DOUBLE parmi les pics ? Signature d'un vrai vecteur du reseau.

    Les pics valent au signe pres : l'autocorrelation est paire, donc -2v confirme autant
    que 2v.
    """
    target = 2.0 * v
    return any(min(float(np.linalg.norm(peak - target)),
                   float(np.linalg.norm(-peak - target))) <= HARMONIC_TOLERANCE
               for peak, _ in vectors)


def _select_iso_basis(
    vectors: list[tuple[NDArray[np.float64], float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
    """Choisit (e_x, e_y) selon la convention iso Dofus.

    La contrainte discriminante est la **symetrie miroir** : en vue iso, e_y est l'image
    de e_x par la verticale, e_y ~= (-e_x[0], e_x[1]). Elle ecarte les bases alternatives
    du meme reseau -- par exemple {(46,23), (0,46)}, qui engendre pourtant les memes
    points mais ne respecte pas la convention (mesure : sans cette contrainte, (0,46)
    passait pour un e_y valide a cause d'un x legerement negatif apres affinage
    sous-pixel).

    Selection en deux temps, car ni la force ni la longueur ne suffisent seules :
      - filtrer sur la FORCE (>= STRENGTH_RATIO x la meilleure) ecarte les pics parasites
        courts, qui sont faibles (mesure : un faux reseau a 0.30 detournait le choix) ;
      - puis prendre la paire la plus COURTE donne les vecteurs primitifs, car un
        sous-reseau (multiple entier de la vraie base) a une force comparable et serait
        sinon retenu a tort.

    Ces deux temps ne suffisent pas, et une capture reelle l'a montre. En phase de
    PLACEMENT (bugcarreblanc.png), un pic parasite a (18, 9) -- force 0,285 -- passait le
    plancher a 0,279 et l'emportait sur la vraie base (46, 23), forte de 0,464, PARCE QU'IL
    EST PLUS COURT. La largeur de cellule tombait a 36 px, sous le minimum admis, et
    `estimate_grid` rendait None : plateau introuvable, aucune entite, tour perdu -- sur
    une frame ou la grille est pourtant la plus nette de tout le jeu.

    Monter le plancher aurait regle ce cas et rien d'autre : un nombre magique cale sur une
    capture. Le troisieme temps est un CRITERE PHYSIQUE : l'autocorrelation d'un reseau
    periodique pique a TOUS les multiples entiers de ses vecteurs. Un vrai vecteur a donc
    son double parmi les pics ; un parasite ne l'a pas. Mesure sur les huit captures --
    2v a 1,0 px d'un pic pour (46, 23), a 11,2 px pour (18, 9), a chaque fois.

    Le filtre n'est pas bloquant : si aucun candidat n'est confirme, on retombe sur la
    selection precedente. Il ne peut donc qu'ecarter des parasites, jamais faire echouer
    une detection qui marchait.
    """
    # Chaque vecteur vaut aussi pour son oppose : on materialise les deux orientations.
    candidates = [(v, s) for v, s in vectors] + [(-v, s) for v, s in vectors]
    down = [(v, s) for v, s in candidates if v[1] > 1.0]  # e_x et e_y descendent

    pairs: list[tuple[NDArray[np.float64], NDArray[np.float64], float, float]] = []
    for a, sa in down:
        if a[0] <= 0:
            continue  # e_x va vers la droite
        mirror = np.array([-a[0], a[1]])
        for b, sb in down:
            if b[0] >= 0:
                continue  # e_y va vers la gauche
            if np.linalg.norm(b - mirror) / np.linalg.norm(a) > MIRROR_TOLERANCE:
                continue
            if abs(a[0] * b[1] - a[1] * b[0]) < 1e-6:
                continue  # colineaires : pas une base
            norm = float(np.linalg.norm(a) + np.linalg.norm(b))
            pairs.append((a, b, min(sa, sb), norm))
    if not pairs:
        return None

    floor = STRENGTH_RATIO * max(p[2] for p in pairs)
    strong = [p for p in pairs if p[2] >= floor]
    confirmed = [p for p in strong if _has_harmonic(vectors, p[0])]
    a, b, strength, _ = min(confirmed or strong, key=lambda p: p[3])
    return a, b, strength


def _edge_offsets(e_x: NDArray[np.float64], e_y: NDArray[np.float64],
                  per_edge: int = 8, inset: float = 0.5) -> NDArray[np.float64]:
    """Points d'echantillonnage du contour d'une cellule, relatifs a son centre.

    `inset` en fraction de la demi-diagonale : 0.5 = le bord exact de la case, moins pour
    echantillonner en retrait (le jeu dessine les marqueurs a ~0.40, cf. entities.py).
    """
    corners = np.array([inset * 2 * (e_x - e_y) / 2, inset * 2 * (e_x + e_y) / 2,
                        -inset * 2 * (e_x - e_y) / 2, -inset * 2 * (e_x + e_y) / 2])
    t = np.linspace(0.0, 1.0, per_edge, endpoint=False)[:, None]
    return np.vstack([corners[i] + t * (corners[(i + 1) % 4] - corners[i])
                      for i in range(4)])


def _estimate_phase(
    response: NDArray[np.float32],
    e_x: NDArray[np.float64],
    e_y: NDArray[np.float64],
    steps: int = 20,
    span: int = 4,
) -> NDArray[np.float64]:
    """Cale la PHASE du reseau : ou tombent reellement les centres de cellules.

    L'autocorrelation donne la base mais **perd la phase** (elle est invariante par
    translation). Sans ce recalage, les contours projetes tombent a cheval sur les
    cellules reelles et le detecteur d'entites echantillonne a cote.

    On balaie donc les decalages du domaine fondamental (le parallelogramme engendre par
    e_x et e_y) et on retient celui qui maximise la reponse de lignes le long des
    contours de cellules : par construction, c'est l'alignement ou les aretes projetees
    coincident avec les traits dessines par le jeu.
    """
    height, width = response.shape
    center = np.array([width / 2.0, height / 2.0])
    offsets = _edge_offsets(e_x, e_y)

    # Voisinage de cellules autour du centre de l'image (le plateau y est centre).
    ij = np.array([(i, j) for i in range(-span, span + 1) for j in range(-span, span + 1)])
    cell_centers = center + ij @ np.vstack([e_x, e_y])
    # (cellules, points, 2) -> tous les points a tester pour un decalage nul.
    base = cell_centers[:, None, :] + offsets[None, :, :]

    best_shift, best_score = np.zeros(2), -np.inf
    for a in np.linspace(0.0, 1.0, steps, endpoint=False):
        for b in np.linspace(0.0, 1.0, steps, endpoint=False):
            shift = a * e_x + b * e_y
            pts = base + shift
            xs = np.round(pts[..., 0]).astype(int)
            ys = np.round(pts[..., 1]).astype(int)
            ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if not ok.any():
                continue
            score = float(response[ys[ok], xs[ok]].mean())
            if score > best_score:
                best_score, best_shift = score, shift
    return center + best_shift


def score_cells(
    response: NDArray[np.float32],
    e_x: NDArray[np.float64],
    e_y: NDArray[np.float64],
    origin: NDArray[np.float64],
    reach: int = 34,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Note chaque position du reseau : son contour suit-il des lignes dessinees ?

    C'est le discriminant qui separe le PLATEAU du reste de l'ecran. Les panneaux d'UI
    ont eux aussi des traits fins (mesure : leur densite depasse celle du plateau, ce qui
    avait fait echouer un ancrage par densite brute), mais **pas a la periode ni a
    l'orientation du reseau**. En notant les contours de cellules a la base et a la phase
    estimees, seules les vraies cases du plateau obtiennent un score eleve.

    -> (scores, ij) ou ij[k] = (i, j) est la position du reseau notee scores[k].
    """
    height, width = response.shape
    offsets = _edge_offsets(e_x, e_y)
    ij = np.array([(i, j) for i in range(-reach, reach + 1)
                   for j in range(-reach, reach + 1)], dtype=np.int64)
    centers = origin + ij @ np.vstack([e_x, e_y])
    pts = centers[:, None, :] + offsets[None, :, :]

    xs = np.round(pts[..., 0]).astype(int)
    ys = np.round(pts[..., 1]).astype(int)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    np.clip(xs, 0, width - 1, out=xs)
    np.clip(ys, 0, height - 1, out=ys)

    sampled = response[ys, xs] * inside
    counts = inside.sum(axis=1)
    scores = np.where(counts > 0, sampled.sum(axis=1) / np.maximum(counts, 1), 0.0)
    # Une cellule majoritairement hors image n'est pas jugeable.
    scores[counts < offsets.shape[0] * 0.8] = 0.0
    return scores, ij


def detect_board_cells(
    response: NDArray[np.float32],
    e_x: NDArray[np.float64],
    e_y: NDArray[np.float64],
    origin: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Positions du reseau qui sont de VRAIES cases du plateau -> (M, 2) de (i, j).

    Seuillage par HYSTERESIS, puis composante connexe dans le repere du reseau.

    Pourquoi l'hysteresis, et pas un seuil unique. Mesure sur une capture reelle
    (`combat1.png`) : les cases OCCUPEES -- personnage, monstres -- scorent 16 a 25 la ou
    Otsu coupe a 31,2. C'est mecanique : le sprite recouvre les aretes de la case, donc
    moins de pixels de ligne sont echantillonnes. Un seuil unique les rejetait, la
    composante connexe se rabattait sur la moitie droite du plateau (60 cases sur ~130),
    et **le personnage tombait hors du plateau** -- donc aucune entite detectee, alors
    que la grille elle-meme etait parfaitement estimee.

    Un seuil unique ne peut pas trancher ici : il doit a la fois accepter une case
    occupee (score 16) et rejeter le decor (median 12,8). Deux seuils le peuvent, parce
    qu'ils exploitent une information que le seuil unique ignore -- la CONNEXITE. Une
    case faible n'est retenue que si elle prolonge des cases fortes.

      - seuil haut (Otsu) : les cases indiscutables, qui amorcent ;
      - seuil bas : les cases douteuses, retenues seulement si elles tiennent aux
        premieres ;
      - la composante gardee est celle qui contient le PLUS D'AMORCES, et non la plus
        etendue : au seuil bas, une zone d'UI ou de decor peut etre vaste sans contenir
        une seule case certaine.
    """
    scores, ij = score_cells(response, e_x, e_y, origin)
    positive = scores[scores > 0]
    if len(positive) < 30:
        return np.empty((0, 2), dtype=np.int64)

    values = (255 * (positive - positive.min())
              / max(positive.max() - positive.min(), 1e-9)).astype(np.uint8)
    otsu, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = positive.min() + (otsu / 255.0) * (positive.max() - positive.min())

    seeds = scores >= cutoff
    if not seeds.any():
        return np.empty((0, 2), dtype=np.int64)

    cells = _largest_seeded_component(ij, scores, seeds, cutoff * LOW_THRESHOLD_RATIO)
    if len(cells) > MAX_PLAUSIBLE_CELLS:
        # L'hysteresis a fui dans le decor : repli sur la selection stricte, qui ne peut
        # pas deborder puisqu'elle n'accepte que les cases indiscutables.
        cells = _largest_seeded_component(ij, scores, seeds, cutoff)
    return _fill_enclosed(cells)


def _fill_enclosed(cells: NDArray[np.int64]) -> NDArray[np.int64]:
    """Ajoute les positions VIDES entierement entourees par le plateau.

    Un trou cerne de plateau EST du plateau : le decor devrait, pour produire un tel trou,
    se trouver au milieu de l'aire de jeu. Ce qu'on y trouve reellement, ce sont les cases
    dont le sprite d'un combattant masque les aretes -- celles-la memes que l'hysteresis
    vise, et qu'elle ne rattrape que si elles TOUCHENT une amorce.

    Mesure sur les captures reelles : +2, +7, +10, +12 et +15 cases. Borne par
    construction, contrairement a l'enveloppe convexe, essayee puis ECARTEE : elle
    ajoutait jusqu'a +94 cases sur combat1920.png, c'est-a-dire qu'elle debordait. Or
    ajouter des cases qui n'en sont pas produit exactement le defaut signale en jeu --
    « il tape une case morte (terrain) ».

    CE QUE CECI NE RESOUT PAS : une troncature de BORDURE. Sur combat1.png, deux monstres
    se tiennent au-dela du plateau detecte. Leurs cases scorent 16,0 et 17,2, soit
    AU-DESSUS du seuil bas (15,6) : ce n'est donc pas le seuil qui les rejette, mais la
    CONNEXITE. Entre le personnage et elles s'etend un couloir de deux cases faibles
    (14,5 puis 9,3), et le decor a une mediane de 12,8 -- aucun seuil ne franchit ce
    couloir sans absorber la pelouse. Le remplissage, lui, interpole sans extrapoler.

    UNE FERMETURE MORPHOLOGIQUE en serait la generalisation naturelle -- elle comble aussi
    les echancrures, pas seulement les trous cernes. Essayee, et ecartee deux fois :
    appliquee ICI elle est sage (+4 a +14 cases selon la capture) et ne rattrape RIEN, meme
    a rayon 3, faute de rive lointaine ; appliquee au masque du seuil bas, avant la
    connexite, elle rattrape les deux monstres des le rayon 1 mais fait passer le plateau
    de 113 a 482 cases, puis 779 au rayon 2.

    Chiffres complets et QUATRE corrections ecartees : cf. TestKnownTruncationOnCombat1.
    """
    if not len(cells):
        return cells
    low = cells.min(axis=0)
    size = tuple(cells.max(axis=0) - low + 3)      # marge de 1 tout autour
    grid = np.zeros(size, np.uint8)
    grid[cells[:, 0] - low[0] + 1, cells[:, 1] - low[1] + 1] = 1
    empty = (grid == 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(empty, connectivity=4)
    # L'exterieur est la composante vide qui touche le coin de la marge.
    enclosed = np.argwhere((labels != labels[0, 0]) & (empty == 1))
    if not len(enclosed):
        return cells
    return np.vstack([cells, enclosed + low - 1]).astype(np.int64)


def _largest_seeded_component(
    ij: NDArray[np.int64],
    scores: NDArray[np.float64],
    seeds: NDArray[np.bool_],
    low: float,
) -> NDArray[np.int64]:
    """Composante connexe la plus riche en amorces, parmi les cases au-dessus de `low`."""
    candidates = ij[scores >= low]
    if not len(candidates):
        return np.empty((0, 2), dtype=np.int64)

    # Composantes connexes sur la grille (i, j) : rasteriser puis etiqueter.
    lo = candidates.min(axis=0)
    size = candidates.max(axis=0) - lo + 1
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    mask[candidates[:, 1] - lo[1], candidates[:, 0] - lo[0]] = 1
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    if count < 2:
        return np.empty((0, 2), dtype=np.int64)

    # Composante la plus RICHE EN AMORCES (et non la plus vaste).
    seed_ij = ij[seeds]
    inside = ((seed_ij >= lo) & (seed_ij <= candidates.max(axis=0))).all(axis=1)
    seed_labels = labels[seed_ij[inside, 1] - lo[1], seed_ij[inside, 0] - lo[0]]
    seed_labels = seed_labels[seed_labels > 0]
    if not len(seed_labels):
        return np.empty((0, 2), dtype=np.int64)
    best = int(np.bincount(seed_labels).argmax())

    rows, cols = np.nonzero(labels == best)
    return np.column_stack([cols + lo[0], rows + lo[1]]).astype(np.int64)


# FENETRES ET POLARITES ESSAYEES PAR LE REPLI, et pourquoi il en faut plusieurs.
#
# L'autocorrelation se calcule sur la part CENTRALE de l'image. Une part fixe suppose un
# plateau de taille fixe : quand le plateau est petit, la fenetre est surtout du decor et
# le pic du reseau s'y noie. Mesure sur `20260819-115025`, un vrai combat :
#
#     crop 0.25 -> force 0.45 (retenue)      crop 0.45 -> force 0.27
#     crop 0.35 -> force 0.34                crop 0.55 -> force 0.23 (rejetee)
#
# Le meme balayage APPLIQUE AUX REFERENCES montre qu'aucune valeur unique ne convient :
# a 0.25, `combat1.png` retient un parasite a 36 px de large. On essaie donc plusieurs
# fenetres et on tranche sur ce qu'elles EXPLIQUENT, pas sur leur force brute.
_FALLBACK_MODES: tuple[tuple[str, int], ...] = (
    ("blackhat", 5), ("tophat", 5), ("both", 5),
    ("blackhat", 9), ("tophat", 9), ("both", 9),
)
_FALLBACK_CROPS: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55)
# Nombre de cases en deca duquel un reseau ne decrit pas un plateau. Mesure : les vrais
# plateaux du depot vont de 50 a 393 cases ; la seule detection parasite du balayage --
# un ecran d'Hotel de vente -- en rend 31. L'intervalle est vide entre les deux.
MIN_FALLBACK_BOARD = 40


def _estimate_grid_fallback(
    frame: NDArray[np.uint8], detect_board: bool
) -> GridEstimate | None:
    """Second essai, quand le chemin nominal n'a rien trouve.

    IL NE PEUT QUE CONVERTIR UN ECHEC EN DETECTION : il n'est appele qu'apres un None, et
    ne revient donc jamais sur une estimation deja acceptee. Aucune capture qui marchait
    ne change de resultat -- la non-regression est structurelle, pas mesuree.

    CE QU'IL CORRIGE. Les trois captures de combat REEL du 19/08 rendaient toutes None,
    et c'est la panne qui empechait le bot de jouer un seul tour : sans plateau, pas
    d'entites, pas de zone de deplacement, pas de plan. Le diagnostic affiche etait
    « aucune grille detectee — hors combat ? » sur des frames ou le combat est
    incontestable (timeline pleine, bouton FIN DE TOUR allume).

    CE QUI REMPLACE LE SEUIL DE FORCE. `MIN_STRENGTH` est cale sur les grands plateaux de
    reference (force 0,57 a 0,62) ; les petits plateaux d'Incarnam plafonnent a 0,23. Le
    seuil ne mesure donc pas « est-ce un vrai reseau » mais « le plateau est-il grand ».
    Le repli lui substitue quatre contraintes independantes, toutes deja ecrites :
    symetrie miroir et confirmation par l'harmonique (`_select_iso_basis`), largeur de
    cellule plausible (`CELL_WIDTH_RANGE`), et surtout un plateau REELLEMENT explique
    (`MIN_FALLBACK_BOARD`). On retient le candidat qui explique le plus de cases : un
    reseau parasite en explique peu, par construction.
    """
    meilleur: GridEstimate | None = None
    for mode, kernel in _FALLBACK_MODES:
        response = _line_response(frame, kernel_size=kernel, mode=mode)
        for crop in _FALLBACK_CROPS:
            basis = _select_iso_basis(_lattice_vectors(_autocorrelation(response, crop)))
            if basis is None:
                continue
            e_x, e_y, strength = basis
            width = abs(e_x[0] - e_y[0])
            if not (CELL_WIDTH_RANGE[0] <= width <= CELL_WIDTH_RANGE[1]):
                continue
            origin = _estimate_phase(response, e_x, e_y)
            board = detect_board_cells(response, e_x, e_y, origin)
            if len(board) < MIN_FALLBACK_BOARD:
                continue
            if meilleur is None or len(board) > len(meilleur.board):
                meilleur = GridEstimate(
                    e_x=e_x, e_y=e_y, origin=origin, strength=strength,
                    board=board if detect_board else np.empty((0, 2), dtype=np.int64))
    return meilleur


def estimate_grid(
    frame: NDArray[np.uint8], crop_frac: float = 0.55, detect_board: bool = True
) -> GridEstimate | None:
    """Capture de combat -> geometrie du plateau, ou None si grille inexploitable.

    Trois inconnues, resolues successivement :
      1. **base** (e_x, e_y) par autocorrelation -- invariante par translation ;
      2. **phase** : ou tombent les centres de cellules (l'autocorrelation la perd,
         il faut une correlation directe) ;
      3. **plateau** : QUELLES positions du reseau sont de vraies cases jouables
         (`detect_board_cells`).

    /!\\ Le plateau de combat n'est PAS la carte 560 cellules. Mesure : a ce zoom
    (cellule 92x46), les 560 cellules couvriraient 2944x1472 px, alors que le plateau
    observe fait ~990 px de large, soit une dizaine de cellules. Les tables
    `CELL_COORDS` / ids 0..559 d'Ankama ne s'appliquent donc pas au plateau ; on utilise
    directement les coordonnees du reseau (i, j), et l'ensemble des cases valides est
    celui qu'on detecte. C'est suffisant : distances, adjacences et portees sont
    invariantes par translation, donc toute la planification tactique fonctionne sans ids
    absolus (cf. §4.4 du doc d'archi).

    `crop_frac` : part centrale analysee pour l'autocorrelation (exclure les bords
    ecarte l'UI laterale et le decor).
    """
    response = _line_response(frame)
    basis = _select_iso_basis(_lattice_vectors(_autocorrelation(response, crop_frac)))
    if basis is not None:
        e_x, e_y, strength = basis
        # Refuser un reseau invraisemblable plutot que le rendre. Un plateau FAUX est pire
        # qu'aucun plateau : il produit un etat complet et entierement fictif, sur lequel
        # tout le reste raisonne serieusement. Rendre None fait dire au bot « pas de
        # grille », ce qui est visible et se retente ; rendre la barre de sorts le fait
        # jouer.
        width = abs(e_x[0] - e_y[0])
        if CELL_WIDTH_RANGE[0] <= width <= CELL_WIDTH_RANGE[1] and strength >= MIN_STRENGTH:
            origin = _estimate_phase(response, e_x, e_y)
            board = (detect_board_cells(response, e_x, e_y, origin) if detect_board
                     else np.empty((0, 2), dtype=np.int64))
            return GridEstimate(e_x=e_x, e_y=e_y, origin=origin, strength=strength,
                                board=board)
    return _estimate_grid_fallback(frame, detect_board)


def estimate_homography_from_grid(
    frame: NDArray[np.uint8], crop_frac: float = 0.55
) -> NDArray[np.float64] | None:
    """Capture de combat -> homographie grille->ecran, ou None si echec.

    L'origine place la cellule (0, 0) au centre de l'image : ancrage COHERENT mais pas
    absolu (suffisant pour la planification tactique, cf. docstring du module).
    """
    estimate = estimate_grid(frame, crop_frac)
    return estimate.homography() if estimate else None


@dataclass(frozen=True)
class BoardMap:
    """Les cases jouables d'un plateau, indexees 0..N-1, avec leur geometrie ecran.

    Remplace la table `CELL_COORDS` (ids 0..559 d'Ankama) pour tout ce qui touche au
    combat : le plateau observe n'est pas la carte 560 (cf. `estimate_grid`). L'ordre des
    indices est deterministe (tri par j puis i) pour qu'un meme plateau donne toujours
    les memes ids d'une frame a l'autre.
    """

    cells: NDArray[np.int64]      # (N, 2) positions reseau (i, j), triees
    e_x: NDArray[np.float64]
    e_y: NDArray[np.float64]
    origin: NDArray[np.float64]

    @classmethod
    def from_estimate(cls, estimate: GridEstimate) -> BoardMap:
        order = np.lexsort((estimate.board[:, 0], estimate.board[:, 1]))
        return cls(cells=estimate.board[order], e_x=estimate.e_x,
                   e_y=estimate.e_y, origin=estimate.origin)

    def __len__(self) -> int:
        return len(self.cells)

    def center(self, index: int) -> NDArray[np.float64]:
        """Centre ecran (px) d'une case -> cible de clic."""
        i, j = self.cells[index]
        return self.origin + i * self.e_x + j * self.e_y

    def outline(self, index: int, per_edge: int = 12,
                inset: float = 0.5) -> NDArray[np.float64]:
        """Points ecran le long du contour d'une case (cf. `_edge_offsets`)."""
        return self.center(index) + _edge_offsets(self.e_x, self.e_y, per_edge, inset)

    def index_at(self, i: int, j: int) -> int | None:
        """Index de la case a la position reseau (i, j), ou None si hors plateau."""
        hit = np.nonzero((self.cells[:, 0] == i) & (self.cells[:, 1] == j))[0]
        return int(hit[0]) if len(hit) else None

    def nearest(self, x: float, y: float) -> int:
        """Case la plus proche d'un point ecran (projection inverse + plus proche)."""
        basis = np.vstack([self.e_x, self.e_y]).T
        ij = np.linalg.solve(basis, np.array([x, y], dtype=float) - self.origin)
        return int(np.linalg.norm(self.cells - ij, axis=1).argmin())

    def neighbours(self, index: int) -> list[int]:
        """Cases adjacentes (4-voisinage du reseau) presentes sur le plateau."""
        i, j = self.cells[index]
        out = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            found = self.index_at(int(i) + di, int(j) + dj)
            if found is not None:
                out.append(found)
        return out


def board_coverage(homography: NDArray[np.float64], shape: tuple[int, int]) -> float:
    """Part des 560 cellules qui tombent dans une image de forme (h, w).

    Sonde de sanite : une base correcte mais une origine mal calee se voit ici.
    """
    hom = np.hstack([CELL_COORDS, np.ones((len(CELL_COORDS), 1))]) @ homography.T
    pts = hom[:, :2] / hom[:, 2:3]
    height, width = shape
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] < width)
              & (pts[:, 1] >= 0) & (pts[:, 1] < height))
    return float(inside.mean())
