"""Pourquoi cette case est-elle vue, ou non ?

Ce module existe a cause d'une enquete qui a pris une iteration entiere : deux monstres
n'etaient pas detectes sur une capture, et il a fallu mesurer a la main, ligne par ligne,
le score de leurs cases, les seuils, la connexite, puis les ratios de marqueur a chaque
inset -- pour finalement decouvrir que je m'etais trompe de case, un sprite isometrique
etant dessine AU-DESSUS de la sienne.

Toute cette mesure, le pipeline la faisait deja. Elle etait simplement jetee.

`BoardInspector` la conserve et la rend interrogeable : on designe un point de l'ecran,
il repond ce qu'il a mesure pour cette case ET pourquoi il en a conclu ce qu'il en a
conclu. C'est la difference entre « le bot ne voit pas ce monstre » et « la case scorait
17,2 pour un seuil bas a 15,6, mais un couloir de deux cases faibles la coupait du
plateau ».

La logique vit ici, pas dans le script : elle doit etre testable sans fenetre ni souris.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.calibration.grid import (
    LOW_THRESHOLD_RATIO,
    BoardMap,
    _line_response,
    estimate_grid,
    score_cells,
)
from jev_tactics.perception.entities import (
    EDGE_INSETS,
    MIN_VOTE_RATIO,
    SAMPLES_PER_EDGE,
    _team_masks,
)
from jev_tactics.state import Team

# Reponse de ligne au centre au-dela de laquelle la case n'est pas creuse. Mesure sur
# trois captures : une case de plateau VIDE a une mediane de 0,7 a 2,2 et un maximum de
# 24,4 ; les deux monstres de combat1.png y montent a 29,5 et 48,1. Le seuil est place
# au-dessus du maximum observe des cases vides.
EMPTY_CENTRE_MAX = 25.0


@dataclass(frozen=True)
class CellDiagnosis:
    """Ce que la perception a mesure pour une case, et ce qu'elle en a conclu."""

    lattice: tuple[int, int]
    centre: tuple[int, int]
    on_board: bool
    line_score: float
    seed_threshold: float       # seuil haut (Otsu) : les cases indiscutables
    low_threshold: float        # seuil bas de l'hysteresis
    markers: dict[str, float] = field(default_factory=dict)
    vote_threshold: float = MIN_VOTE_RATIO
    # Reponse de ligne AU CENTRE de la case. Une case de plateau vide est creuse : mesure
    # sur trois captures, mediane 0,7 a 2,2 et maximum 24,4. Un gros sprite pose dessus y
    # laisse ses propres contours -- 29,5 et 48,1 pour les deux monstres de combat1.png.
    # Le personnage, lui, reste a 2,9 : son sprite est plus etroit. Le critere separe donc
    # les GROS occupants, pas tous, et il ne distingue pas le decor (mediane 9,5 hors
    # plateau). Il est rapporte parce qu'il informe, pas parce qu'il tranche.
    centre_response: float = 0.0
    # Rapport aretes/centre PROPRE A CETTE CASE. Sert a refuter localement un decalage de
    # reseau : sur le couloir de combat1.png il vaut 17, donc la grille y est bien calee
    # et les lignes y sont simplement faibles. Sans cette mesure j'ai cru a un desalignement
    # et failli corriger la mauvaise chose.
    local_phase: float = 0.0

    @property
    def best_team(self) -> tuple[str | None, float]:
        if not self.markers:
            return None, 0.0
        team, ratio = max(self.markers.items(), key=lambda kv: kv[1])
        return (team, ratio) if ratio >= self.vote_threshold else (None, ratio)

    def explain(self) -> str:
        """Une phrase qui dit ce qui s'est passe, avec les chiffres qui l'ont decide.

        Les cas sont enonces dans l'ordre ou ils bloquent : inutile de parler du marqueur
        d'une case qui n'est pas dans le plateau, puisqu'elle n'y sera jamais echantillonnee.
        """
        team, ratio = self.best_team
        if not self.on_board:
            if self.line_score >= self.low_threshold:
                return (f"HORS PLATEAU alors que son score ({self.line_score:.1f}) passe "
                        f"le seuil bas ({self.low_threshold:.1f}) : c'est la CONNEXITE "
                        f"qui l'exclut — elle ne touche aucune case retenue")
            # Le CENTRE tranche entre deux causes que le seul score confond, et que
            # cette phrase confondait : un sprite pose sur la case y laisse ses propres
            # contours (centre charge), alors qu'une case simplement peu dessinee est
            # creuse. La version precedente annoncait « un sprite » dans les deux cas --
            # y compris sur une case dont le centre mesurait 0,9.
            if self.centre_response >= EMPTY_CENTRE_MAX:
                return (f"HORS PLATEAU : score {self.line_score:.1f} sous le seuil bas "
                        f"({self.low_threshold:.1f}), mais le centre repond "
                        f"({self.centre_response:.1f}) — un SPRITE est pose dessus et "
                        f"masque les aretes")
            return (f"HORS PLATEAU : score {self.line_score:.1f} sous le seuil bas "
                    f"({self.low_threshold:.1f}), et le centre est creux "
                    f"({self.centre_response:.1f}) — les lignes sont simplement PEU "
                    f"DESSINEES ici, ce n'est pas un occupant")
        if team is not None:
            return (f"dans le plateau, marqueur {team} a {ratio:.2f} "
                    f"(seuil {self.vote_threshold}) — une entite devrait etre vue ici")
        if ratio > 0:
            return (f"dans le plateau, mais le meilleur marqueur ne vaut que {ratio:.2f} "
                    f"pour un seuil a {self.vote_threshold} : case tenue pour vide")
        return "dans le plateau, aucun marqueur : case vide"


@dataclass
class BoardInspector:
    """Conserve les mesures d'une frame pour pouvoir les INTERROGER case par case.

    Tout est calcule une fois a la construction (~150 ms) : cliquer doit repondre tout de
    suite, sinon on n'explore pas.
    """

    frame: NDArray[np.uint8]
    lattice: BoardMap = field(init=False)
    board: set[tuple[int, int]] = field(init=False)
    scores: NDArray[np.float64] = field(init=False)
    seed_threshold: float = field(init=False)
    low_threshold: float = field(init=False)
    _index: dict[tuple[int, int], int] = field(init=False, repr=False)
    _masks: dict = field(init=False, repr=False)
    # Filtre plein cadre : le recalculer par case rendrait un voisinage de 25 cases
    # inutilisable, alors que cliquer doit repondre tout de suite.
    _response: NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        estimate = estimate_grid(self.frame)
        if estimate is None or not len(estimate.board):
            raise ValueError("aucune grille detectee sur cette capture")
        self._response = _line_response(self.frame)
        scores, ij = score_cells(self._response, estimate.e_x,
                                 estimate.e_y, estimate.origin)
        self.scores = scores
        self.lattice = BoardMap(cells=ij.astype(np.int64), e_x=estimate.e_x,
                                e_y=estimate.e_y, origin=estimate.origin)
        self._index = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}
        self.board = {(int(a), int(b)) for a, b in np.asarray(estimate.board)}

        # Memes seuils que `detect_board_cells`, recalcules ici plutot que rendus par
        # elle : les exposer changerait sa signature pour un usage de diagnostic.
        positive = scores[scores > 0]
        values = (255 * (positive - positive.min())
                  / max(positive.max() - positive.min(), 1e-9)).astype(np.uint8)
        otsu, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.seed_threshold = float(positive.min()
                                    + (otsu / 255.0) * (positive.max() - positive.min()))
        self.low_threshold = self.seed_threshold * LOW_THRESHOLD_RATIO
        self._masks = _team_masks(self.frame)

    def cell_at(self, x: float, y: float) -> tuple[int, int]:
        """Point ECRAN -> maille (i, j), meme si elle n'est pas dans le plateau.

        Repondre pour une case hors plateau est tout l'interet : c'est justement la que se
        posent les questions.
        """
        matrix = np.array([self.lattice.e_x, self.lattice.e_y]).T
        i, j = np.linalg.solve(matrix, np.array([x, y]) - self.lattice.origin)
        return round(float(i)), round(float(j))

    def _marker_ratios(self, index: int) -> dict[str, float]:
        height, width = self.frame.shape[:2]
        out: dict[str, float] = {}
        for team, mask in self._masks.items():
            best = 0.0
            for inset in EDGE_INSETS:
                points = self.lattice.outline(index, per_edge=SAMPLES_PER_EDGE,
                                              inset=inset)
                xs = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
                ys = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
                best = max(best, float((mask[ys, xs] > 0).mean()))
            name = team.name if isinstance(team, Team) else str(team)
            out[name] = best
        return out

    def _responses(self, index: int) -> tuple[float, float]:
        """(reponse sur les ARETES, reponse au CENTRE) pour une case.

        Les deux ensemble disent ce qu'une seule ne dit pas : des aretes faibles AVEC un
        centre creux, c'est une case dont les lignes sont peu dessinees ; des aretes
        faibles avec un centre charge, c'est un sprite pose dessus.
        """
        response = self._response
        height, width = response.shape
        points = self.lattice.outline(index, per_edge=16, inset=0.5)
        xs = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
        centre = self.lattice.center(index)
        cx = int(np.clip(centre[0], 3, width - 4))
        cy = int(np.clip(centre[1], 3, height - 4))
        return (float(response[ys, xs].mean()),
                float(response[cy - 3:cy + 4, cx - 3:cx + 4].mean()))

    def diagnose(self, x: float, y: float) -> CellDiagnosis | None:
        """Diagnostic de la case sous un point ecran, ou None si hors du reseau."""
        key = self.cell_at(x, y)
        index = self._index.get(key)
        if index is None:
            return None
        centre = self.lattice.center(index)
        edges, middle = self._responses(index)
        return CellDiagnosis(
            lattice=key,
            centre=(round(float(centre[0])), round(float(centre[1]))),
            on_board=key in self.board,
            line_score=float(self.scores[index]),
            seed_threshold=self.seed_threshold,
            low_threshold=self.low_threshold,
            markers=self._marker_ratios(index),
            centre_response=middle,
            local_phase=edges / max(middle, 1e-9),
        )

    def neighbourhood(self, x: float, y: float, radius: int = 2) -> list[CellDiagnosis]:
        """Diagnostics des cases autour d'un point, pour LIRE LA CONNEXITE.

        Le defaut le plus difficile a comprendre n'etait pas dans la case elle-meme mais
        dans le couloir qui la separait du plateau. Une case seule ne pouvait pas le dire.
        """
        ci, cj = self.cell_at(x, y)
        out = []
        for i in range(ci - radius, ci + radius + 1):
            for j in range(cj - radius, cj + radius + 1):
                index = self._index.get((i, j))
                if index is None:
                    continue
                centre = self.lattice.center(index)
                out.append(self.diagnose(centre[0], centre[1]))
        return [d for d in out if d is not None]
