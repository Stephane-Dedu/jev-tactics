"""Perception semantique : des pixels vers l'etat structure.

Trois sous-parties :
  - ui : lecture des compteurs PA/PM/PV (classifieur maison, Tesseract en fallback).
  - digits : le classifieur de chiffres lui-meme (police fixe, <1 ms).
  - entities : detecteur v1 par marqueurs de cellule, echantillonnes via la grille
    (cf. scripts/study_markers.py ; necessite une homographie de la carte courante).

L'agent ne consomme QUE des pixels ; la verite terrain (quand disponible) ne sert qu'a
mesurer l'ecart etat-percu / etat-reel, pas a decider.
"""

from jev_tactics.perception.entities import (
    CellMarker,
    cell_outline_points,
    detect_cell_markers,
    detect_entities,
    detect_markers_on_board,
    markers_to_entities,
)
from jev_tactics.perception.ui import DEFAULT_ROIS, Roi, UiReading, read_stat, read_ui

__all__ = [
    "DEFAULT_ROIS",
    "CellMarker",
    "Roi",
    "UiReading",
    "cell_outline_points",
    "detect_cell_markers",
    "detect_entities",
    "detect_markers_on_board",
    "markers_to_entities",
    "read_stat",
    "read_ui",
]
