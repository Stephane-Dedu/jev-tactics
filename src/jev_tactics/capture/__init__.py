"""Capture d'ecran a la demande (tour-par-tour : pas de boucle temps reel).

Si un hot-path haute frequence apparait un jour (ex. recolte temps reel), c'est ici
qu'on isolerait un module Rust via PyO3 -- pas avant profiling.
"""

from jev_tactics.capture.focus import (
    GAME_PATTERNS,
    foreground_title,
    grab_when_focused,
    matches,
    wait_for_game,
)
from jev_tactics.capture.screen import Frame, ScreenCapture

__all__ = [
    "GAME_PATTERNS", "Frame", "ScreenCapture", "foreground_title", "grab_when_focused", "matches",
    "wait_for_game",
]
