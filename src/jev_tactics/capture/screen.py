"""Capture d'ecran a la demande (combat tour-par-tour : pas de boucle temps reel)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mss
import numpy as np
from numpy.typing import NDArray


@dataclass
class Frame:
    """Une image capturee, horodatee. `image` est en BGR (convention OpenCV)."""

    image: NDArray[np.uint8]
    timestamp: float
    region: dict[str, int]


class ScreenCapture:
    """Capture via mss. `monitor=1` = ecran principal ; passer une region pour cpropre."""

    def __init__(self, monitor: int = 1) -> None:
        # `mss.mss` est DEPRECIE depuis mss 10 et disparaitra : la fabrique et la classe
        # rendent le meme objet -- verifie, meme type, meme capture 1920x1080 -- mais la
        # premiere emet un avertissement a chaque construction. C'est le point d'entree de
        # TOUTE la capture, donc de tout le bot.
        #
        # `getattr` plutot qu'un appel direct : le projet declare `mss>=9.0` et je ne peux
        # pas verifier si `MSS` etait deja expose en 9. Relever la contrainte sur une
        # supposition couterait plus que cette ligne.
        self._sct = getattr(mss, "MSS", mss.mss)()
        self._default = self._sct.monitors[monitor]

    def grab(self, region: dict[str, int] | None = None) -> Frame:
        """region = {'left','top','width','height'} en pixels ecran, ou None (moniteur)."""
        mon: Any = region or self._default
        raw = self._sct.grab(mon)
        bgr = np.ascontiguousarray(np.asarray(raw)[:, :, :3])  # BGRA -> BGR
        return Frame(image=bgr, timestamp=time.time(), region=dict(mon))

    @property
    def origin(self) -> tuple[int, int]:
        """Coin haut-gauche du moniteur capture, en pixels ECRAN.

        TOUT LE PROJET SUPPOSE (0, 0), et cette hypothese n'etait ecrite nulle part. La
        chaine est : `grab()` rend une image dont le pixel (0, 0) est ce coin ; la
        detection rend des coordonnees dans cette image ; `pydirectinput.click(x, y)` les
        interprete en coordonnees ECRAN ABSOLUES. Les deux ne coincident que si le coin est
        a l'origine.

        Sur un poste a plusieurs ecrans, `monitors[1]` -- le principal -- peut commencer
        ailleurs : un ecran secondaire place a gauche lui donne un `left` positif. Alors
        CHAQUE clic part decale d'une constante, souvent sur l'autre ecran.

        La panne serait totale et muette : detection parfaite, zero combat, et tous les
        diagnostics de la chasse accusant le detecteur. C'est exactement la forme de defaut
        que `check_layout` couvre pour la RESOLUTION -- meme classe, autre grandeur.
        """
        return int(self._default["left"]), int(self._default["top"])

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
