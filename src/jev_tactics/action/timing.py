"""Delais entre actions.

Les temps de reaction humains sont **asymetriques** : une masse autour d'une valeur
typique, et une longue queue a droite (hesitations, distractions). Un `sleep` constant ou
une gaussienne symetrique sont des signatures machine ; on tire donc dans une
**log-normale**, dont la forme correspond a la mesure de temps de reaction humains.

Ce n'est pas seulement une precaution : le projet vise explicitement un comportement
realiste (§4.5 du doc d'archi), et le budget de calcul (~50-250 ms) doit rester mesurable
SEPAREMENT de ces delais volontaires -- sinon on ne saurait plus ce qui est lent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Delay(str, Enum):
    """Nature du delai : on n'hesite pas autant avant un clic que devant un tour."""

    CLICK = "click"          # entre deux clics d'une meme action
    BETWEEN_SPELLS = "spell"  # entre deux sorts : il faut relire l'etat
    TURN_START = "turn"       # avant d'agir : lecture de la situation
    END_TURN = "end_turn"     # avant de valider la fin de tour


# (mediane en secondes, sigma de la log-normale). Sigma plus grand = queue plus longue.
PROFILES: dict[Delay, tuple[float, float]] = {
    Delay.CLICK: (0.28, 0.35),
    Delay.BETWEEN_SPELLS: (0.85, 0.45),
    Delay.TURN_START: (1.40, 0.55),
    Delay.END_TURN: (0.70, 0.40),
}

MIN_DELAY = 0.05   # jamais instantane
MAX_DELAY = 8.00   # borne la queue : au-dela c'est une anomalie, pas une hesitation


@dataclass
class DelaySampler:
    """Tirage des delais. `seed` rend les tests deterministes."""

    seed: int | None = None

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def sample(self, kind: Delay = Delay.CLICK) -> float:
        """Delai en secondes, tire dans la log-normale du profil demande."""
        median, sigma = PROFILES[kind]
        # mu = ln(mediane) : pour une log-normale, la mediane vaut exp(mu).
        import math

        value = self._random.lognormvariate(math.log(median), sigma)
        return min(MAX_DELAY, max(MIN_DELAY, value))
