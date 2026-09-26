"""Deplacement de souris et clics.

Un saut instantane point-a-point est la signature de bot la plus triviale a detecter.
On trace donc une **courbe de Bezier** entre le point de depart et la cible, avec des
points de controle bruites, une vitesse en cloche (lente au depart et a l'arrivee) et un
leger depassement corrige -- ce que fait une main.

L'interface `InputBackend` separe la DECISION du geste de son EXECUTION : `DryRunBackend`
enregistre les evenements sans toucher la souris, ce qui rend toute la couche testable en
CI, sur une machine sans jeu ni ecran.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol


class InputBackend(Protocol):
    """Ce que la couche d'action attend d'un peripherique d'entree."""

    def move_to(self, x: int, y: int) -> None: ...

    def click(self, x: int, y: int) -> None: ...

    def press_key(self, key: str) -> None: ...

    def type_text(self, text: str) -> None: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass
class DryRunBackend:
    """Backend de test : enregistre les evenements au lieu de les jouer."""

    events: list[tuple] = field(default_factory=list)
    position: tuple[int, int] = (0, 0)

    def move_to(self, x: int, y: int) -> None:
        self.position = (x, y)
        self.events.append(("move", x, y))

    def click(self, x: int, y: int) -> None:
        self.position = (x, y)
        self.events.append(("click", x, y))

    def press_key(self, key: str) -> None:
        self.events.append(("key", key))

    def type_text(self, text: str) -> None:
        self.events.append(("type", text))

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", round(seconds, 3)))

    def moves(self) -> list[tuple[int, int]]:
        return [(x, y) for kind, x, y in
                (e for e in self.events if e[0] == "move")]

    def clicks(self) -> list[tuple[int, int]]:
        return [(x, y) for kind, x, y in
                (e for e in self.events if e[0] == "click")]

    def keys(self) -> list[str]:
        return [e[1] for e in self.events if e[0] == "key"]

    def typed(self) -> list[str]:
        return [e[1] for e in self.events if e[0] == "type"]


class DirectInputBackend:
    """Backend reel (pydirectinput). Importe paresseusement : la CI n'en a pas besoin."""

    def __init__(self) -> None:
        import pydirectinput

        pydirectinput.PAUSE = 0.0  # on gere nous-memes les delais (cf. timing.py)
        self._backend = pydirectinput

    @property
    def screen_size(self) -> tuple[int, int]:
        """Espace de coordonnees dans lequel ce backend CLIQUE, en pixels.

        A COMPARER A LA TAILLE DE LA CAPTURE, et rien ne le faisait. `pydirectinput` place
        le curseur en absolu, normalise contre la taille que Windows lui rapporte. La
        detection, elle, rend des coordonnees dans l'IMAGE. Les deux ne coincident que si
        les deux tailles sont les memes.

        LA MISE A L'ECHELLE D'AFFICHAGE les separe. A 125 %, un processus qui n'est pas
        declare « DPI aware » recoit une taille LOGIQUE (1536 x 864) tandis que la capture
        rend le tampon PHYSIQUE (1920 x 1080). Chaque clic part alors comprime d'un
        facteur, l'erreur croissant avec la distance a l'origine -- nulle en haut a gauche,
        maximale en bas a droite.

        Meme famille que l'origine de capture et que la resolution : une panne totale et
        muette, ou la detection est parfaite et ou tous les diagnostics accusent le
        detecteur. Elle se verifie en une comparaison.
        """
        largeur, hauteur = self._backend.size()
        return int(largeur), int(hauteur)

    def move_to(self, x: int, y: int) -> None:
        self._backend.moveTo(x, y)

    def click(self, x: int, y: int) -> None:
        self._backend.click(x, y)

    def press_key(self, key: str) -> None:
        """Appuie sur une touche, modificateurs compris (« ctrl+shift+1 »).

        Les barres de sorts au-dela de la premiere ligne sont liees a des combinaisons :
        la deuxieme ligne est Ctrl+chiffre, la troisieme Ctrl+Maj+chiffre. Sans support
        des modificateurs, deux tiers des sorts d'un personnage restent injouables --
        et, pire, un « ctrl+1 » envoye tel quel appuierait sur une touche inexistante,
        donc ne lancerait rien, en silence.
        """
        *modifiers, base = [part.strip() for part in key.lower().split("+")]
        for modifier in modifiers:
            self._backend.keyDown(modifier)
        try:
            self._backend.press(base)
        finally:
            # Relacher meme si l'appui echoue : un modificateur reste enfonce
            # transformerait tous les clics suivants en clics modifies.
            for modifier in reversed(modifiers):
                self._backend.keyUp(modifier)

    def type_text(self, text: str) -> None:
        """Saisit du texte par le presse-papier (Ctrl+V), pas touche a touche.

        Le pourquoi est dans `action/clipboard.py` : les noms de ressources sont
        accentues, et une frappe simulee depend de la disposition clavier.

        On ecrit le presse-papier AVANT d'appuyer : si l'ecriture echoue, elle leve, et
        le Ctrl+V n'a pas lieu. L'ordre inverse collerait le contenu precedent -- donc
        chercherait un autre objet, sans que rien ne le signale.
        """
        from jev_tactics.action.clipboard import set_text

        set_text(text)
        self.press_key("ctrl+v")

    def sleep(self, seconds: float) -> None:
        import time

        time.sleep(seconds)


def bezier_path(
    start: tuple[float, float],
    end: tuple[float, float],
    steps: int = 24,
    jitter: float = 0.18,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Trajectoire courbe de `start` a `end`, echantillonnee en `steps` points.

    Bezier cubique dont les points de controle sont decales perpendiculairement a la
    trajectoire (d'ou la courbure), avec une progression non uniforme : la vitesse est
    lente au depart et a l'arrivee, comme un geste humain.
    """
    random_source = rng or random.Random()
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    distance = math.hypot(dx, dy)
    if distance < 1.0:
        return [(int(round(x1)), int(round(y1)))]

    # Normale unitaire : c'est le long de cet axe qu'on ecarte les points de controle.
    nx, ny = -dy / distance, dx / distance
    amplitude = distance * jitter

    def control(t: float) -> tuple[float, float]:
        offset = random_source.uniform(-amplitude, amplitude)
        return (x0 + dx * t + nx * offset, y0 + dy * t + ny * offset)

    c1, c2 = control(0.33), control(0.66)

    points: list[tuple[int, int]] = []
    for step in range(1, steps + 1):
        raw = step / steps
        # Lissage en cloche (smoothstep) : acceleration puis deceleration.
        t = raw * raw * (3.0 - 2.0 * raw)
        u = 1.0 - t
        x = (u**3 * x0 + 3 * u**2 * t * c1[0] + 3 * u * t**2 * c2[0] + t**3 * x1)
        y = (u**3 * y0 + 3 * u**2 * t * c1[1] + 3 * u * t**2 * c2[1] + t**3 * y1)
        points.append((int(round(x)), int(round(y))))

    points[-1] = (int(round(x1)), int(round(y1)))  # finir exactement sur la cible
    return points


def move_along_curve(
    backend: InputBackend,
    target: tuple[float, float],
    start: tuple[float, float] | None = None,
    rng: random.Random | None = None,
) -> None:
    """Amene le curseur sur `target` en suivant une courbe."""
    origin = start if start is not None else getattr(backend, "position", (0, 0))
    for x, y in bezier_path(origin, target, rng=rng):
        backend.move_to(x, y)
