"""Attendre que le jeu soit au premier plan avant de capturer.

Probleme constate a l'usage : lancer un script `--live` depuis le terminal capture le
TERMINAL, parce que c'est lui qui a le focus. L'utilisateur doit alors basculer sur le jeu
« tres vite », en course contre la capture -- et une capture ratee ne se voit pas comme
une erreur, elle se voit comme une perception qui ne trouve rien.

Un simple delai fixe ne resout pas le probleme, il le deguise : trop court on rate, trop
long on attend pour rien, et la bonne valeur depend de la machine. On attend donc la
CONDITION reelle -- la fenetre du jeu est-elle active ? -- ce qui est deterministe.

Le detecteur de fenetre est injectable : les tests n'ont jamais besoin d'un bureau.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

# Motifs cherches dans le titre de la fenetre active, en minuscules ; un seul suffit.
#
# MESURE, pas supposition -- et elle a corrige une erreur. Le titre observe sur le client
# reel est :
#
#     Centvingtkilosbench - Sacrieur - 3.6.10.10 - Release
#
# soit « <personnage> - <classe> - <version> - Release » : **le mot « Dofus » n'y figure
# pas**. Le motif « dofus » seul, choisi a priori, n'aurait jamais correspondu -- et
# l'attente aurait expire a chaque fois, silencieusement, en degradant vers la capture
# du terminal. Exactement la panne que ce module devait supprimer.
#
# « dofus » est conserve : le launcher et certaines configurations le placent bien dans
# le titre.
GAME_PATTERNS = ("dofus", "- release")
# Titres qui portent un motif de jeu SANS etre le jeu, et qu'il faut ecarter AVANT lui.
#
# Le panneau de controle du bot s'appelle « dofus-bot — panneau de controle ». Il contient
# donc « dofus », et `matches` le declarait etre le jeu. Tout ce que ce module protege
# s'effondrait alors d'un coup, et en silence :
#
#   - `FocusGuardedBackend` laissait passer les gestes, croyant le jeu devant. Les clics
#     partaient aux coordonnees du plateau pendant que le PANNEAU etait au premier plan --
#     ils atterrissaient donc sur le panneau, et rien n'arrivait dans le jeu ;
#   - le compteur de gestes avales restait a zero, donc le bilan ne signalait rien ;
#   - le controle de focus de l'orchestrateur concluait « c'est bien le jeu ».
#
# Le bot percevait tout correctement et n'agissait sur rien. Constate en session reelle,
# capture `20260818-212307-combat-vide-recolte.png` : le panneau recouvre la moitie gauche
# du jeu, un combat est en cours derriere, et la session ne fait rien.
#
# L'exclusion passe AVANT les motifs : une fenetre a soi ne doit jamais pouvoir se faire
# passer pour le jeu, quel que soit son titre par ailleurs.
NOT_THE_GAME = ("dofus-bot",)
DEFAULT_TIMEOUT = 30.0
POLL_INTERVAL = 0.25
# Delai laisse au jeu apres la bascule : l'animation de fenetre peut encore etre en cours,
# et capturer pendant produirait une image floue ou partielle.
SETTLE_DELAY = 0.4


def foreground_title() -> str:
    """Titre de la fenetre active. Chaine vide si l'information est indisponible.

    Jamais d'exception : l'indisponibilite doit degrader vers « on capture quand meme »,
    pas empecher d'utiliser les scripts.
    """
    if sys.platform != "win32":
        return ""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        handle = user32.GetForegroundWindow()
        if not handle:
            return ""
        length = user32.GetWindowTextLengthW(handle)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        return buffer.value
    except Exception:
        # Capture large ASSUMEE. Cette fonction est un confort ; si l'API Windows change,
        # est indisponible, ou echoue pour une raison qu'on n'a pas prevue, l'utilisateur
        # doit garder un script qui marche -- pas heriter d'une trace de pile a la place
        # de sa capture.
        return ""


def matches(window_title: str, patterns: tuple[str, ...] = GAME_PATTERNS) -> bool:
    """Vrai si le titre correspond a l'un des motifs (comparaison insensible a la casse).

    Les fenetres du bot lui-meme sont ecartees d'abord : cf. `NOT_THE_GAME`.
    """
    lowered = window_title.lower()
    if any(exclu in lowered for exclu in NOT_THE_GAME):
        return False
    return any(p.lower() in lowered for p in patterns)


def wait_for_game(
    patterns: tuple[str, ...] = GAME_PATTERNS,
    timeout: float = DEFAULT_TIMEOUT,
    title: Callable[[], str] = foreground_title,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_wait: Callable[[], None] | None = None,
    on_timeout: Callable[[str], None] | None = None,
) -> bool:
    """Bloque jusqu'a ce que la fenetre active corresponde a l'un des `patterns`.

    Rend True si la fenetre a ete vue, False si le delai a expire ou si le titre n'est pas
    lisible sur cette plateforme. **L'appelant doit capturer dans les deux cas** : mieux
    vaut une capture peut-etre mauvaise, que l'utilisateur verra, qu'un refus de
    fonctionner sur une plateforme ou le titre n'est pas accessible.

    `on_timeout` recoit le DERNIER TITRE OBSERVE. C'est ce qui rend l'echec reparable :
    sans lui, un motif qui ne correspond pas produit une attente inexpliquee, alors que
    l'utilisateur n'a besoin que de voir le titre reel pour le corriger.
    """
    current = title()
    if not current:
        return False        # titre indisponible : ne pas bloquer inutilement

    if matches(current, patterns):
        # Deja au premier plan : pas d'animation en cours, donc pas de delai de repos.
        return True

    if on_wait is not None:
        on_wait()

    deadline = clock() + timeout
    while clock() < deadline:
        current = title()
        if matches(current, patterns):
            sleep(SETTLE_DELAY)     # laisser l'animation de fenetre se terminer
            return True
        sleep(POLL_INTERVAL)

    if on_timeout is not None:
        on_timeout(current)
    return False


def grab_when_focused(
    capture,
    patterns: tuple[str, ...] = GAME_PATTERNS,
    timeout: float = DEFAULT_TIMEOUT,
    announce: Callable[[str], None] | None = None,
    **kwargs,
):
    """Capture une frame une fois le jeu au premier plan. Dit ce qu'elle attend.

    Le message compte autant que l'attente : un script qui bloque sans rien afficher est
    indiscernable d'un script fige.
    """
    def notify():
        if announce is not None:
            announce("bascule sur le jeu (Alt+Tab) — la capture partira toute seule")

    def explain(last_title: str):
        if announce is not None:
            announce(f"fenetre du jeu non detectee (derniere vue : « {last_title} ») — "
                     f"capture immediate. Ajuster le motif si ce titre est le bon.")

    wait_for_game(patterns=patterns, timeout=timeout, on_wait=notify,
                  on_timeout=explain, **kwargs)
    return capture.grab()
