"""Ecriture dans le presse-papier Windows, pour saisir du texte accentue.

POURQUOI PAS UNE FRAPPE TOUCHE A TOUCHE. Les noms de ressources de Dofus sont accentues
-- *Blé*, *Frêne*, *Orchidée*, *Pierre de Sélénite*. Une frappe simulee envoie des codes
de touches physiques : le caractere obtenu depend alors de la disposition clavier active,
et un `é` tape sur une disposition qui ne le porte pas au meme endroit sort faux ou ne
sort pas du tout. Le champ de recherche recevrait `Bl`, l'HDV afficherait autre chose que
ce qu'on croit, et le prix serait enregistre sous le mauvais nom. Rien ne leverait
d'erreur.

Coller depuis le presse-papier contourne entierement la question : le texte passe en
UTF-16 sans jamais repasser par un code de touche.

Pas de dependance nouvelle : `ctypes` suffit, et `pyperclip` ferait exactement ca.

Le repli evoque ailleurs -- chercher la plus longue sous-chaine SANS accent, puisque la
recherche de l'HDV fonctionne par sous-chaine -- vit dans `market/`, pas ici : c'est une
strategie de recherche, pas une affaire de presse-papier.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class ClipboardError(RuntimeError):
    """Le presse-papier n'a pas pu etre ecrit (verrouille par une autre application)."""


def _user32_kernel32() -> tuple:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    return user32, kernel32


def set_text(text: str) -> None:
    """Place `text` dans le presse-papier. Leve `ClipboardError` en cas d'echec.

    Le presse-papier est une ressource GLOBALE et partagee : une autre application peut
    le tenir ouvert au moment ou l'on ecrit. On leve plutot que de rendre un booleen --
    un collage silencieusement rate collerait le contenu PRECEDENT, donc chercherait
    l'objet d'avant et rangerait son prix sous le nom d'aujourd'hui.
    """
    user32, kernel32 = _user32_kernel32()
    if not user32.OpenClipboard(None):
        raise ClipboardError(
            f"presse-papier inaccessible (erreur {ctypes.get_last_error()}) — une autre "
            f"application le tient ouvert")
    try:
        user32.EmptyClipboard()
        buffer = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(buffer)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            raise ClipboardError("allocation memoire du presse-papier refusee")
        target = kernel32.GlobalLock(handle)
        if not target:
            raise ClipboardError("verrouillage memoire du presse-papier refuse")
        try:
            ctypes.memmove(target, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            raise ClipboardError(
                f"ecriture du presse-papier refusee (erreur {ctypes.get_last_error()})")
    finally:
        user32.CloseClipboard()
