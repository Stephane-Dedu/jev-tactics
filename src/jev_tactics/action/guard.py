"""Ne jamais envoyer d'entree a une fenetre qui n'est pas le jeu.

Le manque etait beant et purement structurel : `wait_for_game` n'est appele qu'UNE FOIS,
au demarrage des scripts. Ensuite plus rien ne verifie quoi que ce soit. Or une session
dure des minutes, et le focus peut partir pour mille raisons ordinaires -- on bascule
pour lire un message, une notification Windows s'invite, le jeu plante.

A partir de cet instant le bot continuait exactement comme avant : il deplacait le
curseur et cliquait dans la fenetre qui se trouvait devant. Navigateur, explorateur de
fichiers, n'importe quoi. Et la boucle de combat, elle, APPUIE SUR DES TOUCHES -- « 3 »,
« ctrl+1 » -- qui partent alors dans un champ de saisie ou un chat.

Deux consequences, la premiere etant la plus grave :

  - des clics et des frappes atterrissent hors du jeu, sans le moindre garde-fou ;
  - le bilan devient une fiction : le bot rapporte des deplacements et des recoltes
    tentes sur des captures d'un jeu qu'il ne pilote plus.

POURQUOI ICI ET NON DANS LES BOUCLES. Une verification par boucle serait a refaire dans
chacune, et oubliee dans la prochaine. La frontiere ou l'entree quitte le programme est
unique : c'est le seul endroit ou la garantie tient sans discipline.

Ce qui est laisse passer, volontairement : `sleep`. Attendre ne touche rien, et bloquer
l'attente desynchroniserait le bot du jeu au retour du focus.

Le compte des gestes ecartes est CONSERVE et rapporte. Une session qui n'a rien fait
parce qu'on avait bascule ailleurs doit le dire -- sans ce chiffre, elle ressemblerait
trait pour trait a une session sur une zone vide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jev_tactics.action.mouse import InputBackend


@dataclass
class FocusGuardedBackend:
    """Enveloppe un backend et n'y laisse passer les gestes que si le jeu a le focus.

    `focused` est injectable : aucun test n'a besoin d'une fenetre, et un script qui ne
    sait pas determiner le focus peut passer `lambda: True` en connaissance de cause.
    """

    inner: InputBackend
    focused: Callable[[], bool]
    dropped: int = field(default=0)

    def describe_drops(self) -> str | None:
        """Ce que le garde a AVALE pendant la session. -> None s'il n'a rien avale.

        Le compteur existait et n'etait lu nulle part -- une mesure calculee puis jetee,
        le defaut que ce projet trouve le plus souvent. Or il explique a lui seul le
        bilan le plus trompeur qu'une session puisse rendre.

        Si la fenetre du jeu n'a jamais le focus -- alt-tab oublie, ou titre qui ne
        correspond a aucun motif connu -- la capture continue de fonctionner, la
        perception aussi, et SEULS les gestes sont avales. Le bilan annonce alors « N
        clics de recolte sans effet » et « N deplacements sans effet », ce qui envoie
        chercher un defaut de ciblage la ou il n'y en a pas.
        """
        if not self.dropped:
            return None
        return (f"{self.dropped} geste(s) avale(s) : la fenetre du jeu n'avait pas le "
                f"focus. Les « sans effet » du bilan s'expliquent par la, et non par un "
                f"defaut de ciblage")

    def _allowed(self) -> bool:
        if self.focused():
            return True
        self.dropped += 1
        return False

    def move_to(self, x: int, y: int) -> None:
        if self._allowed():
            self.inner.move_to(x, y)

    def click(self, x: int, y: int) -> None:
        if self._allowed():
            self.inner.click(x, y)

    def press_key(self, key: str) -> None:
        if self._allowed():
            self.inner.press_key(key)

    def type_text(self, text: str) -> None:
        # Garde AUSSI necessaire ici que pour les touches, et pour la meme raison : sans
        # focus, le texte part dans la fenetre qui se trouve devant -- navigateur, chat,
        # champ de mot de passe.
        if self._allowed():
            self.inner.type_text(text)

    def sleep(self, seconds: float) -> None:
        # Toujours transmis : attendre ne touche rien, et sauter l'attente
        # desynchroniserait le bot du jeu des le retour du focus.
        self.inner.sleep(seconds)

    def describe(self) -> str:
        if not self.dropped:
            return "aucun geste ecarte (le jeu est reste au premier plan)"
        return (f"{self.dropped} geste(s) ECARTES — le jeu n'avait pas le focus. "
                f"Le bilan ci-dessus porte donc sur moins d'actions qu'il n'y parait")
