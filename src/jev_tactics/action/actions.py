"""Primitives de jeu : traduire un Plan en gestes.

C'est le dernier maillon de la boucle percevoir -> decider -> AGIR. La geometrie est deja
resolue (`BoardMap.center` donne la cible de clic exacte) ; ce module compose donc
seulement trajectoire, delais et raccourcis clavier.

Le `Plan` du solveur est execute action par action. Chaque sort a un raccourci clavier
(barre de sorts), le deplacement est un simple clic sur la case.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from jev_tactics.action.mouse import DryRunBackend, InputBackend, move_along_curve
from jev_tactics.action.timing import Delay, DelaySampler
from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import Action, Cast, EndTurn, Move

# Touche de fin de tour (defaut Dofus). Configurable si le joueur l'a remappee.
END_TURN_KEY = "f1"


@dataclass
class Executor:
    """Execute un plan sur le jeu. Par defaut en dry-run : rien ne bouge.

    `spell_keys` associe un nom de sort a son raccourci ; sans entree, le sort est
    ignore avec une trace -- mieux vaut sauter une action que cliquer au hasard.
    """

    board: BoardMap
    backend: InputBackend = field(default_factory=DryRunBackend)
    spell_keys: dict[str, str] = field(default_factory=dict)
    delays: DelaySampler = field(default_factory=DelaySampler)
    end_turn_key: str = END_TURN_KEY
    seed: int | None = None
    skipped: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def _pause(self, kind: Delay) -> None:
        self.backend.sleep(self.delays.sample(kind))

    def _click_cell(self, cell: int) -> None:
        x, y = self.board.center(cell)
        move_along_curve(self.backend, (float(x), float(y)), rng=self._random)
        self.backend.click(int(round(x)), int(round(y)))

    def move_to_cell(self, cell: int) -> None:
        self._click_cell(cell)

    def cast(self, spell: str, target: int) -> bool:
        """Selectionne le sort puis clique la cible. -> False si le sort est inconnu."""
        key = self.spell_keys.get(spell)
        if key is None:
            self.skipped.append(spell)
            return False
        self.backend.press_key(key)
        self._pause(Delay.CLICK)
        self._click_cell(target)
        return True

    def end_turn(self) -> None:
        self._pause(Delay.END_TURN)
        self.backend.press_key(self.end_turn_key)

    def execute(
        self,
        actions: list[Action],
        finish_turn: bool = True,
        still_valid: Callable[[], bool] | None = None,
    ) -> bool:
        """Joue une sequence d'actions. -> True si elle est allee a son terme.

        `still_valid` est consulte ENTRE deux actions. Un plan est calcule d'un bloc sur
        une seule frame, puis joue sans rien reobserver : tout ce qui change entre-temps
        lui echappe. Un allie qui acheve la cible, un piege, un poison -- et les actions
        suivantes partent sur une case vide.

        L'elagage du solveur couvre deja ce QU'IL a tue lui-meme ; ce controle-ci couvre
        ce qu'il ne pouvait pas prevoir.

        Interrompre ne termine PAS le tour : il reste des PA, et la session replanifiera
        sur une observation fraiche. C'est le sens de « changer de cible » -- on ne
        renonce pas au tour, on renonce a la suite d'un plan devenu faux.
        """
        self._pause(Delay.TURN_START)
        for index, action in enumerate(actions):
            if index:
                self._pause(Delay.BETWEEN_SPELLS)
                if still_valid is not None and not still_valid():
                    return False
            if isinstance(action, Move):
                self.move_to_cell(action.cell)
            elif isinstance(action, Cast):
                self.cast(action.spell, action.target)
            elif isinstance(action, EndTurn):
                self.end_turn()
                return True
        if finish_turn:
            self.end_turn()
        return True
