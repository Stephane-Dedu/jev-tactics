"""Faire jouer le bot sur le client reel : se deplacer, engager, combattre.

    python scripts/play.py --route 5,5 6,5 6,4                 # DRY-RUN : rien ne bouge
    python scripts/play.py --route 5,5 6,5 --execute           # pilote vraiment
    python scripts/play.py --goto 7,3 --spells configs/spells/sacrieur.json --execute

**DRY-RUN PAR DEFAUT, et ce n'est pas une precaution de style.** Ce script clique et
appuie sur des touches dans une fenetre de jeu. Le defaut inverse -- agir sauf si l'on
demande le contraire -- transformerait une faute de frappe en session pilotee.

CE QUE CE SCRIPT NE FAIT PAS : les quetes. Parler a un PNJ demande une perception qui
n'existe pas (ni le journal, ni les PNJ sur la carte, ni les fenetres de dialogue). Le
directeur de quetes sait deja quoi faire ; il lui manque des yeux. Ici on exerce donc les
trois briques qui, elles, sont completes -- et c'est la premiere fois qu'elles touchent le
jeu.

A LANCER APRES `scripts/doctor.py`... qui n'est pas encore porte. En attendant, verifier
au moins que la perception voit le plateau : si `--goto` ne lit aucune coordonnee, rien
d'autre ne marchera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jev_tactics.action.actions import Executor
from jev_tactics.action.mouse import DryRunBackend
from jev_tactics.bot.fight import play_fight
from jev_tactics.capture.focus import grab_when_focused
from jev_tactics.capture.screen import ScreenCapture
from jev_tactics.decision.search import SearchDecider
from jev_tactics.perception.coordinates import diagnose_position
from jev_tactics.pipeline import build_board, observe
from jev_tactics.rules.spells import load_spells
from jev_tactics.state.assemble import assemble_combat_state
from jev_tactics.world.navigation import Coord, Direction, MapGraph
from jev_tactics.world.walker import Walker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPELLS = ROOT / "configs" / "spells" / "example.json"


def make_grab(capture: ScreenCapture, announce=print):
    """Une frame, une fois le jeu au premier plan.

    Attendre le focus plutot que capturer tout de suite evite l'Alt+Tab en catastrophe,
    et le message compte autant que l'attente : un script qui bloque sans rien afficher
    est indiscernable d'un script fige.
    """
    return lambda: grab_when_focused(capture, announce=announce).image


def check_origin(capture: ScreenCapture) -> bool:
    """Le moniteur capture doit commencer a (0, 0). -> False sinon.

    TOUT LE PROJET SUPPOSE CETTE ORIGINE : la detection rend des coordonnees dans
    l'image, et `pydirectinput.click` les interprete en pixels ECRAN ABSOLUS. Sur un
    poste a plusieurs ecrans, le principal peut commencer ailleurs -- et alors chaque
    clic part decale d'une constante, souvent sur l'autre ecran.

    La panne serait totale et MUETTE : detection parfaite, zero combat, et tous les
    diagnostics accusant le detecteur. Une ligne ici l'evite.
    """
    left, top = capture.origin
    if (left, top) == (0, 0):
        return True
    print(f"  /!\ le moniteur capture commence a ({left}, {top}), pas a (0, 0).")
    print("      Chaque clic partira decale de cette constante. Capturer l'ecran")
    print("      principal, ou corriger l'origine avant de piloter.")
    return False


def coord(text: str) -> Coord:
    x, _, y = text.partition(",")
    return (int(x), int(y))


def make_backend(execute: bool):
    """Le backend reel n'est importe QUE si l'on pilote vraiment.

    `pydirectinput` est un extra optionnel : une session en dry-run ne doit pas exiger
    son installation, et le depot a deja paye une fois pour un extra importe trop tot
    (cf. `perception/ui.py`).
    """
    if not execute:
        return DryRunBackend()

    # LA CONSTRUCTION EST DANS LE TRY, pas seulement l'import de la classe.
    #
    # `DirectInputBackend` est definie dans `action/mouse.py` et s'importe toujours ; c'est
    # son `__init__` qui fait `import pydirectinput`. Garder l'import de la classe ne
    # capturait donc rien, et l'utilisateur recevait une trace de pile a la place du
    # message qui dit quoi installer. Le garde-fou existait et ne gardait rien -- meme
    # forme que le lint jamais lance.
    try:
        from jev_tactics.action.mouse import DirectInputBackend

        return DirectInputBackend()
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise SystemExit(
            f"pilotage indisponible ({exc.name} manquant) : installer l'extra action\n"
            f'    pip install -e ".[action]"\n'
            f"Sans lui, retirer --execute : le dry-run montre ce qui serait joue."
        ) from exc


def read_state_for(board):
    """Frame -> CombatState, ou None hors combat / etat inexploitable.

    `cursor=None` : sur une capture d'ecran la position reelle de la souris est une
    entree legitime, mais elle rend le resultat dependant de l'endroit ou se trouve le
    curseur. Ici on pilote la souris nous-memes, donc on l'ignore.
    """
    def read(frame, turn, previous):
        return assemble_combat_state(
            observe(frame, board, cursor=None), turn=turn, previous=previous,
            board=board)
    return read


def do_goto(args, backend, grab) -> int:
    graph = MapGraph.from_json(args.graph) if args.graph and Path(args.graph).exists() \
        else MapGraph()
    walker = Walker(grab, backend, graph=graph)
    here = diagnose_position(grab())
    if here.position is None:
        print(f"  coordonnees illisibles : {here.reason}")
        print("  rien d'autre ne peut marcher tant que ceci echoue.")
        return 2
    print(f"  depart {here.position.as_tuple()} -> {args.goto}")
    report = walker.travel_to(args.goto)
    print(f"  {report.describe()}")
    for verdict in report.verdicts:
        print(f"    {verdict.describe()}")
    if args.graph:
        graph.to_json(args.graph)
    return 0 if report.arrived else 1


def do_route(args, backend, grab) -> int:
    walker = Walker(grab, backend)
    ok = 0
    for step in args.route:
        verdict = walker.step(step)
        if verdict is None:
            print("  coordonnees illisibles : on s'arrete plutot que de cliquer a l'aveugle")
            return 2
        print(f"  {verdict.describe()}")
        ok += verdict.changed_map
    print(f"  {ok}/{len(args.route)} deplacement(s) effectif(s)")
    return 0


def do_fight(args, backend, grab) -> int:
    frame = grab()
    board = build_board(frame)
    if board is None:
        print("  plateau non detecte : hors combat, ou calibration a refaire")
        return 2
    spells = load_spells(args.spells)
    executor = Executor(board=board, backend=backend,
                        spell_keys=dict(args.keys or {}), seed=args.seed)
    report = play_fight(board, spells, grab, read_state_for(board),
                        SearchDecider(), executor, max_turns=args.max_turns)
    print(f"  {report.describe()}")
    return 0 if report.turns else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goto", type=coord, metavar="X,Y",
                        help="rejoindre cette carte, en recalculant a chaque pas")
    parser.add_argument("--route", type=Direction, nargs="+", metavar="DIR",
                        help="suite de directions (left/right/top/bottom)")
    parser.add_argument("--fight", action="store_true",
                        help="jouer le combat en cours jusqu'a sa fin")
    parser.add_argument("--spells", type=Path, default=DEFAULT_SPELLS)
    parser.add_argument("--keys", type=lambda s: dict(
        p.split("=", 1) for p in s.split(",")), default=None,
        help="raccourcis des sorts, ex. Ravage=1,Hostilite=2. Un sort absent est SAUTE "
             "en silence par l'executeur -- c'est le defaut le plus difficile a "
             "diagnostiquer du projet")
    parser.add_argument("--graph", type=Path, default=None,
                        help="graphe de cartes a charger et enrichir")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true",
                        help="PILOTER VRAIMENT. Sans ce drapeau, rien ne bouge.")
    args = parser.parse_args()

    if not (args.goto or args.route or args.fight):
        parser.error("choisir au moins --goto, --route ou --fight")

    backend = make_backend(args.execute)
    mode = "EXECUTION" if args.execute else "dry-run (rien ne bouge)"
    print(f"\n  mode : {mode}")

    with ScreenCapture() as capture:
        # L'ORIGINE EST VERIFIEE AVANT DE PILOTER, et seulement alors : en dry-run rien
        # n'est clique, donc un moniteur decale est sans consequence.
        if args.execute and not check_origin(capture):
            sys.exit(2)
        grab = make_grab(capture)

        code = 0
        if args.goto:
            code |= do_goto(args, backend, grab)
        if args.route:
            code |= do_route(args, backend, grab)
        if args.fight:
            code |= do_fight(args, backend, grab)
    sys.exit(code)


if __name__ == "__main__":
    main()
