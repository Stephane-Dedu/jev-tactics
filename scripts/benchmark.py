"""Mesure la qualite du solveur contre l'adversaire de reference.

    python scripts/benchmark.py                       # 50 combats
    python scripts/benchmark.py --fights 200 --enemies 3 --hp 200
    python scripts/benchmark.py --compare             # solveur contre politique naive

Le chiffre produit sert deux fois : garde-fou de non-regression du solveur, et
REFERENCE QUE LE RL DEVRA BATTRE. Une politique apprise qui ne depasse pas la recherche
1 tour est cassee, pas prometteuse -- et sans ce chiffre on ne peut meme pas le dire.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from jev_tactics.planner import best_sequence
from jev_tactics.planner.legal import EndTurn, TurnState
from jev_tactics.planner.prune import useful_actions
from jev_tactics.rules.spells import load_spells
from jev_tactics.sim import benchmark, square_board

DEFAULT_SPELLS = Path(__file__).resolve().parents[1] / "configs" / "spells" / "example.json"


def solver_policy(board, state, spells):
    return best_sequence(board, state, spells).actions


def greedy_policy(board, state, spells):
    """Politique naive : le premier coup utile venu. Sert de plancher de comparaison."""
    turn = TurnState.from_combat(state)
    actions = [a for a in useful_actions(board, state, spells, turn)
               if not isinstance(a, EndTurn)]
    return actions[:1]


def report(name: str, result, elapsed: float) -> None:
    print(f"\n{name}")
    print(f"  {result.describe()}")
    print(f"  {result.defeat_profile()}")
    print(f"  {elapsed:.1f} s ({elapsed / max(result.fights, 1) * 1000:.0f} ms/combat)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=50)
    ap.add_argument("--enemies", type=int, default=2)
    ap.add_argument("--hp", type=int, default=100,
                    help="PV du joueur. Le defaut sature au-dela de deux "
                         "ennemis -- a trois, TOUTE ponderation donne 2,5 %% "
                         "parce que le plafond est arithmetique, pas "
                         "tactique. Compter ~200 PV a trois, ~300 a quatre")
    ap.add_argument("--enemy-hp", type=int, default=60,
                    help="PV de chaque ennemi. Une zone a monstres FAIBLES -- des larves "
                         "-- est l'appariement que le bot jouera des heures durant, et "
                         "c'etait le seul que l'arene ne savait pas exprimer")
    ap.add_argument("--ap", type=int, default=6,
                    help="PA de chaque combattant (le personnage reel mesure en porte 10)")
    ap.add_argument("--mp", type=int, default=3,
                    help="PM de chaque combattant (le personnage reel mesure en porte 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--spells", default=str(DEFAULT_SPELLS))
    ap.add_argument("--compare", action="store_true",
                    help="comparer au plancher (politique naive), memes scenarios")
    args = ap.parse_args()

    spells = load_spells(args.spells)
    board = square_board(args.size)
    print(f"{args.fights} combats, {args.enemies} ennemis a {args.enemy_hp} PV, "
          f"joueur {args.hp} PV / {args.ap} PA / {args.mp} PM, "
          f"plateau {len(board)} cases, graine {args.seed}")

    start = time.perf_counter()
    solver = benchmark(solver_policy, spells, fights=args.fights, seed=args.seed,
                       board=board, enemies=args.enemies, hp=args.hp,
                       enemy_hp=args.enemy_hp, ap=args.ap, mp=args.mp)
    report("solveur, PV ennemis CONNUS (hypothese d'arene)", solver,
           time.perf_counter() - start)

    # Le chiffre qui compte vraiment. Mesure sur 32 captures : 72 ennemis detectes, ZERO
    # avec des PV connus -- rien ne les lit. Sans eux, `evaluate` ne peut ni plafonner les
    # degats ni accorder la prime de mise a mort, le terme le plus lourd du score. Afficher
    # la seule ligne optimiste laisserait croire a une qualite que le bot n'a pas.
    start = time.perf_counter()
    reel = benchmark(solver_policy, spells, fights=args.fights, seed=args.seed,
                     board=board, enemies=args.enemies, hp=args.hp,
                     enemy_hp=args.enemy_hp, ap=args.ap, mp=args.mp,
                     enemy_hp_known=False)
    report("solveur, PV ennemis INCONNUS (ce que fait le bot)", reel,
           time.perf_counter() - start)
    print(f"  ecart du a la seule lecture des PV ennemis : "
          f"{solver.win_rate - reel.win_rate:+.1%}")

    if args.compare:
        start = time.perf_counter()
        naive = benchmark(greedy_policy, spells, fights=args.fights, seed=args.seed,
                          board=board, enemies=args.enemies, hp=args.hp,
                       enemy_hp=args.enemy_hp, ap=args.ap, mp=args.mp)
        report("plancher (premier coup utile)", naive, time.perf_counter() - start)
        gap = solver.win_rate - naive.win_rate
        print(f"\necart : {gap:+.0%} de victoires en faveur du solveur")
        if gap <= 0:
            print("  /!\\ le solveur ne fait pas mieux que le plancher — a investiguer")


if __name__ == "__main__":
    main()
