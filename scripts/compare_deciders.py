"""Compare des decideurs sur LES MEMES combats, dans LES DEUX regimes.

    python scripts/compare_deciders.py
    python scripts/compare_deciders.py --fights 200 --spells configs/spells/sacrieur.json
    python scripts/compare_deciders.py --regime contraint --postures

Ce script existe parce que les chiffres de ce projet se sont deja fait piéger deux fois
par le meme oubli : **un taux de victoire ne veut rien dire sans son regime**.

  - les PV ennemis. Sur 32 captures reelles, 72 ennemis detectes, ZERO avec des PV
    lisibles. Tous les poids de `scoring.py` ont pourtant ete regles a PV connus, ou le
    solveur mesure 80 % contre 60 % sans. C'est le second chiffre qui decrit le jeu ;
  - la CONTRAINTE. A 10 PA et 200 PV un personnage ecrase tout et n'a aucun dilemme :
    17 tours sur 40 n'offrent qu'une seule intention. A 6 PA et 60 PV, aucun. Mesurer les
    postures dans le premier regime revient a mesurer un mur.

D'ou les deux regimes par defaut, et l'affichage systematique des deux colonnes de PV.
Les graines sont partagees : sans cela l'ecart mesure serait celui des scenarios.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from jev_tactics.decision.search import SearchDecider
from jev_tactics.planner.postures import POSTURES, posture_plans
from jev_tactics.rules.spells import load_spells
from jev_tactics.sim import benchmark, square_board

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPELLS = ROOT / "configs" / "spells" / "example.json"

# Deux regimes, choisis parce qu'ils repondent a des questions differentes.
REGIMES = {
    # Ce que le bot joue vraiment : des groupes faibles, des heures durant.
    "courant": dict(enemies=2, hp=100, enemy_hp=60, ap=6, mp=3),
    # La ou les intentions divergent, donc la ou un decideur peut se distinguer.
    "contraint": dict(enemies=2, hp=60, enemy_hp=60, ap=6, mp=3),
    # Le combat encercle, dont l'utilisateur se plaint. Demande 200+ PV pour ne pas
    # mesurer un plafond arithmetique (162 degats recus par tour contre 54 rendus).
    "encercle": dict(enemies=3, hp=200, enemy_hp=60, ap=6, mp=3),
}


def fixed_posture(name: str):
    """Politique qui joue toujours la meme intention, quand elle est proposee.

    Ce sont les BORNES du probleme : la meilleure et la pire intention fixe encadrent ce
    qu'un selecteur peut esperer. Un decideur qui ne bat pas la meilleure borne fixe ne
    justifie pas son cout.
    """
    def policy(board, state, spells):
        plans = posture_plans(board, state, spells)
        if not plans:
            return []
        for posture, cand in plans:
            if posture.name == name:
                return cand.plan.actions
        return plans[0][1].plan.actions
    return policy


def solver_policy(board, state, spells):
    return SearchDecider().decide(board, state, spells).plan.actions


def halfwidth(wins: int, fights: int) -> float:
    """Demi-intervalle a 95 %. AFFICHE, pas calcule en silence.

    A 40 combats il vaut ~8 points : deux politiques separees de cinq points n'y sont pas
    departagees, et les lire comme un classement est la facon la plus simple de se
    tromper avec des chiffres justes.
    """
    if not fights:
        return 0.0
    p = wins / fights
    return 1.96 * math.sqrt(p * (1 - p) / fights)


def run(name, policy, spells, fights, seed, board, regime) -> None:
    line = f"  {name:26}"
    for known in (True, False):
        start = time.time()
        result = benchmark(policy, spells, fights=fights, seed=seed, board=board,
                           enemy_hp_known=known, **regime)
        line += (f"  {result.win_rate:5.1%} +/-{halfwidth(result.wins, result.fights):4.1%}"
                 f" [{time.time() - start:3.0f}s]")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fights", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0,
                        help="graine partagee : sans elle l'ecart mesure serait celui "
                             "des scenarios, pas des politiques")
    parser.add_argument("--spells", type=Path, default=DEFAULT_SPELLS)
    parser.add_argument("--regime", choices=sorted(REGIMES), default="contraint")
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--postures", action="store_true",
                        help="ajouter une ligne par intention fixe (les bornes)")
    args = parser.parse_args()

    spells = load_spells(args.spells)
    board = square_board(args.board)
    regime = REGIMES[args.regime]

    print(f"\n  {args.spells.name} ({len(spells)} sorts) | regime {args.regime} : "
          f"{regime['enemies']} ennemis, {regime['hp']} PV, {regime['ap']} PA")
    print(f"  {args.fights} combats, graine {args.seed}\n")
    print(f"  {'':26}  {'PV connus':^18}{'PV INCONNUS':^18}")
    print(f"  {'':26}  {'(l arene)':^18}{'(le jeu)':^18}")

    run("solveur seul", solver_policy, spells, args.fights, args.seed, board, regime)
    if args.postures:
        for posture in POSTURES:
            run(f"toujours {posture.name}", fixed_posture(posture.name),
                spells, args.fights, args.seed, board, regime)

    print("\n  La colonne qui compte est la SECONDE : sur 32 captures reelles, 72 ennemis")
    print("  detectes et zero avec des PV lisibles. La premiere decrit l'arene, pas le jeu.")


if __name__ == "__main__":
    main()
