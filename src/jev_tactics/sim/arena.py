"""Arene : faire jouer une politique contre l'adversaire de reference, et compter.

Jusqu'ici la qualite du solveur n'etait qu'affirmee -- ses plans « semblent plausibles ».
Ce module la CHIFFRE : taux de victoire, tours moyens, PV restants, sur des scenarios
tires au hasard.

Deux usages :
  - un garde-fou de non-regression : si une modification du solveur fait chuter le taux
    de victoire, on le voit ;
  - **la reference que le RL devra battre**. Une politique apprise qui ne depasse pas la
    recherche 1 tour est cassee, pas prometteuse -- et sans ce chiffre on ne peut meme
    pas le dire.

Les scenarios sont tires d'une graine explicite : deux executions comparent bien la meme
chose. Comparer des politiques sur des combats differents ne voudrait rien dire.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import Action
from jev_tactics.rules.spells import Spell
from jev_tactics.sim.combat import apply_actions, enemy_policy, winner
from jev_tactics.state import CombatState, Entity, Team

Policy = Callable[[BoardMap, CombatState, list[Spell]], list[Action]]

MAX_TURNS = 25


@dataclass
class FightResult:
    winner: Team | None
    turns: int
    hp_left: int
    damage_dealt: float
    # Diagnostic de defaite : un taux de victoire dit COMBIEN on perd, jamais COMMENT.
    # Sans ces trois nombres, toute tentative d'amelioration du solveur revient a
    # deviner ce qui cloche -- et une intuition non mesuree a deja coute une regression
    # (cf. §6.1.b du doc d'archi).
    enemies_left: int = 0
    enemies_start: int = 0
    damage_taken: float = 0.0

    @property
    def kills(self) -> int:
        return self.enemies_start - self.enemies_left


@dataclass
class BenchmarkResult:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0                    # combat non conclu dans la limite de tours
    turns: list[int] = field(default_factory=list)
    hp_left: list[int] = field(default_factory=list)
    # Anatomie des defaites uniquement : les moyennes tous combats confondus melangent
    # deux populations et ne disent rien. Ce qu'on veut savoir, c'est si l'on perd en
    # ayant tue personne (probleme d'agressivite) ou presque tout tue (probleme de
    # finition ou de survie).
    defeats: list[FightResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.fights if self.fights else 0.0

    def defeat_profile(self) -> str:
        """Comment les defaites arrivent, pas seulement combien il y en a."""
        if not self.defeats:
            return "aucune defaite"
        killed = [d.kills for d in self.defeats]
        starts = [d.enemies_start for d in self.defeats]
        turns = [d.turns for d in self.defeats]
        wiped = sum(1 for d in self.defeats if d.kills == 0)
        nearly = sum(1 for d in self.defeats
                     if d.enemies_left == 1 and d.enemies_start > 1)
        return (f"{len(self.defeats)} defaites : "
                f"{float(np.mean(killed)):.1f}/{float(np.mean(starts)):.1f} ennemis tues "
                f"en mediane {float(np.median(turns)):.0f} tours | "
                f"{wiped} sans aucune victime, {nearly} a un ennemi pres")

    @property
    def median_turns(self) -> float:
        return float(np.median(self.turns)) if self.turns else 0.0

    @property
    def median_hp_left(self) -> float:
        return float(np.median(self.hp_left)) if self.hp_left else 0.0

    def describe(self) -> str:
        return (f"{self.wins}/{self.fights} victoires ({self.win_rate:.0%}), "
                f"{self.losses} defaites, {self.draws} nuls | "
                f"mediane {self.median_turns:.0f} tours, "
                f"{self.median_hp_left:.0f} PV restants")


def square_board(size: int = 9) -> BoardMap:
    """Plateau carre plein, sans obstacle : un terrain neutre pour comparer."""
    return BoardMap(
        cells=np.array([(i, j) for j in range(size) for i in range(size)],
                       dtype=np.int64),
        e_x=np.array([46.0, 23.0]), e_y=np.array([-46.0, 23.0]),
        origin=np.array([500.0, 300.0]),
    )


def random_scenario(
    board: BoardMap,
    rng: random.Random,
    enemies: int = 2,
    hp: int = 100,
    enemy_hp: int = 60,
    ap: int = 6,
    mp: int = 3,
) -> CombatState:
    """Placement aleatoire du joueur et des ennemis sur des cases distinctes."""
    cells = rng.sample(range(len(board)), enemies + 1)
    entities = [Entity(entity_id="me", team=Team.ALLY, cell=cells[0], hp=hp,
                       hp_max=hp, ap=ap, mp=mp, is_self=True)]
    entities += [Entity(entity_id=f"e{i}", team=Team.ENEMY, cell=cell, hp=enemy_hp,
                        hp_max=enemy_hp, ap=ap, mp=mp)
                 for i, cell in enumerate(cells[1:])]
    return CombatState(turn=1, entities=entities)


def _refresh(state: CombatState, ap: int, mp: int) -> CombatState:
    """Recharge PA/PM en debut de tour."""
    return state.model_copy(update={
        "entities": [e.model_copy(update={"ap": ap, "mp": mp}) for e in state.entities],
        "turn": state.turn + 1,
    })


def run_fight(
    board: BoardMap,
    state: CombatState,
    spells: list[Spell],
    policy: Policy,
    max_turns: int = MAX_TURNS,
    ap: int = 6,
    mp: int = 3,
) -> FightResult:
    """Deroule un combat complet : politique contre adversaire de reference."""
    damage = 0.0
    turn = 0
    start_enemies = len(state.enemies())
    start_hp = state.self_entity().hp

    for turn in range(1, max_turns + 1):
        outcome = apply_actions(board, state, spells, policy(board, state, spells))
        state, damage = outcome.state, damage + outcome.damage_dealt
        if winner(state) is not None:
            break

        for enemy in [e for e in state.entities if e.team is Team.ENEMY]:
            actions = enemy_policy(board, state, spells, enemy.entity_id)
            state = apply_actions(board, state, spells, actions,
                                  actor_id=enemy.entity_id).state
        if winner(state) is not None:
            break

        state = _refresh(state, ap, mp)

    me = state.find_self()
    hp_left = me.hp if me else 0
    return FightResult(winner=winner(state), turns=turn, hp_left=hp_left,
                       damage_dealt=damage, enemies_left=len(state.enemies()),
                       enemies_start=start_enemies,
                       damage_taken=float(start_hp - hp_left))


def benchmark(
    policy: Policy,
    spells: list[Spell],
    fights: int = 50,
    seed: int = 0,
    board: BoardMap | None = None,
    enemies: int = 2,
    hp: int = 100,
    enemy_hp: int = 60,
    ap: int = 6,
    mp: int = 3,
    enemy_hp_known: bool = True,
) -> BenchmarkResult:
    """Fait jouer `policy` sur `fights` scenarios tires de `seed`.

    La graine est explicite pour que deux politiques soient comparees sur LES MEMES
    combats -- sinon l'ecart mesure serait celui des scenarios, pas des politiques.

    `hp` EXISTE PARCE QUE L'INSTRUMENT SATURE. Mesure du taux de victoire par nombre
    d'ennemis, aux valeurs par defaut :

        1 ennemi  100%    2 ennemis  82%    3 ennemis  2,5%    4 ennemis  0%

    A trois, le resultat est le meme quelle que soit la ponderation testee -- et ce n'est
    pas une faiblesse du solveur : trois adversaires infligent ~162 degats par tour a un
    joueur de 100 PV qui en rend ~54. Le plafond est ARITHMETIQUE. Un balayage lance la
    mesure un mur, pas une politique.

    Porter le joueur a 200 PV rend la fenetre discriminante, et le classement des poids y
    change (cf. `planner/scoring.py`) : c'est donc dans ce reglage-la qu'il faut mesurer
    le combat encercle, celui dont l'utilisateur se plaint.

        ennemis     1      2      3      4
        PV utiles  ~60   ~100   ~200   ~300      (ordre de grandeur, a affiner)

    `enemy_hp_known` EST L ECART LE PLUS IMPORTANT ENTRE CETTE ARENE ET LE JEU. Mesure sur
    les 32 captures disponibles : 72 ennemis detectes, ZERO avec des PV connus. Rien ne les
    lit -- ni l'OCR, reserve au joueur, ni la timeline, qui ne donne qu'un ratio sans
    maximum. `hp_known` est donc faux pour tout ennemi, en permanence.

    La consequence traverse toute la fonction d'evaluation : sans PV, `evaluate` ne peut ni
    plafonner les degats ni accorder la PRIME DE MISE A MORT -- le terme qui, seul, fait
    passer le taux de victoire de 0,6 % a 81 %. Mesure :

        PV ennemis connus (ce que l'arene mesurait)     85,8%      70,0% a 3 ennemis
        PV ennemis inconnus (ce que le bot fait)        36,7%      50,0%

    Quarante-neuf points. Tous les poids regles jusqu'ici l'ont donc ete dans un regime que
    le bot NE CONNAIT PAS. Le defaut par True preserve les mesures deja consignees ; le
    script de benchmark, lui, affiche desormais les DEUX.

    `enemy_hp`, `ap` et `mp` EXISTAIENT DEJA dans `random_scenario` et n'etaient pas
    transmis : l'arene ne savait donc mesurer qu'un seul appariement, 100 PV / 6 PA / 3 PM
    contre des ennemis a 60 PV. Or le personnage reel mesure sur la capture de combat en
    porte 1632, avec 10 PA et 4 PM -- et une zone a larves oppose des monstres faibles.
    L'appariement que le bot va reellement jouer des heures durant etait le seul que
    l'instrument ne pouvait pas exprimer.
    """
    arena = board or square_board()
    rng = random.Random(seed)
    result = BenchmarkResult()

    for _ in range(fights):
        state = random_scenario(arena, rng, enemies=enemies, hp=hp,
                                enemy_hp=enemy_hp, ap=ap, mp=mp)
        if not enemy_hp_known:
            for entity in state.entities:
                if entity.team is not Team.ALLY:
                    entity.hp_known = False
        # `ap`/`mp` DOIVENT etre transmis ici aussi. `run_fight` s'en sert pour recharger
        # les combattants au debut de chaque tour, avec ses PROPRES defauts (6/3) : un
        # scenario construit a 10 PA jouait donc son premier tour a 10 et tous les
        # suivants a 6, sans que rien ne le dise. L'instrument changeait le scenario
        # apres coup -- un chiffre credible et faux.
        fight = run_fight(arena, state, spells, policy, ap=ap, mp=mp)
        result.fights += 1
        result.turns.append(fight.turns)
        result.hp_left.append(fight.hp_left)
        if fight.winner is Team.ALLY:
            result.wins += 1
        elif fight.winner is Team.ENEMY:
            result.losses += 1
            result.defeats.append(fight)
        else:
            result.draws += 1
    return result
