"""Fonction d'evaluation d'un tour : ce que le solveur cherche a maximiser.

Les ponderations sont des constantes NOMMEES, pas des nombres magiques disperses : elles
sont le seul endroit ou l'on encode « ce qu'est un bon tour », et devront etre revues
face au jeu. C'est aussi ce que le RL apprendra a remplacer.

Le score combine, par ordre d'importance :
  1. les degats infliges, plafonnes aux PV restants (frapper un mort ne rapporte rien) ;
  2. une prime de mise a mort, car supprimer un ennemi supprime aussi ses degats futurs
     -- un effet que l'horizon d'un seul tour ne voit pas ;
  3. l'eloignement des ennemis survivants, approximation du risque encouru au tour
     adverse ;
  4. une penalite legere sur les ressources non depensees, pour departager a egalite.
"""

from __future__ import annotations

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.planner.legal import TurnState
from jev_tactics.rules.movement import grid_distance
from jev_tactics.state import CombatState, Team

# Valeur d'une mise a mort, au-dela des degats eux-memes. Le terme etait JUSTIFIE --
# supprimer un ennemi supprime aussi ses degats futurs, ce que l'horizon d'un seul tour ne
# voit pas -- mais sa valeur n'avait jamais ete mesuree. Balayage en arene, 60 combats par
# graine, 3 graines, 2 ennemis :
#
#     KILL_BONUS      0     15     30     60    120    300
#     victoires     0,6%  26,1%  77,8%  81,1%  81,1%  81,1%
#
# DEUX CHOSES QUE CE TABLEAU APPREND.
#
# La premiere : sans lui, le bot gagne 0,6 % des combats. C'est de loin le terme le plus
# important de la fonction d'evaluation -- devant la prudence, devant le plafonnement des
# degats. Un solveur qui maximise les degats bruts repartit ses coups et ne tue personne ;
# les ennemis frappent alors tous, tous les tours, jusqu'au bout.
#
# La seconde : a partir de 60, les resultats sont IDENTIQUES graine par graine (77/82/85
# pour 60, 120 et 300). Ce ne sont pas des moyennes voisines, ce sont les memes plans : le
# bonus y est deja dominant et l'augmenter ne change plus aucune decision. 60 est donc le
# debut d'un plateau, pas un point regle -- et il garde 3,3 points de marge au-dessus du
# genou, situe entre 15 et 30.
#
# REPRISE A TROIS ENNEMIS (200 PV), apres avoir constate que les classements s'inversent
# d'un regime a l'autre :
#
#     KILL_BONUS     15     30     60    120    300
#     victoires      0%   22,5%  70,0%  67,5%  66,7%
#
# Le « plateau a partir de 60 » n'en est pas un : c'est un PIC. Encercle, monter au-dessus
# de 60 degrade (-2,5 puis -3,3 points) la ou deux ennemis donnaient des resultats
# rigoureusement identiques. La valeur retenue est donc juste pour une raison plus etroite
# qu'il n'y paraissait -- au-dela, la mise a mort ecrase la prudence, et un bot qui
# s'acharne sur un mourant se fait entourer par les deux autres.
KILL_BONUS = 60.0
# Valeur d'une case de distance vis-a-vis d'un ennemi vivant.
#
# PREMIERE MESURE, 3 graines x 60 combats, 2 ennemis. Elle concluait a un plateau de 1,5 a
# 3,0 (81,1 % pour les deux) et a une falaise a 6,0 (0,0 %). Le plateau etait un ARTEFACT :
# a 3 graines, deux valeurs separees de 3 points se ressemblent.
#
# REPRISE a 6 graines x 40 combats, 2 ennemis :
#
#     prudence      1,5    2,0    2,5    3,0
#     victoires    83,3%  85,4%  86,2%  80,0%
#
# 2,5 y bat 1,5 sur les SIX graines, jamais moins bien : 82/85/90/90/78/92 contre
# 75/80/90/88/75/92. Et 3,0 se degrade deja, ce que la mesure grossiere ne voyait pas.
#
# TROISIEME MESURE, et la plus utile, parce qu'elle porte sur le cas dont l'utilisateur se
# plaint -- « entoure d'ennemis ». A 3 ennemis, l'arene standard est INGAGNABLE (2,5 %
# quelle que soit la prudence) : trois adversaires infligent ~162 degats par tour a un
# joueur de 100 PV. Ce n'est pas une faiblesse du solveur, c'est un plafond arithmetique.
# En portant le joueur a 200 PV, l'arene discrimine de nouveau :
#
#     prudence      1,5    2,5    3,0    4,0    5,0    6,0
#     victoires    60,0%  70,0%  70,8%  22,5%   0,8%   0,8%
#
# Deux enseignements. L'ecart entre 1,5 et 2,5 passe de 3 points a DIX quand on est
# encercle : la prudence sert la ou elle est cense servir. Et LA FALAISE SE DEPLACE -- 6,0
# a deux ennemis, 4,0 a trois. Une marge mesuree dans un regime ne vaut pas dans un autre.
#
# 2,5 est donc retenu : meilleur dans les deux regimes, jamais pire sur aucune graine, et
# a 1,6x de la falaise la plus proche connue -- la ou 3,0 n'en est qu'a 1,33x pour un gain
# de 0,8 point.
SAFETY_WEIGHT = 2.5
# Penalites sur les ressources NON DEPENSEES. Leur role est de DEPARTAGER a egalite, et
# elles doivent rester assez faibles pour ne jamais decider seules -- ce qu'elles ne
# faisaient pas.
#
# A 2,0 par PA, un lancer en rapportait 6 quel que soit son effet. Trois comportements
# absurdes en decoulaient, tous constates :
#
#   - frapper un ennemi DEJA TUE par la sequence en cours : degats plafonnes donc 0 point
#     gagne, mais 6 points de penalite en moins. Mesure : 0 lancer -> -16,5 | 1 -> 66,5 |
#     2 -> 72,5 | 3 -> 78,5, soit +6 par coup sans effet ;
#   - faire UN PAS AU HASARD quand aucun ennemi n'est detecte : rester avec 3 PM -> -13,5 |
#     avec 0 PM -> -12,0, soit +0,5 par PM depense pour rien ;
#   - le meme mecanisme partout ou plus rien d'autre ne varie.
#
# Les deux premiers ont ete corriges a leur source (elagage des cadavres, refus de jouer
# sans ennemi vu). Restait a savoir si BAISSER les poids coutait quelque chose. Mesure en
# arene, 60 combats par graine :
#
#     PA perdu   2,0    1,0    0,5    0,1    0,0
#     victoires  80,0%  80,0%  80,0%  80,6%  80,6%     (3 graines)
#
#     PM perdu   0,50   0,20   0,05   0,00
#     victoires  80,0%  80,0%  80,0%  81,1%            (3 graines)
#
#     PA=2,0 PM=0,50 -> 80,3%      PA=0,1 PM=0,05 -> 80,7%     (5 graines, 300 combats)
#
# Aucun cout mesurable, donc. L'arene ne modelise PAS les etats degeneres ci-dessus --
# elle a toujours un ennemi vivant a frapper -- ce qui explique qu'elle n'y voie rien :
# le departage n'y decide jamais. C'est precisement pourquoi le baisser est sans risque
# la ou elle mesure, et utile la ou elle ne mesure pas.
#
# 0,1 par PA reste un vrai departage (0,3 point pour un sort a 3 PA) sans jamais pouvoir
# l'emporter sur un seul point de degat reel.
#
# TROISIEME MESURE, a trois ennemis (200 PV), le regime ou tout le reste s'est inverse :
#
#     penalite (PA, PM/2)   0,0    0,1    0,5    2,0
#     victoires            71,7%  70,0%  69,2%  68,3%
#
# Monotone decroissante : la penalite COUTE, environ 1,7 point entre 0,0 et 0,1. L'ecart
# porte sur 120 combats, soit deux victoires -- c'est du bruit, pas un signal. On garde 0,1
# parce que l'arene ne modelise AUCUN des trois etats degeneres listes plus haut : elle a
# toujours un ennemi vivant a frapper, donc le departage n'y decide jamais. Elle ne peut
# pas mesurer ce a quoi ce terme sert ; elle peut seulement dire qu'il ne coute rien de
# mesurable la ou elle voit. C'est ce qu'elle dit.
# Cout d'un allie (ou de soi-meme) pris dans une zone, rapporte a sa part de vie perdue.
# Borne a cette valeur par combattant touche : un allie inflige donc au plus 0,5, ce qui
# reste sous un seul point de degat reel -- c'est un departage, pas un arbitrage.
FRIENDLY_FIRE_PENALTY = 0.5
WASTED_AP_PENALTY = 0.1   # PA non depense
WASTED_MP_PENALTY = 0.05  # PM non depense (moins grave : se replier est parfois juste)

# Poids de la survie. Mesure a l'origine de ce terme (arene, 150 combats) : 40 des 44
# defaites se jouent « a un ennemi pres » -- le solveur en tue un puis meurt sur l'autre
# -- et meme ses victoires finissent a 6 PV en mediane. Le combat est donc une course a
# laquelle il participe sans jamais tenir compte de SA propre survie : la prudence y
# valait autant a 100 PV qu'a 5.
#
# On ne rajoute pas un terme de plus, on rend le terme existant dependant du danger : a
# pleine sante l'eloignement compte peu, moribond il compte beaucoup.
# Balayage en arene. D'abord 3 graines (180 combats), puis 6 (360) pour departager, a
# DEUX ennemis :
#
#     SURVIVAL_WEIGHT   0,0    1,5    3,0    6,0    10,0
#     victoires        72,8%  80,0%  80,8%  81,7%  39,2%
#
# A 6,0 le resultat y etait meilleur ou EGAL sur les six graines : +0,9 point. J'ai garde
# 3,0 quand meme, en argumentant qu'un point de victoire en simulation ne paie pas de
# ramener la marge a la falaise de 3,3x a 1,7x.
#
# LA MESURE A TROIS ENNEMIS a tranche, et bien plus nettement que l'argument :
#
#     SURVIVAL_WEIGHT   0,0    3,0    6,0    10,0
#     victoires        60,8%  70,0%  19,2%   0,0%      (3 ennemis, 200 PV, prudence 2,5)
#
# 6,0 ne gagne pas 0,9 point : il fait TOMBER le taux de 70,0 % a 19,2 % des que le
# personnage est encercle. La falaise s'est deplacee de 10,0 a quelque part entre 3,0 et
# 6,0 -- exactement comme celle de la prudence, passee de 6,0 a 4,0.
#
# 3,0 est l'optimum mesure dans le regime encercle, et le cout de se tromper y est
# asymetrique : au-dessus, l'effondrement ; en dessous, -9 points.
SURVIVAL_WEIGHT = 3.0

# Prime d'ACHEVEMENT quand les PV absolus manquent, proportionnelle a l'entame de la cible.
#
# Raison d'etre. En jeu reel, AUCUN ennemi n'a de PV connus -- mesure sur 32 captures :
# 72 ennemis, zero `hp_known`. Sans eux, `evaluate` ne peut ni plafonner les degats ni
# accorder KILL_BONUS, et le solveur n'a alors plus aucune raison d'achever un blesse : il
# repartit ses coups. Le cout est enorme, mesure en arene (2 ennemis, 6 graines) :
#
#     PV ennemis connus       85,8 %
#     PV ennemis inconnus     35,0 %
#
# Ce terme rattrape la majeure partie de l'ecart avec la seule ENTAME (1 - PV/PVmax), qui
# ne demande pas de connaitre les PV absolus :
#
#     poids       0     x2     x6    x10    x20
#     victoires  35,0%  66,2%  69,6%  75,4%  75,4%
#
# x10 et x20 donnent des resultats IDENTIQUES graine par graine : plateau, pas point regle.
#
# CE QU'IL LUI MANQUE ENCORE, et il faut le dire : en jeu, l'entame n'est attribuable a
# aucun ennemi PRECIS. La timeline donne bien un ratio de PV par portrait, mais rien ne
# relie un portrait a une case -- `reconcile_with_timeline` ne fait correspondre que les
# NOMBRES. Attribuer au hasard ferait viser le mauvais blesse, ce qui est pire que de ne
# rien preferer.
#
# Le terme reste donc INACTIF tant que `hp`/`hp_max` valent le remplissage (entame nulle
# pour tous), et s'active des qu'une lecture par ennemi existe -- celle de l'infobulle de
# survol, `perception/mobinfo.py`, deja ecrite et deja capable de lire « 139/186 ».
WOUNDED_WEIGHT = 10.0


def danger_factor(state: CombatState) -> float:
    """Multiplicateur de prudence, de 1 (intact) a 1 + SURVIVAL_WEIGHT (a l'agonie).

    PV inconnus : facteur neutre. Supposer le pire rendrait le bot craintif sur une
    simple panne de lecture, et un bot qui fuit sans raison ressemble a un bot prudent
    -- meme piege que les sorts introuvables.
    """
    me = state.find_self()
    if me is None or not me.hp_max:
        return 1.0
    return 1.0 + SURVIVAL_WEIGHT * (1.0 - me.hp / me.hp_max)


def expected_damage(spell_damage: float, remaining_hp: int) -> float:
    """Degats utiles : ce qui depasse les PV restants est perdu."""
    return min(spell_damage, float(remaining_hp))


def evaluate(
    board: BoardMap,
    state: CombatState,
    turn: TurnState,
    damage: dict[str, float],
    spells: list | None = None,
) -> float:
    """Score d'un tour termine dans l'etat `turn`, ayant inflige `damage` par entite.

    `spells` est accepte mais inutilise : un terme de POSITION a ete essaye ici et
    RETIRE, cf. §6.1.g du doc d'archi. Le parametre reste pour ne pas casser les appels.
    """
    score = 0.0
    prudence = SAFETY_WEIGHT * danger_factor(state)

    for entity in state.entities:
        if entity.team is not Team.ENEMY:
            # TIR FRATRICIDE. Ce `continue` etait total : les degats infliges a son propre
            # camp valaient exactement zero, et `_apply_damage` ne les enregistrait meme
            # pas. Entre deux cibles a egalite -- l'une epargnant l'allie, l'autre non --
            # le solveur tranchait au hasard de l'ordre d'enumeration.
            #
            # LE POIDS EST UN DEPARTAGE, et pas davantage, pour la meme raison que
            # WASTED_AP_PENALTY : borne par construction a FRIENDLY_FIRE_PENALTY par
            # combattant touche, il ne peut pas l'emporter sur un seul point de degat reel
            # quand un seul allie est en jeu. Le SIGNE, lui, n'est pas discutable.
            #
            # LA BONNE VALEUR EST INMESURABLE ICI, et il faut le dire : l'arene ne place
            # AUCUN allie (cf. `sim/arena.random_scenario`), donc ce terme est inerte dans
            # toutes les mesures deja faites -- KILL_BONUS, prudence, penalite de PA. C'est
            # ce qui rend son ajout sans risque pour l'acquis, et c'est aussi ce qui
            # empeche de le regler. Modeliser des allies dans l'arene est le prealable.
            subi = damage.get(entity.entity_id, 0.0)
            if subi and entity.hp_max:
                score -= FRIENDLY_FIRE_PENALTY * min(subi, entity.hp) / entity.hp_max
            continue
        raw = damage.get(entity.entity_id, 0.0)

        if not entity.hp_known:
            # PV inconnus : on compte les degats bruts, sans prime de mise a mort ni
            # plafonnement -- les deux supposeraient de savoir ce qu'il reste. Les traiter
            # comme mourants (le placeholder valait 1 PV) faisait surestimer chaque
            # attaque et deformait la priorisation des cibles.
            score += raw
            # Prime d'achevement sans PV absolus : plus la cible est entamee, plus la
            # frapper vaut. Nulle tant que les PV sont un remplissage (cf. WOUNDED_WEIGHT).
            if entity.hp_max:
                score += WOUNDED_WEIGHT * (1.0 - entity.hp / entity.hp_max) * raw
            score += prudence * grid_distance(board, turn.cell, entity.cell)
            continue

        dealt = expected_damage(raw, entity.hp)
        score += dealt
        if dealt >= entity.hp:
            score += KILL_BONUS
        else:
            # Seuls les survivants representent une menace au tour suivant.
            score += prudence * grid_distance(board, turn.cell, entity.cell)

    score -= WASTED_AP_PENALTY * turn.ap
    score -= WASTED_MP_PENALTY * turn.mp
    return score
