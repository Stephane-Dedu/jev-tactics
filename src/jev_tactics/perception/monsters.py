"""Detection des groupes de monstres sur la carte, par MOUVEMENT.

===========================================================================================
A LIRE AVANT DE CITER UN CHIFFRE DE CE MODULE : `data/runs/` N'EST PAS CE QU'IL PARAIT.

Les 24 captures de `data/runs/` sont des ECRANS DE COMBAT -- les 24, sans exception.
`read_timeline` y trouve 2 a 5 entrees, et rend 0 sur les trois captures hors combat du
depot (`hors_combat.png`, `capture.png`, `hdv_vide.png`) : la separation est nette, ce
n'est pas un artefact du detecteur. Un test le verifie a chaque execution.

ET ELLES PORTENT L'OVERLAY DU BOT INCRUSTE. `overlay.py` ecrit « MOI », « ENNEMI » et
les noms de sorts ; ces captures ont ete enregistrees ANNOTEES (`play_turn.py --save-image`
y ecrivait le calque, et rien d'autre -- il enregistre desormais aussi la frame brute, dans
`data/runs/brut/`). Ces libelles changent d'une frame a l'autre, donc ils produisent
eux-memes des candidats de mouvement, sur des images que la production ne voit jamais.

COMBIEN, ET LA REPONSE EST RASSURANTE ICI. Part des candidats dont la boite contient au
moins un pixel de calque, sur les 93 paires de meme carte : 129 sur 3578, soit 4 %. En les
retirant, les chiffres tires de ce dossier ne bougent pas :

    part relegue par l'enveloppe      21,4 %  ->  21,3 %
    relegue par l'AIRE seule              21  ->      21
    relegue par la HAUTEUR seule         250  ->     246
    scans portant une trace             8/93  ->    8/93

Les mesures de CHASSE tirees de `data/runs/` tiennent donc, contrairement a celles de la
detection d'ENTITES, ou 98 % des ennemis detectes portaient du calque sur leur contour. La
difference vient de ce que le calque se dessine sur les cases du plateau, la ou le
detecteur d'entites echantillonne exactement, alors que la chasse regarde tout l'ecran.

Or le scan de chasse tourne HORS combat. Toutes les mesures de ce module qui disent
« sur les N paires de data/runs/ » decrivent donc la difference entre deux images de
COMBAT : grille, timeline, barre de sorts, compte a rebours du tour, barres de vie. Un
ecran de chasse porte a la place un suivi de quetes et pas de grille.

CE QUE CA INVALIDE, ET CE QUE CA N'INVALIDE PAS.
  - Les comptages de candidats (20 a 49 par frame) sont vraisemblablement MAJORES : le
    compte a rebours du tour et les barres de vie s'animent, rien de tel a la chasse.
    `SUSPICIOUS_GROUP_COUNT` et les raisonnements de portee en heritent.
  - Les proprietes GEOMETRIQUES tiennent mieux : l'interface partagee (chat, panneaux,
    mini-carte) et le rendu du terrain sont les memes dans les deux etats.
  - Rien de tout cela n'a jamais ete confronte a une vraie paire de chasse, parce qu'il
    n'en existe AUCUNE dans le depot : les deux seules captures hors combat exploitables
    sont sur des cartes differentes, donc impossible d'en faire une difference.

LA SEULE FACON DE LEVER CE DOUTE est d'enregistrer deux captures consecutives hors
combat sur la meme carte. Tant que ce n'est pas fait, tout chiffre de ce module se lit
« mesure sur des ecrans de combat, transpose a la chasse par hypothese ».
===========================================================================================

Le probleme : hors combat, rien ne distingue un groupe de monstres du decor par la
couleur ou la forme. Le terrain est dense, varie d'une carte a l'autre, et un seuil
regle sur une zone echoue sur la suivante -- le projet a deja paye cette lecon sur les
marqueurs de cellule (70 a 120 faux candidats par frame).

Le signal retenu est donc **le mouvement**. Les groupes de monstres se deplacent en
permanence sur la carte ; le decor, non. Deux captures espacees de ~1 s, une difference,
et ce qui a bouge ressort. C'est la meme technique que la surbrillance de recolte
(`farming/probe.py`), mais sur le TEMPS plutot que sur une touche.

Ce que ca attrape aussi, et qu'il faut assumer :
  - **les autres joueurs**, qui bougent exactement pareil. Les engager serait une erreur ;
    aucun signal visuel simple ne les distingue, donc la parade est ailleurs -- verifier
    apres le clic qu'un combat a bien demarre, et sinon reprendre le circuit.
  - **le decor anime** (herbes, eau, feuillages). Filtre par taille et compacite, comme
    pour les ressources. Les seuils ci-dessous sont des POINTS DE DEPART a mesurer sur
    des captures reelles, pas des valeurs relevees -- ils sont marques comme tels.

Ce module ne decide de rien : il propose des candidats. C'est la machine a etats qui
choisit d'engager, et le runner qui verifie que le combat a demarre.

LA GRILLE DU JEU EST LISIBLE HORS COMBAT, et personne ne s'en servait. `estimate_grid`
rend une base identique a celle du combat sur les deux captures de carte du depot --
e_x = (46,0 ; 23,2) -- et un plateau de 263 et 268 cases. La chasse, elle, travaille en
PIXELS purs : distance au centre de l'ecran, liste noire a 30 px, positions de candidats.

CE QUE CA N'AUTORISE PAS, et c'est la premiere idee qui vient : filtrer les candidats
tombant HORS du plateau, au motif qu'ils ne seraient pas du terrain praticable. Le plateau
detecte sur `hors_combat.png` s'arrete a x = 615 alors que l'ecran fait 1920 de large, et
un troupeau de SEPT Bouftous se tient en x 280-580 -- entierement dehors. Le filtre
supprimerait un groupe entier.

C'est la meme troncature que `tests/test_grid.py::TestKnownTruncationOnCombat1` epingle en
combat : les cases masquees par des sprites ou du decor scorent bas, et la composante
connexe s'arrete avant. Elle n'est donc pas propre au combat, et elle interdit ici l'usage
le plus tentant de la grille.

CE QUE CA N'APPORTE PAS NON PLUS : quantifier les clics en cases. Sur les scans rejoues,
UN SEUL clic sur 744 tombe sur une case deja visee dans le meme scan (0,1 %) -- la
detection par composantes connexes separe deja les candidats bien au-dela d'une case.

PISTE MESUREE ET REFUTEE, consignee pour qu'on ne la reprenne pas. « Le decor s'anime SUR
PLACE, un monstre se deplace : il suffit donc d'ecarter les candidats qui reviennent au
meme endroit d'un scan a l'autre. » C'est l'idee qui fonde deja la memoire des positions
ratees du runner (`_spots_here`), et elle semble imparable. Mesure sur les trois series de
vues d'une meme carte de `data/runs/`, en formant toutes les paires et en regroupant les
candidats a 30 px pres :

    emplacements presents dans plus de la moitie des paires      18 a 43 %
    emplacements presents dans moins d'un cinquieme des paires   39 a 66 %

La majorite des candidats NE REVIENNENT PAS au meme endroit. Le bruit n'est donc pas un
decor qui clignote sur place mais un decor dont la zone qui change se DEPLACE dans la
texture -- de l'eau qui defile, un feuillage qui ondule. Un filtre par recurrence ecarterait
donc un cinquieme du bruit au mieux, et rien ne dit qu'il epargnerait les monstres.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import comb

import cv2
import numpy as np
from numpy.typing import NDArray

from jev_tactics.perception.ui import ui_mask

# Ecart de luminosite au-dela duquel un pixel est considere comme ayant bouge. Repris de
# `resources.py`, ou il a ete mesure : en dessous, le bruit de compression passe.
MIN_DIFFERENCE = 18
# Dilatation : un sprite qui se deplace produit deux taches (depart et arrivee) et des
# trous a l'interieur. On les recolle avant de mesurer.
CLOSE_KERNEL = 9

# --- Seuils de forme. ESTIMATIONS, pas mesures : aucune capture de carte avec monstres
# n'etait disponible a l'ecriture. Ils sont volontairement larges, et le script
# `scripts/study_monsters.py` sert a les recaler sur du reel.
# PLANCHER D'AIRE, ABAISSE DE 400 A 300, et c'est le premier reglage de ce fichier fait sur
# un arbitrage chiffre des DEUX cotes.
#
# CE QUI BLOQUAIT. `accepts` REFUSE, et c'est le seul mecanisme de la chasse capable de
# faire passer une carte pour vide : ce qu'il jette n'est jamais clique, donc jamais
# etiquete, donc invisible au journal quelle que soit sa duree. Le cout d'un plancher trop
# haut etait donc inconnaissable, et le cout d'un plancher trop bas -- plus de decor a
# cliquer, a huit secondes l'essai -- ne l'etait pas moins. Faute de pouvoir peser, on ne
# touchait a rien.
#
# LE COTE DROIT SE PESE MAINTENANT. `reach_probability` dit ce que des candidats
# supplementaires coutent en chance d'atteindre le vrai groupe en huit essais. Mesure sur
# les 23 paires de `data/runs/` :
#
#     plancher   candidats (med / max)   atteinte a la mediane   a la pire frame
#          150          34 / 55                  97,1 %              73,0 %
#          200          30 / 51                  98,7 %              79,5 %
#          300          26 / 47                  99,5 %              85,2 %
#          400          23 / 45                  99,8 %              87,8 %
#          700          18 / 37                 100,0 %              95,4 %
#
# De 400 a 300 : trois candidats de plus a la mediane, et TROIS DIXIEMES DE POINT de chance
# d'atteinte. En temps, le nombre attendu d'essais avant de toucher le vrai groupe passe de
# 3,5 a 3,9, soit trois secondes par carte.
#
# CE QUE CA ACHETE. Le combat reel le plus petit du journal fait 492 px, soit 23 % au-dessus
# de 400 -- la marge la plus mince de toutes les bornes (`study_engagements.py`). Sur sept
# combats, en avoir un si pres du plancher indique que la distribution des vrais monstres
# continue en dessous. Trois secondes par carte contre le risque de rendre une carte
# entiere invisible -- une carte perdue coute huit essais, soit 67 s, ET un cran vers
# l'arret de la session.
#
# 300 N'EST PAS UNE TAILLE DE MONSTRE MESUREE, et il ne faut pas le lire ainsi : personne
# n'a mesure l'aire d'un monstre A L'ARRET, dont seule l'animation d'attente bouge. C'est le
# dernier palier ou le cout reste negligeable -- en dessous il double a chaque cran (0,8
# puis 1,6 point). Le critere porte sur le COUT, pas sur le sujet.
#
# LE CLIQUET DE `TestNoBoundMayBeTightenedPastGroundTruth` reste tenu : il interdit de
# RESSERRER au-dela d'un combat deja releve, et desserrer va dans le sens qu'il autorise.
MIN_AREA = 300
# PLAFOND D'AIRE, ET SON VRAI ROLE N'EST PAS LE DECOR. Il figurait parmi les « seuils de
# forme » comme s'il ecartait de grosses taches d'animation. Mesure sur les 23 paires de
# `data/runs/` : il refuse 18 taches en tout, soit 0,8 par frame, de 61 519 a 709 618 px
# (3,0 a 34,2 % de l'ecran) -- et TOUTES sortent de l'enveloppe des combats enregistres,
# donc `beyond_recorded` les releguerait de toute facon.
#
# CE QU'IL ARRETE VRAIMENT, mesure en fabriquant les transitions que le jeu produit :
#
#     carte -> ecran noir (chargement)   1 tache de 2 073 600 px    tout l'ecran
#     ecran noir -> carte                1 tache de 1 630 623 px
#     deux cartes differentes            1 tache de   121 371 px
#
# Un changement de carte, un panneau qui s'ouvre, un fondu : la difference porte sur l'ecran
# entier et rend UNE composante geante. Sans ce plafond, elle devient un candidat cliquable,
# et son point d'ancrage tombe au milieu du plateau -- huit secondes perdues a cliquer une
# transition, au moment precis ou le bot ne devrait pas chasser du tout.
#
# `beyond_recorded` la releguerait (elle depasse largement l'enveloppe) mais reléguer ne
# suffit pas : sur la transition « ecran noir -> carte » il n'y avait que 3 autres
# candidats, donc les huit essais l'auraient atteinte.
MAX_AREA = 60_000
# PLANCHER DE COMPACITE (aire / boite englobante), ABAISSE DE 0,15 A 0,05. Ce n'est plus un
# discriminant, et le journal dit qu'il ne l'a jamais ete :
#
#     ordre (part des paires ou il met le combat devant)     40 %   sous le hasard
#     coupe sans perte (faux positifs retires sans cout)     aucune
#     recouvrement des deux plages                            50 %  le pire du tableau
#
# Un critere qui ORDONNE moins bien qu'un tirage au sort et qui ne peut retirer AUCUN faux
# positif ne separe rien. Il lui restait une utilite possible -- reduire le VOLUME de
# candidats, ce qui ameliore la chance d'atteindre le vrai groupe en huit essais. Mesure sur
# les 23 paires de `data/runs/` :
#
#     plancher   candidats (med / max)   atteinte a la pire frame
#         0,00          27 / 47                  85,2 %
#         0,15          26 / 47                  85,2 %
#         0,30          21 / 42                  91,1 %
#
# A 0,15 il retire 25 candidats sur 606 -- 4,1 % -- soit UN par scan a la mediane, et sur la
# frame la plus chargee il n'en retire AUCUN : 47 des deux cotes, chance d'atteinte
# inchangee. Or c'est la frame chargee qui fait abandonner une carte. Il n'achete donc ni
# separation ni volume la ou le volume compte.
#
# La ligne a 0,30 est ce qui rend les deux precedentes lisibles : un plancher qui MORD se
# voit (42 au lieu de 47, +5,9 points). La mesure n'est pas aveugle, elle est negative.
#
# CE QU'IL POUVAIT COUTER. Le combat reel le moins compact du journal est a 0,198, soit 32 %
# au-dessus de l'ancien plancher -- et le 5e centile de TOUS les candidats est a 0,184. Les
# vrais monstres vivent donc dans la meme region que le plancher, et ce qu'un refus jette ne
# laisse aucune trace. Pur risque, rendement nul.
#
# CE QUI A ETE VERIFIE AVANT D'ABAISSER, parce que le classement est desormais mene par la
# HAUTEUR : une tache etalee et peu remplie pourrait passer devant tout le monde. Sur les
# memes captures, les 25 candidats sous 0,15 mesurent 46 a 134 px de haut, et sur 23 scans
# AUCUN ne serait clique en premier. Le risque redoute ne se materialise pas.
#
# 0,05 N'EST PLUS UN DISCRIMINANT mais un garde-fou contre les masques degeneres -- un
# contour creux, une bordure. Le candidat le moins compact observe est a 0,045, et le plus
# petit combat reel a 0,198 : quatre fois le nouveau plancher.
MIN_EXTENT = 0.05
# ELONGATION MAXIMALE, et elle ne fait presque rien. Mesure sur les 23 paires de
# `data/runs/` : elle refuse 6 taches en tout, soit 0,3 par frame sur 604 candidats. Elle ne
# deplace ni la mediane ni le maximum du compte -- ce qui a d'abord fait conclure, a tort,
# qu'elle ne refusait RIEN (cf. `GroupLimits`).
#
# LA RESSERRER ACHETERAIT QUELQUE CHOSE, et c'est refuse. A 3,17 -- l'elongation du combat
# reel le plus etale -- on gagne deux points d'atteinte (90,1 -> 92,1 %). Mais ce serait
# poser un REFUS exactement sur l'extreme observe de sept combats : le huitieme sera
# peut-etre a 3,5, et un candidat refuse n'est jamais clique, donc jamais etiquete. Deux
# points ne paient pas ce risque-la. La demotion, elle, est deja assuree autrement : le
# classement est mene par la hauteur, et une trainee fine est basse.
MAX_ASPECT = 5.0

# Au-dela de ce nombre de candidats sur UNE frame, la sonde ne mesure plus des monstres.
# Le pendant de `SUSPICIOUS_RESOURCE_COUNT` cote chasse, et il manquait : la recolte
# comptait ses scans invraisemblables depuis longtemps, la chasse cliquait le plus proche
# des siens sans jamais se demander combien il y en avait.
#
# MESURE, sur les 16 paires exploitables de `data/runs/` (les seules captures de carte
# reelles du depot) : 20 a 49 candidats par paire, aire mediane 2129 px, maximum 48142 px,
# pour 12 a 20 % de la surface en mouvement.
#
# ON A LONGTEMPS ECRIT ICI que ces paires, espacees de 12 a 40 s soit cinquante a cent
# fois `MOTION_DELAY`, « majorent largement le mouvement d'une vraie sonde ». C'etait une
# deduction de bon sens, jamais une mesure, et `study_monsters.py --espacement` la refute :
#
#     ecart entre les deux captures        0-30 s   30-60 s   60-120 s
#     candidats, mediane                      29        27         29
#
# PLAT. Ce n'est donc pas la distance parcourue qui produit ces candidats, c'est de
# l'animation permanente -- et rien ne permet d'affirmer qu'un scan a 0,35 s en verrait
# moins. La consolation a disparu avec l'argument.
#
# IL VALAIT 12, ET IL SE DECLENCHAIT SUR 87 % DES SCANS REELS. Un avertissement permanent
# n'est plus lu -- c'est la regle que ce module applique partout ailleurs -- et son compteur
# devenait la copie de `hunt_scans`. Mesure des comptes PLAUSIBLES sur les 23 paires de
# `data/runs/` :
#
#     [4, 7, 11, 14, 15, 16, 16, 18, 18, 20, 20, 21, 22, 23, 23, 23, 24, 26, 28, 31, 33, 35, 43]
#
#     seuil 12   20 scans sur 23 declares   87 %
#     seuil 30    4 scans sur 23            17 %
#     seuil 38    1 scan  sur 23             4 %
#     seuil 43    0 scan  sur 23             0 %
#
# DEUX MODES DE PANNE, et un seul etait garde. Un seuil trop HAUT ne se declenche jamais :
# c'est ce que `tests/test_farming.py` surveillait, en exigeant qu'il reste sous le PLANCHER
# observe. Mais rester sous le plancher garantit qu'il se declenche TOUJOURS, ce qui est
# l'autre panne. Le seuil doit tomber DANS la distribution, pas en dessous.
#
# 43 EST DERIVE, PAS CHOISI. C'est le compte a partir duquel `reach_probability` tombe sous
# 95 % pour `MAX_CONSECUTIVE_FAILURES` essais : au-dela, plus d'une carte sur vingt est
# abandonnee alors qu'elle portait un vrai groupe. Le seuil designe donc une CONSEQUENCE et
# non une invraisemblance -- « une carte de Dofus ne porte pas douze groupes » etait un
# raisonnement sur les monstres appliique a un compte de CANDIDATS, dont la quasi-totalite
# est du decor. Un test recalcule cette derivation et rougit si les deux se separent.
#
# LE CRITERE A 95 % EST PLUS STRICT QUE LE 80 % DU PLANCHER D'AIRE, et c'est voulu : celui-ci
# decide quand AVERTIR, celui-la decide ce qu'on ACCEPTE. Un avertissement doit parler avant
# que la situation ne devienne inacceptable.
#
# « DERIVE, PAS CHOISI » NE VEUT PAS DIRE « SOLIDE », et il faut le lire ici. La derivation
# passe par `reach_probability`, qui AMPLIFIE : la qualite d'ordre ne bouge que de 0,857 a
# 0,964 quand le journal perd une entree, et ce seuil-la va de 31 a 114. Il tient donc a UNE
# observation, et c'est de loin le chiffre le plus fragile du fichier.
#
# Il reste livre a la valeur derivee pour deux raisons. Elle tombe pres du BAS de cette
# bande, donc du cote bavard -- et le defaut qu'on corrige etait le silence par saturation,
# pas l'inverse. Et sur les captures reelles il se declenchait sur 1 scan sur 23 quand il
# valait 38 : un seuil PLUS HAUT ne peut se declencher que moins souvent, donc ce chiffre
# reste un majorant. `study_engagements.py` affiche l'etendue a chaque depouillement.
#
# IL A DEJA BOUGE UNE FOIS, ET POUR LA RAISON EXACTE QUE LE PARAGRAPHE PRECEDENT ANNONCE :
# 38 -> 43 quand la qualite d'ordre est passee de 0,886 a 0,900, apres qu'une entree du
# journal -- un candidat detecte DANS le panneau de controle du bot, pas dans le jeu -- a
# ete marquee invalide. Une seule observation contaminee deplacait le seuil de cinq crans.
# C'est la demonstration en vraie grandeur de l'amplification decrite plus haut, et la
# raison pour laquelle le test qui recalcule la derivation vaut mieux qu'une valeur figee.
#
# IL NE REFUSE RIEN, exactement comme son homologue des ressources : refuser sur un compte
# serait l'erreur deja commise avec la phase de grille. Il compte, et le bilan le dit.
SUSPICIOUS_GROUP_COUNT = 43

# IL Y AVAIT ICI `MIN_PLAUSIBLE_HEIGHT = 35`, hauteur en dessous de laquelle un candidat
# passait en second choix. Elle a disparu, et c'est une SUPPRESSION de seuil, pas un
# deplacement : `best_group` ordonne desormais par la hauteur elle-meme, du plus haut au
# plus bas. Mesure a l'appui la-bas -- le seuil binaire ordonnait a 71,4 %, la hauteur
# continue a 88,6 %, sur les memes 35 paires. 35 etait la plus petite hauteur VUE sur un
# combat reel, donc le treizieme l'aurait probablement invalidee.


@dataclass(frozen=True)
class GroupLimits:
    """Les quatre seuils de forme, REGLABLES sans toucher au paquet.

    Le projet produit deja les chiffres qui les corrigent, par deux instruments :
    `scripts/study_monsters.py` (distribution des aires sur une carte reelle) et le
    journal des engagements, qui etiquette chaque clic par son issue. Il ne manquait que
    la derniere etape -- les appliquer. Constantes de module, elles n'etaient atteignables
    qu'en editant le fichier, ce qui rend la mesure inutilisable par qui ne code pas.

    Les valeurs par defaut restent les constantes ci-dessus, qui sont des ESTIMATIONS
    assumees : aucune capture de carte avec monstres n'existait a leur ecriture.

    VERITE TERRAIN, relevee sur `data/engagements.jsonl` — 12 tentatives, 7 combats et 5
    sans effet. `scripts/study_engagements.py` mesure le recouvrement de chaque critere
    entre les deux classes :

        critere      combats (min/med/max)   sans effet         recouvr.  coupe sans perte
        hauteur         35 /   80 /  125      24 /  33 /   77     42 %    >= 35  -> 3/5
        aire           492 / 3292 / 9342     402 / 563 / 3552     34 %    >= 492 -> 1/5
        compacite     0.20 / 0.40 / 0.76    0.24 / 0.43 / 0.52    50 %    aucune
        elongation    0.38 / 1.00 / 3.17    1.00 / 1.47 / 1.70    25 %    aucune
        largeur         20 /  111 /  157      40 /  44 /  105     47 %    aucune

    LES DEUX DERNIERES COLONNES SE CONTREDISENT, et c'est la lecon principale. Le
    recouvrement designe l'ELONGATION comme le meilleur critere (25 %) ; la coupe sans
    perte dit qu'elle ne peut retirer AUCUN faux positif. Les deux ont raison, et c'est la
    question qui differe.

    Le recouvrement compare deux INTERVALLES. Or les ratés (1.00 a 1.70) sont ici
    entierement CONTENUS dans la plage des combats (0.38 a 3.17) : aucune coupe ne les
    separe, ni par le haut ni par le bas, et le rapport affiche pourtant un beau score
    parce que la plage des combats est large. Le critere le plus inutile passait pour le
    plus prometteur.

    Le tableau precedent, etabli sur neuf tentatives, avait conclu sur cette base que
    « l'elongation separe le mieux », avec une explication qui sonnait juste : un monstre
    se tient DEBOUT, plus haut que large, quand le decor anime -- eau, herbe, feuillage --
    s'etale. Trois combats de plus l'ont demolie. L'un mesure 111 x 35, soit une elongation
    de 3,17 : plus etale que le plus etale des echecs. C'etait un accident d'echantillon
    habille en argument physique.

    CE QUI SEPARE REELLEMENT est la HAUTEUR : garder les candidats d'au moins 35 px retire
    3 des 5 faux positifs sans perdre un seul combat. Le sens reste physique -- ce qui
    bouge sur une carte sans etre un monstre est BAS, une touffe d'herbe, un remous -- mais
    apres ce qui precede, cette explication-la n'est pas non plus tenue pour acquise : voir
    `best_group`, qui ORDONNE au lieu de refuser.

    L'AIRE ne retire qu'un seul raté, et le plus petit combat reel (492 px) reste PLUS
    PETIT que le plus gros echec (3552 px). Le conseil « relever --min-group-area », qui
    figurait dans le message d'arret de la chasse, aurait donc coute de vrais combats sans
    supprimer les faux positifs. Corrige.

    AUCUN SEUIL DE CE DATACLASSE N'EST CHANGE ICI. `accepts` REFUSE, et refuser sur douze
    points est exactement l'erreur que ce tableau vient d'illustrer.

    L'ANGLE MORT, et il est STRUCTUREL. Les douze entrees du journal ont toutes passe
    `accepts` PAR CONSTRUCTION : un candidat refuse n'est jamais clique, donc jamais
    etiquete. Le journal peut valider le CLASSEMENT des candidats retenus -- c'est ce que
    fait le tableau ci-dessus -- mais il ne revelera JAMAIS un faux negatif de ce filtre,
    pas plus avec mille entrees qu'avec douze. Aucune accumulation de donnees ne comblera
    ce trou, et c'est pourtant le SEUL mecanisme de la chasse capable de faire passer une
    carte pour vide.

    Ce qui reste mesurable est la distance entre chaque borne et le combat reel qui s'en
    approche le plus (`study_engagements.py`, section « marges aux bornes de REFUS ») :

        aire            300        combat le plus petit      492      64 % de marge
        elongation      5.00       combat le plus etale      3.17     37 %
        aire maximale   60000      combat le plus gros       9342     84 %
        elongation      0.20       combat le plus dresse     0.38     92 %
        compacite       0.05       combat le moins compact   0.20    296 %

    LES DEUX PREMIERES ETAIENT FROLEES -- 23 % pour l'aire, 32 % pour la compacite -- et ce
    sont celles qui ont bouge depuis, chacune sur un arbitrage chiffre (voir `MIN_AREA` et
    `MIN_EXTENT`). L'elongation haute est desormais la plus mince, a 37 %, et elle ne bouge
    PAS : la resserrer poserait un refus sur l'extreme observe de sept combats.

    UNE BORNE PORTANT QUATRE NOMS. Mesure sur les 23 paires de `data/runs/`, en desactivant
    une borne a la fois :

        borne desactivee    candidats   refuses   par frame
        livre (reference)         604         -           -
        MIN_EXTENT                606         2         0,1
        MAX_ASPECT                610         6         0,3
        MAX_AREA                  622        18         0,8
        MIN_AREA a 100            994       390        17,0

    Trois des quatre refusent une poignee de candidats -- 26 sur 604, soit 4 % -- quand
    `MIN_AREA` en commande quinze fois plus. C'est lui, et lui seul, qui agit sur le NOMBRE
    de candidats ; les trois autres sont des garde-fous pour des cas rares (cf. `MAX_AREA`,
    dont le vrai role est les TRANSITIONS d'ecran).

    CE TABLEAU A DEJA ETE FAUX, et l'erreur vaut d'etre gardee. Il comparait le MAXIMUM du
    compte de candidats par frame, qui vaut 47 avec ou sans ces trois bornes -- d'ou la
    conclusion « elles ne refusent RIEN ». Elles refusent 0,1 a 0,8 candidat par frame :
    trop peu pour deplacer un maximum, pas zero. Mesurer un maximum et conclure sur des
    refus est une faute de statistique, la meme que le sous-echantillonnage corrige dans
    `_pire_frame` -- on ne lit pas une quantite dans un indicateur qui l'ignore.

    L'AMPLEUR DU PLANCHER D'AIRE, honnetement. Descendu a 1 il admet des composantes d'UN
    PIXEL et rend un facteur quinze sur le compte total, ce qui ne decrit aucun reglage
    qu'on envisagerait. La ligne ci-dessus est mesuree a 100, valeur plausible.

    DEUX ROLES QU'ON CONFONDAIT. « Separer les monstres du decor » et « limiter le NOMBRE de
    candidats » ne sont pas la meme chose, et l'aire fait exactement l'un sans faire l'autre.
    Le journal dit qu'elle ne SEPARE pas -- sa meilleure coupe sans perte retire 1 faux
    positif sur 5, et le plus petit combat reel (492 px) est plus petit que le plus gros
    echec (3552 px). Le tableau ci-dessus dit qu'elle est le SEUL controle du volume. Les
    deux sont vrais. Le conseil « relever --min-group-area », retire de quatre endroits
    parce qu'il ne separe rien, reste donc a proscrire pour cette raison-la -- mais on sait
    maintenant que c'est aussi le seul bouton qui agisse sur le nombre de candidats, ce qui
    en fait le plus dangereux a tourner dans les DEUX sens.
    """

    min_area: int = MIN_AREA
    max_area: int = MAX_AREA
    min_extent: float = MIN_EXTENT
    max_aspect: float = MAX_ASPECT

    def accepts(self, area: int, width: int, height: int) -> bool:
        """La tache a-t-elle la forme d'un groupe de monstres ?

        Rassemble les trois criteres en un seul endroit : ils etaient ecrits en ligne
        dans `detect_groups`, et `resources.py` en porte une copie presque identique.
        """
        if not (self.min_area <= area <= self.max_area):
            return False
        if area / max(width * height, 1) < self.min_extent:
            return False
        aspect = width / max(height, 1)
        return not (aspect > self.max_aspect or aspect < 1.0 / self.max_aspect)


DEFAULT_LIMITS = GroupLimits()


# Correlation au-dela de laquelle deux taches de MEME silhouette sont le meme sprite, vu
# avant et apres son deplacement. Mesure sur l'experience de deplacement simule (cf.
# `tests/test_monsters.py::TestWhenTheBlobSplitsInTwo`), sprite et fond reels :
#
#     un sprite scinde en deux lobes     0,689 a 1,000
#     deux blocs de pixels differents    |correlation| <= 0,085
#
# Un ecart de 0,60 entre les deux classes. Le seuil est pose au milieu, et il ne REFUSE
# rien : voir `stale` ci-dessous.
MIN_TRAIL_SCORE = 0.4


@dataclass(frozen=True)
class MonsterGroup:
    """Un candidat de groupe, en coordonnees ecran (cible de clic directe)."""

    x: int
    y: int
    area: int
    width: int
    height: int
    # Cette tache est-elle l'endroit que le sprite vient de QUITTER ?
    #
    # Quand un monstre se deplace de plus que sa propre largeur, sa tache de mouvement se
    # scinde en deux lobes -- depart et arrivee -- de silhouette IDENTIQUE. Le detecteur
    # les rendait tous deux, et le classement tranchait sur leur ecart d'aire, qui n'est
    # que de quelques pour cent : autant dire a pile ou face. Un clic sur le lobe de
    # depart tombe sur du sol vide, ne demarre aucun combat, et fait enregistrer au
    # journal un FAUX NEGATIF sur un vrai monstre.
    #
    # ORDONNE, NE REFUSE PAS. Un lobe marque est clique APRES les autres, jamais retire :
    # si la marque se trompe, elle coute un ordre, pas un combat. C'est la meme regle que
    # pour la hauteur (`best_group`), et pour la meme raison -- les donnees qui la fondent
    # sont trop peu nombreuses pour autoriser un refus.
    stale: bool = False
    # Coin haut-gauche de la BOITE, la ou (x, y) est le point de CLIC recale sur la masse.
    # Les deux different, et c'est voulu (cf. `_anchor`) -- mais toute mesure qui veut
    # relire les pixels du candidat a besoin de la boite, pas du point de clic.
    #
    # Ajoute pour le DECALAGE DE PHASE : dans la boite d'un candidat, un sprite qui se
    # deplace decale sa masse entre les deux captures, une texture qui clignote non. Mesure
    # sur les 604 candidats de `data/runs/` -- deplacement median 0,75 px, p90 1,62 px --
    # contre 3,89 px mesures sur un sprite reel deplace de 4 px. Le decor ne se translate
    # pas ; c'est le premier signal trouve qui distingue physiquement les deux.
    # None quand le candidat n'a pas ete produit par `detect_groups` -- un objet fabrique
    # a la main pour un test, par exemple. Le defaut etait 0, et `box` rendait alors
    # (0, 0, l, h) : une boite PLAUSIBLE et fausse, qui fait mesurer le coin haut-gauche de
    # l'ecran a la place du candidat. Une mesure attribuee aux mauvais pixels est pire
    # qu'une absence -- c'est la regle qui a fait ecarter le journal du dry-run.
    left: int | None = None
    top: int | None = None

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def box(self) -> tuple[int, int, int, int] | None:
        """(gauche, haut, largeur, hauteur), ou None si la boite n'est pas connue."""
        if self.left is None or self.top is None:
            return None
        return self.left, self.top, self.width, self.height


def motion_mask(before: NDArray[np.uint8], after: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Masque des pixels ayant bouge entre deux captures.

    Contrairement a la surbrillance de recolte, on ne cherche PAS un sens : un sprite qui
    se deplace assombrit autant qu'il eclaire. On prend donc la difference absolue.
    """
    if before.shape != after.shape:
        return np.zeros(before.shape[:2], dtype=np.uint8)

    delta = cv2.absdiff(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(before, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(delta, MIN_DIFFERENCE, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _anchor(labels: NDArray, index: int, box: tuple[int, int, int, int],
            cx: float, cy: float) -> tuple[int, int]:
    """Point de CLIC : le pixel de la composante le plus proche de son centroide.

    Le centroide seul ne suffit pas, et la raison est dans `CLOSE_KERNEL` quelques lignes
    plus haut : un sprite qui se deplace laisse DEUX taches, celle de depart et celle
    d'arrivee. Le centroide tombe entre les deux -- c'est-a-dire sur le sol.

    MESURE, sur les 23 paires consecutives de `data/runs/` et les 704 composantes qui
    passent `accepts` :

        centroide SUR la masse en mouvement     579     82 %
        centroide dans le VIDE                  125     18 %

    LA MEME MESURE, APRES CORRECTION, ET ELLE REDIMENSIONNE LE DEFAUT. Les 704 points
    tombent desormais tous sur la masse, mais l'ampleur du deplacement dit ce que ces
    18 % valaient vraiment :

        recalage de plus de  1 px      113 / 704     16 %
        recalage de plus de  5 px       29 / 704      4 %
        recalage de plus de 10 px        6 / 704      1 %
        recalage maximum                17 px

    Le centroide n'etait donc presque jamais entre deux lobes eloignes : il tombait dans
    une petite concavite, a trois pixels du bord. La plupart de ces clics touchaient
    probablement deja le monstre. Ce correctif ne recupere PAS un clic sur cinq, et le
    presenter ainsi serait une exageration -- au mieux quelques pour cent, ceux du bout de
    la distribution.

    CE QU'IL APPORTE VRAIMENT est ailleurs : il supprime une CATEGORIE d'erreur, sans
    seuil, sans capture supplementaire et sans changer le point rendu dans 84 % des cas.
    « Le point de clic est sur la masse en mouvement » devient une propriete du code au
    lieu d'une probabilite, et le jour ou les blobs seront pris a 0,35 s d'intervalle
    plutot qu'a 12-32 s comme ici, c'est cette propriete-la qui restera vraie.

    CE QU'IL NE REPARE PAS. Rien ne garantit que le pixel retenu appartienne au lobe
    d'ARRIVEE : il peut designer l'endroit que le monstre vient de quitter, auquel cas le
    clic fait marcher le personnage au lieu d'engager. Sur ces captures, le cas ne s'est
    pas presente -- 17 px de deplacement maximum, aucun saut d'un lobe a l'autre -- mais
    ces paires sont espacees de dizaines de secondes, donc elles MAJORENT le mouvement :
    l'absence de sauts ici ne prouve pas leur absence ailleurs. Trancher demanderait une
    troisieme capture et de quoi verifier le tri, or le journal n'enregistre qu'un point et
    une issue, pas le masque.
    """
    left, top, width, height = box
    sub = labels[top:top + height, left:left + width] == index
    # Le centroide d'une composante connexe est toujours dans sa boite ; l'arrondi peut
    # le pousser d'un pixel dehors, d'ou le bornage.
    px = min(max(round(float(cx)) - left, 0), width - 1)
    py = min(max(round(float(cy)) - top, 0), height - 1)
    if sub[py, px]:
        return px + left, py + top
    ys, xs = np.nonzero(sub)
    closest = int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))
    return int(xs[closest]) + left, int(ys[closest]) + top


def _mark_trails(
    groups: list[MonsterGroup],
    boxes: list[tuple[int, int, int, int]],
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
) -> list[MonsterGroup]:
    """Marque les lobes de DEPART. -> la meme liste, `stale` renseigne.

    LE RAISONNEMENT. Si un sprite est alle du lobe A au lobe B, alors `apres[B]` contient
    ce que `avant[A]` contenait : c'est le meme sprite, a un autre endroit. La correlation
    croisee des deux imagettes le dit, et le sens du deplacement avec.

    POURQUOI CE N'EST PAS CHER, alors que comparer tous les candidats deux a deux couterait
    104 ms pour trente d'entre eux -- le regime reel mesure sur `data/runs/` -- en croissance
    quadratique, et cela juste avant le clic, dans la fenetre qu'on s'efforce de garder vide.
    Les deux lobes d'un sprite rigide ont la MEME SILHOUETTE : mesure a six deplacements,
    leurs boites font 55x75 des deux cotes, a zero pixel pres. On ne compare donc que les
    paires de meme boite, ce qui laisse 2 paires sur 870 (0,2 %) parmi trente candidats de
    tailles quelconques.

    Ce n'est pas une optimisation opportuniste : l'egalite des boites est une PROPRIETE du
    cas qu'on cherche, pas un raccourci. Deux lobes de tailles differentes ne sont pas le
    meme sprite.

    CE QUE LA SILHOUETTE NE SUFFIT PAS A ECARTER, et c'est le defaut que le degre ci-dessous
    corrige. Une carte de Dofus tire ses monstres d'un petit repertoire d'especes, et son
    decor anime se repete : deux objets DIFFERENTS y ont couramment la meme silhouette et se
    correlent a ~1,0. Le test de correlation les appariait alors comme les deux lobes d'un
    seul sprite, et en reléguait un -- un vrai groupe clique apres le decor.

    MESURE, sur les 23 paires consecutives de `data/runs/` (517 candidats acceptes, 18
    paires de meme boite, 15 appariements au-dela du seuil). En regroupant les candidats
    relies par un appariement :

        groupes de 2 candidats      3
        groupes de 3 candidats      1
        groupes de 5 candidats      1

    UN SPRITE NE LAISSE QU'UN SEUL LOBE DE DEPART. Un ensemble de trois candidats qui
    s'apparient mutuellement n'est donc PAS une trace, quelle que soit la correlation : c'est
    du contenu repete.

    D'ou la regle, qui ne demande AUCUN SEUIL : un appariement ne compte que si ses deux
    extremites n'apparaissent nulle part ailleurs -- autrement dit si la composante du graphe
    d'appariement est exactement une arete. C'est la traduction litterale de « depart et
    arrivee, et rien d'autre ». Sur les memes 517 candidats :

        marquages avant la regle     9
        marquages apres              3      les seuls groupes de taille 2

    Les deux tiers des relegations etaient donc infondees. Le nombre de SCANS portant une
    marque, lui, ne bouge pas (2 sur 23) : `FarmReport.trail_scans` lit la meme frequence
    qu'avant sur ces captures, ce qui est attendu -- la regle retire des marques dans des
    scans qui en gardent d'autres, elle n'en vide aucun.

    CE QU'ELLE COUTE, et il faut l'ecrire. Deux monstres de la MEME espece qui se deplacent
    tous les deux produisent quatre lobes indistinguables : une composante de taille 4, donc
    aucun marquage, alors que deux d'entre eux sont bien des traces. On perd la correction
    dans ce cas. C'est le bon sens du compromis -- ne pas marquer coute un ordre de clic, et
    l'ordre reste celui d'avant ; marquer a tort relegue un vrai groupe.

    CE QUI N'EST PAS FILTRE, faute de mesure. L'ECART entre les deux lobes apparies vaut, sur
    ces memes captures, 47 a 332 px (mediane 156), soit 1,15 a 8,11 fois la taille de leur
    boite. Un ecart de huit fois la silhouette en 0,35 s supposerait une vitesse qu'aucun
    monstre n'a -- une borne haute ecarterait donc encore du faux. Mais la poser demande la
    vitesse des monstres, que personne n'a mesuree, et ces captures-ci sont espacees de 12 a
    40 s : elles ne peuvent pas la donner. C'est le meme mur que `MOTION_DELAY`.
    """
    if len(groups) < 2:
        return groups
    # (i, j, indice du lobe de DEPART). On collecte avant de marquer : la decision depend
    # du nombre d'appariements que chaque candidat a recus, donc de la liste entiere.
    arcs: list[tuple[int, int, int]] = []
    for i, (gi, bi) in enumerate(zip(groups, boxes, strict=True)):
        for j, (gj, bj) in enumerate(zip(groups, boxes, strict=True)):
            if j <= i or (gi.width, gi.height) != (gj.width, gj.height):
                continue
            avant_i, apres_j = _crop(before, bi), _crop(after, bj)
            avant_j, apres_i = _crop(before, bj), _crop(after, bi)
            vers_j = _match(apres_j, avant_i)      # i -> j : i est le depart
            vers_i = _match(apres_i, avant_j)      # j -> i : j est le depart
            if max(vers_i, vers_j) < MIN_TRAIL_SCORE:
                continue
            arcs.append((i, j, i if vers_j >= vers_i else j))

    # Degre 1 des deux cotes <=> la composante est cette seule arete. Un candidat apparie a
    # deux autres appartient a du contenu repete, pas a une trace.
    degre = Counter(k for i, j, _ in arcs for k in (i, j))
    marques = {depart for i, j, depart in arcs if degre[i] == 1 and degre[j] == 1}
    if not marques:
        return groups
    return [replace(g, stale=True) if k in marques else g
            for k, g in enumerate(groups)]


def _crop(frame: NDArray[np.uint8], box: tuple[int, int, int, int]) -> NDArray[np.uint8]:
    left, top, width, height = box
    return frame[top:top + height, left:left + width]


def _match(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    """Correlation normalisee de deux imagettes de meme taille. -1 si incomparables."""
    if a.shape != b.shape or a.size == 0:
        return -1.0
    return float(cv2.matchTemplate(a.astype(np.float32), b.astype(np.float32),
                                   cv2.TM_CCOEFF_NORMED)[0, 0])


def detect_groups(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    exclude: tuple[tuple[int, int, int, int], ...] = (),
    limits: GroupLimits = DEFAULT_LIMITS,
    rejects: list[int] | None = None,
) -> list[MonsterGroup]:
    """Candidats de groupes de monstres, tries du plus gros au plus petit.

    `exclude` : rectangles d'interface a ignorer. L'ATH s'anime (chat qui defile,
    compteurs, icones) et produirait des candidats a chaque capture -- le meme piege que
    les icones de sorts prises pour des marqueurs ennemis.

    `rejects` : liste FOURNIE PAR L'APPELANT, ou sont ajoutees les aires des taches que
    `limits.accepts` a REFUSEES. Elle repond a la seule question que le journal des
    engagements ne pourra jamais poser (cf. `GroupLimits`) : ce filtre est le seul
    mecanisme de la chasse capable de faire passer une carte pour vide, et ce qu'il jette
    ne laisse aucune trace -- un candidat refuse n'est pas clique, donc pas etiquete.

    Un parametre de sortie plutot qu'un second retour, et plutot qu'une fonction jumelle :
    les deux traverseraient le meme masque, et deux traversees d'une meme image finissent
    par diverger. Ici il n'y a qu'un parcours, et l'appelant qui ne demande rien ne paie
    rien.

    Les taches ecartees par `exclude` n'y figurent PAS : celles-la ne sont pas refusees
    pour leur forme, elles sont dans l'interface. Les melanger ferait accuser le filtre
    d'un rejet qui n'est pas le sien.
    """
    mask = motion_mask(before, after)
    # Meme correctif que pour les ressources : les panneaux ouverts s'animent en
    # permanence et ne sont pas dans `UI_ZONES`. Cf. `ui_mask`.
    #
    # NE PAS DILATER CE MASQUE, et la raison est contre-intuitive. Le masque laisse un
    # lisere de candidats le long de ses propres bords -- sur les 93 scans rejoues, dix
    # d'entre eux tombent a x = 405-407 alors que le panneau de chat s'arrete a 402. Elargir
    # le masque de quelques pixels parait donc la correction evidente. Mesure :
    #
    #     dilatation      0 px    2 px    4 px    6 px   10 px
    #     candidats       2757    2823    2850    2837    2732
    #
    # Elle en produit DAVANTAGE. Mettre a zero des pixels supplementaires SCINDE les taches
    # de mouvement en plusieurs composantes connexes, et chaque morceau devient un candidat.
    # Le remede fabrique donc exactement ce qu'il pretend retirer, et il faut aller jusqu'a
    # 10 px -- soit bien au-dela du lisere -- pour repasser sous la valeur de depart.
    if mask.shape == after.shape[:2]:      # tailles discordantes : masque neutre
        mask[ui_mask(after) > 0] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    groups: list[MonsterGroup] = []
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        left, top, width, height, area = (int(stats[index, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA))
        if not limits.accepts(area, width, height):
            if rejects is not None:
                rejects.append(area)
            continue

        # Recale sur la masse AVANT d'exclure : la question posee a `exclude` est « ce
        # candidat est-il dans l'ATH ? », et elle porte sur le point qu'on va cliquer.
        # La juger sur le centroide laisserait passer un candidat dont le clic tombe
        # dans l'interface -- ou refuserait l'inverse.
        x, y = _anchor(labels, index, (left, top, width, height), *centroids[index])
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in exclude):
            continue
        groups.append(MonsterGroup(x=x, y=y, area=area, width=width, height=height,
                                   left=left, top=top))
        boxes.append((left, top, width, height))

    return sorted(_mark_trails(groups, boxes, before, after), key=lambda g: -g.area)


def best_group(
    groups: list[MonsterGroup],
    origin: tuple[int, int],
) -> MonsterGroup | None:
    """Le candidat a cliquer EN PREMIER. N'en ecarte aucun : c'est un ORDRE, pas un filtre.

    Ce que coute un mauvais choix, mesure sur les constantes du runner :

        clic sur un vrai groupe        le combat demarre, la carte est faite
        clic sur un feuillage          ENGAGE_TIMEOUT, soit 8 s perdues
        huit clics rates d'affilee     la carte est declaree INENGAGEABLE et abandonnee

    Une carte peut donc etre perdue en ayant un vrai groupe dessus, simplement parce que
    les huit essais sont partis sur du decor. L'ordre des clics decide du resultat.

    COMMENT UN ORDRE SE MESURE, et c'est ce qui manquait. `study_engagements.py` jugeait
    chaque critere en FILTRE -- recouvrement des plages, coupe sans perte -- alors que cette
    fonction ne filtre rien. La bonne question est : sur toutes les paires (un combat, un
    rate), dans quelle proportion le critere met-il le combat EN PREMIER ? 50 % vaut le
    hasard. Sur les 12 tentatives etiquetees du journal, soit 7 x 5 = 35 paires :

        hauteur, du plus haut au plus bas            88,6 %
        aire, de la plus grande a la plus petite     77,1 %
        elongation, de la plus dressee               75,7 %
        largeur                                      70,0 %
        compacite                                    40,0 %
        DISTANCE A L'ORIGINE, du plus proche         45,7 %

    LA DISTANCE ORDONNE MOINS BIEN QUE LE HASARD. C'etait le second terme de la cle de tri,
    et le tri complet en payait le prix :

        cle precedente   (hauteur < 35, distance)    71,4 %
        cle actuelle     (-hauteur, distance)        88,6 %
        seuil de hauteur seul                        80,0 %

    Le seuil binaire creait des EX AEQUO -- tous les candidats d'au moins 35 px -- que la
    distance tranchait ensuite, et elle les tranchait mal. Le tri complet ordonnait donc
    moins bien que son premier critere pris seul, ce qu'aucune mesure ne pouvait montrer
    tant qu'on jugeait les criteres un par un et en filtre.

    DEUX CHOSES DISPARAISSENT ICI. Le seuil de 35 px, qui etait la plus petite hauteur
    VUE -- du sur-apprentissage dans sa forme la plus pure, et le docstring precedent le
    disait deja. Et le classement par proximite, herite de `nearest_group`. Son argument
    reste vrai sur le papier (un groupe lointain oblige a traverser la carte, donc a
    s'exposer a une agression en chemin) mais il ne se verifie pas ici, et il portait de
    toute facon sur une origine qui n'est PAS le personnage : `character_position` vaut le
    centre de l'ecran, faute de savoir localiser le personnage hors combat (cf.
    `FarmingSession`). La distance reste en dernier rang, ou elle ne departage plus que des
    hauteurs strictement egales.

    CE QUE CES CHIFFRES NE DISENT PAS. 35 paires issues de 12 clics, sur une poignee de
    cartes : l'ecart entre 45,7 % et 50 % n'est pas significatif, et « la distance nuit »
    serait surinterpreter. Ce qui est solide est plus modeste et suffit : la distance
    n'apporte RIEN de mesurable, la hauteur si, et faire passer la premiere devant la
    seconde -- ce que faisait le seuil binaire -- coute 17 points d'ordre.

    L'ENVELOPPE PAR LE HAUT, ET POURQUOI ELLE MANQUAIT. Ordonner par la hauteur SANS BORNE
    SUPERIEURE suppose que « plus haut » veuille toujours dire « plus probablement un
    monstre ». C'est vrai dans la plage du journal (24 a 125 px) et faux au-dela : une nappe
    de mouvement etalee -- de l'eau qui defile, plusieurs sprites dont les taches ont
    fusionne -- est plus haute que n'importe quel monstre. Mesure sur les 23 paires de
    `data/runs/`, avec la seule cle (stale, -hauteur, distance) :

        premier candidat plus HAUT que tout combat reel     23 / 23
        premier candidat plus GROS que tout combat reel     21 / 23

    Cent pour cent des premiers clics partaient sur des taches de 167x310, 170x444, 609x358.
    LE JOURNAL NE POUVAIT PAS LE MONTRER : ses douze entrees ont toutes ete cliquees sous
    l'ancien classement -- le plus PROCHE -- donc la population sur laquelle 88,6 % a ete
    mesure ne contient aucune de ces nappes. La mesure qui a justifie le changement portait
    sur un echantillon que le changement rend caduc.

    LE PLAFOND N'A PAS DE FACTEUR DE SECURITE, et ce n'est pas un oubli : il n'en existe
    pas. La population des nappes commence juste au-dessus de l'enveloppe reelle, sans
    intervalle vide ou poser une marge --

        plafond de relegation      premier clic hors enveloppe
        aucun                             21 / 23
        5,0 x le plus gros combat         20 / 23
        2,0 x                             20 / 23
        1,5 x                             18 / 23
        1,0 x (livre)                      0 / 23

    -- si bien que tout facteur superieur a 1 laisse le defaut entier. On s'en tient donc au
    constat : ce qui est plus gros que TOUT combat jamais enregistre passe en dernier. Le
    cliquet se desserre de lui-meme au prochain combat plus gros.

    LA HAUTEUR AUSSI, et il a fallu la mesurer separement. Le plafond d'aire seul ramenait
    l'aire du premier clic dans l'enveloppe mais pas sa hauteur -- une tache etroite et tres
    haute (70 x 251, 51 x 156) passe sous 9342 px tout en etant deux fois plus haute que
    tout combat reel, et le tri par hauteur la met alors en tete :

        cle                    hauteur du 1er clic (min/med/max)   hors enveloppe
        plafond d'aire seul              104 / 154 / 251              20 / 23
        enveloppe complete                81 / 116 / 125               0 / 23

    Vingt scans sur vingt-trois changent de premier clic. Ce qui est choisi a la place tient
    dans la plage des combats reels (49 a 185 de large, 81 a 125 de haut) la ou le choix
    precedent la depassait presque toujours.

    A LIRE AVEC SA RESERVE. Ces paires sont espacees de 12 a 40 s, soit cinquante a cent
    fois `MOTION_DELAY` : elles MAJORENT enormement les fusions de taches. A 0,35 s la
    relegation devrait etre rare, et c'est le regime souhaite -- un garde-fou qui ne se
    declenche pas est un garde-fou qui n'avait rien a corriger.

    RIEN N'EST REFUSE, et c'est ce qui rend le changement sur : un candidat relegue est
    clique APRES les autres, pas jamais. Le runner memorise les positions ratees par carte
    (`_spots_here`) et rescanne au cycle suivant, si bien que la liste entiere finit par
    etre parcourue. Se tromper d'ordre coute quelques secondes ; se tromper de filtre coute
    le combat. Avec douze points, seul le premier prix est payable.
    """
    ordre = click_order(groups, origin)
    return ordre[0] if ordre else None


def phase_shift(before: NDArray[np.uint8], after: NDArray[np.uint8],
                box: tuple[int, int, int, int] | None) -> float | None:
    """De combien la masse du candidat s'est-elle DEPLACEE ? -> pixels, ou None.

    LE SIGNAL QUE CE MODULE CHERCHE DEPUIS SON EN-TETE. « Hors combat, rien ne distingue un
    groupe de monstres du decor par la couleur ou la forme » -- et la piste physique
    evidente, « le decor s'anime SUR PLACE, un monstre se deplace », avait ete essayee puis
    refutee. Elle l'avait ete sous la forme d'une RECURRENCE entre deux scans successifs,
    qui ne mesure pas le deplacement mais la persistance d'un emplacement.

    Ici on mesure le deplacement LA OU IL A LIEU : dans la boite du candidat, entre les deux
    captures d'UNE MEME paire. Mesure :

        sprite reel deplace de  4 px    decalage 3,89   confiance 0,90
        sprite reel deplace de 20 px    decalage 20,29  confiance 0,64
        604 candidats de data/runs      decalage median 0,75, p90 1,62

    Le decor ne se translate pas, meme sur des paires espacees de 12 a 40 s.

    CE QUE LA MESURE ISOLE, et c'est le chiffre qui compte pour s'en servir un jour :

        decalage >=  2 px    41 candidats sur 604    6,8 %
        decalage >=  5 px    11 candidats sur 604    1,8 %
        decalage >= 10 px     7 candidats sur 604    1,2 %   (maximum observe 82,7 px)

    Une petite minorite se translate donc VRAIMENT. C'est la forme qu'on attend d'un
    discriminant, et elle tranche avec la hauteur : celle-ci ordonne bien (88,6 %) mais
    n'isole rien, tous les candidats en ayant une.

    LA MESURE PEUT VOIR CE QU'ELLE PRETEND VOIR. Une translation reste UNE tache tant
    qu'elle ne depasse pas la largeur de la boite, et 60 % des boites font au moins 60 px
    de large : les deplacements jusqu'a ~60 px y sont donc observables. Le p90 a 1,62 px
    n'est pas un aveuglement de l'instrument, c'est un constat sur le decor.

    CE QU'IL NE MESURE PAS, et il faut le savoir avant de s'en servir. Quand le sprite s'est
    deplace de plus que sa largeur, la tache se scinde et chaque lobe ne contient le sprite
    que dans UNE des deux frames : il n'y a alors rien a aligner, et le chiffre rendu n'a
    pas de sens (mesure : 41 px et 12 px pour un meme deplacement de 64 px, confiances 0,13
    et 0,22). Ce cas-la est deja traite ailleurs, par `stale`.

    None quand la boite est trop petite pour une fenetre de Hann.
    """
    if box is None:
        return None
    left, top, width, height = box
    if width < 8 or height < 8:
        return None
    if before.shape != after.shape:
        return None
    # UNE SEULE garde pour le debordement, et elle vient AVANT la conversion. Une boite qui
    # sort du cadre donne une decoupe tronquee par numpy (mesure sur un demi-candidat, sans
    # que rien ne le signale) ou vide -- et cv2.cvtColor LEVE sur une image vide, ce qui
    # ferait tomber la boucle de chasse. Comparer la forme obtenue a la forme demandee
    # couvre les deux cas d'un coup ; le faire avant cvtColor est ce qui evite l'exception.
    # Tester en plus les bornes en amont serait un second chemin vers la meme decision.
    # Une seule des deux vues est verifiee : les cadres ont deja la meme forme (garde
    # ci-dessus), donc la meme decoupe y produit la meme taille. Verifier la seconde
    # serait une condition qui ne peut pas echouer -- donc un test qu'aucune
    # contre-epreuve ne pourrait tuer, ce qui la rend indistinguable d'un oubli.
    vue_avant = before[top:top + height, left:left + width]
    vue_apres = after[top:top + height, left:left + width]
    if vue_avant.shape[:2] != (height, width):
        return None
    a = cv2.cvtColor(vue_avant, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(vue_apres, cv2.COLOR_BGR2GRAY).astype(np.float64)
    fenetre = cv2.createHanningWindow((width, height), cv2.CV_64F)
    (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), fenetre)
    return float(np.hypot(dx, dy))


def click_order(groups: list[MonsterGroup],
                origin: tuple[int, int]) -> list[MonsterGroup]:
    """L'ordre COMPLET des clics. `best_group` en est le premier element.

    UNE SEULE DEFINITION DE L'ORDRE, et c'est la raison d'etre de cette fonction. La cle
    vivait dans `best_group`, qui ne rend que le vainqueur ; tout ce qui avait besoin du
    RANG la recopiait -- les tests par une boucle « prends le meilleur, retire-le », et le
    runner ne pouvait pas du tout. Deux ecritures d'un meme ordre finissent par diverger, et
    celle des tests aurait alors garde sa propre copie.

    Le rang sert a autre chose qu'a trier : c'est la seule mesure qui puisse CONFRONTER
    `reach_probability` au reel. Ce modele pilote desormais `SUSPICIOUS_GROUP_COUNT`, la
    ligne des cartes abandonnees et le plafond d'essais, et il n'a jamais ete verifie sur
    une session -- il repose sur une hypothese d'independance qu'aucune donnee ne soutient.
    Enregistrer le rang du candidat clique, c'est enregistrer de quoi le refuter.
    """
    ox, oy = origin
    # `stale` EN PREMIER, avant la hauteur. Un lobe de depart n'est pas un mauvais
    # candidat : c'est un endroit dont on sait que le monstre vient de PARTIR. Le cliquer
    # ne peut rien donner, alors qu'un candidat bas peut encore etre un vrai groupe.
    #
    # CE QUE CHAQUE CLE PESE VRAIMENT, mesure en rejouant les 40 scans de `data/runs/` :
    #
    #     stale             marque 7 candidats sur 1659 (0,42 %)
    #     beyond_recorded   relegue 324 sur 1659 (19,5 %)
    #     hauteur           departage presque toujours -- paquets d'ex aequo de 1 a 3
    #     distance          tranche 40 des 320 choix de la fenetre d'essais (12,5 %),
    #                       et le PREMIER clic une seule fois sur 40 scans
    #
    # POURQUOI PAS LE PLUS PROCHE D'ABORD ? C'est la premiere chose qu'on propose en
    # lisant cet ordre, et le modele de cout y repond sans arbitrage.
    #
    # Un ECHEC coute `ENGAGE_TIMEOUT` en entier -- huit secondes -- parce que rien ne
    # permet de savoir plus tot que le clic n'a rien engage. Un SUCCES ne coute que la
    # marche. Viser au plus pres n'economise donc QUE de la marche : de la mediane
    # observee (13 cases) au plus pres plausible (5), on gagne 2,4 a 4,0 s selon la vitesse
    # -- soit un tiers a une moitie d'echec.
    #
    # Ce que couterait l'echange, avec le modele de `reach_probability` et 25 candidats :
    #
    #     ordre par HAUTEUR (q = 0,886)          2,7 echecs attendus
    #     meme ordre, borne basse (q = 0,657)    8,2
    #     ordre SANS valeur predictive (q = 0,5) 12,0
    #
    # La distance n'a aucune valeur predictive mesuree : la mettre en tete revient donc au
    # dernier cas. Neuf echecs de plus, soit 74 s, pour economiser 3,2 s de marche --
    # RAPPORT DE 23 CONTRE UN. Meme en supposant la qualite de la hauteur a sa borne basse,
    # l'echange reste perdant d'un facteur dix.
    #
    # LA DISTANCE EST DONC UN DEPARTAGE, PAS UN CRITERE, et c'est ce qui a fait ecarter une
    # version isometrique de ce terme : Dofus est en vue isometrique, donc une meme distance
    # a l'ecran ne coute pas le meme trajet selon la direction, mais corriger cela ne peut
    # deplacer que ces 12,5 % de choix, a l'interieur de paquets de trois au plus, et rien
    # ne mesure que le trajet isometrique predise mieux l'issue. Le gain plafonne sous le
    # bruit du reste.
    return sorted(groups, key=lambda g: (g.stale, beyond_recorded(g), -g.height,
                                         (g.x - ox) ** 2 + (g.y - oy) ** 2))


# SA FRAGILITE. Le retrait d'une entree (`study_engagements.py`, section « stabilite »)
# donnait 0,857 a 0,964, d'ou un « solide a +/- 0,05 » qui etait FAUX : enlever un point sur
# douze ne peut pas deplacer la mesure de plus d'un douzieme, donc cette methode mesure sa
# propre granularite, pas l'incertitude. Un bootstrap sur les ENGAGEMENTS (20 000 tirages,
# 7 combats / 5 rates reechantillonnes separement pour respecter la dependance des paires)
# donne l'intervalle reel :
#
#     hauteur   0,886  IC95 [0,657, 1,000]   P(<= hasard) = 0,004
#     aire      0,771  IC95 [0,429, 1,000]   P(<= hasard) = 0,052
#     largeur   0,700  IC95 [0,371, 1,000]   P(<= hasard) = 0,118
#
# CE QUE CA ETABLIT ET CE QUE CA N'ETABLIT PAS. La hauteur bat le hasard (4 chances sur
# mille de se tromper) : la cle primaire du tri est justifiee. Mais l'ecart hauteur - aire
# vaut 0,114 avec un IC95 de [-0,114, 0,343] et P(aire >= hauteur) = 0,21 -- ces douze
# points NE PERMETTENT PAS de dire que la hauteur bat l'aire. Si elle est gardee, c'est sur
# un argument de mecanisme et non de chiffre : un aplat de decor etale gonfle l'aire sans
# gagner en hauteur, ce que les 23 premiers clics sur du tres large avaient deja montre.
#
# Qualite d'ORDRE de `best_group`, mesuree sur les 35 paires (combat, rate) du journal :
# part de celles ou le vrai groupe est clique AVANT le faux. Etablie dans le docstring
# ci-dessus, et ici sous forme de constante parce qu'elle sert a autre chose qu'a se lire
# -- voir `reach_probability`. Une seule copie : `scripts/study_engagements.py` la
# recalcule depuis le journal, et si les deux divergent c'est le journal qui a raison.
#
# RELEVEE LE 18/08/2026 : 40 paires au lieu de 35, et 0,900 au lieu de 0,886.
#
# Le journal est passe de 12 a 14 engagements ce soir-la. Pris tels quels, les deux
# nouveaux faisaient TOMBER la mesure a 0,8125 et sa borne basse jusqu'au hasard -- une
# degradation qui aurait condamne la cle de tri. En regardant la ligne fautive : un RATE de
# hauteur 110 a (x=156, y=170), c'est-a-dire DANS le panneau de controle du bot, qui
# recouvrait cette partie de l'ecran (capture `20260818-212307-combat-vide-recolte.png`).
# Le detecteur avait pris un element de l'interface du bot pour un groupe de monstres.
#
# Cette tentative ne dit rien du TRI : elle dit que la capture contenait une fenetre qui
# n'est pas le jeu. Elle est donc marquee `invalid` dans le journal -- annotee, pas
# effacee, avec sa raison -- et `Engagement.decided` l'ecarte. Voir `farming/journal.py`
# pour la regle, qui est etroite : un rate ORDINAIRE reste un rate.
#
# L'autre nouvelle entree, elle, est un vrai combat de hauteur 125, bien classe : c'est
# elle qui fait monter la mesure. La cause de la pollution est corrigee en amont
# (`capture/focus.py` : les fenetres du bot ne peuvent plus passer pour le jeu).
MEASURED_ORDER_QUALITY = 0.900

# BORNE BASSE DE L'INTERVALLE CI-DESSUS (2,5e centile du bootstrap). Elle existe parce que
# `reach_probability` AMPLIFIE l'incertitude au lieu de la diluer : la meme incertitude de
# +/- 0,23 sur la qualite donne, pour 8 essais,
#
#      5 candidats   100 % contre  99 %   -> la conclusion tient
#     12 candidats    97 % contre  45 %   -> elle vacille
#     27 candidats    66 % contre   1 %   -> il ne reste RIEN a conclure
#
# Publier la seule estimation ponctuelle sur une carte encombree, c'est donc annoncer 66 %
# la ou l'honnete reponse est « entre 1 % et 66 % ». Le diagnostic des cartes abandonnees
# encadre.
#
# Relevee avec la mesure ci-dessus : 0,675 sur 40 paires, contre 0,657 sur 35.
#
# CE QUE LA LIGNE INVALIDEE AURAIT COUTE. En comptant le faux candidat detecte sur le
# panneau du bot, cette borne tombait a 0,52 -- soit le hasard. Le depot aurait alors
# affiche, en toute rigueur, que l'ordre de `best_group` ne bat plus le tirage a pile ou
# face, et le plafond d'essais aurait ete releve pour compenser une degradation qui
# n'existait pas. Une seule tentative sur quatorze, et toute la justification basculait :
# c'est la fragilite d'un echantillon de cette taille, et la raison d'etre de cette borne.
ORDER_QUALITY_LOW = 0.675

# PART DES SCANS PORTANT AU MOINS UNE TRACE DE DEPART, sur les 40 paires de meme carte de
# `data/runs/` : 6 / 40, IC95 [7 %, 29 %]. Le bilan de session compare son propre taux a
# celui-la ; sans reference, « une proportion elevee » n'etait pas une consigne.
#
# NE PAS CONFONDRE AVEC LE TAUX PAR CANDIDAT, qui vaut 7 / 1659 = 0,42 %. Trente-cinq fois
# plus petit, parce qu'un scan porte 38 candidats en mediane. Le bilan compte des scans.
#
# C'EST UN PLAFOND. Ces paires sont espacees de 10 a 400 s quand `MOTION_DELAY` vaut
# 0,35 s. Le taux ne montre aucune dependance detectable a l'intervalle sur cette plage
# (test de permutation, p = 0,73 : la decroissance apparente 0,64 % -> 0 % est du bruit sur
# sept marquages), et le deplacement mesure est PLAT -- mediane 0,78 px de 10 s a 400 s.
# Autrement dit la population detectee est faite de choses qui ne se deplacent pas, a
# aucune echelle de temps ; a 0,35 s ce sera encore plus vrai. Depasser ce taux est donc un
# signal, rester en dessous n'en est pas un.
MEASURED_TRAIL_SCAN_RATE = 0.15

# ---------------------------------------------------------------------------------------
# DEUX PISTES SEDUISANTES QUE LEUR TEMOIN A TUEES. Ecrites ici parce qu'elles reviendront :
# elles sont plausibles, elles ont l'air gratuites, et rien dans le code ne dirait qu'elles
# ont deja ete essayees.
#
# 1. UNE MEMOIRE DE DECOR PAR CARTE. « Le decor s'anime au meme endroit a chaque passage,
#    un monstre non » : il suffirait de retenir les positions qui reproduisent un candidat
#    d'un scan a l'autre. Mesure sur des scans DISJOINTS (aucune frame partagee, sinon la
#    recurrence est une tautologie) :
#
#        meme carte          44 %   des candidats reapparaissent a moins de 30 px
#        cartes DIFFERENTES  30 %   <- le temoin
#
#    L'ecart reel n'est donc que de 14 points (p < 1e-4, mais 14 points). Les candidats se
#    concentrent dans les memes regions de l'ecran QUELLE QUE SOIT la carte, et sans ce
#    temoin j'aurais lu 44 % comme « la moitie des candidats sont du decor persistant » et
#    bati une suppression qui aurait jete de vrais groupes a presque le meme taux. Un
#    pouvoir separateur de 14 points ne soutient pas un mecanisme quand la hauteur, elle,
#    en offre 88,6.
#
# 2. DES ZONES D'IU DEDUITES DES CAPTURES. Chercher les cellules qui produisent un candidat
#    sur TOUTES les cartes -- ce serait de l'interface, pas du decor. Observe : 26 cellules
#    de 40 px communes aux 3 cartes. Attendu par hasard, a tailles d'ensembles egales :
#    18,5 [12, 25]. L'exces tient en sept cellules, p = 0,02 parmi plusieurs tests de cette
#    session. TROIS CARTES NE SUFFISENT PAS : le critere « vu partout » est presque atteint
#    par coincidence. `UI_ZONES` reste donc inchangee, et cette fois pour une raison
#    chiffree et non par prudence.
# ---------------------------------------------------------------------------------------

# ENVELOPPE DES COMBATS ENREGISTRES : aire et hauteur du plus gros et du plus haut combat
# de `data/engagements.jsonl` (le meme, 157 x 125). Au-dela, un candidat passe en dernier
# choix -- voir `best_group` et `beyond_recorded`.
#
# LEUR FRAGILITE, par retrait d'une entree : l'aire va de 4957 a 9342, la hauteur de 112 a
# 125. Retirer le seul plus gros combat DIVISE PAR DEUX le plafond d'aire. C'est la borne la
# plus fragile de ce fichier -- mais dans la direction sure : un plafond trop LARGE ne
# relegue pas assez, il n'egare aucun monstre.
#
# UN CLIQUET, PAS UN SEUIL, et c'est ce qui le rend acceptable : il ne fait que constater
# l'enveloppe deja observee, il se DESSERRE tout seul des qu'un combat plus grand est
# enregistre, et il ne se resserre jamais. Le miroir exact de
# `TestNoBoundMayBeTightenedPastGroundTruth`, applique par le haut.
#
# CES DEUX CHIFFRES SONT DANS LE BON REGIME, et c'est ce qui les distingue de tout ce qui
# se mesure sur `data/runs/`. Le journal est ecrit par le runner, donc ses taches sont
# prises a `MOTION_DELAY` -- 0,35 s. Les paires de `data/runs/` sont espacees de 12 a 40 s
# et gonflent enormement les taches ; elles servent a MONTRER le defaut, jamais a fixer la
# borne.
#
# LE RISQUE, ET IL FAUT L'ECRIRE. Un cliquet qui se desserre sur ses propres observations
# peut s'auto-bloquer : un groupe reel plus grand que tous les precedents est relegue, donc
# clique tard, donc peut-etre jamais si la carte est abandonnee au bout de
# MAX_CONSECUTIVE_FAILURES essais -- et son combat n'est alors jamais enregistre, si bien
# que la borne ne bouge pas.
#
# CE BLOCAGE EST PARTIEL, ET LA MESURE LE PRECISE. Simulation sur 320 essais, quatre
# candidats hors enveloppe a chaque scan, avec la liste noire plafonnee :
#
#     20 plausibles par scan     0 % des essais tombent sur un relegue
#     12 plausibles              0 %
#      8 plausibles              5 %
#      5 plausibles             40 %
#
# Le cliquet reste donc gele sur les cartes ENCOMBREES et se desserre tout seul sur les
# cartes CALMES -- exactement celles ou essayer un relegue coute le moins, puisqu'il y a
# peu d'autre chose a cliquer, et celles ou un gros groupe reel a le plus de chances d'etre
# ce qui est la. Ce n'est pas une garantie, c'est mieux qu'un blocage total et il fallait
# le mesurer pour le savoir : la premiere redaction disait « peut-etre jamais ».
LARGEST_RECORDED_FIGHT = 9342
TALLEST_RECORDED_FIGHT = 125


def beyond_recorded(group: MonsterGroup) -> bool:
    """Ce candidat sort-il de l'enveloppe de tout combat deja enregistre ?

    LES DEUX BORNES TRAVAILLENT, et il fallait le verifier : elles viennent du MEME combat
    (157 x 125), donc rien ne garantissait que le « ou » ne soit pas a moitie mort. Sur les
    1659 candidats de `data/runs/`, 324 relegues (19,5 %) dont 208 par les deux bornes,
    11 par l'AIRE seule et 105 par la HAUTEUR seule. Aucune n'est inerte -- un conjoint qui
    ne se declenche jamais est indistinguable d'un oubli, et il y en avait un ici la veille.

    UNE seule notion et un seul endroit : l'aire et la hauteur repondent a la meme question
    -- « a-t-on deja vu un vrai groupe aussi grand ? » -- et les separer en deux termes de
    tri ferait deux cliquets a desserrer, donc deux occasions de n'en desserrer qu'un.
    """
    return (group.area > LARGEST_RECORDED_FIGHT
            or group.height > TALLEST_RECORDED_FIGHT)


def reach_probability(candidates: int, attempts: int,
                      quality: float = MEASURED_ORDER_QUALITY) -> float:
    """P(un vrai groupe present parmi `candidates` soit clique en `attempts` essais).

    CE QUE CA REPOND. `MAX_CONSECUTIVE_FAILURES` vaut 8 depuis toujours, sans un mot de
    justification : c'est un nombre d'essais choisi de tete. Or les deux chiffres qui le
    determinent sont desormais mesures -- combien de candidats une frame rend (20 a 49 sur
    les captures reelles du depot) et a quel point l'ordre est bon (`best_group`). Il n'y
    avait plus qu'a poser le calcul.

    LE MODELE, en une phrase : chaque faux candidat devance le vrai avec la probabilite
    1 - `quality`, independamment des autres, donc le rang du vrai suit une loi binomiale.

        candidats   8 essais, ordre actuel   ordre precedent   au hasard
              12            100,0 %                 99,7 %        88,7 %
              20             99,9 %                 85,3 %        18,0 %
              30             98,7 %                 38,3 %         0,4 %
              49             82,5 %                  1,9 %         0,0 %

    Deux lectures. La premiere justifie le 8 : aux comptes habituels il touche le vrai
    groupe presque a coup sur. La seconde dit ce que la cle de tri vaut REELLEMENT -- a 30
    candidats, la corriger a fait passer une carte de 38 % a 99 % de chances d'etre jouee,
    et c'est la meme mecanique, le meme plafond, le meme temps.

    OU LE 8 CESSE DE SUFFIRE : a 49 candidats -- la pire frame observee -- il reste 17,5 %
    de chances d'abandonner une carte qui portait un groupe. Il en faudrait 10. Le calcul
    n'est pas cable dans le plafond pour autant : il est REPORTE quand le bot abandonne
    effectivement des cartes (cf. `bot/diagnostics.py`), parce que rallonger le plafond
    coute 8,4 s par essai supplementaire sur toutes les cartes, y compris les calmes.

    CE QUE LE MODELE SUPPOSE, et il faut le lire avec : un seul vrai groupe par carte, des
    candidats independants, et une qualite d'ordre etablie sur 35 paires. C'est une
    ESTIMATION d'ordre de grandeur, pas une garantie -- et elle est plutot pessimiste, une
    carte portant souvent plus d'un groupe.
    """
    if candidates <= 1:
        return 1.0
    if attempts <= 0:
        return 0.0
    rate = min(max(1.0 - quality, 0.0), 1.0)
    autres = candidates - 1
    # Le vrai groupe est atteint si moins de `attempts` faux le devancent.
    return float(sum(comb(autres, i) * rate ** i * (1.0 - rate) ** (autres - i)
                     for i in range(min(attempts, autres + 1))))
