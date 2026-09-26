"""Reglages communs a toute la suite.

OPENCV EN UN SEUL FIL, pour que la perception soit REPRODUCTIBLE.

Constat, sur cette machine et sans plugin d'ordre aleatoire (l'ordre des tests est donc
identique d'une execution a l'autre) :

    execution 1   1635 passes
    execution 2   test_entities : 26 comptages justes au lieu de 27, et la LISTE des
                  captures en desaccord avait change
    execution 3   test_pipeline : test_timeline_trims_phantom_markers

Deux tests differents, sur deux executions differentes, avec les MEMES fichiers d'entree
et le meme ordre. Chacun passe seul, et passe apparie avec les nouveaux tests de boucle.
Ce ne sont donc ni des donnees qui changent, ni une dependance d'ordre.

CAUSE INFEREE, non demontree : `cv2.getNumThreads()` vaut 12 ici. OpenCV parallelise
filtres, correlations et composantes connexes ; l'ordre de reduction en virgule flottante
n'y est pas garanti, et une difference au dernier bit suffit a faire basculer une mesure
POSEE SUR UN SEUIL. C'est exactement le profil observe : ce sont les captures limites qui
entrent et sortent de la liste des desaccords, jamais les nettes.

Ce reglage ne corrige pas le bot -- en session, une detection marginale qui bascule d'une
frame a l'autre est sans consequence, la frame suivante arrive. Il rend la SUITE
reproductible, ce qui est la condition pour que ses temoins veuillent dire quelque chose :
un test qui rougit une fois sur cinq finit par se lire comme du bruit, et c'est ainsi
qu'on cesse de croire un temoin qui a raison.

Cout mesure : la suite passe de ~3 min 30 a ~4 min. Le prix d'un temoin auquel on peut se
fier.
"""

import cv2


def pytest_configure(config):
    """`config` est impose par pytest, et inutilise ici."""
    cv2.setNumThreads(1)
