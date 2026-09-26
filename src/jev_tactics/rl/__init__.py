"""Apprentissage par renforcement : encodage d'etat et masque d'actions.

Le masquage d'actions est la condition d'existence du RL ici, pas un raffinement : a
chaque etat l'immense majorite des actions sont illegales, et sans masque PPO
echantillonne du vide. Le masque est construit A PARTIR de `planner/legal.py` -- le meme
generateur que le solveur -- pour qu'une politique ne puisse pas apprendre d'autres
regles que celles du jeu simule.

La reference a battre est deja mesuree : `sim/arena.py` donne 65 % de victoires pour la
recherche 1 tour. Un RL qui ne la depasse pas est casse, pas prometteur.
"""

from jev_tactics.rl.space import ActionSpace, action_mask, legal_table

__all__ = ["ActionSpace", "action_mask", "legal_table"]
