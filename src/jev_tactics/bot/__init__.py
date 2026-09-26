"""La boucle vivante : jouer un combat sur le client reel.

`sim/` fait jouer des combats simules, `decision/` choisit quoi jouer, `action/` sait
cliquer. Il manquait ce qui relie les trois face au jeu : percevoir une frame, en tirer un
etat, decider, executer, recommencer jusqu'a la fin du combat.
"""
