"""Les quetes : ce qu'il faut faire, et dans quel ordre.

**Rien de tout ceci n'existait**, ni ici ni dans le depot d'origine. Les seules mentions de
« quetes » en amont visent a MASQUER le panneau comme source de bruit visuel -- soit
l'inverse exact de le lire.

Ce paquet se scinde en deux moities de nature tres differente :

  - **la logique** (`model`, `director`) : quelle est l'etape courante, ou faut-il aller,
    que faut-il faire en arrivant. Pure, deterministe, testable sans le jeu ni une seule
    capture. C'est ce qui est ecrit ;
  - **la perception** (lire le journal de quetes, reperer un PNJ, reconnaitre une fenetre
    de dialogue) : impossible a ecrire sans captures de ces ecrans, dont le depot n'a
    aucune. C'est ce qui manque.

La separation n'est pas un accident de calendrier, c'est le meme decoupage que partout
ailleurs ici : `director` decide, il ne regarde rien et ne clique rien. Il rend une
INTENTION, que les couches deja ecrites savent executer -- `world.walker` pour se rendre
quelque part, `world.engage` puis `decision` pour un combat.
"""
