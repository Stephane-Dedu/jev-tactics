"""Le monde hors combat : carte, deplacement, reperage.

`navigation` vivait sous `farming/` dans le depot d'origine, par accident d'histoire :
il n'importe RIEN de la recolte -- seulement la bibliotheque standard, cv2 et numpy. Le
ranger la rendait le graphe de cartes inaccessible a tout ce qui n'etait pas du farming,
alors que se deplacer est la brique la plus reutilisable du bot.
"""
