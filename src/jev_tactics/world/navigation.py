"""Navigation entre cartes : circuits de recolte et trajets vers un point precis.

Deux besoins distincts, deliberement traites par deux outils :

  - **Circuit** (`Route`) : une suite cyclique de directions. C'est ce qu'il faut pour
    farmer -- on tourne en rond sur une zone, sans jamais avoir besoin de savoir OU l'on
    est. Robuste : aucune lecture d'ecran ne peut le mettre en defaut.

  - **Trajet** (`MapGraph`) : un plus court chemin vers une carte donnee, pour rejoindre
    la banque. Celui-la exige de connaitre sa position, donc une lecture de coordonnees.

Le monde de Dofus est une grille : depuis (x, y), aller a droite mene a (x+1, y), en bas
a (x, y+1). Les voisins se deduisent donc des coordonnees, et le graphe n'a besoin que de
la LISTE DES CARTES PRATICABLES -- pas d'enumerer les aretes une par une.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class Direction(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    @property
    def delta(self) -> tuple[int, int]:
        return {
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
            Direction.TOP: (0, -1),
            Direction.BOTTOM: (0, 1),
        }[self]

    @property
    def opposite(self) -> Direction:
        return {
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.TOP: Direction.BOTTOM,
            Direction.BOTTOM: Direction.TOP,
        }[self]


Coord = tuple[int, int]


def direction_of(delta: Coord) -> Direction | None:
    """Direction correspondant a un deplacement d'UNE carte, ou None sinon.

    Sert a nommer un deplacement CONSTATE : on voulait aller a droite, les coordonnees
    disent qu'on est monte. Rendre None pour tout ce qui n'est pas un pas unitaire est
    voulu -- un zaap ou un rappel deplacent de plusieurs cartes d'un coup, et la seule
    correction honnete est alors de ne pas en inventer une.
    """
    for direction in Direction:
        if direction.delta == delta:
            return direction
    return None


@dataclass
class MapGraph:
    """Cartes praticables d'une zone, et les transitions qu'on a vraiment CONSTATEES.

    Deux sources d'aretes, de fiabilite tres inegale, et il importe de ne pas les
    confondre :

      - `links` : une transition OBSERVEE en jouant. « Depuis (3,8), aller a droite mene
        en (4,8) » parce que le bot l'a fait et a relu ses coordonnees. C'est un fait.

      - l'ADJACENCE : depuis (3,8), la droite mene sans doute en (4,8) parce que c'est la
        carte voisine. C'est une supposition, et elle est fausse plus souvent qu'on ne le
        croirait : la detection de derive montre qu'un clic de bord fait MARCHER le
        personnage, et qu'un obstacle peut le faire sortir par un autre cote.

    Les faits l'emportent, l'adjacence bouche les trous. Cette hierarchie est la raison
    d'etre du graphe : il ne demandait jusqu'ici qu'une liste ecrite a la main que
    personne n'avait, ce qui le rendait inutilisable. Il se remplit desormais tout seul
    pendant la recolte.
    """

    walkable: set[Coord] = field(default_factory=set)
    # (origine, direction) -> destination reellement atteinte.
    links: dict[tuple[Coord, Direction], Coord] = field(default_factory=dict)
    # Rendement CUMULE par carte : (recoltes, passages). Il existait deja, mais seulement
    # le temps d'une session -- `FarmingSession.yields()` l'affichait puis le perdait.
    #
    # Or c'est la seule decision que la recolte demande : un circuit de dix cartes dont
    # trois portent tout le rendement se parcourt mieux en trois cartes. Cette decision se
    # prend sur des HEURES, pas sur une session, et jusqu'ici chaque session repartait de
    # zero. Le graphe est deja persistant ; y loger le rendement ne coute rien de plus.
    # (recoltes, COMBATS, passages). Les combats sont cumules pour la meme raison que les
    # recoltes, et elle pese plus lourd encore : sur un circuit de CHASSE -- une zone a
    # larves n'offre rien a recolter -- le combat EST le rendement. Sans eux, le classement
    # persistant vaut zero partout et ne designe aucune carte, precisement dans le mode ou
    # l'on s'en sert pour tailler le circuit.
    yields: dict[Coord, tuple[int, int, int]] = field(default_factory=dict)

    def record_visit(self, position: Coord, harvested: int, fights: int = 0) -> None:
        """Cumule un passage sur une carte et ce qu'il a rapporte."""
        taken, fought, visits = self.yields.get(position, (0, 0, 0))
        self.yields[position] = (taken + harvested, fought + fights, visits + 1)

    def best_maps(self, minimum_visits: int = 2,
                  by: str = "harvests") -> list[tuple[Coord, float]]:
        """Cartes triees PAR PASSAGE, des plus riches aux plus pauvres.

        `by` choisit ce qu'on classe : « harvests » ou « fights ». Un choix EXPLICITE et
        non une somme -- les deux ne se comparent pas, et les additionner inventerait une
        grandeur qui ne veut rien dire. C'est l'appelant qui sait s'il recolte ou s'il
        chasse.

        `minimum_visits` ecarte les cartes vues une seule fois : une carte visitee une
        fois et ayant rapporte trois ressources afficherait 3,0 par passage et dominerait
        le classement sur un seul echantillon. C'est le meme piege que les seuils regles
        sur une capture, et il a deja coute assez de temps a ce projet.
        """
        index = 1 if by == "fights" else 0
        rated = [(position, row[index] / row[2])
                 for position, row in self.yields.items()
                 if row[2] >= minimum_visits]
        return sorted(rated, key=lambda row: -row[1])

    @classmethod
    def from_json(cls, path: str | Path) -> MapGraph:
        """Charge un graphe.

        Deux formats acceptes : la simple liste `[[x, y], ...]` d'origine, ecrite a la
        main, et le format complet `{"walkable": [...], "links": [...]}` que le bot
        produit. Les circuits deja rediges continuent de fonctionner.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return cls(walkable={(int(x), int(y)) for x, y in data})
        graph = cls(walkable={(int(x), int(y)) for x, y in data.get("walkable", [])})
        for origin, direction, destination in data.get("links", []):
            graph.observe((int(origin[0]), int(origin[1])), Direction(direction),
                          (int(destination[0]), int(destination[1])))
        for row in data.get("yields", []):
            # Deux longueurs acceptees : (carte, recoltes, passages) d'avant les combats,
            # et (carte, recoltes, combats, passages) depuis. Refuser les anciennes
            # lignes jetterait un rendement accumule sur des heures pour un champ ajoute
            # apres coup -- c'est justement ce que la persistance sert a eviter.
            position = (int(row[0][0]), int(row[0][1]))
            taken, fights, visits = (
                (int(row[1]), 0, int(row[2])) if len(row) == 3
                else (int(row[1]), int(row[2]), int(row[3])))
            graph.yields[position] = (taken, fights, visits)
        return graph

    def describe_learning(self, maps_visited: int) -> str | None:
        """Ce que le graphe a appris pendant la session, ou POURQUOI il n'a rien appris.

        La combinaison qui compte n'etait nommee nulle part : des cartes parcourues et
        ZERO transition constatee. Elle a une cause unique et precise -- une liaison ne
        s'enregistre que si les coordonnees sont lues AVANT et APRES le changement, et
        `_travel` retombe sinon sur la comparaison d'images, qui confirme le changement
        sans savoir vers ou.

        Sans ce message, le bilan affiche « 6 cartes » puis « 0 transitions » sur deux
        lignes voisines, et cette contradiction apparente se lit comme un graphe casse.
        C'est l'inverse : le graphe fonctionne, ce sont les coordonnees qui manquent.
        -> None quand il n'y a rien a signaler.
        """
        if maps_visited <= 0:
            return None
        if self.links:
            return None
        return (f"{maps_visited} carte(s) parcourue(s) mais AUCUNE transition apprise — "
                f"les coordonnees n'ont jamais ete lisibles des deux cotes d'un "
                f"changement de carte. Le circuit fonctionne, le graphe reste aveugle : "
                f"il ne pourra ni corriger une derive ni conseiller un circuit")

    def to_json(self, path: str | Path) -> None:
        """Ecrit le graphe. Trie, pour que deux sessions successives se comparent."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Une entree par ligne, sans indentation imbriquee : `indent=` eclaterait chaque
        # coordonnee sur trois lignes et rendrait le fichier illisible a l'oeil, alors
        # qu'on veut pouvoir le relire et le corriger a la main.
        walkable = ",\n  ".join(json.dumps(list(c)) for c in sorted(self.walkable))
        links = ",\n  ".join(
            json.dumps([list(origin), direction.value, list(destination)])
            for (origin, direction), destination
            in sorted(self.links.items(), key=lambda kv: (kv[0][0], kv[0][1].value)))
        rendement = ",\n  ".join(
            json.dumps([list(position), taken, fights, visits])
            for position, (taken, fights, visits) in sorted(self.yields.items()))
        target.write_text(
            f'{{\n "walkable": [\n  {walkable}\n ],\n "links": [\n  {links}\n ],\n'
            f' "yields": [\n  {rendement}\n ]\n}}\n',
            encoding="utf-8")

    def suggest_route(self, start: Coord, keep: int = 4, minimum_visits: int = 2,
                      by: str = "fights") -> list[Direction] | None:
        """Circuit FERME propose a partir de ce que la session a appris. -> pas, ou None.

        Ferme la boucle que le depot laissait ouverte. Le bot mesure deja quelles cartes
        rapportent (`yields`, cumule d'une session a l'autre) et quelles sorties menent ou
        (`links`, constatees en jouant) -- mais rien ne joignait les deux, et composer un
        circuit restait un travail a la main sur une zone qu'on ne connait pas encore.

        La methode : garder les `keep` meilleures cartes, les enchainer de proche en
        proche depuis `start` -- la plus proche d'abord, au sens du nombre de cartes --
        puis revenir au depart. Ce n'est PAS un optimum : le probleme est celui du
        voyageur de commerce, et le resoudre exactement sur des donnees aussi bruitees
        que « 3 combats en 2 passages » serait une precision inventee. Un circuit ferme
        qui passe par les bonnes cartes vaut mieux qu'un circuit optimal sur des chiffres
        faux.

        Rend None plutot qu'un circuit partiel des qu'une etape est inatteignable : un
        circuit qui ne revient pas a son point de depart derive d'une carte par tour, et
        c'est la panne la plus difficile a constater du farming -- le bilan continue
        d'annoncer « N cartes parcourues » pendant que le bot s'eloigne.
        """
        classees = [position for position, _ in self.best_maps(minimum_visits, by=by)]
        cibles = [p for p in classees[:keep] if p != start]
        if not cibles:
            return None

        pas: list[Direction] = []
        ici = start
        restantes = list(cibles)
        while restantes:
            # La plus PROCHE d'abord, et non la meilleure : l'ordre du classement dit
            # quelles cartes valent le detour, pas dans quel ordre les parcourir.
            legs = [(self.route(ici, cible), cible) for cible in restantes]
            atteignables = [(chemin, cible) for chemin, cible in legs if chemin is not None]
            if not atteignables:
                return None
            chemin, cible = min(atteignables, key=lambda couple: len(couple[0]))
            pas.extend(chemin)
            ici = cible
            restantes.remove(cible)

        retour = self.route(ici, start)
        if retour is None:
            return None
        pas.extend(retour)
        return pas or None

    def save_suggested_route(self, start: Coord, path: str | Path,
                             overwrite: bool = False, **kwargs) -> str:
        """Ecrit le circuit suggere. -> une ligne a afficher, jamais d'exception.

        Ce que le depot demandait encore a la main : lire six directions dans un bilan et
        les recopier dans un JSON. C'est le dernier pas d'une chaine ou tout le reste se
        mesure tout seul -- et un pas manuel au bout d'une mesure automatique est
        exactement ce qui fait qu'on ne s'en sert pas.

        N'ECRASE PAS sans qu'on le demande. Un circuit existant a souvent ete corrige a la
        main, sur des heures de jeu ; le remplacer par une suggestion tiree de la derniere
        session serait un echange perdant, et silencieux.
        """
        pas = self.suggest_route(start, **kwargs)
        if not pas:
            return ("aucun circuit a suggerer : le graphe n'a pas encore vu deux fois la "
                    "meme carte")
        target = Path(path)
        if target.exists() and not overwrite:
            return (f"{target} existe deja — non ecrase. Le circuit suggere est "
                    f"{[d.value for d in pas]}")
        Route(steps=list(pas)).to_json(target)
        return f"circuit suggere ecrit dans {target} ({len(pas)} pas)"

    def barren_on(self, route: Route, start: Coord,
                  minimum_visits: int = 2) -> list[Coord]:
        """Cartes du circuit qui n ont JAMAIS rien donne, en assez de passages pour le dire.

        Le circuit ne connait que des directions RELATIVES -- c est ce qui le rend
        increvable, aucune lecture ratee ne peut le derailler. La position de depart, elle,
        est absolue. Leur composition dit donc quelles cartes il va parcourir, et le graphe
        dit ce qu elles ont rapporte.

        C est la premiere chose de ce projet qui touche le RENDEMENT et non la fiabilite :
        un circuit de six cartes dont trois sont steriles fait perdre la moitie du temps en
        deplacements. Le bot ne peut pas reecrire le circuit a la place de l utilisateur --
        une carte peut etre traversee pour en atteindre une autre -- mais il peut le lui
        DIRE, chiffres a l appui.

        `minimum_visits` ecarte les cartes vues une seule fois : declarer sterile une carte
        sur un seul passage est le meme piege que regler un seuil sur une capture.

        STERILE veut dire « n'a rien donne DU TOUT » : ni recolte, ni combat. Juger sur
        les seules recoltes conseillait de retirer du circuit les meilleures cartes d'une
        zone de chasse -- une carte a larves ne rapporte aucune ressource par
        construction, et c'est celle qu'on veut garder. Le conseil etait exactement
        inverse de ce qu'il fallait faire.
        """
        x, y = start
        steriles = []
        for dx, dy in route.offsets()[:-1]:
            position = (x + dx, y + dy)
            taken, fights, visits = self.yields.get(position, (0, 0, 0))
            if visits >= minimum_visits and taken == 0 and fights == 0:
                steriles.append(position)
        return steriles

    def observe(self, origin: Coord, direction: Direction, destination: Coord) -> None:
        """Enregistre une transition CONSTATEE. Les deux cartes deviennent praticables.

        Une observation ecrase la precedente : si la meme sortie mene ailleurs
        aujourd'hui, c'est la mesure d'aujourd'hui qui vaut. Le monde peut changer
        (obstacle retire, personnage qui contourne autrement), et garder la premiere
        valeur a jamais serait s'attacher a un fait perime.
        """
        self.walkable.add(origin)
        self.walkable.add(destination)
        self.links[(origin, direction)] = destination

    def __contains__(self, coord: Coord) -> bool:
        return coord in self.walkable

    def neighbours(self, coord: Coord) -> list[tuple[Direction, Coord]]:
        """Cartes atteignables depuis `coord`, avec la direction a prendre.

        Les transitions observees priment ; l'adjacence ne sert que la ou l'on n'a jamais
        essaye. Une arete supposee peut donc echouer -- le runner verifie chaque
        deplacement de toute facon, et l'observation qu'il en tire corrige le graphe.
        """
        x, y = coord
        out = []
        for direction in Direction:
            observed = self.links.get((coord, direction))
            if observed is not None:
                out.append((direction, observed))
                continue
            dx, dy = direction.delta
            candidate = (x + dx, y + dy)
            if candidate in self.walkable:
                out.append((direction, candidate))
        return out

    def route(self, start: Coord, goal: Coord) -> list[Direction] | None:
        """Plus court chemin en nombre de cartes, ou None si inatteignable.

        BFS : tous les changements de carte se valent, inutile de ponderer.
        """
        if start == goal:
            return []
        if start not in self.walkable or goal not in self.walkable:
            return None

        previous: dict[Coord, tuple[Coord, Direction]] = {}
        queue = deque([start])
        seen = {start}
        while queue:
            current = queue.popleft()
            for direction, neighbour in self.neighbours(current):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                previous[neighbour] = (current, direction)
                if neighbour == goal:
                    steps: list[Direction] = []
                    node = goal
                    while node != start:
                        node, taken = previous[node]
                        steps.append(taken)
                    return steps[::-1]
                queue.append(neighbour)
        return None


@dataclass
class Route:
    """Circuit de recolte : une suite cyclique de directions.

    Ne demande AUCUNE connaissance de la position : on avance, et quand la suite est
    epuisee on recommence. C'est ce qui la rend increvable -- une lecture de coordonnees
    ratee ne peut pas la faire derailler.

    Elle sait toutefois EN PROFITER quand elles sont lisibles : `divert()` insere un pas
    correctif sans toucher a l'index du circuit, de sorte qu'un detour subi (on voulait
    aller a droite, on est monte) se rattrape puis reprend exactement ou il en etait.
    """

    steps: list[Direction]
    index: int = 0
    laps: int = 0
    # Pas correctifs a jouer AVANT de reprendre le circuit. Separes des `steps` a dessein :
    # un detour est ponctuel et ne doit ni decaler l'index, ni compter un tour de circuit.
    detours: deque[Direction] = field(default_factory=deque)
    # Melangeur des pas, applique a chaque tour de circuit. None pour garder l'ordre
    # ecrit -- utile pour rejouer une session a l'identique.
    shuffle: random.Random | None = None
    _last: Direction | None = field(default=None, repr=False)
    _last_was_detour: bool = field(default=False, repr=False)

    # /!\ __len__ ci-dessous rend une Route VIDE falsy. Ne jamais ecrire
    # `route or defaut` : une route volontairement vide serait remplacee en silence.
    # Tester `route is None`.

    @classmethod
    def from_json(cls, path: str | Path) -> Route:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(steps=[Direction(step) for step in data])

    def to_json(self, path: str | Path) -> None:
        """Ecrit le circuit, dans le format que `from_json` relit.

        Sur UNE ligne : un circuit se lit d'un coup d'oeil, et se corrige a la main --
        c'est meme l'usage principal du fichier.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([step.value for step in self.steps]) + "\n", encoding="utf-8")

    def __len__(self) -> int:
        return len(self.steps)

    def offsets(self) -> list[Coord]:
        """Positions relatives visitees, en partant de (0, 0) inclus.

        Sert a decrire la FORME du circuit sans rien savoir de la zone : le nombre de
        positions distinctes est le nombre de cartes reellement parcourues, qui n'est pas
        le nombre de pas des qu'on repasse quelque part.
        """
        x = y = 0
        seen = [(0, 0)]
        for step in self.steps:
            dx, dy = step.delta
            x, y = x + dx, y + dy
            seen.append((x, y))
        return seen

    def net_displacement(self) -> Coord:
        """Deplacement cumule d'un TOUR complet. (0, 0) pour un circuit ferme."""
        return self.offsets()[-1]

    @property
    def is_closed(self) -> bool:
        """Un tour ramene-t-il au point de depart ?

        C'est l'invariant d'un circuit de recolte, et rien ne le verifiait. Une route
        ecrite a la main avec un pas en trop -- l'erreur la plus facile a commettre --
        fait avancer le bot en ligne droite pour toujours, a raison d'une carte par tour.
        Il quitte sa zone en quelques minutes, ne trouve plus rien, et le bilan continue
        d'annoncer « N cartes parcourues » comme si tout allait bien.
        """
        return self.net_displacement() == (0, 0)

    def describe_shape(self) -> str:
        """Resume lisible, a afficher au lancement plutot qu'a decouvrir apres coup."""
        if not self.steps:
            return "circuit vide"
        maps = len(set(self.offsets()[:-1]))
        if self.is_closed:
            return f"{len(self.steps)} pas, {maps} carte(s), ferme"
        dx, dy = self.net_displacement()
        return (f"{len(self.steps)} pas, {maps} carte(s) — NON FERME : chaque tour "
                f"deplace de ({dx:+d}, {dy:+d}), donc le bot s'eloigne de sa zone "
                f"d'autant a chaque passage")

    def peek(self) -> Direction | None:
        if self.detours:
            return self.detours[0]
        return self.steps[self.index] if self.steps else None

    def divert(self, *directions: Direction) -> None:
        """Insere un ou plusieurs pas correctifs avant la suite du circuit.

        Appele quand les coordonnees revelent qu'on a change de carte dans la MAUVAISE
        direction. Sans cela le circuit continuerait depuis une carte decalee -- et comme
        il ne connait que des directions relatives, le decalage serait definitif.

        Plusieurs pas, parce qu'un ecart d'une carte n'est que le cas facile : un saut
        (zaap, rappel) demande un trajet, que le graphe des cartes sait calculer. On les
        insere dans l'ORDRE donne -- d'ou le parcours a l'envers, `appendleft` empilant.
        """
        for direction in reversed(directions):
            self.detours.appendleft(direction)

    def advance(self) -> Direction | None:
        """Direction suivante : les detours d'abord, puis le circuit."""
        if self.detours:
            self._last, self._last_was_detour = self.detours.popleft(), True
            return self._last
        if not self.steps:
            return None
        step = self.steps[self.index]
        self.index += 1
        if self.index >= len(self.steps):
            self.index = 0
            self.laps += 1
            self._reshuffle()
        self._last, self._last_was_detour = step, False
        return step

    def _reshuffle(self) -> None:
        """Rebat l'ordre des pas a chaque tour de circuit, si `shuffle` le demande.

        UNE TRAJECTOIRE EXACTEMENT PERIODIQUE est la signature la plus simple qui existe.
        Les travaux sur la detection de bots (F-score 99,4 a 99,7 %) exploitent justement
        les trajectoires de deplacement, et le geste — delais log-normaux, courbes de
        Bezier — ne sert a rien si le CHEMIN, lui, se repete a l'identique indefiniment.

        Ce remede est gratuit, et c'est ce qui le rend evident une fois vu : la FERMETURE
        d'un circuit ne depend que du MULTI-ENSEMBLE de ses pas, pas de leur ordre. Le
        deplacement net est une somme, et une somme est commutative. Toute permutation
        d'un circuit ferme est donc un circuit ferme -- l'invariant que ce projet protege
        depuis le debut survit intact.

        Ce qui change en revanche est le CHEMIN parcouru : les cartes visitees en cours de
        route different d'un tour a l'autre, meme si l'on revient au point de depart. Un
        detour subi se rattrape toujours par `divert()`, qui ne touche pas a l'ordre.
        """
        if self.shuffle is not None:
            self.shuffle.shuffle(self.steps)

    def rewind(self) -> None:
        """Revient d'un pas : le changement de carte n'a pas eu lieu."""
        if self._last_was_detour and self._last is not None:
            # Un detour rate reste a faire : le remettre en tete plutot que de le perdre,
            # sinon la correction serait abandonnee et le decalage rendu permanent.
            self.detours.appendleft(self._last)
            self._last_was_detour = False
            return
        if not self.steps:
            return
        if self.index == 0 and self.shuffle is not None:
            # LE TOUR VIENT DE BOUCLER, DONC `_reshuffle` A DEJA REBATTU LES PAS. Reculer
            # l'indice ne redonne alors PAS le pas rate -- `steps[-1]` designe un autre pas
            # depuis le rebattage -- et, pire, il place le curseur a la FIN du tour neuf :
            # le prochain `advance` consomme ce dernier pas, reboucle aussitot, et les
            # len-1 pas du debut ne sont JAMAIS joues. Le tour ne vaut plus (0, 0), et le
            # circuit n'est plus ferme.
            #
            # Mesure sur le circuit d'exemple, 200 suites de 200 pas avec 35 % d'echecs :
            # 199 suites sur 200 derivaient. La sonde de derive existante ne l'avait jamais
            # vu parce qu'elle construit ses routes SANS rebattage -- le defaut ne nait que
            # de la rencontre des deux.
            #
            # Le pas rate est remis en DETOUR, mecanisme deja en place trois lignes plus
            # haut pour un detour rembobine : il sera rejoue au prochain `advance`, sans
            # toucher a l'ordre du tour neuf. Le multi-ensemble d'un tour reste donc exact,
            # et c'est lui seul qui ferme le circuit.
            #
            # `laps` n'est PAS decremente ici, contrairement au cas sans rebattage : un
            # tour neuf a bel et bien commence -- les pas ont ete rebattus -- et le pas en
            # souffrance voyage avec les detours, pas avec le compteur. Le decrementer
            # desynchroniserait le compte des rebattages, qui eux ont eu lieu.
            if self._last is not None:
                self.detours.appendleft(self._last)
            return
        if self.index == 0:
            self.index = len(self.steps) - 1
            self.laps = max(0, self.laps - 1)
        else:
            self.index -= 1


# Ecart moyen d'intensite au-dela duquel on considere que la carte a change.
MAP_CHANGE_THRESHOLD = 18.0


def map_changed(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    threshold: float = MAP_CHANGE_THRESHOLD,
) -> bool:
    """Vrai si les deux captures montrent des cartes differentes.

    On compare des vignettes : un changement de carte redessine tout l'ecran, alors
    qu'un deplacement du personnage ou une animation n'en modifie qu'une fraction. Le
    sous-echantillonnage rend justement la mesure insensible a ces petits mouvements.

    /!\\ Ce que la methode compare est la STRUCTURE A GRANDE ECHELLE. Mesure : deux
    images de bruit uniforme pourtant differentes donnent un ecart de 7,2 seulement (le
    vignettage moyenne le bruit), la ou deux cartes aux zones distinctes le depassent
    largement. Corollaire honnete : deux cartes visuellement TRES SEMBLABLES pourraient
    ne pas declencher la detection. Le seuil n'a pas encore ete valide sur une vraie
    transition de carte, faute de capture avant/apres.

    Sert de VERIFICATION apres un clic de bord : sans elle, le bot croirait avoir change
    de carte alors que le deplacement a echoue (obstacle, clic hors zone), et tout le
    circuit se decalerait.

    MESURE, faite depuis, sur les 28 paires que forment les huit captures reelles -- dont
    les coordonnees sont lues et verifiees a l'oeil, ce qui donne la verite terrain :

        meme carte (1 paire)          49,4
        cartes differentes            12,0 a 50,0, mediane 33,0

    Les deux classes se CHEVAUCHENT, et la raison est instructive : la seule paire « meme
    carte » oppose une capture hors combat a une capture EN combat. La vignette mesure donc
    surtout l'interface et l'etat de combat, pas le terrain.

    Ce chevauchement ne condamne pas le seuil, parce que 27 de ces 28 paires ne sont PAS la
    situation ou cette fonction sert. Elle est appelee entre deux frames encadrant un clic
    de bord, hors combat. Une seule paire correspond -- capture.png (5,8) contre
    hors_combat.png (2,9) -- et elle donne 29,4, au-dessus des 18. Preuve faible, mais dans
    le bon sens ; le seuil reste donc inchange, faute de mieux.

    TROIS PISTES MESUREES ET ECARTEES pour lever le chevauchement :

      - masquer l'ATH avant de comparer, puisqu'il domine la mesure. Il descend bien la
        paire « meme carte » de 49,4 a 26,4, mais fait tomber le minimum des cartes
        differentes de 12,0 a 6,4 : la separation ne s'ameliore pas ;
      - comparer le NOM DE ZONE du bandeau superieur, signal unilateral ideal (s'il change,
        la carte a change). Meme zone 4,0 a 42,1 contre zones differentes 32,6 a 50,0 --
        et les quatre valeurs aberrantes impliquent TOUTES la capture dont le panneau de
        quetes recouvre ce bandeau. Un panneau ouvert suffit a le detruire ;
      - se fier aux seules coordonnees. C'est deja ce que fait l'appelant : cette fonction
        n'est qu'un REPLI quand elles sont illisibles. Depuis que les gabarits couvrent
        huit captures sur huit, ce repli sert moins souvent -- mais quand il se trompe en
        disant « pas de changement », le circuit rembobine un deplacement REUSSI et se
        decale pour de bon.
    """
    if before.shape != after.shape:
        return True

    small_before = cv2.resize(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (64, 36))
    small_after = cv2.resize(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (64, 36))
    difference = np.abs(small_after.astype(np.int16) - small_before.astype(np.int16))
    return float(difference.mean()) >= threshold
