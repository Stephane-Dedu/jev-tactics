"""Etat de combat structure, serialisable et versionne.

Invariant du projet : on doit pouvoir rejouer une decision hors-ligne a partir d'un
etat logge, sans le jeu. Donc tout ici est pydantic (validation + JSON) et porte un
`schema_version` : un replay ne doit jamais casser silencieusement apres un refactor.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

# Grille iso Dofus : 560 cellules indexees 0..559, lignes alternees avec decalage.
GRID_CELLS = 560


class Team(str, Enum):
    ALLY = "ally"
    ENEMY = "enemy"


class Entity(BaseModel):
    """Un combattant sur le plateau. `cell` est un index de BoardMap, pas des pixels."""

    entity_id: str
    team: Team
    # Index dans le BoardMap COURANT, dont la taille varie d'un combat a l'autre (les
    # cases jouables sont detectees). Il n'y a donc pas de borne superieure fixe.
    #
    # Cette borne valait GRID_CELLS (560), reliquat de l'hypothese « le plateau est la
    # carte Ankama 560 cases » -- que le projet a lui-meme refutee par la mesure (a ce
    # zoom, 560 cases couvriraient 2944 px de large ; le plateau en fait ~990). La
    # contrainte a survecu a l'hypothese qui la justifiait, et a fini par faire tomber
    # tout le pipeline sur une frame ou le plateau detecte comptait ~2000 cases.
    cell: int = Field(ge=0)
    hp: int = Field(ge=0)
    hp_max: int = Field(gt=0)
    ap: int = Field(ge=0)  # PA
    mp: int = Field(ge=0)  # PM
    is_self: bool = False
    # Faux quand les PV sont un PLACEHOLDER, pas une mesure. La timeline donne bien des
    # ratios de PV, mais rien ne permet encore d'apparier un emplacement de timeline a
    # une entite precise du plateau. Le planificateur doit alors s'abstenir de raisonner
    # sur des seuils de mise a mort (cf. planner/scoring.py).
    hp_known: bool = True
    # LE PENDANT DE `hp_known`, POUR LA POSITION. Faux quand la case ne vient QUE des
    # marqueurs de couleur, sans confirmation par la zone de deplacement du jeu.
    #
    # La distinction n'est pas theorique. Mesure sur les captures de session ou les deux
    # signaux repondent : la case issue du marqueur couvre 0 % de la zone de deplacement
    # sur huit captures sur quatorze -- le marqueur n'est pas « un peu decale », il est
    # ailleurs, jusqu'a seize cases plus loin. C'est `movement_centre` qui rattrape, et
    # quand la zone n'est pas lisible il ne rattrape rien.
    #
    # Une position fausse ne se voit NULLE PART : les numeros de case restent plausibles,
    # le plan reste credible, et tout y est calcule autour d'un personnage qui n'est pas
    # la. Ce booleen est le seul endroit ou la difference peut etre lue.
    cell_confirmed: bool = False


class CombatState(BaseModel):
    """Etat complet d'un tour, tel que consomme par le planificateur."""

    schema_version: int = SCHEMA_VERSION
    turn: int = Field(ge=0)
    # None tant qu'on ne sait pas qui joue : la perception identifie de facon fiable
    # NOTRE tour (cf. is_our_turn), pas encore l'identite du combattant actif adverse.
    active_entity_id: str | None = None
    is_our_turn: bool = False
    entities: list[Entity]
    # Cellules infranchissables (decor). Les cellules occupees se deduisent des entities.
    obstacles: set[int] = Field(default_factory=set)
    # Cases surlignees par le JEU comme atteignables ce tour-ci (perception active).
    # Quand elle est renseignee, cette information prime sur notre propre pathfinding :
    # c'est le resultat exact du moteur, obstacles compris. None hors de notre tour.
    reachable_hint: set[int] | None = None

    def self_entity(self) -> Entity:
        """Le personnage joue. Leve si absent : en combat, son absence est une panne de
        perception, et rendre None la ferait voyager silencieusement jusqu'au planificateur.

        Pour les contextes ou la mort du personnage est une issue NORMALE (simulation,
        entrainement), utiliser `find_self()`.
        """
        me = self.find_self()
        if me is None:
            raise LookupError(
                f"aucune entite marquee is_self parmi {len(self.entities)} entites")
        return me

    def find_self(self) -> Entity | None:
        return next((e for e in self.entities if e.is_self), None)

    def occupied_cells(self) -> set[int]:
        return {e.cell for e in self.entities}

    def enemies(self) -> list[Entity]:
        return [e for e in self.entities if e.team is Team.ENEMY]

    def blocked_cells(self) -> set[int]:
        """Cases infranchissables : decor + occupants. Entree du pathfinding."""
        return self.obstacles | self.occupied_cells()
