"""Definition des sorts et conditions de lancer.

Les plafonds de lancer (par tour, par cible) ne sont pas un detail de regles : ce sont
les elagages les plus efficaces du solveur (cf. §6.1 du doc d'archi). Sans eux l'arbre
d'un tour atteint ~10^8 sequences ; avec eux il devient enumerable.

La ligne de vue se calcule sur les coordonnees du reseau : on echantillonne le segment
entre les deux cases et on regarde s'il traverse un obstacle. Les extremites sont
exclues -- une cible n'obstrue pas sa propre visee, et on ne se bloque pas soi-meme.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from jev_tactics.calibration.grid import BoardMap
from jev_tactics.rules.movement import grid_distance


class Spell(BaseModel):
    """Un sort du personnage. Les degats sont une fourchette (min, max)."""

    name: str
    # D'ou viennent les valeurs de ce sort. Ce n'est pas de la documentation : le solveur
    # compare des coups, et comparer une valeur mesuree a une valeur inventee produit un
    # arbitrage qui a l'air fonde sans l'etre. Savoir lesquelles sont approximatives est
    # la seule facon de ne pas prendre un ordre de grandeur pour une mesure.
    #   "mesure" : releve dans le jeu (infobulle du sort)
    #   "role"   : archetype (rapports coherents, valeurs approximatives)
    #   "saisie" : entre a la main, non verifie
    source: str = "saisie"
    # Raccourci clavier explicite (« 1 », « ctrl+3 », « ctrl+shift+2 »). Prioritaire sur
    # la deduction par `slot` : au-dela de la premiere ligne, les barres sont liees a des
    # COMBINAISONS, que la position seule ne permet pas de deviner.
    key: str | None = None
    ap_cost: int = Field(gt=0)
    # Emplacement dans la barre de sorts (lecture par lignes, 0 = haut-gauche). Sert a
    # deux choses : savoir si le sort est lancable (perception/spellbar.py) et le
    # selectionner par raccourci. None = non renseigne : le sort est alors suppose
    # toujours disponible.
    slot: int | None = None
    range_min: int = Field(ge=0, default=1)
    range_max: int = Field(ge=0)
    needs_line_of_sight: bool = True
    # Contraintes d'alignement du jeu. « En ligne » signifie que la cible partage une
    # coordonnee du reseau avec le lanceur -- c'est-a-dire qu'elle est sur l'une des
    # quatre lignes de cases qui partent de lui.
    #
    # Sans ces contraintes, le solveur visait n'importe quelle case a portee. Constate en
    # jeu : « Ravage en ligne droite mais tente sur un ennemi pas dans la ligne ». Le
    # coup etait planifie, le clic parti, et le jeu le refusait -- tour perdu, sans que
    # rien ne signale l'erreur. Quatre sorts de la classe sont concernes, dont les deux
    # plus puissants (Ravage 28-32, Desolation 26-30).
    # Element des degats. Determinant pour un personnage SPECIALISE : en Dofus, les
    # degats dependent de la caracteristique de l'element, et une Force elevee ne
    # renforce QUE les sorts Terre. Les degats de base de dofusdb ne sont donc pas
    # comparables entre elements des que le build est oriente -- or le solveur ne fait
    # que comparer.
    element: str | None = None
    cast_in_line: bool = False
    cast_in_diagonal: bool = False
    max_casts_per_turn: int = Field(gt=0, default=99)
    max_casts_per_target: int = Field(gt=0, default=99)
    damage_min: int = Field(ge=0, default=0)
    damage_max: int = Field(ge=0, default=0)
    # Rayon d'effet de zone (0 = cible unique).
    area_radius: int = Field(ge=0, default=0)
    # FORME de la zone. Le jeu en distingue plusieurs, et les confondre avec un cercle
    # fausse l'evaluation : une croix de rayon 1 touche 4 cases, un cercle 8. Le solveur
    # SOMME les degats sur les cases touchees, donc surestimer la zone lui fait choisir un
    # sort de groupe la ou une cible unique rapportait plus.
    #
    # Formes relevees sur les sorts Sacrieur (champ `zoneDescr.shape`, en ASCII) :
    #   'P' cible unique  |  'C' cercle  |  '+' croix  |  'X' croix diagonale
    #   'G' et 'U' : semantique non verifiee, traitees en cercle et SIGNALEES comme telles
    area_shape: str = "circle"

    def area_cells(self, board: BoardMap, target: int) -> list[int]:
        """Cases touchees par un lancer sur `target` (la cible incluse)."""
        if self.area_radius == 0 or self.area_shape == "point":
            return [target]

        hit = []
        ti, tj = (int(v) for v in board.cells[target])
        for cell in range(len(board)):
            di = int(board.cells[cell][0]) - ti
            dj = int(board.cells[cell][1]) - tj

            if self.area_shape == "diagonal":
                # Les cases diagonales sont a DEUX pas de grille : mesurer la zone en
                # `grid_distance` n'en toucherait aucune a rayon 1. On compte donc les
                # pas EN DIAGONALE, ce qui est l'unite dans laquelle le jeu exprime
                # cette forme.
                if abs(di) != abs(dj) or abs(di) > self.area_radius:
                    continue
            else:
                if grid_distance(board, target, cell) > self.area_radius:
                    continue
                # A rayon 1 la croix et le cercle coincident : la distance de grille est
                # deja en croix. Ils ne divergent qu'a partir de 2.
                if self.area_shape == "cross" and not (di == 0 or dj == 0):
                    continue
            hit.append(cell)
        return hit

    @property
    def average_damage(self) -> float:
        return (self.damage_min + self.damage_max) / 2.0

    def in_range(self, board: BoardMap, caster: int, target: int) -> bool:
        distance = grid_distance(board, caster, target)
        if not (self.range_min <= distance <= self.range_max):
            return False
        return self.is_aligned(board, caster, target)

    def is_aligned(self, board: BoardMap, caster: int, target: int) -> bool:
        """Respecte les contraintes d'alignement, s'il y en a."""
        if not (self.cast_in_line or self.cast_in_diagonal):
            return True
        di = int(board.cells[target][0]) - int(board.cells[caster][0])
        dj = int(board.cells[target][1]) - int(board.cells[caster][1])
        # « En ligne » : une seule coordonnee du reseau varie. « En diagonale » : les deux
        # varient d'autant. Un sort declarant les deux accepte l'une ou l'autre.
        if self.cast_in_line and (di == 0 or dj == 0):
            return True
        return bool(self.cast_in_diagonal and abs(di) == abs(dj) and di != 0)



# Raccourcis par defaut de la premiere ligne de la barre : 1..9 puis 0.
# HYPOTHESE, pas une mesure -- c'est la configuration Dofus par defaut, mais un joueur
# peut l'avoir remappee. D'ou `--spell-keys`, qui prime toujours.
DEFAULT_ROW_KEYS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")


def default_spell_keys(spells: list[Spell]) -> dict[str, str]:
    """Raccourcis deduits du `slot` : nom du sort -> touche.

    Le lien slot -> touche existe deja dans le jeu ; le redemander a l'utilisateur
    l'obligeait a ressaisir une information qu'il avait deja donnee, avec une faute de
    frappe possible a chaque fois. Or un sort sans raccourci est SAUTE en silence, ce qui
    produit un bot qui n'attaque pas -- le mode de defaillance le plus couteux du projet.

    Seule la premiere ligne est couverte : au-dela, les raccourcis dependent de
    modificateurs (Maj, Ctrl) dont la convention n'a pas ete verifiee. Mieux vaut ne rien
    proposer que proposer une touche fausse, qui lancerait le mauvais sort.
    """
    keys: dict[str, str] = {}
    for spell in spells:
        if spell.key:
            keys[spell.name] = spell.key          # raccourci declare : il fait autorite
        elif spell.slot is not None and 0 <= spell.slot < len(DEFAULT_ROW_KEYS):
            keys[spell.name] = DEFAULT_ROW_KEYS[spell.slot]
    return keys


# Facteur applique aux sorts de l'element principal. MODELE, pas mesure : la formule du
# jeu multiplie les degats de base par (1 + caracteristique/100). Un build specialise a
# 200-300 dans sa caracteristique et ~0 ailleurs, d'ou un rapport de 3 a 4 entre un sort
# de son element et un autre. La valeur exacte importe peu -- ce qui compte est que
# l'ecart existe, sans quoi le solveur choisit un sort hors-element pour 4 points de
# degats de base qu'il n'infligera jamais.
DEFAULT_ELEMENT_FACTOR = 3.0


def apply_element_focus(
    spells: list[Spell], element: str | None, factor: float = DEFAULT_ELEMENT_FACTOR
) -> list[Spell]:
    """Renforce les sorts de l'element principal du personnage.

    Rend des COPIES : la liste d'origine reste celle du jeu, et l'on peut comparer les
    deux. Un `element` absent laisse tout inchange -- ne rien supposer par defaut.
    """
    if not element:
        return spells
    target = element.lower()
    scaled = []
    for spell in spells:
        if spell.element != target:
            scaled.append(spell)
            continue
        scaled.append(spell.model_copy(update={
            "damage_min": int(round(spell.damage_min * factor)),
            "damage_max": int(round(spell.damage_max * factor)),
        }))
    return scaled


def audit_spell_config(spells: list[Spell], keys: dict[str, str]) -> list[str]:
    """Ce que la config PROMET, et ce que le bot en fera reellement. -> lignes prefixees.

    Le silence est le cas normal ; « /!\\ » designe ce qui appelle une action.

    Trois ecarts, tous invisibles autrement, et le deuxieme est le defaut que ce projet
    designe comme le plus couteux -- un sort planifie puis SAUTE a l'execution, ce qui
    produit un bot qui n'attaque pas.

    Mesure sur la vraie configuration du Sacrieur, 20 sorts : **9 sont utilitaires**,
    donc invisibles au solveur (cf. `planner.prune.offensive_spells`, qui les ecarte pour
    garder la recherche exhaustive). Pres de la moitie d'une config obtenue par
    `fetch_spells.py` ne sert donc a rien, et rien ne le disait -- ce n'est pas une panne,
    c'est une attente a corriger.
    """
    lignes: list[str] = []
    offensifs = [s for s in spells if s.damage_max > 0]
    utilitaires = [s.name for s in spells if s.damage_max <= 0]

    if not offensifs:
        # Aucun sort offensif : le solveur ne pourra JAMAIS frapper. C'est la meme panne
        # que la barre mal configuree, un cran plus tot -- et elle est certaine, pas
        # probable.
        lignes.append("/!\\ aucun sort offensif dans la config : le solveur ne pourra "
                      "lancer aucun sort (tous ont damage_max = 0)")
    elif utilitaires:
        # Observation, pas avertissement : l'exclusion est DELIBEREE et mesuree. Sans la
        # dire, l'utilisateur croit disposer de vingt sorts.
        lignes.append(f"sorts : {len(offensifs)} offensifs utilises par le solveur, "
                      f"{len(utilitaires)} utilitaires ignores ({', '.join(utilitaires)})"
                      f" — soins, buffs et deplacements ne sont pas modelises")

    muets = [s.name for s in offensifs if s.name not in keys]
    if muets:
        # LE defaut le plus couteux : le solveur planifie le sort, l'executeur le saute
        # faute de raccourci, et le tour part sans lui. `default_spell_keys` ne couvre que
        # la premiere ligne de la barre ; au-dela il faut un `key` explicite.
        lignes.append(f"/!\\ {len(muets)} sort(s) offensif(s) SANS raccourci : "
                      f"{', '.join(muets)}. Ils seront planifies puis SAUTES a "
                      f"l'execution — ajouter « key » dans la config, ou --spell-keys")

    connus = {s.name for s in spells}
    inconnus = sorted(name for name in keys if name not in connus)
    if inconnus:
        # Un raccourci pour un sort qui n'existe pas est une faute de frappe, et elle ne
        # fait RIEN : le sort vise garde son raccourci d'origine. La correction que
        # l'utilisateur croyait avoir apportee n'a pas eu lieu.
        lignes.append(f"/!\\ raccourci(s) pour un sort absent de la config : "
                      f"{', '.join(inconnus)} — faute de frappe ? Ils sont sans effet")

    return lignes


def load_spells(path: str | Path) -> list[Spell]:
    """Charge une liste de sorts depuis un JSON (cf. configs/spells/).

    Deux formes acceptees : la liste nue d'origine, et `{"placeholder": ..., "spells":
    [...]}`. La seconde existe pour pouvoir MARQUER une config d'exemple -- voir
    `is_placeholder` et le refus dans `bot/setup.py`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["spells"] if isinstance(data, dict) else data
    return [Spell.model_validate(entry) for entry in entries]


def is_placeholder(path: str | Path) -> bool:
    """Cette config est-elle un EXEMPLE, et non les sorts d'un vrai personnage ?

    Le danger n'est pas theorique. `configs/spells/example.json` contient trois sorts
    inventes -- « frappe », « trait », « souffle » -- avec de vrais raccourcis clavier.
    Jouer avec elle ne rate pas des sorts : le solveur planifie sur des portees et des
    degats FICTIFS, puis appuie sur ces touches, donc de VRAIS sorts partent, choisis par
    un raisonnement faux. C'est la meme famille que la barre remappee, en pire, parce que
    rien dans la sortie ne le trahit.

    Et le garde-fou existant ne l'attrape pas : « executer exige des raccourcis » est
    satisfait, l'exemple en declare trois.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return bool(data.get("placeholder")) if isinstance(data, dict) else False


def available_spells(spells: list[Spell], ready: set[int] | None) -> list[Spell]:
    """Filtre les sorts reellement lancables d'apres la barre de sorts.

    Un sort sans `slot` renseigne est conserve : on ne dispose d'aucune information sur
    lui, et l'ecarter par defaut priverait le planificateur de tout un pan de coups.
    De meme, `ready=None` (barre non lue) laisse tout passer.
    """
    if ready is None:
        return spells
    return [s for s in spells if s.slot is None or s.slot in ready]


def has_line_of_sight(
    board: BoardMap, source: int, target: int, blocked: set[int]
) -> bool:
    """Vrai si aucun obstacle ne se dresse entre `source` et `target` (bornes exclues).

    On echantillonne le segment en coordonnees de reseau, a un pas assez fin pour ne
    manquer aucune case traversee, et on arrondit chaque point vers la case la plus
    proche.

    **Une position ABSENTE du plateau bloque.** Le plateau ne contient que les cases
    jouables detectees ; ce qui n'y figure pas est un mur, un trou, ou du hors-plateau --
    et rien de tout cela ne se traverse. La version precedente laissait passer ces
    positions, d'ou le defaut constate en jeu : le bot tirait a travers les murs, le jeu
    refusait le coup, et le tour etait perdu sans que rien ne le signale.

    Le sens de l'erreur est choisi. La detection de plateau est imparfaite : elle rate
    parfois des cases jouables, qui seront alors prises pour des murs. Le bot renoncera
    donc a des tirs valides -- coup manque, mais tour joue. L'inverse fait perdre le tour
    entier.
    """
    if source == target:
        return True

    start = board.cells[source].astype(float)
    end = board.cells[target].astype(float)
    steps = max(2, int(2 * grid_distance(board, source, target)))

    for step in range(1, steps):
        point = start + (end - start) * (step / steps)
        i, j = int(round(point[0])), int(round(point[1]))
        cell = board.index_at(i, j)
        if cell is None:
            return False          # hors plateau : mur, trou, ou vide
        if cell in (source, target):
            continue
        if cell in blocked:
            return False
    return True


def castable_targets(
    board: BoardMap,
    spell: Spell,
    caster: int,
    blocked: set[int],
    candidates: list[int] | None = None,
) -> list[int]:
    """Cases sur lesquelles `spell` peut etre lance depuis `caster`."""
    pool = range(len(board)) if candidates is None else candidates
    return [
        cell for cell in pool
        if spell.in_range(board, caster, cell)
        and (not spell.needs_line_of_sight
             or has_line_of_sight(board, caster, cell, blocked))
    ]
