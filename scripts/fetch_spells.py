"""Recupere les sorts d'une classe depuis l'API publique dofusdb.fr.

    python scripts/fetch_spells.py --breed 11 --out configs/spells/sacrieur.json
    python scripts/fetch_spells.py --list-breeds

Pourquoi cette source plutot qu'une saisie a la main : dofusdb expose les DONNEES DU JEU
(cout en PA, portees, plafonds de lancer, zones), pas une retranscription. C'est donc du
`source: "dofusdb"` -- ni « mesure » (rien n'a ete releve sur l'ecran de l'utilisateur),
ni « saisie » (personne n'a recopie), mais une reference verifiable et reproductible.

**Limite importante, qui change ce qu'on peut en attendre.** Les degats renvoyes sont les
degats de BASE du sort, avant caracteristiques, dommages et resistances. Le personnage en
inflige beaucoup plus. Ce n'est pas genant ici : le solveur COMPARE des coups, et les
rapports entre sorts sont conserves. Ca le devient si l'on veut predire les PV restants
d'une cible -- ce que le bot ne fait pas (`hp_known=False` sur les ennemis).

Politesse : les reponses sont mises en cache sur disque, et un delai separe les requetes.
Un fichier de sorts ne change qu'a un reequilibrage ; le re-telecharger a chaque essai
serait du gaspillage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.dofusdb.fr"
USER_AGENT = "dofus-bot-research/0.1 (projet de recherche personnel)"
REQUEST_DELAY = 0.3

CACHE = Path(__file__).resolve().parents[1] / "data" / "dofusdb"

# Effets de degats directs, par element. Les « vols » (91-95) soignent en plus, mais
# infligent les memes degats -- ce que le solveur modelise, c'est le degat.
DAMAGE_EFFECTS = {
    91: "eau", 92: "terre", 93: "air", 94: "feu", 95: "neutre",       # vols
    96: "eau", 97: "terre", 98: "air", 99: "feu", 100: "neutre",      # degats directs
}


def element(effects: list[dict]) -> str | None:
    """Element du premier effet de degat. Determinant pour un build specialise."""
    for effect in effects:
        found = DAMAGE_EFFECTS.get(effect.get("effectId"))
        if found:
            return found
    return None

BREEDS = {
    1: "feca", 2: "osamodas", 3: "enutrof", 4: "sram", 5: "xelor", 6: "ecaflip",
    7: "eniripsa", 8: "iop", 9: "cra", 10: "sadida", 11: "sacrieur", 12: "pandawa",
    13: "roublard", 14: "zobal", 15: "steamer", 16: "eliotrope", 17: "huppermage",
    18: "ouginak", 19: "forgelance",
}


def fetch(path: str) -> dict:
    """GET JSON, avec cache disque. Le cache evite de solliciter le site pour rien."""
    cache_file = CACHE / (path.replace("/", "_").replace("?", "_") + ".json")
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    request = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(REQUEST_DELAY)
    return data


def damage_range(effects: list[dict]) -> tuple[int, int]:
    """Fourchette de degats d'un niveau de sort, tous elements confondus.

    Un sort multi-elements cumule ses effets ; on les additionne. Un sort sans effet de
    degat rend (0, 0) -- il sera signale comme non modelise plutot qu'invente.
    """
    low = high = 0
    for effect in effects:
        if effect.get("effectId") not in DAMAGE_EFFECTS:
            continue
        # diceNum = borne basse, diceSide = borne haute (0 quand la valeur est fixe).
        minimum = int(effect.get("diceNum") or 0)
        maximum = int(effect.get("diceSide") or 0) or minimum
        low += minimum
        high += max(minimum, maximum)
    return low, high


# Formes de zone du jeu (`zoneDescr.shape`, en code ASCII). Relevees sur les sorts
# Sacrieur ; les deux dernieres n'ont pas ete verifiees et sont approximees par un cercle,
# ce qui SURESTIME la zone -- signale plutot que masque.
ZONE_SHAPES = {
    ord("P"): "point",       # cible unique
    ord("C"): "circle",      # cercle
    # /!\ Les lettres decrivent l'apparence A L'ECRAN, pas les coordonnees du reseau.
    # En vue isometrique les deux sont echangees, ce qui les rend faciles a inverser --
    # et je les avais inversees. Demonstration, avec e_x=(46,23) et e_y=(-46,23) :
    #
    #   voisins ADJACENTS du reseau (distance 1) :
    #     i+1 -> bas-droite   i-1 -> haut-gauche
    #     j+1 -> bas-gauche   j-1 -> haut-droite     => forment un X a l'ecran
    #
    #   voisins DIAGONAUX du reseau (distance 2) :
    #     i+1,j-1 -> droite   i-1,j+1 -> gauche
    #     i+1,j+1 -> bas      i-1,j-1 -> haut        => forment un + a l'ecran
    #
    # Donc « + » touche les cases a distance de reseau 2 en diagonale, et « X » les
    # voisines immediates. Constate en jeu : Decimation (« + ») « tapait dans le vide »
    # -- le bot visait une case en croyant toucher sa voisine, alors que la zone porte
    # sur les diagonales.
    ord("+"): "diagonal",
    ord("X"): "cross",
    ord("G"): "circle",      # non verifie
    ord("U"): "circle",      # non verifie
}


def area_shape(effects: list[dict]) -> str:
    """Forme de zone du premier effet de degat. « circle » par defaut."""
    for effect in effects:
        if effect.get("effectId") not in DAMAGE_EFFECTS:
            continue
        shape = (effect.get("zoneDescr") or {}).get("shape")
        return ZONE_SHAPES.get(shape, "circle")
    return "circle"


def area_radius(effects: list[dict]) -> int:
    """Rayon d'effet, lu sur la zone du premier effet de degat qui en declare une.

    `rawZone` encode la forme ET la taille (« Cc2 » = cercle de rayon 2). On ne retient
    que le rayon, seule notion que le solveur modelise ; une croix ou une ligne y sont
    approximees par un cercle, ce qui SURESTIME la zone. Assume et signale : mieux vaut
    une zone trop large -- le solveur y verra des cibles qui n'y sont pas et sera decu --
    qu'une zone trop etroite, qui lui ferait ignorer des coups groupes.
    """
    for effect in effects:
        if effect.get("effectId") not in DAMAGE_EFFECTS:
            continue
        zone = effect.get("zoneDescr") or {}
        if ZONE_SHAPES.get(zone.get("shape")) == "point":
            return 0
        return min(int(zone.get("param1") or 0), 6)
    return 0


def convert(spell: dict, level: dict, index: int) -> dict | None:
    """Sort dofusdb + niveau -> entree de configuration du bot."""
    low, high = damage_range(level.get("effects") or [])
    name = (spell.get("name") or {}).get("fr") or f"sort_{spell.get('id')}"
    ap_cost = int(level.get("apCost") or 0)
    if ap_cost <= 0:
        return None  # sort passif ou sans cout : rien a planifier

    return {
        "name": name,
        "source": "dofusdb",
        "slot": index,
        "ap_cost": ap_cost,
        "range_min": int(level.get("minRange") or 0),
        "range_max": int(level.get("range") or 0),
        "needs_line_of_sight": bool(level.get("castTestLos", True)),
        "cast_in_line": bool(level.get("castInLine", False)),
        "cast_in_diagonal": bool(level.get("castInDiagonal", False)),
        "max_casts_per_turn": int(level.get("maxCastPerTurn") or 0) or 99,
        "max_casts_per_target": int(level.get("maxCastPerTarget") or 0) or 99,
        "damage_min": low,
        "damage_max": high,
        "area_radius": area_radius(level.get("effects") or []),
        "area_shape": area_shape(level.get("effects") or []),
        "element": element(level.get("effects") or []),
    }


def collect(breed_id: int, max_level: int | None,
            extra: list[int] | None = None) -> list[dict]:
    breed = fetch(f"breeds/{breed_id}?lang=fr")
    spell_ids = list(breed.get("breedSpellsId") or [])
    # `breedSpellsId` n'est pas exhaustif : Decimation (id 12731) est un sort Sacrieur
    # absent de la liste, decouvert parce qu'il figurait dans la barre du joueur. D'ou
    # `--spell-id`, plutot que de supposer la liste complete.
    spell_ids += [i for i in (extra or []) if i not in spell_ids]
    print(f"classe {(breed.get('shortName') or {}).get('fr', breed_id)} : "
          f"{len(spell_ids)} sorts", file=sys.stderr)

    entries: list[dict] = []
    for spell_id in spell_ids:
        try:
            spell = fetch(f"spells/{spell_id}?lang=fr")
        except urllib.error.HTTPError as error:
            print(f"  sort {spell_id} indisponible ({error.code})", file=sys.stderr)
            continue

        levels = spell.get("spellLevels") or []
        if not levels:
            continue
        # Le dernier niveau est le plus haut ; c'est celui qu'un personnage joue.
        chosen = None
        for level_id in reversed(levels):
            level = fetch(f"spell-levels/{level_id}?lang=fr")
            if max_level is None or int(level.get("minPlayerLevel") or 0) <= max_level:
                chosen = level
                break
        if chosen is None:
            continue

        entry = convert(spell, chosen, len(entries))
        if entry is not None:
            entries.append(entry)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breed", type=int, default=11, help="identifiant de classe")
    ap.add_argument("--out", help="fichier de sortie (sinon : affichage)")
    ap.add_argument("--max-level", type=int,
                    help="ne garder que les niveaux de sort accessibles a ce niveau de "
                         "personnage")
    ap.add_argument("--spell-id", type=int, nargs="*", default=[],
                    help="ids de sorts a ajouter (breedSpellsId n'est pas exhaustif)")
    ap.add_argument("--list-breeds", action="store_true")
    args = ap.parse_args()

    if args.list_breeds:
        for identifier, name in sorted(BREEDS.items()):
            print(f"  {identifier:>2}  {name}")
        return

    entries = collect(args.breed, args.max_level, args.spell_id)
    if not entries:
        sys.exit("aucun sort exploitable recupere")

    payload = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"
    if not args.out:
        print(payload)
        return

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} existe deja — refus d'ecraser")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")

    print(f"{len(entries)} sorts -> {out}")
    print("\nLes `slot` sont attribues DANS L'ORDRE DE LA CLASSE, pas celui de ta barre.")
    print("A corriger : ce qui compte est la position reelle, sinon le bot lancera le")
    print("mauvais sort. Verifier avec :")
    print(f"  python scripts/spells.py --live --spells {out}")


if __name__ == "__main__":
    main()
