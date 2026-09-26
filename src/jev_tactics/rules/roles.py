"""Roles de sorts : decrire sa barre par ARCHETYPES plutot que sort par sort.

Le probleme que ca resout est celui du passage a l'echelle. Saisir a la main les couts,
portees et degats de chaque sort de chaque personnage est long, refait a chaque
changement de build, et faux des la premiere faute de frappe -- sans que rien ne le dise
(cf. `diagnose_spells`).

L'inversion : au lieu que le bot apprenne TES sorts, tu ranges ta barre selon une
convention qu'il connait deja. « emplacement 1 = corps a corps, emplacement 2 = dps a
distance » se declare en une ligne et vaut pour n'importe quelle classe.

**Difference importante avec `scaffold_spells`.** Le squelette lit la barre et refuse
d'inventer des degats : rien sur la capture ne les montre, une valeur credible y serait
une invention deguisee en mesure. Ici l'information vient de TOI -- tu affirmes qu'un
emplacement est un dps longue portee -- et l'archetype ne fait que traduire cette
affirmation en ordres de grandeur coherents entre eux.

Ce que ca donne, et ce que ca ne donne pas : le solveur compare des coups, donc ce sont
les rapports (un sort longue portee coute plus cher et frappe moins fort qu'un corps a
corps) qui gouvernent ses choix, bien plus que les valeurs absolues. Un jeu de roles
plausible produit donc des decisions sensees. Il ne remplace pas des valeurs exactes
pour autant : les degats affiches restent a corriger si l'on veut que le solveur arbitre
finement entre deux sorts proches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from jev_tactics.rules.spells import Spell


@dataclass(frozen=True)
class SpellRole:
    """Archetype de sort : la forme d'un sort, pas ses valeurs exactes."""

    role: str
    description: str
    ap_cost: int
    range_min: int
    range_max: int
    damage_min: int
    damage_max: int
    needs_line_of_sight: bool = True
    max_casts_per_turn: int = 2
    max_casts_per_target: int = 99
    area_radius: int = 0

    def to_spell(self, slot: int, name: str | None = None) -> Spell:
        fields = asdict(self)
        fields.pop("role")
        fields.pop("description")
        return Spell(name=name or f"{self.role}_{slot}", slot=slot, source="role",
                     **fields)


# Catalogue volontairement court : des formes NETTEMENT distinctes, pour que le solveur
# ait de vrais arbitrages a faire. Multiplier les variantes proches n'ajouterait que du
# temps de recherche.
ROLES: dict[str, SpellRole] = {
    "cac": SpellRole(
        role="cac", description="corps a corps : bon marche, frappe fort, portee 1",
        ap_cost=3, range_min=1, range_max=1, damage_min=28, damage_max=34),
    "dps_courte": SpellRole(
        role="dps_courte", description="degats a courte portee, polyvalent",
        ap_cost=3, range_min=1, range_max=3, damage_min=22, damage_max=27),
    "dps_longue": SpellRole(
        role="dps_longue", description="degats a distance : plus cher, frappe moins fort",
        ap_cost=4, range_min=2, range_max=6, damage_min=18, damage_max=23,
        max_casts_per_target=1),
    "zone": SpellRole(
        role="zone", description="degats de zone : faibles par cible, utiles groupes",
        ap_cost=4, range_min=1, range_max=4, damage_min=12, damage_max=16,
        area_radius=1, max_casts_per_turn=1),
    "finition": SpellRole(
        role="finition", description="coup lourd, un seul lancer par tour",
        ap_cost=5, range_min=1, range_max=4, damage_min=40, damage_max=50,
        max_casts_per_turn=1),
    "soutien": SpellRole(
        role="soutien", description="soin ou buff — NON MODELISE, degats nuls",
        ap_cost=3, range_min=0, range_max=6, damage_min=0, damage_max=0,
        max_casts_per_turn=1),
}


def parse_assignment(text: str) -> dict[int, str]:
    """Lit « 0=cac,2=dps_longue,3=zone » -> {0: 'cac', 2: 'dps_longue', 3: 'zone'}.

    Un role inconnu ou un emplacement illisible leve : une affectation silencieusement
    ignoree donnerait un personnage ampute d'un sort sans que rien ne le signale, la
    panne meme que ce module cherche a eviter.
    """
    assignment: dict[int, str] = {}
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"affectation illisible : {part!r} (attendu « slot=role »)")
        raw_slot, role = (piece.strip() for piece in part.split("=", 1))
        if not raw_slot.isdigit():
            raise ValueError(f"emplacement invalide : {raw_slot!r}")
        if role not in ROLES:
            raise ValueError(
                f"role inconnu : {role!r} (connus : {', '.join(sorted(ROLES))})")
        assignment[int(raw_slot)] = role
    if not assignment:
        raise ValueError("aucune affectation")
    return assignment


def spells_from_roles(assignment: dict[int, str]) -> list[Spell]:
    """Affectation emplacement -> role, traduite en sorts utilisables par le solveur."""
    return [ROLES[role].to_spell(slot) for slot, role in sorted(assignment.items())]


def describe_roles() -> str:
    width = max(len(name) for name in ROLES)
    return "\n".join(
        f"  {name:<{width}}  {role.description}"
        f"  [{role.ap_cost} PA, portee {role.range_min}-{role.range_max}"
        f"{', zone' if role.area_radius else ''}]"
        for name, role in ROLES.items())
