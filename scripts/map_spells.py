"""Associe les sorts d'une config aux EMPLACEMENTS REELS de la barre.

    # 1. Planche de contact : la barre avec ses numeros d'emplacement
    python scripts/map_spells.py --sheet combat1.png --out data/barre_numerotee.png

    # 2. Appliquer l'affectation lue dessus
    python scripts/map_spells.py --spells configs/spells/sacrieur.json \
        --assign "0=Absorption,1=Ravage,2=Hemorragie"

Pourquoi ce script existe : `fetch_spells.py` attribue les `slot` dans l'ORDRE DE LA
CLASSE, pas dans celui de la barre du joueur -- l'API ignore comment chacun range la
sienne. Sans correction, le bot appuie sur la touche d'un sort et en lance un autre.

**L'appariement automatique a ete essaye et MESURE comme insuffisant.** Comparer les
icones de la barre aux icones de reference de dofusdb donne :

    correlation en niveaux de gris : scores 0,18 a 0,61, marges de +0,00 a +0,02
    correlation d'histogramme HSV  : mediane 0,90, mais marge mediane de 0,033
                                     et seulement 17 noms distincts pour 26 cases

Or un sort n'occupe qu'une case : proposer six fois « Hostilite » prouve que la mesure ne
discrimine pas. Les icones du jeu sont petites (46 px), encadrees et re-rendues ; les
references sont des dessins propres de 96 px. Livrer cet appariement aurait produit une
config fausse avec l'apparence d'une config verifiee -- exactement ce que le reste du
projet s'interdit.

Le partage retenu : la machine fait ce qu'elle fait bien (localiser les cases, decouper,
numeroter, reecrire le JSON), l'humain fait ce qu'il fait instantanement et que la
machine rate (reconnaitre une icone).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from jev_tactics.perception.spellbar import (
    COLUMNS,
    ORIGIN_X,
    ORIGIN_Y,
    ROWS,
    SLOT_STEP,
    SlotState,
    read_spell_slots,
)

SCALE = 3


def contact_sheet(frame: np.ndarray) -> np.ndarray:
    """Barre de sorts agrandie, chaque case occupee portant son numero."""
    step = int(SLOT_STEP)
    x0, y0 = ORIGIN_X, ORIGIN_Y
    x1, y1 = x0 + int(COLUMNS * SLOT_STEP) + 4, y0 + int(ROWS * SLOT_STEP) + 4
    if x1 > frame.shape[1] or y1 > frame.shape[0]:
        sys.exit("la barre de sorts depasse de l'image — capture plein ecran attendue ?")

    sheet = cv2.resize(frame[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_NEAREST)
    states = read_spell_slots(frame)
    for index, state in enumerate(states):
        if state is SlotState.EMPTY:
            continue
        row, column = divmod(index, COLUMNS)
        x = int(column * SLOT_STEP) * SCALE
        y = int(row * SLOT_STEP) * SCALE
        # Numero sur fond plein : lisible quelle que soit l'icone dessous.
        cv2.rectangle(sheet, (x + 2, y + 2), (x + 40, y + 24), (0, 0, 0), -1)
        cv2.putText(sheet, str(index), (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(sheet, (x, y), (x + step * SCALE, y + step * SCALE),
                      (0, 255, 255), 1)
    return sheet


def apply_assignment(spells_path: Path, assignment: dict[int, str]) -> None:
    """Reecrit les `slot` d'une config. Un nom inconnu leve plutot que d'etre ignore."""
    entries = json.loads(spells_path.read_text(encoding="utf-8"))
    by_name = {e["name"].lower(): e for e in entries}

    unknown = [name for name in assignment.values() if name.lower() not in by_name]
    if unknown:
        sys.exit(f"sorts inconnus dans {spells_path.name} : {unknown}\n"
                 f"connus : {', '.join(sorted(e['name'] for e in entries))}")

    # Retirer d'abord TOUS les slots : ceux qui ne sont pas affectes ne doivent pas
    # garder un numero herite de l'ordre de la classe, qui serait faux.
    for entry in entries:
        entry["slot"] = None
    for slot, name in assignment.items():
        by_name[name.lower()]["slot"] = slot

    spells_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    placed = sorted(assignment.items())
    print(f"{len(placed)} sorts places dans {spells_path}")
    for slot, name in placed:
        print(f"   emplacement {slot:>2} : {name}")
    orphans = [e["name"] for e in entries if e["slot"] is None]
    if orphans:
        print(f"\n{len(orphans)} sorts sans emplacement — supposes toujours disponibles, "
              f"donc jamais lances faute de raccourci :")
        print(f"   {', '.join(orphans)}")


def parse_assignment(text: str) -> dict[int, str]:
    assignment: dict[int, str] = {}
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"affectation illisible : {part!r} (attendu « slot=NomDuSort »)")
        raw, name = (piece.strip() for piece in part.split("=", 1))
        if not raw.isdigit():
            sys.exit(f"emplacement invalide : {raw!r}")
        assignment[int(raw)] = name
    if not assignment:
        sys.exit("aucune affectation")
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", metavar="CAPTURE",
                    help="capture PLEIN ECRAN en combat -> planche numerotee")
    ap.add_argument("--out", default="data/barre_numerotee.png")
    ap.add_argument("--spells", help="config a corriger")
    ap.add_argument("--assign", help='"0=Absorption,1=Ravage,..."')
    args = ap.parse_args()

    if args.sheet:
        frame = cv2.imread(args.sheet)
        if frame is None:
            sys.exit(f"image illisible : {args.sheet}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), contact_sheet(frame))
        occupied = sum(1 for s in read_spell_slots(frame) if s is not SlotState.EMPTY)
        print(f"{occupied} emplacements occupes -> {out}")
        print("Ouvrir l'image, lire les numeros, puis :")
        print('  python scripts/map_spells.py --spells <config> --assign "0=Nom,1=Nom"')
        return

    if args.spells and args.assign:
        apply_assignment(Path(args.spells), parse_assignment(args.assign))
        print("\nVerifier en combat :")
        print(f"  python scripts/spells.py --live --spells {args.spells}")
        return

    ap.error("preciser --sheet CAPTURE, ou --spells CONFIG --assign \"...\"")


if __name__ == "__main__":
    main()
