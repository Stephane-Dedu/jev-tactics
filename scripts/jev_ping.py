"""Un seul appel reel a Jev, pour verifier la cle, la forme du SDK et la latence.

    python scripts/jev_ping.py --check      # ne sort PAS sur le reseau
    python scripts/jev_ping.py              # un appel reel (quelques millimes)

A lancer en premier des qu'une cle est posee dans `.env`. Trois choses y sont verifiees, et
chacune peut casser independamment :

  - **la cle** est-elle lue ? Le fichier `.env` etait ignore par git avant que quoi que ce
    soit ne sache le lire ;
  - **la forme de la reponse.** `transport._from_sdk_result` a ete ecrit d'apres la
    documentation, sans jamais voir le SDK. Les noms d'attributs (`result.choices`,
    `.confidence`) sont des SUPPOSITIONS jusqu'a ce premier appel ;
  - **la latence.** `ENGAGE_TIMEOUT` et les budgets de tour sont poses par analogie. Un
    appel par tour ne coute rien en euros -- de l'ordre du quart de centime par combat --
    mais il coute du temps, et c'est cette grandeur qui n'est pas mesuree.

`--check` s'arrete avant l'appel : il dit si la cle est visible et si le SDK est installe,
sans rien consommer. C'est le mode a utiliser pour diagnostiquer une configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from jev_tactics.config import ENV_FILE, describe_key
from jev_tactics.decision.transport import DEFAULT_MODEL, HttpTransport

# Une question minuscule, mais de la MEME FORME que celles du combat : un `choice` sur des
# etiquettes opaques avec une description par option. Si elle passe, la question reelle
# passe aussi -- ce qui ne serait pas garanti avec un « bonjour ».
STATE = {
    "me": {"hp_pct": 35, "ap": 6, "mp": 3},
    "enemies": [{"id": "e0", "distance": 1}, {"id": "e1", "distance": 2}],
    "plans": [
        {"id": "a", "damage_total": 60, "ends_at_distance": 1},
        {"id": "b", "damage_total": 20, "ends_at_distance": 6},
    ],
}

QUESTIONS = {
    "plan": {
        "type": "choice",
        "instructions": (
            "Tu choisis l'intention tactique d'un tour de combat. Le personnage est a "
            "35 % de ses points de vie avec deux ennemis au contact. Reponds uniquement "
            "par l'identifiant."
        ),
        "criteria": {
            "a": "frapper fort, 60 degats, mais finir au contact",
            "b": "se mettre a l'abri, 20 degats, finir a 6 cases",
        },
    }
}


def check() -> int:
    """Diagnostic hors ligne. -> code de sortie."""
    print(f"  cle        : {describe_key()}")
    print(f"  fichier    : {ENV_FILE} "
          f"({'present' if ENV_FILE.exists() else 'ABSENT — cf. .env.example'})")
    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        print('  sdk        : ABSENT — `pip install -e ".[jev]"`')
        return 1
    print("  sdk        : typesafe-sdk installe")
    print("\n  Tout est en place. Relancer sans --check pour un appel reel.")
    return 0


def ping(model: str, timeout: float) -> int:
    transport = HttpTransport(model=model, timeout=timeout)
    if not transport.api_key:
        print(f"  {describe_key()}")
        print(f"  Poser la cle dans {ENV_FILE} (cf. .env.example), ou exporter "
              "TYPESAFE_API_KEY.")
        return 2

    payload = json.dumps(STATE, separators=(",", ":"))
    print(f"  modele     : {model}")
    print(f"  etat       : {len(payload)} caracteres")
    start = time.monotonic()
    try:
        answer = transport(payload, QUESTIONS)
    except Exception as exc:
        # Aucune trace de la cle ici : le message d'une erreur HTTP peut contenir
        # l'en-tete d'autorisation selon les bibliotheques.
        print(f"  ECHEC      : {type(exc).__name__}")
        print(f"  detail     : {str(exc)[:300]}")
        return 1
    elapsed = (time.monotonic() - start) * 1000

    print(f"  latence    : {elapsed:.0f} ms")
    print(f"  reponse    : {json.dumps(answer, ensure_ascii=False)[:400]}")

    plan = answer.get("answers", {}).get("plan", {})
    if not plan:
        print("\n  /!\\ la reponse ne porte pas answers.plan : la forme attendue par")
        print("      transport._from_sdk_result ne correspond pas au SDK reel.")
        return 1
    print(f"\n  choix      : {plan.get('choice')!r}  "
          f"confiance {plan.get('confidence')}")
    if plan.get("choice") not in QUESTIONS["plan"]["criteria"]:
        print("  /!\\ etiquette hors de l'ensemble propose — le decodage est a revoir.")
        return 1
    print("  La forme du SDK est confirmee : le decideur peut etre branche.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verifier cle et SDK SANS appeler (rien n'est consomme)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    print()
    sys.exit(check() if args.check else ping(args.model, args.timeout))


if __name__ == "__main__":
    main()
