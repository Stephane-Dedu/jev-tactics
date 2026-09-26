"""Transport vers Jev, et son double enregistre.

Le depot compte ~1 400 tests deterministes qui ne demandent ni le jeu ni le reseau. Un
decideur distant met fin a cette propriete s'il est appele directement : la documentation
de Jev n'expose ni graine, ni temperature, ni mode deterministe, donc deux executions du
meme test peuvent diverger -- et une panne reseau ferait echouer une suite qui ne teste
pas le reseau.

D'ou l'indirection. `JevDecider` ne connait qu'un appelable ; le choix entre appel reel
et reponse enregistree se fait au montage. Les tests ne posent aucune condition sur la
presence d'une cle API, parce qu'ils n'en ont jamais besoin.

La cle d'enregistrement est un hachage de la REQUETE (etat + questions). Un refactor de
la projection change l'etat, donc la cle, donc l'absence d'enregistrement : le test dit
« cassette manquante » au lieu de rejouer une reponse qui ne correspond plus a ce qu'on
demande aujourd'hui. Un enregistrement perime est pire qu'aucun -- il valide l'ancien code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "jev-latest"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class Transport(Protocol):
    def __call__(self, state: str, questions: dict[str, Any]) -> dict[str, Any]: ...


def cassette_key(state: str, questions: dict[str, Any]) -> str:
    blob = json.dumps({"state": state, "questions": questions},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class CassetteMissing(LookupError):
    """Aucune reponse enregistree pour cette requete."""


class ReplayTransport:
    """Rejoue des reponses enregistrees. Aucun reseau, aucune cle API.

    C'est le transport utilise en test et en CI.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists() else {}
        )

    def __call__(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        key = cassette_key(state, questions)
        if key not in self._data:
            raise CassetteMissing(
                f"cassette {key} absente de {self.path}. Reenregistrer avec "
                f"RecordingTransport, ou verifier que la projection n'a pas change.")
        return self._data[key]


class RecordingTransport:
    """Appelle un transport reel et consigne la reponse, pour la rejouer ensuite."""

    def __init__(self, inner: Transport, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self._data: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists() else {}
        )

    def __call__(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        key = cassette_key(state, questions)
        if key not in self._data:
            self._data[key] = self.inner(state, questions)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        return self._data[key]


class HttpTransport:
    """Appel reel, via le SDK officiel `typesafe-sdk`.

    Importe A L'APPEL et non au chargement du module : le SDK est une dependance
    optionnelle, et le depot doit s'installer et se tester sans lui -- comme il le fait
    deja pour `pydirectinput` (pilotage) ou `pytesseract` (OCR).
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 timeout: float = 2.0, max_retries: int = 2):
        self.model = model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

    def __call__(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        try:
            from typesafe_sdk import RetryPolicy, TypeSafeClient
        except ImportError as exc:  # pragma: no cover - depend de l'environnement
            raise RuntimeError(
                "typesafe-sdk absent : `pip install 'jev-tactics[jev]'` pour les appels "
                "reels. Les tests utilisent ReplayTransport et n'en ont pas besoin."
            ) from exc

        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY non definie.")

        client = TypeSafeClient(api_key=self.api_key)
        result = client.system_one(
            state,
            _to_sdk_questions(questions),
            model=self.model,
            retry=RetryPolicy(max_retries=self.max_retries, timeout=self.timeout),
        )
        return _from_sdk_result(result)


def _to_sdk_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Du dict brut (forme HTTP) vers les objets du SDK.

    Le depot manipule la forme HTTP partout -- c'est elle qu'on enregistre, hache et
    relit -- et ne convertit qu'au bord. Une cassette reste ainsi lisible et independante
    de la version du SDK.
    """
    from typesafe_sdk import Choice, Noul, Score

    built: dict[str, Any] = {}
    for name, q in questions.items():
        kind = q["type"]
        if kind == "choice":
            built[name] = Choice(instructions=q["instructions"], criteria=q["criteria"])
        elif kind == "score":
            built[name] = Score(instructions=q["instructions"], criteria=q["criteria"])
        elif kind == "noul":
            built[name] = Noul(instructions=q["instructions"])
        else:
            raise ValueError(f"type de question inconnu : {kind}")
    return built


def _from_sdk_result(result: Any) -> dict[str, Any]:
    """Normalise la reponse du SDK vers la forme HTTP `{"answers": {...}}`."""
    answers: dict[str, Any] = {}
    for name, ans in getattr(result, "choices", {}).items():
        answers[name] = {"type": "choice", "choice": ans.choice,
                         "confidence": ans.confidence,
                         "probabilities": dict(getattr(ans, "probabilities", {}) or {})}
    for name, ans in getattr(result, "scores", {}).items():
        answers[name] = {"type": "score", "score": ans.score,
                         "confidence": ans.confidence}
    for name, ans in getattr(result, "nouls", {}).items():
        answers[name] = {"type": "noul", "noul": ans.noul}
    return {"model": getattr(result, "model", DEFAULT_MODEL), "answers": answers}
