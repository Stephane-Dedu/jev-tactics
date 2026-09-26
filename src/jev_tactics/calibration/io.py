"""Persistance d'une calibration (homographie + metadonnees) en JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def save_homography(path: str | Path, h: NDArray[np.float64], meta: dict[str, Any] | None = None) -> None:
    data = {"H": np.asarray(h, dtype=float).tolist(), "meta": meta or {}}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_homography(path: str | Path) -> tuple[NDArray[np.float64], dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["H"], dtype=np.float64), data.get("meta", {})
