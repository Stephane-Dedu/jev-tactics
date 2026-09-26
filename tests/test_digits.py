"""Classifieur maison des chiffres : exact sur les fixtures reelles, et il REJETTE
plutot que deviner (le rejet declenche le fallback Tesseract dans ui.read_stat).
Hermetique : aucun appel Tesseract ici."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.digits import (
    DigitTemplates,
    extract_glyphs,
    read_digits_from_mask,
)
from jev_tactics.perception.ui import _glyph_mask

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*_*.png"))
TEMPLATES = DigitTemplates.load()


def _expected(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])  # {img}_{roi}_{valeur}.png


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.stem for f in FIXTURES])
def test_reads_reference_crops_exactly(fixture):
    crop = cv2.imread(str(fixture))
    assert read_digits_from_mask(_glyph_mask(crop), TEMPLATES) == _expected(fixture)


def test_glyph_count_matches_digit_count():
    crop = cv2.imread(str(FIXTURES[0]))  # dofus_pa_10 -> 2 glyphes
    n_digits = len(str(_expected(FIXTURES[0])))
    assert len(extract_glyphs(_glyph_mask(crop))) == n_digits


def test_rejects_noise_blob():
    # Un blob aleatoire ne doit correspondre a aucun chiffre -> rejet (None), pas un guess.
    rng = np.random.default_rng(42)
    mask = np.zeros((28, 30), dtype=np.uint8)
    mask[6:22, 8:22] = (rng.random((16, 14)) > 0.5).astype(np.uint8) * 255
    assert read_digits_from_mask(mask, TEMPLATES) is None


def test_rejects_empty_mask():
    assert read_digits_from_mask(np.zeros((28, 30), dtype=np.uint8), TEMPLATES) is None


def test_templates_cover_observed_digits():
    # Les chiffres presents dans les fixtures doivent etre couverts ; 3 et 8 sont connus
    # manquants (documente dans build_digit_templates) et passeront par le fallback.
    observed = {int(c) for f in FIXTURES for c in str(_expected(f))}
    assert observed <= TEMPLATES.covered_digits()
