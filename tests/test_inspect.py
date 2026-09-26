"""Interroger la perception case par case : « pourquoi n'as-tu pas vu ce monstre ? »

Ce module existe a cause d'une enquete qui a pris une iteration entiere -- mesurer a la
main le score de deux cases, les seuils, la connexite, puis les ratios de marqueur a
chaque inset. Le diagnostic final tenait en une phrase. Ces tests verifient que la phrase
sort toute seule, et surtout qu'elle designe le BON etage : un seuil, une connexite ou un
marqueur appellent trois corrections differentes."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from jev_tactics.perception.inspect import BoardInspector, CellDiagnosis

ROOT = Path(__file__).resolve().parents[1]


def _diagnosis(**kwargs) -> CellDiagnosis:
    base = {"lattice": (0, 0), "centre": (100, 100), "on_board": True,
            "line_score": 40.0, "seed_threshold": 31.2, "low_threshold": 15.6,
            "markers": {"ENEMY": 0.0, "ALLY": 0.0}}
    return CellDiagnosis(**{**base, **kwargs})


class TestTheVerdictNamesTheRightStage:
    """Un seuil, une connexite et un marqueur appellent trois corrections OPPOSEES.
    Confondre les trois est exactement ce qui a coute une iteration."""

    def test_a_sprite_covered_cell_is_named_as_such(self):
        """Centre charge : le sprite y laisse ses propres contours."""
        verdict = _diagnosis(on_board=False, line_score=9.3,
                             centre_response=48.1).explain()
        assert "SPRITE" in verdict and "48.1" in verdict

    def test_a_faintly_drawn_cell_is_NOT_blamed_on_a_sprite(self):
        """Le verdict disait « un sprite » dans les deux cas — y compris sur une case dont
        le centre mesure 0,9. Deux causes distinctes, deux corrections distinctes : l'une
        invite a chercher un occupant, l'autre a regarder le rendu du plateau."""
        verdict = _diagnosis(on_board=False, line_score=14.5,
                             centre_response=0.9).explain()
        assert "PEU DESSINEES" in verdict and "sprite" not in verdict.lower()

    def test_above_the_threshold_but_off_board_blames_CONNECTIVITY(self):
        """Le cas reel de combat1.png : score 17,2 pour un seuil bas a 15,6. Ce n'est pas
        le seuil qui rejette la case, c'est qu'elle ne touche rien."""
        verdict = _diagnosis(on_board=False, line_score=17.2).explain()
        assert "CONNEXITE" in verdict

    def test_a_marker_above_the_vote_announces_an_entity(self):
        verdict = _diagnosis(markers={"ENEMY": 0.29, "ALLY": 0.02}).explain()
        assert "ENEMY" in verdict and "devrait etre vue" in verdict

    def test_a_weak_marker_is_distinguished_from_none(self):
        """« 0,12 pour un seuil a 0,20 » et « rien du tout » ne se corrigent pas pareil :
        le premier invite a regarder le seuil, le second la bande de couleur."""
        weak = _diagnosis(markers={"ENEMY": 0.12}).explain()
        empty = _diagnosis(markers={"ENEMY": 0.0}).explain()
        assert "0.12" in weak and "aucun marqueur" in empty

    def test_the_verdict_always_carries_its_numbers(self):
        """Un verdict sans mesure ne permet pas de corriger -- c'est toute la lecon des
        iterations precedentes."""
        for diagnosis in (_diagnosis(on_board=False, line_score=9.3, centre_response=48.0),
                          _diagnosis(on_board=False, line_score=17.2),
                          _diagnosis(markers={"ENEMY": 0.29})):
            assert any(character.isdigit() for character in diagnosis.explain())


class TestBestTeam:
    def test_the_strongest_marker_wins(self):
        assert _diagnosis(markers={"ENEMY": 0.29, "ALLY": 0.12}).best_team[0] == "ENEMY"

    def test_below_the_vote_nobody_wins(self):
        team, ratio = _diagnosis(markers={"ENEMY": 0.12}).best_team
        assert team is None and ratio == 0.12


class TestOnTheRealCapture:
    """La capture qui a motive l'outil. Si ces valeurs changent, c'est que la detection a
    bouge -- et l'outil doit continuer a dire la verite sur ce qu'elle fait."""

    @pytest.fixture
    def inspector(self):
        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent (capture locale, non versionnee)")
        return BoardInspector(frame=cv2.imread(str(path)))

    def test_it_answers_for_a_cell_OUTSIDE_the_board(self, inspector):
        """Tout l'interet : c'est hors plateau que se posent les questions. Refuser d'y
        repondre laisserait l'utilisateur sans explication la ou il en veut une."""
        diagnosis = inspector.diagnose(455, 300)
        assert diagnosis is not None and not diagnosis.on_board

    def test_the_monster_cell_carries_its_marker(self, inspector):
        """J'avais conclu l'inverse en relevant la case A L'OEIL : un sprite isometrique
        est dessine AU-DESSUS de sa case. Cliquer supprime cette erreur."""
        diagnosis = inspector.diagnose(455, 300)
        assert diagnosis.markers["ENEMY"] >= diagnosis.vote_threshold

    def test_it_blames_connectivity_there(self, inspector):
        assert "CONNEXITE" in inspector.diagnose(455, 300).explain()

    def test_the_player_cell_is_seen_normally(self, inspector):
        diagnosis = inspector.diagnose(707, 248)
        assert diagnosis.on_board and diagnosis.best_team[0] == "ALLY"

    def test_the_neighbourhood_shows_the_corridor(self, inspector):
        """Un ilot ne s'explique jamais par sa seule case : c'est le couloir de cases
        faibles qui le coupe du plateau, et il faut le VOIR."""
        around = inspector.neighbourhood(455, 300, radius=2)
        assert len(around) == 25
        assert not any(d.on_board for d in around), "le couloir a disparu"


class TestRobustness:
    def test_a_frame_without_a_grid_is_refused_clearly(self):
        with pytest.raises(ValueError, match="grille"):
            BoardInspector(frame=np.zeros((400, 400, 3), np.uint8))

    def test_a_point_outside_the_lattice_returns_none(self):
        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        inspector = BoardInspector(frame=cv2.imread(str(path)))
        assert inspector.diagnose(-99999, -99999) is None


def _study():
    """Charge scripts/study_cells.py comme un module."""
    import importlib.util
    import sys

    path = ROOT / "scripts" / "study_cells.py"
    if not path.exists():
        pytest.skip("study_cells.py absent")
    argv, sys.argv = sys.argv, ["study_cells"]
    try:
        spec = importlib.util.spec_from_file_location("_study_cells", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def _row(**kwargs) -> dict:
    base = {"label": "enemy", "on_board": True, "line_score": 40.0,
            "low_threshold": 15.6, "markers": {"ENEMY": 0.0, "ALLY": 0.0}}
    return {**base, **kwargs}


class TestTheTwoVocabulariesAgree:
    """L'inspecteur EXPLIQUE une case, l'etude COMPTE les etages sur un lot d'etiquettes.
    S'ils decoupaient les causes differemment, ils raconteraient deux histoires du meme
    defaut -- et l'on ne saurait plus laquelle croire."""

    @pytest.mark.parametrize(("changes", "stage", "phrase"), [
        ({"on_board": False, "line_score": 9.3, "centre_response": 48.0},
         "score de ligne", "sous le seuil bas"),
        ({"on_board": False, "line_score": 17.2}, "connexite", "CONNEXITE"),
        ({"markers": {"ENEMY": 0.29}}, "VUE", "devrait etre vue"),
        ({"markers": {"ENEMY": 0.12}}, "seuil de vote", "0.12"),
        ({"markers": {"ENEMY": 0.0}}, "bande de couleur", "aucun marqueur"),
    ])
    def test_same_cause_same_stage(self, changes, stage, phrase):
        study = _study()
        assert stage in study.stage_of(_row(**changes))
        assert phrase in _diagnosis(**changes).explain()


class TestStudyAnalysis:
    def test_overlap_is_zero_for_disjoint_ranges(self):
        study = _study()
        assert study.overlap([0.27, 0.33], [0.0, 0.06]) == 0.0

    def test_overlap_is_high_for_confused_ranges(self):
        """Un critere dont les plages se chevauchent ne separe rien : le durcir couterait
        des cases manquees sans supprimer les faux positifs."""
        study = _study()
        assert study.overlap([10.0, 40.0], [12.0, 38.0]) > 0.8

    def test_a_truncated_line_does_not_lose_the_rest(self, tmp_path):
        """Une session interrompue laisse une derniere ligne a moitie ecrite."""
        study = _study()
        path = tmp_path / "labels.jsonl"
        good = json.dumps(_row())
        truncated = '{"label": "ene'
        path.write_text(f"{good}\n{good}\n{truncated}", encoding="utf-8")
        assert len(study.load(path)) == 2

class TestCentreResponseSeparatesTwoCauses:
    """La reponse au CENTRE distingue « un sprite masque les aretes » de « les lignes sont
    peu dessinees ». Mesure sur trois captures : une case de plateau vide a une mediane de
    0,7 a 2,2 et un maximum de 24,4 ; les deux monstres de combat1.png y montent a 29,5 et
    48,1. Le seuil est place au-dessus du maximum observe des cases vides.

    LIMITE MESUREE : le personnage, lui, reste a 2,9 — son sprite est plus etroit. Le
    critere separe les GROS occupants, pas tous. Et il ne distingue pas le decor (mediane
    9,5 hors plateau). Il informe, il ne tranche pas."""

    def test_the_real_monster_cell_is_above_the_threshold(self):
        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        from jev_tactics.perception.inspect import EMPTY_CENTRE_MAX

        inspector = BoardInspector(frame=cv2.imread(str(path)))
        assert inspector.diagnose(455, 300).centre_response > EMPTY_CENTRE_MAX

    def test_the_empty_corridor_is_below(self):
        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        from jev_tactics.perception.inspect import EMPTY_CENTRE_MAX

        inspector = BoardInspector(frame=cv2.imread(str(path)))
        assert inspector.diagnose(640, 275).centre_response < EMPTY_CENTRE_MAX

    def test_local_phase_refutes_a_misalignment(self):
        """J'ai cru a un desalignement du reseau dans ce couloir et failli corriger la
        mauvaise chose. La phase LOCALE y vaut 19 : la grille est bien calee."""
        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        inspector = BoardInspector(frame=cv2.imread(str(path)))
        assert inspector.diagnose(640, 275).local_phase > 3.0

