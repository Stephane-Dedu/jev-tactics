"""La mesure qui a fait tomber un argument, verifiee avant qu'on s'appuie dessus.

`survey_spacing` repond a « les paires de `data/runs/`, espacees de dizaines de secondes,
majorent-elles le scan reel pris a 0,35 s ? ». La reponse etait AFFIRMEE depuis le debut,
dans trois fichiers, sans avoir jamais ete mesuree -- et elle est fausse : le compte de
candidats ne suit pas l'ecart. C'est sur ce verdict que reposent desormais les commentaires
de `SUSPICIOUS_GROUP_COUNT`, donc il se garde comme un comportement.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def etude():
    spec = importlib.util.spec_from_file_location(
        "study_monsters", ROOT / "scripts" / "study_monsters.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheCaptureInstantIsReadFromTheName:
    """`_seconds` : de l'arithmetique, pas un instant date. Seuls les ECARTS servent, et
    fabriquer un instant obligerait a lui inventer un fuseau."""

    def test_two_captures_a_minute_apart(self, etude):
        assert (etude._seconds("20260807-220329")
                - etude._seconds("20260807-220229")) == 60

    def test_it_crosses_the_hour(self, etude):
        assert (etude._seconds("20260807-220001")
                - etude._seconds("20260807-215921")) == 40

    def test_a_name_that_says_no_time_is_refused(self, etude):
        """Rendre 0 ferait passer un fichier sans horodatage pour une capture prise au
        meme instant que toutes les autres -- soit un ecart nul, la tranche la plus
        courte, exactement celle dont la conclusion depend."""
        assert etude._seconds("capture") is None
        assert etude._seconds("failures-ecran-inconnu") is None


class TestTheSpacingVerdict:
    """Le verdict PLAT/CROISSANT, sur des mesures fabriquees.

    Le tester sur `data/runs/` reviendrait a le faire dependre du contenu du depot ; ce
    qu'on garde ici est la REGLE de lecture, qui doit rester juste quel que soit le fond.
    """

    def test_flat_counts_are_called_flat(self, etude, capsys, monkeypatch):
        self._verdict(etude, monkeypatch, [29, 27, 29])
        assert "PLAT" in capsys.readouterr().out

    def test_a_growing_count_is_called_growing(self, etude, capsys, monkeypatch):
        """CONTRE-EPREUVE : si le compte suivait vraiment l'ecart, l'argument d'origine
        aurait ete bon et l'instrument doit le dire."""
        self._verdict(etude, monkeypatch, [8, 20, 45])
        assert "CROISSANT" in capsys.readouterr().out

    def _verdict(self, etude, monkeypatch, medianes):
        """Fabrique des paires dont chaque tranche porte la mediane voulue."""
        mesures = []
        for (bas, haut), median in zip(etude.TRANCHES, medianes, strict=False):
            mesures.extend([((bas + haut) // 2, median)] * 3)
        monkeypatch.setattr(etude, "_mesures_espacement", lambda _: mesures)
        etude.survey_spacing(Path("inexistant"))


class TestTheUiLeakSurvey:
    """`survey_ui_leak` : reste-t-il de l'INTERFACE hors de `UI_ZONES` ?

    Un element d'interface oublie produit des candidats a chaque frame, toujours au meme
    endroit de l'ECRAN. Chaque clic dessus coute `ENGAGE_TIMEOUT` sans jamais engager, et
    la liste noire ne le retient pas puisqu'elle est rangee par CARTE.

    UN TEMOIN UNIQUE NE SUFFIT PAS, et la premiere version de cette mesure s'y est trompee :
    elle comparait le coin suspect (0,92) a UNE case de terrain choisie a la main (0,75) et
    concluait que rien ne se separait. Sur la grille entiere, les cases deja exclues sont a
    +0,76 de mediane et les autres a +0,11. L'ecart entre deux points choisis ne dit rien de
    l'ecart entre deux populations.
    """

    def _captures(self, tmp_path, monkeypatch, etude, combien, *, bandeau=True,
                  anime=True, terrain_partage=False):
        """`combien` cartes : terrain different a chaque fois, bandeau fixe en option.

        `terrain_partage` recopie un morceau de decor sur les DEUX PREMIERES cartes
        seulement -- ce que font des cartes adjacentes d'une meme zone, qui partagent leur
        tileset. C'est le cas qui distingue `min` de `max` : ce morceau correle
        parfaitement sur une paire et pas du tout sur les autres.
        """
        import cv2
        import numpy as np

        fixe = np.random.default_rng(99).integers(0, 255, size=(240, 300, 3),
                                                  dtype=np.uint8)
        commun = np.random.default_rng(7).integers(0, 255, size=(120, 120, 3),
                                                   dtype=np.uint8)
        # DEUX captures par carte, et le bandeau ANIME entre les deux. Un panneau
        # d'interface qui ne bouge pas ne produit aucun candidat, donc ne pose aucun
        # probleme : ce qui coute, c'est celui qui clignote. La premiere version de ce
        # decor etait statique et ne modelisait donc pas le defaut qu'on cherche.
        for i in range(combien):
            for prise in (0, 1):
                rng = np.random.default_rng(i)
                frame = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
                if bandeau:
                    frame[400:640, 1560:1860] = fixe
                    if prise and anime:
                        frame[430:520, 1600:1740] = 255 - fixe[30:120, 40:180]
                if terrain_partage and i < 2:
                    frame[240:360, 300:420] = commun
                    if prise:
                        frame[270:340, 330:400] = 255 - commun[30:100, 30:100]
                cv2.imwrite(str(tmp_path / f"2026080{i}-12000{prise}.png"), frame)

        # La carte est deduite du coin haut-gauche, different a chaque capture : pas de
        # table a tenir a jour, donc pas de KeyError quand la scene change.
        compteur = {"n": 0}

        def _diagnose(frame, *a, **k):
            # Deux captures par carte, dans l'ordre du tri : 0,0,1,1,2,2...
            compteur["n"] += 1
            carte = ((compteur["n"] - 1) // 2 % max(combien, 1), 0)
            return type("L", (), {
                "position": type("P", (), {"as_tuple": lambda _s, c=carte: c})()})()

        monkeypatch.setattr(etude, "diagnose_position", _diagnose)
        return tmp_path

    def test_it_finds_a_panel_that_ui_zones_misses(self, tmp_path, monkeypatch, etude,
                                                  capsys):
        dossier = self._captures(tmp_path, monkeypatch, etude, 3)
        suspectes = etude.survey_ui_leak(dossier)
        sortie = capsys.readouterr().out
        assert suspectes > 0, sortie
        assert "1590" in sortie or "1650" in sortie, sortie

    def test_terrain_alone_raises_nothing(self, tmp_path, monkeypatch, etude, capsys):
        """CONTRE-EPREUVE : sans bandeau fixe, aucune case ne doit sortir. Une mesure qui
        designe toujours quelque chose ferait exclure du terrain."""
        dossier = self._captures(tmp_path, monkeypatch, etude, 3, bandeau=False)
        assert etude.survey_ui_leak(dossier) == 0
        capsys.readouterr()

    def test_a_still_panel_is_not_reported(self, tmp_path, monkeypatch, etude, capsys):
        """UN PANNEAU QUI NE BOUGE PAS NE COUTE RIEN, et la premiere version le signalait
        quand meme.

        Elle demandait « cette case est-elle hors de `UI_ZONES` et stable ? » -- une
        question a laquelle un panneau statique repond oui, alors qu'il ne produit AUCUN
        candidat et que le bot ne le cliquera jamais. Sur les vraies captures, cela faisait
        75 signalements dont un bloc entier a droite ; en demandant ce que `detect_groups`
        rend REELLEMENT, il en reste 4.

        Ce qui coute, c'est l'interface qui CLIGNOTE.
        """
        dossier = self._captures(tmp_path, monkeypatch, etude, 3, anime=False)
        assert etude.survey_ui_leak(dossier) == 0, capsys.readouterr().out

    def test_terrain_shared_by_two_maps_is_not_ui(self, tmp_path, monkeypatch, etude,
                                                  capsys):
        """POURQUOI C'EST LA PIRE PAIRE QUI COMPTE. Des cartes adjacentes partagent leur
        tileset : un morceau de decor peut etre identique sur deux cartes et absent de la
        troisieme. L'interface, elle, est sur TOUTES.

        Retenir la meilleure paire designerait ce decor ; retenir la pire ne le designe
        pas. Sans ce cas, les deux statistiques rendent le meme verdict et rien ne dit
        laquelle est juste.
        """
        dossier = self._captures(tmp_path, monkeypatch, etude, 3, bandeau=False,
                                 terrain_partage=True)
        assert etude.survey_ui_leak(dossier) == 0, capsys.readouterr().out

    def test_one_map_says_nothing(self, tmp_path, monkeypatch, etude, capsys):
        """Comparer une carte a elle-meme n'a aucun sens : la mesure doit se taire."""
        dossier = self._captures(tmp_path, monkeypatch, etude, 1)
        assert etude.survey_ui_leak(dossier) == 0
        assert "moins de deux cartes" in capsys.readouterr().out
