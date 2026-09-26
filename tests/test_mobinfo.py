"""Infobulle de monstre : les PV exacts d'un ennemi, que rien d'autre ne donne."""

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from jev_tactics.perception.mobinfo import (
    MobReading,
    glyph_groups,
    has_tooltip,
    load_templates,
    read_mob_hp,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = "data/runs/20260807-224524.png"
ROI = (740, 820, 836, 845)


def _frame(name=CAPTURE):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} absent")
    return cv2.imread(str(path))


class TestReadingEnemyHealth:
    """Ce que cette lecture vaut, chiffre. Sur les 32 captures du depot, 72 ennemis sont
    detectes et AUCUN n'a de PV connus : ni l'OCR (reserve au joueur) ni la timeline (un
    ratio sans maximum) ne les fournit. Sans eux, `evaluate` ne peut ni plafonner les
    degats ni accorder la prime de mise a mort -- le terme le plus lourd du score.

        PV ennemis connus      85,8 %      70,0 % a trois ennemis
        PV ennemis inconnus    36,7 %      50,0 %

    Quarante-neuf points, contre les deux ou trois que rapporte le meilleur ajustement de
    ponderation mesure jusqu'ici.
    """

    def test_the_real_tooltip_is_read(self):
        reading = read_mob_hp(_frame(), ROI)
        assert (reading.hp, reading.hp_max) == (139, 186), reading.reason

    def test_the_two_numbers_are_segmented_apart(self):
        """Le courant et le maximum sont separes par leur ESPACEMENT, pas par leur
        couleur : l'un est blanc, l'autre gris, mais la forme est la meme et un lecteur
        cale sur la teinte casserait au premier theme different."""
        groups = glyph_groups(_frame(), ROI)
        assert [len(g) for g in groups] == [3, 3]


class TestRefusals:
    """Tout-ou-rien, comme les trois autres lecteurs de chiffres du projet. Un maximum de
    PV faux ferait croire un monstre mourant ou invulnerable, et la prime de mise a mort
    deciderait sur cette base -- c'est le pire endroit ou se tromper."""

    def test_an_empty_roi_is_refused_with_a_reason(self):
        reading = read_mob_hp(_frame(), (0, 0, 5, 5))
        assert reading.hp is None and "attendus" in reading.reason

    def test_a_frame_without_a_tooltip_yields_nothing(self):
        reading = read_mob_hp(np.zeros((200, 200, 3), dtype=np.uint8), (0, 0, 150, 40))
        assert reading.hp is None

    def test_missing_digits_are_named(self):
        """Couverture actuelle {1, 3, 6, 8, 9} : une seule capture. Le refus doit DIRE
        lesquels manquent, sinon on cherche la panne au mauvais endroit -- c'est ce que
        les gabarits de coordonnees ont deja coute une fois."""
        missing = sorted(set(range(10)) - load_templates().covered_digits())
        assert missing, "alphabet complet : mettre a jour ce test"
        reading = MobReading(reason=f"chiffre rejete — gabarits MANQUANTS : {missing}")
        assert "MANQUANTS" in reading.reason

    def test_an_incoherent_pair_is_rejected(self):
        assert MobReading(reason="200/100 : incoherent").hp is None


class TestTheTooltipSitsAtAFixedPlace:
    """Je croyais l'infobulle attachee au curseur, donc la ROI inutilisable telle quelle.

    Mesure : sur les deux captures du depot qui en portent une, l'icone de coeur tombe au
    MEME pixel (733, 825), correlation 1,000. Le jeu la place a un endroit fixe.

    Un gabarit de coeur reste souhaitable comme ancre, mais il n'est pas gratuit : le meme
    coeur apparait dans le CHAT (« -99 PV » sur bug.png), a 0,899 de correlation. Dix
    centiemes de marge, la ou le crane du chat en a trente-six.
    """

    def test_the_heart_is_at_the_same_pixel_in_both(self):
        reference = _frame()[825:841, 733:753]
        other = _frame("data/runs/20260807-221715.png")[825:841, 733:753]
        assert float(np.abs(reference.astype(int) - other.astype(int)).mean()) < 3.0

    def test_the_second_tooltip_segments_in_the_same_roi(self):
        """La preuve utile : la MEME ROI decoupe deux nombres sur les deux captures."""
        groups = glyph_groups(_frame("data/runs/20260807-221715.png"), ROI)
        assert [len(g) for g in groups] == [3, 3]


class TestAGroupCanBeSkipped:
    """Sur data/runs/20260807-221715.png, le maximum « 200 » est en GRIS et son « 2 » se
    fragmente en deux composantes de 3 et 4 px. Balayage du seuil de clarte -- 150, 130,
    110, 90 -- aucun ne le recolle : a 110 le groupe voisin fusionne en 14 px. Le gris de
    cette police ne segmente pas.

    Sa valeur COURANTE, elle, est en blanc et segmente parfaitement : [6, 6, 6]. Il serait
    absurde de perdre ses trois chiffres parce que son voisin resiste, d'ou l'etiquette
    vide du constructeur.

        avant   {1, 3, 6, 8, 9}
        apres   {0, 1, 2, 3, 6, 8, 9}
    """

    def test_the_alphabet_grew(self):
        covered = load_templates().covered_digits()
        assert {0, 2} <= covered

    def test_the_first_capture_still_reads(self):
        """Le controle qui compte : ajouter des exemplaires ne doit pas casser ce qui
        marchait. C'est exactement ce qui s'etait produit avec une etiquette fausse."""
        reading = read_mob_hp(_frame(), ROI)
        assert (reading.hp, reading.hp_max) == (139, 186), reading.reason

    def test_the_unsegmentable_one_refuses_rather_than_guesses(self):
        reading = read_mob_hp(_frame("data/runs/20260807-221715.png"), ROI)
        assert reading.hp is None and reading.reason


class TestTheTooltipTellsAMonsterFromScenery:
    """Ce que la CHASSE doit trancher n'est pas « combien de PV » mais « y a-t-il un
    monstre sous le curseur ». Deux questions, deux couts :

        survoler un candidat, puis lire        0,35 s     TOOLTIP_DELAY
        cliquer un candidat, puis constater    8 s        ENGAGE_TIMEOUT

    Vingt fois moins cher — et le journal de la premiere session reelle ne contient QUE
    des clics rates. La sonde de mouvement, elle, rend 20 a 49 candidats par frame sur les
    captures du depot : elle ne separe rien.

    MESURE, sur les 24 captures de `data/runs/` :

        22 sans survol   ->  0 nombre     aucun faux positif
         2 avec survol   ->  2 nombres

    Le predicat ne CLASSE aucun glyphe, a dessein : `read_mob_hp` refuse une des deux
    infobulles faute des gabarits 4, 5 et 7. S'en servir pour la question de la chasse
    ferait passer un monstre pour du decor a cause d'un gabarit absent.
    """

    SANS_SURVOL: ClassVar[list[str]] = [
        "data/runs/20260807-215714.png", "data/runs/20260807-220257.png",
        "data/runs/20260807-222431.png", "data/runs/20260807-224545.png"]

    def test_a_hovered_monster_is_seen(self):
        assert has_tooltip(_frame(), ROI)

    def test_it_does_not_need_the_digits_to_be_readable(self):
        """LE point. Cette capture porte une infobulle que `read_mob_hp` REFUSE — le
        maximum « 200 » n'y est pas segmentable. Le predicat doit quand meme la voir :
        un monstre reste un monstre quand ses PV sont illisibles."""
        frame = _frame("data/runs/20260807-221715.png")
        assert read_mob_hp(frame, ROI).hp is None
        assert has_tooltip(frame, ROI), (
            "un gabarit manquant ferait prendre un monstre pour du decor")

    @pytest.mark.parametrize("nom", SANS_SURVOL)
    def test_nothing_hovered_reads_as_nothing(self, nom):
        """La moitie qui compte. Un predicat toujours vrai ferait cliquer tout le decor —
        exactement ce que la sonde de mouvement fait deja."""
        assert not has_tooltip(_frame(nom), ROI)

    def test_the_corpus_does_not_separate_one_number_from_two(self):
        """Ce que la mesure NE dit PAS, ecrit pour qu'on ne le prenne pas pour acquis.

        `minimum=2` est un raisonnement -- un glyphe clair isole tromperait un seuil a un
        -- et non un releve : sur ces 24 captures les deux valeurs rendent le meme verdict.
        Le jour ou l'infobulle hors combat ne portera qu'un seul nombre, c'est ce
        raisonnement qu'il faudra revoir, pas ce test.
        """
        dossier = ROOT / "data" / "runs"
        if not dossier.is_dir():
            pytest.skip("data/runs absent")
        frames = [cv2.imread(str(f)) for f in sorted(dossier.glob("*.png"))]
        un = [has_tooltip(f, ROI, minimum=1) for f in frames]
        deux = [has_tooltip(f, ROI, minimum=2) for f in frames]
        assert un == deux, ("le corpus separe desormais les deux seuils : la remarque "
                            "ci-dessus est perimee, mesurer et choisir")

    def test_the_whole_repository_agrees(self):
        """Les 24 captures d'un coup : deux positives, vingt-deux negatives. C'est ce
        rapport qui fait du survol un discriminant et non une deuxieme sonde bruyante."""
        dossier = ROOT / "data" / "runs"
        if not dossier.is_dir():
            pytest.skip("data/runs absent")
        frames = sorted(dossier.glob("*.png"))
        vues = [f.name for f in frames if has_tooltip(cv2.imread(str(f)), ROI)]
        assert len(vues) == 2, f"attendu 2 survols sur {len(frames)}, vu {vues}"
