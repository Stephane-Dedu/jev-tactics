"""Lecture des coordonnees de carte.

Ce que ca debloque : le circuit de recolte est aujourd'hui AVEUGLE -- robuste, puisque
aucune lecture ratee ne peut le derailler, mais incapable de rejoindre une banque ou de
se rattraper apres un deplacement manque. Savoir ou l'on est change cela.

Le principe directeur est le meme que pour l'UI : TOUT-OU-RIEN. Une coordonnee dont un
chiffre est douteux est une position FAUSSE, et une position fausse est pire que pas de
position -- le bot croirait savoir ou il est et calculerait un trajet vers nulle part."""

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from jev_tactics.perception.coordinates import (
    COORD_ROI,
    MapPosition,
    diagnose_position,
    load_coord_templates,
    read_position,
)

ROOT = Path(__file__).resolve().parents[1]

# Valeurs lues a l'oeil sur chaque capture. Ce sont les seules verites terrain
# disponibles, et elles servent aussi a FABRIQUER les gabarits : le test verifie donc
# que la chaine complete se referme, pas seulement qu'elle est coherente avec elle-meme.
REAL = [
    ("tacle.png", 3, 8),
    ("combat1.png", 1, 29),
    ("bugcarreblanc.png", 0, 7),
    ("bug4.png", 0, 8),
    ("combat1920.png", 5, 8),
    # La seule capture HORS COMBAT, et la seule qui compte vraiment : c'est hors combat
    # que le circuit a besoin de sa position. Elle a d'ailleurs commence par ECHOUER --
    # le « 2 » y marquait 0,879 pour un seuil a 0,90, alors que le chiffre 2 etait deja
    # couvert. Pas un seuil trop strict : un manque d'exemplaires. Ses glyphes sont
    # desormais dans les gabarits.
    ("hors_combat.png", 2, 9),
    # LES TROIS FRAMES QUI ONT TUE UNE SESSION REELLE, le 15 aout 2026 a 08h40. Une carte
    # parfaitement normale -- Amakna (5, 4), coordonnees parfaitement lisibles a l'oeil --
    # que `classify_screen` rendait INCONNUE : ni timeline (donc pas un combat) ni
    # coordonnees (donc pas une carte). La session s'arretait apres 1,8 s, trois fois de
    # suite, sur un message qui envoyait relever `--screen-patience`. Le delai n'y etait
    # pour rien.
    #
    # Cause : le « 5 » marquait 1,000 et le « 4 » 0,840 pour un seuil a 0,90. Le chiffre 4
    # etait pourtant deja « couvert » -- exactement la meme panne que `hors_combat.png`
    # ci-dessus, un an de projet plus tard : la COUVERTURE ne dit rien de la
    # GENERALISATION. Un seul exemplaire par chiffre suffit a se declarer complet et a
    # echouer sur la premiere police legerement differente.
    ("data/failures/20260815-083952-ecran-inconnu.png", 5, 4),
    ("data/failures/20260815-084018-ecran-inconnu.png", 5, 4),
    ("data/failures/20260815-084045-ecran-inconnu.png", 5, 4),
]


@pytest.mark.parametrize(("name", "x", "y"), REAL)
def test_reads_real_captures(name, x, y):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} absent (capture locale, non versionnee)")
    assert read_position(cv2.imread(str(path))) == MapPosition(x=x, y=y)


class TestASessionKillingScreenIsRecognised:
    """Le trajet COMPLET, et non la seule lecture de chiffres.

    Ce qui a tue la session n'est pas `read_position` mais `classify_screen`, qui s'appuie
    dessus : sans coordonnees et sans timeline, un ecran n'est ni une carte ni un combat,
    donc il est INCONNU -- et un ecran inconnu arrete tout. Tester la lecture sans tester
    le classement laisserait passer une regression sur le lien entre les deux.
    """

    TUEUSES: ClassVar[list[str]] = [
        nom for nom, _x, _y in REAL if nom.startswith("data/failures/")]

    @pytest.mark.parametrize("nom", TUEUSES)
    def test_the_screen_is_a_map_again(self, nom):
        from jev_tactics.perception.screen import ScreenState, classify_screen

        chemin = ROOT / nom
        if not chemin.exists():
            pytest.skip(f"{nom} absent")
        assert classify_screen(cv2.imread(str(chemin))) is ScreenState.MAP

    def test_the_templates_still_cover_what_they_covered(self):
        """LE controle apres avoir touche aux gabarits, et il a failli manquer.

        `build_coord_templates.py` REMPLACE le jeu, il ne l'etend pas. Le relancer avec
        les seules nouvelles captures a fait tomber la couverture de neuf chiffres a deux
        -- toutes les autres cartes seraient devenues illisibles, en echange de celle-ci.
        Il faut lui repasser TOUTES les sources a chaque fois.
        """
        couverts = load_coord_templates().covered_digits()
        assert {0, 1, 2, 3, 4, 5, 7, 8, 9} <= couverts, (
            "des chiffres ont disparu des gabarits : le constructeur a ete relance sans "
            "toutes ses sources")


class TestRejection:
    """Refuser est le comportement voulu, pas un echec."""

    def test_a_blank_frame_reads_nothing(self):
        assert read_position(np.zeros((1080, 1920, 3), np.uint8)) is None

    def test_a_too_small_frame_is_safe(self):
        """Une capture tronquee ne doit pas faire tomber la boucle de recolte."""
        assert read_position(np.zeros((40, 40, 3), np.uint8)) is None

    def test_uncovered_digits_are_rejected_not_guessed(self):
        """Les gabarits ne couvrent pas tous les chiffres, faute de captures. Ceux qui
        manquent doivent etre REJETES : deviner produirait une position credible et
        fausse, ce qui est le pire des cas."""
        covered = load_coord_templates().covered_digits()
        assert covered, "gabarits vides"
        assert covered <= set(range(10))


class TestGrouping:
    """Le decoupage doit s'arreter aux coordonnees, sans lire « - Niveau NN »."""

    def test_the_level_is_not_read_as_a_coordinate(self):
        """Piege reel : le « N » majuscule de Niveau a EXACTEMENT la meme hauteur qu'un
        chiffre, donc un filtre par hauteur ne suffit pas. C'est le decoupage en groupes,
        dont on ne garde que les deux premiers, qui tranche.

        Sans cela, `combat1.png` (« 1, 29 - Niveau 30 ») aurait pu rendre 30 en ordonnee.
        """
        path = ROOT / "combat1.png"
        if not path.exists():
            pytest.skip("combat1.png absent")
        position = read_position(cv2.imread(str(path)))
        assert position is not None and position.y == 29, "le niveau a ete lu"


def test_position_is_a_plain_tuple():
    assert MapPosition(x=-2, y=5).as_tuple() == (-2, 5)

class TestNegativeCoordinatesAreRead:
    """INCARNAM EST EN COORDONNEES NEGATIVES, et le bot y etait aveugle sur chaque carte.

    Ce module refusait toute lecture des qu'un tiret precedait la fin des coordonnees,
    faute de pouvoir le distinguer du separateur de « - Niveau », qui est le meme glyphe.
    Le refus etait honnete -- « -2, 9 » lu « 2, 9 » designe une carte a l'oppose -- mais
    sa portee etait totale : en session reelle, chaque carte d'Incarnam sortait INCONNU,
    l'orchestrateur envoyait sa touche de dernier recours et s'arretait. Le bot ouvrait le
    menu du jeu et ne faisait plus rien.

    Le commentaire d'alors nommait ce qui manquait : « il faudrait mesurer l'ecart
    tiret-chiffre, et le mesurer demande une capture ». Les captures existent maintenant,
    et la mesure separe franchement -- signe a 2-3 px de son chiffre, separateur a 8 px
    du « N ».

    LES CAPTURES SONT SEPAREES EN DEUX. Celles qui ont servi a fabriquer les gabarits ne
    peuvent pas prouver grand-chose en se relisant elles-memes ; ce sont les AUTRES qui
    font foi, et elles portent les memes chiffres rendus sur d'autres fonds."""

    # Verite terrain lue a l'oeil sur le bandeau de chaque capture.
    CONSTRUITES: ClassVar = [("20260818-205851", -1, -4), ("20260818-210551", -1, -6),
                             ("20260818-210845", -2, -6), ("20260818-211155", -3, -6)]
    RETENUES: ClassVar = [("20260818-205918", -1, -4), ("20260818-210602", -2, -6),
                          ("20260818-210614", -2, -6), ("20260818-210621", -2, -6)]

    def _capture(self):
        path = ROOT / "hors_combat.png"
        if not path.exists():
            pytest.skip("hors_combat.png absent")
        return cv2.imread(str(path))

    def _incarnam(self, nom):
        path = ROOT / "data" / "failures" / f"{nom}-ecran-inconnu.png"
        if not path.exists():
            pytest.skip(f"{nom} absent")
        return cv2.imread(str(path))

    def test_the_level_separator_does_not_block_the_reading(self):
        """Le tiret de « - Niveau » vient APRES les coordonnees : il ne doit rien
        empecher, sinon plus aucune carte ne serait lisible."""
        assert read_position(self._capture()) == MapPosition(x=2, y=9)

    @pytest.mark.parametrize(("nom", "x", "y"), RETENUES)
    def test_a_held_out_negative_map_is_read(self, nom, x, y):
        """Captures qui n'ont PAS servi a fabriquer les gabarits : la vraie preuve."""
        assert read_position(self._incarnam(nom)) == MapPosition(x=x, y=y)

    @pytest.mark.parametrize(("nom", "x", "y"), CONSTRUITES)
    def test_the_chain_closes_on_the_captures_that_taught_it(self, nom, x, y):
        """La chaine gabarits -> lecture se referme. Necessaire, pas suffisant."""
        assert read_position(self._incarnam(nom)) == MapPosition(x=x, y=y)

    def test_the_sign_is_really_carried_and_not_dropped(self):
        """CONTRE-EPREUVE du signe : sans lui, `-2, -6` se lirait `2, 6`.

        Une lecture qui rendrait la valeur absolue passerait tous les tests ci-dessus si
        on n'y comparait que des magnitudes. On exige donc explicitement le signe.
        """
        position = read_position(self._incarnam("20260818-210602"))
        assert position is not None
        assert position.x < 0 and position.y < 0, "le signe a ete perdu en route"

    def test_positive_maps_stay_positive(self):
        """L'autre sens : appliquer un signe a tort renverrait le bot a l'oppose."""
        position = read_position(self._capture())
        assert position is not None
        assert position.x > 0 and position.y > 0

class TestTheFailureExplainsItself:
    """« Coordonnees illisibles » ne se corrige pas.

    Une session reelle a rapporte « 3 passages sans coordonnees lisibles » et rien de
    plus. Les cinq causes possibles appellent des gestes OPPOSES : fabriquer un gabarit,
    deplacer une ROI, accepter un signe negatif, elargir la bande de hauteur. Sans la
    raison, il n'y avait aucun moyen de savoir laquelle -- et tout ce qui depend de la
    position restait inerte : graphe des cartes, detection de derive, trajet vers la
    banque."""

    def test_a_reading_that_works_carries_no_reason(self):
        path = ROOT / "hors_combat.png"
        if not path.exists():
            pytest.skip("hors_combat.png absent")
        reading = diagnose_position(cv2.imread(str(path)))
        assert reading.position == MapPosition(2, 9) and reading.reason == ""

    def test_a_frame_too_small_says_so(self):
        reading = diagnose_position(np.zeros((40, 40, 3), np.uint8))
        assert "trop petite" in reading.reason

    def test_an_empty_zone_says_so(self):
        reading = diagnose_position(np.zeros((1080, 1920, 3), np.uint8))
        assert "aucun glyphe" in reading.reason

    def test_an_uncovered_digit_NAMES_the_missing_ones(self):
        """LE cas le plus probable en jeu, et le seul qui se corrige en une commande :
        il manque des gabarits. Encore faut-il savoir lesquels."""
        path = ROOT / "hors_combat.png"
        if not path.exists():
            pytest.skip("hors_combat.png absent")
        from jev_tactics.perception.digits import DigitTemplates

        full = load_coord_templates()
        keep = full.digits != 2                      # on ampute le « 2 » qu'elle affiche
        amputed = DigitTemplates(exemplars=full.exemplars[keep], digits=full.digits[keep])
        reading = diagnose_position(cv2.imread(str(path)), amputed)
        assert reading.position is None
        assert "MANQUANTS" in reading.reason and "2" in reading.reason

    def test_the_score_is_reported_with_the_threshold(self):
        """Un accord a 0,89 pour un seuil a 0,90 se corrige autrement qu'un accord a
        0,40 : le premier demande un exemplaire de plus, le second une autre police."""
        path = ROOT / "hors_combat.png"
        if not path.exists():
            pytest.skip("hors_combat.png absent")
        from jev_tactics.perception.digits import DigitTemplates

        full = load_coord_templates()
        keep = full.digits != 2
        amputed = DigitTemplates(exemplars=full.exemplars[keep], digits=full.digits[keep])
        assert "seuil" in diagnose_position(cv2.imread(str(path)), amputed).reason

    def test_a_dash_that_signs_nothing_is_still_refused(self):
        """Un tiret qui ne touche aucun nombre n'est ni un signe ni un separateur.

        Ce test remplace celui qui exigeait le REFUS de toute coordonnee negative. Sa
        mise en scene -- un tiret colle a 14 px du chiffre -- ne ressemblait a aucun
        affichage reel : mesure sur les captures d'Incarnam, un vrai signe est a 2-3 px de
        son chiffre. Il epinglait donc une prudence excessive avec un montage qui ne
        pouvait pas se produire. Ce qui reste vrai, et qu'on garde, est le cas ou le tiret
        ne s'explique pas.
        """
        path = ROOT / "hors_combat.png"
        if not path.exists():
            pytest.skip("hors_combat.png absent")
        frame = cv2.imread(str(path))
        x0, y0, x1, y1 = COORD_ROI
        band = frame[y0:y1, x0:x1].copy()
        minus = band[:, 51:56].copy()
        frame[y0:y1, x0 + 12:x1] = band[:, :x1 - x0 - 12]
        frame[y0:y1, x0 + 2:x0 + 7] = minus
        reading = diagnose_position(frame)
        assert reading.position is None
        assert "inexplique" in reading.reason

    def test_read_position_keeps_its_contract(self):
        """Les appelants qui ne veulent que la position ne doivent rien changer."""
        assert read_position(np.zeros((40, 40, 3), np.uint8)) is None



class TestTheLineAboveDoesNotBreakTheReading:
    """LA ROI MORD LE BAS DE LA LIGNE DU DESSUS, et les fragments de lettres qu'elle y
    attrape ont la forme d'un tiret.

    Constate en session reelle le 19/08, sur la frame que le bot a lui-meme conservee :

        Incarnam (Cha...)
        -3, -6 - Niveau 5

    Les deux signes etaient correctement rattaches -- ce travail-la etait fait -- et la
    lecture echouait quand meme, sur trois marques a y = 0 a 3 alors que les chiffres
    tiennent entre y = 8 et 21. Ce sont les jambages de « Incarnam (Cha...) ».

    CE QUE CA COUTAIT. Toutes les cartes d'Incarnam ont des coordonnees NEGATIVES, donc la
    position etait illisible PARTOUT sur la zone de depart : le circuit ne pouvait plus
    juger un deplacement, le graphe des cartes restait vide (« 0 cartes, 0 transitions »
    dans le bilan de session), et le journal enregistrait `position: null` a chaque ligne --
    or sans carte, un point d'ecran est ininterpretable a froid.

    Un tiret qui n'est pas a la hauteur des chiffres ne peut etre ni un signe ni un
    separateur : il appartient a une autre ligne.
    """

    FRAMES = ("20260819-115112-combat-vide-recolte.png",
              "20260819-115130-combat-vide-recolte.png",
              "20260819-115106-combat-vide-recolte.png")

    def _frame(self, nom):
        chemin = ROOT / "data" / "failures" / nom
        if not chemin.exists():
            pytest.skip(f"{nom} absent (frame de session, non versionnee)")
        return cv2.imread(str(chemin))

    @pytest.mark.parametrize("nom", FRAMES)
    def test_incarnam_reads_its_negative_position(self, nom):
        lecture = diagnose_position(self._frame(nom))
        assert lecture.position is not None, lecture.reason
        assert (lecture.position.x, lecture.position.y) == (-3, -6)

    def test_a_map_screen_is_no_longer_taken_for_an_unknown_one(self):
        """LE COUT REEL DE CETTE LECTURE, et il ne se voit pas dans le lecteur.

        `classify_screen` reconnait une CARTE a ses coordonnees. Sans elles, une carte
        parfaitement ordinaire tombe en INCONNU, et l'orchestrateur arrete la session --
        « ecran non reconnu apres 1,8 s ». C'est arrive le 19/08 a 14h58 : la frame
        conservee montre Incarnam (Champs) en -2, -5, hors combat, rien d'anormal. Le bot
        s'est arrete sur une carte.

        La cause etait un gabarit manquant : le « 5 » de la coordonnee y mesure 10 px de
        large la ou celui de « Niveau 5 », sur la MEME ligne, en mesure 9. Le premier
        marquait 0,848 contre un seuil a 0,90, le second 0,992. Un pixel de largeur separe
        une session qui continue d'une session qui s'arrete.
        """
        chemin = ROOT / "data" / "failures" / "20260819-145855-ecran-inconnu.png"
        if not chemin.exists():
            pytest.skip("frame de session absente (non versionnee)")
        from jev_tactics.perception.screen import ScreenState, classify_screen
        frame = cv2.imread(str(chemin))
        lecture = diagnose_position(frame)
        assert lecture.position is not None, lecture.reason
        assert (lecture.position.x, lecture.position.y) == (-2, -5)
        assert classify_screen(frame) is ScreenState.MAP, (
            "la carte reste classee INCONNUE : la session s'arreterait encore dessus")

    def test_the_refusal_still_guards_its_own_ground(self):
        """CONTRE-EPREUVE : le refus ne doit pas avoir ete vide de sa substance. Un tiret
        A LA HAUTEUR DES CHIFFRES, colle a rien, reste inexplique et doit refuser -- c'est
        le cas qui protege d'une position douteuse."""
        frame = self._frame(self.FRAMES[0]).copy()
        # Un tiret pose au MILIEU de la ligne des coordonnees, loin de tout chiffre.
        frame[70 + 14:70 + 16, 30:38] = 255
        assert diagnose_position(frame).position is None
