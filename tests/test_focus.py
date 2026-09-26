"""Attente de la fenetre du jeu avant capture.

Probleme constate a l'usage : lancer un script `--live` depuis le terminal capture le
TERMINAL, et l'utilisateur doit basculer « tres vite » sur le jeu. Une capture ratee ne
se voit pas comme une erreur -- elle se voit comme une perception qui ne trouve rien,
c'est-a-dire le mode de defaillance le plus couteux a diagnostiquer du projet.

Ces tests portent sur les cas degrades autant que sur le cas nominal : ce module doit
toujours degrader vers « on capture quand meme », jamais vers « on refuse »."""

from jev_tactics.capture.focus import (
    foreground_title,
    grab_when_focused,
    matches,
    wait_for_game,
)


class _Clock:
    """Horloge factice : les tests n'attendent jamais reellement."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Capture:
    def __init__(self):
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        return "frame"


class TestWaitForGame:
    def test_returns_immediately_when_already_focused(self):
        clock = _Clock()
        assert wait_for_game(title=lambda: "Dofus 3.6 — Perso", sleep=clock.sleep,
                             clock=clock.time)
        assert clock.now == 0.0

    def test_matching_is_case_insensitive(self):
        clock = _Clock()
        assert wait_for_game(title=lambda: "DOFUS", sleep=clock.sleep, clock=clock.time)

    def test_the_real_window_title_matches(self):
        """MESURE, et elle a corrige une erreur : le titre du client reel est
        « Centvingtkilosbench - Sacrieur - 3.6.10.10 - Release » -- **sans le mot
        Dofus**. Le motif choisi a priori n'aurait jamais correspondu, l'attente aurait
        expire a chaque fois et on serait revenu a capturer le terminal, sans rien qui
        le signale."""
        assert matches("Centvingtkilosbench - Sacrieur - 3.6.10.10 - Release")

    def test_an_unrelated_window_does_not_match(self):
        assert not matches("Windows PowerShell")
        assert not matches("combat1.png - Photos")

    def test_waits_then_succeeds_when_the_user_switches(self):
        clock = _Clock()
        titles = iter(["Terminal", "Terminal", "Terminal", "Dofus 3.6"])
        assert wait_for_game(title=lambda: next(titles, "Dofus 3.6"),
                             sleep=clock.sleep, clock=clock.time)
        assert clock.now > 0.0

    def test_gives_up_after_the_timeout(self):
        """Ne jamais bloquer indefiniment : l'utilisateur doit reprendre la main."""
        clock = _Clock()
        assert not wait_for_game(title=lambda: "Terminal", timeout=5.0,
                                 sleep=clock.sleep, clock=clock.time)

    def test_unavailable_title_does_not_block(self):
        """Plateforme sans titre lisible : degrader vers la capture immediate plutot que
        d'attendre un evenement qui n'arrivera jamais."""
        clock = _Clock()
        assert not wait_for_game(title=lambda: "", sleep=clock.sleep, clock=clock.time)
        assert clock.now == 0.0

    def test_announces_only_when_it_actually_waits(self):
        """Un message affiche alors qu'il n'y a pas d'attente serait du bruit."""
        clock = _Clock()
        calls = []
        wait_for_game(title=lambda: "Dofus", sleep=clock.sleep, clock=clock.time,
                      on_wait=lambda: calls.append(1))
        assert calls == []

    def test_announces_when_it_waits(self):
        clock = _Clock()
        calls = []
        wait_for_game(title=lambda: "Terminal", timeout=2.0, sleep=clock.sleep,
                      clock=clock.time, on_wait=lambda: calls.append(1))
        assert calls == [1]

    def test_settles_after_an_actual_switch(self):
        """Apres une bascule, l'animation de fenetre peut etre en cours : capturer
        pendant donnerait une image partielle. Un delai de repos s'impose -- mais
        seulement la, pas quand le jeu etait deja au premier plan."""
        clock = _Clock()
        titles = iter(["Terminal", "Dofus"])
        wait_for_game(title=lambda: next(titles, "Dofus"), sleep=clock.sleep,
                      clock=clock.time)
        assert clock.now > 0.0

    def test_reads_the_title_once_when_already_focused(self):
        """Le premier controle ne doit pas consommer deux lectures : sur une source
        sequentielle, la seconde masquerait la premiere."""
        calls = []
        clock = _Clock()

        def title():
            calls.append(1)
            return "Dofus"

        wait_for_game(title=title, sleep=clock.sleep, clock=clock.time)
        assert len(calls) == 1


class TestGrabWhenFocused:
    def test_captures_once_focused(self):
        clock = _Clock()
        capture = _Capture()
        assert grab_when_focused(capture, title=lambda: "Dofus", sleep=clock.sleep,
                                 clock=clock.time) == "frame"
        assert capture.grabs == 1

    def test_timeout_reports_the_observed_title(self):
        """Ce qui rend l'echec REPARABLE : sans le titre reel, un motif qui ne
        correspond pas produit une attente inexpliquee. Avec, l'utilisateur voit
        immediatement quoi corriger."""
        clock = _Clock()
        messages = []
        grab_when_focused(_Capture(), timeout=1.0, title=lambda: "Mon Jeu Bizarre",
                          sleep=clock.sleep, clock=clock.time,
                          announce=messages.append)
        assert any("Mon Jeu Bizarre" in m for m in messages)

    def test_captures_anyway_after_a_timeout(self):
        """Le point de conception : mieux vaut une capture peut-etre mauvaise, que
        l'utilisateur verra, qu'un refus de fonctionner."""
        clock = _Clock()
        capture = _Capture()
        messages = []
        grab_when_focused(capture, timeout=1.0, title=lambda: "Terminal",
                          sleep=clock.sleep, clock=clock.time,
                          announce=messages.append)
        assert capture.grabs == 1
        assert any("non detectee" in m for m in messages)

    def test_says_what_it_is_waiting_for(self):
        """Un script qui bloque sans rien afficher est indiscernable d'un script fige."""
        clock = _Clock()
        messages = []
        grab_when_focused(_Capture(), timeout=1.0, title=lambda: "Terminal",
                          sleep=clock.sleep, clock=clock.time,
                          announce=messages.append)
        assert any("Alt+Tab" in m for m in messages)


def test_foreground_title_never_raises():
    """Quelle que soit la plateforme : l'indisponibilite doit degrader, pas planter."""
    assert isinstance(foreground_title(), str)


class TestTheBotsOwnWindowIsNotTheGame:
    """Le panneau de controle s'appelle « dofus-bot — panneau de controle ».

    Il contient donc « dofus », et `matches` le declarait etre le jeu. Tout ce que ce
    module protege s'effondrait alors d'un coup, en silence : `FocusGuardedBackend`
    laissait passer les gestes en croyant le jeu devant, les clics partaient aux
    coordonnees du plateau PENDANT que le panneau etait au premier plan, le compteur de
    gestes avales restait a zero, et le bilan ne signalait rien.

    Le bot percevait tout correctement et n'agissait sur rien -- « il ne fait rien ».
    """

    def test_the_control_panel_is_not_mistaken_for_the_game(self):
        assert not matches("dofus-bot — panneau de controle")

    def test_the_exclusion_ignores_case(self):
        assert not matches("DOFUS-BOT — Panneau de controle")

    def test_the_real_game_still_matches(self):
        """CONTRE-EPREUVE : l'exclusion ne doit pas emporter le jeu avec elle.

        Le titre reel n'a pas le mot « Dofus » -- il vaut « <personnage> - <classe> -
        <version> - Release » -- d'ou le motif « - release ». Un tour de vis qui le
        perdrait ferait attendre le bot indefiniment.
        """
        assert matches("Centvingtkilosbench - Sacrieur - 3.6.10.10 - Release")

    def test_a_launcher_named_dofus_still_matches(self):
        assert matches("Dofus 3.6")

    def test_the_exclusion_is_checked_before_the_patterns(self):
        """Un titre qui porte les DEUX signaux doit etre refuse : une fenetre a soi ne
        peut pas se racheter en ressemblant par ailleurs au jeu."""
        assert not matches("dofus-bot — panneau - Release")
