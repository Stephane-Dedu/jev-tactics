"""La capture d'ecran, point d'entree de tout le bot.

Aucun test ne la couvrait : il lui faut un vrai ecran, donc elle etait laissee de cote --
et c'est justement ce qui rend une deprecation silencieuse dangereuse ici. `mss.mss` est
deprecie depuis mss 10 et disparaitra ; le jour ou il disparait, RIEN ne fonctionne, et la
suite serait restee verte jusque-la.

Ces tests s'ignorent proprement sur une machine sans affichage.
"""

import warnings
from typing import ClassVar

import pytest


@pytest.fixture
def capture():
    from jev_tactics.capture import ScreenCapture

    try:
        instance = ScreenCapture()
    except Exception:                        # pragma: no cover - machine sans affichage
        pytest.skip("pas d'ecran capturable")
    yield instance
    instance.close()


class TestTheCaptureOriginIsTheClickOrigin:
    """L'hypothese la plus basse de tout le bot, ecrite et verifiee nulle part.

    `grab()` rend une image dont le pixel (0, 0) est le coin du moniteur capture. La
    detection rend des coordonnees DANS cette image. `pydirectinput.click(x, y)` les
    interprete en coordonnees ECRAN ABSOLUES. Les deux ne coincident que si ce coin est a
    l'origine -- ce qui n'est pas garanti des qu'il y a plusieurs ecrans.

    La panne serait totale et muette : chaque clic decale d'une constante, souvent sur
    l'autre ecran, avec une detection parfaite et zero combat. Tous les diagnostics de la
    chasse accuseraient le detecteur.
    """

    def _capture(self, monkeypatch, left, top):
        import jev_tactics.capture.screen as module

        class _Faux:
            monitors: ClassVar[list] = [
                {"left": 0, "top": 0, "width": 0, "height": 0},
                {"left": left, "top": top, "width": 1920, "height": 1080}]

            def close(self):
                pass

        monkeypatch.setattr(module, "mss", type("m", (), {"MSS": _Faux, "mss": _Faux}))
        return module.ScreenCapture()

    def test_it_reads_the_monitor_corner(self, monkeypatch):
        assert self._capture(monkeypatch, 1920, 0).origin == (1920, 0)

    def test_a_primary_monitor_at_the_origin_reads_zero(self, monkeypatch):
        """CONTRE-EPREUVE : une origine toujours non nulle ferait refuser toutes les
        sessions, y compris les correctes."""
        assert self._capture(monkeypatch, 0, 0).origin == (0, 0)

    def test_a_negative_corner_is_seen_too(self, monkeypatch):
        """Un ecran place a GAUCHE du principal donne des coordonnees negatives. Le
        controle doit les voir : le decalage est le meme probleme dans l'autre sens."""
        assert self._capture(monkeypatch, -1920, -200).origin == (-1920, -200)


class TestTheCaptureUsesASupportedEntryPoint:
    def test_building_it_warns_about_nothing(self):
        """`mss.mss` emettait un DeprecationWarning A CHAQUE construction. La fabrique et
        la classe rendent le meme objet -- verifie : meme type, meme capture -- donc le
        seul effet du changement est de faire taire un avertissement qui annonce une
        rupture."""
        from jev_tactics.capture import ScreenCapture

        with warnings.catch_warnings(record=True) as vus:
            warnings.simplefilter("always")
            try:
                instance = ScreenCapture()
            except Exception:                # pragma: no cover - machine sans affichage
                pytest.skip("pas d'ecran capturable")
            instance.close()
        deprecations = [v for v in vus if issubclass(v.category, DeprecationWarning)]
        assert deprecations == [], [str(v.message) for v in deprecations]

    def test_a_grab_returns_a_bgr_image(self, capture):
        """Contre-epreuve : un test qui ne verifie que l'absence d'avertissement serait
        satisfait par une capture qui ne capture rien."""
        frame = capture.grab()
        assert frame.image.ndim == 3 and frame.image.shape[2] == 3
        assert frame.image.shape[0] > 0 and frame.image.shape[1] > 0

    def test_a_region_is_honoured(self, capture):
        frame = capture.grab({"left": 0, "top": 0, "width": 64, "height": 32})
        assert frame.image.shape[:2] == (32, 64)
