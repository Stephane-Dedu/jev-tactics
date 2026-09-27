"""La lecture de `.env`, et les trois regles qui evitent une panne connue.

Le fichier etait deja ignore par git avant que quoi que ce soit ne sache le lire -- l'etat
le plus trompeur possible : on pose sa cle dedans, on croit avoir configure le depot, et
rien ne la lit. Ces tests gardent le contraire.
"""

from __future__ import annotations

import os

import pytest

from jev_tactics import config


@pytest.fixture
def env_file(tmp_path):
    def write(contents: str):
        path = tmp_path / ".env"
        path.write_text(contents, encoding="utf-8")
        return path
    return write


class TestReading:
    def test_a_simple_pair_is_loaded(self, env_file, monkeypatch):
        monkeypatch.delenv("UN_TEST_JEV", raising=False)
        posees = config.load_env(env_file("UN_TEST_JEV=abc\n"))
        assert posees == ["UN_TEST_JEV"]
        assert os.environ["UN_TEST_JEV"] == "abc"

    def test_quotes_and_spaces_are_stripped(self, env_file, monkeypatch):
        monkeypatch.delenv("UN_TEST_JEV", raising=False)
        config.load_env(env_file('  UN_TEST_JEV = "abc"  \n'))
        assert os.environ["UN_TEST_JEV"] == "abc"

    def test_an_export_prefix_is_tolerated(self, env_file, monkeypatch):
        """C'est ce que les gens copient depuis une documentation shell."""
        monkeypatch.delenv("UN_TEST_JEV", raising=False)
        config.load_env(env_file("export UN_TEST_JEV=abc\n"))
        assert os.environ["UN_TEST_JEV"] == "abc"

    def test_comments_and_blanks_are_ignored(self, env_file, monkeypatch):
        monkeypatch.delenv("UN_TEST_JEV", raising=False)
        posees = config.load_env(env_file("# commentaire\n\nUN_TEST_JEV=abc\n"))
        assert posees == ["UN_TEST_JEV"]

    def test_a_value_containing_equals_survives(self, env_file, monkeypatch):
        """Une cle base64 peut contenir « = » : couper sur le dernier separateur la
        tronquerait, et l'erreur ne se verrait qu'a l'appel distant."""
        monkeypatch.delenv("UN_TEST_JEV", raising=False)
        config.load_env(env_file("UN_TEST_JEV=abc==\n"))
        assert os.environ["UN_TEST_JEV"] == "abc=="


class TestTheRealEnvironmentWins:
    """Sinon la CI ne pourrait plus imposer sa cle, et essayer une autre valeur
    demanderait d'editer un fichier."""

    def test_an_exported_variable_is_not_overwritten(self, env_file, monkeypatch):
        monkeypatch.setenv("UN_TEST_JEV", "depuis_le_shell")
        posees = config.load_env(env_file("UN_TEST_JEV=depuis_le_fichier\n"))
        assert posees == []
        assert os.environ["UN_TEST_JEV"] == "depuis_le_shell"

    def test_override_is_possible_but_explicit(self, env_file, monkeypatch):
        monkeypatch.setenv("UN_TEST_JEV", "depuis_le_shell")
        config.load_env(env_file("UN_TEST_JEV=depuis_le_fichier\n"), override=True)
        assert os.environ["UN_TEST_JEV"] == "depuis_le_fichier"


class TestAbsenceIsNormal:
    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert config.load_env(tmp_path / "pas_la") == []

    def test_no_key_reads_as_none_not_empty_string(self, monkeypatch):
        """None et "" se comportent differemment dans un `or` : une chaine vide prise
        pour une cle donnerait un 401 au lieu d'un message clair."""
        monkeypatch.setenv("TYPESAFE_API_KEY", "")
        assert config.api_key() is None


class TestNothingLeaks:
    def test_the_description_never_shows_the_value(self, monkeypatch):
        """Une cle apparue une fois dans une trace est une cle a revoquer."""
        monkeypatch.setenv("TYPESAFE_API_KEY", "sk-tres-secret-0123456789")
        described = config.describe_key()
        assert "sk-tres-secret" not in described
        assert "0123456789" not in described
        assert "presente" in described

    def test_it_says_when_the_key_is_missing(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setattr(config, "_loaded", True)   # ne pas lire le .env local
        assert "absente" in config.describe_key()

    def test_loading_returns_names_not_values(self, env_file, monkeypatch):
        monkeypatch.delenv("UN_TEST_JEV", raising=False)
        posees = config.load_env(env_file("UN_TEST_JEV=secret\n"))
        assert posees == ["UN_TEST_JEV"]
        assert "secret" not in str(posees)


class TestTheTransportUsesIt:
    def test_the_http_transport_finds_a_key_from_env(self, monkeypatch):
        """Le defaut evite : lire `os.environ` directement rendait « cle absente » sur une
        machine ou le `.env` existe."""
        from jev_tactics.decision.transport import HttpTransport

        monkeypatch.setenv("TYPESAFE_API_KEY", "depuis_l_environnement")
        assert HttpTransport().api_key == "depuis_l_environnement"

    def test_an_explicit_key_still_wins(self, monkeypatch):
        from jev_tactics.decision.transport import HttpTransport

        monkeypatch.setenv("TYPESAFE_API_KEY", "depuis_l_environnement")
        assert HttpTransport(api_key="explicite").api_key == "explicite"
