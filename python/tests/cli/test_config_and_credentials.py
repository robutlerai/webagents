"""
Phase 3: one config contract, one credential store (2026-09-23).

What these pin, and why each mattered:

  * The two CLIs COLLIDED on `~/.webagents/config.json` with incompatible
    schemas, and DIVERGED on the credential filename, so logging in with one
    SDK left the other logged out.
  * `config get|set|edit|reset` were TODO no-ops. `set` printed "Config
    updated" and wrote nothing, which is the worst possible shape: the user is
    told it worked.
  * The token was written at 0644 (S-211).
  * There was no precedence, no env layer and no flag layer at all.
"""

import json
import os

import pytest

from webagents.cli.config_store import (
    DEFAULTS,
    ConfigStore,
    env_chain,
    expand,
    global_dir,
    profile_name,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestPrecedence:
    """flag > env > project > global > default, and it is documented."""

    def test_layers_are_ordered_highest_first(self, tmp_path):
        store = ConfigStore(cwd=tmp_path, overrides={"model": "from-flag"})
        assert [name for name, _ in store.layers()] == ["flag", "project", "global", "default"]

    def test_flag_beats_project_beats_default(self, tmp_path):
        (tmp_path / ".webagents").mkdir()
        (tmp_path / ".webagents" / "config.json").write_text(
            json.dumps({"daemon.port": 4321, "model": "from-project"})
        )

        store = ConfigStore(cwd=tmp_path, overrides={"model": "from-flag"})
        assert store.get("model") == "from-flag"
        assert store.source_of("model") == "flag"

        assert store.get("daemon.port") == 4321
        assert store.source_of("daemon.port") == "project"

        assert store.get("daemon.host") == DEFAULTS["daemon.host"]
        assert store.source_of("daemon.host") == "default"

    def test_reading_config_creates_nothing(self, tmp_path):
        # `LocalState.__init__` used to scaffold eight directories plus a
        # .gitignore in whatever directory you were standing in, reached from
        # `is_authenticated()`, so `auth whoami` littered any repo.
        ConfigStore(cwd=tmp_path).get("daemon.port")
        assert list(tmp_path.iterdir()) == []


class TestDotenvAndExpansion:
    """`${VAR}` references, so a committed config never holds a secret."""

    def test_dotenv_is_read_but_never_overrides_the_process_env(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text('FROM_FILE=file\nSHARED=file\n# comment\nQUOTED="q"\n')
        monkeypatch.setenv("SHARED", "process")

        env = env_chain(cwd=tmp_path)
        assert env["FROM_FILE"] == "file"
        assert env["QUOTED"] == "q"
        # A developer who exports something in their shell means it.
        assert env["SHARED"] == "process"

    def test_expansion_forms(self, tmp_path):
        env = {"SET": "value"}
        assert expand("${SET}", env) == "value"
        assert expand("${MISSING:-fallback}", env) == "fallback"
        # LEFT AS WRITTEN, not replaced with "". A silent empty string is how a
        # misconfigured URL becomes an inscrutable request to nowhere.
        assert expand("${MISSING}", env) == "${MISSING}"

    def test_a_config_value_can_reference_an_env_var(self, tmp_path, monkeypatch):
        (tmp_path / ".webagents").mkdir()
        (tmp_path / ".webagents" / "config.json").write_text(
            json.dumps({"platform.url": "${PORTAL_URL}"})
        )
        monkeypatch.setenv("PORTAL_URL", "https://staging.example")
        assert ConfigStore(cwd=tmp_path).get("platform.url") == "https://staging.example"


class TestValidation:
    """Unknown keys are reported, not silently accepted."""

    def test_unknown_key_is_reported_with_a_suggestion(self, tmp_path):
        (tmp_path / ".webagents").mkdir()
        (tmp_path / ".webagents" / "config.json").write_text(json.dumps({"daemon.prot": 1}))
        problems = ConfigStore(cwd=tmp_path).validate()
        assert any("unknown key" in p and "daemon.prot" in p for p in problems)
        assert any("daemon.port" in p for p in problems), "no suggestion offered"

    def test_unresolvable_reference_is_reported(self, tmp_path):
        (tmp_path / ".webagents").mkdir()
        (tmp_path / ".webagents" / "config.json").write_text(
            json.dumps({"platform.url": "${NOT_SET_ANYWHERE}"})
        )
        problems = ConfigStore(cwd=tmp_path).validate()
        assert any("NOT_SET_ANYWHERE" in p for p in problems)

    def test_malformed_json_is_reported_but_does_not_break_reads(self, tmp_path):
        (tmp_path / ".webagents").mkdir()
        (tmp_path / ".webagents" / "config.json").write_text("{not json")
        store = ConfigStore(cwd=tmp_path)
        assert any("not valid JSON" in p for p in store.validate())
        # Every other command still works, on the layers below.
        assert store.get("daemon.host") == DEFAULTS["daemon.host"]


class TestProfiles:
    """`--profile` moves EVERY path, not some of them."""

    def test_profile_changes_the_global_directory(self, monkeypatch):
        assert global_dir().name == ".webagents"
        monkeypatch.setenv("WEBAGENTS_PROFILE", "staging")
        # Resolved from the environment, not just from an argument. Taking the
        # argument literally is what let the token go to the profile keystore
        # while its metadata landed in the shared directory.
        assert global_dir().name == ".webagents-staging"
        assert profile_name() == "staging"

    def test_an_explicit_profile_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("WEBAGENTS_PROFILE", "from-env")
        assert global_dir("from-arg").name == ".webagents-from-arg"


class TestCredentialPrecedence:
    """flag > env > keystore, and the token is never in the config file."""

    def test_flag_beats_env_beats_store(self, monkeypatch):
        from webagents.cli import credentials

        monkeypatch.setattr(credentials, "_store", lambda *a, **k: _FakeStore("from-store"))
        assert credentials.get_token() == "from-store"

        monkeypatch.setenv("WEBAGENTS_TOKEN", "from-env")
        assert credentials.get_token() == "from-env"
        assert credentials.get_token(explicit="from-flag") == "from-flag"

    def test_a_missing_keystore_reads_as_not_logged_in(self, monkeypatch):
        # It must not raise: the caller reports "not logged in", which is true.
        from webagents.cli import credentials

        def boom(*a, **k):
            raise RuntimeError("no keystore here")

        monkeypatch.setattr(credentials, "_store", boom)
        assert credentials.get_token() is None

    def test_the_token_never_lands_in_the_plaintext_metadata_file(self, tmp_path, monkeypatch):
        # S-211. `_save_json` is a bare write_text with no mode, so anything it
        # touches is 0644. The token must not go through it.
        from webagents.cli.state import local as local_mod

        captured = {}
        monkeypatch.setattr(
            "webagents.cli.credentials.set_token",
            lambda token, profile=None, quiet=False: captured.setdefault("token", token) or "keystore",
        )
        state = local_mod.LocalState(project_dir=tmp_path)
        state.global_dir = tmp_path / "global"
        state.global_dir.mkdir(parents=True, exist_ok=True)

        state.set_credentials(access_token="secret-jwt", username="tester")

        assert captured["token"] == "secret-jwt", "token did not reach the keystore"
        on_disk = json.loads((state.global_dir / "credentials.json").read_text())
        assert on_disk == {"username": "tester"}
        assert "access_token" not in on_disk


class _FakeStore:
    """Stands in for the keystore, so these tests never touch a real one."""

    def __init__(self, value):
        self._value = value

    def get(self, _name):
        return self._value
