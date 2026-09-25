"""
S-222 and the portal-URL split (2026-09-24).

Two defects that a unit test could not see, because each one lived in the gap
between a function that was right and a caller that never reached it.

  * `--profile` and `--token` were read into `ctx.obj` by the root callback and
    dropped there. Nothing under the callback reads `ctx.obj`: the token
    lookup, the keystore namespace and `global_dir()` all resolve from the
    ENVIRONMENT. `tests/cli/test_config_and_credentials.py` already pinned
    `get_token(explicit=...)` and `global_dir("name")`, and both passed the
    whole time, because the argument they were given by hand is the argument
    the CLI never passed. So `--profile local login` overwrote the DEFAULT
    profile's token, and `--token "$T" deploy` ran as whoever was logged in on
    that machine.
  * `login` resolved the portal from `ROBUTLER_API_URL` at import time and
    ignored `platform.url`; `RobutlerAPI` read `platform.url` and ignored the
    variable. Setting either alone signed you in to one portal and deployed to
    another.

Everything here therefore goes through `runner.invoke(app, ...)` rather than
calling the resolver directly: the contract being pinned is that the FLAG
arrives, not that the function honours an argument.
"""

import os

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """An isolated HOME with the keystore refused, so no real Keychain write."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The file backend, explicitly: these tests store dummy tokens, and the
    # developer's own Keychain is not a test fixture.
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "ROBUTLER_API_URL",
                "ROBUTLER_INTERNAL_API_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    from webagents.cli import credentials

    # The flag override is process state; a leak between tests would make one
    # test pass because of another.
    credentials.set_flag_token(None)
    yield home
    credentials.set_flag_token(None)
    # `--profile` writes WEBAGENTS_PROFILE into the environment itself (the
    # point of `test_profile_flag_reaches_child_processes`), past monkeypatch,
    # which only undoes what it set. Left there it made every later test run
    # under that profile, and every hint name it (`cli_command`). Removed
    # here; monkeypatch then restores whatever the shell had.
    os.environ.pop("WEBAGENTS_PROFILE", None)


def _store_token_in_default_profile(value: str) -> None:
    from webagents.cli.credentials import set_token

    set_token(value)


@pytest.fixture
def bearer(monkeypatch):
    """`whoami` against a fake portal; returns the bearers it was sent.

    `whoami` is how the token is observed: `auth token`, which printed it, is
    gone with the rest of the Python-only commands.
    """
    import httpx

    seen = []

    class _Response:
        status_code = 200

        def json(self):
            return {"user": {"username": "me"}}

    def get(url, headers=None, timeout=None):
        seen.append((headers or {}).get("Authorization", "").removeprefix("Bearer "))
        return _Response()

    monkeypatch.setattr(httpx, "get", get)
    return seen


class TestProfileFlagReachesTheCredentialStore:
    """The measurement that found S-222, as a test."""

    def test_profile_flag_isolates_the_token(self, bearer):
        _store_token_in_default_profile("DEFAULT-PROFILE-TOKEN")

        # Sanity: the default profile has one.
        assert runner.invoke(app, ["whoami"]).exit_code == 0

        # THE BUG: this used to find the default profile's token, so
        # `--profile local login` then overwrote it.
        result = runner.invoke(app, ["--profile", "other", "whoami"])
        assert result.exit_code == 1
        assert bearer == ["DEFAULT-PROFILE-TOKEN"]

    def test_profile_flag_and_env_var_agree(self, bearer):
        _store_token_in_default_profile("DEFAULT-PROFILE-TOKEN")

        flag = runner.invoke(app, ["--profile", "other", "whoami"])
        env = runner.invoke(app, ["whoami"], env={"WEBAGENTS_PROFILE": "other"})
        # The env form always worked. The point is that they now MATCH: it was
        # exercising only this form that hid the defect.
        assert flag.exit_code == env.exit_code == 1

    def test_profile_flag_reaches_child_processes(self):
        """It goes into the environment, where every later lookup (and a
        child process) reads it; carried on `ctx.obj` it reached nothing."""
        import os

        runner.invoke(app, ["--profile", "spawned", "config", "path"])
        assert os.environ.get("WEBAGENTS_PROFILE") == "spawned"

    def test_no_flag_leaves_the_environment_alone(self):
        import os

        runner.invoke(app, ["config", "path"])
        assert os.environ.get("WEBAGENTS_PROFILE") is None


class TestTokenFlagReachesTheLookup:
    def test_token_flag_outranks_the_stored_token(self, bearer):
        _store_token_in_default_profile("STORED-TOKEN")

        result = runner.invoke(app, ["--token", "FLAG-TOKEN", "whoami"])
        assert result.exit_code == 0
        # THE BUG: the stored one was sent, so `--token` on a shared CI runner
        # silently ran as whoever last logged in there.
        assert bearer == ["FLAG-TOKEN"]

    def test_token_flag_beats_the_env_var(self, bearer):
        runner.invoke(app, ["--token", "FLAG-TOKEN", "whoami"], env={"WEBAGENTS_TOKEN": "ENV-TOKEN"})
        assert bearer == ["FLAG-TOKEN"]

    def test_token_flag_is_not_exported_to_child_processes(self, bearer):
        """Deliberately process-local: a child never asked for the bearer."""
        import os

        runner.invoke(app, ["--token", "FLAG-TOKEN", "whoami"])
        assert os.environ.get("WEBAGENTS_TOKEN") is None


class TestOnePortalUrlForLoginAndDeploy:
    """`login` and `publish` must never resolve different portals."""

    def _api_base(self):
        from webagents.cli.platform.api import RobutlerAPI

        return RobutlerAPI().base_url

    def _login_base(self):
        from webagents.cli.platform import auth

        return auth._base()

    def test_env_var_moves_both(self, monkeypatch):
        monkeypatch.setenv("ROBUTLER_API_URL", "https://cluster.example")
        # THE BUG, one direction: `deploy` ignored the variable.
        assert self._api_base() == "https://cluster.example"
        assert self._login_base() == "https://cluster.example"

    def test_config_key_moves_both(self, tmp_path):
        from webagents.cli.config_store import ConfigStore

        ConfigStore().set("platform.url", "https://configured.example")
        # THE BUG, the other direction and the one that bit: `login` ignored
        # `platform.url`, so it opened production while `deploy` went local.
        assert self._login_base() == "https://configured.example"
        assert self._api_base() == "https://configured.example"

    def test_environment_outranks_config(self, monkeypatch):
        from webagents.cli.config_store import ConfigStore

        ConfigStore().set("platform.url", "https://configured.example")
        monkeypatch.setenv("ROBUTLER_API_URL", "https://from-env.example")
        assert self._login_base() == "https://from-env.example"
        assert self._api_base() == "https://from-env.example"

    def test_trailing_slash_does_not_double(self, monkeypatch):
        monkeypatch.setenv("ROBUTLER_API_URL", "https://cluster.example/")
        assert self._login_base() == "https://cluster.example"

    def test_internal_url_is_not_consulted(self, monkeypatch):
        """It is an in-cluster address; a developer's browser cannot open it."""
        monkeypatch.setenv("ROBUTLER_INTERNAL_API_URL", "http://portal.ns.svc.cluster.local")
        assert self._login_base() == "https://robutler.ai"

    def test_the_profile_carries_its_own_portal(self, monkeypatch):
        from webagents.cli.config_store import ConfigStore

        ConfigStore().set("platform.url", "https://production.example")
        monkeypatch.setenv("WEBAGENTS_PROFILE", "local")
        ConfigStore().set("platform.url", "https://cluster.example")

        # The whole point of a profile: a local login cannot deploy to prod.
        assert self._login_base() == "https://cluster.example"
        assert self._api_base() == "https://cluster.example"

        monkeypatch.delenv("WEBAGENTS_PROFILE")
        assert self._login_base() == "https://production.example"

    def test_resolver_reports_its_source(self, monkeypatch):
        from webagents.cli.config_store import resolve_platform_url

        assert resolve_platform_url() == ("https://robutler.ai", "default")

        from webagents.cli.config_store import ConfigStore

        ConfigStore().set("platform.url", "https://configured.example")
        assert resolve_platform_url() == ("https://configured.example", "global")

        monkeypatch.setenv("ROBUTLER_API_URL", "https://from-env.example")
        assert resolve_platform_url() == ("https://from-env.example", "ROBUTLER_API_URL")


class TestDoctorNamesThePortal:
    """Which portal the CLI talks to, before anything is published there.

    An agent username is minted once and cannot be renamed, so publishing to
    the wrong portal is not an undo away. The sign-in check names the portal
    `publish` would use (and `publish --dry-run` names the whole route).
    """

    def test_it_names_the_configured_portal(self):
        from webagents.cli.config_store import ConfigStore

        ConfigStore().set("platform.url", "https://cluster.example")
        result = runner.invoke(app, ["doctor"])
        assert "Not signed in to cluster.example." in result.stdout

    def test_logged_out_is_a_warning_not_a_silent_ok(self):
        result = runner.invoke(app, ["doctor"])
        assert "▲ sign-in" in result.stdout

    def test_the_environment_variable_moves_it(self, monkeypatch):
        monkeypatch.setenv("ROBUTLER_API_URL", "https://from-env.example")
        result = runner.invoke(app, ["doctor"])
        assert "from-env.example" in result.stdout
