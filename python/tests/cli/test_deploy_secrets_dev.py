"""
`link`, `publish`, `secrets` (2026-09-23; `deploy` became `publish` and `dev`
went on 2026-09-24, with the command surface of the TypeScript CLI).

New surface, so these are about the contracts rather than about defects, with
two exceptions that are both pinned bugs:

* S-219: `--profile` isolated the stored token only on machines WITHOUT an OS
  keystore, because the profile reached the fallback file's directory and
  never reached the keystore namespace.
* The platform client pointed every call at `/v1/*`, which exists nowhere in
  the portal.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()

AGENT_WITH_SKILLS = """---
name: shipper
description: Ships things
model: openai/gpt-4o-mini
intents:
  - ship the thing
skills:
  - memory
  - mcp:
      servers: [filesystem]
---

Ship it.
"""


class TestTheDeployPayloadMatchesThePortalSchema:
    """`POST /api/agents` takes name/description/instructions/model/intents/skills."""

    def _dry_run(self, tmp_path, monkeypatch, body=AGENT_WITH_SKILLS):
        (tmp_path / "AGENT.md").write_text(body)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["publish", "--dry-run"])
        assert result.exit_code == 0, result.output
        start = result.output.index("{")
        return json.loads(result.output[start:])

    def test_skills_become_a_mapping_not_a_list(self, tmp_path, monkeypatch):
        # The one shape that does not carry across: AGENT.md writes a LIST,
        # the portal's zod schema is `z.record(z.unknown())`.
        payload = self._dry_run(tmp_path, monkeypatch)
        assert payload["skills"] == {
            "memory": {},
            "mcp": {"servers": ["filesystem"]},
        }

    def test_it_sends_the_fields_the_schema_declares(self, tmp_path, monkeypatch):
        payload = self._dry_run(tmp_path, monkeypatch)
        assert payload["name"] == "shipper"
        assert payload["description"] == "Ships things"
        assert payload["model"] == "openai/gpt-4o-mini"
        assert payload["intents"] == ["ship the thing"]
        assert "Ship it." in payload["instructions"]

    def test_absent_fields_are_omitted_rather_than_sent_empty(self, tmp_path, monkeypatch):
        payload = self._dry_run(
            tmp_path, monkeypatch, "---\nname: bare\n---\n\nBody.\n"
        )
        assert "skills" not in payload
        assert "intents" not in payload

    def test_dry_run_works_without_being_logged_in(self, tmp_path, monkeypatch):
        # It sends nothing, and checking your payload before you log in is
        # exactly when you want it. The auth gate used to fire first.
        monkeypatch.delenv("WEBAGENTS_TOKEN", raising=False)
        payload = self._dry_run(tmp_path, monkeypatch)
        assert payload["name"] == "shipper"

    def test_dry_run_names_the_method_and_route(self, tmp_path, monkeypatch):
        (tmp_path / "AGENT.md").write_text(AGENT_WITH_SKILLS)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["publish", "--dry-run"])
        assert "Would POST " in result.output and "/api/agents:" in result.output


class TestPublishRefusesToGuess:
    def test_an_unlinked_directory_is_asked_before_creating(self, tmp_path, monkeypatch):
        # A platform username is minted once and cannot be renamed, so an
        # accidental create leaves a handle you are stuck with. Nobody can
        # answer here (no terminal), so it is refused without --yes.
        (tmp_path / "AGENT.md").write_text(AGENT_WITH_SKILLS)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WEBAGENTS_TOKEN", "fake-token-for-the-auth-gate")
        sent = _fake_portal(monkeypatch, {})

        result = runner.invoke(app, ["publish"])
        assert result.exit_code == 1
        assert "Publishing creates a new agent named shipper. Its username is set once and cannot be changed." in result.output
        assert "Pass --yes to create it without a prompt." in result.output
        assert sent == []

    def test_no_agent_file_is_a_clear_refusal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["publish", "--dry-run"])
        assert result.exit_code == 1
        assert "No AGENT.md or agent.json at" in result.output
        assert "webagents init" in result.output


class _FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _fake_portal(monkeypatch, body: dict, status: int = 201) -> list:
    """`httpx.AsyncClient` answering every request with `body`; returns what was sent."""
    import httpx

    sent = []

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, json=None, headers=None):
            sent.append((method, url, json))
            return _FakeResponse(status, body)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return sent


class TestTheAgentApiKeyIsNotLost:
    """`POST /api/agents` answers `{agent: {...}, rawApiKey}` and the platform
    returns that key ONCE. The old `deploy` dropped it on the floor, so
    creating an agent through the CLI meant regenerating the key."""

    def _publish_creating(self, tmp_path, monkeypatch, response):
        (tmp_path / "AGENT.md").write_text(AGENT_WITH_SKILLS)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WEBAGENTS_TOKEN", "fake-token-for-the-auth-gate")
        _fake_portal(monkeypatch, response)
        return runner.invoke(app, ["publish", "--yes"])

    def test_the_key_is_stored_rather_than_discarded(self, tmp_path, monkeypatch):
        stored = {}

        class _Store:
            def set(self, name, value):
                stored[name] = value
                return "keystore"

        import webagents.cli.commands.secrets as secrets_cmd

        monkeypatch.setattr(secrets_cmd, "_store", lambda quiet=False: _Store())

        result = self._publish_creating(
            tmp_path, monkeypatch,
            {"agent": {"id": "a-1", "username": "vs.shipper"}, "rawApiKey": "rok_secret"},
        )

        assert result.exit_code == 0, result.output
        assert "rok_secret" in stored.values()
        # And NOT printed: a credential in scrollback is the mistake S-212 fixed.
        assert "rok_secret" not in result.output
        assert "Stored its API key as AGENT_KEY_VS_SHIPPER (your keychain)." in result.output

    def test_a_store_failure_is_loud_because_the_key_cannot_be_recovered(
        self, tmp_path, monkeypatch
    ):
        class _Broken:
            def set(self, name, value):
                raise RuntimeError("keychain locked")

        import webagents.cli.commands.secrets as secrets_cmd

        monkeypatch.setattr(secrets_cmd, "_store", lambda quiet=False: _Broken())

        result = self._publish_creating(
            tmp_path, monkeypatch,
            {"agent": {"id": "a-1", "username": "vs.shipper"}, "rawApiKey": "rok_secret"},
        )

        assert "Could not store the agent's API key: keychain locked" in result.output
        assert "cannot be recovered" in result.output

    def test_a_create_records_the_binding(self, tmp_path, monkeypatch):
        # Otherwise the next publish mints a SECOND agent, which is the whole
        # thing `link` exists to prevent.
        from webagents.cli.publish import LINK_ID_KEY

        self._publish_creating(tmp_path, monkeypatch, {"agent": {"id": "agent-xyz", "username": "vs.shipper"}})

        written = json.loads((tmp_path / ".webagents" / "config.json").read_text())
        assert written[LINK_ID_KEY] == "agent-xyz"

    def test_the_next_publish_updates_it_without_renaming(self, tmp_path, monkeypatch):
        # PATCH to the linked agent, and never `name`: a platform username is
        # minted once, and a rename moved `me.shipper` to `me.shipper-2`.
        self._publish_creating(tmp_path, monkeypatch, {"agent": {"id": "agent-xyz", "username": "vs.shipper"}})
        sent = _fake_portal(monkeypatch, {"agent": {"username": "vs.shipper"}}, status=200)

        result = runner.invoke(app, ["publish"])
        assert result.exit_code == 0, result.output
        method, url, body = sent[-1]
        assert method == "PATCH" and url.endswith("/api/agents/agent-xyz")
        assert "name" not in body
        assert "Updated vs.shipper." in result.output


class TestTheLinkBinding:
    def test_link_show_reports_an_unlinked_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["link", "--show"])
        assert result.exit_code == 0
        assert "This folder is not linked to an agent on Robutler." in result.output

    def test_the_link_is_read_from_the_project_only(self, tmp_path, monkeypatch):
        # A `link.agentId` in the GLOBAL config made every unlinked folder on
        # the machine update that one agent.
        from webagents.cli.config_store import ConfigStore
        from webagents.cli.publish import LINK_ID_KEY, project_link

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        ConfigStore(cwd=tmp_path).set(LINK_ID_KEY, "agent-global", scope="global")
        assert project_link(tmp_path) == {}
        ConfigStore(cwd=tmp_path).set(LINK_ID_KEY, "agent-here", scope="project")
        assert project_link(tmp_path) == {"agent_id": "agent-here"}


class TestTheClientPointsAtRoutesThatExist:
    def test_no_verified_method_still_calls_v1(self):
        # Every call used to go to `/v1/*`, which exists nowhere in the portal
        # (`find app -type d -name v1` returns nothing), and update used PUT
        # where the portal has PATCH.
        source = (
            Path(__file__).resolve().parents[2]
            / "webagents" / "cli" / "platform" / "api.py"
        ).read_text()

        for method in ("get_user", "list_agents", "get_agent", "register_agent",
                       "update_agent", "delete_agent"):
            body = source.split(f"async def {method}(")[1].split("async def ")[0]
            assert "/v1/" not in body, f"{method} still calls /v1"
            assert "/api/" in body, f"{method} does not call /api"

    def test_update_uses_patch(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "webagents" / "cli" / "platform" / "api.py"
        ).read_text()
        body = source.split("async def update_agent(")[1].split("async def ")[0]
        assert ".patch(" in body and ".put(" not in body


class TestProfileIsolationOfSecrets:
    """S-219, pinned."""

    def test_the_namespace_carries_the_profile(self):
        from webagents.cli.config_store import scoped_namespace

        assert scoped_namespace("cli", None) == "cli"
        assert scoped_namespace("cli", "ci") == "cli-ci"

    def test_the_default_profile_keeps_the_bare_name(self):
        # So a token stored before the fix stays readable from the profile
        # that wrote it.
        from webagents.cli.config_store import scoped_namespace

        assert scoped_namespace("cli") == "cli"

    def test_credentials_scopes_the_namespace_not_only_the_directory(self):
        # The keychain is keyed by namespace ALONE, so scoping only
        # `secrets_dir` isolated two profiles on a machine with no keystore
        # and shared one entry on a machine with one.
        source = (
            Path(__file__).resolve().parents[2]
            / "webagents" / "cli" / "credentials.py"
        ).read_text()
        body = source.split("def _store(")[1].split("\ndef ")[0]
        assert "scoped_namespace" in body
        assert "namespace=CLI_NAMESPACE," not in body

    def test_provider_secrets_are_scoped_the_same_way(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "webagents" / "cli" / "commands" / "secrets.py"
        ).read_text()
        body = source.split("def _store(")[1].split("\ndef ")[0]
        assert "scoped_namespace" in body


class TestSecretsIsHonestAboutWhatItCanDo:
    def test_it_never_takes_a_value_as_an_argument(self):
        # An argument lands in shell history and in the process list, where
        # every other user on the machine can read it.
        import inspect

        from webagents.cli.commands.secrets import set_secret

        assert "value" not in inspect.signature(set_secret).parameters

    def test_the_environment_wins_over_the_store(self, monkeypatch):
        # So adding a key here can never change the behaviour of a shell that
        # was already exporting one.
        from webagents.cli.commands.secrets import load_into_environment

        monkeypatch.setenv("OPENAI_API_KEY", "from-the-shell")
        load_into_environment()
        assert os.environ["OPENAI_API_KEY"] == "from-the-shell"

    def test_the_known_variables_come_from_the_provider_registry(self):
        # Not a second hardcoded list: a provider added to the registry shows
        # up here without anyone remembering to.
        from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS
        from webagents.cli.commands.secrets import _known_env_vars

        known = _known_env_vars()
        for provider in LLM_PROVIDERS:
            for env_var in provider.env_vars:
                # Keys only: the Robutler socket's URL is not a secret.
                assert (env_var in known) == (provider.credential == "api_key")


class TestImportingTheSdkDoesNotRewriteTheEnvironment:
    """S-216, pinned.

    Importing `webagents.cli.main` used to add 37 variables to `os.environ`
    (`OPENAI_API_KEY`, `STRIPE_SECRET_KEY`, `POSTGRES_URL`, ...) from whatever
    `.env` a `find_dotenv()` walk happened to reach. The cause was an eager
    import chain into `litellm`, which calls `load_dotenv()` in its own
    module body.
    """

    def test_the_platform_skills_package_does_not_import_eagerly(self):
        # Deferring THIS file is what breaks the chain. If someone restores
        # the eager imports, the environment rewrite comes back.
        source = (
            Path(__file__).resolve().parents[2]
            / "webagents" / "agents" / "skills" / "robutler" / "__init__.py"
        ).read_text()

        assert "__getattr__" in source, "the lazy resolver is gone"
        for eager in ("from .auth import", "from .crm import", "from .payments import"):
            assert eager not in source, f"eager import restored: {eager}"

    def test_every_exported_name_still_resolves(self):
        # Laziness must not cost anyone a name.
        import webagents.agents.skills.robutler as platform_skills

        for name in platform_skills.__all__:
            assert getattr(platform_skills, name) is not None, name

    def test_dir_still_answers(self):
        # Star-imports and tab completion rely on it.
        import webagents.agents.skills.robutler as platform_skills

        assert set(platform_skills.__all__) <= set(dir(platform_skills))

    def test_an_unknown_name_still_raises_attribute_error(self):
        import webagents.agents.skills.robutler as platform_skills

        with pytest.raises(AttributeError):
            platform_skills.NoSuchSkill

    def test_the_sdk_calls_load_dotenv_nowhere(self):
        # An SDK that rewrites its host's environment on import is wrong
        # regardless of which dependency also does it.
        #
        # Parsed, not grepped: the modules that FIXED this describe the old
        # chain in their docstrings, and a text search cannot tell prose about
        # `load_dotenv()` from a call to it.
        import ast

        root = Path(__file__).resolve().parents[2] / "webagents"
        offenders = []
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "load_dotenv":
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")

        assert offenders == [], f"load_dotenv() is called in the SDK: {offenders}"


class TestAKeyIssuedOnceCanBeReadBack:
    """Found deploying to a real cluster, 2026-09-24.

    `deploy` stored the agent API key and printed "Read it with `webagents
    secrets list`". `list` shows NAMES and never values, and in keystore mode
    it showed nothing at all, because the index it reads is written by
    `note_index` and neither `deploy` nor `secrets set` called it. So the one
    credential the platform does not reissue went into the login Keychain and
    the CLI could neither list it nor hand it back.

    These use the REAL store rather than a stub. The existing test above fakes
    `_store` with a class that only has `set`, which is exactly why the index
    and the read-back path were never exercised.
    """

    def _real_store_in(self, tmp_path, monkeypatch, keystore: bool):
        """Point the CLI's provider store at tmp_path, on either backend."""
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
        if keystore:
            from tests.skills.local.test_secrets_skill import FakeKeyring

            # A fake keychain, so the real one is never asked (or offered to be reset).
            monkeypatch.delenv("WEBAGENTS_SECRETS_BACKEND", raising=False)

            fake = FakeKeyring()
            monkeypatch.setattr(
                "webagents.agents.skills.local.secrets.store._load_keyring",
                lambda: (fake, "tests.FakeKeyring"),
            )
        else:
            monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")

    def test_publish_names_a_command_that_can_actually_read_it(self, tmp_path, monkeypatch):
        self._real_store_in(tmp_path, monkeypatch, keystore=True)
        (tmp_path / "AGENT.md").write_text(AGENT_WITH_SKILLS)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("WEBAGENTS_TOKEN", "fake-token-for-the-auth-gate")
        _fake_portal(monkeypatch, {"agent": {"id": "a-1", "username": "vs.shipper"}, "rawApiKey": "rok_once"})

        result = runner.invoke(app, ["publish", "--yes"])
        assert result.exit_code == 0, result.output

        # It pointed at `secrets list`, which cannot show a value at all.
        assert "secrets get" in result.output
        assert "rok_once" not in result.output

        name = "AGENT_KEY_VS_SHIPPER"
        # The name is now listed...
        listed = runner.invoke(app, ["secrets", "list"])
        assert name in listed.output
        # ...and the value comes back, which is the whole point.
        shown = runner.invoke(app, ["secrets", "get", name, "--show"])
        assert shown.exit_code == 0
        assert "rok_once" in shown.output

    def test_get_redacts_unless_asked(self, tmp_path, monkeypatch):
        self._real_store_in(tmp_path, monkeypatch, keystore=True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-dummy")
        runner.invoke(app, ["secrets", "set", "OPENAI_API_KEY"])

        result = runner.invoke(app, ["secrets", "get", "OPENAI_API_KEY"])
        assert result.exit_code == 0
        assert "sk-dummy" not in result.output
        assert "--show" in result.output

    def test_get_is_non_zero_when_absent(self, tmp_path, monkeypatch):
        self._real_store_in(tmp_path, monkeypatch, keystore=True)
        result = runner.invoke(app, ["secrets", "get", "OPENAI_API_KEY"])
        assert result.exit_code == 1

    def test_secrets_set_is_listed_afterwards(self, tmp_path, monkeypatch):
        """The CLI's own `set` forgot the index too, not just `deploy`."""
        self._real_store_in(tmp_path, monkeypatch, keystore=True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-dummy")
        runner.invoke(app, ["secrets", "set", "ANTHROPIC_API_KEY"])

        result = runner.invoke(app, ["secrets", "list"])
        assert "ANTHROPIC_API_KEY" in result.output

    def test_an_empty_list_does_not_claim_certainty(self, tmp_path, monkeypatch):
        """It said "No provider keys stored" while a keychain held some.

        The caveat has to survive a non-empty list too: another test in this
        file leaves provider keys in the environment, so the table branch is
        the one that runs here, and it was silent.
        """
        self._real_store_in(tmp_path, monkeypatch, keystore=True)

        result = runner.invoke(app, ["secrets", "list"])
        assert "An OS keychain cannot be listed" in result.output

    def test_the_file_backend_makes_no_such_caveat(self, tmp_path, monkeypatch):
        self._real_store_in(tmp_path, monkeypatch, keystore=False)

        result = runner.invoke(app, ["secrets", "list"])
        assert "cannot be listed" not in result.output
