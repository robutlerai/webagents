"""
Tests for SecretsSkill and its backends.

The two paths these tests are FOR are the fallback and the delete, because
those are the ones a developer only meets when something has already gone
wrong. The happy keystore path is covered too, but through an injected fake:
a test that reaches the real macOS Keychain passes on one laptop and is
meaningless in CI, and would also write a credential into a human's login
keychain to prove a point.

The fallback is reachable on any machine because ``WEBAGENTS_SECRETS_BACKEND``
/ ``backend="file"`` selects it deliberately. Before that option existed the
plaintext path could only be exercised on a box with no keystore, which is
precisely the box nobody runs tests on.
"""

import json
import logging
import stat

import pytest

from webagents.agents.skills.local.secrets.skill import SecretsSkill
from webagents.agents.skills.local.secrets.store import (
    KeystoreUnavailableError,
    SecretStore,
    open_secret_store,
    service_key,
)

# Obvious dummies. Nothing here is or resembles a real credential.
DUMMY = "dummy-value-not-a-real-secret"
OTHER_DUMMY = "another-dummy-value-not-a-real-secret"


@pytest.fixture
def warnings():
    """Capture this module's warnings by attaching a handler to its logger.

    NOT ``caplog``: ``webagents/utils/logging.py`` sets ``propagate = False``
    across the ``webagents`` logger tree, so records never reach the root
    handler pytest installs and ``caplog.records`` comes back empty while the
    warnings are plainly visible on stdout. A handler on the logger itself is
    the only reliable way to assert on them here.
    """
    logger = logging.getLogger("webagents.skills.secrets")
    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collector(level=logging.WARNING)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


class FakeKeyring:
    """A fake ``keyring`` module over a dict, keyed the way the real one is.

    It lets the keystore path be asserted on every platform, and it lets a
    test prove a delete swept the FILE while the keystore was live, which is
    the case a real keychain would make miserable to set up.
    """

    def __init__(self):
        self.items = {}

    def get_password(self, service, name):
        return self.items.get((service, name))

    def set_password(self, service, name, value):
        self.items[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.items:
            raise RuntimeError("no such password")
        del self.items[(service, name)]


def file_store(tmp_path, namespace="test-agent", **kwargs):
    return open_secret_store(
        namespace=namespace, secrets_dir=str(tmp_path), backend="file", **kwargs
    )


def keystore_store(tmp_path, keyring, namespace="test-agent", quiet=False):
    return SecretStore(
        namespace=namespace,
        keyring_module=keyring,
        unavailable_reason="tests.FakeKeyring",
        file_path=tmp_path / f"{namespace}.json",
        quiet=quiet,
    )


# ---------------------------------------------------------------------------
# File fallback
# ---------------------------------------------------------------------------


def test_file_fallback_round_trip(tmp_path):
    store = file_store(tmp_path)
    assert store.set("api_key", DUMMY) == "file"
    assert store.get("api_key") == DUMMY

    status = store.status()
    assert status["backend"] == "file"
    assert status["keystore"] is False
    assert status["path"] == str(tmp_path / "test-agent.json")


def test_file_fallback_says_why_and_where(tmp_path):
    status = file_store(tmp_path).status()
    assert "WEBAGENTS_SECRETS_BACKEND" in status["reason"]
    assert "NOT in an OS keystore" in status["warning"]
    assert "PLAINTEXT" in status["warning"]
    assert status["path"] in status["warning"]


def test_file_fallback_warns_on_open_and_on_every_write(tmp_path, warnings):
    store = file_store(tmp_path)
    store.set("api_key", DUMMY)
    store.set("other_key", OTHER_DUMMY)

    assert any("NOT in an OS keystore" in line for line in warnings)
    assert len([line for line in warnings if "PLAINTEXT to" in line]) == 2
    assert any("api_key" in line for line in warnings)
    # The whole point: names travel, values do not.
    assert not any(DUMMY in line for line in warnings)
    assert not any(OTHER_DUMMY in line for line in warnings)


def test_file_is_owner_only_inside_an_owner_only_directory(tmp_path):
    store = file_store(tmp_path / "nested")
    store.set("api_key", DUMMY)
    path = tmp_path / "nested" / "test-agent.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_repairs_a_world_readable_file_left_by_an_earlier_write(tmp_path):
    path = tmp_path / "test-agent.json"
    path.write_text("{}")
    path.chmod(0o644)
    file_store(tmp_path).set("api_key", DUMMY)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_file_does_not_stop_the_agent(tmp_path):
    (tmp_path / "test-agent.json").write_text("not json at all")
    store = file_store(tmp_path)
    assert store.get("api_key") is None
    store.set("api_key", DUMMY)
    assert store.get("api_key") == DUMMY


def test_file_persists_across_store_instances(tmp_path):
    file_store(tmp_path).set("api_key", DUMMY)
    assert file_store(tmp_path).get("api_key") == DUMMY


def test_list_returns_names_never_values(tmp_path):
    store = file_store(tmp_path)
    store.set("b_key", DUMMY)
    store.set("a_key", OTHER_DUMMY)
    names, complete = store.list()
    assert (names, complete) == (["a_key", "b_key"], True)
    assert DUMMY not in json.dumps(names)


# ---------------------------------------------------------------------------
# Namespacing
# ---------------------------------------------------------------------------


def test_two_agents_on_one_machine_do_not_collide(tmp_path):
    file_store(tmp_path, "agent-one").set("platform_token", DUMMY)
    assert file_store(tmp_path, "agent-two").get("platform_token") is None
    assert file_store(tmp_path, "agent-one").get("platform_token") == DUMMY


def test_service_key_scopes_the_keystore_the_same_way():
    assert service_key("agent-one") == "webagents:agent-one"
    assert service_key("agent-one") != service_key("agent-two")


def test_keystore_entries_are_separated_by_namespace(tmp_path):
    keyring = FakeKeyring()
    one = keystore_store(tmp_path, keyring, "agent-one", quiet=True)
    two = keystore_store(tmp_path, keyring, "agent-two", quiet=True)
    one.set("platform_token", DUMMY)
    two.set("platform_token", OTHER_DUMMY)
    assert len(keyring.items) == 2
    assert one.get("platform_token") == DUMMY
    assert two.get("platform_token") == OTHER_DUMMY


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_and_reports_whether_anything_went(tmp_path):
    store = file_store(tmp_path)
    store.set("api_key", DUMMY)
    assert store.delete("api_key") is True
    assert store.get("api_key") is None
    assert store.delete("api_key") is False


def test_delete_leaves_no_trace_of_the_value(tmp_path):
    store = file_store(tmp_path)
    store.set("api_key", DUMMY)
    store.delete("api_key")
    raw = (tmp_path / "test-agent.json").read_text()
    assert DUMMY not in raw
    assert json.loads(raw) == {}


def test_delete_touches_only_the_named_secret(tmp_path):
    store = file_store(tmp_path)
    store.set("api_key", DUMMY)
    store.set("keep_me", OTHER_DUMMY)
    store.delete("api_key")
    assert store.get("keep_me") == OTHER_DUMMY


def test_delete_sweeps_a_plaintext_copy_written_before_the_keystore_existed(tmp_path):
    # The exact sequence that motivates sweeping both: a headless boot wrote
    # the file, the machine later grew a keystore, and a delete that only
    # cleared the active backend would leave the plaintext behind.
    file_store(tmp_path, quiet=True).set("platform_token", DUMMY)

    after = keystore_store(tmp_path, FakeKeyring(), quiet=True)
    assert after.keystore is True
    assert after.delete("platform_token") is True
    assert DUMMY not in (tmp_path / "test-agent.json").read_text()


# ---------------------------------------------------------------------------
# Keystore backend
# ---------------------------------------------------------------------------


def test_keystore_round_trip_touches_no_file(tmp_path, warnings):
    keyring = FakeKeyring()
    store = keystore_store(tmp_path, keyring)
    assert store.set("platform_token", DUMMY) == "keystore"
    assert store.get("platform_token") == DUMMY

    assert keyring.items[("webagents:test-agent", "platform_token")] == DUMMY
    assert warnings == []
    assert not (tmp_path / "test-agent.json").exists()


def test_keystore_status_carries_no_warning_and_no_path(tmp_path):
    status = keystore_store(tmp_path, FakeKeyring()).status()
    assert status["backend"] == "keystore"
    assert status["keystore"] is True
    assert "warning" not in status
    assert "path" not in status


def test_keystore_returns_none_for_a_name_it_does_not_hold(tmp_path):
    assert keystore_store(tmp_path, FakeKeyring()).get("never_set") is None


def test_keystore_list_admits_it_may_be_short(tmp_path):
    store = keystore_store(tmp_path, FakeKeyring())
    store.set("platform_token", DUMMY)
    store.note_index("platform_token", True)
    assert store.list() == (["platform_token"], False)

    store.delete("platform_token")
    store.note_index("platform_token", False)
    assert store.list() == ([], False)


def test_keystore_index_holds_no_secret_material(tmp_path):
    store = keystore_store(tmp_path, FakeKeyring())
    store.set("platform_token", DUMMY)
    store.note_index("platform_token", True)
    raw = (tmp_path / "test-agent.index.json").read_text()
    assert "platform_token" in raw
    assert DUMMY not in raw


# ---------------------------------------------------------------------------
# Refusing the fallback
# ---------------------------------------------------------------------------


def test_require_keystore_throws_rather_than_writing_plaintext(tmp_path):
    with pytest.raises(KeystoreUnavailableError) as excinfo:
        file_store(tmp_path, require_keystore=True)
    assert "WEBAGENTS_SECRETS_REQUIRE_KEYSTORE" in str(excinfo.value)
    assert "webagents[keyring]" in str(excinfo.value)
    assert not (tmp_path / "test-agent.json").exists()


def test_require_keystore_honours_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAGENTS_SECRETS_REQUIRE_KEYSTORE", "1")
    with pytest.raises(KeystoreUnavailableError):
        open_secret_store(
            namespace="test-agent", secrets_dir=str(tmp_path), backend="file"
        )


def test_environment_can_select_the_file_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("WEBAGENTS_SECRETS_DIR", str(tmp_path))
    store = open_secret_store(namespace="test-agent", quiet=True)
    assert store.keystore is False
    assert store.status()["path"] == str(tmp_path / "test-agent.json")


# ---------------------------------------------------------------------------
# Names and values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["", "has space", "has/slash", "a" * 129, "sneaky\nnewline"]
)
def test_bad_names_are_rejected_rather_than_mangled(tmp_path, name):
    with pytest.raises(ValueError, match="invalid secret name"):
        file_store(tmp_path, quiet=True).get(name)


@pytest.mark.parametrize("name", ["platform_token", "openai.api-key", "A_1"])
def test_names_a_credential_actually_takes_are_accepted(tmp_path, name):
    store = file_store(tmp_path, quiet=True)
    store.set(name, DUMMY)
    assert store.get(name) == DUMMY


def test_empty_value_is_refused_rather_than_stored_as_a_hole(tmp_path):
    with pytest.raises(ValueError, match="empty value"):
        file_store(tmp_path, quiet=True).set("api_key", "")


# ---------------------------------------------------------------------------
# The skill surface
# ---------------------------------------------------------------------------


def skill(tmp_path, **extra):
    return SecretsSkill(
        {
            "namespace": "skill-agent",
            "secrets_dir": str(tmp_path),
            "backend": "file",
            "quiet": True,
            **extra,
        }
    )


@pytest.mark.asyncio
async def test_every_result_carries_the_backend_and_the_warning(tmp_path):
    result = await skill(tmp_path).secrets_set("api_key", DUMMY)
    assert result["backend"] == "file"
    assert result["keystore"] is False
    # The LLM sees this even when nobody is reading the process log.
    assert "PLAINTEXT" in result["warning"]


@pytest.mark.asyncio
async def test_get_reports_existence_without_returning_the_value(tmp_path):
    s = skill(tmp_path)
    await s.secrets_set("api_key", DUMMY)
    got = await s.secrets_get("api_key")
    assert got["exists"] is True
    assert "value" not in got
    assert DUMMY not in json.dumps(got)


@pytest.mark.asyncio
async def test_reveal_is_refused_by_default_and_says_why(tmp_path):
    s = skill(tmp_path)
    await s.secrets_set("api_key", DUMMY)
    got = await s.secrets_get("api_key", reveal=True)
    assert "value" not in got
    assert "reveal refused" in got["note"]


@pytest.mark.asyncio
async def test_reveal_works_only_when_the_developer_opted_in(tmp_path):
    s = skill(tmp_path, allow_reveal=True)
    await s.secrets_set("api_key", DUMMY)
    assert (await s.secrets_get("api_key", reveal=True))["value"] == DUMMY


@pytest.mark.asyncio
async def test_a_missing_secret_is_absent_rather_than_an_error(tmp_path):
    got = await skill(tmp_path).secrets_get("never_set")
    assert got["exists"] is False
    assert "value" not in got


@pytest.mark.asyncio
async def test_delete_through_the_tool(tmp_path):
    s = skill(tmp_path)
    await s.secrets_set("api_key", DUMMY)
    assert (await s.secrets_delete("api_key"))["removed"] is True
    assert (await s.secrets_get("api_key"))["exists"] is False
    assert (await s.secrets_delete("api_key"))["removed"] is False


@pytest.mark.asyncio
async def test_list_through_the_tool(tmp_path):
    s = skill(tmp_path)
    await s.secrets_set("api_key", DUMMY)
    listed = await s.secrets_list()
    assert listed["names"] == ["api_key"]
    assert DUMMY not in json.dumps(listed)


@pytest.mark.asyncio
async def test_status_names_the_reason_and_the_path(tmp_path):
    status = await skill(tmp_path).secrets_status()
    assert status["keystore"] is False
    assert status["reason"]
    assert "skill-agent.json" in status["path"]


def test_every_tool_is_owner_scoped(tmp_path):
    info = skill(tmp_path).get_skill_info()
    assert info["tools"] == [
        "secrets_delete",
        "secrets_get",
        "secrets_list",
        "secrets_set",
        "secrets_status",
    ]
    assert info["scope"] == "owner"


@pytest.mark.asyncio
async def test_initialize_defaults_the_namespace_to_the_agent_name(tmp_path):
    class FakeAgent:
        name = "agent-from-initialize"

        def register_tool(self, *args, **kwargs):
            pass

    s = SecretsSkill(
        {"secrets_dir": str(tmp_path), "backend": "file", "quiet": True}
    )
    await s.initialize(FakeAgent())
    assert s.get_store().namespace == "agent-from-initialize"


@pytest.mark.asyncio
async def test_initialize_warns_rather_than_waiting_for_the_first_tool_call(
    tmp_path, warnings
):
    class FakeAgent:
        name = "noisy-agent"

    s = SecretsSkill({"secrets_dir": str(tmp_path), "backend": "file"})
    await s.initialize(FakeAgent())
    assert any("NOT in an OS keystore" in line for line in warnings)
