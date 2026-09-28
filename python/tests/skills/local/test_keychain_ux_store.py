"""
Keychain dialogs on macOS, the store's half (the keychain-ux lane, 2026-09-27).

A FAKE macOS keychain drives these, modelled on what the lane measured on a
real one (`webagents-test-artifacts/2026-09-28-keychain-ux/probe-results.json`):
an item trusts the program that made it; any other program gets a dialog, or,
with user interaction switched off, errSecAuthFailed (-25293) for a read and
errSecInvalidOwnerEdit (-25244) for a delete, with no dialog. The real keychain
is exercised only by `test_keychain_ux_macos_probe.py`, with probe items of its
own. Every name and sentence comes from `fixtures/keychain_ux/keychain_ux.json`,
which the TypeScript twin (`tests/unit/skills/keychain-ux-store.test.ts`) reads
too.
"""

import contextlib
import json
import os
import stat
from pathlib import Path

import pytest

from webagents.agents.skills.local.secrets import keychain_ux as kx
from webagents.agents.skills.local.secrets.store import SecretStore, open_secret_store, service_key

FIXTURE = json.loads((Path(__file__).parents[2] / "fixtures" / "keychain_ux" / "keychain_ux.json").read_text())

# Obvious dummies. Nothing here is or resembles a real credential.
DUMMY = "dummy-value-not-a-real-secret"
OTHER_DUMMY = "another-dummy-value-not-a-real-secret"


class FakeApiError(Exception):
    """`keyring`'s macOS `api.Error`: `(OSStatus, message)`."""


class MacWorld:
    """A fake macOS keychain. Items remember the program that made them and
    the programs the person chose Always Allow for."""

    def __init__(self, program: str = "Python 3.14.2") -> None:
        self.items = {}
        self.program = program
        self.ui = True
        self.answer = "always"
        self.events = []
        self.reads = []

    def add(self, service, account, value, creator=None):
        self.items[(service, account)] = {"value": value, "creator": creator or self.program, "trusted": set()}

    def keyring(self):
        return FakeMacKeyring(self)

    def mac(self):
        return FakeMac(self)

    def dialogs(self):
        return [event for event in self.events if event[0] == "dialog"]

    def said(self):
        return [event[1] for event in self.events if event[0] == "say"]


class FakeMacKeyring:
    """The `keyring` calls the store makes, with macOS's answers."""

    def __init__(self, world: MacWorld) -> None:
        self.w = world

    def _trusted(self, item) -> bool:
        return self.w.program == item["creator"] or self.w.program in item["trusted"]

    def _ask(self, op, service, account, item) -> None:
        if not self.w.ui:
            raise RuntimeError(f"Can't {op} password on keychain") from FakeApiError(-25293 if op == "read" else -25244, "Unknown Error")
        self.w.events.append(("dialog", op, service, account))
        if self.w.answer == "deny":
            raise RuntimeError("denied") from FakeApiError(-25293, "Unknown Error")
        item["trusted"].add(self.w.program)

    def get_password(self, service, account):
        self.w.reads.append((service, account, self.w.ui))
        item = self.w.items.get((service, account))
        if item is None:
            return None
        if not self._trusted(item):
            self._ask("read", service, account, item)
        return item["value"]

    def set_password(self, service, account, value):
        # `keyring` replaces an item by deleting it first.
        item = self.w.items.get((service, account))
        if item is not None:
            if not self._trusted(item):
                self._ask("delete", service, account, item)
            del self.w.items[(service, account)]
        self.w.add(service, account, value)

    def delete_password(self, service, account):
        item = self.w.items.get((service, account))
        if item is None:
            raise RuntimeError("Can't delete password in keychain") from FakeApiError(-25300, "Item not found")
        if not self._trusted(item):
            self._ask("delete", service, account, item)
        del self.w.items[(service, account)]


class FakeMac:
    """`MacKeychain`: attribute lookups and the switch that stops dialogs."""

    def __init__(self, world: MacWorld) -> None:
        self.w = world

    def exists(self, service, account=None, cache=False):
        return any(s == service and (account is None or a == account) for (s, a) in self.w.items)

    @contextlib.contextmanager
    def quiet(self):
        previous, self.w.ui = self.w.ui, False
        try:
            yield
        finally:
            self.w.ui = previous


@pytest.fixture(autouse=True)
def fresh_run(monkeypatch):
    """Each test is a new run: nothing said, nothing forbidden, not a terminal."""
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    kx.reset_for_tests(interactive=False)
    yield
    kx.reset_for_tests()


def run(world: MacWorld, interactive: bool) -> None:
    kx.reset_for_tests(interactive=interactive, writer=lambda text: world.events.append(("say", text)))


def mac_store(tmp_path, world: MacWorld, namespace="cli") -> SecretStore:
    secrets = tmp_path / "secrets"
    keyring = world.keyring()
    access = kx.KeychainAccess(keyring, namespace, secrets, mac=world.mac())
    return SecretStore(namespace=namespace, keyring_module=keyring, unavailable_reason="fake macOS", file_path=secrets / f"{namespace}.json", quiet=True, keychain=access)


def record_of(tmp_path) -> dict:
    return json.loads((tmp_path / "keychain.json").read_text())


OWN = "webagents (Python) cli"
OLD = "webagents:cli"


# ---------------------------------------------------------------------------
# Names and words, from the fixture both SDKs read
# ---------------------------------------------------------------------------


class TestTheNames:
    def test_each_sdk_files_its_items_under_its_own_name(self):
        assert FIXTURE["service"]["template"] == kx.SERVICE_TEMPLATE
        assert FIXTURE["service"]["legacy_template"] == kx.LEGACY_SERVICE_TEMPLATE
        assert FIXTURE["service"]["runtimes"] == kx.RUNTIME_LABELS
        for example in FIXTURE["service"]["examples"]:
            assert kx.service_name(example["namespace"], example["runtime"]) == example["service"]
            assert kx.legacy_service_name(example["namespace"]) == example["legacy"]
        assert service_key("cli") == OWN

    def test_the_program_macos_names(self):
        for case in FIXTURE["program_name"]["cases"]:
            assert kx.program_name(case["path"]) == case["program"], case

    def test_what_a_recorded_program_means_for_the_next_read(self):
        for case in FIXTURE["record"]["predictions"]:
            assert kx.prediction(case["recorded"], case["current"]) == case["prediction"], case["about"]

    def test_the_words(self):
        assert list(kx.EXPLANATION) == FIXTURE["explanation"]
        assert len(kx.EXPLANATION) == 4
        assert kx.BLOCKED == FIXTURE["blocked"]
        assert kx.BLOCKED_COMMAND == FIXTURE["blocked_command"]
        assert kx.LEFT_BEHIND == FIXTURE["left_behind"]
        assert kx.DENIED == FIXTURE["denied"]
        assert kx.NO_DEFAULT_KEYCHAIN == FIXTURE["no_default_keychain"]
        assert kx.OTHER_SIGNED_IN == FIXTURE["other_runtime"]["signed_in"]
        assert kx.OTHER_KEYS == FIXTURE["other_runtime"]["keys"]
        assert kx.OTHER_KEYS_COMMAND == FIXTURE["other_runtime"]["keys_command"]
        assert kx.INTERPRETER_NAMES == FIXTURE["names"]["interpreter"]
        assert kx.CLI_NAMES == FIXTURE["names"]["cli"]
        assert kx.RECORD_FILE == FIXTURE["record"]["file"]
        assert kx.RECORD_FILE_ELSEWHERE == FIXTURE["record"]["file_elsewhere"]
        assert kx.RECORD_ABOUT == FIXTURE["record"]["about"]
        assert kx.STATUS_ROW == FIXTURE["status_row"]
        assert {str(k): v for k, v in kx.DIALOG_STATUSES.items()} == FIXTURE["no_quiet_read"]["dialog_statuses"]
        assert kx.ITEM_NOT_FOUND == FIXTURE["no_quiet_read"]["item_not_found"]
        words = {key: value for key, value in FIXTURE["doctor"].items() if key != "cases"}
        assert kx.DOCTOR_WORDS == words

    def test_no_em_dash_in_any_sentence(self):
        text = json.dumps(FIXTURE, ensure_ascii=False)
        assert "—" not in text and "–" not in text


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestTheRecord:
    def test_it_lives_beside_the_items_and_never_above_a_test_folder(self, tmp_path):
        assert kx.record_path(tmp_path / ".webagents-local" / "secrets") == tmp_path / ".webagents-local" / "keychain.json"
        assert kx.record_path(tmp_path / "anything") == tmp_path / "anything" / "keychain.record.json"

    def test_it_is_0600_and_holds_no_value(self, tmp_path):
        world = MacWorld()
        store = mac_store(tmp_path, world)
        store.set("platform_token", DUMMY)
        path = tmp_path / "keychain.json"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        text = path.read_text()
        assert DUMMY not in text
        entry = record_of(tmp_path)["runtimes"]["python"]["items"][OWN]["platform_token"]
        assert sorted(entry) == sorted(FIXTURE["record"]["fields"])
        assert entry["program"] == kx.current_program()["program"]
        assert entry["version"] == kx.current_program()["version"]

    def test_a_looser_record_is_repaired_when_read(self, tmp_path):
        path = tmp_path / "keychain.json"
        path.write_text(json.dumps(FIXTURE["record"]["sample"]))
        os.chmod(path, 0o644)
        kx.KeychainRecord(path).items()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_the_sample_reads_the_same_in_both_sdks(self, tmp_path):
        path = tmp_path / "keychain.json"
        path.write_text(json.dumps(FIXTURE["record"]["sample"]))
        reads = FIXTURE["record"]["sample_reads"]
        record = kx.KeychainRecord(path)
        assert ("platform_token" in record.items().get("webagents (Python) cli", {})) is reads["python_signed_in"]
        assert sorted(record.items().get("webagents (Python) providers", {})) == reads["python_key_names"]
        assert ("platform_token" in record.items("typescript").get("webagents (TypeScript) cli", {})) is reads["typescript_signed_in"]
        pending = [{key: entry[key] for key in ("service", "account", "namespace", "legacy")} for entry in record.pending("typescript")]
        assert pending == reads["typescript_pending"]
        assert record.legacy_done("webagents:cli", "platform_token")


# ---------------------------------------------------------------------------
# Reads: said before, never waited on
# ---------------------------------------------------------------------------


class TestReads:
    def test_an_item_this_interpreter_made_is_read_quietly(self, tmp_path):
        world = MacWorld()
        world.add(OWN, "platform_token", DUMMY)
        run(world, interactive=False)
        assert mac_store(tmp_path, world).get("platform_token") == DUMMY
        assert world.dialogs() == [] and world.said() == []
        # Read with the dialogs switched off, every time.
        assert world.reads == [(OWN, "platform_token", False)]

    def test_nobody_to_answer_is_never_asked(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        world.add(OWN, "OTHER", DUMMY, creator="Python 3.14.2")
        run(world, interactive=False)
        store = mac_store(tmp_path, world)
        assert store.get("platform_token") is None
        assert store.get("OTHER") is None
        assert world.dialogs() == []
        assert all(ui is False for _, _, ui in world.reads)
        # The one sentence, once per run, naming the command that settles it.
        expected = FIXTURE["blocked"].format(program=kx.current_program()["program"], item=OWN, command="webagents whoami")
        assert world.said() == [expected]
        pending = kx.KeychainRecord(tmp_path / "keychain.json").pending()
        assert {(entry["service"], entry["account"], entry["legacy"]) for entry in pending} == {(OWN, "platform_token", False), (OWN, "OTHER", False)}

    def test_a_copy_in_the_file_stands_in_and_nothing_is_said(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        run(world, interactive=False)
        store = mac_store(tmp_path, world)
        store._write_file_map({"platform_token": OTHER_DUMMY})
        assert store.get("platform_token") == OTHER_DUMMY
        assert world.said() == [] and world.dialogs() == []

    def test_a_terminal_hears_the_four_lines_before_the_dialog(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        world.add(OWN, "OTHER", OTHER_DUMMY, creator="Python 3.14.2")
        run(world, interactive=True)
        store = mac_store(tmp_path, world)
        assert store.get("platform_token") == DUMMY
        assert store.get("OTHER") == OTHER_DUMMY
        program = kx.current_program()["program"]
        lines = [line.format(program=program, item=OWN) for line in FIXTURE["explanation"]]
        # Once per run, and BEFORE the first dialog.
        assert world.events[0] == ("say", "\n".join(lines))
        assert [event[0] for event in world.events] == ["say", "dialog", "dialog"]
        # Always Allow: the next run reads quietly.
        world.events.clear()
        run(world, interactive=False)
        assert mac_store(tmp_path, world).get("platform_token") == DUMMY
        assert world.events == []

    def test_serve_and_the_daemon_are_never_asked_even_at_a_terminal(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        run(world, interactive=True)
        kx.forbid_dialogs("serve")
        assert mac_store(tmp_path, world).get("platform_token") is None
        assert world.dialogs() == []
        assert kx.was_blocked(OWN, "platform_token")

    def test_deny_carries_on_without_the_item(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        world.answer = "deny"
        run(world, interactive=True)
        assert mac_store(tmp_path, world).get("platform_token") is None
        assert world.said()[-1] == FIXTURE["denied"].format(program=kx.current_program()["program"], item=OWN)


# ---------------------------------------------------------------------------
# The old shared name
# ---------------------------------------------------------------------------


class TestTheOldSharedName:
    def test_an_old_item_this_interpreter_made_is_copied_quietly_and_left_in_place(self, tmp_path):
        world = MacWorld()
        world.add(OLD, "platform_token", DUMMY)
        run(world, interactive=True)
        store = mac_store(tmp_path, world)
        assert store.get("platform_token") == DUMMY
        assert world.dialogs() == [] and world.said() == []
        assert world.items[(OWN, "platform_token")]["value"] == DUMMY
        assert (OLD, "platform_token") in world.items
        assert kx.KeychainRecord(tmp_path / "keychain.json").legacy_done(OLD, "platform_token")
        assert store.own_index_names() == ["platform_token"]

    def test_an_old_item_another_program_made_asks_after_the_explanation(self, tmp_path):
        world = MacWorld(program="node 24.7.0")
        world.add(OLD, "platform_token", DUMMY, creator="Python 3.14.2")
        run(world, interactive=True)
        assert mac_store(tmp_path, world).get("platform_token") == DUMMY
        assert [event[0] for event in world.events] == ["say", "dialog"]
        assert world.events[0][1].startswith(FIXTURE["explanation"][0].format(program=kx.current_program()["program"], item=OLD))

    def test_a_run_with_nobody_to_answer_never_reads_an_old_item(self, tmp_path):
        world = MacWorld()
        # Made by this very interpreter: a quiet read WOULD succeed, and still
        # none is made, because such a run never reads an old item at all.
        world.add(OLD, "platform_token", DUMMY)
        run(world, interactive=False)
        assert mac_store(tmp_path, world).get("platform_token") is None
        assert [read for read in world.reads if read[0] == OLD] == []
        assert (OWN, "platform_token") not in world.items
        pending = kx.KeychainRecord(tmp_path / "keychain.json").pending()
        assert [(entry["service"], entry["account"], entry["legacy"]) for entry in pending] == [(OWN, "platform_token", True)]
        assert world.said() == [FIXTURE["blocked"].format(program=kx.current_program()["program"], item=OLD, command="webagents whoami")]

    def test_removing_takes_the_old_item_too_when_no_dialog_is_needed(self, tmp_path):
        world = MacWorld()
        world.add(OLD, "platform_token", DUMMY)
        run(world, interactive=True)
        store = mac_store(tmp_path, world)
        assert store.get("platform_token") == DUMMY
        assert store.delete("platform_token") is True
        assert store.left_behind == []
        assert world.items == {}

    def test_an_old_item_only_a_dialog_could_remove_is_named_and_never_read_again(self, tmp_path):
        world = MacWorld(program="node 24.7.0")
        world.add(OLD, "platform_token", DUMMY, creator="Python 3.14.2")
        run(world, interactive=True)
        store = mac_store(tmp_path, world)
        assert store.delete("platform_token") is True
        assert store.left_behind == [{"item": OLD, "account": "platform_token"}]
        assert (OLD, "platform_token") in world.items
        assert world.dialogs() == []
        sentence = FIXTURE["left_behind"].format(item=OLD, account="platform_token")
        assert kx.left_behind_sentence(OLD, "platform_token") == sentence
        # Removed means removed: the next read does not copy it back.
        assert store.get("platform_token") is None
        assert world.dialogs() == []

    def test_list_names_what_the_old_index_holds_until_it_is_copied_or_removed(self, tmp_path):
        world = MacWorld()
        world.add("webagents:providers", "OPENAI_API_KEY", DUMMY)
        store = mac_store(tmp_path, world, namespace="providers")
        (tmp_path / "secrets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "secrets" / "providers.index.json").write_text(json.dumps(["OPENAI_API_KEY"]))
        assert store.list() == (["OPENAI_API_KEY"], False)
        run(world, interactive=True)
        store.delete("OPENAI_API_KEY")
        assert store.list() == ([], False)

    def test_elsewhere_the_old_item_is_copied_in_any_run(self, tmp_path):
        """No per-program dialog outside macOS: the copy needs no terminal."""

        class PlainKeyring:
            def __init__(self):
                self.items = {(OLD, "platform_token"): DUMMY}

            def get_password(self, service, name):
                return self.items.get((service, name))

            def set_password(self, service, name, value):
                self.items[(service, name)] = value

            def delete_password(self, service, name):
                del self.items[(service, name)]

        keyring = PlainKeyring()
        store = SecretStore(namespace="cli", keyring_module=keyring, unavailable_reason="plain", file_path=tmp_path / "secrets" / "cli.json", quiet=True)
        assert store._access.mac is None
        assert store.get("platform_token") == DUMMY
        assert keyring.items[(OWN, "platform_token")] == DUMMY


# ---------------------------------------------------------------------------
# Writes and removals after an upgrade
# ---------------------------------------------------------------------------


class TestWrites:
    def test_replacing_an_item_an_earlier_interpreter_made(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        run(world, interactive=False)
        with pytest.raises(kx.KeychainDialogBlocked) as blocked:
            mac_store(tmp_path, world).set("platform_token", OTHER_DUMMY)
        assert str(blocked.value) == FIXTURE["blocked"].format(program=kx.current_program()["program"], item=OWN, command="webagents whoami")
        assert world.dialogs() == []
        run(world, interactive=True)
        assert mac_store(tmp_path, world).set("platform_token", OTHER_DUMMY) == "keystore"
        assert [event[0] for event in world.events] == ["say", "dialog"]
        assert world.items[(OWN, "platform_token")] == {"value": OTHER_DUMMY, "creator": "Python 3.14.3", "trusted": set()}

    def test_removing_an_item_an_earlier_interpreter_made_with_nobody_to_answer(self, tmp_path):
        world = MacWorld(program="Python 3.14.3")
        world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
        run(world, interactive=False)
        with pytest.raises(kx.KeychainDialogBlocked):
            mac_store(tmp_path, world).delete("platform_token")
        assert (OWN, "platform_token") in world.items
        assert world.dialogs() == []


# ---------------------------------------------------------------------------
# Where the keychain is not used at all
# ---------------------------------------------------------------------------


class TestNoKeychain:
    def test_the_file_backend_makes_no_keychain_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
        store = open_secret_store(namespace="cli", secrets_dir=str(tmp_path), quiet=True)
        assert store.keystore is False
        assert store._access is None

    def test_no_default_keychain_means_the_file(self, tmp_path, monkeypatch):
        """With HOME pointed elsewhere macOS has no default keychain, and the
        first add would show its "keychain cannot be found" prompt and wait (a
        test suite hung on exactly that, 2026-09-27). The file answers instead."""
        keyring = pytest.importorskip("keyring")

        class MacBackend:
            pass

        MacBackend.__module__ = "keyring.backends.macOS"
        monkeypatch.setattr(keyring, "get_keyring", lambda: MacBackend())
        monkeypatch.setattr(kx, "default_keychain_available", lambda: False)
        monkeypatch.delenv("WEBAGENTS_SECRETS_BACKEND", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        store = open_secret_store(namespace="cli", secrets_dir=str(tmp_path), quiet=True)
        assert store.keystore is False
        assert store.status()["reason"] == FIXTURE["no_default_keychain"].format(home=str(tmp_path))


# ---------------------------------------------------------------------------
# `webagents whoami` settles what a background run could not read
# ---------------------------------------------------------------------------


def test_whoami_in_a_terminal_reads_what_a_background_run_could_not(tmp_path):
    world = MacWorld(program="Python 3.14.3")
    world.add(OWN, "platform_token", DUMMY, creator="Python 3.14.2")
    run(world, interactive=False)
    assert mac_store(tmp_path, world).get("platform_token") is None
    record = kx.KeychainRecord(tmp_path / "keychain.json")
    assert len(record.pending()) == 1
    assert [event[0] for event in world.events] == ["say"]

    world.events.clear()
    run(world, interactive=True)
    settled = kx.settle_pending([tmp_path / "keychain.json"], lambda namespace, folder: mac_store(tmp_path, world, namespace))
    assert settled == 1
    assert record.pending() == []
    assert [event[0] for event in world.events] == ["say", "dialog"]
    # The background run reads it quietly now.
    world.events.clear()
    run(world, interactive=False)
    assert mac_store(tmp_path, world).get("platform_token") == DUMMY
    assert world.events == []


# ---------------------------------------------------------------------------
# `doctor`'s `keychain` line
# ---------------------------------------------------------------------------


class TestDoctorLine:
    def test_the_fixture_cases(self):
        for case in FIXTURE["doctor"]["cases"]:
            line = kx.doctor_line(dict(case["facts"]))
            assert line["name"] == "keychain"
            assert line["status"] == case["status"], case["about"]
            assert line["detail"] == case["detail"], case["about"]
            assert line.get("fix") == case.get("fix"), case["about"]

    def test_facts_are_gathered_without_reading_a_value(self, tmp_path):
        world = MacWorld()
        world.add(OWN, "platform_token", DUMMY)
        world.add(OLD, "platform_token", DUMMY)
        store = mac_store(tmp_path, world)
        record = kx.KeychainRecord(tmp_path / "keychain.json")
        me = kx.current_program()
        record.note_used(OWN, "platform_token")
        facts = kx.doctor_facts([store], record, {OLD: ["platform_token"]}, mac=world.mac())
        assert world.reads == []
        assert facts == {"backend": "keychain", "runtime": "python", "count": 1, "interpreter_version": me["version"], "prediction": "quiet", "legacy": True, "pending": 0, "other": False, "env_token": False}
        # An upgrade since: the next use asks once.
        data = record.read()
        data["runtimes"]["python"]["items"][OWN]["platform_token"]["version"] = "3.13.0"
        (tmp_path / "keychain.json").write_text(json.dumps(data))
        assert kx.doctor_facts([store], record, {}, mac=world.mac())["prediction"] == "upgraded"

    def test_the_other_cli_is_named_when_this_one_has_nothing(self, tmp_path):
        world = MacWorld()
        store = mac_store(tmp_path, world)
        path = tmp_path / "keychain.json"
        path.write_text(json.dumps({"runtimes": {"typescript": {"items": {"webagents (TypeScript) cli": {"platform_token": {"program": "node"}}}}}}))
        facts = kx.doctor_facts([store], kx.KeychainRecord(path), {}, mac=world.mac())
        assert facts["count"] == 0 and facts["other"] is True
        assert kx.doctor_line(facts)["detail"] == "macOS keychain: nothing stored by the Python CLI yet; the TypeScript CLI keeps its own items here"
