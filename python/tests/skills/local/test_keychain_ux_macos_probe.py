"""
The no-dialog read, proved on the REAL macOS keychain (the keychain-ux lane,
2026-09-27). `test_keychain_ux_store.py` covers the logic with a fake; this
file proves the one thing a fake cannot: that with nobody to answer, a read
macOS would ask about returns at once instead of showing a dialog, through the
store's own code and the real `keyring` backend.

SAFETY, because this touches the person's login keychain:
  * Only items this file creates, named `webagents (Python) keychain-ux-probe-<hex>`
    or `webagents:keychain-ux-probe-<hex>` for a fresh hex each test. Nothing
    else is read, listed or touched.
  * Every keychain call runs in a CHILD process under a timeout, so a call that
    unexpectedly asked would be killed and the test would fail, not wait.
  * A creator deleting its own item needs no authorization, so every item is
    removed in a `finally` by the program that made it (`/usr/bin/security` for
    the foreign ones, this interpreter for its own), and its absence is checked
    with an attribute-only lookup, which cannot ask.
  * The children run with the REAL home: with HOME pointed at a scratch folder
    macOS finds no default keychain, and an add would show its "keychain cannot
    be found" prompt (a suite hung on that on 2026-09-27). The whole file skips
    unless `security default-keychain` answers for the real home.
"""

import json
import os
import pwd
import subprocess
import sys
import textwrap
import uuid

import pytest

SECURITY = "/usr/bin/security"
REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
TIMEOUT = 30


def _child_env():
    env = {key: value for key, value in os.environ.items() if key not in ("WEBAGENTS_SECRETS_BACKEND", "WEBAGENTS_PROFILE")}
    env["HOME"] = REAL_HOME
    return env


def _real_keychain_here() -> bool:
    if sys.platform != "darwin" or not os.path.exists(SECURITY):
        return False
    try:
        import keyring

        if type(keyring.get_keyring()).__module__ != "keyring.backends.macOS":
            return False
        done = subprocess.run([SECURITY, "default-keychain", "-d", "user"], env=_child_env(), capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - no keyring, or no keychain: skip
        return False
    return done.returncode == 0


pytestmark = pytest.mark.skipif(not _real_keychain_here(), reason="needs macOS with a default keychain and the keyring package")

CHILD = textwrap.dedent(
    r"""
    import json, os, sys, time
    from pathlib import Path

    import keyring

    from webagents.agents.skills.local.secrets import keychain_ux as kx
    from webagents.agents.skills.local.secrets.store import SecretStore

    op, namespace, folder, interactive = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
    said = []
    kx.reset_for_tests(interactive=interactive, writer=said.append)
    value = os.environ.get("PROBE_VALUE", "")
    started = time.monotonic()
    out = {"op": op}
    if op in ("get", "set", "delete"):
        store = SecretStore(namespace=namespace, keyring_module=keyring, unavailable_reason="probe",
                            file_path=Path(folder) / "secrets" / f"{namespace}.json", quiet=True)
        out["guarded"] = store._access.mac is not None
        if op == "get":
            got = store.get("probe")
            out["found"] = got is not None
            out["value_matches"] = got == value
            out["blocked"] = kx.was_blocked(store._access.service, "probe") or kx.was_blocked(store._access.legacy_service, "probe")
        elif op == "set":
            out["backend"] = store.set("probe", value)
        else:
            out["removed"] = store.delete("probe")
            out["left_behind"] = store.left_behind
    elif op == "raw-set":
        keyring.set_password(namespace, "probe", value)
    elif op == "raw-delete":
        try:
            keyring.delete_password(namespace, "probe")
            out["deleted"] = True
        except Exception as error:
            out["deleted"] = False
    out["said"] = said
    out["secs"] = round(time.monotonic() - started, 3)
    print(json.dumps(out))
    """
)


def child(op, namespace, folder, interactive=False, value=""):
    env = _child_env()
    env["PROBE_VALUE"] = value
    try:
        done = subprocess.run([sys.executable, "-c", CHILD, op, namespace, str(folder), "1" if interactive else "0"], env=env, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{op} on {namespace} did not return within {TIMEOUT}s: a keychain call waited, which is the bug")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def exists(service) -> bool:
    """Attribute-only lookup: no secret requested, so it cannot ask."""
    code = subprocess.run([SECURITY, "find-generic-password", "-s", service, "-a", "probe"], env=_child_env(), capture_output=True, timeout=10).returncode
    assert code in (0, 44), code
    return code == 0


def security_add(service, value):
    done = subprocess.run([SECURITY, "add-generic-password", "-s", service, "-a", "probe", "-l", f"{service} (a test probe, safe to Deny)", "-w", value], env=_child_env(), capture_output=True, timeout=10)
    assert done.returncode == 0, done.stderr


def security_delete(service):
    subprocess.run([SECURITY, "delete-generic-password", "-s", service, "-a", "probe"], env=_child_env(), capture_output=True, timeout=10)


@pytest.fixture
def probe():
    tag = f"keychain-ux-probe-{uuid.uuid4().hex[:12]}"
    return {"namespace": tag, "own": f"webagents (Python) {tag}", "old": f"webagents:{tag}", "value": f"probe-{uuid.uuid4().hex}"}


def test_an_item_another_program_made_is_refused_at_once_not_asked(tmp_path, probe):
    """The proof item 3 asks for: /usr/bin/security makes the item, this
    interpreter reads it with nobody to answer, and gets the one sentence
    instead of a dialog."""
    security_add(probe["own"], probe["value"])
    try:
        got = child("get", probe["namespace"], tmp_path)
        assert got["guarded"] is True
        assert got["found"] is False and got["blocked"] is True
        assert len(got["said"]) == 1 and got["said"][0].startswith("macOS may ask before")
        assert got["secs"] < 5
    finally:
        security_delete(probe["own"])
    assert not exists(probe["own"])


def test_an_item_this_interpreter_made_is_read_with_nobody_to_answer(tmp_path, probe):
    try:
        assert child("set", probe["namespace"], tmp_path, value=probe["value"])["backend"] == "keystore"
        got = child("get", probe["namespace"], tmp_path, value=probe["value"])
        assert got["value_matches"] is True and got["said"] == []
        gone = child("delete", probe["namespace"], tmp_path)
        assert gone["removed"] is True and gone["left_behind"] == []
    finally:
        child("raw-delete", probe["own"], tmp_path)
    assert not exists(probe["own"])


def test_an_old_item_another_program_made_is_never_read_and_is_named_when_removed(tmp_path, probe):
    security_add(probe["old"], probe["value"])
    try:
        got = child("get", probe["namespace"], tmp_path, value=probe["value"])
        assert got["found"] is False and got["blocked"] is True
        assert not exists(probe["own"])
        gone = child("delete", probe["namespace"], tmp_path)
        assert gone["left_behind"] == [{"item": probe["old"], "account": "probe"}]
        assert exists(probe["old"])
    finally:
        security_delete(probe["old"])
        child("raw-delete", probe["own"], tmp_path)
    assert not exists(probe["old"]) and not exists(probe["own"])


def test_an_old_item_this_interpreter_made_is_copied_in_a_terminal_without_asking(tmp_path, probe):
    """Safe with a terminal switched on: this interpreter made the old item, so
    even the read that may ask cannot ask."""
    try:
        child("raw-set", probe["old"], tmp_path, value=probe["value"])
        got = child("get", probe["namespace"], tmp_path, interactive=True, value=probe["value"])
        assert got["value_matches"] is True and got["said"] == []
        assert exists(probe["own"]) and exists(probe["old"])
        gone = child("delete", probe["namespace"], tmp_path, interactive=True)
        assert gone["removed"] is True and gone["left_behind"] == []
    finally:
        child("raw-delete", probe["old"], tmp_path)
        child("raw-delete", probe["own"], tmp_path)
    assert not exists(probe["old"]) and not exists(probe["own"])
