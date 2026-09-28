"""
The daemon keeps its agents' keys where the server keeps every key (S-309,
2026-09-27, the agent-secrets lane): a key an earlier daemon left in the
agent folder's `.webagents/keys` moves into the store on load with its
thumbprint unchanged, a key that git ever tracked is said to need rotation,
two folders sharing an agent name are told they share a key, a conflicting
or unusable legacy file is never moved. The paths and lines are the shared
fixture's (`tests/fixtures/daemon/signedhooks.json`, `store`); the TypeScript
daemon runs the same in `tests/unit/daemon/agent-secrets-daemon-keys.test.ts`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from webagents.cli.daemon import identity as identity_module
from webagents.cli.daemon.identity import (
    DAEMON_KEY_LINES,
    agent_key_origin_file,
    daemon_agent_identity,
    daemon_key_line,
    daemon_keys_dir,
    legacy_daemon_keys_dir,
)
from webagents.crypto.jwks import JWKSManager

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "daemon" / "signedhooks.json").read_text())
STORE = FIXTURE["store"]
PUBLIC_URL = "https://agent.example"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("WEBAGENTS_PUBLIC_URL", "WEBAGENTS_KEYS_DIR", "WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    identity_module._said_shared.clear()


def project(tmp_path: Path, name: str = "project") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "AGENT.md").write_text(FIXTURE["agent_file"].replace("{url}", "https://hook.example/hook"))
    return root


def store_file(agent: str = "reporter") -> Path:
    return Path.home() / STORE["default"] / FIXTURE["key_file"].format(agent=agent)


def legacy_file(root: Path, agent: str = "reporter") -> Path:
    return root / STORE["legacy_dir"] / FIXTURE["key_file"].format(agent=agent)


def mint_legacy(root: Path, agent: str = "reporter") -> str:
    """A key at the old place, minted by the store itself so its bytes are exactly what a daemon wrote."""
    manager = JWKSManager({"keys_dir": str(legacy_daemon_keys_dir(root))})
    return manager.ensure_ed25519_key(agent)


def lines_of(capsys) -> list:
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("[webagents]")]


def test_the_words_and_the_places_are_the_fixture_s(monkeypatch):
    assert DAEMON_KEY_LINES["moved"] == STORE["moved_line"]
    assert DAEMON_KEY_LINES["tracked"] == STORE["tracked_line"]
    assert DAEMON_KEY_LINES["shared"] == STORE["shared_line"]
    assert daemon_keys_dir() == Path.home() / STORE["default"]
    monkeypatch.setenv(STORE["env"], "/mnt/agent-keys")
    assert daemon_keys_dir() == Path("/mnt/agent-keys")
    assert legacy_daemon_keys_dir("/x") == Path("/x") / STORE["legacy_dir"]
    assert agent_key_origin_file("reporter") == STORE["origin_file"].format(agent="reporter")


def test_a_legacy_key_moves_into_the_store_same_thumbprint_and_the_folder_is_left_without_one(tmp_path, capsys):
    root = project(tmp_path)
    kid = mint_legacy(root)
    legacy = legacy_file(root)
    assert legacy.exists()
    capsys.readouterr()

    identity = daemon_agent_identity("reporter", root, PUBLIC_URL)
    assert identity.held_keys()[0].thumbprint == kid
    assert store_file().exists()
    assert not legacy.exists()
    assert not (root / STORE["legacy_dir"]).exists()
    said = lines_of(capsys)
    assert daemon_key_line("moved", agent="reporter", legacy=str(legacy), store=str(store_file())) in said
    assert not any("rotate the key" in line for line in said)

    # The sidecar names this folder; a second load from it says nothing more.
    origin = json.loads((daemon_keys_dir() / agent_key_origin_file("reporter")).read_text())
    assert sorted(origin) == sorted(STORE["origin_keys"])
    assert origin["folder"] == os.path.realpath(root)
    assert daemon_agent_identity("reporter", root, PUBLIC_URL).held_keys()[0].thumbprint == kid
    assert lines_of(capsys) == []


def test_a_key_git_ever_tracked_is_said_to_need_rotation(tmp_path, capsys):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = project(tmp_path)
    mint_legacy(root)
    legacy = legacy_file(root)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example"}
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", "-f", os.path.relpath(legacy, root)], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "oops"], check=True, env=env)
    capsys.readouterr()

    daemon_agent_identity("reporter", root, PUBLIC_URL)
    assert daemon_key_line("tracked", legacy=str(legacy), folder=str(root), store=str(store_file())) in lines_of(capsys)


def test_a_legacy_file_that_is_not_a_key_is_said_and_never_moved(tmp_path):
    root = project(tmp_path)
    legacy = legacy_file(root)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("not a key")
    with pytest.raises(RuntimeError) as caught:
        daemon_agent_identity("reporter", root, PUBLIC_URL)
    assert str(legacy) in str(caught.value)
    assert legacy.read_text() == "not a key"
    assert not store_file().exists()


def test_a_different_key_already_in_the_store_is_a_conflict_and_neither_file_moves(tmp_path):
    root = project(tmp_path)
    mint_legacy(root)
    JWKSManager({}).ensure_ed25519_key("reporter")
    legacy = legacy_file(root)
    before = store_file().read_bytes()
    with pytest.raises(RuntimeError) as caught:
        daemon_agent_identity("reporter", root, PUBLIC_URL)
    assert "two different agent keys" in str(caught.value)
    assert legacy.exists()
    assert store_file().read_bytes() == before


def test_the_store_pointed_at_the_folder_itself_moves_nothing(tmp_path, monkeypatch, capsys):
    root = project(tmp_path)
    monkeypatch.setenv(STORE["env"], str(legacy_daemon_keys_dir(root)))
    kid = mint_legacy(root)
    assert daemon_agent_identity("reporter", root, PUBLIC_URL).held_keys()[0].thumbprint == kid
    assert legacy_file(root).exists()
    assert not any("moved the signing key" in line for line in lines_of(capsys))


def test_two_folders_with_one_agent_name_are_told_they_share_a_key_once(tmp_path, capsys):
    first = project(tmp_path, "first")
    second = project(tmp_path, "second")
    kid = daemon_agent_identity("reporter", first, PUBLIC_URL).held_keys()[0].thumbprint
    capsys.readouterr()
    assert daemon_agent_identity("reporter", second, PUBLIC_URL).held_keys()[0].thumbprint == kid
    expected = daemon_key_line(
        "shared", agent="reporter", store=str(store_file()), origin=os.path.realpath(first), folder=os.path.realpath(second)
    )
    assert lines_of(capsys) == [expected]
    daemon_agent_identity("reporter", second, PUBLIC_URL)
    assert lines_of(capsys) == []
    # The sidecar still names the first folder.
    origin = json.loads((daemon_keys_dir() / agent_key_origin_file("reporter")).read_text())
    assert origin["folder"] == os.path.realpath(first)
