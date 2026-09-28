"""
The signing identity a daemon-served agent holds (2026-09-27, the
a2a-delegate-webhooks lane): the same shape `WebAgentsServer` gives a static
agent, so a `cron:` webhook is signed exactly as a served agent signs one.
The TypeScript twin is `daemon/agent-identity.ts`; the shared fixture
`tests/fixtures/daemon/signedhooks.json` pins the paths and words.

WHY. The server mints an Ed25519 key per static agent and publishes its key
set at `{agent URL}/.well-known/jwks.json` (`crypto/jwks.py`); an agent loaded
from a file by the daemon (`server/extensions/local_file_source.py`) got no
identity at all, so `cli/daemon/deliver.py` posted every webhook unsigned and
the run's record said so. A receiver that verifies Web Bot Auth
(`crypto/web_bot_auth_verify.py`) could not accept a scheduled report from a
daemon.

WHERE THE KEY LIVES (S-309, 2026-09-27, the agent-secrets lane): in the
store the server keeps every agent's key in, `WEBAGENTS_KEYS_DIR` else
`~/.webagents/keys`, as `<name>.ed25519.jwk.json`, 0600 in a 0700 directory
(`crypto/jwks.py`). For one day the daemon alone kept it in the agent
folder, at `.webagents/keys/` beside the agent file, so that the file, its
schedule state and its key travelled together. They must not: the agent
folder is where git commits go (nothing writes a `.gitignore` there, so
`git add -A` committed the private key), where folders get copied and
shared, and where the file tools look, while the sandbox denies confined
commands `~/.webagents` and only write-denies the folder's own
`.webagents/`. In a container, point `WEBAGENTS_KEYS_DIR` at a mounted
secret, as for the server.

A key found at the old place is moved into the store on load: the same
bytes, the same thumbprint, so the agent's identity does not change, and one
line says so. When the folder is a git work tree and the file was ever
tracked, a second line says to rotate the key, because a key that was
committed is a key someone else may hold. A key in both places holding two
different keys is a conflict (`RuntimeError`, the store's own sentence), and
the agent serves without an identity, as it does for any unusable key file.
A legacy file that is not a usable key is never moved.

ONE NAME, ONE KEY, SAID OUT LOUD. The store names a file by agent name, so
two folders whose agent files share a name would share one key, and
silently. A non-secret sidecar, `<name>.ed25519.origin.json`, records the
folder a daemon key was first made for (or first used from), and a daemon
loading the same name from another folder says so once per process rather
than sharing in silence. The identity still loads: refusing would take an
agent down for a naming clash the operator can fix by renaming one file. The
fixture's `store` section pins the paths and the lines.

THE ISSUER is `{public URL}{url_prefix}/{name}`: the daemon mounts every agent
under `/agents/{name}`, and serves that agent's key set at
`/agents/{name}/.well-known/jwks.json` (`server/core/app.py`), so a verifier
that strips the well-known suffix off `Signature-Agent` fetches the set from
the daemon itself. The public URL is `WEBAGENTS_PUBLIC_URL` when set, the
daemon's own address otherwise. The signer refuses a loopback or plain http
issuer as it always has (`http_signature.py`), so a daemon with no public URL
still posts unsigned and the record says why; set `WEBAGENTS_PUBLIC_URL` to
the https address the daemon is reachable at to sign, as the server asks.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

#: Where the daemon kept an agent's key for one day, under the agent's folder (the fixture's `store.legacy_dir`).
LEGACY_DAEMON_KEYS_DIR = Path(".webagents") / "keys"
#: Where the daemon mounts its agents (the fixture's `issuer` and `jwks_route`).
DAEMON_AGENTS_PREFIX = "/agents"

#: The lines the key move and the shared-name check say (the fixture's `store`).
DAEMON_KEY_LINES = {
    "moved": "[webagents] moved the signing key of agent {agent} from {legacy} to {store}: the same key with the same thumbprint, so its identity is unchanged.",
    "tracked": "[webagents] {legacy} was committed to git in {folder}, so rotate the key: move {store} aside and restart to mint a new one, then remove the old key from the repository's history.",
    "shared": "[webagents] the signing key of agent {agent} in {store} was made for {origin}; {folder} now signs with the same key. Give one of the two agents another name to give it a key of its own.",
}

#: The shared-name line is said once per process per agent and folder, not on every rescan.
_said_shared: Set[str] = set()

__all__ = [
    "DAEMON_AGENTS_PREFIX",
    "DAEMON_KEY_LINES",
    "LEGACY_DAEMON_KEYS_DIR",
    "agent_key_origin_file",
    "daemon_agent_identity",
    "daemon_agent_url",
    "daemon_key_line",
    "daemon_keys_dir",
    "daemon_public_url",
    "legacy_daemon_keys_dir",
]


def daemon_key_line(kind: str, **values: str) -> str:
    """One of `DAEMON_KEY_LINES`, filled in."""
    return DAEMON_KEY_LINES[kind].format(**values)


def daemon_public_url(host: str, port: int, public_url: Optional[str] = None) -> str:
    """The address the daemon publishes for its agents: `public_url`, else
    `WEBAGENTS_PUBLIC_URL`, else its own bind address (the fixture's
    `public_url_cases`)."""
    configured = (public_url if public_url is not None else os.environ.get("WEBAGENTS_PUBLIC_URL", "")).strip()
    if configured:
        return configured.rstrip("/")
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{url_host}:{port}"


def daemon_agent_url(public_url: str, agent_name: str, url_prefix: str = DAEMON_AGENTS_PREFIX) -> str:
    """The agent URL, the principal every signature names: `{public_url}{url_prefix}/{name}`."""
    from webagents.server.core.registration import compose_principal, resolve_public_base_url

    return compose_principal(resolve_public_base_url(public_url, agent_name), agent_name, url_prefix)


def daemon_keys_dir() -> Path:
    """The store every served agent's key lives in: `WEBAGENTS_KEYS_DIR`, else `~/.webagents/keys` (`crypto/jwks.py`)."""
    from webagents.crypto.jwks import JWKSManager

    return JWKSManager({}).keys_dir


def legacy_daemon_keys_dir(agent_dir: Any) -> Path:
    """The old key directory of the agent whose file lives in `agent_dir` (module docstring)."""
    return Path(agent_dir) / LEGACY_DAEMON_KEYS_DIR


def agent_key_origin_file(agent_name: str) -> str:
    """The non-secret sidecar beside an agent's key file, naming the folder the key was made for (the fixture's `store.origin_file`)."""
    from webagents.crypto.jwks import JWKSManager

    return f"{JWKSManager._jwk_stem(agent_name)}.ed25519.origin.json"


def _real_path(target: Path) -> str:
    try:
        return os.path.realpath(target)
    except OSError:
        return os.path.abspath(target)


def _was_tracked_by_git(agent_dir: Path, file: Path) -> bool:
    """Whether `file` (inside `agent_dir`) was ever committed to, or is staged in, the git work tree `agent_dir` is in."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(agent_dir), *args], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()

    try:
        if git("rev-parse", "--is-inside-work-tree") != "true":
            return False
        relative = os.path.relpath(file, agent_dir)
        if git("log", "--all", "--format=%H", "-n", "1", "--", relative):
            return True
        return bool(git("ls-files", "--", relative))
    except (OSError, subprocess.SubprocessError):
        return False


def _adopt_legacy_keys(agent_name: str, agent_dir: Path, manager: Any) -> None:
    """Move the keys an earlier daemon left in `<agent_dir>/.webagents/keys`
    into the store (module docstring): the current key and a previous one
    held for rotation. Raises `RuntimeError` naming the file for a legacy
    file that is not a usable key (it is left where it is) and for a store
    that already holds a different key under the same name."""
    stem = manager._jwk_stem(agent_name)
    store: Path = manager.keys_dir
    # `WEBAGENTS_KEYS_DIR` pointed at the folder itself: nothing to move, and
    # a move onto itself would delete the key.
    if _real_path(store) == _real_path(legacy_daemon_keys_dir(agent_dir)):
        return
    for part in (".ed25519", ".ed25519.previous"):
        file_name = f"{stem}{part}.jwk.json"
        legacy = legacy_daemon_keys_dir(agent_dir) / file_name
        if not os.path.lexists(legacy):
            continue
        # A legacy file that cannot be used is said and never moved: the
        # store's own reader raises naming it.
        key = manager._load_ed25519_jwk(legacy)
        target = store / file_name
        if os.path.lexists(target):
            theirs = manager._load_ed25519_jwk(target)
            if theirs.thumbprint != key.thumbprint:
                raise RuntimeError(
                    f"{legacy} and {target} hold two different agent keys. One agent has one identity, "
                    "so neither is chosen: keep the one the platform knows and move the other aside."
                )
            # The store already holds this very key: the folder copy is a duplicate.
        else:
            store.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                store.chmod(0o700)
            except OSError:
                pass
            data = legacy.read_bytes()
            # Created exclusively, 0600, never replacing: the same rule as a new key.
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        legacy.unlink()
        try:
            legacy.parent.rmdir()
        except OSError:
            pass  # Not empty, or already gone: either is fine.
        print(daemon_key_line("moved", agent=agent_name, legacy=str(legacy), store=str(target)), flush=True)
        if part == ".ed25519" and _was_tracked_by_git(agent_dir, legacy):
            print(daemon_key_line("tracked", legacy=str(legacy), folder=str(agent_dir), store=str(target)), flush=True)


def _record_key_origin(agent_name: str, agent_dir: Path, manager: Any) -> None:
    """Record the folder this agent's key was made for, or say that another folder made it (module docstring)."""
    store: Path = manager.keys_dir
    file = store / agent_key_origin_file(agent_name)
    folder = _real_path(agent_dir)
    existing: Optional[Dict[str, Any]] = None
    try:
        existing = json.loads(file.read_text())
    except FileNotFoundError:
        existing = None
    except (OSError, ValueError):
        return  # Unreadable or not JSON: not ours to rewrite.
    if not isinstance(existing, dict):
        if existing is not None:
            return
        try:
            store.mkdir(parents=True, exist_ok=True, mode=0o700)
            record = {"agent": agent_name, "folder": folder, "recorded_at": datetime.now(timezone.utc).isoformat()}
            fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass  # A store this process cannot write (an ephemeral key was said already): nothing to record.
        return
    origin = existing.get("folder")
    if isinstance(origin, str) and origin != folder:
        once = f"{agent_name}\n{folder}"
        if once in _said_shared:
            return
        _said_shared.add(once)
        key_file = store / f"{manager._jwk_stem(agent_name)}.ed25519.jwk.json"
        print(daemon_key_line("shared", agent=agent_name, store=str(key_file), origin=origin, folder=folder), flush=True)


def daemon_agent_identity(agent_name: str, agent_dir: Any, public_url: Optional[str], url_prefix: str = DAEMON_AGENTS_PREFIX) -> Any:
    """Load, or create and persist, the identity of the agent `agent_name`
    whose file lives in `agent_dir`, in the store the server uses (module
    docstring), after moving any key an earlier daemon left in the agent
    folder and noting the folder the key belongs to. Raises `RuntimeError`
    when a key file exists and cannot be used, as the server does: a key the
    platform may have pinned is never replaced."""
    from webagents.crypto.identity import AgentSigningIdentity
    from webagents.crypto.jwks import JWKSManager

    folder = Path(os.path.abspath(agent_dir))
    manager = JWKSManager({})
    _adopt_legacy_keys(agent_name, folder, manager)
    manager.ensure_ed25519_key(agent_name)
    _record_key_origin(agent_name, folder, manager)
    return AgentSigningIdentity(daemon_agent_url(public_url or "", agent_name, url_prefix), manager)
