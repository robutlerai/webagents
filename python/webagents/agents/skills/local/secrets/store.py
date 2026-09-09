"""
Secret storage backends, with no agent machinery attached.

WHY THIS IS SEPARATE FROM ``skill.py``: the first consumer is not an LLM, it
is ``register_with_platform``, which runs at boot before any agent exists. A
store reachable only through a ``Skill`` would have forced the server to
construct an agent in order to persist a token. ``skill.py`` is a thin
decorator wrapper over what lives here.

WHY IT EXISTS AT ALL: ``register_with_platform`` now answers with a PLATFORM
BEARER valid for seven days (``EXPIRES_IN_SECONDS = 7 * 24 * 60 * 60`` on the
platform's ``/api/auth/cli/token``), and the examples had nowhere to put it
but a dotfile or ``WEBAGENTS_AGENT_TOKEN``. That token is worse to leak than
it looks, for two reasons recorded in the platform's security log:

  * **S-037**: it carries ``agents:own`` and the signer sets no ``jti``, so
    there is no revocation lever at all. Rotating the key published on the
    agent card does NOT invalidate it, because nothing on its validation path
    reads that key. A leaked one is good for a week.
  * **S-034**: the AOAuth assertion it was exchanged for needs no ``exp`` and
    has no replay defence, so the exchange can be repeated.

A credential that cannot be revoked is one you must not lose in the first
place, which is the whole argument for putting it in the operating system's
keystore rather than in a file next to the code.

BACKENDS, in the order they are tried:

  1. ``keystore`` - the ``keyring`` package, which is the maintained Python
     answer for exactly this: macOS Keychain, Linux Secret Service (libsecret
     over DBus) and Windows Credential Manager behind one API, and it is what
     pip, twine and poetry already use, so a developer's machine is usually
     configured for it already. It is an OPTIONAL extra
     (``pip install webagents[keyring]``) rather than a hard dependency, and
     the import is lazy. Agents are served from containers and headless hosts
     by default, where there IS no keystore behind the library, so making
     every install carry a DBus-adjacent dependency to serve the minority
     that can use it is the wrong default. It is declared in
     ``pyproject.toml`` either way, because a documented import that is not a
     declared dependency is the F-040 bug.
  2. ``file`` - ``~/.webagents/secrets/<namespace>.json``, 0600 in a 0700
     directory. PLAINTEXT. It exists because a container or a headless CI box
     has no keystore and an agent still has to run there.

The fallback is the part that has to be got right, because the failure mode
worth avoiding is not "no keystore", it is "the developer believed there was
one". So it announces itself three times over: once when the store opens
(naming the reason, the fix and the path), again on every write, and a third
time in the ``backend``/``warning`` fields of every tool result, which is the
copy an LLM reading the result actually sees. Set
``WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1`` (or ``require_keystore=True``) and
the fallback becomes an error instead, which is the lever for a deployment
that would rather fail than write a bearer to disk.

NOTHING HERE EVER LOGS A SECRET VALUE. Names only, at every level.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("webagents.skills.secrets")

#: Secret names go into a keychain service key and a JSON object key, so keep
#: them boring. Rejecting rather than sanitising: silently mapping two
#: distinct names onto one storage key is how a ``set`` overwrites a secret
#: its caller never named.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class KeystoreUnavailableError(RuntimeError):
    """Raised when ``require_keystore`` is set and there is no keystore."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"no OS keystore available ({reason}), and "
            "WEBAGENTS_SECRETS_REQUIRE_KEYSTORE refuses the plaintext fallback. "
            "Install the optional keystore support (`pip install webagents[keyring]`) "
            "on a machine with a keystore, or unset the variable to accept a 0600 file."
        )


def _assert_valid_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise ValueError(
            f"invalid secret name {name!r}: use 1-128 characters from A-Z a-z 0-9 . _ -"
        )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def service_key(namespace: str) -> str:
    """The keychain service key. One function, so both backends agree on it."""
    return f"webagents:{namespace}"


def _file_stem(namespace: str) -> str:
    """Filesystem-safe file stem, matching ``crypto/jwks.py``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", namespace)


def resolve_secrets_dir(configured: Optional[str] = None) -> Path:
    """Where the fallback file lives: explicit config, then
    ``WEBAGENTS_SECRETS_DIR``, then ``~/.webagents/secrets`` beside the keys
    directory the identity store already uses."""
    base = (configured or os.getenv("WEBAGENTS_SECRETS_DIR") or "").strip()
    return Path(base) if base else Path.home() / ".webagents" / "secrets"


def _load_keyring() -> Tuple[Any, str]:
    """Import ``keyring`` and confirm it has a backend that actually works.

    Returns ``(module_or_None, reason)``. The reason is the actual failure
    text rather than a generic sentence, because "package not installed" and
    "installed but this container has no DBus session" want different fixes
    and read identically otherwise.

    The backend check is the load-bearing half. ``import keyring`` succeeds on
    a headless Linux box and then hands back ``keyring.backends.fail.Keyring``,
    which raises only when you try to USE it. Treating a successful import as
    proof of a keystore is how a store reports ``keystore`` while storing
    nothing.
    """
    try:
        import keyring  # noqa: PLC0415 - optional extra, imported lazily on purpose
        from keyring.backends import fail as keyring_fail
    except ImportError as e:
        return None, (
            "optional package `keyring` is not installed "
            f"(pip install webagents[keyring]): {e}"
        )

    try:
        backend = keyring.get_keyring()
    except Exception as e:  # noqa: BLE001 - any failure here means no keystore
        return None, f"keyring could not resolve a backend: {e}"

    if isinstance(backend, keyring_fail.Keyring):
        return None, (
            "keyring is installed but found no usable backend on this machine "
            "(a headless container with no DBus session is the usual cause)"
        )
    return keyring, f"{type(backend).__module__}.{type(backend).__name__}"


class SecretStore:
    """Named secrets in the OS keystore, or in a 0600 file that says so.

    Build one with :func:`open_secret_store`, which probes the keystore up
    front so :meth:`status` is answerable before the first read.
    """

    def __init__(
        self,
        namespace: str,
        keyring_module: Any,
        unavailable_reason: str,
        file_path: Path,
        quiet: bool = False,
    ) -> None:
        self.namespace = namespace
        self._keyring = keyring_module
        self._reason = unavailable_reason
        self._file_path = file_path
        self._quiet = quiet
        self._warned_on_open = False

    # -- reporting ---------------------------------------------------------

    @property
    def keystore(self) -> bool:
        """True only when secrets are in the operating system's own keystore."""
        return self._keyring is not None

    def status(self) -> Dict[str, Any]:
        """Where a value would go right now, and why."""
        if self._keyring is not None:
            return {
                "backend": "keystore",
                "keystore": True,
                "namespace": self.namespace,
                "detail": self._reason,
            }
        return {
            "backend": "file",
            "keystore": False,
            "namespace": self.namespace,
            "reason": self._reason,
            "path": str(self._file_path),
            "warning": self._fallback_warning(),
        }

    def _fallback_warning(self) -> str:
        """The sentence a developer has to read. Names the path, because "not
        in a keystore" without "here is where it is instead" is not
        actionable."""
        return (
            f'secrets for "{self.namespace}" are NOT in an OS keystore: {self._reason}. '
            f"They are stored as PLAINTEXT in {self._file_path} (0600). Anything that "
            "can read that file, including a backup and any process running as this "
            "user, can read the secrets. Set WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1 to "
            "refuse this fallback."
        )

    def warn_if_fallback(self) -> None:
        """Print the fallback warning once for this store.

        Called by :func:`open_secret_store` at OPEN rather than at first
        access, deliberately. A warning that waits for the first read arrives
        at a moment the developer has no reason to be watching, and for a
        store only ever written to at shutdown it may not arrive at all.
        """
        if self._keyring is not None or self._quiet or self._warned_on_open:
            return
        self._warned_on_open = True
        logger.warning(self._fallback_warning())

    # -- file backend ------------------------------------------------------

    def _read_file_map(self) -> Dict[str, str]:
        try:
            raw = self._file_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            # Missing, unreadable and corrupt all mean "no secrets here". A
            # corrupt file is deliberately not fatal: refusing to boot because
            # a cache of a re-derivable token failed to parse is worse than
            # re-deriving it.
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {k: v for k, v in parsed.items() if isinstance(v, str)}

    def _write_file_map(self, values: Dict[str, str]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self._file_path.parent.chmod(0o700)
        except OSError:  # e.g. a dir we do not own; the file mode still holds
            pass
        # Create with 0600 rather than write-then-chmod: the latter leaves a
        # window where the secrets exist world-readable. Same pattern as the
        # agent key in `crypto/jwks.py`.
        fd = os.open(self._file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=2, sort_keys=True)
        # `mode=` on open only applies when the file is CREATED, so repair one
        # an earlier version already left at 0644.
        os.chmod(self._file_path, 0o600)

    # -- operations --------------------------------------------------------

    def get(self, name: str) -> Optional[str]:
        """The value, or ``None`` when there is none. Never logged."""
        _assert_valid_name(name)
        if self._keyring is not None:
            try:
                return self._keyring.get_password(service_key(self.namespace), name)
            except Exception as e:  # noqa: BLE001
                logger.warning("could not read secret %r from the keystore: %s", name, e)
                return None
        self.warn_if_fallback()
        return self._read_file_map().get(name)

    def set(self, name: str, value: str) -> str:
        """Store a value. Returns the backend it actually landed in."""
        _assert_valid_name(name)
        if not isinstance(value, str) or value == "":
            raise ValueError(f"refusing to store an empty value for {name}")
        if self._keyring is not None:
            self._keyring.set_password(service_key(self.namespace), name, value)
            return "keystore"

        self.warn_if_fallback()
        if not self._quiet:
            # Every write, not just the first: a developer who scrolled past
            # the open-time warning still sees this one next to the operation
            # that caused it. The NAME is logged, never the value.
            logger.warning(
                'wrote secret "%s" as PLAINTEXT to %s (no OS keystore available)',
                name,
                self._file_path,
            )
        values = self._read_file_map()
        values[name] = value
        self._write_file_map(values)
        return "file"

    def delete(self, name: str) -> bool:
        """Remove a secret. Sweeps BOTH backends, returns whether anything went.

        Both, because the interesting case is a secret written to the file on
        a headless box that later grew a keystore: deleting only the active
        backend leaves the plaintext copy behind, which is exactly the state a
        developer calling ``delete`` believes they have escaped.
        """
        _assert_valid_name(name)
        removed = False
        if self._keyring is not None:
            try:
                self._keyring.delete_password(service_key(self.namespace), name)
                removed = True
            except Exception:  # noqa: BLE001 - PasswordDeleteError means absent
                removed = False
        values = self._read_file_map()
        if name in values:
            del values[name]
            self._write_file_map(values)
            removed = True
        return removed

    def list(self) -> Tuple[List[str], bool]:
        """``(names, complete)``. NEVER values.

        The keystore backend cannot enumerate: ``keyring`` has no portable
        "list everything under this service", and a list that is silently
        short is worse than no list. So the store keeps its own index of names
        it has written, in a file that holds no secret material, and says so
        through ``complete``.
        """
        if self._keyring is None:
            return sorted(self._read_file_map().keys()), True
        return self._read_index(), False

    @property
    def _index_path(self) -> Path:
        return self._file_path.with_suffix(".index.json")

    def _read_index(self) -> List[str]:
        try:
            parsed = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        return sorted(n for n in parsed if isinstance(n, str)) if isinstance(parsed, list) else []

    def note_index(self, name: str, present: bool) -> None:
        """Record or forget a name in the keystore-mode index. Names only."""
        if self._keyring is None:
            return
        current = set(self._read_index())
        if present:
            current.add(name)
        else:
            current.discard(name)
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self._index_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(sorted(current), fh, indent=2)
        except OSError:
            # The index is a convenience. Losing it must never fail a store
            # that already succeeded.
            pass


def open_secret_store(
    namespace: Optional[str] = None,
    require_keystore: Optional[bool] = None,
    secrets_dir: Optional[str] = None,
    quiet: bool = False,
    backend: Optional[str] = None,
) -> SecretStore:
    """Open a secret store, probing the keystore first.

    Args:
        namespace: Collision boundary. Two agents on one machine must not read
            each other's secrets, and the OS keystore is machine-wide, so
            everything is filed under ``webagents:<namespace>``. Defaults to
            ``WEBAGENTS_SECRETS_NAMESPACE``, then ``"webagents"``. Two agents
            that share a name DO share secrets; that is a property of the
            name, not a bug to work around here.
        require_keystore: Refuse the plaintext fallback rather than take it.
            Defaults to ``WEBAGENTS_SECRETS_REQUIRE_KEYSTORE``.
        secrets_dir: Directory for the fallback file. Defaults to
            ``WEBAGENTS_SECRETS_DIR``, then ``~/.webagents/secrets``.
        quiet: Suppress the log warnings. Does NOT suppress the ``warning``
            field on results, which must never be switchable off.
        backend: ``"auto"`` (default) probes for a keystore and falls back.
            ``"file"`` skips the probe and uses the plaintext file
            deliberately. Defaults to ``WEBAGENTS_SECRETS_BACKEND``. It exists
            for two reasons. One: an operator who has decided the file is what
            they want should be able to say so rather than arrange for the
            keystore to fail. Two, and this is the one that made it
            non-negotiable: without it the fallback is UNTESTABLE on any
            machine that has a working keystore, so the path that matters most
            would only ever be exercised where it matters least. Choosing
            ``"file"`` does not quieten a single warning.

    Raises:
        KeystoreUnavailableError: when there is no keystore and the caller (or
            the environment) refused the fallback.
    """
    ns = (namespace or os.getenv("WEBAGENTS_SECRETS_NAMESPACE") or "webagents").strip()
    required = (
        require_keystore
        if require_keystore is not None
        else _env_flag("WEBAGENTS_SECRETS_REQUIRE_KEYSTORE")
    )
    forced = (backend or os.getenv("WEBAGENTS_SECRETS_BACKEND") or "auto").strip().lower()

    if forced == "file":
        module, reason = None, (
            "the file backend was selected explicitly (WEBAGENTS_SECRETS_BACKEND)"
        )
    else:
        module, reason = _load_keyring()

    if module is None and required:
        raise KeystoreUnavailableError(reason)

    # Always resolved, even in keystore mode. `delete()` sweeps the file so a
    # plaintext copy written before the keystore existed does not survive, and
    # the name index lives beside it; both need a real path.
    file_path = resolve_secrets_dir(secrets_dir) / f"{_file_stem(ns)}.json"

    store = SecretStore(
        namespace=ns,
        keyring_module=module,
        unavailable_reason=reason,
        file_path=file_path,
        quiet=quiet,
    )
    # At open, not at first use. See `warn_if_fallback`.
    store.warn_if_fallback()
    return store
