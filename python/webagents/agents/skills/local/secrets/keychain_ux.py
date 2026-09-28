"""
Keychain dialogs on macOS: the item names each SDK files its secrets under,
the four lines said before macOS can ask, and the rule that nothing waits on
a question nobody can answer.

WHY (the owner's request, 2026-09-27). macOS asks before a program reads a
keychain item it did not create, in a dialog that names the program ("Python",
"node") and the item. People met that dialog with no idea what it was:

  * Both SDKs filed their items under the SAME names (`webagents:<ns>`), so
    signing in with one and running the other raised a dialog naming the
    other's interpreter.
  * The creator macOS trusts is the INTERPRETER, not webagents. Homebrew's
    Python and node are ad-hoc signed, so the keychain knows each by a hash of
    its binary, and after an upgrade even the creator is asked once more.
  * A process with nobody to answer waits forever. The TypeScript CLI hung on
    exactly that in a headless shell on 2026-09-23.

What this module does about it, identically to the TypeScript twin
(`typescript/src/skills/secrets/keychain-ux.ts`), with every name and sentence
pinned by `tests/fixtures/keychain_ux/keychain_ux.json`:

  1. SERVICE NAMES PER SDK: `webagents (Python) <ns>` here,
     `webagents (TypeScript) <ns>` there, readable in the dialog. Each SDK
     reads only its own. An item under the old shared name is copied the first
     time this SDK finds none of its own, and left in place.
  2. A RECORD beside the items (`keychain.json` in the profile's folder, 0600,
     no secret in it) of which program last used each one, so `doctor` can say
     whether the next use will ask, and the other SDK can say where a sign-in is.
  3. THE EXPLANATION, four lines, printed at most once per run, in a terminal,
     before a read that may raise a dialog.
  4. NEVER BLOCK. The no-dialog read here is PROVED, not assumed:
     `SecKeychainSetUserInteractionAllowed(false)` makes the keychain answer
     `errSecAuthFailed` (-25293) instead of asking. Measured on macOS 26.2 on
     2026-09-27 with probe items that `/usr/bin/security` and node created,
     read by this interpreter: an error in about 20 ms, no dialog, and the
     creator still reads its own item with the switch off. The switch is per
     process (Apple's `KeychainCore::Globals::mUI`, a process singleton), so it
     cannot quiet another app's prompt. Every keychain call tries it first; only
     a terminal gets the call that may ask, after the explanation. `serve`, the
     daemon and a pipe are refused in one sentence naming the command that
     settles it (`webagents whoami`, in a terminal).
  5. NO DEFAULT KEYCHAIN MEANS NO KEYCHAIN. With HOME pointed somewhere else (a
     test, a CI runner) macOS finds no default keychain, and an add with the
     dialog allowed shows its "keychain cannot be found" prompt and waits: a
     test suite hung on exactly that on 2026-09-27. `default_keychain_available`
     is checked before the store picks the keychain.

Names only, never a value, in the record, the sentences and the logs.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("webagents.skills.secrets")

#: This SDK, and the other one, as the fixture names them.
RUNTIME = "python"
OTHER_RUNTIME = "typescript"
RUNTIME_LABELS = {"python": "Python", "typescript": "TypeScript"}
INTERPRETER_NAMES = {"python": "Python", "typescript": "Node"}
CLI_NAMES = {"python": "Python CLI", "typescript": "TypeScript CLI"}

SERVICE_TEMPLATE = "webagents ({runtime}) {namespace}"
LEGACY_SERVICE_TEMPLATE = "webagents:{namespace}"

RECORD_FILE = "keychain.json"
RECORD_FILE_ELSEWHERE = "keychain.record.json"
RECORD_ABOUT = (
    "Which program last used each webagents keychain item on this machine. macOS asks before "
    "a different program reads an item, and webagents reads this file to say so first. It holds no secret."
)

EXPLANATION = (
    'macOS is about to ask whether {program} may use "{item}" in your keychain: {program} is what webagents runs on.',
    "Choose Always Allow so it does not ask again.",
    "If it asks for your Mac password, the password goes to macOS, never to webagents.",
    'webagents only ever asks for items whose name starts with "webagents", so choose Deny for anything else.',
)
BLOCKED = (
    'macOS may ask before {program} can use "{item}" in your keychain, and nothing here can answer it: '
    "run `{command}` once in a terminal, then try again."
)
BLOCKED_COMMAND = "whoami"
LEFT_BEHIND = (
    '"{item}" ({account}) from an earlier version of webagents is still in your keychain, because removing it '
    'would make macOS ask. To remove it, open Keychain Access, search for "{item}" and delete it.'
)
DENIED = 'macOS did not allow {program} to use "{item}", so webagents carries on without it.'
NO_DEFAULT_KEYCHAIN = (
    "macOS has no default keychain for this user here (HOME is {home}), and making one would raise a dialog"
)
OTHER_SIGNED_IN = "The {other} is signed in on this machine, but each CLI keeps its own sign-in in the keychain."
OTHER_KEYS = (
    "The {other} has keys stored on this machine, but each CLI keeps its own: "
    "store them for this one with `{command}`."
)
OTHER_KEYS_COMMAND = "secrets set NAME"

#: The `keychain` line of `doctor` and the `Keychain` row of the chat's /status.
DOCTOR_WORDS = {
    "name": "keychain",
    "file": "not used: the sign-in and keys are in owner-only files in {dir}",
    "keystore": "the system keystore, which asks no per-program questions",
    "keychain": "macOS keychain: {parts}",
    "items": '{count} {noun} named "webagents ({runtime}) ...", last used by {interpreter} {version}',
    "none": "nothing stored by the {cli} yet",
    "quiet": "the next use will not ask",
    "upgraded": "{interpreter} was upgraded since the last use: macOS will ask once",
    "changed": "{interpreter} changed since the last use: macOS will ask once",
    "unknown": "no record says which program made them: macOS may ask once",
    "legacy": "items from an earlier version are here: the next use in a terminal copies them, and macOS may ask once",
    "pending": "{count} {noun} could not be read by a run with nobody to answer macOS",
    "other": "the {other} keeps its own items here",
    "env_token": "WEBAGENTS_TOKEN is set, so the sign-in is not read from here",
    "fix": "`{command}` once in a terminal, then choose Always Allow",
    "noun": {"one": "item", "many": "items"},
}
STATUS_ROW = "Keychain"

#: OSStatus values that, with user interaction off, mean "macOS would have
#: asked". The first two were measured (the module docstring); the others are
#: the documented answers for a locked keychain and a cancelled prompt.
DIALOG_STATUSES = {
    -25293: "errSecAuthFailed",
    -25308: "errSecInteractionNotAllowed",
    -25244: "errSecInvalidOwnerEdit",
    -128: "errSecUserCanceled",
}
ITEM_NOT_FOUND = -25300
SECURITY_TOOL = "/usr/bin/security"


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def service_name(namespace: str, runtime: str = RUNTIME) -> str:
    """The keychain service this SDK files `namespace` under."""
    return SERVICE_TEMPLATE.format(runtime=RUNTIME_LABELS[runtime], namespace=namespace)


def legacy_service_name(namespace: str) -> str:
    """The name both SDKs shared before 2026-09-27. Read once, to copy, and left in place."""
    return LEGACY_SERVICE_TEMPLATE.format(namespace=namespace)


def program_name(path: str) -> str:
    """What macOS calls the running program in its dialog: the bundle's name
    when it runs inside `NAME.app/Contents/MacOS/` (Homebrew's Python does),
    else the file's name (`node`)."""
    parts = Path(path).parts
    for index in range(len(parts) - 2):
        if parts[index].endswith(".app") and parts[index + 1] == "Contents" and parts[index + 2] == "MacOS":
            return parts[index][: -len(".app")]
    return Path(path).name


_program: Optional[Dict[str, str]] = None


def _process_image() -> Optional[str]:
    """The executable macOS runs for this process. For a framework Python that
    is `Python.app/Contents/MacOS/Python`, not the `python3.14` that started it,
    and it is what the keychain knows (measured 2026-09-27)."""
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buffer = ctypes.create_string_buffer(4096)
        if libproc.proc_pidpath(os.getpid(), buffer, ctypes.c_uint32(4096)) > 0:
            return buffer.value.decode("utf-8", "replace")
    except (OSError, AttributeError, ValueError):
        return None
    return None


def current_program() -> Dict[str, str]:
    """This interpreter as the record keeps it: the program name macOS shows,
    the real path, the version, and a stamp (size and modification time) that
    changes when the binary is reinstalled at the same path and version."""
    global _program
    if _program is None:
        path = _process_image() or os.path.realpath(sys.executable or "python")
        try:
            info = os.stat(path)
            stamp = f"{info.st_size}:{int(info.st_mtime)}"
        except OSError:
            stamp = ""
        _program = {
            "runtime": RUNTIME,
            "program": program_name(path),
            "path": path,
            "version": platform.python_version(),
            "stamp": stamp,
        }
    return dict(_program)


def prediction(recorded: Optional[Dict[str, Any]], current: Dict[str, Any]) -> str:
    """What the recorded program and this one mean for the next read of an
    item that exists: `quiet`, `upgraded`, `changed` or `unknown`."""
    if not recorded:
        return "unknown"
    same_stamp = not recorded.get("stamp") or not current.get("stamp") or recorded.get("stamp") == current.get("stamp")
    if recorded.get("path") == current.get("path") and recorded.get("version") == current.get("version") and same_stamp:
        return "quiet"
    if recorded.get("version") != current.get("version"):
        return "upgraded"
    return "changed"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def record_path(secrets_dir: Any) -> Path:
    """Where the record for a store with this secrets folder lives: in its
    parent when the folder is called `secrets` (the profile's folder), else
    inside it, so a test's temporary folder never writes above itself."""
    folder = Path(secrets_dir)
    return folder.parent / RECORD_FILE if folder.name == "secrets" else folder / RECORD_FILE_ELSEWHERE


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repair_mode(path: Path) -> None:
    """0600, whatever an earlier write left."""
    try:
        if path.stat().st_mode & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


_RECORD_LOCK = threading.RLock()


class KeychainRecord:
    """Which program last used each item, what was copied from an old name,
    and what a run with nobody to answer could not read. Names only."""

    def __init__(self, path: Any, runtime: str = RUNTIME) -> None:
        self.path = Path(path)
        self.runtime = runtime

    def read(self) -> Dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return {}
        _repair_mode(self.path)
        try:
            data = json.loads(raw)
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _section(self, data: Dict[str, Any], runtime: Optional[str] = None, create: bool = False) -> Dict[str, Any]:
        name = runtime or self.runtime
        runtimes = data.get("runtimes")
        if not isinstance(runtimes, dict):
            if not create:
                return {}
            runtimes = data["runtimes"] = {}
        section = runtimes.get(name)
        if not isinstance(section, dict):
            if not create:
                return {}
            section = runtimes[name] = {}
        if create:
            for key in ("items", "legacy_done", "pending"):
                if not isinstance(section.get(key), dict):
                    section[key] = {}
        return section

    def _update(self, change: Callable[[Dict[str, Any]], bool]) -> None:
        with _RECORD_LOCK:
            data = self.read()
            if not change(data):
                return
            data["about"] = RECORD_ABOUT
            self._write(data)

    def _write(self, data: Dict[str, Any]) -> None:
        """Atomically, 0600 from the first byte. A record that cannot be
        written costs an explanation later, never the operation."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, temp = tempfile.mkstemp(prefix=".keychain-", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temp)
                raise
            os.chmod(self.path, 0o600)
        except OSError as error:
            logger.debug("could not write the keychain record %s: %s", self.path, error)

    # -- items ---------------------------------------------------------------

    def items(self, runtime: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        found = self._section(self.read(), runtime).get("items")
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for service, accounts in (found or {}).items():
            if isinstance(accounts, dict):
                entries = {account: entry for account, entry in accounts.items() if isinstance(entry, dict)}
                if entries:
                    out[service] = entries
        return out

    def item(self, service: str, account: str, runtime: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.items(runtime).get(service, {}).get(account)

    def note_used(self, service: str, account: str) -> None:
        """This program read or wrote the item: record it (only when it
        changed), and forget any pending entry for it."""
        me = current_program()
        entry = {key: me[key] for key in ("program", "path", "version", "stamp")}

        def change(data: Dict[str, Any]) -> bool:
            section = self._section(data, create=True)
            accounts = section["items"].setdefault(service, {})
            old = accounts.get(account)
            changed = False
            if not isinstance(old, dict) or any(old.get(key) != value for key, value in entry.items()):
                accounts[account] = {**entry, "at": _now()}
                changed = True
            return self._drop_pending(section, service, account) or changed

        self._update(change)

    def forget(self, service: str, account: str) -> None:
        def change(data: Dict[str, Any]) -> bool:
            section = self._section(data)
            if not section:
                return False
            changed = False
            accounts = section.get("items", {}).get(service)
            if isinstance(accounts, dict) and account in accounts:
                del accounts[account]
                if not accounts:
                    del section["items"][service]
                changed = True
            return self._drop_pending(section, service, account) or changed

        self._update(change)

    # -- the old shared names ------------------------------------------------

    def legacy_done(self, legacy_service: str, account: str) -> bool:
        """True once this SDK copied, removed or replaced the old item, so it is
        never read again: a `secrets remove` must not bring it back."""
        done = self._section(self.read()).get("legacy_done", {}).get(legacy_service)
        return isinstance(done, list) and account in done

    def mark_legacy_done(self, legacy_service: str, account: str) -> None:
        def change(data: Dict[str, Any]) -> bool:
            section = self._section(data, create=True)
            done = section["legacy_done"].setdefault(legacy_service, [])
            if not isinstance(done, list):
                done = section["legacy_done"][legacy_service] = []
            if account in done:
                return False
            done.append(account)
            done.sort()
            return True

        self._update(change)

    # -- what a run with nobody to answer could not read -----------------------

    @staticmethod
    def _drop_pending(section: Dict[str, Any], service: str, account: str) -> bool:
        pending = section.get("pending", {}).get(service)
        if isinstance(pending, dict) and account in pending:
            del pending[account]
            if not pending:
                del section["pending"][service]
            return True
        return False

    def note_pending(self, service: str, account: str, namespace: str, secrets_dir: str, legacy: bool) -> None:
        def change(data: Dict[str, Any]) -> bool:
            section = self._section(data, create=True)
            entries = section["pending"].setdefault(service, {})
            if account in entries:
                return False
            entries[account] = {"namespace": namespace, "secrets_dir": secrets_dir, "legacy": bool(legacy), "at": _now()}
            return True

        self._update(change)

    def clear_pending(self, service: str, account: str) -> None:
        self._update(lambda data: self._drop_pending(self._section(data), service, account) if self._section(data) else False)

    def pending(self, runtime: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for service, accounts in (self._section(self.read(), runtime).get("pending") or {}).items():
            if not isinstance(accounts, dict):
                continue
            for account, entry in accounts.items():
                if isinstance(entry, dict):
                    out.append({"service": service, "account": account, **entry})
        return out


# ---------------------------------------------------------------------------
# Who may be asked
# ---------------------------------------------------------------------------

_forbidden: Optional[str] = None
_interactive_override: Optional[bool] = None
_explained = False
_said_blocked = False
_blocked: List[Tuple[str, str]] = []
_writer: Optional[Callable[[str], None]] = None


def forbid_dialogs(reason: str = "serve") -> None:
    """This process never makes a keychain call that may raise a dialog.
    Called by `serve`, the daemon and every server entry point: a dialog there
    would appear on someone's screen at a random later moment, or on none."""
    global _forbidden
    _forbidden = reason or "serve"


def dialogs_allowed() -> bool:
    """A person is at a terminal to answer: stdin and stderr are terminals and
    nothing forbade it."""
    if _forbidden:
        return False
    if _interactive_override is not None:
        return _interactive_override
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stderr and sys.stderr.isatty())
    except (ValueError, OSError, AttributeError):
        return False


def _say(text: str) -> None:
    """Straight to stderr, never through logging: `doctor` and the chat hold
    their log while an agent starts, and this must not be held."""
    if _writer is not None:
        _writer(text)
        return
    try:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
    except (OSError, ValueError, AttributeError):
        pass


def _cli_command(rest: str) -> str:
    try:
        from webagents.cli.config_store import cli_command

        return cli_command(rest)
    except Exception:  # noqa: BLE001 - a library user may not have the CLI's config
        return f"webagents {rest}"


def explanation_lines(item: str, program: Optional[str] = None) -> List[str]:
    name = program or current_program()["program"]
    return [line.format(program=name, item=item) for line in EXPLANATION]


def explain_once(item: str, program: Optional[str] = None) -> bool:
    """The four lines, at most once per run. True when they were printed now."""
    global _explained
    if _explained:
        return False
    _explained = True
    _say("\n".join(explanation_lines(item, program)))
    return True


def blocked_sentence(item: str, program: Optional[str] = None) -> str:
    return BLOCKED.format(program=program or current_program()["program"], item=item, command=_cli_command(BLOCKED_COMMAND))


def note_blocked(item: str, account: str) -> str:
    """A read was refused because nobody can answer: said once per run."""
    global _said_blocked
    _blocked.append((item, account))
    sentence = blocked_sentence(item)
    if not _said_blocked:
        _said_blocked = True
        _say(sentence)
    return sentence


def was_blocked(item: str, account: Optional[str] = None) -> bool:
    return any(seen == item and (account is None or name == account) for seen, name in _blocked)


def left_behind_sentence(item: str, account: str) -> str:
    return LEFT_BEHIND.format(item=item, account=account)


def reset_for_tests(interactive: Optional[bool] = None, writer: Optional[Callable[[str], None]] = None) -> None:
    """Forget what this run said and forbade. Tests only."""
    global _forbidden, _interactive_override, _explained, _said_blocked, _writer, _program
    _forbidden = None
    _interactive_override = interactive
    _explained = False
    _said_blocked = False
    _blocked.clear()
    _writer = writer
    _program = None


# ---------------------------------------------------------------------------
# The macOS calls
# ---------------------------------------------------------------------------

_UI_LOCK = threading.RLock()
_security_lib: Any = None
_default_keychain: Optional[bool] = None


class QuietUnavailable(RuntimeError):
    """The switch that stops a dialog could not be set, so the call may ask."""


def _security() -> Any:
    global _security_lib
    if _security_lib is None:
        lib = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
        lib.SecKeychainSetUserInteractionAllowed.restype = ctypes.c_int32
        lib.SecKeychainSetUserInteractionAllowed.argtypes = [ctypes.c_ubyte]
        lib.SecKeychainGetUserInteractionAllowed.restype = ctypes.c_int32
        lib.SecKeychainGetUserInteractionAllowed.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
        _security_lib = lib
    return _security_lib


@contextlib.contextmanager
def no_interaction() -> Iterator[None]:
    """Keychain calls inside answer an error instead of showing a dialog, and
    the previous setting comes back after. Per process, so no other program's
    prompt is affected; serialised, so two threads cannot interleave it."""
    lib = _security()
    with _UI_LOCK:
        previous = ctypes.c_ubyte(1)
        if lib.SecKeychainGetUserInteractionAllowed(ctypes.byref(previous)) != 0:
            previous = ctypes.c_ubyte(1)
        status = lib.SecKeychainSetUserInteractionAllowed(0)
        if status != 0:
            raise QuietUnavailable(f"SecKeychainSetUserInteractionAllowed answered {status}")
        try:
            yield
        finally:
            lib.SecKeychainSetUserInteractionAllowed(previous.value)


def os_status(error: BaseException) -> Optional[int]:
    """The OSStatus a `keyring` macOS error carries, from anywhere in its chain."""
    seen = 0
    current: Optional[BaseException] = error
    while current is not None and seen < 6:
        args = getattr(current, "args", ())
        if args and isinstance(args[0], int) and not isinstance(args[0], bool):
            return args[0]
        current = current.__cause__ or current.__context__
        seen += 1
    return None


def dialog_needed(error: BaseException) -> bool:
    """With interaction off, this error means macOS would have asked."""
    if isinstance(error, QuietUnavailable):
        return True
    try:
        from keyring.errors import KeyringLocked

        if isinstance(error, KeyringLocked):
            return True
    except ImportError:
        pass
    return os_status(error) in DIALOG_STATUSES


def absent_error(error: BaseException) -> bool:
    return os_status(error) == ITEM_NOT_FOUND


def default_keychain_available() -> Optional[bool]:
    """Whether macOS has a default keychain for this user under this HOME.
    `security default-keychain` only reads the search list, never an item."""
    global _default_keychain
    if sys.platform != "darwin":
        return None
    if _default_keychain is None:
        try:
            done = subprocess.run(
                [SECURITY_TOOL, "default-keychain", "-d", "user"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            _default_keychain = done.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return None
    return _default_keychain


class MacKeychain:
    """The macOS-only calls, behind one object so tests can replace it."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, Optional[str]], Optional[bool]] = {}

    def exists(self, service: str, account: Optional[str] = None, cache: bool = False) -> Optional[bool]:
        """Whether an item exists, from its attributes alone: no secret is
        requested, so no dialog is possible (measured 2026-09-27 on an item
        another program created). None when it cannot be told."""
        key = (service, account)
        if cache and key in self._cache:
            return self._cache[key]
        args = [SECURITY_TOOL, "find-generic-password", "-s", service] + (["-a", account] if account else [])
        try:
            code = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False).returncode
        except (OSError, subprocess.SubprocessError):
            return None
        found = True if code == 0 else False if code == 44 else None
        if cache:
            self._cache[key] = found
        return found

    def quiet(self) -> Any:
        return no_interaction()


def macos_keychain_for(keyring_module: Any) -> Optional[MacKeychain]:
    """The macOS calls, when the store's `keyring` backend is the macOS
    keychain. Any other backend (Secret Service, Credential Manager, a test's
    fake) asks no per-program question, so there is nothing to guard."""
    if sys.platform != "darwin":
        return None
    try:
        backend = keyring_module.get_keyring()
    except Exception:  # noqa: BLE001 - a module without backends is not the macOS keychain
        return None
    if type(backend).__module__ != "keyring.backends.macOS":
        return None
    try:
        _security()
    except (OSError, AttributeError):
        return None
    return MacKeychain()


class KeychainDialogBlocked(RuntimeError):
    """A write or delete that needs macOS to ask, in a run where nobody can
    answer. The message is the one sentence."""

    def __init__(self, sentence: str, item: str, account: str) -> None:
        super().__init__(sentence)
        self.item = item
        self.account = account


# ---------------------------------------------------------------------------
# The store's keychain calls
# ---------------------------------------------------------------------------

_AUTO: Any = object()


class KeychainAccess:
    """Every keychain call a `SecretStore` makes, under this SDK's own names,
    with the old shared name copied once and the dialogs said before and
    never waited on."""

    def __init__(
        self,
        keyring_module: Any,
        namespace: str,
        secrets_dir: Any,
        mac: Any = _AUTO,
        record: Optional[KeychainRecord] = None,
    ) -> None:
        self.keyring = keyring_module
        self.namespace = namespace
        self.service = service_name(namespace)
        self.legacy_service = legacy_service_name(namespace)
        self.secrets_dir = Path(secrets_dir)
        self.mac: Optional[MacKeychain] = macos_keychain_for(keyring_module) if mac is _AUTO else mac
        self.record = record or KeychainRecord(record_path(self.secrets_dir))
        #: Old items `delete` could not remove without a dialog: `{item, account}`.
        self.left_behind: List[Dict[str, str]] = []
        #: The item and whether it was the old one, for the last read that was refused.
        self.last_blocked: Optional[Tuple[str, bool]] = None

    # -- reads ---------------------------------------------------------------

    def get(self, name: str) -> Tuple[Optional[str], str]:
        """`(value, outcome)`: `found`, `adopted` (copied from the old name
        now), `absent`, `denied` (the person said no) or `blocked` (nobody
        can answer here)."""
        self.last_blocked = None
        value, outcome = self._read(self.service, name, legacy=False)
        if outcome != "absent":
            return value, outcome
        if self.record.legacy_done(self.legacy_service, name):
            return None, "absent"
        value, outcome = self._read(self.legacy_service, name, legacy=True)
        if outcome == "found" and value is not None:
            self._adopt(name, value)
            return value, "adopted"
        return value, outcome

    def _read(self, service: str, name: str, legacy: bool) -> Tuple[Optional[str], str]:
        if self.mac is None:
            try:
                value = self.keyring.get_password(service, name)
            except Exception as error:  # noqa: BLE001
                logger.warning("could not read secret %r from the keystore: %s", name, error)
                return None, "absent"
            if value is None:
                return None, "absent"
            if not legacy:
                self.record.note_used(service, name)
            return value, "found"

        if legacy:
            # One lookup for the whole old name first: with no old item at
            # all (a fresh machine, or everything copied) nothing else runs.
            if self.mac.exists(service, None, cache=True) is False:
                return None, "absent"
            if not dialogs_allowed():
                # An old item is never read in a run with nobody to answer,
                # not even quietly: its existence is enough to say what to do.
                if self.mac.exists(service, name, cache=True):
                    return None, self._blocked(service, legacy=True)
                return None, "absent"

        try:
            with self.mac.quiet():
                value = self.keyring.get_password(service, name)
        except Exception as error:  # noqa: BLE001 - classified below
            if not dialog_needed(error):
                if not absent_error(error):
                    logger.warning("could not read secret %r from the keychain: %s", name, error)
                return None, "absent"
        else:
            if value is None:
                return None, "absent"
            if not legacy:
                self.record.note_used(service, name)
            return value, "found"

        # macOS would ask. Only a person at a terminal is asked, and told first.
        if not dialogs_allowed():
            return None, self._blocked(service, legacy)
        explain_once(service)
        try:
            value = self.keyring.get_password(service, name)
        except Exception:  # noqa: BLE001 - Deny, or the keychain stayed locked
            _say(DENIED.format(program=current_program()["program"], item=service))
            return None, "denied"
        if value is None:
            return None, "absent"
        if not legacy:
            self.record.note_used(service, name)
        return value, "found"

    def _blocked(self, service: str, legacy: bool) -> str:
        self.last_blocked = (service, legacy)
        return "blocked"

    def report_blocked(self, name: str) -> str:
        """Say the sentence (once per run) and remember the item for the next
        `webagents whoami` in a terminal. The store calls this when no file
        copy could stand in."""
        item, legacy = self.last_blocked or (self.service, False)
        self.record.note_pending(self.service, name, self.namespace, str(self.secrets_dir), legacy)
        return note_blocked(item, name)

    def _adopt(self, name: str, value: str) -> None:
        """Copy an old item to this SDK's name. The old one stays: another
        tool, or the other SDK's earlier version, may still read it."""
        try:
            if self.mac is None:
                self.keyring.set_password(self.service, name, value)
            else:
                with self.mac.quiet():
                    self.keyring.set_password(self.service, name, value)
        except Exception as error:  # noqa: BLE001 - the value is still returned
            logger.warning("could not copy secret %r to %s: %s", name, self.service, error)
            return
        self.record.note_used(self.service, name)
        self.record.mark_legacy_done(self.legacy_service, name)

    # -- writes --------------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        if self.mac is None:
            self.keyring.set_password(self.service, name, value)
        else:
            try:
                with self.mac.quiet():
                    self.keyring.set_password(self.service, name, value)
            except Exception as error:  # noqa: BLE001 - classified below
                if not dialog_needed(error):
                    raise
                # `keyring` replaces an item by deleting it first, and macOS
                # asks before a program deletes an item another program made
                # (an earlier interpreter, after an upgrade).
                if not dialogs_allowed():
                    raise KeychainDialogBlocked(blocked_sentence(self.service), self.service, name) from error
                explain_once(self.service)
                self.keyring.set_password(self.service, name, value)
        self.record.note_used(self.service, name)
        # A new value replaces whatever the old shared name held.
        self.record.mark_legacy_done(self.legacy_service, name)

    def delete(self, name: str) -> bool:
        """Remove this SDK's item, and retire the old one: removed when that
        needs no dialog, else listed in `left_behind`. Either way the old item
        is never read again, so a removed secret does not come back."""
        self.left_behind = []
        removed = self._delete_own(name)
        # Whether or not it was copied before: removing means the old item
        # goes too when that needs no dialog (`logout` must not leave a
        # sign-in an earlier version can still use), else it is named.
        removed = self._retire_legacy(name) or removed
        self.record.forget(self.service, name)
        return removed

    def _delete_own(self, name: str) -> bool:
        if self.mac is None:
            try:
                self.keyring.delete_password(self.service, name)
                return True
            except Exception:  # noqa: BLE001 - absent
                return False
        try:
            with self.mac.quiet():
                self.keyring.delete_password(self.service, name)
            return True
        except Exception as error:  # noqa: BLE001 - classified below
            if absent_error(error) or not dialog_needed(error):
                return False
            if not dialogs_allowed():
                raise KeychainDialogBlocked(blocked_sentence(self.service), self.service, name) from error
            explain_once(self.service)
            try:
                self.keyring.delete_password(self.service, name)
                return True
            except Exception:  # noqa: BLE001 - Deny
                return False

    def _retire_legacy(self, name: str) -> bool:
        removed = False
        try:
            if self.mac is None:
                try:
                    self.keyring.delete_password(self.legacy_service, name)
                    removed = True
                except Exception:  # noqa: BLE001 - absent
                    pass
            elif self.mac.exists(self.legacy_service, None, cache=True) is not False:
                try:
                    with self.mac.quiet():
                        self.keyring.delete_password(self.legacy_service, name)
                    removed = True
                except Exception as error:  # noqa: BLE001 - classified below
                    if dialog_needed(error) and not absent_error(error):
                        # Removing it would ask: say which item stays, and stop
                        # reading it, which is what removing means to this CLI.
                        self.left_behind.append({"item": self.legacy_service, "account": name})
                        removed = True
        finally:
            self.record.mark_legacy_done(self.legacy_service, name)
        return removed

    # -- names ---------------------------------------------------------------

    def legacy_names(self, legacy_index: List[str]) -> List[str]:
        """The old index's names this SDK has not copied or retired yet: still
        readable through the copy, so `list` shows them."""
        return [name for name in legacy_index if not self.record.legacy_done(self.legacy_service, name)]


# ---------------------------------------------------------------------------
# `webagents whoami`, the other SDK, and `doctor`
# ---------------------------------------------------------------------------


def settle_pending(record_paths: List[Path], open_store: Callable[[str, str], Any]) -> int:
    """`webagents whoami` in a terminal: read every item a run with nobody to
    answer could not read (macOS asks now, after the explanation), so that run
    can read it next time. Values are read and dropped. Returns how many were
    read."""
    if not dialogs_allowed():
        return 0
    settled = 0
    seen = set()
    for path in dict.fromkeys(Path(p) for p in record_paths):
        record = KeychainRecord(path)
        for entry in record.pending():
            key = (entry.get("namespace"), entry.get("secrets_dir"), entry.get("account"))
            if key in seen or not all(key):
                continue
            seen.add(key)
            try:
                store = open_store(str(entry["namespace"]), str(entry["secrets_dir"]))
                value = store.get(str(entry["account"]))
            except Exception:  # noqa: BLE001 - one item must not stop the rest
                continue
            if value is not None:
                settled += 1
            else:
                # Gone, or the person said no: in a terminal nothing is left
                # to settle, so the entry has said its piece.
                record.clear_pending(str(entry["service"]), str(entry["account"]))
    return settled


def other_runtime_items(record: KeychainRecord, namespace: str) -> List[str]:
    """The names the other SDK recorded under `namespace`, never values."""
    return sorted(record.items(OTHER_RUNTIME).get(service_name(namespace, OTHER_RUNTIME), {}))


def other_signed_in_sentence() -> str:
    return OTHER_SIGNED_IN.format(other=CLI_NAMES[OTHER_RUNTIME])


def other_keys_sentence() -> str:
    return OTHER_KEYS.format(other=CLI_NAMES[OTHER_RUNTIME], command=_cli_command(OTHER_KEYS_COMMAND))


def short_path(path: Any) -> str:
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


def doctor_line(facts: Dict[str, Any]) -> Dict[str, Any]:
    """The `keychain` line from what `doctor_facts` found: `{name, status,
    detail, fix?}`. Pure, so both SDKs are held to the fixture's cases."""
    words = DOCTOR_WORDS
    name = words["name"]
    backend = facts.get("backend")
    if backend == "file":
        return {"name": name, "status": "ok", "detail": words["file"].format(dir=facts.get("dir", ""))}
    if backend != "keychain":
        return {"name": name, "status": "ok", "detail": words["keystore"]}
    runtime = facts.get("runtime", RUNTIME)
    other = OTHER_RUNTIME if runtime == RUNTIME else RUNTIME
    interpreter = INTERPRETER_NAMES[runtime]

    def noun(count: int) -> str:
        return words["noun"]["one"] if count == 1 else words["noun"]["many"]

    count = int(facts.get("count") or 0)
    parts: List[str] = []
    warn = False
    if count:
        parts.append(words["items"].format(count=count, noun=noun(count), runtime=RUNTIME_LABELS[runtime], interpreter=interpreter, version=facts.get("interpreter_version", "")))
        kind = facts.get("prediction", "quiet")
        parts.append(words[kind].format(interpreter=interpreter) if kind in ("quiet", "upgraded", "changed", "unknown") else words["quiet"])
        warn = kind in ("upgraded", "changed", "unknown")
    else:
        parts.append(words["none"].format(cli=CLI_NAMES[runtime]))
    if facts.get("legacy"):
        parts.append(words["legacy"])
        warn = True
    pending = int(facts.get("pending") or 0)
    if pending:
        parts.append(words["pending"].format(count=pending, noun=noun(pending)))
        warn = True
    if not count and facts.get("other"):
        parts.append(words["other"].format(other=CLI_NAMES[other]))
    if facts.get("env_token"):
        parts.append(words["env_token"])
    line: Dict[str, Any] = {"name": name, "status": "warn" if warn else "ok", "detail": words["keychain"].format(parts="; ".join(parts))}
    if warn:
        line["fix"] = words["fix"].format(command=facts.get("command") or _cli_command(BLOCKED_COMMAND))
    return line


def doctor_facts(
    stores: List[Any],
    record: KeychainRecord,
    legacy_accounts: Dict[str, List[str]],
    env_token: bool = False,
    mac: Any = _AUTO,
) -> Dict[str, Any]:
    """What the `keychain` line says, gathered WITHOUT reading a value: the
    record, this program, and attribute-only lookups. `stores` are the CLI's
    open stores (the sign-in's and the keys'); `legacy_accounts` maps an old
    service name to the accounts worth looking for under it."""
    first = stores[0] if stores else None
    status = first.status() if first is not None else {"backend": "file", "path": ""}
    if status.get("backend") != "keystore":
        return {"backend": "file", "runtime": RUNTIME, "dir": short_path(Path(str(status.get("path") or "")).parent)}
    keyring_module = getattr(first, "_keyring", None)
    probe = macos_keychain_for(keyring_module) if mac is _AUTO else mac
    if probe is None:
        return {"backend": "keystore", "runtime": RUNTIME}
    me = current_program()
    services = {service_name(store.namespace) for store in stores}
    mine = {service: accounts for service, accounts in record.items().items() if service in services}
    for store in stores:
        for account in _own_index(store):
            mine.setdefault(service_name(store.namespace), {}).setdefault(account, {})
    count = 0
    worst = "quiet"
    order = ["quiet", "unknown", "changed", "upgraded"]
    latest: Optional[Dict[str, Any]] = None
    for service, accounts in mine.items():
        for account, entry in accounts.items():
            if probe.exists(service, account) is not True:
                continue
            count += 1
            kind = prediction(entry or None, me)
            if order.index(kind) > order.index(worst):
                worst = kind
            if entry and (latest is None or str(entry.get("at", "")) > str(latest.get("at", ""))):
                latest = entry
    legacy = False
    for legacy_service, accounts in legacy_accounts.items():
        if probe.exists(legacy_service, None) is not True:
            continue
        if any(not record.legacy_done(legacy_service, account) and probe.exists(legacy_service, account) for account in accounts):
            legacy = True
            break
    other = any(service_name(store.namespace, OTHER_RUNTIME) in record.items(OTHER_RUNTIME) for store in stores)
    return {
        "backend": "keychain",
        "runtime": RUNTIME,
        "count": count,
        "interpreter_version": (latest or {}).get("version") or me["version"],
        "prediction": worst if count else "none",
        "legacy": legacy,
        "pending": len(record.pending()),
        "other": other,
        "env_token": env_token,
    }


def _own_index(store: Any) -> List[str]:
    try:
        return list(store.own_index_names())
    except Exception:  # noqa: BLE001 - a store without an index has no names to add
        return []
