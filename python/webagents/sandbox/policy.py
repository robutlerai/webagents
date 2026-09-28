"""
What a `sandbox:` declaration means, resolved into something a kernel can hold.

The declaration in an agent file is a wish. This module turns it into a
`SandboxPolicy`: absolute, symlink-resolved paths plus a network decision, with
the escalation set subtracted. `srt.py` turns that into an srt settings file
(`@anthropic-ai/sandbox-runtime`, the engine since 2026-09-26; before that
`runner.py` wrote Seatbelt profiles and bubblewrap argument lists itself).

WHAT THE OS CAN AND CANNOT ENFORCE, stated plainly because the schema implies
more than is deliverable:

  * `allowed_folders`  -> ENFORCED. Becomes the write roots.
  * `preset`           -> ENFORCED, as defaults for folders and network.
                          `unrestricted` is the exception and says so: it is
                          NOT confined at all (see `PRESETS`).
  * `network`          -> ENFORCED, per host. srt runs a proxy outside the
                          sandbox and the kernel lets the command reach only
                          that proxy, which admits the listed hosts. Empty
                          means no network. There is no allow-all entry.
  * `allowed_commands` -> NOT ENFORCEMENT. Kept, and relabelled: it decides
                          what runs without prompting. See the package
                          docstring for why a command allow-list cannot be a
                          boundary.
  * `allowed_imports`  -> NOT ENFORCEABLE HERE AT ALL. It is about Python
                          imports inside a process, and an OS sandbox confines
                          file and socket access, not `import` statements.
                          Reported as inert rather than silently ignored.

The words below (presets, the escalation set, the credential folders, the
refusal texts) are pinned by `tests/fixtures/sandbox/srt.json`, which the
TypeScript suite reads too, so the same agent file is confined the same way
under either CLI.

THE SHAPE IS GRANULAR, AND THE SANDBOX IS ON BY DEFAULT (owner decision,
2026-09-27; the sandbox-default lane; TypeScript `sandbox/policy.ts` is the
reference). Every key maps to something srt enforces, and an agent with no
`sandbox:` block gets exactly this:

    sandbox:
      preset: development     # strict | development | off
      files:
        write: [.]            # the agent's folder; the private scratch folder is always added
        read: all             # development: all; strict: the write folders plus system folders
        deny: []              # more paths commands may never read, on top of the built-in list
      network:
        hosts: []             # host names, or the groups npm, pypi, github (`HOST_GROUPS`)
        local: false          # connecting to local servers and listening on a port (srt allowLocalBinding)
        sockets: []           # unix sockets by path (srt allowUnixSockets; macOS only)
      env: []                 # variables a command may see; none that look secret by default

The flat keys keep working as aliases: `allowed_folders` is `files.write`, a
bare `network:` list is `network.hosts`, `env_passthrough` is `env`;
`allowed_commands` and `allowed_imports` stay as they are. `sandbox: off`
(PyYAML reads it as the boolean False, which is taken too) is the explicit
opt-out and means what `preset: unrestricted` means, which stays accepted.
`normalize_declaration` folds the aliases in; the fixture pins the mapping.

THE BUILT-IN DENIES apply under every confined preset and no key removes them
(S-311, S-315, S-309): reads of `CREDENTIAL_DIRS` under $HOME with
`Library/Keychains` (the keychain trusts the program that created an item,
and every item this SDK stores was created by the interpreter a confined
command can start, so with the file readable it read the CLI's platform
token and the secrets store with no dialog; exercised); reads of every
per-profile folder `~/.webagents-<profile>` (srt's deny is a path prefix, so
`.webagents` never covered its siblings, which hold history, sessions,
checkpoints and, with the file backend, the token and secrets; exercised);
reads of `.env`, `.env.*` and `.webagents/` in the working folder and every
write root (provider keys live in `.env`, and the daemon's signing key lived
under `.webagents/`); writes to `ESCALATION_DENY` and the agent-file
patterns (S-283); and writes to the SDK's own install whenever it lies
inside a write root (S-316, `install_write_denies`: a project-local venv or
`node_modules` holding webagents, srt or the node srt runs on). macOS takes
globs; Linux enumerates what exists when the command starts, as
`AGENT_FILE_PATTERNS` already does.
"""

from __future__ import annotations

import fnmatch
import os
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class SandboxUnavailable(RuntimeError):
    """No enforcement backend, and the policy asked for one.

    Raised rather than degraded. See rule 2 in the package docstring.
    """


class SandboxDeclarationError(ValueError):
    """A declaration that cannot be used as written, with a sentence for its author."""


#: The keys of the normalised `sandbox:` block (the loader's `SandboxConfig` fields), sorted.
SANDBOX_KEYS = ("allowed_commands", "allowed_imports", "env", "files", "network", "preset")

#: The flat spellings still accepted, and the key each folds into (fixture `schema.aliases`).
SANDBOX_ALIASES = {"allowed_folders": "files.write", "env_passthrough": "env"}

#: Every key a `sandbox:` block may carry, as the refusal lists them: the keys and the aliases, sorted.
SANDBOX_ACCEPTED_KEYS = tuple(sorted((*SANDBOX_KEYS, *SANDBOX_ALIASES)))

SANDBOX_FILES_KEYS = ("deny", "read", "write")
SANDBOX_NETWORK_KEYS = ("hosts", "local", "sockets")

DEFAULT_PRESET = "development"

#: The spelling of the opt-out in an agent file: `sandbox: off` (False once PyYAML has read it).
OFF_PRESET = "off"

#: `network.hosts` entries that name a group rather than a host, expanded to
#: the hosts each truly needs and nothing more (fixture `host_groups`).
HOST_GROUPS: Dict[str, Sequence[str]] = {
    "npm": ("registry.npmjs.org",),
    "pypi": ("pypi.org", "files.pythonhosted.org"),
    "github": ("github.com", "api.github.com", "codeload.github.com", "objects.githubusercontent.com", "raw.githubusercontent.com"),
}

#: The command-line opt-out: `webagents --no-sandbox` puts this in the
#: environment (as `--profile` puts `WEBAGENTS_PROFILE`), and the shell reads
#: it when it resolves its policy. One run, the owner's commands only: callers
#: other than the owner are refused under it (S-248), and SKILL.md scripts
#: keep running confined.
ENV_NO_SANDBOX = "WEBAGENTS_NO_SANDBOX"


def no_sandbox_requested(environ: Optional[Dict[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return (env.get(ENV_NO_SANDBOX) or "").strip().lower() in {"1", "true", "yes"}


def sandbox_state(policy: Optional["SandboxPolicy"], origin: str) -> str:
    """The state as the status row and `/sandbox` print it: `development
    (default)`, `off (agent file)`, `off (--no-sandbox)` (fixture `status`).
    `origin` is one of `default`, `agent file`, `--no-sandbox`."""
    if policy is None or not policy.confined:
        return f"{OFF_PRESET} ({origin})"
    return f"{policy.preset} ({origin})"


#: Paths that stay WRITE-DENIED even inside an allowed root.
#:
#: Every serious implementation converged on a list like this independently
#: (Claude Code's is mandatory and cannot be exempted; Codex calls its version
#: `read_only_subpaths`). The reasoning is the same in each: a command that can
#: write here can grant itself permissions for the NEXT command, so allowing it
#: makes the whole policy advisory.
#:
#: Relative to each write root, matched on any path component.
#:
#: THE AGENT'S OWN FILES ARE IN THIS SET (S-283, 2026-09-26). Under
#: `development`, and under `strict` with the default `allowed_folders:
#: ["."]`, the agent's folder is a write root, and the review's probe wrote
#: `AGENT.md`, `WEBAGENTS.md`, `mcp.json` and another skill's `SKILL.md` from
#: a confined command: a persistent injection into every later turn, a wider
#: `sandbox: network:`, an open `access:` block or a new `cron:` schedule,
#: all reloaded by the daemon on the write itself. So the agent definition
#: (`AGENT.md`, and `AGENT-<name>.md` through `AGENT_FILE_PATTERNS`), the
#: inherited context (`WEBAGENTS.md`), the file the MCP skill reads
#: (`mcp.json`, undotted, beside the `.mcp.json` already here) and the whole
#: `.agents/skills` folder (every SKILL.md skill, the ones not installed yet
#: included, since the folder itself is denied) are write-denied in every
#: write root, on both engines. Literal entries hold whether or not the file
#: exists yet on macOS; on Linux srt takes only paths that exist (`srt.py`,
#: `build_settings`).
ESCALATION_DENY = (
    ".git/hooks",
    ".git/config",
    ".claude",
    ".webagents",
    ".vscode",
    ".idea",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".gitconfig",
    ".mcp.json",
    ".env",
    "AGENT.md",
    "WEBAGENTS.md",
    "mcp.json",
    ".agents/skills",
)

#: Agent-file NAME PATTERNS write-denied in every write root (S-283): a
#: planted `AGENT-<name>.md` is a new agent the daemon serves on its next
#: scan, with whatever `sandbox:`, `access:` and `cron:` the planter wrote.
#: The daemon's own rule for an agent file name (`cli/daemon/registry.py`,
#: `AGENT.md` or `AGENT-<name>.md`) is what the pattern spells.
#:
#: HOW A PATTERN REACHES THE KERNEL, since the two engines differ:
#:
#:   * macOS: srt compiles a deny glob into a Seatbelt regex, so the pattern
#:     itself is passed and covers a file CREATED during the command too
#:     (probed 2026-09-26: `AGENT-evil.md` refused, `AGENT-helper.md` refused
#:     and not renamable).
#:   * Linux: bubblewrap binds concrete paths only, and srt strips a write
#:     glob silently (`stripWriteGlobs` in its `sandbox-manager`). So the
#:     files that exist when the settings are built are enumerated and denied
#:     by name, and a match that does not exist yet is NOT covered there.
#:     Stated in the fixture (`agent_file_deny.linux`) rather than papered
#:     over with a placeholder, which bubblewrap would create on the host.
#:
#: Both engines get the enumerated names, so the settings differ by exactly
#: the glob entry. Pinned by `tests/fixtures/sandbox/srt.json`.
AGENT_FILE_PATTERNS = ("AGENT-*.md",)


def _within(path: str, root: str) -> bool:
    """Whether `path` is `root` or lies beneath it (both absolute, realpath'd)."""
    root = root.rstrip("/") or "/"
    return root == "/" or path == root or path.startswith(root + "/")


def install_write_denies(write_roots: Sequence[str], installs: Sequence[str]) -> List[str]:
    """The parts of the SDK's own install that a command could otherwise
    write, to be write-denied (S-316, the ptypass-fixes lane, 2026-09-27).

    `installs` are where the running CLI and its engine live (`srt.
    sdk_install_paths`): the interpreter's environment, the webagents
    package, srt and the node that runs it. Under `development` the agent's
    folder is a write root, so a project-local `.venv` or `node_modules`
    holding webagents was writable from a confined command, and srt, which
    runs OUTSIDE the sandbox for every command, and the SDK code the next
    `webagents` start imports, were both rewritable: the sandbox's guarantee
    ended at the next command. An install inside a write root is denied
    whole; a write root inside an install is denied whole too (`files.write`
    naming a folder of the install). An install outside every write root
    adds nothing, which is the usual case (pipx, a global install, a venv
    elsewhere). Nothing else is denied: a command may still write a
    DIFFERENT venv or `node_modules` the project keeps. Pinned by the
    fixture's `sdk_install_deny`."""
    denied: List[str] = []
    for install in installs:
        for root in write_roots:
            if _within(install, root):
                hit = install
            elif _within(root, install):
                hit = root
            else:
                continue
            if hit not in denied:
                denied.append(hit)
    return denied


def matches_agent_file_pattern(name: str) -> bool:
    """Whether `name` (one path segment) matches one of `AGENT_FILE_PATTERNS`;
    `*` matches within the segment, case-sensitively, as srt's glob does."""
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in AGENT_FILE_PATTERNS)


def _matching_entries(folder: str, pattern: str) -> List[str]:
    """The entries of `folder` whose names match `pattern`, sorted; none when it cannot be listed."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return [os.path.join(folder, name) for name in sorted(names) if fnmatch.fnmatchcase(name, pattern)]


def agent_file_denies(root: str, system: Optional[str] = None) -> List[str]:
    """The agent-file denies for one write root: every existing file whose
    name matches a pattern, by name, and on macOS the pattern itself. A root
    that cannot be listed contributes only the pattern (macOS) or nothing."""
    system = system or platform.system()
    try:
        names = os.listdir(root)
    except OSError:
        names = []
    denied = [os.path.join(root, name) for name in sorted(names) if matches_agent_file_pattern(name)]
    if system == "Darwin":
        denied.extend(os.path.join(root, pattern) for pattern in AGENT_FILE_PATTERNS)
    return denied

#: `preset` -> (write the working directory?, scope reads?, confined at all?).
#:
#: Reads and writes are deliberately asymmetric, the way Claude Code's are:
#: writes are enumerated everywhere, reads are broad by default and scoped only
#: under `strict`. Scoping reads is the expensive option, because module
#: resolution and toolchains traverse far more of the filesystem than anyone
#: expects, so it is opt-in rather than the default.
#:
#: `unrestricted` IS NOT CONFINED (2026-09-26). It used to mean "writes as
#: `development`, network open". srt has no allow-all network entry (`*` is
#: rejected), so that policy cannot be expressed any more, and the two ways
#: of keeping the name were both wrong: confining files under a name that
#: says otherwise is the S-217 shape (a declaration that does something else
#: than it says), and demanding a domain list would make it a second spelling
#: of `development` plus `network:`. So it is an explicit opt-out, reported
#: loudly wherever the sandbox is reported, and a caller other than the owner
#: is refused under it exactly as under no declaration.
PRESETS = {
    "strict": (False, True, True),
    "development": (True, False, True),
    "unrestricted": (True, False, False),
}

#: Folders under the home directory a `development` command cannot read: the
#: credentials of the person running the agent. `strict` denies every read
#: outside the declared folders, so this list only matters there. Literal
#: paths (srt's Linux rules take no globs), realpath'd against `$HOME`.
#: `Library/Keychains` is the login keychain file (S-311, module docstring).
CREDENTIAL_DIRS = (
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".gnupg",
    ".kube",
    ".docker",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".webagents",
    "Library/Keychains",
)

#: The per-profile folders beside `~/.webagents` (S-315): `~/.webagents-local` and its siblings.
PROFILE_DIR_PATTERN = ".webagents-*"

#: Read-denied in the working folder and every write root, under every confined preset (S-309, S-312).
ROOT_READ_DENY = (".env", ".webagents")
ROOT_READ_DENY_PATTERNS = (".env.*",)


def profile_dir_denies(home: str, system: Optional[str] = None) -> List[str]:
    """The profile folders to deny under `home`: the glob itself on macOS
    (Seatbelt takes it as a regex, so a profile made after the settings were
    built is covered too), the folders that exist on Linux (bubblewrap binds
    concrete paths only; a placeholder would be created on the host)."""
    system = system or platform.system()
    if system == "Darwin":
        return [os.path.join(home, PROFILE_DIR_PATTERN)]
    return [entry for entry in _matching_entries(home, PROFILE_DIR_PATTERN) if os.path.isdir(entry)]


def root_read_denies(root: str, system: Optional[str] = None) -> List[str]:
    """The read denies for one folder a command may write (the working folder,
    every write root): `.env`, `.webagents` and every `.env.*`. Literal entries
    hold whether or not the file exists yet on macOS, and the pattern is passed
    as a glob there; Linux gets only what exists when the command starts."""
    system = system or platform.system()
    denied: List[str] = []
    if system == "Darwin":
        denied.extend(os.path.join(root, relative) for relative in ROOT_READ_DENY)
        denied.extend(os.path.join(root, pattern) for pattern in ROOT_READ_DENY_PATTERNS)
        return denied
    for relative in ROOT_READ_DENY:
        if os.path.exists(os.path.join(root, relative)):
            denied.append(os.path.join(root, relative))
    for pattern in ROOT_READ_DENY_PATTERNS:
        for entry in _matching_entries(root, pattern):
            if entry not in denied:
                denied.append(entry)
    return denied

#: Read access every command needs before it can do anything at all, under
#: `strict`, where reads are denied everywhere else. Without these the dynamic
#: loader fails and the process dies before `main`. `/private/var/select` is
#: what Apple's `python3` and `git` shims read to find the developer tools.
SYSTEM_READ = {
    "Darwin": (
        "/usr", "/bin", "/sbin", "/System", "/Library", "/private/etc",
        "/private/var/db", "/private/var/select", "/dev", "/opt/homebrew", "/opt/local",
    ),
    "Linux": (
        "/usr", "/etc", "/opt", "/bin", "/sbin", "/lib", "/lib64", "/lib32",
        "/dev", "/proc", "/sys", "/run",
    ),
}

#: A `network:` entry is a host (`github.com`), a wildcard under a domain
#: (`*.example.com`) or a host with a port (`127.0.0.1:8080`, `[::1]:80`).
#: srt rejects anything else at run time; checking here refuses it at load,
#: where the author of the file can see it.
_NETWORK_ENTRY = re.compile(
    r"^(\*\.)?(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*(?::\d{1,5})?$"
)
_NETWORK_ENTRY_V6 = re.compile(r"^\[[0-9A-Fa-f:.]+\](?::\d{1,5})?$")


def check_network_entry(entry: Any) -> str:
    """The entry as srt will see it, or a `ValueError` saying what is wrong."""
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("sandbox: a network entry must be a host name")
    value = entry.strip()
    if value == "*" or value.startswith("*.") and "." not in value[2:]:
        raise ValueError(
            f"sandbox: network entry {value!r} is not allowed: srt has no allow-all; "
            "list the hosts (`*.example.com` needs a domain under the wildcard)"
        )
    if "://" in value or "/" in value:
        raise ValueError(f"sandbox: network entry {value!r} must be a host name, not a URL or an address range")
    if not (_NETWORK_ENTRY.match(value) or _NETWORK_ENTRY_V6.match(value)):
        raise ValueError(f"sandbox: network entry {value!r} is not a host name (`github.com`, `*.example.com`, `127.0.0.1:8080`)")
    return value


def expand_hosts(entries: Iterable[Any]) -> List[str]:
    """The hosts a `network.hosts` list names: groups expanded in place, each
    entry checked, duplicates dropped."""
    hosts: List[str] = []
    for entry in entries:
        group = HOST_GROUPS.get(entry.strip().lower()) if isinstance(entry, str) else None
        for host in group if group is not None else [check_network_entry(entry)]:
            if host not in hosts:
                hosts.append(host)
    return hosts


# ---------------------------------------------------------------------------
# The declaration, normalised
# ---------------------------------------------------------------------------


def _unknown_keys_message(unknown: Sequence[str], known: Sequence[str], what: str) -> str:
    """The refusal for keys the schema does not define, naming the nearest
    match: the loader's sentence (`cli/loader/schema.py`, `_reject_unknown_keys`)."""
    import difflib

    problems = []
    for key in unknown:
        close = difflib.get_close_matches(str(key), sorted(known), n=1, cutoff=0.6)
        problems.append(f"unknown key '{key}' (did you mean '{close[0]}'?)" if close else f"unknown key '{key}'")
    return f"{what}: " + "; ".join(problems) + f". Known keys: {', '.join(sorted(known))}"


def _string_list(value: Any, key: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise SandboxDeclarationError(f"sandbox: {key} must be a list of strings")
    return list(value)


def is_sandbox_off(raw: Any) -> bool:
    """Whether a `sandbox:` value spells the opt-out: `off`, or the boolean PyYAML reads `off` as."""
    return raw is False or (isinstance(raw, str) and raw.strip().lower() == OFF_PRESET)


_SHAPE_MESSAGE = "sandbox: must be a mapping of settings (preset, files, network, env, ...) or `off`"


def normalize_declaration(raw: Any) -> Dict[str, Any]:
    """The declaration as written, checked and normalised (the TypeScript
    `parseSandboxDeclaration`): unknown keys refused with the did-you-mean at
    every level, aliases folded in, lists typed, defaults filled. `off` (or
    False) is the opt-out and becomes `preset: unrestricted`; `on` (or True,
    or an empty mapping) is the defaults. `files.read` defaults by preset:
    `all` unless `strict`, which scopes reads to the write folders. A
    `SandboxConfig` model (anything with `model_dump`) is taken as its
    mapping. The preset's NAME is checked when the policy is built."""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if is_sandbox_off(raw):
        raw = {"preset": "unrestricted"}
    if raw is True or (isinstance(raw, str) and raw.strip().lower() == "on"):
        raw = {}
    if not isinstance(raw, dict):
        raise SandboxDeclarationError(_SHAPE_MESSAGE)
    unknown = [key for key in raw if key not in SANDBOX_ACCEPTED_KEYS]
    if unknown:
        raise SandboxDeclarationError(_unknown_keys_message(unknown, SANDBOX_ACCEPTED_KEYS, "sandbox"))
    if "allowed_folders" in raw and raw.get("files") is not None:
        raise SandboxDeclarationError("sandbox: allowed_folders is the old spelling of files.write; use one of them")
    if "env_passthrough" in raw and raw.get("env") is not None:
        raise SandboxDeclarationError("sandbox: env_passthrough is the old spelling of env; use one of them")
    preset = raw.get("preset")
    preset = DEFAULT_PRESET if preset is None else preset
    if not isinstance(preset, str):
        raise SandboxDeclarationError("sandbox: preset must be a string")
    if preset.strip().lower() == OFF_PRESET:
        preset = "unrestricted"

    write: List[str] = ["."]
    read: Any = None
    deny: List[str] = []
    files = raw.get("files")
    if files is not None:
        if not isinstance(files, dict):
            raise SandboxDeclarationError("sandbox: files must be a mapping (write, read, deny)")
        bad = [key for key in files if key not in SANDBOX_FILES_KEYS]
        if bad:
            raise SandboxDeclarationError(_unknown_keys_message(bad, SANDBOX_FILES_KEYS, "sandbox.files"))
        if "write" in files and files["write"] is not None:
            write = _string_list(files["write"], "files.write")
        if files.get("read") is not None:
            if isinstance(files["read"], str) and files["read"].strip().lower() == "all":
                read = "all"
            else:
                read = _string_list(files["read"], "files.read (`all`, or a list of folders)")
        deny = _string_list(files.get("deny"), "files.deny")
    elif "allowed_folders" in raw and raw["allowed_folders"] is not None:
        write = _string_list(raw["allowed_folders"], "allowed_folders")
    if read is None:
        read = [] if preset.lower() == "strict" else "all"

    hosts: List[str] = []
    local = False
    sockets: List[str] = []
    network = raw.get("network")
    if isinstance(network, (list, tuple)):
        hosts = _string_list(network, "network")
    elif network is not None:
        if not isinstance(network, dict):
            raise SandboxDeclarationError("sandbox: network must be a list of hosts or a mapping (hosts, local, sockets)")
        bad = [key for key in network if key not in SANDBOX_NETWORK_KEYS]
        if bad:
            raise SandboxDeclarationError(_unknown_keys_message(bad, SANDBOX_NETWORK_KEYS, "sandbox.network"))
        hosts = _string_list(network.get("hosts"), "network.hosts")
        if network.get("local") is not None:
            if not isinstance(network["local"], bool):
                raise SandboxDeclarationError("sandbox: network.local must be true or false")
            local = network["local"]
        sockets = _string_list(network.get("sockets"), "network.sockets")

    env = _string_list(raw["env"], "env") if raw.get("env") is not None else _string_list(raw.get("env_passthrough"), "env_passthrough")
    return {
        "preset": preset,
        "files": {"write": write, "read": read, "deny": deny},
        "network": {"hosts": hosts, "local": local, "sockets": sockets},
        "env": env,
        "allowed_commands": _string_list(raw.get("allowed_commands"), "allowed_commands"),
        "allowed_imports": _string_list(raw.get("allowed_imports"), "allowed_imports"),
    }

#: A PRIVATE scratch directory, not the whole of `$TMPDIR`.
#:
#: An earlier version added `$TMPDIR` itself as a write root, which was wrong
#: in a way the first test caught: an agent working inside a temp directory,
#: which is extremely normal, had its whole working tree made writable AND
#: readable, so a policy naming one subdirectory silently granted its siblings.
#: Claude Code does the same thing this does and sets `TMPDIR=/tmp/claude`.
SCRATCH_DIR_NAME = "webagents-sandbox"


def _real(path: os.PathLike | str) -> str:
    """Absolute and symlink-resolved. See rule 3 in the package docstring."""
    return os.path.realpath(os.path.expanduser(str(path)))


@dataclass
class SandboxPolicy:
    """A resolved, enforceable policy."""

    #: Directories the command may write to. Already realpath'd.
    write_roots: List[str] = field(default_factory=list)
    #: Directories the command may read when `scoped_reads` is set.
    read_roots: List[str] = field(default_factory=list)
    #: Whether reads are confined to `read_roots`: `strict`, or a `files.read` list.
    scoped_reads: bool = False
    #: More paths the command may never read (`files.deny`), realpath'd.
    read_deny: List[str] = field(default_factory=list)
    #: The private scratch directory, exported to the child as `TMPDIR`.
    scratch: Optional[str] = None
    #: Outbound network: True when any host is reachable (a `network:` list,
    #: or the unconfined preset). `network_domains` says which.
    network: bool = False
    #: The hosts the command may reach, as srt's `allowedDomains`, groups expanded.
    network_domains: List[str] = field(default_factory=list)
    #: Connecting to local servers and listening on a port (`network.local`, srt `allowLocalBinding`).
    local_network: bool = False
    #: Unix sockets the command may reach by path (`network.sockets`, srt `allowUnixSockets` on macOS).
    unix_sockets: List[str] = field(default_factory=list)
    #: False for `unrestricted`: nothing below is applied and the command runs
    #: with the agent's permissions. Reported everywhere the sandbox is.
    confined: bool = True
    #: The preset the policy came from, for reports.
    preset: str = "development"
    #: Where the command runs.
    cwd: Optional[str] = None
    #: Carried through for reporting; NOT enforced here. See the module docstring.
    advisory_commands: List[str] = field(default_factory=list)
    #: Declared but unenforceable. Reported so it cannot be believed.
    unenforceable: List[str] = field(default_factory=list)
    #: Secret-looking variables the command may still see (S-220).
    env_passthrough: List[str] = field(default_factory=list)
    #: Folders that stay write-denied even inside a write root, on top of the
    #: escalation set: EVERY SKILL.md skill's folder while one skill's script
    #: runs (plan item 1.4, 2026-09-26; widened from the running skill's own
    #: folder for S-283, since a `pdf` script wrote `.agents/skills/other/
    #: SKILL.md`), so third-party code cannot rewrite the instructions and
    #: scripts the next activation of any skill will trust. Absolute.
    read_only: List[str] = field(default_factory=list)

    @property
    def deny_writes(self) -> List[str]:
        """Escalation paths to deny inside each write root (the literal set,
        then the agent-file patterns as `agent_file_denies` resolves them for
        this platform), then `read_only`. Built per command, so the
        enumeration sees the files as they are when the command starts."""
        denied = []
        for root in self.write_roots:
            for relative in ESCALATION_DENY:
                denied.append(os.path.join(root, relative))
            for entry in agent_file_denies(root):
                if entry not in denied:
                    denied.append(entry)
        for folder in self.read_only:
            if folder not in denied:
                denied.append(folder)
        return denied

    @property
    def deny_reads(self) -> List[str]:
        """What the command cannot read. Scoped reads (`strict`, or a
        `files.read` list): everything, with `allow_reads` re-allowed beneath.
        Otherwise the credential folders and the profile folders under $HOME.
        In both cases the built-in root denies (`.env`, `.env.*`, `.webagents`
        in the working folder and every write root but the scratch) and
        `read_deny`, which srt re-emits after its allows so a deny nested
        inside an allowed folder still holds."""
        system = platform.system()
        denied: List[str] = []

        def add(entry: str) -> None:
            if entry not in denied:
                denied.append(entry)

        if self.scoped_reads:
            add("/")
        else:
            home = _real(Path.home())
            for relative in CREDENTIAL_DIRS:
                add(os.path.join(home, relative))
            for entry in profile_dir_denies(home, system):
                add(entry)
        roots = [root for root in [self.cwd, *self.write_roots] if root and root != self.scratch]
        for root in roots:
            for entry in root_read_denies(root, system):
                add(entry)
        for entry in self.read_deny:
            if system == "Darwin" or os.path.exists(entry):
                add(entry)
        return denied

    @property
    def allow_reads(self) -> List[str]:
        """Under `strict`, the reads re-allowed beneath the denied root."""
        if not self.scoped_reads:
            return []
        system = SYSTEM_READ.get(platform.system(), ())
        readable = list(system) + list(self.read_roots) + list(self.write_roots)
        if self.cwd:
            readable.append(self.cwd)
        return _dedupe_paths(_existing(readable))

    def describe(self) -> str:
        """One line, for a refusal message or a log."""
        if not self.confined:
            return f"not confined (preset {self.preset})"
        roots = ", ".join(self.write_roots) or "(nothing)"
        network = ", ".join(self.network_domains) if self.network_domains else "off"
        switches = []
        if self.local_network:
            switches.append("local network on")
        if self.unix_sockets:
            switches.append(f"sockets: {', '.join(self.unix_sockets)}")
        return f"writes: {roots}; network: {network}" + (f"; {'; '.join(switches)}" if switches else "")


def policy_from_metadata(
    sandbox: Any,
    *,
    cwd: Optional[str] = None,
    tmpdir: Optional[str] = None,
) -> Optional[SandboxPolicy]:
    """Resolve an agent file's `sandbox:` into a policy, or None if absent.

    `sandbox` is the `SandboxConfig` from `cli/loader/schema.py`, a mapping in
    the nested or the flat spelling, or `off` (`normalize_declaration`). None
    means the file declared nothing and the CALLER decides what that means:
    the shell takes the defaults (`default_policy`), the SKILL.md runner a
    synthesised `strict`.

    Args:
        sandbox: the declaration, or None
        cwd: where the command runs; defaults to the process's cwd
        tmpdir: a writable scratch directory; defaults to `$TMPDIR`
    """
    if sandbox is None:
        return None

    declaration = normalize_declaration(sandbox)
    preset = str(declaration["preset"] or DEFAULT_PRESET).lower()
    if preset == OFF_PRESET:
        preset = "unrestricted"
    if preset not in PRESETS:
        raise ValueError(
            f"unknown sandbox preset {preset!r}; expected one of "
            f"{', '.join(sorted([*PRESETS, OFF_PRESET]))}"
        )
    # The preset's read scoping is already in the declaration's `files.read`.
    writes_cwd, _preset_scoped_reads, confined = PRESETS[preset]

    # Checked here, at load, rather than by srt at run time: the author of the
    # file is the one who can fix it, and the run-time refusal would name a
    # settings file they have never seen.
    network_domains = expand_hosts(declaration["network"]["hosts"])

    working = _real(cwd or os.getcwd())

    # A directory of ours inside the temp area, never the temp area itself.
    scratch_base = _real(tmpdir or os.environ.get("TMPDIR") or "/tmp")
    scratch = os.path.join(scratch_base, SCRATCH_DIR_NAME)
    try:
        os.makedirs(scratch, mode=0o700, exist_ok=True)
    except OSError:
        # A scratch directory we cannot create is one we must not promise.
        scratch = ""

    # `files.write` is relative to the working directory when relative, which
    # is how `["."]`, the schema default, is meant to read.
    def relative_to(entry: str) -> str:
        candidate = Path(str(entry))
        if candidate.is_absolute() or str(entry).startswith("~"):
            return _real(candidate)
        return _real(Path(working) / candidate)

    declared = [relative_to(entry) for entry in declaration["files"]["write"]]

    write_roots = list(declared)
    if writes_cwd and working not in write_roots:
        write_roots.append(working)
    # The private scratch directory is always writable: too much tooling breaks
    # without somewhere to put a temp file, and this one is ours.
    if scratch and scratch not in write_roots:
        write_roots.append(scratch)

    # `files.read`: `all` reads broadly; a list scopes reads to it, the write
    # folders, the working folder and the system folders (what `strict` does).
    read = declaration["files"]["read"]
    scoped_reads = read != "all"
    read_roots = _dedupe_paths([relative_to(entry) for entry in read] + declared + [working]) if scoped_reads else []

    unenforceable = []
    if declaration["allowed_imports"]:
        unenforceable.append("allowed_imports")
    # srt takes unix socket paths on macOS only ("seccomp cannot filter by
    # path"); the safe direction is to grant nothing elsewhere, and to say so.
    unix_sockets = [_real(entry) for entry in declaration["network"]["sockets"]]
    if unix_sockets and platform.system() != "Darwin":
        unenforceable.append("network.sockets")

    return SandboxPolicy(
        write_roots=_dedupe_paths(write_roots),
        read_roots=read_roots,
        scoped_reads=scoped_reads,
        read_deny=[relative_to(entry) for entry in declaration["files"]["deny"]],
        network=(not confined) or bool(network_domains) or bool(declaration["network"]["local"]),
        network_domains=network_domains,
        local_network=bool(declaration["network"]["local"]),
        unix_sockets=unix_sockets,
        confined=confined,
        preset=preset,
        cwd=working,
        scratch=scratch or None,
        advisory_commands=list(declaration["allowed_commands"]),
        unenforceable=unenforceable,
        env_passthrough=[str(name) for name in declaration["env"]],
    )


def default_policy(*, cwd: Optional[str] = None, tmpdir: Optional[str] = None) -> SandboxPolicy:
    """The policy an agent with no `sandbox:` block gets: the defaults, resolved against `cwd`."""
    policy = policy_from_metadata({}, cwd=cwd, tmpdir=tmpdir)
    assert policy is not None
    return policy


def _existing(paths: Sequence[str]) -> List[str]:
    """Only paths that are there, in order, once each. A rule for a missing
    path is harmless on macOS and a hard failure for a Linux bind, so the
    settings never name one."""
    seen: List[str] = []
    for path in paths:
        if path not in seen and os.path.exists(path):
            seen.append(path)
    return seen


def _dedupe_paths(paths: Iterable[str]) -> List[str]:
    """Order-preserving, and drops a path already covered by an ancestor.

    Two overlapping roots are not wrong, just noise in the generated profile,
    and a smaller profile is a profile someone will read.
    """
    out: List[str] = []
    for path in paths:
        if any(path == kept or path.startswith(kept.rstrip("/") + "/") for kept in out):
            continue
        out = [kept for kept in out if not kept.startswith(path.rstrip("/") + "/")]
        out.append(path)
    return out
