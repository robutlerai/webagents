"""
Running one command under srt, `@anthropic-ai/sandbox-runtime` (2026-09-26).

WHY srt, AND WHY ITS CLI. The first engine (2026-09-23, S-217) wrote Seatbelt
profiles and bubblewrap argument lists itself, in this package, and only for
Python. srt is the same two primitives, maintained by Anthropic for Claude
Code, with the thing neither primitive has: a proxy outside the sandbox that
admits a LIST OF HOSTS, so `network:` can name `github.com` instead of being
on or off. It is a declared dependency of the TypeScript package, so both
SDKs confine a command through the same code and the same settings shape,
which is what parity of an agent file requires.

The CLI, not the library: srt's `SandboxManager` is a process-wide singleton
whose network lists are global, so two agents with different `network:`
lists in one process (the daemon) would share one policy. Running
`[node, cli.js, --settings <file>, -c <command>]` per command gives each
command its own policy and returns the command's own exit code.

THE ENGINE SHIPS INSIDE THIS PACKAGE (the sandbox-engine lane, 2026-09-27).
The sandbox is on by default and fails closed, and a pip package cannot
declare an npm dependency, so a Python user without Node and a global srt
had every shell command refused. srt 0.0.77 and its four dependencies are
vendored under `sandbox_engine/node_modules/` as npm publishes them
(`scripts/sandbox_engine_vendor.py` checks each tarball against the
TypeScript lockfile's integrity and records every file's sha256), and node
is `node` from PATH when it is 20.11 or later, else the one the
`nodejs-wheel-binaries` dependency installed. `webagents sandbox setup`
(`setup_checks`) says what this machine still lacks: the Linux programs
with this distribution's install line, what a container must allow,
Windows needing WSL 2. Pinned by `tests/fixtures/sandbox/sandbox_engine.json`.

WHAT THIS MODULE DOES THAT srt DOES NOT (spec section 1.6, pinned by
`tests/fixtures/sandbox/srt.json`):

1. **Pins the binaries.** srt resolves `bash`, `env`, `bwrap`, `socat` and
   `rg` from PATH, and the pnpm shim resolves `node` from PATH. The process
   being confined is the one we distrust, so `node` and `cli.js` are absolute
   (`cli.js` is the bundled one unless `WEBAGENTS_SRT_CLI` names another),
   srt itself runs with a root-owned PATH, and the agent's PATH is restored
   INSIDE the sandbox (`export PATH=...` as the first line of the command), so
   a binary planted in a user-writable folder only ever runs confined.
2. **Scrubs the environment** (S-220): secret-looking names are withheld
   (`runner.scrub_environment`), and the names that would change what srt
   ITSELF does are dropped (`DROPPED_ENV`): `NODE_OPTIONS` would run code in
   srt, outside the sandbox.
3. **Fails closed with the true reason.** srt only warns about a missing
   seccomp helper, does not probe whether a sandbox can be built, and its
   exit 1 is ambiguous with a command's own. So: node and cli.js must be found,
   the package version must be exactly `SRT_VERSION` (read from its
   package.json; `srt --version` prints npm's variable or "1.0.0"), the Linux
   dependencies must be present, and a one-time `-c true` preflight must pass.
4. **A private scratch folder.** srt's `/tmp/claude` is shared by every run
   on the machine and stays writable regardless. `CLAUDE_CODE_TMPDIR` points
   srt at `$TMPDIR/webagents-sandbox` instead, which is in `allowWrite`.
5. **Timeouts and interrupts.** srt exits 0 when its child is killed by
   SIGTERM, so a timed out command would look like success. The whole process
   group is killed here and `TimeoutExpired` is raised, which the shell
   reports as a timeout. An interrupt (the chat's Esc or Ctrl+C, passed in as
   `cancel`) kills the group the same way and raises `CommandInterrupted`
   (the ptypass-fixes lane, 2026-09-27): before, cancelling the turn only
   stopped the reply, and the command ran on in its thread for 13 to 18
   seconds, until its own timeout.
6. **The settings file cannot be rewritten from inside.** It is 0600 in a
   0700 directory that is outside every writable root (checked, refused
   otherwise), written for one command and removed after it.
7. **No terminal (S-317, 2026-09-27).** Every command, confined or not,
   starts with stdin at `/dev/null`, as the TypeScript engine's `stdio:
   ['ignore', ...]` always did. Popen with no `stdin` handed the command fd 0,
   the owner's terminal: `start_new_session` takes away the controlling
   terminal but not an open descriptor, and srt does not close it. The PTY
   pass watched a confined command read what the owner typed into the chat
   and draw a line on the screen, and srt's node, exiting after an
   interrupted command, restored the terminal mode it had inherited under the
   input box, so Enter arrived as Ctrl+J and nothing sent any more. The
   preflight and `node --version` get `/dev/null` too.

Never set: `enableWeakerNestedSandbox` (binds the host `/proc`),
`allowAppleEvents`, `allowAllUnixSockets`, `allowLocalBinding`, `allowPty`.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import signal
import site
import stat
import subprocess
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .policy import SandboxPolicy, SandboxUnavailable, install_write_denies

SRT_PACKAGE = "@anthropic-ai/sandbox-runtime"
SRT_VERSION = "0.0.77"

#: What a refusal says after the reason when srt cannot run here, declared
#: sandbox or default (fixture `refusals.unavailable_tail`). The engine ships
#: inside both packages (the sandbox-engine lane, 2026-09-27), so the reason
#: before it says what THIS machine lacks, with its fix (`unavailable_message`),
#: and the tail names the check and the opt-out.
UNAVAILABLE_TAIL = (
    "`webagents sandbox setup` checks this machine. To run commands with your permissions instead, "
    "pass --no-sandbox for this run or put `sandbox: off` in the agent file. The command was not run."
)

#: The one sentence the shell appends when a confined command's output shows
#: the sandbox refused something, naming the switch that opens it (fixture
#: `hints`; TypeScript `srt.ts` `REFUSAL_HINTS` is the reference). Detection
#: reads the command's own output: srt's proxy answers a host that is not
#: listed with 403 (`CONNECT tunnel failed, response 403`), a direct socket to
#: a local port or a unix socket gets EPERM, and a DNS lookup fails because
#: the sandbox has no resolver (the proxy resolves the hosts it admits).
REFUSAL_HINTS = {
    "hosts": "The sandbox refused a network connection: list the host under `sandbox: network: hosts:` in the agent file (or a group: npm, pypi, github).",
    "local": "The sandbox refused a local connection or a listening port: set `sandbox: network: local: true` in the agent file to allow them.",
    "sockets": "The sandbox refused a unix socket: list its path under `sandbox: network: sockets:` in the agent file to allow it.",
    "env": "The sandbox keeps `.env` and secret-looking variables from commands: list a variable name under `sandbox: env:` in the agent file to pass it through.",
}

_EPERM = re.compile(r"\bEPERM\b|Operation not permitted|Couldn't connect to server|Could not connect to server", re.I)
_PROXY_403 = re.compile(r"CONNECT tunnel failed, response 403|Tunnel connection failed: 403|403 Forbidden|blocked by network allowlist|Received HTTP code 403 from proxy", re.I)
_DNS_FAILURE = re.compile(r"Could not resolve host|getaddrinfo (?:ENOTFOUND|EAI_AGAIN)|nodename nor servname provided|Temporary failure in name resolution|Name or service not known|Name does not resolve|Failed to resolve", re.I)
_LOOPBACK = re.compile(r"\blocalhost\b|127\.0\.0\.1|\[?::1\]?|0\.0\.0\.0|\bbind\b|\blisten(?:ing)?\b|EADDRNOTAVAIL|http\.server|\bserver?\b|--port\b", re.I)
_UNIX_SOCKET = re.compile(r"\.sock\b|unix socket|unix://|ssh-agent|SSH_AUTH_SOCK|authentication agent|Docker daemon", re.I)
_DOTENV = re.compile(r"(?:^|[\s/'\"=])\.env(?:\.[A-Za-z0-9_.-]+)?\b")
_SOCKET_WORDS = re.compile(r"Cannot connect to the Docker daemon|Could not open a connection to your authentication agent", re.I)


def refusal_kind(command: str, output: str) -> Optional[str]:
    """Which refusal the output of `command` shows, if any (`hosts`, `local`, `sockets`, `env`)."""
    both = f"{command}\n{output}"
    if _EPERM.search(output) and _DOTENV.search(command):
        return "env"
    if _UNIX_SOCKET.search(both) and (_EPERM.search(output) or _SOCKET_WORDS.search(output)):
        return "sockets"
    if _PROXY_403.search(output) or _DNS_FAILURE.search(output):
        return "hosts"
    if _EPERM.search(output) and _LOOPBACK.search(both):
        return "local"
    return None


def refusal_hint(command: str, output: str) -> Optional[str]:
    """The hint sentence for `command`'s output, or None when nothing was refused."""
    kind = refusal_kind(command, output)
    return REFUSAL_HINTS[kind] if kind else None


#: srt's own record of a refused host, on its stderr under `--debug`:
#: `[SandboxDebug] Connection blocked to <host>:<port>` from its proxy (the
#: same line for a CONNECT and a plain HTTP request). Read from srt's output,
#: never guessed from the command.
_BLOCKED_LINE = re.compile(r"^\[SandboxDebug\] (?:Connection blocked to|HTTP request blocked to) (\S+):(\d+)\s*$")


def refused_hosts_from_srt_log(stderr: str) -> List[str]:
    hosts: List[str] = []
    for line in stderr.splitlines():
        match = _BLOCKED_LINE.match(line.strip())
        if not match:
            continue
        host = match.group(1).strip("[]").lower()
        if host not in hosts:
            hosts.append(host)
    return hosts


def _parse_dotenv(path: str) -> Dict[str, str]:
    """`KEY=value` lines of one `.env` file, quotes stripped, as the CLI's config store reads them."""
    out: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def env_from_dotenv(names: Sequence[str], cwd: str, environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The listed `env` names that the process environment does not have, read
    from the `.env` files the CLI loads (`./.env`, then the profile folder's
    `~/.webagents[-<profile>]/.env`), and nothing else from those files: the
    command cannot read `.env` itself, and S-220's rule stands, only listed
    names pass. The process environment wins where it has the name."""
    if not names:
        return {}
    env = os.environ if environ is None else environ
    profile = (env.get("WEBAGENTS_PROFILE") or "").strip()
    files = [
        os.path.join(cwd, ".env"),
        os.path.join(os.path.expanduser("~"), f".webagents-{profile}" if profile else ".webagents", ".env"),
    ]
    found: Dict[str, str] = {}
    for file in files:
        values = _parse_dotenv(file)
        for name in names:
            if name in env or name in found:
                continue
            hit = next((key for key in values if key.upper() == name.upper()), None)
            if hit is not None:
                found[name] = values[hit]
    return found

#: The PATH srt itself runs with: root-owned directories only.
SRT_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

#: Where an explicit install is named: the absolute path of srt's `dist/cli.js`,
#: and the node binary to run it with.
ENV_CLI = "WEBAGENTS_SRT_CLI"
ENV_NODE = "WEBAGENTS_SRT_NODE"

#: srt's own grace between SIGTERM and SIGKILL, matched here.
KILL_GRACE_SECONDS = 2.0

#: How often a running command looks at its `cancel` event (the chat's Esc
#: or Ctrl+C), in seconds: the group is killed within this of the keypress.
INTERRUPT_POLL_SECONDS = 0.1

#: What the shell tool and the SKILL.md script runner answer for a command an
#: interrupt stopped (fixture `srt.json` `interrupt.result`, the TypeScript
#: `INTERRUPTED_RESULT`).
INTERRUPTED_RESULT = "Interrupted: the command and everything it started were stopped."


class CommandInterrupted(RuntimeError):
    """The command's turn was interrupted (`cancel` was set): its whole
    process group has been killed, as a timeout kills it."""

#: How long the preflight may take before it counts as a failure.
PREFLIGHT_TIMEOUT_SECONDS = 60.0

#: What srt needs on Linux, looked for in root-owned directories only.
LINUX_DEPS = ("bwrap", "socat", "rg")
ROOT_OWNED_BIN_DIRS = ("/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/local/bin")

#: Names dropped from srt's environment (case-insensitively), on top of the
#: secret scrub: they change what srt or the wrapped bash does, outside or
#: before the sandbox. Pinned by the fixture's `env.dropped_from_srt`.
DROPPED_ENV = (
    "SRT_DEBUG",
    "CLAUDE_TMPDIR",
    "CLAUDE_CODE_TMPDIR",
    "NODE_OPTIONS",
    "NODE_PATH",
    "NODE_REPL_EXTERNAL_MODULE",
    "BASH_ENV",
    "ENV",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "FTP_PROXY",
    "GRPC_PROXY",
    "RSYNC_PROXY",
    "SANDBOX_RUNTIME",
)

#: The engine that ships inside this package (the sandbox-engine lane,
#: 2026-09-27): srt and its dependencies as npm publishes them, vendored by
#: `scripts/sandbox_engine_vendor.py`, which checks each tarball against the
#: TypeScript lockfile's integrity and lists every file with its sha256 in
#: `sandbox_engine/sandbox_engine_provenance.json`.
ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_engine")
BUNDLED_CLI = os.path.join(ENGINE_DIR, "node_modules", *SRT_PACKAGE.split("/"), "dist", "cli.js")

#: srt's `engines.node`: the oldest node it runs on.
NODE_MINIMUM = (20, 11, 0)

#: The PyPI package that brings node where there is none (a dependency of
#: this package on macOS and Linux), and the module folder it installs.
NODE_WHEEL = "nodejs-wheel-binaries"
_NODE_WHEEL_MODULE = "nodejs_wheel"

#: How each piece was found, as `doctor` and `sandbox setup` say it (fixture
#: `sandbox_engine.json`, `engine.python.cli_from` and `node.python_from`).
CLI_FROM_BUNDLED = "the engine that ships with webagents"
NODE_FROM_PATH = "PATH"

#: The fixes the reasons below come with (fixture `sandbox_engine.json` `fixes`).
FIX_ENGINE_BUNDLED = "reinstall webagents (`pip install --force-reinstall webagents`)"
FIX_WINDOWS = "run webagents inside WSL 2 (Windows Subsystem for Linux), where the Linux sandbox works"
FIX_PLATFORM = "run webagents on macOS, Linux or WSL 2"
FIX_TMPDIR = "point TMPDIR at a short path, such as /tmp/wa: srt's socket path must fit in 104 bytes on macOS"
FIX_CONTAINER = (
    "run the container with `--security-opt seccomp=unconfined --security-opt apparmor=unconfined` "
    "(or `--privileged`) so bubblewrap can create its namespaces, and as a user other than root"
)
FIX_NAMESPACES = {
    "userns_apparmor": "`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, or an AppArmor profile that grants bwrap `userns`",
    "userns_clone": "`sudo sysctl -w kernel.unprivileged_userns_clone=1`",
    "userns_max": "`sudo sysctl -w user.max_user_namespaces=15000`",
}


@dataclass
class SrtLocation:
    """An srt install this process may use."""

    node: str
    cli: str
    version: str
    package_dir: str
    #: How each was found, for `doctor`.
    node_from: str
    cli_from: str


def _executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _safe_binary(path: str) -> Tuple[bool, str]:
    """A regular, executable file that neither it nor its folder lets other
    users replace. The process being confined must not choose its own node."""
    real = os.path.realpath(path)
    if not _executable(real):
        return False, f"{path} is not an executable file"
    try:
        mode = os.stat(real).st_mode
        dir_mode = os.stat(os.path.dirname(real)).st_mode
    except OSError as error:
        return False, f"{path} cannot be inspected: {error}"
    if mode & stat.S_IWOTH or dir_mode & stat.S_IWOTH:
        return False, f"{path} or its folder is world-writable"
    return True, real


def _package_version(package_dir: str) -> Optional[str]:
    try:
        with open(os.path.join(package_dir, "package.json"), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if version else None


def _minimum_text() -> str:
    return ".".join(str(part) for part in NODE_MINIMUM)


def parse_node_version(output: str) -> Optional[Tuple[int, int, int]]:
    """`(major, minor, patch)` from `node --version`'s output, or None."""
    match = re.match(r"^v(\d+)\.(\d+)\.(\d+)", (output or "").strip())
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def _version_text(version: Optional[Tuple[int, int, int]]) -> str:
    return "v" + ".".join(str(part) for part in version) if version else "of no readable version"


#: `node --version` per binary, once per process.
_NODE_VERSIONS: Dict[str, Optional[Tuple[int, int, int]]] = {}


def _node_version(node: str) -> Optional[Tuple[int, int, int]]:
    """The version a node binary reports. It runs with an EMPTY environment:
    NODE_OPTIONS (`--require`) would otherwise run code in it, outside any
    sandbox, before srt has been reached."""
    if node not in _NODE_VERSIONS:
        try:
            result = subprocess.run([node, "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, env={}, cwd="/")
            _NODE_VERSIONS[node] = parse_node_version(result.stdout) if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            _NODE_VERSIONS[node] = None
    return _NODE_VERSIONS[node]


def _site_roots() -> List[str]:
    """Where an installed package lives: the folder holding this `webagents`
    package, then the running interpreter's own site-packages (sysconfig,
    and the user site). NEVER `sys.path`, which `python -m webagents` starts
    with the current folder, a folder a confined command may write: a
    `nodejs_wheel/bin/node` planted there would otherwise be run, unconfined,
    as the engine's node."""
    roots = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
    paths = sysconfig.get_paths()
    roots += [paths[key] for key in ("purelib", "platlib") if paths.get(key)]
    try:
        if site.ENABLE_USER_SITE:
            roots.append(site.getusersitepackages())
    except AttributeError:
        pass
    return list(dict.fromkeys(os.path.realpath(root) for root in roots))


def wheel_node() -> Optional[str]:
    """The node binary `nodejs-wheel-binaries` installed, or None: its
    `nodejs_wheel/bin/node` in one of `_site_roots()`. Never imported, so
    nothing of that package runs in this process."""
    for root in _site_roots():
        candidate = os.path.join(root, _NODE_WHEEL_MODULE, "bin", "node")
        if _executable(candidate):
            return candidate
    return None


def _absolute_path(value: Optional[str]) -> str:
    """PATH without its relative entries (`.`, `bin`, empty): `which` would
    resolve those against the current folder, which a confined command may
    write, and the node found here runs unconfined."""
    return os.pathsep.join(entry for entry in (value or "").split(os.pathsep) if os.path.isabs(entry))


def choose_node(environ: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str, str, str]:
    """The node srt runs with: `(path, how it was found, "", "")`, or
    `(None, "", reason, fix)`.

    `WEBAGENTS_SRT_NODE` when set (never second-guessed, but it must be a
    safe binary of srt's minimum); else `node` from PATH when it is safe and
    20.11 or later; else the node `nodejs-wheel-binaries` installed, which
    this package depends on so that a machine with no Node still confines
    commands. Nothing is downloaded, ever."""
    env = os.environ if environ is None else environ
    minimum = _minimum_text()
    explicit = (env.get(ENV_NODE) or "").strip()
    if explicit:
        ok, detail = _safe_binary(os.path.expanduser(explicit))
        if ok:
            version = _node_version(detail)
            if version is not None and version >= NODE_MINIMUM:
                return detail, ENV_NODE, "", ""
            detail = (
                f"{explicit} is node {_version_text(version)}; srt needs {minimum} or later"
                if version is not None
                else f"{explicit} does not report a node version; srt needs {minimum} or later"
            )
        return None, "", f"node was not found for srt ({ENV_NODE}: {detail})", f"point {ENV_NODE} at node {minimum} or later, or unset it"

    looked: List[str] = []
    search = _absolute_path(env.get("PATH"))
    on_path = shutil.which("node", path=search) if search else None
    if on_path:
        ok, detail = _safe_binary(on_path)
        if ok:
            version = _node_version(detail)
            if version is not None and version >= NODE_MINIMUM:
                return detail, NODE_FROM_PATH, "", ""
            looked.append(f"PATH has node {_version_text(version)}" if version is not None else "PATH's node does not report a version")
        else:
            looked.append(detail)
    else:
        looked.append("none on PATH")
    wheel = wheel_node()
    if wheel:
        ok, detail = _safe_binary(wheel)
        if ok:
            version = _node_version(detail)
            if version is not None and version >= NODE_MINIMUM:
                return detail, NODE_WHEEL, "", ""
            looked.append(f"{NODE_WHEEL} has node {_version_text(version)}" if version is not None else f"{NODE_WHEEL}'s node does not report a version")
        else:
            looked.append(detail)
    else:
        looked.append(f"{NODE_WHEEL} is not installed")
    return (
        None,
        "",
        f"no node {minimum} or later for srt ({'; '.join(looked)})",
        f"reinstall webagents, which brings node through {NODE_WHEEL}, or install Node {minimum} or later",
    )


def _engine_env_fix() -> str:
    return f"point {ENV_CLI} at srt {SRT_VERSION}'s dist/cli.js, or unset it to use the engine that ships with webagents"


def locate_cli(environ: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str, str, str]:
    """srt's `cli.js`: `(path, how it was found, "", "")`, or `(None, "",
    reason, fix)`. `WEBAGENTS_SRT_CLI` wins when set (an explicit install is
    never second-guessed); otherwise the engine that ships with this package.
    Either must be exactly `SRT_VERSION`, read from its own `package.json`:
    srt's settings schema strips unknown keys silently, so a version this SDK
    has not been checked against could drop a rule without a word."""
    env = os.environ if environ is None else environ
    explicit = (env.get(ENV_CLI) or "").strip()
    if explicit:
        cli = os.path.realpath(os.path.expanduser(explicit))
        if not os.path.isfile(cli):
            return None, "", f"{SRT_PACKAGE}@{SRT_VERSION} was not found ({ENV_CLI}={explicit} is not a file)", _engine_env_fix()
        cli_from = ENV_CLI
    else:
        if not os.path.isfile(BUNDLED_CLI):
            return None, "", f"{SRT_PACKAGE}@{SRT_VERSION} is missing from this webagents install ({BUNDLED_CLI} is not a file)", FIX_ENGINE_BUNDLED
        cli = os.path.realpath(BUNDLED_CLI)
        cli_from = CLI_FROM_BUNDLED
    version = _package_version(os.path.dirname(os.path.dirname(cli)))
    if version != SRT_VERSION:
        fix = _engine_env_fix() if cli_from == ENV_CLI else FIX_ENGINE_BUNDLED
        return None, "", f"{cli} is {SRT_PACKAGE} {version or 'of no readable version'}; this SDK requires exactly {SRT_VERSION}", fix
    return cli, cli_from, "", ""


def locate_engine(environ: Optional[Dict[str, str]] = None) -> Tuple[Optional[SrtLocation], str, str]:
    """Where srt and its node are: `(location, "", "")`, or `(None, reason,
    fix)` (`locate_cli`, then `choose_node`)."""
    cli, cli_from, reason, fix = locate_cli(environ)
    if cli is None:
        return None, reason, fix
    node, node_from, reason, fix = choose_node(environ)
    if node is None:
        return None, reason, fix
    location = SrtLocation(
        node=node,
        cli=cli,
        version=SRT_VERSION,
        package_dir=os.path.dirname(os.path.dirname(cli)),
        node_from=node_from,
        cli_from=cli_from,
    )
    return location, "", ""


def locate_srt(environ: Optional[Dict[str, str]] = None) -> Tuple[Optional[SrtLocation], str]:
    """Where srt is, or why it cannot be used: `(location, "")` or `(None, reason)`."""
    location, reason, _fix = locate_engine(environ)
    return location, reason


# ---------------------------------------------------------------------------
# The SDK's own install (S-316)
# ---------------------------------------------------------------------------


def _outermost_node_modules(path: str) -> str:
    """The outermost `node_modules` folder `path` lies in, else `path`: an npm
    or pnpm install keeps a package's dependencies somewhere under it."""
    parts = path.split(os.sep)
    if "node_modules" in parts:
        return os.sep.join(parts[: parts.index("node_modules") + 1]) or os.sep
    return path


#: `sdk_install_paths` per (WEBAGENTS_SRT_CLI, WEBAGENTS_SRT_NODE, PATH), once per process.
_INSTALLS: Dict[Tuple[str, str, str], List[str]] = {}


def sdk_install_paths(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """Where the running CLI and its sandbox engine live, realpath'd (S-316,
    fixture `sdk_install_deny.python`): the interpreter's environment
    (`sys.prefix`: every package the CLI imports, and the `.pth` files Python
    runs at every start), the `webagents` package itself (an editable install
    keeps it outside `sys.prefix`), the interpreter's site-packages and user
    site, srt's install when `WEBAGENTS_SRT_CLI` names one (its outermost
    `node_modules`; the bundled engine is inside the package), and the folder
    holding the node srt runs on. `build_settings` write-denies the ones that
    lie inside a write root (`policy.install_write_denies`)."""
    import sys

    from .policy import _dedupe_paths

    env = os.environ if environ is None else environ
    key = (env.get(ENV_CLI, ""), env.get(ENV_NODE, ""), env.get("PATH", ""))
    cached = _INSTALLS.get(key)
    if cached is not None:
        return list(cached)
    found: List[str] = []

    def add(path: Optional[str]) -> None:
        if path:
            real = os.path.realpath(path)
            if real not in found:
                found.append(real)

    add(sys.prefix)
    add(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = sysconfig.get_paths()
    for name in ("purelib", "platlib"):
        add(paths.get(name))
    try:
        if site.ENABLE_USER_SITE:
            add(site.getusersitepackages())
    except AttributeError:
        pass
    cli, cli_from, _reason, _fix = locate_cli(env)
    if cli is not None:
        add(_outermost_node_modules(os.path.dirname(os.path.dirname(cli))))
    node, _node_from, _reason, _fix = choose_node(env)
    if node is not None:
        add(os.path.dirname(os.path.realpath(node)))
    installs = _dedupe_paths(found)
    _INSTALLS[key] = installs
    return list(installs)


#: What `doctor` and `webagents sandbox setup` say when the install lies inside
#: the agent's folder (fixture `sdk_install_deny.report`, the TypeScript words).
INSTALL_INSIDE_DETAIL = (
    "webagents runs from inside this folder ({where}): shell commands may not write there, "
    "so no command can change the code that confines the next one"
)
INSTALL_INSIDE_FIX = "if commands here must install packages into that environment, install webagents outside this folder"
INSTALL_IS_THE_FOLDER = "this folder itself"


def install_inside_check(folders: Sequence[str], environ: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    """The `install` check for `doctor` and `sandbox setup`: a warning naming
    the parts of the install inside `folders` (the agent's folder, or the
    policy's write roots), relative to the first folder where they can be;
    None when there are none. The sandbox protects them either way; the
    warning says why a confined `pip install` into the CLI's own venv fails."""
    roots = [os.path.realpath(folder) for folder in folders if folder]
    hits = install_write_denies(roots, sdk_install_paths(environ))
    if not hits:
        return None
    base = roots[0].rstrip("/") or "/"
    shown = [INSTALL_IS_THE_FOLDER if hit == base else os.path.relpath(hit, base) if hit.startswith(base + "/") else hit for hit in hits]
    return _check("install", "warn", INSTALL_INSIDE_DETAIL.format(where=", ".join(shown)), INSTALL_INSIDE_FIX)


def _linux_programs() -> Tuple[Dict[str, str], List[str]]:
    """Absolute, root-owned paths of srt's Linux programs, and the ones missing."""
    found: Dict[str, str] = {}
    missing: List[str] = []
    for dep in LINUX_DEPS:
        for directory in ROOT_OWNED_BIN_DIRS:
            candidate = os.path.join(directory, dep)
            if _executable(candidate):
                found[dep] = candidate
                break
        else:
            missing.append(dep)
    return found, missing


def _programs_reason(missing: Sequence[str]) -> str:
    return f"{', '.join(missing)} not found in {', '.join(ROOT_OWNED_BIN_DIRS)}; srt needs bubblewrap, socat and ripgrep on Linux"


def _linux_deps() -> Tuple[Dict[str, str], str]:
    """Absolute, root-owned paths of srt's Linux dependencies, or why not."""
    found, missing = _linux_programs()
    return found, _programs_reason(missing) if missing else ""


#: The Linux packages that hold srt's programs, and the install line per
#: distribution family (fixture `sandbox_engine.json` `install`).
LINUX_PACKAGES = {"bwrap": "bubblewrap", "socat": "socat", "rg": "ripgrep"}
_INSTALL_FAMILIES = (
    (("debian", "ubuntu"), "sudo apt-get install {packages}"),
    (("fedora", "rhel", "centos", "rocky", "almalinux"), "sudo dnf install {packages}"),
    (("arch", "manjaro"), "sudo pacman -S {packages}"),
    (("alpine",), "sudo apk add {packages}"),
    (("opensuse", "suse", "sles"), "sudo zypper install {packages}"),
)


def _os_release_ids(text: str) -> List[str]:
    """`ID` then each `ID_LIKE` word of an os-release file, lower-cased."""
    values: Dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition("=")
        if key in ("ID", "ID_LIKE"):
            values[key] = value.strip().strip("\"'").lower()
    return [word for word in [values.get("ID", ""), *values.get("ID_LIKE", "").split()] if word]


def linux_install_fix(os_release: str, missing: Sequence[str]) -> str:
    """The line that installs the `missing` programs on the distribution
    `os_release` describes, in backquotes; a plain sentence when the family
    is not known."""
    packages = " ".join(LINUX_PACKAGES[dep] for dep in LINUX_DEPS if dep in missing)
    for identifier in _os_release_ids(os_release):
        for ids, line in _INSTALL_FAMILIES:
            if identifier in ids or any(identifier.startswith(f"{known}-") for known in ids):
                return "`" + line.format(packages=packages) + "`"
    return f"install {packages} with your package manager"


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _os_release(root: str = "/") -> str:
    return _read_text(os.path.join(root, "etc", "os-release")) or _read_text(os.path.join(root, "usr", "lib", "os-release"))


_CGROUP_WORDS = re.compile(r"docker|kubepods|containerd|lxc|podman")


def in_container(root: str = "/", environ: Optional[Dict[str, str]] = None) -> bool:
    """Whether this Linux process runs in a container (fixture `container`)."""
    env = os.environ if environ is None else environ
    if os.path.exists(os.path.join(root, ".dockerenv")) or os.path.exists(os.path.join(root, "run", ".containerenv")):
        return True
    if env.get("container"):
        return True
    return bool(_CGROUP_WORDS.search(_read_text(os.path.join(root, "proc", "1", "cgroup"))))


#: What restricts unprivileged user namespaces, read from /proc/sys in this
#: order (fixture `namespaces`): file, the value that restricts, the fix.
_NAMESPACE_CHECKS = (
    ("kernel/apparmor_restrict_unprivileged_userns", "1", "userns_apparmor"),
    ("kernel/unprivileged_userns_clone", "0", "userns_clone"),
    ("user/max_user_namespaces", "0", "userns_max"),
)


def namespace_restriction(root: str = "/") -> Tuple[str, str, str]:
    """`(file, value, fix)` for the first setting that stops bubblewrap from
    creating user namespaces here, or `("", "", "")`."""
    for relative, restricts, fix in _NAMESPACE_CHECKS:
        value = _read_text(os.path.join(root, "proc", "sys", relative)).strip()
        if value == restricts:
            return f"/proc/sys/{relative}", value, FIX_NAMESPACES[fix]
    return "", "", ""


#: The longest unix socket path srt can listen on and connect to on macOS
#: (fixture `tmpdir.limit`): `sun_path` holds 104 bytes, and node uses all
#: of them. Measured with real srt on 2026-09-27: 104 bytes listened and
#: carried a request, 105 failed with `listen EINVAL`.
SOCKET_PATH_LIMIT = 104

#: The socket name srt gives its proxy, `srt-mux-<pid>-<n>.sock` (its
#: `sandbox/mux-proxy.js`), with the widest pid macOS hands out (xnu
#: `PID_MAX` is 99999) and srt's first socket (fixture `tmpdir.socket`).
SRT_SOCKET_NAME = "srt-mux-99999-0.sock"


def srt_socket_probe(tmpdir: Optional[str] = None) -> str:
    """The longest socket path srt builds here (fixture `tmpdir`).

    srt listens on `join(os.tmpdir(), "srt-mux-<pid>-<n>.sock")`, and its
    `os.tmpdir()` is the TMPDIR this module gives it, as spelled: the
    preflight's `mkdtemp` folder under the temp folder (not realpath'd; the
    kernel measures the string), or a command's scratch folder, which the
    policy realpaths (`$TMPDIR/webagents-sandbox`). The longer of the two is
    measured.

    MEASURED, NOT GUESSED (the ptypass-fixes lane, 2026-09-27). The first
    version realpath'd the temp folder (`/private` added, which srt never
    sees) and counted a 7-digit pid, so every stock Mac read 110 bytes and
    `webagents sandbox setup` failed, while a confined `true` and a confined
    curl both ran (the PTY pass, `06b`, `06d`)."""
    from .policy import SCRATCH_DIR_NAME

    base = tmpdir or tempfile.gettempdir()
    preflight = os.path.join(os.path.abspath(base), "webagents-srt-preflight-" + "X" * 8)
    scratch = os.path.join(os.path.realpath(tmpdir or os.environ.get("TMPDIR") or "/tmp"), SCRATCH_DIR_NAME)
    longest = max(preflight, scratch, key=lambda folder: len(folder.encode("utf-8")))
    return os.path.join(longest, SRT_SOCKET_NAME)


def tmpdir_too_long(tmpdir: Optional[str] = None) -> bool:
    """Whether srt's socket path passes macOS's 104-byte limit here: exactly when srt itself would fail."""
    return len(srt_socket_probe(tmpdir).encode("utf-8")) > SOCKET_PATH_LIMIT


def _preflight_fix(system: str) -> str:
    """The most likely cause of a failed `-c true`, as the thing to do."""
    if system == "Darwin":
        return FIX_TMPDIR if tmpdir_too_long() else ""
    if system == "Linux":
        if in_container():
            return FIX_CONTAINER
        return namespace_restriction()[2]
    return ""


def _platform_refusal(system: str) -> Tuple[str, str]:
    if system == "Windows":
        return "there is no sandbox on native Windows", FIX_WINDOWS
    return f"there is no sandbox on {system}; srt supports macOS and Linux", FIX_PLATFORM


#: The status, per process and per (cli, node) setting, so a test that points
#: `WEBAGENTS_SRT_CLI` somewhere else gets a fresh answer.
_STATUS: Dict[Tuple[str, str], Dict[str, Any]] = {}


def reset_backend_status() -> None:
    _STATUS.clear()


def backend_status() -> Dict[str, Any]:
    """What backend is in use, and why not, so `doctor` can say something true.

    A sandbox that reports itself working while enforcing nothing is the exact
    failure this whole area is about, so the reason is always carried, and
    `available` is only True after a real `-c true` has run. `fix` is what to
    do on THIS machine (the sandbox-engine lane, 2026-09-27): the install line
    for the missing Linux programs, what a container must allow, WSL 2 on
    Windows; empty when there is nothing better than `webagents sandbox setup`.
    """
    key = (os.environ.get(ENV_CLI, ""), os.environ.get(ENV_NODE, ""))
    cached = _STATUS.get(key)
    if cached is not None:
        return dict(cached)

    system = platform.system()
    status: Dict[str, Any] = {
        "platform": system,
        "backend": None,
        "path": None,
        "node": None,
        "version": None,
        "available": False,
        "reason": "",
        "fix": "",
        "found": "",
    }

    def finish() -> Dict[str, Any]:
        _STATUS[key] = status
        return dict(status)

    if system not in ("Darwin", "Linux"):
        status["reason"], status["fix"] = _platform_refusal(system)
        return finish()

    # THE ENGINE FIRST, then the Linux programs it runs (2026-09-28). The
    # engine is what everything else depends on, and it ships with the
    # package, so a real install passes this step and meets the program check
    # next. The other order reported "bwrap, socat, rg not found" for a machine
    # whose engine was missing or the wrong version, which hid the actual
    # fault, and on a Linux CI runner without those programs it broke every
    # test of the engine's own refusals. The TypeScript twin has the same order.
    location, reason, fix = locate_engine()
    if location is None:
        status["reason"], status["fix"] = reason, fix
        return finish()

    deps: Dict[str, str] = {}
    if system == "Linux":
        deps, missing = _linux_programs()
        if missing:
            status["reason"] = _programs_reason(missing)
            status["fix"] = linux_install_fix(_os_release(), missing)
            return finish()

    status.update(
        path=location.cli,
        node=location.node,
        version=location.version,
        found=f"cli.js from {location.cli_from}, node from {location.node_from}",
    )
    failure = _preflight(location, deps)
    if failure:
        status["reason"] = failure
        status["fix"] = _preflight_fix(system)
    else:
        status["backend"] = "srt"
        status["available"] = True
    return finish()


def unavailable_message(status: Dict[str, Any]) -> str:
    """The refusal when the engine cannot run (fixture `sandbox_engine.json`
    `unavailable`): what this machine lacks, its fix when it has one, then
    `UNAVAILABLE_TAIL`, which names the check and the opt-out."""
    reason = status.get("reason") or "no sandbox backend here"
    fix = status.get("fix") or ""
    return f"{reason}: {fix}. {UNAVAILABLE_TAIL}" if fix else f"{reason}. {UNAVAILABLE_TAIL}"


#: The fix line's first half when this machine has no specific fix.
FIX_SETUP_POINTER = "run `webagents sandbox setup` for the details"


def unavailable_fix(status: Dict[str, Any]) -> str:
    """doctor's fix line for an engine that cannot run, the chat's
    `sandboxFix` in the same words (fixture `sandbox_engine.json` `doctor`):
    this machine's fix, then the opt-out."""
    fix = status.get("fix") or FIX_SETUP_POINTER
    return f"{fix}, or pass --no-sandbox to run commands with your permissions for this run"


def _check(name: str, status: str, detail: str, fix: Optional[str] = None) -> Dict[str, str]:
    out = {"name": name, "status": status, "detail": detail}
    if fix:
        out["fix"] = fix
    return out


def setup_checks(environ: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """`webagents sandbox setup`: whether the engine runs on this machine,
    as doctor-style checks (`name`, `status`, `detail`, `fix`), in the order
    of the fixture's `setup.checks`. A CHECK, NOT AN INSTALLER: it installs
    and changes nothing, and never downloads. The last check runs a real
    confined `true` through the engine (a fresh preflight, not the cached
    status), so an `ok` here means commands are sandboxed here."""
    env = os.environ if environ is None else environ
    system = platform.system()
    checks: List[Dict[str, str]] = []
    not_run = "not run: fix the checks above first"

    if system not in ("Darwin", "Linux"):
        reason, fix = _platform_refusal(system)
        checks.append(_check("platform", "fail", reason, fix))
        checks.append(_check("confined", "fail", not_run))
        return checks
    if system == "Darwin":
        checks.append(_check("platform", "ok", f"macOS {platform.mac_ver()[0] or platform.release()}"))
    else:
        pretty = next((line.partition("=")[2].strip().strip("\"'") for line in _os_release().splitlines() if line.startswith("PRETTY_NAME=")), "")
        checks.append(_check("platform", "ok", f"Linux ({pretty})" if pretty else "Linux"))

    blocked = False
    cli, cli_from, reason, fix = locate_cli(env)
    if cli is None:
        checks.append(_check("engine", "fail", reason, fix))
        blocked = True
    else:
        checks.append(_check("engine", "ok", f"srt {SRT_VERSION}, {cli_from}"))

    node, node_from, reason, fix = choose_node(env)
    if node is None:
        checks.append(_check("node", "fail", reason, fix))
        blocked = True
    else:
        checks.append(_check("node", "ok", f"{_version_text(_node_version(node))} from {node_from} ({node})"))

    deps: Dict[str, str] = {}
    if system == "Linux":
        deps, missing = _linux_programs()
        if missing:
            checks.append(_check("programs", "fail", _programs_reason(missing), linux_install_fix(_os_release(), missing)))
            blocked = True
        else:
            checks.append(_check("programs", "ok", "bwrap, socat and rg"))

    confined: Dict[str, str]
    if blocked or cli is None or node is None:
        confined = _check("confined", "fail", not_run)
    else:
        location = SrtLocation(node=node, cli=cli, version=SRT_VERSION, package_dir=os.path.dirname(os.path.dirname(cli)), node_from=node_from, cli_from=cli_from)
        failure = _preflight(location, deps)
        confined = (
            _check("confined", "fail", f"{failure}: shell commands are refused", _preflight_fix(system) or None)
            if failure
            else _check("confined", "ok", "a confined `true` ran: shell commands are sandboxed here")
        )
    ran = confined["status"] == "ok"

    if system == "Linux":
        # Said only when the confined `true` failed: a restriction bubblewrap
        # is exempt from (an AppArmor profile for bwrap) is not a problem.
        where, value, namespace_fix = namespace_restriction()
        if where and not ran:
            checks.append(_check("namespaces", "fail", f"unprivileged user namespaces are restricted here ({where} is {value})", namespace_fix))
        if in_container(environ=env):
            checks.append(
                _check("container", "ok", "inside a container, and bubblewrap works here")
                if ran
                else _check("container", "fail", "inside a container, where bubblewrap needs namespaces the container may not allow", FIX_CONTAINER)
            )
    if system == "Darwin" and tmpdir_too_long():
        probe = srt_socket_probe()
        checks.append(_check("tmpdir", "fail", f"srt's socket path here is {len(probe.encode('utf-8'))} bytes ({probe}), over macOS's 104", FIX_TMPDIR))
    # The install inside the folder the commands run in (S-316): said, since
    # it is why a confined `pip install` into the CLI's own venv fails.
    install = install_inside_check([os.getcwd()], env)
    if install is not None:
        checks.append(install)
    checks.append(confined)
    return checks


def _preflight(location: SrtLocation, deps: Dict[str, str]) -> str:
    """`-c true` under a minimal policy. Empty when it ran; else why not."""
    scratch = tempfile.mkdtemp(prefix="webagents-srt-preflight-")
    try:
        settings = {
            "network": {"allowedDomains": [], "deniedDomains": [], "strictAllowlist": True},
            "filesystem": {"denyRead": [], "allowWrite": [os.path.realpath(scratch)], "denyWrite": []},
        }
        settings.update(_linux_pins(deps))
        settings_dir = tempfile.mkdtemp(prefix="webagents-srt-", dir=scratch)
        os.chmod(settings_dir, 0o700)
        settings_path = _write_json_0600(os.path.join(settings_dir, "settings.json"), settings)
        env = srt_environment(dict(os.environ), scratch)
        try:
            result = subprocess.run(
                [location.node, location.cli, "--settings", settings_path, "-c", "true"],
                # Never the owner's terminal (S-317, module docstring rule 7).
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
                env=env,
                cwd=scratch,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return f"srt cannot start a sandbox here: {error}"
        if result.returncode != 0:
            lines = [line for line in (result.stderr or result.stdout).splitlines() if line.strip()]
            detail = lines[0].strip() if lines else f"exit {result.returncode}"
            return f"srt cannot start a sandbox here: {detail}"
        return ""
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _linux_pins(deps: Dict[str, str]) -> Dict[str, Any]:
    """srt's own pins for the Linux helpers, so it never takes them from PATH."""
    pins: Dict[str, Any] = {}
    if deps.get("bwrap"):
        pins["bwrapPath"] = deps["bwrap"]
    if deps.get("socat"):
        pins["socatPath"] = deps["socat"]
    if deps.get("rg"):
        pins["ripgrep"] = {"command": deps["rg"]}
    return pins


def _engine_reads() -> List[str]:
    """What srt itself must read inside the sandbox on Linux: its seccomp
    helper, which bubblewrap runs INSIDE the new mount namespace before the
    command. Under scoped reads (`strict`, and an agent with no `sandbox:`
    block) every read is denied but the listed roots, and an install outside
    them (a virtualenv, a home folder's site-packages) hid the helper, so
    every confined command failed with "apply-seccomp: No such file or
    directory". Found 2026-10-02, the first time the enforcement tests ran on
    a Linux runner. The TypeScript twin is `engineReads`."""
    cli, _how, _reason, _fix = locate_cli()
    if cli is None:
        return []
    helper = os.path.join(os.path.dirname(os.path.dirname(cli)), "vendor", "seccomp")
    return [helper] if os.path.isdir(helper) else []


def build_settings(policy: SandboxPolicy, deps: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The srt settings for a policy. Pinned by the fixture's `settings_cases`.

    `network.local` is srt's `allowLocalBinding` (bind and listen on any
    local port, connect to loopback directly; on Linux the command has its
    own network namespace, so a port it opens is reachable only inside it),
    and `network.sockets` is `allowUnixSockets`, macOS only. Neither key
    appears unless asked for, and `allowAllUnixSockets` never does.
    """
    if deps is None and platform.system() == "Linux":
        deps, _missing = _linux_deps()
    linux = platform.system() == "Linux"
    deny_writes = policy.deny_writes
    # The SDK's own install, when it lies inside a write root (S-316).
    for entry in install_write_denies(policy.write_roots, sdk_install_paths()):
        if entry not in deny_writes:
            deny_writes.append(entry)
    if linux:
        # A `denyWrite` for a missing path makes bubblewrap create a
        # placeholder on the host while the command runs; srt's own mandatory
        # set covers `.git/hooks` and `.git/config` when they exist.
        deny_writes = [path for path in deny_writes if os.path.exists(path)]
    filesystem: Dict[str, Any] = {
        "denyRead": list(policy.deny_reads),
        "allowWrite": list(policy.write_roots),
        "denyWrite": deny_writes,
    }
    if policy.scoped_reads:
        filesystem["allowRead"] = list(policy.allow_reads) + (_engine_reads() if linux else [])
    network: Dict[str, Any] = {
        "allowedDomains": list(policy.network_domains),
        "deniedDomains": [],
        # Never consult an ask callback: the CLI has none, and saying so
        # keeps the file honest if it is ever read by the library.
        "strictAllowlist": True,
    }
    if policy.local_network:
        network["allowLocalBinding"] = True
    if policy.unix_sockets and platform.system() == "Darwin":
        network["allowUnixSockets"] = list(policy.unix_sockets)
    settings: Dict[str, Any] = {"network": network, "filesystem": filesystem}
    settings.update(_linux_pins(deps or {}))
    return settings


def _write_json_0600(path: str, data: Dict[str, Any]) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def _inside(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root + "/") or root == "/"


def settings_base(policy: SandboxPolicy) -> str:
    """A folder for the settings file that no write root contains, or a
    refusal: the file must not be rewritable by the command it confines."""
    candidates = [
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath(os.path.join(os.path.expanduser("~"), ".cache", "webagents-srt")),
    ]
    for base in candidates:
        if not any(_inside(base, root) for root in policy.write_roots):
            os.makedirs(base, mode=0o700, exist_ok=True)
            return base
    raise SandboxUnavailable(
        f"no place for the settings file outside the writable folders ({', '.join(policy.write_roots)}). {UNAVAILABLE_TAIL}"
    )


def write_settings(policy: SandboxPolicy, settings: Optional[Dict[str, Any]] = None) -> str:
    """The settings file for one command: 0600, in a fresh 0700 directory
    outside every write root. The caller removes the directory afterwards."""
    base = settings_base(policy)
    directory = tempfile.mkdtemp(prefix="webagents-srt-", dir=base)
    os.chmod(directory, 0o700)
    return _write_json_0600(os.path.join(directory, "settings.json"), settings or build_settings(policy))


def srt_environment(scrubbed: Dict[str, str], scratch: Optional[str]) -> Dict[str, str]:
    """The environment srt runs with, from an already secret-scrubbed one:
    the names in `DROPPED_ENV` gone, a root-owned PATH, and the scratch folder
    named for srt to export as TMPDIR inside."""
    dropped = {name.upper() for name in DROPPED_ENV}
    env = {name: value for name, value in scrubbed.items() if name.upper() not in dropped}
    env["PATH"] = SRT_PATH
    if scratch:
        env["CLAUDE_CODE_TMPDIR"] = scratch
        env["TMPDIR"] = scratch
    return env


#: The NO_PROXY srt exports inside the sandbox (its `generateProxyEnvVars`,
#: the list of `SRT_VERSION` exactly): loopback and the private ranges are
#: sent direct rather than through srt's proxy. A LISTED LOOPBACK HOST WAS
#: UNREACHABLE (2026-09-27, the final e2e re-run, `g1b-netdebug`): every
#: ordinary client honours NO_PROXY, the sandbox refuses its direct socket
#: (`Operation not permitted`), and only the proxy, which the same list told it
#: to skip, could have reached the host `network:` named. So the command starts
#: with a NO_PROXY from which every entry covering a `network:` host has been
#: removed (`no_proxy_for`), and those hosts go through the proxy, which admits
#: them. Pinned by the fixture's `no_proxy`.
SRT_NO_PROXY = ("localhost", "127.0.0.1", "::1", "169.254.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")


def network_host(entry: str) -> str:
    """The host a `network:` entry names: without a port, a leading `*.` or
    IPv6 brackets, lower-cased."""
    import re

    value = entry.strip().lower()
    if value.startswith("*."):
        value = value[2:]
    v6 = re.match(r"^\[([^\]]+)\](?::\d{1,5})?$", value)
    if v6:
        return v6.group(1)
    if value.count(":") == 1:
        host, _, port = value.rpartition(":")
        if host and port.isdigit():
            value = host
    return value.rstrip(".")


def no_proxy_entry_covers(entry: str, host: str) -> bool:
    """Whether one NO_PROXY entry covers `host`: a name covers itself and every
    host under it, an address covers that address, a range covers an address
    inside it (the matching every client does, so what a client would bypass
    the proxy for)."""
    import ipaddress

    target = network_host(host)
    try:
        address: Optional[ipaddress._BaseAddress] = ipaddress.ip_address(target)
    except ValueError:
        address = None
    if "/" in entry:
        if address is None:
            return False
        network = ipaddress.ip_network(entry, strict=False)
        return address.version == network.version and address in network
    try:
        literal = ipaddress.ip_address(entry)
    except ValueError:
        literal = None
    if literal is not None:
        return address is not None and address == literal
    return address is None and (target == entry or target.endswith("." + entry))


def no_proxy_for(network_domains: Sequence[str]) -> List[str]:
    """srt's NO_PROXY less every entry that covers a host in `network_domains`, in srt's order."""
    return [entry for entry in SRT_NO_PROXY if not any(no_proxy_entry_covers(entry, host) for host in network_domains)]


def wrapped_command(command: str, user_path: str, network_domains: Sequence[str] = (), merge_stderr: bool = False) -> str:
    """The `-c` string: the agent's PATH restored inside, Node told to honour
    the proxy variables (`NODE_USE_ENV_PROXY=1`: Node 24 and later ignore
    HTTP_PROXY otherwise, so `fetch` to a listed host was stopped at the
    socket while curl reached it), the NO_PROXY the `network:` list leaves
    (only when it removed something; `unset` when nothing is left), then the
    command on its own line, so a trailing comment or `&` in the command
    cannot swallow anything. With `merge_stderr`, the command's stderr joins
    its stdout, so srt's own stderr carries only srt's lines
    (`refused_hosts_from_srt_log`)."""
    # npm's cache lives in `~/.npm`, which no policy makes writable, and npm
    # refuses to run without a writable cache (probed 2026-09-27: it asked for
    # a chown). The scratch folder is writable by construction.
    lines = [f"export PATH={shlex.quote(user_path)}", "export NODE_USE_ENV_PROXY=1", 'export npm_config_cache="${TMPDIR:-/tmp}/npm-cache"']
    kept = no_proxy_for(network_domains)
    if len(kept) != len(SRT_NO_PROXY):
        value = shlex.quote(",".join(kept))
        lines.append(f"export NO_PROXY={value} no_proxy={value}" if kept else "unset NO_PROXY no_proxy")
    if merge_stderr:
        lines.append("exec 2>&1")
    return "\n".join([*lines, command])


def _kill_group(process: subprocess.Popen) -> None:
    """SIGTERM the whole group, then SIGKILL what ignores it."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + KILL_GRACE_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _communicate(
    process: subprocess.Popen,
    argv: Any,
    timeout: Optional[float],
    cancel: Optional[threading.Event],
) -> Tuple[str, str]:
    """The command's output, once it ends, with the two ways it is stopped
    early: its timeout and an interrupt (`cancel`, set by the chat's Esc or
    Ctrl+C). Either kills the whole process group first. With a `cancel` the
    wait is taken in `INTERRUPT_POLL_SECONDS` steps; `communicate` may be
    called again after its own timeout without losing output."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        left = None if deadline is None else max(0.0, deadline - time.monotonic())
        step = left if cancel is None else (INTERRUPT_POLL_SECONDS if left is None else min(INTERRUPT_POLL_SECONDS, left))
        try:
            return process.communicate(timeout=step)
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                _kill_group(process)
                raise CommandInterrupted(INTERRUPTED_RESULT) from None
            if deadline is not None and time.monotonic() >= deadline:
                _kill_group(process)
                raise subprocess.TimeoutExpired(argv, timeout or 0) from None


def run(
    command: str,
    policy: SandboxPolicy,
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    capture_refusals: bool = False,
    cancel: Optional[threading.Event] = None,
) -> subprocess.CompletedProcess:
    """Run `command` under srt with `policy`, or refuse.

    With `capture_refusals` (the interactive chat, to ask the owner about a
    host by name) srt runs with `--debug`, the command's stderr is merged
    into its stdout, and the result carries `refused_hosts`, read from srt's
    own log; srt's log itself reaches nobody (it names the settings and the
    wrapped command), so `stderr` is empty then. `cancel` is the interrupt:
    once it is set the command's whole process group is killed.

    Raises:
        SandboxUnavailable: no usable srt here (see `backend_status`).
        subprocess.TimeoutExpired: the command outlived `timeout`; its whole
            process group has been killed first.
        CommandInterrupted: `cancel` was set; the group has been killed.
    """
    # Through `runner.backend_status`, the package's one answer, so a caller
    # (or a test) that replaces it there is honoured here too.
    from . import runner

    status = runner.backend_status()
    if not status["available"]:
        raise SandboxUnavailable(unavailable_message(status))

    given = dict(env if env is not None else os.environ)
    # A listed `env` name the process lacks is read from the `.env` files the
    # CLI loads; the files themselves stay unreadable to the command.
    scrubbed, _withheld = runner.scrub_environment({**env_from_dotenv(policy.env_passthrough, policy.cwd or os.getcwd(), given), **given}, policy.env_passthrough)
    user_path = scrubbed.get("PATH") or os.defpath
    child_env = srt_environment(scrubbed, policy.scratch)

    deps: Dict[str, str] = {}
    if platform.system() == "Linux":
        deps, _missing = _linux_deps()
    if cancel is not None and cancel.is_set():
        raise CommandInterrupted(INTERRUPTED_RESULT)
    settings_path = write_settings(policy, build_settings(policy, deps))
    argv = [str(status["node"]), str(status["path"]), *(["--debug"] if capture_refusals else []), "--settings", settings_path, "-c", wrapped_command(command, user_path, policy.network_domains, capture_refusals)]
    try:
        process = subprocess.Popen(
            argv,
            cwd=policy.cwd,
            env=child_env,
            # Never the owner's terminal (S-317, module docstring rule 7).
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Its own process group, so a timeout or an interrupt can kill
            # everything the command started, not just srt.
            start_new_session=True,
        )
        stdout, stderr = _communicate(process, argv, timeout, cancel)
        if not capture_refusals:
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, "")
        result.refused_hosts = refused_hosts_from_srt_log(stderr)  # type: ignore[attr-defined]
        return result
    finally:
        shutil.rmtree(os.path.dirname(settings_path), ignore_errors=True)


def run_unconfined(
    command: str,
    policy: SandboxPolicy,
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cancel: Optional[threading.Event] = None,
) -> subprocess.CompletedProcess:
    """`unrestricted`: no sandbox, by declaration. The environment is still
    scrubbed and the scratch folder still exported, because those cost the
    command nothing and the declaration asked for neither to be undone.

    Run as a confined command is, apart from srt: stdin at `/dev/null`
    (S-317; the terminal-mode half of that bug did not need srt), and its own
    process group, so a timeout or an interrupt stops everything it started
    (`subprocess.run` killed only the shell and left its children running,
    which the TypeScript twin never did)."""
    from .runner import scrub_environment

    child_env, _withheld = scrub_environment(dict(env if env is not None else os.environ), policy.env_passthrough)
    if policy.scratch:
        child_env["TMPDIR"] = policy.scratch
    if cancel is not None and cancel.is_set():
        raise CommandInterrupted(INTERRUPTED_RESULT)
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=policy.cwd,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    stdout, stderr = _communicate(process, command, timeout, cancel)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
