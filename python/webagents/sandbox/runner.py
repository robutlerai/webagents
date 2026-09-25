"""
Running one command under an OS sandbox.

macOS uses Seatbelt (`sandbox-exec`), Linux uses bubblewrap (`bwrap`). Those
are the two primitives every comparable tool settled on: Claude Code, OpenAI
Codex CLI, Gemini CLI and Cursor CLI all use this pair. Codex has DEMOTED
Landlock to "legacy/backup" in favour of bubblewrap, which is the clearest
available signal about which Linux primitive survives contact with real
toolchains, so this does not use Landlock either.

THE BINARY IS PINNED BY ABSOLUTE PATH, and that is not fussiness. The process
being confined is the one we distrust. Resolving `sandbox-exec` through `PATH`
would let a command that can write to any earlier `PATH` entry choose its own
sandbox binary and be "confined" by a no-op. Codex pins `/usr/bin/sandbox-exec`
for exactly this reason and says so in a comment; Claude Code, measured against
its shipped binary on 2026-09-23, invokes it unpinned.

`sandbox-exec` has been marked DEPRECATED since OS X 10.8 (2012) and remains
the only option on macOS: the copy on a current machine is rebuilt with each OS
release, and four vendors ship it in production today.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from .policy import SandboxPolicy, SandboxUnavailable

#: macOS. Absolute and fixed; see the module docstring.
SEATBELT = "/usr/bin/sandbox-exec"

#: Linux. Searched, because distributions disagree, but only in directories
#: that an unprivileged process should not be able to write to. A `bwrap` found
#: anywhere else is not trusted, which is the same defence as pinning.
BWRAP_CANDIDATES = (
    "/usr/bin/bwrap",
    "/usr/local/bin/bwrap",
    "/bin/bwrap",
)

#: Read access every command needs before it can do anything at all. Without
#: these the dynamic loader fails and the process dies before `main`.
_SYSTEM_READ = (
    "/usr",
    "/bin",
    "/sbin",
    "/System",
    "/Library",
    "/private/etc",
    "/private/var/db",
    "/dev",
    "/opt/homebrew",
)


def _macos_backend() -> Optional[str]:
    return SEATBELT if os.path.isfile(SEATBELT) and os.access(SEATBELT, os.X_OK) else None


def _linux_backend() -> Optional[str]:
    for candidate in BWRAP_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


#: The probe result, per process. A few milliseconds, but `backend_status` is
#: consulted once per command and the answer cannot change under us.
_LINUX_PROBE: Optional[Tuple[bool, str]] = None


def _probe_bwrap(backend: str) -> Tuple[bool, str]:
    """Can bubblewrap actually build the sandbox we ask for, HERE?

    WHY A PROBE AND NOT A FILE CHECK (found 2026-09-23 by the first ever run of
    this branch, in a Linux container). The binary was present, so this module
    reported the backend available, and then EVERY command failed, the
    legitimate ones included, with `bwrap: Can't mount proc on /proc:
    Operation not permitted`. That is fail-closed in the narrow sense that
    nothing ran, but the report was a lie and the error was a riddle.

    The cause is a kernel rule rather than a bubblewrap one: a fresh procfs
    cannot be mounted in a user namespace while the existing `/proc` carries
    locked overmounts, which is exactly what a container runtime puts there.
    User, PID and network namespaces all worked in the same container; only
    the private `/proc` was refused.

    That `/proc` is NOT optional and is NOT quietly dropped: with broad reads
    the outer `/proc` would otherwise be visible, and `/proc/<pid>/environ` of
    the agent itself is readable by the same user, so dropping it would hand a
    command the agent's secrets by another door (see S-220).
    """
    global _LINUX_PROBE
    if _LINUX_PROBE is not None:
        return _LINUX_PROBE

    try:
        result = subprocess.run(
            [
                backend,
                "--new-session",
                "--die-with-parent",
                "--unshare-pid",
                "--proc", "/proc",
                "--dev", "/dev",
                "--ro-bind", "/", "/",
                "/bin/true",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            _LINUX_PROBE = (True, "")
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            detail = detail[0] if detail else f"exit {result.returncode}"
            hint = ""
            if "proc" in detail.lower():
                hint = (
                    " (usually a nested container: the runtime's masked /proc "
                    "prevents a private one being mounted)"
                )
            elif "namespace" in detail.lower() or "permission" in detail.lower():
                hint = (
                    " (unprivileged user namespaces are disabled; on Ubuntu "
                    "24.04+ AppArmor restricts them)"
                )
            _LINUX_PROBE = (False, f"bubblewrap cannot build the sandbox here: {detail}{hint}")
    except (OSError, subprocess.SubprocessError) as error:
        _LINUX_PROBE = (False, f"bubblewrap could not be run: {error}")
    return _LINUX_PROBE


def sandbox_available() -> bool:
    """Whether this machine can enforce anything at all."""
    return backend_status()["available"]


def backend_status() -> Dict[str, object]:
    """What backend is in use, and why not, so `doctor` can say something true.

    A sandbox that reports itself working while enforcing nothing is the exact
    failure this whole area is about, so the reason is always carried.
    """
    system = platform.system()
    if system == "Darwin":
        path = _macos_backend()
        return {
            "platform": system,
            "backend": "seatbelt" if path else None,
            "path": path,
            "available": path is not None,
            "reason": "" if path else f"{SEATBELT} is missing or not executable",
        }
    if system == "Linux":
        path = _linux_backend()
        if path is None:
            return {
                "platform": system,
                "backend": None,
                "path": None,
                "available": False,
                "reason": "bwrap not found in " + ", ".join(BWRAP_CANDIDATES) + " (install bubblewrap)",
            }
        works, why = _probe_bwrap(path)
        return {
            "platform": system,
            "backend": "bubblewrap" if works else None,
            "path": path,
            "available": works,
            "reason": why,
        }
    return {
        "platform": system,
        "backend": None,
        "path": None,
        "available": False,
        "reason": f"no sandbox backend for {system}; only macOS and Linux are supported",
    }


def build_seatbelt_profile(policy: SandboxPolicy) -> str:
    """The SBPL profile for a policy.

    `(deny default)` and then an allow-list, which is the shape Gemini CLI's
    profiles use and which its own comment insists on keeping
    ("do not switch to `(allow default)`").

    `(allow file-read-metadata)` is global on purpose: scoping reads breaks
    `stat()` walks, and Node and Python both traverse intermediate directories
    during module resolution. Gemini CLI's strict profile carries the same
    carve-out with the same note.
    """
    lines = [
        "(version 1)",
        "(deny default)",
        # Apple's own system profile. It is private interface and subject to
        # change, which is the price of a profile that does not SIGABRT the
        # moment a process touches something mundane.
        '(import "system.sb")',
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        "(allow pseudo-tty)",
    ]

    if policy.scoped_reads:
        # `strict`: reads confined to what was declared, plus the system paths
        # a process needs to start at all.
        readable = list(_SYSTEM_READ) + list(policy.read_roots) + list(policy.write_roots)
        if policy.cwd:
            readable.append(policy.cwd)
        lines.append(
            "(allow file-read* "
            + " ".join(f'(subpath "{_escape(path)}")' for path in _existing(readable))
            + ")"
        )
    else:
        # Reads broad, writes enumerated. The same asymmetry Claude Code uses,
        # and the reason is cost: scoping reads breaks module resolution and
        # toolchains in ways that are tedious to chase, so it is opt-in.
        lines.append("(allow file-read*)")
    lines.append(
        "(allow file-map-executable "
        + " ".join(
            f'(subpath "{_escape(path)}")'
            for path in _existing(_SYSTEM_READ)
        )
        + ")"
    )

    if policy.write_roots:
        lines.append(
            "(allow file-write* "
            + " ".join(f'(subpath "{_escape(path)}")' for path in policy.write_roots)
            + ' (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr")'
            + ' (regex #"^/dev/ttys[0-9]*$"))'
        )
    else:
        lines.append(
            '(allow file-write* (literal "/dev/null") (literal "/dev/stdout") '
            '(literal "/dev/stderr") (regex #"^/dev/ttys[0-9]*$"))'
        )

    # AFTER the allow, because SBPL is last-match-wins. This is what keeps a
    # command from granting itself permissions for the next one.
    if policy.deny_writes:
        lines.append(
            "(deny file-write* "
            + " ".join(f'(subpath "{_escape(path)}")' for path in policy.deny_writes)
            + ")"
        )

    if policy.network:
        lines.append("(allow network-outbound)")
        lines.append("(allow network-bind)")
        lines.append("(allow system-socket)")
    # Omitting the network rules entirely is the denial: `(deny default)`
    # already covers it, and it bites at the socket layer rather than at DNS.

    return "\n".join(lines) + "\n"


#: What a Linux process needs to START under `strict`, where the root is empty.
#: `/lib*` carry the dynamic loader: without them `execvp /bin/sh` fails with
#: "No such file or directory", which is what the first run of this branch did.
_LINUX_SYSTEM_DIRS = ("/usr", "/etc", "/opt")
#: On merged-usr distributions these are SYMLINKS into `/usr`; binding one would
#: bind its target at the wrong path, so they are recreated as links instead.
_LINUX_MAYBE_SYMLINKED = ("/bin", "/sbin", "/lib", "/lib64", "/lib32", "/libx32")


def _linux_system_mounts() -> List[str]:
    args: List[str] = []
    for path in _LINUX_SYSTEM_DIRS:
        if os.path.isdir(path):
            args += ["--ro-bind", path, path]
    for path in _LINUX_MAYBE_SYMLINKED:
        if os.path.islink(path):
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    return args


def build_bwrap_argv(policy: SandboxPolicy, backend: str) -> List[str]:
    """The bubblewrap argument list for a policy.

    bubblewrap's own README is blunt that it supplies a mechanism and not a
    policy: "the level of protection ... is entirely determined by the
    arguments passed to bubblewrap". So the arguments are the security model.

    ORDER IS THE SECURITY MODEL TOO, and the first version had it wrong (found
    2026-09-23 by the first ever run of this branch). In bubblewrap a later
    mount covers an earlier one. `--ro-bind / /` came AFTER `--proc /proc` and
    `--dev /dev`, so under broad reads it laid the HOST's `/proc` and `/dev`
    back over the private ones. The private `/proc` exists precisely so a
    command cannot read `/proc/<pid>/environ` of the agent that launched it,
    so this undid the one thing it was there for; and `/dev/null` became
    read-only, which is how it was noticed. The layout is now, strictly:

        1. the root filesystem   (broad: `/` read-only; strict: empty tmpfs)
        2. `/proc` and `/dev`    (private, ON TOP of the root)
        3. the write roots       (read-write)
        4. the escalation set    (read-only again, on top of the write roots)
    """
    argv = [
        backend,
        # Without this, TIOCSTI can inject into the controlling terminal and
        # run commands OUTSIDE the sandbox (CVE-2017-5226). bubblewrap
        # documents this under Limitations.
        "--new-session",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    if not policy.network:
        argv.append("--unshare-net")

    # 1. The root.
    if policy.scoped_reads:
        # Nothing visible that is not named: an empty root, then the system
        # directories a process needs to start, then what was declared.
        argv += ["--tmpfs", "/"]
        argv += _linux_system_mounts()
        for path in _existing(policy.read_roots):
            argv += ["--ro-bind", path, path]
    else:
        argv += ["--ro-bind", "/", "/"]

    # 2. Private /proc and /dev, AFTER the root so they are not covered by it.
    argv += ["--proc", "/proc", "--dev", "/dev"]

    # 3. Writable roots.
    for path in policy.write_roots:
        if os.path.exists(path):
            argv += ["--bind", path, path]

    # 4. The escalation set, read-only again, on top of the writable roots.
    for path in policy.deny_writes:
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]

    if policy.cwd:
        argv += ["--chdir", policy.cwd]
    return argv


def run_sandboxed(
    command: str,
    policy: SandboxPolicy,
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run `command` under the OS sandbox, or refuse.

    Raises:
        SandboxUnavailable: when no backend can enforce the policy. Refusing is
            the point; see rule 2 in the package docstring.
    """
    status = backend_status()
    if not status["available"]:
        raise SandboxUnavailable(
            f"{status['reason']}. The agent declared a sandbox, so the command "
            f"was not run."
        )

    backend = str(status["path"])
    if status["backend"] == "seatbelt":
        argv = [
            backend,
            "-p",
            build_seatbelt_profile(policy),
            "--",
            "/bin/sh",
            "-c",
            command,
        ]
    else:
        argv = build_bwrap_argv(policy, backend) + ["/bin/sh", "-c", command]

    # Secrets are withheld from the command (S-220). Confining its files and
    # network while handing it every API key in the process was the hole.
    child_env, _withheld = scrub_environment(
        dict(env if env is not None else os.environ), policy.env_passthrough
    )

    # Point the child at the scratch directory it is actually allowed to use.
    # Without this a tool writes to the real `$TMPDIR`, which is outside the
    # policy, and fails in a way that reads like the sandbox is broken.
    if policy.scratch:
        child_env["TMPDIR"] = policy.scratch

    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=policy.cwd if status["backend"] == "seatbelt" else None,
        env=child_env,
    )


#: A variable whose NAME contains one of these is withheld from a sandboxed
#: command unless the agent lists it in `sandbox.env_passthrough` (S-220).
#:
#: Matched on the name, case-insensitively, because the value tells you nothing
#: and the name is how every credential in practice announces itself. OpenAI's
#: Codex CLI defaults to excluding `*KEY*`, `*SECRET*` and `*TOKEN*` for the
#: same reason; this adds the other spellings that turn up in real
#: environments. Over-matching (an innocent `KEYBOARD_LAYOUT`) costs a command
#: one variable it almost certainly did not need; under-matching costs a key.
SECRET_NAME_PARTS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE",
    "AUTH",
    "SESSION",
    "COOKIE",
)


def scrub_environment(
    environ: Dict[str, str], passthrough: Sequence[str] = ()
) -> Tuple[Dict[str, str], List[str]]:
    """The environment a sandboxed command gets, and the names withheld.

    WHY THIS EXISTS (S-220, measured 2026-09-23). The first version of the
    sandbox confined files and network and then ran the command with a copy of
    the agent's ENTIRE environment. Under the strictest preset,
    `echo $OPENAI_API_KEY` printed the key: into the tool result, the model's
    context, the transcript, and the saved session. It needed no file access
    and no network at all.
    """
    allowed = {name.upper() for name in passthrough}
    kept: Dict[str, str] = {}
    withheld: List[str] = []
    for name, value in environ.items():
        upper = name.upper()
        if upper not in allowed and any(part in upper for part in SECRET_NAME_PARTS):
            withheld.append(name)
            continue
        kept[name] = value
    return kept, sorted(withheld)


def _existing(paths: Sequence[str]) -> List[str]:
    """Only paths that are there.

    A Seatbelt rule for a missing path is harmless; a bwrap `--ro-bind` for one
    is a hard failure, so both filter through here and stay in step.
    """
    seen = []
    for path in paths:
        if path not in seen and os.path.exists(path):
            seen.append(path)
    return seen


def _escape(path: str) -> str:
    """SBPL string literal escaping."""
    return path.replace("\\", "\\\\").replace('"', '\\"')
