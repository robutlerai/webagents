"""
What a `sandbox:` declaration means, resolved into something a kernel can hold.

The declaration in an agent file is a wish. This module turns it into a
`SandboxPolicy`: absolute, symlink-resolved paths plus a network decision, with
the escalation set subtracted. `runner.py` turns that into a Seatbelt profile
or a bubblewrap argument list.

WHAT THE OS CAN AND CANNOT ENFORCE, stated plainly because the schema implies
more than is deliverable:

  * `allowed_folders`  -> ENFORCED. Becomes the write roots.
  * `preset`           -> ENFORCED, as defaults for folders and network.
  * `network`          -> ENFORCED, but only as on/off. No kernel primitive
                          expresses "allow github.com": Landlock's network
                          support is ports only and Seatbelt has no hostname
                          concept, which is why Claude Code, Codex and `srt`
                          each run an out-of-sandbox proxy for domain
                          allow-listing. Adding one is separate work.
  * `allowed_commands` -> NOT ENFORCEMENT. Kept, and relabelled: it decides
                          what runs without prompting. See the package
                          docstring for why a command allow-list cannot be a
                          boundary.
  * `allowed_imports`  -> NOT ENFORCEABLE HERE AT ALL. It is about Python
                          imports inside a process, and an OS sandbox confines
                          file and socket access, not `import` statements.
                          Reported as inert rather than silently ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


class SandboxUnavailable(RuntimeError):
    """No enforcement backend, and the policy asked for one.

    Raised rather than degraded. See rule 2 in the package docstring.
    """


#: Paths that stay WRITE-DENIED even inside an allowed root.
#:
#: Every serious implementation converged on a list like this independently
#: (Claude Code's is mandatory and cannot be exempted; Codex calls its version
#: `read_only_subpaths`). The reasoning is the same in each: a command that can
#: write here can grant itself permissions for the NEXT command, so allowing it
#: makes the whole policy advisory.
#:
#: Relative to each write root, matched on any path component.
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
)

#: `preset` -> (write the working directory?, allow network?, scope reads?).
#:
#: Reads and writes are deliberately asymmetric, the way Claude Code's are:
#: writes are enumerated everywhere, reads are broad by default and scoped only
#: under `strict`. Scoping reads is the expensive option, because module
#: resolution and toolchains traverse far more of the filesystem than anyone
#: expects, so it is opt-in rather than the default.
PRESETS = {
    "strict": (False, False, True),
    "development": (True, False, False),
    "unrestricted": (True, True, False),
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
    #: Whether reads are confined to `read_roots`. Only `strict` does this.
    scoped_reads: bool = False
    #: The private scratch directory, exported to the child as `TMPDIR`.
    scratch: Optional[str] = None
    #: Outbound network.
    network: bool = False
    #: Where the command runs.
    cwd: Optional[str] = None
    #: Carried through for reporting; NOT enforced here. See the module docstring.
    advisory_commands: List[str] = field(default_factory=list)
    #: Declared but unenforceable. Reported so it cannot be believed.
    unenforceable: List[str] = field(default_factory=list)
    #: Secret-looking variables the command may still see (S-220).
    env_passthrough: List[str] = field(default_factory=list)

    @property
    def deny_writes(self) -> List[str]:
        """Escalation paths to deny inside each write root."""
        denied = []
        for root in self.write_roots:
            for relative in ESCALATION_DENY:
                denied.append(os.path.join(root, relative))
        return denied

    def describe(self) -> str:
        """One line, for a refusal message or a log."""
        roots = ", ".join(self.write_roots) or "(nothing)"
        return f"writes: {roots}; network: {'on' if self.network else 'off'}"


def policy_from_metadata(
    sandbox: Any,
    *,
    cwd: Optional[str] = None,
    tmpdir: Optional[str] = None,
) -> Optional[SandboxPolicy]:
    """Resolve an agent file's `sandbox:` into a policy, or None if absent.

    `sandbox` is the `SandboxConfig` from `cli/loader/schema.py` (or any object
    or mapping carrying the same four fields). None means the file declared
    nothing, which is NOT the same as declaring an empty policy: the first
    means "no sandbox was asked for", the second means "deny everything".

    Args:
        sandbox: the declaration, or None
        cwd: where the command runs; defaults to the process's cwd
        tmpdir: a writable scratch directory; defaults to `$TMPDIR`
    """
    if sandbox is None:
        return None

    def _get(name: str, default: Any) -> Any:
        if isinstance(sandbox, dict):
            return sandbox.get(name, default)
        return getattr(sandbox, name, default)

    preset = str(_get("preset", "development") or "development").lower()
    if preset not in PRESETS:
        raise ValueError(
            f"unknown sandbox preset {preset!r}; expected one of "
            f"{', '.join(sorted(PRESETS))}"
        )
    writes_cwd, network, scoped_reads = PRESETS[preset]

    working = _real(cwd or os.getcwd())

    # A directory of ours inside the temp area, never the temp area itself.
    scratch_base = _real(tmpdir or os.environ.get("TMPDIR") or "/tmp")
    scratch = os.path.join(scratch_base, SCRATCH_DIR_NAME)
    try:
        os.makedirs(scratch, mode=0o700, exist_ok=True)
    except OSError:
        # A scratch directory we cannot create is one we must not promise.
        scratch = ""

    # `allowed_folders` is relative to the working directory when relative,
    # which is how `["."]`, the schema default, is meant to read.
    declared: List[str] = []
    for entry in _get("allowed_folders", None) or []:
        candidate = Path(str(entry))
        declared.append(
            _real(candidate if candidate.is_absolute() else Path(working) / candidate)
        )

    write_roots = list(declared)
    if writes_cwd and working not in write_roots:
        write_roots.append(working)
    # The private scratch directory is always writable: too much tooling breaks
    # without somewhere to put a temp file, and this one is ours.
    if scratch and scratch not in write_roots:
        write_roots.append(scratch)

    unenforceable = []
    if _get("allowed_imports", None):
        unenforceable.append("allowed_imports")

    return SandboxPolicy(
        write_roots=_dedupe_paths(write_roots),
        # Under `strict` reads are confined to what was declared plus the
        # working directory. Otherwise `read_roots` stays empty, which
        # `runner.py` reads as "the broad default".
        read_roots=_dedupe_paths(declared + [working]) if scoped_reads else [],
        scoped_reads=scoped_reads,
        network=bool(network),
        cwd=working,
        scratch=scratch or None,
        advisory_commands=list(_get("allowed_commands", None) or []),
        unenforceable=unenforceable,
        env_passthrough=[str(name) for name in (_get("env_passthrough", None) or [])],
    )


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
