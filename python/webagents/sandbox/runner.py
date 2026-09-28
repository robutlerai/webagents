"""
Running one command under the OS sandbox: the entry points `ShellSkill` and
the SKILL.md runner call, and the environment scrub they share.

THE ENGINE IS srt (`@anthropic-ai/sandbox-runtime`, `srt.py`, 2026-09-26).
This module used to write Seatbelt profiles for macOS and bubblewrap argument
lists for Linux itself. srt is the same two primitives (`sandbox-exec` and
`bwrap`), maintained for Claude Code, plus the proxy outside the sandbox that
lets `network:` name hosts rather than being on or off, and it is a declared
dependency of the TypeScript package, so both SDKs confine the same agent
file through the same engine. What this package still does on top of srt
(pinning, scrubbing, failing closed, the private scratch folder, timeouts, the
private settings file) is listed at the top of `srt.py`.

THE BINARIES ARE PINNED BY ABSOLUTE PATH, and that is not fussiness. The
process being confined is the one we distrust. Resolving `node` or `cli.js`
through the agent's `PATH` would let a command that can write to any earlier
`PATH` entry choose its own sandbox and be "confined" by a no-op. Codex pins
`/usr/bin/sandbox-exec` for exactly this reason; srt itself resolves its
helpers from PATH, which is why it runs with a root-owned one here.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .policy import SandboxPolicy, SandboxUnavailable


def sandbox_available() -> bool:
    """Whether this machine can enforce anything at all."""
    return bool(backend_status()["available"])


def backend_status() -> Dict[str, object]:
    """What backend is in use, and why not, so `doctor` can say something true.

    A sandbox that reports itself working while enforcing nothing is the exact
    failure this whole area is about, so the reason is always carried. The
    answer is `srt.backend_status()`: found and preflighted, or the reason.
    """
    from . import srt

    return srt.backend_status()


def sandbox_required_reason(policy: Optional[SandboxPolicy]) -> Optional[str]:
    """Why a command from a caller other than the owner cannot run: None when
    `policy` confines and the engine is here, else the reason (pinned by the
    fixture's `refusals.not_owner_reasons`). The owner may run unconfined;
    nobody else may, which is defense in depth behind the owner-only default
    of the shell tool (S-248)."""
    if policy is None:
        return "this agent declares none"
    if not policy.confined:
        return "this agent's sandbox is unrestricted, which is no sandbox"
    status = backend_status()
    if not status["available"]:
        return f"the sandbox is unavailable: {status['reason']}"
    return None


def run_sandboxed(
    command: str,
    policy: SandboxPolicy,
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    capture_refusals: bool = False,
    cancel: Optional[threading.Event] = None,
) -> subprocess.CompletedProcess:
    """Run `command` under the policy, or refuse.

    A confined policy runs under srt. An UNCONFINED one (`preset:
    unrestricted`, `sandbox: off`, `--no-sandbox`) runs with the agent's own
    permissions, because that is what the declaration says; callers that must
    not do that check `policy.confined` or `sandbox_required_reason` first,
    as `ShellSkill` does for callers other than the owner. `capture_refusals`
    is `srt.run`'s: the refused hosts from srt's own log, for the chat.
    `cancel` is the interrupt (the chat's Esc or Ctrl+C): once it is set,
    the command's whole process group is killed. Either way the command's
    stdin is `/dev/null`, never the owner's terminal (S-317).

    Raises:
        SandboxUnavailable: when srt cannot enforce the policy here. Refusing
            is the point; see rule 2 in the package docstring.
        subprocess.TimeoutExpired: the command outlived `timeout`. Under srt
            the whole process group is killed first; srt alone would exit 0
            after a SIGTERM, which would read as success.
        srt.CommandInterrupted: `cancel` was set; the group has been killed.
    """
    from . import srt

    if not policy.confined:
        return srt.run_unconfined(command, policy, timeout=timeout, env=env, cancel=cancel)
    return srt.run(command, policy, timeout=timeout, env=env, capture_refusals=capture_refusals, cancel=cancel)


async def run_interruptibly(run: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """`run(*args, cancel=..., **kwargs)` in a worker thread, stopped with the
    task that awaits it (the ptypass-fixes lane, 2026-09-27).

    The chat's Esc and Ctrl+C cancel the turn's task. Cancelling
    `asyncio.to_thread` only abandons the thread, so the command ran on for
    13 to 18 seconds after "Interrupted" was on the screen, until its own
    timeout. Here the cancellation sets the thread's `cancel` event, which
    kills the command's whole process group (`srt._communicate`), and the
    cancellation carries on up the stack as before. The shell tool and the
    SKILL.md script runner both run their commands through this."""
    cancel = threading.Event()
    try:
        return await asyncio.to_thread(run, *args, cancel=cancel, **kwargs)
    except asyncio.CancelledError:
        cancel.set()
        raise


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


