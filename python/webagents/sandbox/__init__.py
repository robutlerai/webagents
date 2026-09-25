"""
OS-level sandboxing for commands an agent runs.

THE POINT OF THIS PACKAGE, in one sentence: an allow-list decides what the
model ASKED FOR, and a kernel sandbox decides what the process may TOUCH. Only
the second is a boundary.

`ShellSkill` gates commands with `shlex.split` plus an argv[0] allow-list.
Measured on macOS 26.2, that allow-list passes all of these:

    allow-list says PASS (saw 'echo')  ->  shell ran: 'vs'         # echo $(id -un)
    allow-list says PASS (saw 'echo')  ->  shell ran: 'vs'         # echo `whoami`
    allow-list says PASS (saw 'echo')  ->  shell ran: 'x CHAINED'  # echo x && echo CHAINED

`shlex` treats `$(...)` as a literal token; `shell=True` expands it afterwards,
so the guard inspects one string and the kernel runs another. This is not a
local quirk: CSA Labs' GuardFall study (2026-06-30) bypassed the command guards
of 10 of 11 open-source coding agents this way, and Anthropic documents the
same limit for Claude Code's own deny rules ("isn't a security boundary around
the program").

The same payloads under the wrapper in `runner.py`, measured:

    read a secret via $()     ->  Operation not permitted
    write outside the root    ->  Operation not permitted
    curl with network denied  ->  refused at the socket layer
    legitimate write in root  ->  ok
    widen its own sandbox     ->  sandbox_apply: Operation not permitted

The command string is never inspected. The restriction is inherited by children
and cannot be widened from inside, which is what makes it hold for `python -c`
and for pipelines.

FOUR RULES this package follows, each taken from a place someone else got it
wrong:

1. **Wrap each command, never the agent process.** `sandbox_init` is one-way
   and process-wide, so sandboxing the interpreter would also cut the agent off
   from its own LLM API.
2. **Fail closed.** If a file declares `sandbox:` and the backend is missing,
   refuse to run. Claude Code's default is to warn and run unsandboxed; that is
   a defensible UX choice for an interactive tool and the wrong one for a
   declared restriction, which is the defect S-217 is about.
3. **`realpath` every path.** `/tmp`, `/etc` and `/var` are symlinks into
   `/private` on macOS, so a rule written against `/tmp/x` silently matches
   nothing.
4. **Pin the sandbox binary by absolute path.** The process being confined is
   the one we distrust; if it can write anywhere earlier on `PATH` it would
   otherwise choose its own `sandbox-exec`.
"""

from .policy import (
    ESCALATION_DENY,
    SandboxPolicy,
    SandboxUnavailable,
    policy_from_metadata,
)
from .runner import backend_status, run_sandboxed, sandbox_available

__all__ = [
    "ESCALATION_DENY",
    "SandboxPolicy",
    "SandboxUnavailable",
    "policy_from_metadata",
    "backend_status",
    "run_sandboxed",
    "sandbox_available",
]
