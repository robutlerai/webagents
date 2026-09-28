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

THE ENGINE IS srt (2026-09-26): `@anthropic-ai/sandbox-runtime` 0.0.77, run
as a CLI per command (`srt.py`). It is Seatbelt and bubblewrap underneath,
plus a proxy outside the sandbox that admits the hosts a file lists under
`network:`, and it is the same engine the TypeScript SDK runs, so an agent
file is confined identically under either CLI (`tests/fixtures/sandbox/srt.json`).
The four rules above still hold; `srt.py` says what this package does that
srt does not.
"""

from .policy import (
    AGENT_FILE_PATTERNS,
    CREDENTIAL_DIRS,
    DEFAULT_PRESET,
    ENV_NO_SANDBOX,
    ESCALATION_DENY,
    HOST_GROUPS,
    OFF_PRESET,
    PRESETS,
    PROFILE_DIR_PATTERN,
    ROOT_READ_DENY,
    ROOT_READ_DENY_PATTERNS,
    SANDBOX_ACCEPTED_KEYS,
    SANDBOX_ALIASES,
    SANDBOX_FILES_KEYS,
    SANDBOX_KEYS,
    SANDBOX_NETWORK_KEYS,
    SandboxDeclarationError,
    SandboxPolicy,
    SandboxUnavailable,
    agent_file_denies,
    check_network_entry,
    default_policy,
    expand_hosts,
    install_write_denies,
    is_sandbox_off,
    matches_agent_file_pattern,
    no_sandbox_requested,
    normalize_declaration,
    policy_from_metadata,
    profile_dir_denies,
    root_read_denies,
    sandbox_state,
)
from .runner import backend_status, run_interruptibly, run_sandboxed, sandbox_available, sandbox_required_reason
from .srt import (
    INTERRUPTED_RESULT,
    REFUSAL_HINTS,
    SRT_PACKAGE,
    SRT_VERSION,
    UNAVAILABLE_TAIL,
    CommandInterrupted,
    env_from_dotenv,
    refusal_hint,
    refusal_kind,
    refused_hosts_from_srt_log,
    setup_checks,
    unavailable_fix,
    unavailable_message,
)

__all__ = [
    "AGENT_FILE_PATTERNS",
    "CREDENTIAL_DIRS",
    "DEFAULT_PRESET",
    "ENV_NO_SANDBOX",
    "ESCALATION_DENY",
    "HOST_GROUPS",
    "INTERRUPTED_RESULT",
    "OFF_PRESET",
    "PRESETS",
    "PROFILE_DIR_PATTERN",
    "REFUSAL_HINTS",
    "ROOT_READ_DENY",
    "ROOT_READ_DENY_PATTERNS",
    "SANDBOX_ACCEPTED_KEYS",
    "SANDBOX_ALIASES",
    "SANDBOX_FILES_KEYS",
    "SANDBOX_KEYS",
    "SANDBOX_NETWORK_KEYS",
    "SRT_PACKAGE",
    "SRT_VERSION",
    "UNAVAILABLE_TAIL",
    "CommandInterrupted",
    "SandboxDeclarationError",
    "SandboxPolicy",
    "SandboxUnavailable",
    "agent_file_denies",
    "check_network_entry",
    "default_policy",
    "env_from_dotenv",
    "expand_hosts",
    "install_write_denies",
    "is_sandbox_off",
    "matches_agent_file_pattern",
    "no_sandbox_requested",
    "normalize_declaration",
    "policy_from_metadata",
    "profile_dir_denies",
    "refusal_hint",
    "refusal_kind",
    "refused_hosts_from_srt_log",
    "root_read_denies",
    "sandbox_state",
    "backend_status",
    "run_interruptibly",
    "run_sandboxed",
    "sandbox_available",
    "sandbox_required_reason",
    "setup_checks",
    "unavailable_fix",
    "unavailable_message",
]
