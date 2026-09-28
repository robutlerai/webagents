"""
Shell Skill

Shell command execution with proper sandboxing using command whitelist/blacklist.

OWNER-ONLY BY DEFAULT (2026-09-26, S-248). `run_command` took the decorator's
default scope, `all`, so anyone the platform let message an agent connected
through Portal Connect, or anyone who could reach a served agent with any
bearer, could run commands as the developer's user on the machine serving it.
The allow-list is not a boundary (its own comment below says so). The tool is
`scope="owner"` now: the owner and admins see it; nobody else does unless the
agent file hands it to a group with `access: tools:` (ADR-0045), which
replaces the scope with that group's. The TypeScript `ShellSkill` declares
the same.
"""

import asyncio
import os
import shlex
import subprocess
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Set, Dict, Any, Tuple
from pathlib import Path

from ...base import Skill
from webagents.agents.tools.decorators import tool
from webagents.sandbox.srt import INTERRUPTED_RESULT, CommandInterrupted

#: Said once when an agent loads with `sandbox: off` or `preset:
#: unrestricted` (pinned by `tests/fixtures/sandbox/srt.json`,
#: `unrestricted.warning`).
#:
#: ONE WORDING (the ptypass-fixes lane, 2026-09-27): this line said "Use
#: `development` or `strict`" while `/sandbox` and `doctor` said "Remove
#: `sandbox: off`". It is now `/sandbox`'s headline and fix, word for word
#: (`chat_words` `sandboxOff`, `sandboxOffFix`, `sandboxFlagFix`).
UNRESTRICTED_WARNING = (
    "Sandbox: off (agent file): commands are not confined and run with your permissions. "
    "Remove `sandbox: off` (or `preset: unrestricted`) from the agent file to confine them."
)

#: Said once when the run started with `--no-sandbox` (fixture `status.warnings.off_flag`).
NO_SANDBOX_WARNING = (
    "Sandbox: off (--no-sandbox): commands are not confined and run with your permissions. "
    "Run without --no-sandbox to confine them."
)


def _say_on_stderr(message: str, origin: str) -> None:
    import sys

    print(message, file=sys.stderr)


#: Where the opt-out is said when an agent loads (the ptypass-fixes lane,
#: 2026-09-27): on stderr by default (`-p`, `serve()`, the daemon). The chat
#: says it as one of its own notices, wrapped at words (the raw line wrapped
#: mid-word above the welcome card), and `doctor` has a line of its own and
#: does not repeat it (`opt_out_announcer`).
_OPT_OUT_ANNOUNCER: List[Callable[[str, str], None]] = [_say_on_stderr]


@contextmanager
def opt_out_announcer(announce: Callable[[str, str], None]) -> Iterator[None]:
    """While agents load in this block, say the opt-out through
    `announce(message, origin)` instead of on stderr."""
    previous = _OPT_OUT_ANNOUNCER[0]
    _OPT_OUT_ANNOUNCER[0] = announce
    try:
        yield
    finally:
        _OPT_OUT_ANNOUNCER[0] = previous

# THE COMMAND LIST GATES UNCONFINED COMMANDS ONLY (the ptypass-fixes lane,
# 2026-09-27). The PTY pass found `sleep` and `mkdir` refused in both SDKs,
# `python3` in TypeScript only and `rg` in Python only, before the sandbox was
# ever reached, while `docs/cli/sandbox.md` says the kernel is the boundary.
# A confined command now runs whatever it is, and the kernel decides what it
# may touch; only a name the agent file lists under `blocked_commands` is
# still refused. With `sandbox: off` (or `--no-sandbox`) the list below is the
# only gate there is, so it stays, as ONE list and ONE wording in both SDKs
# (TypeScript `DEFAULT_ALLOWED`, `DEFAULT_BLOCKED`), pinned by the fixture
# `sandbox/srt.json` (`shell_allowlist`). A command's name is the base name of
# the word in each command position (the first word, and the first after
# `&&`, `||`, `|` or `;`).
DEFAULT_ALLOWED = (
    "ls", "cat", "grep", "find", "head", "tail", "wc", "echo", "date",
    "pwd", "which", "whereis", "git", "npm", "pip", "python", "python3", "node",
    "uvx", "curl", "wget", "rg", "fd",
)
DEFAULT_BLOCKED = (
    "rm", "rmdir", "dd", "mkfs", "fdisk", "kill", "killall", "pkill",
    "shutdown", "reboot", "halt", "su", "sudo", "chmod", "chown",
)
NOT_ALLOWED = "Command '{name}' is not in the allowlist"
IS_BLOCKED = "Command '{name}' is blocked"

# OS-LEVEL ENFORCEMENT, ON BY DEFAULT (2026-09-27, the sandbox-default lane,
# owner decision; the TypeScript `ShellSkill` is the reference). An agent with
# no `sandbox:` block gets the defaults (`development`: writes in its folder
# and a scratch folder, no network, no local servers, `.env` and the
# credential folders unreadable). The opt-out is explicit and said loudly:
# `sandbox: off` in the agent file (`preset: unrestricted` stays accepted),
# or `--no-sandbox` for one run, which lifts confinement from the OWNER'S
# commands only. A sandbox that cannot be enforced here refuses the command
# rather than running it free, declared or default. When a confined command's
# output shows a refusal, ONE sentence names the switch that opens it
# (`refusal_hint`), and in the interactive chat only, a refused host is asked
# about by name (`asker`), read from srt's own proxy log.


class ShellSkill(Skill):
    """Shell command execution with sandboxing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Command sandboxing
        self.allowed_commands: Set[str] = self._load_allowed_commands()
        self.blocked_commands: Set[str] = self._load_blocked_commands()
        #: The names the agent file itself blocks: refused even when confined.
        self.declared_blocked: Set[str] = set((config or {}).get("blocked_commands") or [])
        
        # Filesystem sandboxing
        if config and config.get("base_dir"):
            self.working_dir = Path(config["base_dir"]).resolve()
            self.sandbox_enabled = True
        else:
            self.working_dir = Path.cwd()
            self.sandbox_enabled = False

        # OS-LEVEL ENFORCEMENT, when the agent file declared `sandbox:`
        # (2026-09-23, S-217; the engine is srt since 2026-09-26, see
        # `webagents/sandbox/srt.py`).
        #
        # The allow/deny lists above stay, and they are NOT the boundary: they
        # decide what runs without asking. They cannot be a boundary, because
        # they inspect the text the model proposed while `shell=True` runs
        # something else. Measured: `echo $(id -un)` passes an argv[0]
        # allow-list and the shell executes `id`.
        #
        # `self.policy` is what actually holds. ON BY DEFAULT (module
        # comment): no block means the defaults, not "no sandbox".
        # `--no-sandbox` (the environment the CLI's root option sets) wins
        # over the file for this run; it is the owner's explicit choice.
        from webagents.sandbox import OFF_PRESET, no_sandbox_requested, policy_from_metadata

        self.policy = None
        self._sandbox_error = None
        #: The chat's ask-on-first-use hooks (`HostAsker` in the TypeScript
        #: shell): an object with `ask_host(host, command) -> "once" |
        #: "always" | "no"` and `allow_host_always(host)`, both awaitable.
        #: Set by the chat alone.
        self.asker = (config or {}).get("asker")
        declared = (config or {}).get("sandbox")
        raw: Any = {} if declared is None else declared
        self.sandbox_origin = "default" if declared is None else "agent file"
        if no_sandbox_requested((config or {}).get("env")):
            raw = OFF_PRESET
            self.sandbox_origin = "--no-sandbox"
        try:
            self.policy = policy_from_metadata(raw, cwd=str(self.working_dir))
        except ValueError as error:
            # A malformed declaration must not silently become "no
            # sandbox". Remembered and refused at execution time, where
            # there is somewhere to report it.
            self._sandbox_error = str(error)
        if self.policy is not None and not self.policy.confined:
            # An opt-out that nobody sees is S-217 again. Said once, where
            # the agent loads (`opt_out_announcer`).
            _OPT_OUT_ANNOUNCER[0](NO_SANDBOX_WARNING if self.sandbox_origin == "--no-sandbox" else UNRESTRICTED_WARNING, self.sandbox_origin)

    @property
    def sandbox_error(self) -> Optional[str]:
        """Why the `sandbox:` declaration did not resolve, when it did not (the
        TypeScript shell's `sandboxError`): the chat's `/sandbox` and `/status`
        say `Invalid: ...` from it (D6, 2026-09-26). None when it resolved."""
        return self._sandbox_error

    def declared_in_agent_file(self) -> None:
        """The chat wrote a `sandbox:` block into the agent file for the
        policy in use (an `always` answer, the ptypass-fixes lane,
        2026-09-27): the defaults are now the file's own, so the state reads
        `(agent file)` at once rather than after /reload. An opt-out stays
        what it was. The TypeScript `declaredInAgentFile` is the twin."""
        if self.sandbox_origin == "default":
            self.sandbox_origin = "agent file"

    def sandbox_state_line(self) -> str:
        """The state as the status row prints it: `development (default)`,
        `off (agent file)`, `off (--no-sandbox)` (fixture `status`)."""
        from webagents.sandbox import sandbox_state

        return sandbox_state(self.policy, self.sandbox_origin)

    def _caller_is_owner(self) -> bool:
        """Whether the command was asked for by the agent's owner (or an
        admin). Outside any run there is no caller: that is code calling the
        skill directly, which is the process itself."""
        context = self.get_context()
        if context is None:
            return True
        from webagents.agents.core.scopes import scope_allows

        return scope_allows("owner", context.auth_scopes)

    def _docker_sandbox(self):
        """The Docker `sandbox` skill beside this one, if the agent has it."""
        if not self.agent:
            return None
        for skill in self.agent.skills.values():
            if skill.__class__.__name__ == "SandboxSkill":
                return skill
        return None
    
    def _load_allowed_commands(self) -> Set[str]:
        """The names an unconfined command may start with: the defaults and the file's `allowed_commands`."""
        allowed = set(DEFAULT_ALLOWED)
        if self.config and "allowed_commands" in self.config:
            allowed.update(self.config["allowed_commands"])
        return allowed

    def _load_blocked_commands(self) -> Set[str]:
        """The names refused: the defaults (unconfined only) and the file's `blocked_commands` (always)."""
        blocked = set(DEFAULT_BLOCKED)
        if self.config and "blocked_commands" in self.config:
            blocked.update(self.config["blocked_commands"])
        return blocked

    def _check_command(self, command: str, confined: bool = False) -> Tuple[bool, str]:
        """Whether `command` may run, before it is handed to the runner.

        Confined (the module comment above `DEFAULT_ALLOWED`): only a name
        the agent file blocks is refused; the kernel decides the rest.
        Unconfined: the one list, then the path checks below."""
        try:
            # Use shlex to parse command line correctly handling quotes
            cmd_parts = shlex.split(command)
        except ValueError as e:
            return False, f"Invalid command format: {e}"

        if not cmd_parts:
            return False, "Empty command"

        # The first word is the command, and so is the first after each
        # chaining operator: a simple heuristic, not a shell parser.
        command_separators = {';', '&&', '||', '|'}

        current_cmd_start = True

        for token in cmd_parts:
            if token in command_separators:
                current_cmd_start = True
                continue

            if current_cmd_start:
                name = os.path.basename(token)
                if confined:
                    if name in self.declared_blocked:
                        return False, IS_BLOCKED.format(name=name)
                elif name in self.blocked_commands:
                    return False, IS_BLOCKED.format(name=name)
                elif name not in self.allowed_commands:
                    return False, NOT_ALLOWED.format(name=name)
                current_cmd_start = False
            elif confined:
                continue
            else:
                # This token is an argument
                if self.sandbox_enabled:
                    # Sandbox check 1: No absolute paths (outside sandbox)
                    # Exception: legitimate flags/options might start with /? unlikely but possible.
                    # Usually / implies path.
                    if token.startswith("/") and Path(token).exists():
                         # Check if it resolves inside sandbox
                         try:
                             p = Path(token).resolve()
                             if not (self.working_dir in p.parents or p == self.working_dir):
                                 return False, f"Access denied: Absolute path outside sandbox: {token}"
                         except Exception:
                             pass # If not a valid path, maybe just a weird string
                             
                    # Sandbox check 2: No parent directory traversal
                    if ".." in token:
                         # This blocks ".." even in strings like "version..1", which is a trade-off.
                         # Tighter check: match path components
                         parts = Path(token).parts
                         if ".." in parts:
                             return False, f"Access denied: Parent directory traversal ('..') not allowed in sandbox: {token}"
                    
                    # Sandbox check 3: No home directory expansion
                    if token.startswith("~"):
                        return False, f"Access denied: Home directory expansion ('~') not allowed in sandbox: {token}"

        return True, ""
    
    # The TypeScript tool's sentence (`skills/shell/skill.ts` `runCommand`),
    # pinned by the shared fixture `sandbox/srt.json` (`shell_tool`): the
    # model reads the same description under either SDK (2026-09-27).
    @tool(scope="owner", description="Run a shell command in the working folder")
    async def run_command(self, command: str, timeout: int = 30) -> str:
        """Run a shell command in the working folder

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (default: 30)

        Returns:
            Command output or error message
        """
        if self._sandbox_error:
            # Declared and unparseable. Refusing is the only honest answer:
            # running unsandboxed is precisely what the declaration forbade.
            return f"Access denied: invalid sandbox declaration: {self._sandbox_error}"

        # DEFENSE IN DEPTH behind the owner-only default (S-248, 2026-09-26):
        # a command asked for by anyone but the owner runs ONLY confined. An
        # agent file that hands `shell` to a group with `access: tools:` and
        # declares no sandbox, or declares `unrestricted`, or runs where srt
        # is missing, refuses that group rather than running as the owner.
        if not self._caller_is_owner():
            from webagents.sandbox import sandbox_required_reason

            reason = sandbox_required_reason(self.policy)
            if reason is not None:
                return f"Access denied: commands from callers other than the owner run only in a sandbox, and {reason}"

        # The Docker `sandbox` skill, when the agent has it and the file
        # declares no kernel policy. A declared `sandbox:` wins over it: the
        # declaration is the explicit one, and the container ignores it.
        if self.sandbox_origin == "default":
            docker = self._docker_sandbox()
            if docker is not None:
                return await docker.run_sandbox_command(command)

        allowed, reason = self._check_command(command, confined=bool(self.policy is not None and self.policy.confined))
        if not allowed:
            return f"Access denied: {reason}"

        try:
            if self.policy is not None:
                from webagents.sandbox import SandboxUnavailable, run_interruptibly, run_sandboxed

                # Ask on first use, in the interactive chat only, for the
                # owner only: a refused host is read from srt's own proxy log
                # and asked about by name; `once` re-runs with it for this
                # command, `always` writes it into the agent file first.
                # Every other mode returns the hint.
                asking = self.asker is not None and self.policy.confined and self._caller_is_owner()
                policy = self.policy
                while True:
                    try:
                        # In a worker thread that the turn's Esc or Ctrl+C stops,
                        # process group and all (`run_interruptibly`).
                        result = await run_interruptibly(run_sandboxed, command, policy, timeout=timeout, capture_refusals=asking)
                    except SandboxUnavailable as error:
                        # FAIL CLOSED. The file said sandboxed, or the
                        # defaults do; this machine cannot enforce it; so the
                        # command does not run. Claude Code's default is to
                        # warn and continue, which is a reasonable UX choice
                        # for an interactive tool and the wrong one for a
                        # declared restriction.
                        return f"Access denied: {error}"
                    refused = [host for host in (getattr(result, "refused_hosts", None) or []) if host not in policy.network_domains]
                    if not asking or not refused:
                        break
                    allowed_hosts: List[str] = []
                    for host in refused:
                        answer = await self.asker.ask_host(host, command)
                        if answer == "no":
                            continue
                        if answer == "always":
                            await self.asker.allow_host_always(host)
                        allowed_hosts.append(host)
                    if not allowed_hosts:
                        break
                    import dataclasses

                    policy = dataclasses.replace(policy, network=True, network_domains=[*policy.network_domains, *allowed_hosts])
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.working_dir,
                    # Never the owner's terminal (S-317).
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                policy = None

            output = result.stdout or ""
            if result.stderr:
                output += f"\nStderr: {result.stderr}"

            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"

            if policy is not None and policy.confined:
                from webagents.sandbox import refusal_hint

                hint = refusal_hint(command, f"{result.stdout or ''}\n{result.stderr or ''}")
                if hint:
                    output += f"\n{hint}"

            return output if output else "(No output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except CommandInterrupted:
            return INTERRUPTED_RESULT
        except Exception as e:
            return f"Error executing command: {e}"
