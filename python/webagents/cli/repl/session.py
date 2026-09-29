"""
The chat (2026-09-24): one inline chat that runs the agent in its own process,
the same as the TypeScript one (`typescript/src/cli/app.ts`).

WHY ONE CHAT, IN PROCESS. The Python CLI had two chats, a full-screen Textual
one and this line-by-line one, and both reached the agent through the daemon:
auto-started, polled for up to five seconds, and left running with whatever
code it started with. The TypeScript chat builds the agent itself. So the two
SDKs looked and behaved differently, and a command like `/model` or `/login`
had to go through a server to touch the agent. Now both chats build the agent
in-process (`cli/agent_builder.py` here), take the same commands
(`repl/commands.py`, word for word the TypeScript list), keep conversations in
the same files (`cli/sessions.py`), and draw the same box, card, notices and
turns.
"""

from __future__ import annotations

import asyncio
import getpass
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text
from rich.theme import Theme as RichTheme

from webagents.agents.skills.core.llm.pricing import NO_COST, RunningCost, add_turn_cost, cost_words

from .. import checkpoints
from ..config_store import cli_command
from ..robutler_sessions import (
    ConversationsTarget,
    linked_platform_agent,
    list_platform_conversations,
    merge_conversations,
    read_platform_conversation,
    record_words,
    unavailable_reason,
    words_since,
)
from ..sessions import list_sessions, load_session, mark_recorded, new_session_id, save_session, sessions_dir, when_label
from ..turn_history import spoken_count
from ..ui.banner import WelcomeInfo, play_wordmark, welcome_card
from ..ui.prompt_box import PromptBox, sent_message
from ..ui.screen import record_screen
from ..ui.terminal import query_background, query_cursor_row, stream_keys
from ..ui.theme import markdown_styles, theme_for
from . import agent_reload
from .agent_reload import LoadedAgent
from .chat_words import CHAT_WORDS, fill, plural
from .commands import (
    CHAT_COMMANDS,
    CHAT_HISTORY,
    MOVED_COMMANDS,
    chat_command,
    help_lines,
    notice,
    takes_no_arguments,
)
from .failures import FailureText

#: The content tools the TypeScript agent registers for one turn (`present`,
#: `read_content`, `save_content`) and that its chat leaves out of `/tools`;
#: named here too so the two lists are filtered alike.
TRANSIENT_TOOLS = frozenset({"present", "read_content", "save_content"})


def chat_history_file(profile: Optional[str] = None) -> Path:
    """The profile's typed-line history file (owner decision D1, S-291): under
    `global_dir()`, so `--profile x` keeps its own."""
    from ..config_store import global_dir

    return global_dir(profile) / CHAT_HISTORY["file"]


def secure_chat_history(file: Path) -> bool:
    """The folder at 0700 and the file at 0600, whether they exist already or
    not (S-291: the file used to take the umask, 0644 in a 0755 folder, so
    another account on the machine could read every line typed). Never raises:
    a home folder that cannot be written leaves the chat without history
    rather than without a chat."""
    try:
        file.parent.mkdir(parents=True, exist_ok=True, mode=CHAT_HISTORY["dir_mode"])
        os.chmod(file.parent, CHAT_HISTORY["dir_mode"])
        # Opened for append, so an existing file keeps its entries; created owner-only.
        os.close(os.open(str(file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, CHAT_HISTORY["file_mode"]))
        os.chmod(file, CHAT_HISTORY["file_mode"])
        return True
    except OSError:
        return False


def scope_label(scope: Any) -> str:
    """Who may use a tool, from its scope (`/tools`, spec 3.7, the same words
    as the TypeScript chat): `only you` for one scoped to the owner or admin,
    `you and {groups}` for one an `access.tools` grant named, `every caller`
    for one open to all (or with no scope at all)."""
    scopes = [str(s) for s in (scope if isinstance(scope, (list, tuple, set)) else ([scope] if scope else []))]
    groups = [s[len("group:"):] for s in scopes if s.startswith("group:")]
    if groups:
        return fill("scopeYouAnd", groups=", ".join(groups))
    if not scopes or "all" in scopes or "none" in scopes:
        return CHAT_WORDS["scopeEveryCaller"]
    if all(s in ("owner", "admin") for s in scopes):
        return CHAT_WORDS["scopeOnlyYou"]
    return ", ".join(scopes)


def _pattern_text(pattern: Any) -> str:
    """A pattern as the file wrote it, for `/access`."""
    return f"agent:*.{pattern.value}" if pattern.kind == "domain" else f"{pattern.kind}:{pattern.value}"


def _group_text(rule: Any) -> str:
    """A group's members and trust, in one phrase (`/access`)."""
    trust = ""
    if rule.trust is not None:
        minimum = rule.trust.min
        shown = str(int(minimum)) if float(minimum).is_integer() else str(minimum)
        trust = fill("accessTrust", min=shown, topic=rule.trust.topic) if rule.trust.topic else fill("accessTrustOverall", min=shown)
    if rule.members is None:
        return trust
    members = ", ".join(_pattern_text(p) for p in rule.members) if rule.members else CHAT_WORDS["accessNobody"]
    return f"{members}, {trust}" if trust else members

#: The embedded agent's name, offered by /agent in every folder.
BUILT_IN_AGENT = "robutler"

#: "Find the agent file the usual way" as opposed to None, the built-in agent.
_UNSET: Any = object()


def skillmd_origin_line(name: str, entry: Any) -> str:
    """One installed SKILL.md skill's line in `/skills` (2026-09-29): from the
    lock's entry, its source and short commit (`skillmdFrom`), or its local
    folder when the install had no commit (`skillmdFromLocal`); a folder the
    lock does not know is the person's own (`skillmdIn`). The TypeScript
    chat's `skillmdOriginLine` is the twin; `cli/chat_edits.json` pins both."""
    source = entry.get("source") if isinstance(entry, dict) else None
    commit = entry.get("commit") if isinstance(entry, dict) else None
    if isinstance(source, str) and source:
        if isinstance(commit, str) and commit:
            return fill("skillmdFrom", skill=name, source=source, commit=commit[:7])
        return fill("skillmdFromLocal", skill=name, source=source)
    return fill("skillmdIn", skill=name, folder=".agents/skills/" + name)


def _saved_cost(metadata: Any) -> RunningCost:
    """The conversation's saved cost (`save_conversation`), for /resume; nothing when the file has none."""
    credits = metadata.get("cost_credits") if isinstance(metadata, dict) else None
    if isinstance(credits, bool) or not isinstance(credits, (int, float)):
        return NO_COST
    return RunningCost(float(credits), metadata.get("cost_estimated") is True, True)


def _short_path(path: Path) -> str:
    text = str(path)
    home = str(Path.home())
    if text == home or text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def _control_path(file: str, folder: Optional[Path] = None) -> str:
    """A control file's path as the person reads it in the chat's question
    (the ptypass-fixes lane, 2026-09-27): relative to this folder when it is
    inside it (`AGENT.md`), else under `~`, else as given. The file tools name
    a file as the model did, often absolutely, and the question wrapped such a
    path mid-name across three lines. The TypeScript `controlPath` is the
    twin."""
    base = Path(folder or Path.cwd())
    absolute = Path(os.path.normpath(os.path.join(str(base), file)))
    relative = os.path.relpath(str(absolute), str(base))
    if relative != "." and relative != ".." and not relative.startswith(".." + os.sep):
        return relative
    return _short_path(absolute)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _truncate_start(text: str, width: int) -> str:
    return text if len(text) <= width else "…" + text[-(width - 1):]


class _HostAsker:
    """The chat's ask-on-first-use hooks for the shell (`_attach_host_asker`):
    the question at the terminal with the turn's live display taken down, as
    `_confirm_control_write` does, and the agent-file write for `always`."""

    def __init__(self, session: "WebAgentsSession", shell: Any) -> None:
        self.session = session
        self.shell = shell

    async def ask_host(self, host: str, command: str) -> str:
        from ..sandbox_default_hosts import HOST_WORDS, fill_words, host_answer

        s = self.session
        if not s._chatting or not s._at_terminal():
            return "no"
        with s._turn_paused():
            p = s.theme.palette
            s.console.print()
            s.console.print(Text(f"  {fill_words(HOST_WORDS['hostRefused'], host=host, command=command)}", style=p.warning))
            answer = host_answer(await s._ask(f"  {HOST_WORDS['hostQuestion']}"))
            s._answer_line("host", answer, host=host)
            return answer

    async def allow_host_always(self, host: str) -> None:
        from ..sandbox_default_hosts import HOST_WORDS, add_network_host, fill_words

        s = self.session
        file = s.built.file if s.built else None
        shown = file.name if file is not None else "the built-in agent"
        if file is None:
            s.notice("warn", fill_words(HOST_WORDS["hostNotWritten"], host=host, file=shown, problem=HOST_WORDS["hostNoFile"]))
            return
        text, problem = add_network_host(Path(file).read_text(), host)
        if text is None:
            s.notice("warn", fill_words(HOST_WORDS["hostNotWritten"], host=host, file=shown, problem=problem or ""))
            return
        # Whether the file was still the version the chat loaded, before its own write.
        untouched = s._file_changed_now() is None
        Path(file).write_text(text)
        # This session keeps the host too; the file is read again on the next start or /reload.
        policy = getattr(self.shell, "policy", None)
        if policy is not None and host not in policy.network_domains:
            policy.network_domains.append(host)
            policy.network = True
        # THE CHAT'S OWN WRITE IS NOT A CHANGE (the ptypass-fixes lane,
        # 2026-09-27): the edit above made the next prompt say "AGENT.md
        # changed during the last reply", and `/sandbox` said "(default)"
        # until /reload, though this session already runs what the file now
        # says. When nothing else had changed the file, the written version
        # is the loaded one; the policy is now the file's own block.
        if untouched:
            written = s._file_changed_now()
            if written is not None:
                s._loaded = written
                s._version_noticed = written.sha
        declared = getattr(self.shell, "declared_in_agent_file", None)
        if callable(declared):
            declared()
        s.notice("ok", fill_words(HOST_WORDS["hostWritten"], host=host, file=shown))


class WebAgentsSession:
    """One chat with one agent at a time, in this process."""

    def __init__(
        self,
        agent_path: Optional[Path] = None,
        model: Optional[str] = None,
        streaming: bool = True,
        chosen: bool = False,
        interactive: Optional[bool] = None,
    ) -> None:
        self.console = Console()
        self.theme = theme_for(self.console)
        self.console.push_theme(RichTheme(markdown_styles(self.theme)))

        #: The agent file given on the command line; None finds it (AGENT.md, then the built-in agent).
        self.agent_path = agent_path
        #: /agent's choice (or `-a`'s, when `chosen`): a path, None for the built-in agent, or unset.
        self.selected_file: Any = agent_path if chosen else _UNSET
        #: `--no-streaming` shows each reply whole, when it is done.
        self.streaming = streaming
        #: A model the person chose (`-m` or /model), ahead of the file's.
        self.explicit_model = model
        #: `/rounds <n>`: this chat's tool-round budget, above the flag and the
        #: file (2026-09-28, `agents/core/tool_budget.py`); kept across rebuilds.
        self.session_rounds: Optional[int] = None
        #: `agent_builder.BuiltAgent`, set by `initialize()`.
        self.built: Any = None

        self.messages: List[Dict[str, Any]] = []
        self.session_id = new_session_id()
        self.session_created_at = ""
        #: The platform chat this conversation is recorded into, once it is
        #: (`session: {backend: robutler}`, `robutler_sessions.py`), and how
        #: many of `messages` are there.
        self.platform_chat_id: Optional[str] = None
        self.recorded_count = 0
        #: Recording runs behind the conversation, one turn after another.
        self._recording: Optional["asyncio.Future[None]"] = None
        #: Said once per conversation, before the next prompt: why it is not on Robutler.
        self._recording_problem: Optional[str] = None
        self._recording_noticed = False
        #: The snapshot taken before each message of this conversation, oldest
        #: first (`checkpoints.py`), for /undo. Taken only when the agent can
        #: change files: it has `filesystem` or `shell`.
        self.turn_snapshots: List[str] = []
        self._snapshots_noticed = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.turns = 0
        self.session_started = time.time()
        self.running = True
        #: `run()` has started the interactive loop (S-314, 2026-09-27). Only
        #: then may a file-tool write to a control file ask the person at the
        #: terminal; `-p` never builds this object, and a test that calls
        #: `initialize()` alone gets the refusal, as `serve` and the daemon do.
        self._chatting = False
        #: The turn's live display and its key listener, so a question asked
        #: mid-turn (the control-file prompt) can take them down and put them back.
        self._turn_live: Any = None
        self._turn_keys: Any = None
        self._turn_reenter_keys: Optional[Callable[[], Any]] = None
        #: The turn's SIGINT handler while a turn runs: `_ask` puts it back
        #: after a mid-turn question instead of leaving none.
        self._turn_interrupt: Optional[Callable[[], None]] = None
        #: The turn's renderer while a turn runs, for `_turn_paused` to take
        #: the time spent answering out of the running tools' timers.
        self._turn_renderer: Any = None

        #: Whether the chat may ask before it changes a file (spec W2): stdin
        #: and stdout are terminals. A test sets it; a pipe leaves it false.
        self.interactive = interactive if interactive is not None else (sys.stdin.isatty() and sys.stdout.isatty())
        #: The loaded version's fingerprint (`agent_reload.py`), for /reload and
        #: the before-prompt notice; None for the built-in agent.
        self._loaded: Optional[LoadedAgent] = None
        #: The running agent's tool names, sorted, for the reload diff.
        self._tool_names: List[str] = []
        #: The version whose "changed" notice was shown, said once per version.
        self._version_noticed: Optional[str] = None
        #: A change seen during the last reply: `/reload shows what changed`.
        self._changed_during_reply = False

        #: The tokens THIS chat has spent, for the goodbye line; the
        #: conversation's own totals (`input_tokens`, `output_tokens`) count a
        #: resumed conversation's earlier turns too, which the goodbye line
        #: used to repeat as if they were this chat's (2026-09-26).
        self.session_tokens = 0
        #: The conversation's cost in credits (plan item 2.4, `llm/pricing.py`):
        #: what Robutler's models reported, or an estimate from the list prices
        #: for a provider key. `session_cost` is what THIS chat spent, for the
        #: goodbye line, as `session_tokens` is for tokens.
        self.cost = NO_COST
        self.session_cost = NO_COST
        #: The owner's note keys, read before each prompt for `/memory forget` completion.
        self._memory_keys: List[str] = []
        #: This agent's schedule names, read before each prompt for `/cron run` completion.
        self._schedule_names: List[str] = []

        # The typed-line history, per profile and owner-only (S-291). A file
        # that cannot be made stays in memory for this chat.
        history_file = chat_history_file()
        history = FileHistory(str(history_file)) if secure_chat_history(history_file) else None
        self.prompt_box = PromptBox(
            self.theme,
            commands=[(f"/{c.name}", c.description) for c in CHAT_COMMANDS],
            footer=self._footer_parts,
            history=history,
            completers=self._completers(),
        )

        self._handlers: Dict[str, Callable[[str], Any]] = {
            "help": self.cmd_help,
            "new": lambda _args: self.start_new_conversation(),
            "clear": lambda _args: self.clear_screen(),
            "resume": self.cmd_resume,
            "undo": lambda _args: self.cmd_undo(),
            "rewind": self.cmd_rewind,
            "reload": lambda _args: self.cmd_reload(),
            "model": self.cmd_model,
            "rounds": self.cmd_rounds,
            "agent": self.cmd_agent,
            "skills": self.cmd_skills,
            "tools": lambda _args: self.cmd_tools(),
            "mcp": lambda _args: self.cmd_mcp(),
            "access": lambda _args: self.cmd_access(),
            "cron": self.cmd_cron,
            "memory": self.cmd_memory,
            "status": lambda _args: self.cmd_status(),
            "login": lambda _args: self.sign_in(),
            "logout": lambda _args: self.cmd_logout(),
            "keys": self.cmd_keys,
            "secrets": self.cmd_secrets,
            "sandbox": lambda _args: self.cmd_sandbox(),
            "publish": self.cmd_publish,
            "exit": lambda _args: self._stop(),
        }
        missing = [c.name for c in CHAT_COMMANDS if c.name not in self._handlers]
        if missing:
            raise RuntimeError(f"No handler for /{missing[0]}")

    # -- the agent ------------------------------------------------------------

    def _resolve_agent_file(self) -> Optional[Path]:
        if self.selected_file is not _UNSET:
            return self.selected_file
        if self.agent_path is not None:
            return self.agent_path
        from ..agent_files import default_agent_file

        return default_agent_file(Path.cwd())

    def _at_terminal(self) -> bool:
        """Whether the chat may ask before it changes a file (spec W2)."""
        return self.interactive

    async def initialize(self) -> None:
        """Build the agent the way the daemon would (`cli/agent_builder.py`),
        then swap it in and clean up the one it replaces (spec 3.1). A build
        that raises (a bad file) leaves the running agent untouched, so
        /reload, /model and the rest keep it."""
        from ..agent_builder import build_agent
        from ..credentials import get_token

        # `get_token`, read per search: `discovery` searches as the person at
        # this terminal when the agent has no platform credential of its own.
        # The file tools ask here, and nowhere else, before they change one
        # of the agent's control files (S-314).
        # The opt-out, in the chat, is one of its notices, the words
        # `/sandbox` prints (`opt_out_announcer`; the ptypass-fixes lane,
        # 2026-09-27): the shell's raw stderr line wrapped mid-word above the
        # welcome card and said something else than `/sandbox`.
        from contextlib import nullcontext

        from webagents.agents.skills.local.shell.skill import opt_out_announcer

        opted_out: List[str] = []
        with opt_out_announcer(lambda _message, origin: opted_out.append(origin)) if self._chatting else nullcontext():
            new = await build_agent(
                self._resolve_agent_file(),
                working_dir=Path.cwd(),
                model=self.explicit_model,
                person_token=get_token,
                confirm_control_write=self._confirm_control_write,
            )
        previous = self.built
        self.built = new
        if self.session_rounds is not None and getattr(new, "agent", None) is not None:
            new.agent.max_tool_iterations = self.session_rounds
            new.agent.max_tool_rounds_source = "session"
        self._loaded = self._loaded_of(new)
        self._tool_names = sorted(name for name, _ in self.tool_list())
        self._version_noticed = self._loaded.sha if self._loaded else None
        self._changed_during_reply = False
        # Only the interactive loop asks (`run()` attaches for the first
        # agent); a `/reload` while chatting attaches to the new one here.
        # `-p` never has it.
        if self._chatting:
            self._attach_host_asker()
        if previous is not None and previous is not new:
            await self._cleanup_agent(previous)
        if opted_out:
            # `/sandbox`'s own notice: the state, and the switch that confines.
            kind, headline, detail = self.sandbox_summary()
            self.notice(kind, f"Sandbox: {headline}", detail)

    def _loaded_of(self, built: Any) -> Optional[LoadedAgent]:
        """The fingerprint of the version `built` was built from; None for the built-in agent."""
        file = getattr(built, "file", None)
        if file is None:
            return None
        try:
            from ..loader.hierarchy import load_agent

            merged = load_agent(file)
            return agent_reload.loaded_agent_of(str(file), str(Path(file).parent), merged.metadata, merged.instructions)
        except Exception:  # noqa: BLE001 - a fingerprint we cannot take is no worse than none
            return None

    async def _cleanup_agent(self, built: Any) -> None:
        """Close the previous agent's skills (its MCP connections), which every /model, /login and /keys left open."""
        skills = getattr(getattr(built, "agent", None), "skills", None) or {}
        for skill in list(skills.values()) if isinstance(skills, dict) else []:
            cleanup = getattr(skill, "cleanup", None)
            if cleanup is None:
                continue
            try:
                result = cleanup()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - a cleanup that fails must not fail the reload
                pass

    @property
    def agent_name(self) -> str:
        return self.built.name if self.built else "agent"

    @property
    def model_problem(self) -> Optional[str]:
        return self.built.model_problem if self.built else None

    def model_label(self) -> str:
        return self.built.model_label if self.built else ""

    def agent_folder(self) -> Path:
        return self.built.file.parent if self.built and self.built.file else Path.cwd()

    def tool_list(self) -> List[Tuple[str, str]]:
        return [(name, description) for name, description, _scope in self.tool_rows()]

    def tool_rows(self) -> List[Tuple[str, str, Any]]:
        """`(name, first line of the description, scope)` per tool, by name."""
        if not self.built:
            return []
        out = []
        for tool in self.built.agent.get_all_tools():
            name = str(tool.get("name"))
            if name in TRANSIENT_TOOLS:
                continue
            description = (tool.get("description") or "").strip().split("\n")[0]
            out.append((name, description, tool.get("scope")))
        # By name, as the TypeScript chat lists them: registration order is
        # each SDK's own business, and the list reads the same in both.
        return sorted(out, key=lambda row: row[0])

    # -- drawing --------------------------------------------------------------

    def notice(self, kind: str, text: str, detail: Optional[str] = None) -> None:
        notice(self.console, self.theme, kind, text, detail)

    def welcome_info(self) -> WelcomeInfo:
        from webagents import __version__

        return WelcomeInfo(
            agent=self.agent_name,
            description=self.built.description if self.built else "",
            model=self.model_label(),
            tools=[name for name, _ in self.tool_list()],
            folder=_short_path(self.agent_folder()),
            warnings=[self.model_problem] if self.model_problem else [],
            version=__version__,
        )

    def print_card(self) -> None:
        for line in welcome_card(self.theme, self.console.width, self.welcome_info()):
            self.console.print(line)
        self.console.print()

    def _footer_parts(self) -> List[str]:
        """Under the box: who, which model, what it has cost so far, where."""
        from .render import compact_number

        parts = [self.agent_name]
        model = self.model_label()
        if model:
            parts.append(model)
        # The conversation's tokens (a resumed one's included), as /status counts them.
        tokens = self.input_tokens + self.output_tokens
        if tokens:
            parts.append(f"{compact_number(tokens)} tokens")
        # What it has cost, in credits, next to the tokens (plan item 2.4): the
        # platform's number for Robutler's models, an estimate (tilde) when it
        # reported none; nothing for a key's turn, which costs Robutler nothing.
        if self.cost.known:
            parts.append(cost_words(self.cost.credits, self.cost.estimated))
        parts.append(_truncate_start(_short_path(Path.cwd()), 28))
        return parts

    def cost_model(self) -> Optional[str]:
        """The model a turn's cost is estimated for: the one that answered (a failover may have moved it), else the agent's."""
        failover = self.failover_skill()
        answered = getattr(failover, "answered_model", None) if failover is not None else None
        if answered:
            return answered
        access = self.built.access if self.built else None
        return getattr(access, "model", None) if access is not None else None

    def turn_ran_through_robutler(self) -> bool:
        """Whether the last turn's answering model ran through Robutler: the
        failover's answering member when it says so, else the agent's access."""
        failover = self.failover_skill()
        via = getattr(failover, "answered_via_robutler", None) if failover is not None else None
        if via is not None:
            return bool(via)
        access = self.built.access if self.built else None
        return getattr(access, "kind", None) == "proxy"

    def turn_cost_model(self) -> Optional[str]:
        """The model a turn's cost is ESTIMATED for, or None when nothing is
        (2026-09-29): only a turn that ran through Robutler costs credits; the
        platform reports the number, and the estimate stands in when it did
        not. A turn on the person's own provider key costs Robutler nothing,
        and the footer showed `~<0.0001 credits` for it: tokens alone now. The
        TypeScript chat's `turnCostModel` is the twin; `w2ops/cost.json` pins both."""
        return self.cost_model() if self.turn_ran_through_robutler() else None

    def failover_skill(self) -> Any:
        """The agent's failover skill (`fallback_models:`, plan item 2.8), when it has one."""
        skills = getattr(getattr(self.built, "agent", None), "skills", None) if self.built else None
        if not isinstance(skills, dict):
            return None
        return next((s for s in skills.values() if getattr(s, "is_failover", False)), None)

    # -- completion (spec 3.8) ------------------------------------------------

    def _completers(self) -> Dict[str, Callable[[str], List[Tuple[str, str]]]]:
        """What the box offers after `/<command> `: the next word's values, by
        command. Read-only lookups, computed as the menu opens; what needs a
        store (notes, schedules) is read before each prompt
        (`_refresh_completion_data`)."""

        def agents() -> List[Tuple[str, str]]:
            out = [(a.name, a.description) for a in self.folder_agents() if not a.problem]
            out.append((BUILT_IN_AGENT, "The general assistant"))
            return out

        # A word after the last one the command takes closes the menu, so
        # enter then sends the line; a list command (`skills add a b`) keeps
        # offering the names not yet typed, and esc closes its menu.
        def parts(args: str) -> Tuple[str, List[str]]:
            words = args.split()
            return (words[0] if words else ""), words[1:]

        def agent(args: str) -> List[Tuple[str, str]]:
            verb, rest = parts(args)
            if verb == "edit":
                return [] if rest else [a for a in agents() if a[0] != BUILT_IN_AGENT]
            if verb:
                return []
            return agents() + [("new", "make one here"), ("edit", "open its file in your editor")]

        def skills(args: str) -> List[Tuple[str, str]]:
            verb, rest = parts(args)
            if verb == "add":
                from ..agent_builder import SKILL_CLASSES

                return [(name, "") for name in sorted(SKILL_CLASSES) if name not in rest]
            if verb == "remove":
                names = list(self._loaded.skills) + list(self._loaded.skillmd) if self._loaded else []
                return [(name, "") for name in names if name not in rest]
            if verb:
                return []
            return [("list", "every name an agent file can name"), ("add", "give the agent a skill"), ("remove", "take one away")]

        def help_(args: str) -> List[Tuple[str, str]]:
            return [] if args.split() else [(c.name, c.description) for c in CHAT_COMMANDS]

        def keys(args: str) -> List[Tuple[str, str]]:
            verb, rest = parts(args)
            if verb in ("set", "unset"):
                if rest:
                    return []
                from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

                return [(p.env_vars[0], p.id) for p in LLM_PROVIDERS if p.credential == "api_key" and p.env_vars]
            if verb:
                return []
            return [("set", "store a key"), ("unset", "remove a stored key")]

        def cron(args: str) -> List[Tuple[str, str]]:
            verb, rest = parts(args)
            if verb == "run":
                return [] if rest else [(name, "") for name in self._schedule_names]
            if verb:
                return []
            return [("run", "run a schedule now")]

        def memory(args: str) -> List[Tuple[str, str]]:
            verb, rest = parts(args)
            if verb == "forget":
                return [] if rest else [(key, "") for key in self._memory_keys]
            if verb:
                return []
            return [("forget", "remove one of your notes")]

        return {"agent": agent, "skills": skills, "help": help_, "keys": keys, "cron": cron, "memory": memory}

    async def _refresh_completion_data(self) -> None:
        """What the completers need that is async: read before the box opens, quietly."""
        try:
            skill = self._memory_skill()
            self._memory_keys = [n["key"] for n in (await skill.owner_summary())["recent"]] if skill is not None else []
            self._schedule_names = []
            if self.built is not None and self.built.file is not None and self._loaded is not None and self._loaded.cron:
                from ..cron_command import folder_schedules

                found = next((f for f in folder_schedules(self.agent_folder(), lambda _line: None) if f.name == self.agent_name), None)
                self._schedule_names = [s.name for s in found.schedules] if found is not None else []
        except Exception:  # noqa: BLE001 - completion is a convenience
            pass

    def _print_lines(self, lines: List[Text]) -> None:
        self.console.print()
        for line in lines:
            self.console.print(line)
        self.console.print()

    # -- conversations --------------------------------------------------------

    def start_new_conversation(self, say: bool = True) -> None:
        self.messages = []
        self.session_id = new_session_id()
        self.session_created_at = ""
        self.platform_chat_id = None
        self.recorded_count = 0
        self._recording_noticed = False
        self.turn_snapshots = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = NO_COST
        if say:
            self.notice("ok", "Started a new conversation.")

    def clear_screen(self) -> None:
        self.start_new_conversation(False)
        if self.console.is_terminal:
            # Screen and scrollback, then the card: what a fresh start looks like.
            self.console.file.write("\x1b[2J\x1b[3J\x1b[H")
            self.console.file.flush()
            self.print_card()
        else:
            self.console.clear()

    def session_dir(self) -> Path:
        return sessions_dir(self.agent_folder(), self.agent_name)

    def save_conversation(self) -> None:
        """Saved after every turn, so /resume finds it after a crash as well as after /exit."""
        if not self.messages:
            return
        try:
            save_session(
                self.session_dir(),
                {
                    "session_id": self.session_id,
                    "agent_name": self.agent_name,
                    "created_at": self.session_created_at,
                    "updated_at": "",
                    "messages": self.messages,
                    "metadata": {
                        "model": self.model_label(),
                        "sdk": "python",
                        **(
                            {"robutler_chat_id": self.platform_chat_id, "robutler_recorded": self.recorded_count}
                            if self.platform_chat_id
                            else {}
                        ),
                        # The conversation's cost, so /resume shows it again (plan item 2.4).
                        **({"cost_credits": self.cost.credits, "cost_estimated": self.cost.estimated} if self.cost.known else {}),
                    },
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
            )
        except OSError:
            pass  # A conversation that cannot be saved is still a conversation.

    @property
    def session_backend(self) -> str:
        """Where conversations are kept: `local`, or on Robutler too (the session skill's `backend`)."""
        return getattr(self.built, "session_backend", "local") if self.built else "local"

    def conversations_target(self) -> Any:
        """Where this agent's conversations are kept on Robutler, as whom, or
        why they cannot be (`robutler_sessions.py`): the person's sign-in, and
        the folder's link to the agent `webagents publish` made."""
        from ..config_store import resolve_platform_url
        from ..credentials import get_token

        token = get_token()
        if not token:
            return "signed_out"
        agent_id = linked_platform_agent(self.agent_folder(), self.agent_name)
        if not agent_id:
            return "not_published"
        return ConversationsTarget(base=resolve_platform_url()[0], token=token, agent_id=agent_id)

    def record_on_robutler(self) -> None:
        """After a turn: record what the conversation added into the person's
        chat with the agent on Robutler, behind the conversation (the next
        prompt does not wait). A problem is said once, before the next prompt."""
        if self.session_backend != "robutler":
            return
        words = words_since(self.messages, self.recorded_count)
        if not words:
            return
        session_id, directory, up_to, chat_id = self.session_id, self.session_dir(), len(self.messages), self.platform_chat_id
        previous = self._recording

        async def record() -> None:
            if previous is not None:
                try:
                    await previous
                except Exception:  # noqa: BLE001 - its problem was said already
                    pass
            target = self.conversations_target()
            if isinstance(target, str):
                self._recording_problem = unavailable_reason(target, cli_command)
                return
            try:
                recorded_into = await record_words(target, session_id, chat_id, words, cli_command)
            except Exception as error:  # noqa: BLE001 - said, and the conversation goes on
                self._recording_problem = f"This conversation is not being kept on Robutler: {error}."
                return
            if not recorded_into:
                return
            if session_id == self.session_id:
                self.platform_chat_id = recorded_into
                self.recorded_count = up_to
                self.save_conversation()
            else:
                # The person moved on (/new, /resume): note it on the one recorded.
                mark_recorded(directory, session_id, recorded_into, up_to)

        self._recording = asyncio.ensure_future(record())

    def say_recording_problem(self) -> None:
        """Before a prompt: the recording problem, once per conversation."""
        if not self._recording_problem:
            return
        if not self._recording_noticed:
            self.notice("warn", self._recording_problem)
        self._recording_noticed = True
        self._recording_problem = None

    async def settle_recording(self, timeout: float = 10.0) -> None:
        """Wait for the recording behind the conversation (bounded: an
        unreachable platform must not hold the terminal)."""
        if self._recording is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._recording), timeout)
        except Exception:  # noqa: BLE001 - timed out, or failed and said
            pass

    @property
    def can_change_files(self) -> bool:
        """Whether the agent can change the folder: it has `filesystem` or `shell`."""
        names = (getattr(self.built, "skills", None) or []) if self.built else []
        return any(str(name).lower() in ("filesystem", "shell") for name in names)

    def _turn_snapshot_folder(self) -> Optional[Path]:
        """The folder a turn's snapshot is taken of: None when the agent cannot
        change files, or in the home folder or above."""
        if not self.can_change_files:
            return None
        folder = self.agent_folder()
        return None if checkpoints.snapshots_off_reason(folder) else folder

    def _snapshot_failed(self, error: OSError) -> None:
        if not self._snapshots_noticed:
            self.notice("warn", checkpoints.snapshot_failed(str(error)))
        self._snapshots_noticed = True

    def snapshot_before_turn(self, message: str) -> None:
        """Before a message: a snapshot of the folder, for /undo, when the agent
        can change files. Never in the home folder or above, and quiet about
        it until /undo is asked for. A snapshot that cannot be taken is said
        once, and the message goes ahead."""
        folder = self._turn_snapshot_folder()
        if folder is None:
            return
        try:
            self.turn_snapshots.append(checkpoints.take_snapshot(folder, checkpoints.turn_label(message))["id"])
        except OSError as error:
            self._snapshot_failed(error)

    async def snapshot_before_turn_async(self, message: str) -> None:
        """What a turn runs once its spinner is up: the same snapshot, walked in
        a thread so the spinner keeps drawing. Taken before the spinner, a
        large folder froze the chat with no sign of life after Enter
        (2026-09-28). It still finishes before the model is asked anything, so
        no tool can change a file first."""
        folder = self._turn_snapshot_folder()
        if folder is None:
            return
        try:
            manifest = await asyncio.to_thread(checkpoints.take_snapshot, folder, checkpoints.turn_label(message))
        except OSError as error:
            self._snapshot_failed(error)
            return
        self.turn_snapshots.append(manifest["id"])

    async def _confirm(self, question: str) -> bool:
        """A yes to ``question``, asked only at a terminal (spec W2)."""
        if not self._at_terminal():
            return False
        answer = await self._ask(f"  {question}")
        return (answer or "").strip().lower() in ("y", "yes")

    async def _confirm_control_write(self, file: str, diff: str) -> bool:
        """The file tools' question before they change one of the agent's
        control files (S-314, 2026-09-27; `filesystem/agent_secrets_guard.py`):
        the diff, then a yes or no from the person at the terminal, with the
        turn's live display and key listener taken down so the question is
        not drawn over and the answer is read as a line. Anything but the
        interactive chat at a terminal answers no, and the tool says the
        owner declined."""
        from webagents.agents.skills.local.filesystem.agent_secrets_guard import CONTROL_HEADER, CONTROL_QUESTION

        if not self._chatting or not self._at_terminal():
            return False
        with self._turn_paused():
            p = self.theme.palette
            # The path as this folder knows it (`_control_path`), in the
            # header, the diff's own two header lines and the answer.
            shown = _control_path(file)
            self.console.print()
            self.console.print(Text(f"  {CONTROL_HEADER.format(path=shown)}", style=p.warning))
            for line in diff.split("\n"):
                header = re.match(r"^(---|\+\+\+) ", line)
                text = f"{header.group(1)} {shown}" if header else line
                style = p.muted if header else p.success if line.startswith("+") else p.error if line.startswith("-") else p.muted
                self.console.print(Text(f"  {text}", style=style))
            self.console.print()
            allowed = await self._confirm(CONTROL_QUESTION)
            self._answer_line("control", "yes" if allowed else "no", path=shown)
            return allowed

    @contextmanager
    def _turn_paused(self) -> Iterator[None]:
        """The turn's live display and Esc listener taken down while the chat
        asks a question mid-turn, and put back after it.

        THE ANSWER STAYS ON THE SCREEN (the ptypass-fixes lane, 2026-09-27).
        Rich's Live, stopped and started again, remembers how tall it last
        was, and its first redraw moved up that many lines: over the
        question, the typed answer and the diff's last line, so the
        scrollback kept no record of what was allowed (the PTY pass, `02`).
        The height is forgotten before it starts again, so it draws below
        what the question left."""
        live = self._turn_live
        keys = self._turn_keys
        if live is not None:
            live.stop()
        if keys is not None:
            keys.close()
        paused_at = time.time()
        try:
            yield
        finally:
            # The time the person spent answering is not the tool's (a
            # declined write read "1m 14s", the minute being the owner's):
            # every call still running starts that much later.
            renderer = self._turn_renderer
            if renderer is not None:
                renderer.shift_running_tools(time.time() - paused_at)
            if keys is not None and self._turn_reenter_keys is not None:
                self._turn_reenter_keys()
            if live is not None:
                render = getattr(live, "_live_render", None)
                if render is not None and hasattr(render, "_shape"):
                    render._shape = None
                live.start()

    def _answer_line(self, question: str, answer: str, **values: str) -> None:
        """The one line that says what was decided, under the question and
        the answer (`chat_words.ANSWER_WORDS`, the TypeScript chat's too)."""
        from .chat_words import ANSWER_WORDS

        text = ANSWER_WORDS[question][answer]
        for key, value in values.items():
            text = text.replace("{" + key + "}", str(value))
        p = self.theme.palette
        mark, rest = text[:1], text[1:]
        self.console.print(Text.assemble(("  ", ""), (mark, p.success if mark == "✓" else p.warning), (rest, p.muted)))

    async def _confirm_and_restore(
        self, folder: Path, target: Dict[str, Any], plan: Any, header: str, before_label: str, done: Any = None
    ) -> None:
        """Show what a restore would do, ask, and do it."""
        p = self.theme.palette
        self.console.print()
        self.console.print(Text(f"  {header}", style=p.text))
        for line in checkpoints.plan_lines(plan):
            self.console.print(Text(line, style=p.muted))
        if target.get("partial") is True:
            self.console.print(Text(f"  {checkpoints.PARTIAL_NOTE}", style=p.muted))
        self.console.print()
        if not await self._confirm(checkpoints.CONFIRM):
            self.notice("info", checkpoints.LEFT_AS_IS)
            return
        result = checkpoints.restore_snapshot(folder, target["id"], before_label)
        if done is not None:
            done()
        if result.written or result.removed:
            self.notice("ok", checkpoints.restored_sentence(len(result.written), len(result.removed)))
        for failure in result.failed:
            self.notice("warn", checkpoints.failed_sentence(failure["path"], failure["reason"]))

    async def cmd_undo(self) -> None:
        """`/undo`: put back what the last message changed (`checkpoints.py`)."""
        folder = self.agent_folder()
        off = checkpoints.snapshots_off_reason(folder)
        if off and self.can_change_files:
            self.notice("info", off)
            return
        snapshot_id = self.turn_snapshots[-1] if self.turn_snapshots else None
        store = checkpoints.checkpoints_dir(folder)
        everything = checkpoints.list_checkpoints(store) if snapshot_id else []
        target = next((m for m in everything if m["id"] == snapshot_id), None)
        if target is None:
            if snapshot_id:
                self.turn_snapshots.pop()
            self.notice("info", checkpoints.NOTHING_TO_UNDO)
            return
        plan = checkpoints.plan_restore(target, checkpoints.scan_folder(folder, store, everything[0]))
        if not checkpoints.plan_changes_anything(plan):
            self.turn_snapshots.pop()
            self.notice("ok", checkpoints.NOTHING_CHANGED)
            return
        await self._confirm_and_restore(folder, target, plan, checkpoints.UNDO_HEADER, "before /undo", self.turn_snapshots.pop)

    async def cmd_rewind(self, args: str) -> None:
        """`/rewind`: this folder's snapshots; `/rewind <number>`: put the folder back as that one has it."""
        p = self.theme.palette
        folder = self.agent_folder()
        store = checkpoints.checkpoints_dir(folder)
        everything = checkpoints.list_checkpoints(store)
        if not everything:
            self.notice("info", checkpoints.NO_SNAPSHOTS)
            return
        pick = args.strip()
        if not pick:
            lines = [Text(checkpoints.REWIND_TITLE, style=f"bold {p.text}")]
            for index, m in enumerate(everything[:9]):
                lines.append(
                    Text.assemble(
                        "  ",
                        (str(index + 1), p.accent),
                        "  ",
                        (when_label(m["created_at"]).ljust(12), p.muted),
                        (_truncate(m["label"], self.console.width - 20), p.text),
                    )
                )
            lines.append(Text(""))
            lines.append(Text(f"  {checkpoints.REWIND_HINT}", style=p.faint))
            self._print_lines(lines)
            return
        target = everything[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(everything) else None
        if target is None:
            self.notice("error", checkpoints.rewind_missing(pick), checkpoints.REWIND_MISSING_HINT)
            return
        plan = checkpoints.plan_restore(target, checkpoints.scan_folder(folder, store, everything[0]))
        if not checkpoints.plan_changes_anything(plan):
            self.notice("ok", checkpoints.REWIND_SAME)
            return
        await self._confirm_and_restore(
            folder, target, plan, checkpoints.rewind_header(when_label(target["created_at"]), target["label"]), "before /rewind"
        )

    async def cmd_resume(self, args: str) -> None:
        """`/resume`: the earlier conversations, here and on Robutler; `/resume <number>`: continue one."""
        p = self.theme.palette
        directory = self.session_dir()
        target: Optional[ConversationsTarget] = None
        platform: List[Any] = []
        if self.session_backend == "robutler":
            found = self.conversations_target()
            if isinstance(found, str):
                self.notice("info", unavailable_reason(found, cli_command))
            else:
                target = found
                try:
                    platform = await list_platform_conversations(found, cli_command)
                except Exception as error:  # noqa: BLE001 - said; this machine's list still answers
                    self.notice("warn", f"Could not list the conversations on Robutler: {error}.")

        def current(entry: Any) -> bool:
            return bool(self.messages) and (
                (entry.id is not None and entry.id == self.session_id)
                or (entry.chat_id is not None and entry.chat_id == self.platform_chat_id)
            )

        entries = [e for e in merge_conversations(list_sessions(directory), platform) if not current(e)]
        if not entries:
            where = ", in this folder or on Robutler." if self.session_backend == "robutler" and target else " in this folder."
            self.notice("info", f"No earlier conversations with {self.agent_name}{where}")
            return
        pick = args.strip()
        if not pick:
            width = self.console.width - 1
            lines = [Text("Earlier conversations", style=f"bold {p.text}")]
            for index, e in enumerate(entries[:9]):
                room = max(10, width - 32)
                tag = "Robutler: " if e.only_on_robutler else ""
                lines.append(
                    Text.assemble(
                        "  ",
                        (str(index + 1), p.accent),
                        "  ",
                        (when_label(e.updated_at).ljust(12), p.muted),
                        (f"{max(e.local_count, e.platform_count)} messages".ljust(14), p.faint),
                        (tag, p.muted),
                        (_truncate(e.preview or "(no text)", room - len(tag)), p.text),
                    )
                )
            lines.append(Text(""))
            lines.append(Text("  Continue one with /resume <number>.", style=p.faint))
            self._print_lines(lines)
            return
        if pick.isdigit():
            index = int(pick) - 1
            chosen = entries[index] if 0 <= index < len(entries) else None
        else:
            chosen = next((e for e in entries if (e.id or "").startswith(pick) or (e.chat_id or "").startswith(pick)), None)
        if chosen is None:
            self.notice("error", f"There is no conversation {pick}.", "Type /resume to see the list.")
            return
        local = load_session(directory, chosen.id) if chosen.id else None
        # Robutler has more of it (continued on the web, or only there): read it from there.
        if target is not None and chosen.chat_id and (local is None or chosen.platform_count > chosen.local_count):
            try:
                words = await read_platform_conversation(target, chosen.chat_id, cli_command)
            except Exception as error:  # noqa: BLE001
                self.notice("error", f"Could not read that conversation from Robutler: {error}.")
                return
            self.messages = list(words)
            self.session_id = chosen.id or chosen.session_id or new_session_id()
            self.session_created_at = (local or {}).get("created_at") or ""
            self.input_tokens = int((local or {}).get("input_tokens") or 0)
            self.output_tokens = int((local or {}).get("output_tokens") or 0)
            self.cost = _saved_cost((local or {}).get("metadata"))
            self.platform_chat_id = chosen.chat_id
            self.recorded_count = len(words)
            # Kept on this machine as well from now on.
            self.save_conversation()
        elif local is not None:
            self.messages = list(local["messages"])
            self.session_id = local["session_id"]
            self.session_created_at = local["created_at"]
            self.input_tokens = int(local["input_tokens"])
            self.output_tokens = int(local["output_tokens"])
            self.cost = _saved_cost(local["metadata"])
            chat_id = local["metadata"].get("robutler_chat_id")
            recorded = local["metadata"].get("robutler_recorded")
            self.platform_chat_id = chat_id if isinstance(chat_id, str) and chat_id else None
            self.recorded_count = recorded if self.platform_chat_id and isinstance(recorded, int) else 0
        else:
            self.notice("error", f"There is no conversation {pick}.", "Type /resume to see the list.")
            return
        self._recording_noticed = False
        self.turn_snapshots = []
        self.print_recap()
        self.notice(
            "ok",
            f"Continuing the conversation from {when_label(chosen.updated_at)} ({spoken_count(self.messages)} messages).",
        )

    def print_recap(self) -> None:
        """The last few exchanges of a resumed conversation, so it reads as a continuation."""
        p = self.theme.palette
        columns = self.console.width
        said = [
            m
            for m in self.messages
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ]
        shown = said[-6:]
        self.console.print()
        self.console.print(Text("── Earlier in this conversation ──", style=p.faint))
        if len(said) > len(shown):
            self.console.print(Text(f"   … {len(said) - len(shown)} earlier messages", style=p.faint))
        for message in shown:
            text = message["content"].strip()
            self.console.print()
            if message["role"] == "user":
                for line in sent_message(self.theme, columns, text[:240] + "…" if len(text) > 240 else text):
                    self.console.print(line)
            else:
                lines = [line for line in text.split("\n") if line.strip()]
                for index, line in enumerate(lines[:3]):
                    marker = ("✦ ", p.agent) if index == 0 else ("  ", "")
                    self.console.print(Text.assemble(marker, (_truncate(line.strip(), columns - 3), p.muted)))
                if len(lines) > 3:
                    self.console.print(Text("  …", style=p.faint))
        self.console.print()
        self.console.print(Text("── Continue below ──", style=p.faint))

    # -- commands ---------------------------------------------------------------

    def _stop(self) -> None:
        self.running = False

    def cmd_help(self, args: str) -> None:
        asked = args.strip()
        if asked:
            spec = chat_command(asked)
            if spec is None:
                self._say_unknown_command(asked)
                return
            self.notice("info", spec.usage, "\n".join([spec.description, *spec.details]))
            return
        self._print_lines(help_lines(self.theme))

    def _say_unknown_command(self, name: str) -> None:
        """`✗ Unknown command /x.`, with a did-you-mean for a moved name (`/edit`) or a near one."""
        from ..help_format import suggest_similar

        bare = name.lstrip("/").lower()
        moved = MOVED_COMMANDS.get(bare)
        if moved:
            self.notice("error", fill("unknownCommand", name=bare), f"Did you mean /{moved}? Type / to see the commands.")
            return
        match = re.search(r"Did you mean (\w+)\?", suggest_similar(bare, [c.name for c in CHAT_COMMANDS]))
        if match:
            self.notice("error", fill("unknownCommand", name=bare), fill("didYouMean", name=match.group(1)))
            return
        self.notice("error", fill("unknownCommand", name=bare), CHAT_WORDS["unknownCommandHint"])

    def _declared_provider(self) -> Any:
        """The provider whose skill the agent file names, if any (`find_provider`)."""
        from webagents.agents.skills.core.llm.providers import find_provider

        names = list(self._loaded.skills) if self._loaded is not None else (list(self.built.skills) if self.built else [])
        return next((p for p in (find_provider(n) for n in names) if p is not None), None)

    def _running_model(self) -> Optional[str]:
        """The `provider/model` the agent runs on now, or None without one."""
        if not self.built or self.model_problem:
            return None
        access = self.built.access
        if access is not None and getattr(access, "kind", None) == "none":
            return None
        if access is not None and getattr(access, "model", None):
            return str(access.model)
        from ..agent_builder import _named_skill_model

        return _named_skill_model(self.built.agent.skills) if self.built.agent is not None else None

    async def cmd_model(self, args: str) -> None:
        """`/model`: which model and how it is reached; `/model <provider/model>`:
        switch, for this chat; `--save` also keeps it in the agent file (spec 3.5).

        THE FILE'S PROVIDER DECIDES WHAT IT RUNS (D3, 2026-09-26). An agent
        naming `openai` asked for `anthropic/claude-sonnet-4` used to answer
        "Model set to openai/gpt-4o-mini", keeping its own model as if that
        were a switch; the TypeScript chat claimed the switch and ran OpenAI's
        default. Now such a switch is refused with the way out, in both.
        """
        parts = args.split()
        save = "--save" in parts
        rest = [p for p in parts if p != "--save"]
        if len(rest) > 1 or any(p.startswith("-") for p in rest):
            self.notice("error", fill("usage", usage="/model [provider/model] [--save]"))
            return
        if not rest and not save:
            self.notice("info", f"Model: {self.model_label() or '(none)'}", "Switch with /model <provider/model>.")
            return
        from webagents.agents.skills.core.llm.providers import find_provider

        current = self._running_model()
        wanted = rest[0] if rest else current
        if not wanted:
            self.notice("error", fill("usage", usage="/model [provider/model] [--save]"))
            return
        file = self.built.file if self.built else None
        if save and file is None:
            self.notice("error", CHAT_WORDS["builtInNoFileToKeep"])
            return
        # The file names a provider's skill: it runs that provider's models,
        # and asking for another's changes nothing (D3). Robutler's proxy
        # serves them all.
        declared = self._declared_provider()
        asked = find_provider(wanted.split("/", 1)[0]) if "/" in wanted else None
        if declared is not None and declared.id != "proxy" and asked is not None and asked.id != declared.id:
            self.notice(
                "error",
                fill("modelRefusal", name=self.agent_name, provider=declared.id),
                fill("modelRefusalHint", provider=declared.id, other=asked.id),
            )
            return
        if wanted != current or not save:
            # REBUILD THE AGENT, do not just set the field: the built LLM skill
            # keeps its own model.
            previous = self.explicit_model
            self.explicit_model = wanted
            try:
                await self.initialize()
                # A model with no way to run here (no key, not signed in) is not a
                # switch; keep the one that works.
                if self.model_problem:
                    raise RuntimeError(self.model_problem)
            except Exception as error:  # noqa: BLE001 - said, and the working agent put back
                self.explicit_model = previous
                self.notice("error", f"Could not switch model: {error}")
                try:
                    await self.initialize()
                except Exception:  # noqa: BLE001
                    pass
                return
            model = self._running_model() or wanted
            hint = None if save or file is None else fill("modelSwitchHint", model=model, file=file.name)
            self.notice("ok", f"Model set to {self.model_label()}", hint)
        if not save:
            return
        await self._save_model(self._running_model() or wanted)

    def rounds_now(self) -> Tuple[int, str]:
        """The running agent's tool-round budget, and where it came from."""
        from webagents.agents.core.tool_budget import DEFAULT_MAX_TOOL_ITERATIONS

        agent = self.built.agent if self.built else None
        rounds = getattr(agent, "max_tool_iterations", DEFAULT_MAX_TOOL_ITERATIONS)
        return rounds, getattr(agent, "max_tool_rounds_source", "default")

    def rounds_status(self) -> str:
        """`/status`'s line: the budget and its source."""
        from webagents.agents.core.tool_budget import ROUNDS_WORDS, rounds_source_words

        rounds, source = self.rounds_now()
        file = self.built.file.name if self.built and self.built.file else None
        return ROUNDS_WORDS["status"].format(rounds=rounds, source=rounds_source_words(source, file))

    async def cmd_rounds(self, args: str) -> None:
        """`/rounds`: the tool rounds one turn may run, and where that came
        from; `/rounds <n>`: set it for this chat; `--save` also keeps it in
        the agent file as `max_tool_rounds`, as `/model --save` keeps a model
        (2026-09-28, `agents/core/tool_budget.py`)."""
        from webagents.agents.core.tool_budget import ROUNDS_WORDS, parse_max_tool_rounds, rounds_source_words

        parts = args.split()
        save = "--save" in parts
        rest = [p for p in parts if p != "--save"]
        file = self.built.file if self.built else None
        if len(rest) > 1 or (save and not rest):
            self.notice("error", fill("usage", usage="/rounds [n] [--save]"))
            return
        if not rest:
            rounds, source = self.rounds_now()
            said = ROUNDS_WORDS["show"].format(rounds=rounds, source=rounds_source_words(source, file.name if file else None))
            self.notice("info", said, ROUNDS_WORDS["showHint"])
            return
        try:
            rounds = parse_max_tool_rounds(rest[0], "/rounds")
        except ValueError as error:
            self.notice("error", str(error))
            return
        if save and file is None:
            self.notice("error", CHAT_WORDS["builtInNoFileToKeep"])
            return
        self.session_rounds = rounds
        if self.built and self.built.agent is not None:
            self.built.agent.max_tool_iterations = rounds
            self.built.agent.max_tool_rounds_source = "session"
        if not save:
            hint = ROUNDS_WORDS["setHint"].format(rounds=rounds, file=file.name) if file else None
            self.notice("ok", ROUNDS_WORDS["set"].format(rounds=rounds), hint)
            return
        await self._save_rounds(rounds)

    async def _save_rounds(self, rounds: int) -> None:
        """`/rounds <n> --save`: `max_tool_rounds` into the file's front matter, then reload."""
        from webagents.agents.core.tool_budget import ROUNDS_WORDS

        from ..skills_edit import edit_front_matter_scalar, unsafe_target_reason, write_by_rename

        file = self.built.file if self.built else None
        if file is None:
            return
        if not self._at_terminal():
            self.notice("error", fill("notAtTerminal", command="rounds --save"))
            return
        unsafe = unsafe_target_reason(file, file.parent, "the chat")
        if unsafe is not None:
            self.notice("error", unsafe)
            return
        try:
            edit = edit_front_matter_scalar(file.read_text(), "max_tool_rounds", str(rounds), str(file))
        except Exception as error:  # noqa: BLE001 - the editor's own sentence
            self.notice("error", str(error))
            return
        if not edit.changed:
            self.notice("info", ROUNDS_WORDS["already"].format(file=file.name, rounds=rounds))
            return
        p = self.theme.palette
        self.console.print(Text(f"  max_tool_rounds: {rounds}", style=p.muted))
        if not await self._confirm(CHAT_WORDS["makeThisChange"]):
            self.notice("info", CHAT_WORDS["leftAsIs"])
            return
        self._snapshot_for_command("/rounds")
        write_by_rename(file, edit.text)
        # The file carries the budget now: it is no longer this chat's alone.
        self.session_rounds = None
        self.notice("ok", ROUNDS_WORDS["kept"].format(rounds=rounds, file=file.name))
        await self._reload_now()

    async def _save_model(self, model: str) -> None:
        """`/model --save`: the model into the file's front matter (W2 to W6), then reload."""
        from ..skills_edit import edit_front_matter_scalar, unsafe_target_reason, write_by_rename

        file = self.built.file if self.built else None
        if file is None:
            return
        if not self._at_terminal():
            self.notice("error", fill("notAtTerminal", command="model --save"))
            return
        unsafe = unsafe_target_reason(file, file.parent, "the chat")
        if unsafe is not None:
            self.notice("error", unsafe)
            return
        try:
            edit = edit_front_matter_scalar(file.read_text(), "model", model, str(file))
        except Exception as error:  # noqa: BLE001 - the editor's own sentence
            self.notice("error", str(error))
            return
        if not edit.changed:
            self.notice("info", f"{file.name} already keeps {fill('planModel', model=model)}.")
            return
        p = self.theme.palette
        self.console.print(Text(f"  {fill('planModel', model=model)}", style=p.muted))
        if not await self._confirm(CHAT_WORDS["makeThisChange"]):
            self.notice("info", CHAT_WORDS["leftAsIs"])
            return
        self._snapshot_for_command("/model")
        write_by_rename(file, edit.text)
        # The file carries the model now: the switch is no longer this chat's alone.
        self.explicit_model = None
        self.notice("ok", fill("modelKept", model=model, file=file.name))
        await self._reload_now()

    def folder_agents(self) -> List[Tuple[str, Path, str]]:
        """The agent files in this folder, with their names: AGENT.md and AGENT-<name>.md."""
        from ..agent_files import folder_agents

        return list(folder_agents(Path.cwd()))

    async def cmd_agent(self, args: str) -> None:
        parts = args.split()
        verb = parts[0] if parts else ""
        rest = parts[1:]
        if verb == "new" and rest:
            await self.cmd_agent_new(rest)
            return
        if verb == "edit":
            await self.cmd_agent_edit(rest)
            return

        p = self.theme.palette
        agents = self.folder_agents()
        current = self.agent_name
        wanted = args.strip()
        if not wanted:
            width = max([10, len(BUILT_IN_AGENT)] + [len(a.name) for a in agents]) + 2
            room = max(10, self.console.width - width - 24)

            def row(name: str, where: str, description: str) -> Text:
                mark = ("●", p.accent) if name == current else ("○", p.faint)
                return Text.assemble(
                    "  ", mark, " ", (name.ljust(width), f"bold {p.text}"),
                    (where.ljust(18), p.faint), (_truncate(description, room), p.muted),
                )

            lines = [Text("Agents", style=f"bold {p.text}")]
            for a in agents:
                # A file that does not load is listed with its sentence (D2).
                if a.problem:
                    lines.append(Text.assemble("  ", ("✗", f"bold {p.error}"), " ", (a.name, p.text), "  ", (a.file.name, p.faint), "  ", (a.problem, p.muted)))
                else:
                    lines.append(row(a.name, a.file.name, a.description))
            lines.append(row(BUILT_IN_AGENT, "built in", "The general assistant"))
            lines.append(Text(""))
            lines.append(Text("  Switch with /agent <name>.", style=p.faint))
            self._print_lines(lines)
            return
        target = next((a for a in agents if a.name == wanted), None)
        if target is None and wanted != BUILT_IN_AGENT:
            self.notice("error", fill("noAgentCalled", name=wanted), CHAT_WORDS["seeTheList"])
            return
        if target is not None and target.problem:
            # A broken agent is not switched to; the chat keeps the one it has (D2).
            self.notice("error", target.problem, fill("keepsTalking", name=current))
            return
        if wanted == current:
            self.notice("info", f"Already talking to {current}.")
            return
        await self._switch_to(target.file if target else None, wanted)

    async def _switch_to(self, file: Optional[Path], wanted: str) -> None:
        """Switch to `file` (None: the built-in agent), rebuild and greet, keeping the running agent if the build fails."""
        previous_file, previous_model = self.selected_file, self.explicit_model
        self.selected_file = file
        self.explicit_model = None
        try:
            await self.initialize()
        except Exception as error:  # noqa: BLE001 - a bad file keeps the running agent
            self.selected_file, self.explicit_model = previous_file, previous_model
            try:
                await self.initialize()
            except Exception:  # noqa: BLE001
                pass
            self.notice("error", str(error), fill("keepsTalking", name=self.agent_name))
            return
        self.start_new_conversation(False)
        if self.console.is_terminal:
            self.console.print()
            for line in welcome_card(self.theme, self.console.width, self.welcome_info()):
                self.console.print(line)
        self.notice("ok", fill("nowTalking", name=self.agent_name), self.model_problem)

    async def cmd_reload(self) -> None:
        """`/reload`: read the agent file again and use it (spec 3.1), asking when a policy part changed."""
        file = self.built.file if self.built else None
        if file is None or self._loaded is None:
            self.notice("info", CHAT_WORDS["builtInNoFileToRead"])
            return
        after = self._loaded_of(self.built)
        if after is None:
            # The file no longer loads: refuse the reload with the loader's sentence.
            try:
                from ..loader.hierarchy import load_agent

                load_agent(file)
            except Exception as error:  # noqa: BLE001
                self.notice("error", str(error), CHAT_WORDS["keepsAgent"])
                return
            return
        shown = file.name
        if agent_reload.same_version(self._loaded, after):
            self.notice("info", fill("upToDate", name=self.agent_name, file=shown))
            return
        diff = agent_reload.reload_diff(self._loaded, after)
        p = self.theme.palette
        self.console.print()
        self.console.print(Text(f"  {fill('changedHeader', file=shown)}", style=p.text))
        for part in diff.parts:
            self.console.print(Text(f"  {part}", style=p.muted))
        self.console.print()
        if diff.policy_changed and not await self._confirm(CHAT_WORDS["useThisVersion"]):
            self.notice("info", CHAT_WORDS["keepsAgent"])
            return
        await self._reload_now()

    async def _reload_now(self) -> None:
        """Build from the file again and swap it in, saying what changed; a failed build keeps the running agent."""
        before = self._tool_names
        was_name = self.agent_name
        shown = self.built.file.name if self.built and self.built.file else ""
        try:
            await self.initialize()
        except Exception as error:  # noqa: BLE001
            self.notice("error", str(error), CHAT_WORDS["keepsAgent"])
            return
        detail = self.model_problem or (" ".join(agent_reload.tool_change_lines(before, self._tool_names)) or None)
        self.notice("ok", fill("reloaded", name=self.agent_name, file=shown), detail)
        if was_name and self.agent_name != was_name:
            self.start_new_conversation(False)

    async def cmd_agent_edit(self, rest: List[str]) -> None:
        """`/agent edit [name]`: open the agent's file in $VISUAL/$EDITOR, then use it (spec 3.2)."""
        if len(rest) > 1:
            self.notice("error", fill("usage", usage="/agent edit [name]"))
            return
        name = rest[0] if rest else None
        if name:
            target = next((a for a in self.folder_agents() if a.name == name), None)
            if target is None:
                self.notice("error", fill("noAgentCalled", name=name), CHAT_WORDS["seeTheList"])
                return
            file = target.file
        else:
            if self.built is None or self.built.file is None:
                self.notice("info", CHAT_WORDS["builtInNoFileToEdit"], CHAT_WORDS["makeOneHint"])
                return
            file = self.built.file
        if not self._at_terminal():
            self.notice("error", fill("notAtTerminal", command="agent edit"))
            return
        from ..skills_edit import unsafe_target_reason

        unsafe = unsafe_target_reason(file, file.parent, "the chat")
        if unsafe is not None:
            self.notice("error", unsafe)
            return
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            self.notice("info", fill("fileIsAt", file=file.name, path=str(file)), CHAT_WORDS["setEditor"])
            return
        code = self._run_editor(editor, file)
        if code != 0:
            self.notice("error", fill("editorExited", code=code), CHAT_WORDS["keepsAgent"])
            return
        if self.built is not None and file == self.built.file:
            await self._reload_now()
        else:
            self.notice("ok", fill("saved", file=file.name), fill("switchHint", name=name or ""))

    def _run_editor(self, editor: str, file: Path) -> int:
        """Run the person's editor on `file`, stdio inherited; the exit code, or 1 when it could not start."""
        import subprocess

        try:
            if os.name == "nt":
                result = subprocess.run(["cmd.exe", "/d", "/s", "/c", f'{editor} "{file}"'])
            else:
                result = subprocess.run(["/bin/sh", "-c", f'{editor} "$1"', "sh", str(file)])
        except OSError:
            return 1
        return result.returncode

    async def cmd_agent_new(self, rest: List[str]) -> None:
        """`/agent new <name> [chatbot|tool-agent]`: make an agent file here and switch to it (spec 3.3)."""
        from ..init_templates import AGENT_NAME_RE, INIT_TEMPLATES, agent_markdown
        from ..skills_edit import unsafe_target_reason

        name = rest[0] if rest else ""
        template = rest[1] if len(rest) > 1 else "chatbot"
        if not name or len(rest) > 2:
            self.notice("error", fill("usage", usage="/agent new <name> [chatbot|tool-agent]"))
            return
        if not AGENT_NAME_RE.match(name):
            self.notice("error", CHAT_WORDS["badName"])
            return
        if template not in INIT_TEMPLATES:
            self.notice("error", fill("unknownTemplate", template=template), CHAT_WORDS["templates"])
            return
        folder = Path.cwd()
        if any(a.name == name for a in self.folder_agents()):
            self.notice("error", fill("taken", name=name), fill("switchHint", name=name))
            return
        if checkpoints.snapshots_off_reason(folder):
            self.notice("error", CHAT_WORDS["homeFolder"], CHAT_WORDS["homeFolderHint"])
            return
        if not self._at_terminal():
            self.notice("error", fill("notAtTerminal", command="agent new"))
            return
        has_agent = bool(self.folder_agents()) or (folder / "AGENT.md").exists()
        file = folder / (f"AGENT-{name}.md" if has_agent else "AGENT.md")
        unsafe = unsafe_target_reason(file, folder, "the chat")
        if unsafe is not None:
            self.notice("error", unsafe)
            return
        if not await self._confirm(fill("makeFile", file=file.name, template=template)):
            self.notice("info", CHAT_WORDS["leftAsIs"])
            return
        self._snapshot_for_command("/agent new")
        access = self.built.access if self.built else None
        model = self.model_label_for_new(access)
        file.write_text(agent_markdown(name, template, model))
        self.notice("ok", fill("made", file=file.name), CHAT_WORDS["madeHint"])
        await self._switch_to(file, name)

    def model_label_for_new(self, access: Any) -> Optional[str]:
        """The provider/model to write into a new agent: the one the chat runs on with a key here, else None (the template default)."""
        if access is not None and getattr(access, "kind", None) == "direct" and getattr(access, "model", None):
            return access.model
        return None

    def _snapshot_for_command(self, typed: str) -> None:
        """A W5 snapshot before a command writes, so /undo can put it back; quiet where snapshots are off."""
        folder = self.agent_folder()
        if checkpoints.snapshots_off_reason(folder):
            return
        try:
            self.turn_snapshots.append(checkpoints.take_snapshot(folder, fill("beforeCommand", command=typed))["id"])
        except OSError as error:
            if not self._snapshots_noticed:
                self.notice("warn", checkpoints.snapshot_failed(str(error)))
            self._snapshots_noticed = True

    async def cmd_skills(self, args: str) -> None:
        """`/skills`: this agent's skills; `/skills add|remove` change them; `/skills list` shows every name (spec 3.4)."""
        parts = args.split()
        action = parts[0] if parts else ""
        rest = parts[1:]
        if not action:
            self._command_skills_show()
            return
        if action == "list":
            from ..agent_builder import SKILL_CLASSES
            from ..skills_edit import skillmd_list_lines

            p = self.theme.palette
            lines = [Text("Skills an agent file can name:", style=f"bold {p.text}"), Text("")]
            for name in sorted(SKILL_CLASSES):
                lines.append(Text(f"  {name}", style=p.text))
            lines.append(Text(""))
            for line in skillmd_list_lines(Path.cwd()):
                lines.append(Text(line, style=p.muted))
            self._print_lines(lines)
            return
        if action not in ("add", "remove"):
            self.notice("error", fill("usage", usage="/skills [list|add|remove]"))
            return
        if not rest:
            usage = (
                "/skills add <name>... | <owner/repo | git URL | folder> [--skill <name>]"
                if action == "add"
                else "/skills remove <name>..."
            )
            self.notice("error", fill("usage", usage=usage))
            return
        await self._command_skills_edit(action, rest)

    def _command_skills_show(self) -> None:
        """`/skills`: this agent's coded skills and the folder's SKILL.md skills."""
        file = self.built.file if self.built else None
        p = self.theme.palette
        if file is None:
            # The built-in agent has skills too (2026-09-27): it used to refuse
            # with "no agent file to change", which is /skills add's answer.
            lines = [Text(fill("skillsOfBuiltIn", name=self.agent_name), style=f"bold {p.text}")]
            declared = self.built.declared_skills if self.built else []
            for name in declared:
                lines.append(Text(f"  {name}", style=p.text))
            if not declared:
                lines.append(Text(f"  {CHAT_WORDS['none']}", style=p.faint))
            lines.append(Text(""))
            lines.append(Text(f"  {CHAT_WORDS['builtInSkillsHint']}", style=p.faint))
            self._print_lines(lines)
            return
        lines = [Text(fill("skillsOf", name=self.agent_name, file=file.name), style=f"bold {p.text}")]
        skills = self._loaded.skills if self._loaded else []
        for name in skills:
            lines.append(Text(f"  {name}", style=p.text))
        if not skills:
            lines.append(Text(f"  {CHAT_WORDS['none']}", style=p.faint))
        md = self._loaded.skillmd if self._loaded else []
        if md:
            # Where each one came from (2026-09-29): the lock's source and
            # short commit, its local folder, or the person's own folder when
            # the lock does not know it. The TypeScript chat says the same.
            from webagents.agents.skills.local.skillmd.skillmd_install import read_lock

            installed = read_lock(str(self.agent_folder())).get("skills") or {}
            lines.append(Text(""))
            lines.append(Text(CHAT_WORDS["skillmdHeading"], style=p.text))
            for name in md:
                lines.append(Text(f"  {skillmd_origin_line(name, installed.get(name))}", style=p.muted))
        lines.append(Text(""))
        lines.append(Text(f"  {CHAT_WORDS['skillsHint']}", style=p.faint))
        self._print_lines(lines)

    async def _command_skills_edit(self, action: str, rest: List[str]) -> None:
        """`/skills add|remove`: plan, show, ask, snapshot, apply, reload (spec 3.4)."""
        from ..skills_edit import apply_skills, plan_skills, spoken_list

        file = self.built.file if self.built else None
        if file is None:
            self.notice("info", CHAT_WORDS["builtInNoFileToChange"], CHAT_WORDS["makeOneHint"])
            return
        folder = self.agent_folder()
        asked_yes = "--yes" in rest or "-y" in rest
        skill_at = rest.index("--skill") if "--skill" in rest else -1
        wanted_skill = rest[skill_at + 1] if skill_at != -1 and skill_at + 1 < len(rest) and not rest[skill_at + 1].startswith("-") else None
        names = [t for i, t in enumerate(rest) if t not in ("--yes", "-y", "--skill") and not (skill_at != -1 and i == skill_at + 1)]

        plan = plan_skills(action, names, folder=folder, file=file, who="the chat")
        if plan.errors:
            for line in plan.errors:
                self.notice("error", line)
            return
        if plan.sources:
            await self._install_sources(plan.sources, wanted_skill, asked_yes)
            return
        if not plan.changed:
            if plan.already:
                self.notice("info", f"{file.name} already names {spoken_list(plan.already)}.", f"Skills: {', '.join(plan.skills_after) if plan.skills_after else 'none'}")
            else:
                self.notice("info", f"{file.name} does not name {spoken_list(plan.absent)}.", f"Skills: {', '.join(plan.skills_after) if plan.skills_after else 'none'}")
            return
        if not self._at_terminal():
            self.notice("error", fill("notAtTerminal", command=f"skills {action}"), fill("inAScript", command=f"webagents skills {action} {' '.join(names)}"))
            return
        p = self.theme.palette
        self.console.print()
        if plan.add:
            self.console.print(Text(f"  {fill('planAdd', names=', '.join(plan.add))}", style=p.muted))
        if plan.remove:
            self.console.print(Text(f"  {fill('planRemove', names=', '.join(plan.remove))}", style=p.muted))
        for name in plan.installed_removals:
            self.console.print(Text(f"  {fill('planRemoveInstalled', skill=name, files=plural(self._installed_file_count(folder, name), 'file'))}", style=p.muted))
        if plan.add or plan.remove:
            self.console.print(Text(f"  {fill('planAfter', list=', '.join(plan.skills_after) if plan.skills_after else 'none')}", style=p.muted))
        if not await self._confirm(CHAT_WORDS["makeThisChange"]):
            self.notice("info", CHAT_WORDS["leftAsIs"])
            return
        self._snapshot_for_command(f"/skills {action}")
        shown = file.name
        applied = apply_skills(action, plan, folder)
        for name in applied.installed_removed:
            self.notice("ok", f"Removed {name} from .agents/skills.")
        if applied.added:
            self.notice("ok", f"Added {spoken_list(applied.added)} to {shown}.")
        if applied.removed:
            self.notice("ok", f"Removed {spoken_list(applied.removed)} from {shown}.")
        if applied.added or applied.removed:
            self.console.print(Text(f"  Skills: {', '.join(applied.skills_after) if applied.skills_after else 'none'}", style=p.muted))
        await self._reload_now()

    def _installed_file_count(self, folder: Path, name: str) -> int:
        try:
            import json as _json

            lock = _json.loads((folder / ".webagents" / "skills.lock").read_text())
            return len(((lock.get("skills") or {}).get(name) or {}).get("files") or [])
        except Exception:  # noqa: BLE001
            return 0

    async def _install_sources(self, sources: List[str], skill: Optional[str], asked_yes: bool) -> None:
        """A SKILL.md source: refuse --yes (W2), then the CLI installer with the chat's confirm; reload after."""
        if asked_yes:
            self.notice("error", CHAT_WORDS["alwaysAsks"], CHAT_WORDS["alwaysAsksHint"])
            return
        if not self._at_terminal():
            self.notice("error", fill("notAtTerminal", command="skills add"), fill("inAScript", command=f"webagents skills add {' '.join(sources)}"))
            return
        from webagents.agents.skills.local.skillmd.skillmd_install import install_from_source, parse_source

        p = self.theme.palette
        folder = self.agent_folder()
        installed_any = False
        snapped = {"done": False}

        # The installer is synchronous and asks through a sync callback; run it
        # in a thread and answer its question from this loop.
        loop = asyncio.get_event_loop()

        def ask(question: str) -> bool:
            future = asyncio.run_coroutine_threadsafe(self._confirm_source(question, snapped), loop)
            return future.result()

        for source in sources:
            code = await asyncio.to_thread(
                install_from_source,
                parse_source(source),
                str(folder),
                skill=skill,
                yes=False,
                tty=True,
                confirm=ask,
                out=lambda line: self.console.print(Text(f"  {line}", style=p.muted)),
                err=lambda line: self.notice("error", line),
            )
            if code == 0:
                installed_any = True
        if installed_any:
            await self._reload_now()

    async def _confirm_source(self, question: str, snapped: Dict[str, bool]) -> bool:
        yes = await self._confirm(question)
        if yes and not snapped["done"]:
            self._snapshot_for_command("/skills add")
            snapped["done"] = True
        return yes

    # -- the before-prompt notice and the tip (spec 3.1, 3.3) -----------------

    def _say_new_agent_tip(self) -> None:
        """After the card, on the built-in agent in a folder with no agent file: `/agent new <name> makes one`."""
        if self.built is not None and self.built.file is not None:
            return
        if self.folder_agents():
            return
        self.console.print(Text(CHAT_WORDS["tip"], style=self.theme.palette.faint))
        self.console.print()

    def _file_changed_now(self) -> Optional[LoadedAgent]:
        """The new fingerprint when the file differs from the loaded version; None otherwise or for the built-in agent."""
        if self.built is None or self.built.file is None or self._loaded is None:
            return None
        after = self._loaded_of(self.built)
        if after is None or agent_reload.same_version(self._loaded, after):
            return None
        return after

    def _say_file_changed(self) -> None:
        """Before a prompt: the agent file changed since the chat loaded it (S-283); said once per version, never reloaded."""
        after = self._file_changed_now()
        if after is None or self.built is None or self.built.file is None:
            return
        if self._version_noticed == after.sha:
            return
        self._version_noticed = after.sha
        shown = self.built.file.name
        if self._changed_during_reply:
            self.notice("warn", fill("changedDuringReply", file=shown))
        else:
            self.notice("info", fill("changedSinceLoaded", file=shown))
        self._changed_during_reply = False

    def cmd_tools(self) -> None:
        """`/tools`: what the agent can use, by name, and who else may (the scopes column, spec 3.7)."""
        p = self.theme.palette
        tools = self.tool_rows()
        if not tools:
            self.notice("info", "This agent has no tools.", "Add skills to its AGENT.md, for example `filesystem`.")
            return
        width = min(28, max(len(name) for name, _, _ in tools)) + 2
        labels = [scope_label(scope) for _, _, scope in tools]
        scope_width = max(len(label) for label in labels) + 2
        room = self.console.width - width - scope_width - 6
        lines = [Text(f"Tools ({len(tools)})", style=f"bold {p.text}")]
        for (name, description, _scope), label in zip(tools, labels):
            lines.append(
                Text.assemble(
                    "  ",
                    ("●", p.success),
                    " ",
                    (_truncate(name, width - 2).ljust(width), f"bold {p.text}"),
                    (label.ljust(scope_width), p.faint),
                    (_truncate(description, max(10, room)), p.muted),
                )
            )
        self._print_lines(lines)

    def cmd_access(self) -> None:
        """`/access`: who may call this agent, and what each caller gets, from the parsed block (spec 3.7)."""
        p = self.theme.palette
        policy = getattr(self.built, "access_policy", None) if self.built else None
        if policy is None:
            yours = [name for name, _, scope in self.tool_rows() if scope_label(scope) == CHAT_WORDS["scopeOnlyYou"]]
            self.notice("info", fill("noAccessBlock", tools=", ".join(yours) if yours else "no tools"), CHAT_WORDS["noAccessBlockHint"])
            return
        rows: List[Tuple[str, str]] = []
        if policy.deny:
            rows.append((CHAT_WORDS["accessRefusedLabel"], ", ".join(_pattern_text(d) for d in policy.deny)))
        for group, rule in policy.groups.items():
            rows.append((CHAT_WORDS["accessGroupsLabel"], f"{group}: {_group_text(rule)}"))
        rows.append(
            (
                CHAT_WORDS["accessOthersLabel"],
                fill("accessOthersGroup", group=policy.default) if policy.default else CHAT_WORDS["accessRefused"],
            )
        )
        # Tools by the set of groups that name them, so `shell, todo: you and staff` reads as one line.
        by_groups: Dict[str, List[str]] = {}
        for group, names in policy.tools.items():
            for name in names:
                groups = [g for g, lst in policy.tools.items() if name in lst]
                key = ", ".join(groups)
                lst = by_groups.setdefault(key, [])
                if name not in lst and groups[0] == group:
                    lst.append(name)
        for groups, names in by_groups.items():
            rows.append((CHAT_WORDS["accessToolsLabel"], fill("accessToolsRow", names=", ".join(names), groups=groups)))
        rows.append((CHAT_WORDS["accessToolsLabel"], CHAT_WORDS["accessEveryOther"]))
        from rich.table import Table

        file = self.built.file.name if self.built and self.built.file else "AGENT.md"
        width = max(len(label) for label, _ in rows) + 3
        grid = Table.grid(padding=0)
        grid.add_column(width=width + 2, no_wrap=True)
        grid.add_column(ratio=1, overflow="fold")
        last = ""
        for label, value in rows:
            shown = "" if label == last else label
            last = label
            grid.add_row(Text("  " + shown, style=p.muted), Text(value, style=p.text))
        self.console.print()
        self.console.print(Text(fill("accessTitle", file=file), style=f"bold {p.text}"))
        self.console.print(grid, width=max(40, self.console.width - 1))
        self.console.print()

    def _mcp_skill(self) -> Any:
        """The agent's MCP skill, when it has one."""
        return self.built.agent.skills.get("mcp") if self.built else None

    def cmd_mcp(self) -> None:
        """`/mcp`: the servers the agent uses, from the skill's own report, values masked (spec 3.7, S-292)."""
        p = self.theme.palette
        file = self.built.file.name if self.built and self.built.file else "AGENT.md"
        skill = self._mcp_skill()
        rows = skill.server_report() if skill is not None and hasattr(skill, "server_report") else []
        if not rows:
            self.notice("info", fill("mcpNone", name=self.agent_name), fill("mcpNoneHint", file=file))
            return
        heading = CHAT_WORDS["mcpFromJson"] if getattr(skill, "config_source", "config") == "mcp.json" else fill("mcpFromFile", file=file)
        lines = [Text(heading, style=f"bold {p.text}")]
        for row in rows:
            if row.get("rejected"):
                text = fill("mcpRejected", server=row["name"], reason=row["rejected"])
            elif not row.get("connected"):
                text = fill("mcpNotConnected", server=row["name"], transport=row.get("transport") or "unknown", error=row.get("error") or "not connected")
            else:
                names = list(row.get("tools") or [])
                shown = ", ".join(names[:3])
                more = fill("mcpMore", count=len(names) - 3) if len(names) > 3 else ""
                tools = f"{plural(len(names), 'tool')}: {shown}{more}" if names else plural(0, "tool")
                text = fill("mcpConnected", server=row["name"], transport=row.get("transport") or "unknown", tools=tools)
            lines.append(Text(f"  {text}", style=p.text if row.get("connected") else p.muted))
        lines.append(Text(""))
        lines.append(Text(f"  {CHAT_WORDS['mcpServe']}", style=p.faint))
        self._print_lines(lines)

    async def cmd_cron(self, args: str) -> None:
        """`/cron`: this agent's schedules as the daemon runs them; `/cron run <name>`: one now, after asking (spec 3.7)."""
        from ..cron_command import folder_schedules, list_schedules_command, run_schedule

        parts = args.split()
        name = self.agent_name
        file = self.built.file if self.built else None
        shown = file.name if file is not None else "AGENT.md"
        folder = self.agent_folder()
        if parts and (parts[0] != "run" or len(parts) != 2):
            self.notice("error", fill("usage", usage="/cron [run <name>]"))
            return
        p = self.theme.palette
        if not parts:
            if file is None:
                self.notice("info", fill("cronNone", name=name), fill("cronNoneHint", file=shown))
                return
            lines: List[str] = []
            list_schedules_command(folder, log=lines.append, error=lambda line: self.notice("error", line), agent=name)
            if not lines or lines[0].startswith("No schedules"):
                self.notice("info", fill("cronNone", name=name), fill("cronNoneHint", file=shown))
                return
            self._print_lines([Text(f"  {line}", style=p.text) for line in lines] + [Text(""), Text(f"  {CHAT_WORDS['cronDaemon']}", style=p.faint)])
            return
        schedule = parts[1]
        found = next((f for f in folder_schedules(folder, lambda _line: None) if f.name == name), None) if file is not None else None
        entry = next((s for s in found.schedules if s.name == schedule), None) if found is not None else None
        if entry is None:
            self.notice("error", fill("cronNoSchedule", schedule=schedule), CHAT_WORDS["seeTheSchedules"])
            return
        deliver = entry.deliver
        if deliver.kind == "file":
            target = deliver.path
        elif deliver.kind == "webhook":
            from urllib.parse import urlsplit

            target = urlsplit(deliver.url).netloc
        else:
            target = CHAT_WORDS["cronChatTarget"]
        if not await self._confirm(fill("cronRun", schedule=schedule, target=target)):
            self.notice("info", CHAT_WORDS["notRun"])
            return
        outcome: List[str] = []
        code = await run_schedule(name, schedule, folder, log=outcome.append, error=lambda line: self.notice("error", line))
        if outcome:
            self.notice("ok" if code == 0 else "error", outcome[-1])

    def _memory_skill(self) -> Any:
        """The agent's memory skill, when it has one and it answers the owner's view."""
        skill = self.built.agent.skills.get("memory") if self.built else None
        return skill if skill is not None and hasattr(skill, "owner_summary") else None

    async def cmd_memory(self, args: str) -> None:
        """`/memory`: what the agent remembers, and where; `/memory forget <key>`: one of your notes, after asking (spec 3.7)."""
        parts = args.split()
        name = self.agent_name
        if parts and (parts[0] != "forget" or len(parts) != 2):
            self.notice("error", fill("usage", usage="/memory [forget <key>]"))
            return
        skill = self._memory_skill()
        if skill is None:
            self.notice("info", fill("memoryNone", name=name), CHAT_WORDS["memoryNoneHint"])
            return
        if parts:
            key = parts[1]
            if not await skill.has_own_note(key):
                self.notice("error", fill("memoryNoNote", key=key))
                return
            if not await self._confirm(fill("memoryForget", key=key)):
                self.notice("info", CHAT_WORDS["leftAsIs"])
                return
            if await skill.forget_own(key):
                self.notice("ok", fill("memoryForgot", key=key))
            else:
                self.notice("error", fill("memoryNoNote", key=key))
            return
        summary = await skill.owner_summary()
        p = self.theme.palette
        # "on Robutler" only when the tier has the agent's key to get there (B5).
        robutler = summary["portal"] and summary.get("portal_key", True)
        kept_in = (
            CHAT_WORDS["memoryLocal"]
            + (
                CHAT_WORDS["memoryAndRobutler"] if robutler
                else CHAT_WORDS["memoryRobutlerNoKey"] if summary["portal"]
                else ""
            )
            if summary["local"]
            else CHAT_WORDS["memoryRobutlerOnly"] if robutler else CHAT_WORDS["memoryRobutlerOnlyNoKey"]
        )
        rows = [
            (CHAT_WORDS["memoryKeptInLabel"], kept_in),
            (CHAT_WORDS["memoryYoursLabel"], plural(summary["owner"], "note")),
            (CHAT_WORDS["memorySharedLabel"], plural(summary["shared"], "note")),
            (
                CHAT_WORDS["memoryCallersLabel"],
                fill("memoryCallers", callers=plural(summary["callers"], "caller"), notes=plural(summary["caller_notes"], "note")),
            ),
        ]
        width = max(len(label) for label, _ in rows) + 3
        lines = [Text(fill("memoryTitle", name=name), style=f"bold {p.text}")]
        for label, value in rows:
            lines.append(Text.assemble("  ", (label.ljust(width), p.muted), (value, p.text)))
        if summary["recent"]:
            lines.append(Text(""))
            key_width = min(32, max(len(n["key"]) for n in summary["recent"])) + 2
            for note in summary["recent"]:
                lines.append(
                    Text.assemble(
                        "  ",
                        (_truncate(note["key"], key_width - 2).ljust(key_width), p.text),
                        (_truncate(note["first_line"], max(10, self.console.width - key_width - 4)), p.muted),
                    )
                )
        lines.append(Text(""))
        lines.append(Text(f"  {CHAT_WORDS['memoryHint']}", style=p.faint))
        self._print_lines(lines)

    async def _who_am_i(self, portal: str, token: str) -> Any:
        """The username behind the stored sign-in, "expired", or None when unknown."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=4) as client:
                response = await client.get(f"{portal}/api/users/me", headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            return None
        if response.status_code == 401:
            return "expired"
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        user = data.get("user") if isinstance(data.get("user"), dict) else data
        return user.get("username") or None

    def _provider_env_var(self) -> Optional[str]:
        from ..agent_builder import provider_env_var

        return provider_env_var(self.built) if self.built else None

    def model_route(self) -> str:
        """How the agent reaches its model, in words: for /status."""
        access = self.built.access if self.built else None
        if self.model_problem or (access is not None and access.kind == "none"):
            kind = getattr(access, "kind", None)
            reason = "not signed in" if kind == "proxy" else (getattr(access, "reason", "") or "no model")
            return f"none ({reason}). /login, or /keys set <NAME>."
        if access is not None and access.kind == "proxy":
            return f"{self.model_label()}, paid from your Robutler credits"
        # A local model says where it is reached (plan item 2.8).
        local = access.local_route() if access is not None and hasattr(access, "local_route") else None
        if local:
            return local
        env_var = self._provider_env_var()
        return f"{self.model_label()}, with your {env_var}" if env_var else self.model_label()

    async def cmd_status(self) -> None:
        from ..config_store import platform_url
        from ..credentials import get_token

        p = self.theme.palette
        portal = platform_url().rstrip("/")
        host = re.sub(r"^https?://", "", portal)
        token = get_token()
        account = "Not signed in. /login signs in."
        if token:
            who = await self._who_am_i(portal, token)
            if who == "expired":
                account = f"Sign-in expired on {host}. /login signs in again."
            elif who:
                account = f"@{who} on {host}"
            else:
                account = f"Signed in on {host}"
        from ..config_store import profile_name
        from .render import compact_number

        tokens = self.input_tokens + self.output_tokens
        file = self.built.file if self.built else None
        profile = profile_name()
        # Where the sign-in and keys live and whether the next use makes macOS
        # ask: doctor's `keychain` line (keychain-ux, 2026-09-27).
        keychain = self._keychain_row()
        rows = [
            ("Account", account),
            *([("Profile", profile)] if profile else []),
            *([("Keychain", keychain)] if keychain else []),
            ("Agent", f"{self.agent_name} ({file.name})" if file else f"{self.agent_name} (built in)"),
            ("Model", self.model_route()),
            ("Tool rounds", self.rounds_status()),
            ("Sandbox", self.sandbox_summary()[1]),
            ("Folder", _short_path(self.agent_folder())),
            (
                "Conversation",
                f"{spoken_count(self.messages)} messages{f', {compact_number(tokens)} tokens' if tokens else ''}"
                + (f", {cost_words(self.cost.credits, self.cost.estimated)}" if self.cost.known else "")
                + (", also on Robutler" if self.platform_chat_id else ""),
            ),
            ("Robutler", self._robutler_row(bool(token))),
        ]
        from rich.table import Table

        width = max(len(label) for label, _ in rows) + 3
        grid = Table.grid(padding=0)
        grid.add_column(width=width + 2, no_wrap=True)
        grid.add_column(ratio=1, overflow="fold")
        for label, value in rows:
            grid.add_row(Text("  " + label, style=p.muted), Text(value, style=p.text))
        self.console.print()
        self.console.print(Text("Status", style=f"bold {p.text}"))
        # One column short of the edge, where the TypeScript chat wraps.
        self.console.print(grid, width=max(40, self.console.width - 1))
        self.console.print()

    @staticmethod
    def _keychain_row() -> str:
        """The `Keychain` row of /status: the detail of doctor's `keychain`
        line, found without reading a value. Empty when it cannot be told."""
        try:
            from webagents.agents.skills.local.secrets.keychain_ux import doctor_line

            from ..doctor import keychain_facts

            return str(doctor_line(keychain_facts())["detail"])
        except Exception:  # noqa: BLE001 - a status row, never a failure
            return ""

    def _left_behind_notices(self, items: List[Dict[str, str]]) -> None:
        """An old `webagents:` item only a macOS dialog could remove, named
        with where to remove it (keychain-ux, 2026-09-27)."""
        from ..account import left_behind_lines

        for line in left_behind_lines(items):
            self.notice("info", line)

    def _robutler_row(self, signed_in: bool) -> str:
        """The `Robutler` row of /status (spec 3.6): published as what, or what stands in the way."""
        if self.built is None or self.built.file is None:
            return CHAT_WORDS["statusBuiltIn"]
        from ..publish import project_link

        link = project_link(self.agent_folder())
        if linked_platform_agent(self.agent_folder(), self.agent_name):
            return fill("statusPublished", agentName=link.get("agent_name") or self.agent_name)
        return CHAT_WORDS["statusNotPublished"] if signed_in else CHAT_WORDS["statusSignedOut"]

    async def cmd_logout(self) -> None:
        from ..config_store import platform_url
        from ..credentials import TOKEN_ENV_VAR, get_token
        from ..platform.auth import logout

        from webagents.agents.skills.local.secrets.keychain_ux import KeychainDialogBlocked

        from ..credentials import left_behind

        host = re.sub(r"^https?://", "", platform_url().rstrip("/"))
        if not get_token():
            self.notice("info", "Not signed in.")
            return
        try:
            logout()
        except KeychainDialogBlocked as blocked:
            self.notice("error", str(blocked))
            return
        await self.initialize()
        if os.environ.get(TOKEN_ENV_VAR):
            self.notice("warn", f"{TOKEN_ENV_VAR} is set in this shell, and it keeps you signed in.", "Unset it to sign out completely.")
            return
        self.notice("ok", f"Signed out of {host}.", self.model_problem or f"{self.agent_name} runs on {self.model_label()}.")
        self._left_behind_notices(left_behind())

    def _key_names(self) -> List[str]:
        from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

        return [p.env_vars[0] for p in LLM_PROVIDERS if p.credential == "api_key" and p.env_vars]

    async def cmd_keys(self, args: str) -> None:
        from ..commands.secrets import _store

        p = self.theme.palette
        parts = args.split()
        verb = parts[0] if parts else ""
        name = parts[1].upper() if len(parts) > 1 else ""
        known = self._key_names()
        if not verb:
            try:
                store = _store(quiet=True)
                backend = "stored in your keychain" if store.keystore else "stored in an owner-only file"
            except Exception:  # noqa: BLE001 - the label is a nicety
                store, backend = None, "stored"
            width = max(len(k) for k in known) + 3
            lines = [Text("Model provider keys", style=f"bold {p.text}")]
            for key in known:
                stored = None
                if store is not None:
                    try:
                        stored = store.get(key)
                    except Exception:  # noqa: BLE001
                        stored = None
                exported = os.environ.get(key)
                # Stored keys are loaded into this process's environment, so a
                # value equal to the stored one is the stored one.
                where = backend if stored and (not exported or exported == stored) else "set in this shell" if exported else "not set"
                mark = ("○", p.faint) if where == "not set" else ("●", p.success)
                lines.append(Text.assemble("  ", mark, " ", (key.ljust(width), p.text), (where, p.faint if where == "not set" else p.muted)))
            lines.append(Text(""))
            lines.append(Text("  /keys set <NAME> stores one; /keys unset <NAME> removes a stored one.", style=p.faint))
            self._print_lines(lines)
            return
        if verb not in ("set", "unset") or not name:
            self.notice("error", "Usage: /keys [set|unset NAME]", f"NAME is one of {', '.join(known)}.")
            return
        if name not in known:
            self.notice("error", f"{name} is not a model provider key.", f"One of {', '.join(known)}.")
            return
        store = _store(quiet=True)
        if verb == "set":
            value = (await asyncio.to_thread(getpass.getpass, f"  {name} (hidden): ")).strip()
            if not value:
                self.notice("info", "Nothing entered; nothing stored.")
                return
            try:
                backend = store.set(name, value)
            except Exception as error:  # noqa: BLE001
                self.notice("error", f"Could not store {name}: {error}")
                return
            exported = os.environ.get(name)
            os.environ[name] = value if not exported else exported
            await self.initialize()
            self.notice(
                "ok",
                f"Stored {name} ({'your keychain' if backend == 'keystore' else 'an owner-only file'}).",
                f"{name} is also set in this shell, which wins." if exported else self.model_problem or f"{self.agent_name} runs on {self.model_label()}.",
            )
            return
        try:
            stored = store.get(name)
            removed = store.delete(name)
        except Exception as error:  # noqa: BLE001
            self.notice("error", f"Could not remove {name}: {error}")
            return
        exported = os.environ.get(name)
        if stored and exported == stored:
            os.environ.pop(name, None)
            exported = None
        await self.initialize()
        if not removed:
            self.notice("info", f"{name} was not stored.", "It is set in this shell; unset it there." if exported else None)
            return
        self.notice("ok", f"Removed {name}.", "It is still set in this shell." if exported else self.model_problem)
        self._left_behind_notices(store.left_behind)

    async def cmd_secrets(self, args: str) -> None:
        """`/secrets`: the names in the store an MCP server's `${secret:NAME}`
        reads (S-292), and `set`/`remove` to change them. Handled like `/keys
        set`: the value is taken at a hidden local prompt, goes to the store
        and nowhere else, never to the model. The words are pinned by
        `tests/fixtures/cli/secrets.json` (`chat`)."""
        from webagents.agents.skills.local.secrets.references import REFERENCE_NAME

        from ..commands.secrets import _store

        p = self.theme.palette
        parts = args.split()
        verb = parts[0] if parts else ""
        name = parts[1] if len(parts) > 1 else ""
        if not verb:
            names: List[str] = []
            complete = True
            where = "stored"
            try:
                store = _store(quiet=True)
                names, complete = store.list()
                where = "stored in your keychain" if store.keystore else "stored in an owner-only file"
            except Exception:  # noqa: BLE001 - the listing is a nicety; the hint still says how to add one
                pass
            lines = [Text("Secrets stored on this machine", style=f"bold {p.text}")]
            if not names:
                lines.append(Text("  (none)", style=p.faint))
            width = max([len(n) for n in names] + [0]) + 3
            for stored in names:
                lines.append(Text.assemble("  ", ("●", p.success), " ", (stored.ljust(width), p.text), (where, p.muted)))
            if not complete:
                lines.append(Text("  An OS keychain cannot be listed, so secrets other tools wrote there do not appear.", style=p.faint))
            lines.append(Text(""))
            lines.append(Text("  /secrets set <NAME> stores one; /secrets remove <NAME> removes it. An MCP server uses one as ${secret:NAME} in its env, headers or url.", style=p.faint))
            self._print_lines(lines)
            return
        if verb not in ("set", "remove") or not name:
            self.notice("error", "Usage: /secrets [set|remove NAME]")
            return
        if not REFERENCE_NAME.match(name):
            self.notice("error", f"{name} does not look like an environment variable name.")
            return
        store = _store(quiet=True)
        if verb == "set":
            value = (await asyncio.to_thread(getpass.getpass, f"  {name} (hidden): ")).strip()
            if not value:
                self.notice("info", "Nothing entered; nothing stored.")
                return
            try:
                backend = store.set(name, value)
            except Exception as error:  # noqa: BLE001
                self.notice("error", f"Could not store {name}: {error}")
                return
            # A provider key stored this way reaches the model as `/keys set`
            # would; any other name stays out of the environment (S-292) and
            # is read by the MCP skill at connect time, so the rebuild below
            # connects a server that named it.
            if name in self._key_names() and not os.environ.get(name):
                os.environ[name] = value
            await self.initialize()
            self.notice(
                "ok",
                f"Stored {name} ({'your keychain' if backend == 'keystore' else 'an owner-only file'}).",
                f"An MCP server uses it as ${{secret:{name}}} in its env, headers or url.",
            )
            return
        try:
            stored = store.get(name)
            removed = store.delete(name)
        except Exception as error:  # noqa: BLE001
            self.notice("error", f"Could not remove {name}: {error}")
            return
        if stored and os.environ.get(name) == stored:
            os.environ.pop(name, None)
        await self.initialize()
        if not removed:
            self.notice("info", f"{name} was not stored.")
            return
        self.notice("ok", f"Removed {name}.")
        self._left_behind_notices(store.left_behind)

    def _attach_host_asker(self) -> None:
        """Ask on first use (the sandbox-default lane, 2026-09-27; the
        TypeScript chat's `attachHostAsker`): the shell asks the person at the
        terminal about a host a confined command was refused, by name, read
        from srt's own proxy log (`shell/skill.py`, `asker`). `once` re-runs
        the command with the host for that run; `always` writes it into the
        agent file's `network.hosts` (`sandbox_default_hosts.py`), keeps it
        for this session and re-runs; anything else returns the output with
        the hint. Only the interactive chat attaches this: `-p`, `serve` and
        the daemon refuse with the hint, and the shell asks for the owner's
        commands only."""
        skills = self.built.agent.skills if self.built else {}
        shell = skills.get("shell")
        if shell is None:
            return
        shell.asker = _HostAsker(self, shell) if self.interactive else None

    def _sandbox_detail(self, policy: Any) -> str:
        """The effective folders, hosts and switches of a confined policy, one line (`sandboxDetail`)."""
        from pathlib import Path as _Path

        def folders(roots: List[str]) -> str:
            return ", ".join(_short_path(_Path(root)) for root in roots if root != getattr(policy, "scratch", None))

        return fill(
            "sandboxDetail",
            writes=f"{folders(policy.write_roots) or 'none'} and a scratch folder",
            reads=f"{folders(policy.read_roots) or 'none'} and the system folders" if policy.scoped_reads else "all but credential folders and .env",
            hosts=", ".join(policy.network_domains) or "none",
            local="on" if getattr(policy, "local_network", False) else "off",
            sockets=", ".join(getattr(policy, "unix_sockets", None) or []) or "none",
            env=", ".join(policy.env_passthrough) or "none",
        )

    def sandbox_summary(self) -> Tuple[str, str, Optional[str]]:
        """(kind, headline, detail): what the agent's commands may do, for
        /sandbox and /status (the TypeScript chat's `sandboxSummary`). The
        headline is the STATE, preset and origin (`development (default)`,
        `off (agent file)`, `off (--no-sandbox)`), since the sandbox is on by
        default (2026-09-27); the detail lists the effective folders, hosts
        and switches."""
        from webagents.sandbox import backend_status, sandbox_state
        from webagents.sandbox.srt import FIX_SETUP_POINTER

        skills = self.built.agent.skills if self.built else {}
        shell = skills.get("shell")
        if shell is None:
            # SKILL.md scripts are confined commands too.
            scripts = skills.get("agent_skills")
            script_state = scripts.script_state() if scripts is not None and hasattr(scripts, "script_state") else None
            if not script_state:
                return ("info", "Not needed: this agent cannot run commands.", None)
            status = backend_status()
            if not status.get("available"):
                return (
                    "warn",
                    fill("sandboxUnavailable", state=script_state, reason=status.get("reason") or "no sandbox backend here"),
                    fill("sandboxFix", fix=status.get("fix") or FIX_SETUP_POINTER),
                )
            return ("ok", fill("sandboxScripts", state=script_state), None)
        # A declaration that did not resolve refuses every command (D6,
        # 2026-09-26): the shell keeps the reason, and this used to say "Off".
        sandbox_error = getattr(shell, "sandbox_error", None) or getattr(shell, "_sandbox_error", None)
        if sandbox_error:
            return ("warn", f"Invalid: {sandbox_error}", "Every command is refused until the declaration is fixed.")
        policy = getattr(shell, "policy", None)
        origin = getattr(shell, "sandbox_origin", None) or "default"
        state = sandbox_state(policy, origin)
        if policy is None or not getattr(policy, "confined", True):
            return ("warn", fill("sandboxOff", state=state), CHAT_WORDS["sandboxFlagFix" if origin == "--no-sandbox" else "sandboxOffFix"])
        # The engine, as `doctor` checks it (G9, 2026-09-26): a policy srt
        # cannot enforce here refuses every command, and the state alone would be a lie.
        status = backend_status()
        if not status.get("available"):
            return (
                "warn",
                fill("sandboxUnavailable", state=state, reason=status.get("reason") or "no sandbox backend here"),
                fill("sandboxFix", fix=status.get("fix") or FIX_SETUP_POINTER),
            )
        return ("ok", state, self._sandbox_detail(policy))

    def cmd_sandbox(self) -> None:
        kind, headline, detail = self.sandbox_summary()
        self.notice(kind, f"Sandbox: {headline}", detail)

    async def cmd_publish(self, args: str) -> None:
        """`/publish`: send this folder's agent to Robutler, or update the
        linked one after asking (owner decision 6); `/publish --dry-run`: show
        what would be sent and send nothing (spec 3.6)."""
        from ..publish import PublishIO, publish_agent

        parts = args.split()
        if any(p != "--dry-run" for p in parts):
            self.notice("error", fill("usage", usage="/publish [--dry-run]"))
            return
        dry_run = "--dry-run" in parts
        p = self.theme.palette
        file = self.built.file if self.built else None
        if file is None:
            self.notice("warn", "Publishing needs an AGENT.md in this folder.", "Create one with `webagents init`, then /publish.")
            return

        async def confirm(question: str) -> bool:
            return await self._confirm(f"{question} [y/N] ")

        self.console.print()
        result = await publish_agent(
            file,
            PublishIO(
                ok=lambda line: self.notice("ok", line),
                print=lambda line: self.console.print(Text(f"  {line}", style=p.muted)),
                error=lambda line: self.notice("error", line),
                confirm=confirm,
                confirm_update=self._confirm,
            ),
            dry_run=dry_run,
        )
        if result.ok:
            self.console.print()

    # -- signing in and keys, when there is no model -----------------------------

    async def sign_in(self) -> None:
        """Sign in through the browser, then rebuild the agent: with no key of its
        own it now runs on Robutler's models. /login and the offer both come here."""
        from ..config_store import platform_url
        from ..platform.auth import login

        p = self.theme.palette
        host = re.sub(r"^https?://", "", platform_url().rstrip("/"))
        self.console.print()
        try:
            result = await login(say=lambda line: self.console.print(Text(f"  {line}", style=p.muted)))
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - said, and the chat goes on
            message = str(error) or "Sign-in cancelled."
            self.notice("error", f"Could not sign in: {message}")
            return
        await self.initialize()
        username = (result or {}).get("username") or ""
        who = f" as @{username}" if username else ""
        if self.model_problem:
            self.notice("warn", f"Signed in{who} on {host}.", self.model_problem)
        else:
            self.notice("ok", f"Signed in{who} on {host}.", f"{self.agent_name} runs on {self.model_label()}.")

    def _key_candidates(self) -> List[Any]:
        """The providers whose key would give this agent a model, for the offer."""
        from webagents.agents.skills.core.llm.providers import LLM_PROVIDERS

        takes_key = [p for p in LLM_PROVIDERS if p.credential == "api_key" and p.env_vars]
        access = self.built.access if self.built else None
        named = getattr(access, "provider", None)
        if named is not None:
            return [p for p in takes_key if p.id == named.id]
        if (getattr(access, "reason", "") or "").endswith("is served by Robutler"):
            return []
        return takes_key

    async def _ask(self, question: str) -> Optional[str]:
        """One visible line, or None on Ctrl+C or Ctrl+D (as the TypeScript
        chat's `promptLine` answers).

        NOT A THREAD (2026-09-26, the e2e run). `input()` used to run in a
        worker thread, and Ctrl+C at the first-run "Choose 1-3" prompt
        reached the main thread, which could not interrupt the thread's read:
        the chat echoed ^C and sat until enter, then died with exit 130. On
        POSIX the line is read on the event loop instead (the terminal, in
        cooked mode, delivers it whole on enter; a pipe delivers chunks), and
        SIGINT is a loop handler that answers None at once. Elsewhere the
        thread stays, and Ctrl+C stays what it was.
        """
        import signal

        sys.stdout.write(question)
        sys.stdout.flush()
        loop = asyncio.get_running_loop()
        try:
            fd = sys.stdin.fileno()
            if os.name != "posix":
                raise OSError("no reader")
            future: "asyncio.Future[Optional[str]]" = loop.create_future()
            buffer = bytearray()

            def readable() -> None:
                try:
                    data = os.read(fd, 4096)
                except OSError as error:
                    if not future.done():
                        future.set_exception(error)
                    return
                if not data:
                    # Ctrl+D or the pipe's end: what was typed so far, else nothing.
                    if not future.done():
                        future.set_result(buffer.decode("utf-8", "replace") if buffer else None)
                    return
                buffer.extend(data)
                if b"\n" in buffer:
                    line, _, _rest = bytes(buffer).partition(b"\n")
                    if not future.done():
                        future.set_result(line.decode("utf-8", "replace").rstrip("\r"))

            def interrupt() -> None:
                if not future.done():
                    future.set_result(None)

            loop.add_reader(fd, readable)
            try:
                loop.add_signal_handler(signal.SIGINT, interrupt)
                installed = True
            except (NotImplementedError, RuntimeError, ValueError):
                installed = False
            try:
                answer = await future
            finally:
                loop.remove_reader(fd)
                if installed:
                    loop.remove_signal_handler(signal.SIGINT)
                    # A QUESTION MID-TURN HANDS CTRL+C BACK TO THE TURN (the
                    # ptypass-fixes lane, 2026-09-27). Removing the handler
                    # left none: a Ctrl+C later in the same turn ended it
                    # without the interrupt, and the chat said "The model
                    # returned no answer" instead of "Interrupted" (the PTY
                    # pass, `14` against `15`).
                    if self._turn_interrupt is not None:
                        loop.add_signal_handler(signal.SIGINT, self._turn_interrupt)
            if answer is None:
                self.console.print()
            return answer
        except (OSError, ValueError, AttributeError):
            pass
        try:
            return await asyncio.to_thread(input)
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return None

    async def offer_model_access(self) -> None:
        """No model to run on: ask once, before the chat opens.

        The ways out, there and then: sign in to Robutler (first, since it needs
        nothing the person has to go and find), type a provider key (kept for
        next time, in the store both CLIs read), or carry on without a model.
        """
        p = self.theme.palette
        choices: List[Tuple[str, Callable[[], Any]]] = [
            ("Sign in to Robutler and use its models, paid from your credits", self.sign_in),
        ]
        candidates = self._key_candidates()
        if candidates:
            which = candidates[0].env_vars[0] if len(candidates) == 1 else "a provider key"
            choices.append((f"Enter {which}, kept for next time", lambda: self._enter_key(candidates)))
        choices.append(("Continue without a model", None))

        self.notice("warn", self.model_problem or "")
        for index, (label, _action) in enumerate(choices):
            self.console.print(Text.assemble("  ", (str(index + 1), p.accent), "  ", (label, p.text)))
        self.console.print()
        answer = await self._ask(f"  Choose 1-{len(choices)} [1]: ")
        if answer is None:
            # Ctrl+C (or Ctrl+D) at the offer cancels the offer, not the chat,
            # and says so; the TypeScript chat does the same (2026-09-26).
            self.notice("info", CHAT_WORDS["continuingWithoutModel"])
            return
        answer = answer.strip()
        index = 0 if not answer else int(answer) - 1 if answer.isdigit() else -1
        if not 0 <= index < len(choices):
            self.notice("info", "Continuing without a model.")
            return
        action = choices[index][1]
        if action is not None:
            await action()

    async def _enter_key(self, candidates: List[Any]) -> None:
        from ..commands.secrets import _store

        p = self.theme.palette
        provider = candidates[0]
        if len(candidates) > 1:
            self.console.print()
            for index, candidate in enumerate(candidates):
                self.console.print(
                    Text.assemble("  ", (str(index + 1), p.accent), "  ", (candidate.id, p.text), " ", (candidate.env_vars[0], p.faint))
                )
            self.console.print()
            answer = await self._ask(f"  Which provider? 1-{len(candidates)} [1]: ")
            if answer is None:
                return
            answer = answer.strip()
            index = 0 if not answer else int(answer) - 1 if answer.isdigit() else -1
            if not 0 <= index < len(candidates):
                self.notice("info", "No provider chosen; continuing without a model.")
                return
            provider = candidates[index]
        env_var = provider.env_vars[0]
        try:
            value = (await asyncio.to_thread(getpass.getpass, f"  {env_var} (hidden): ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not value:
            self.notice("info", "Nothing entered; continuing without a model.")
            return
        os.environ[env_var] = value
        try:
            backend = _store(quiet=True).set(env_var, value)
            kept = "Kept in your OS keystore" if backend == "keystore" else "Kept in an owner-only file"
            kept += f"; `webagents secrets remove {env_var}` removes it."
        except Exception as error:  # noqa: BLE001
            kept = f"Set for this session only; it could not be kept: {error}"
        await self.initialize()
        if self.model_problem:
            self.notice("warn", self.model_problem, kept)
        else:
            self.notice("ok", f"{self.agent_name} runs on {self.model_label()}.", kept)

    # -- a turn -----------------------------------------------------------------

    def _explain_failure(self, message: str) -> "FailureText":
        """The headline and hint for a failed turn (`failures.py`, the same rules
        and cases as the TypeScript chat): Robutler's own refusals when the turn
        ran on its models, else the provider's words and `render.error_hint`."""
        from ..model_access import platform_llm_url
        from .failures import present_failure
        from .render import error_hint

        access = self.built.access if self.built else None
        on_robutler = access is not None and getattr(access, "kind", None) == "proxy"
        return present_failure(
            message,
            proxy_url=platform_llm_url() if on_robutler else None,
            generic_hint=lambda text: error_hint(self.model_label(), text),
            model=self.cost_model(),
        )

    async def handle_input(self, user_input: str) -> None:
        text = user_input.strip()
        if not text:
            return
        if text.startswith("/"):
            # Only a typed line reaches here (spec W1): the model, a tool, an
            # `@file` and a resumed transcript never do.
            name, _, args = text[1:].partition(" ")
            spec = chat_command(name)
            handler = self._handlers.get(name.lower()) if spec is not None else None
            if spec is None or handler is None:
                self._say_unknown_command(name)
                return
            # W8: a command whose usage names no argument refuses one.
            if args.strip() and takes_no_arguments(spec):
                self.notice("error", fill("usage", usage=spec.usage))
                return
            # W7: a handler's exception becomes `✗ {message}`, the chat goes on.
            try:
                result = handler(args.strip())
                if asyncio.iscoroutine(result):
                    await result
            except Exception as error:  # noqa: BLE001 - shown; a failed rebuild keeps the running agent
                self.notice("error", str(error))
            return
        if self.model_problem:
            self.notice("warn", self.model_problem, "Type /login to sign in without leaving the chat.")
            return
        message = self._expand_file_references(text)
        # The file as it is when the turn starts, so a change made while the
        # chat sat idle at the prompt is not called one made "during the last
        # reply" (2026-09-26): only a version that appears between here and
        # the end of the turn is the agent's own doing.
        before_turn = self._file_changed_now()
        try:
            # The /undo snapshot is taken inside the turn, once its spinner is up.
            await self._turn(message, snapshot=True)
        finally:
            self.save_conversation()
            self.record_on_robutler()
            # If the agent rewrote its own file during the reply, the next
            # before-prompt notice says so with `▲` rather than `✦` (S-283).
            after_turn = self._file_changed_now()
            if after_turn is not None and (before_turn is None or before_turn.sha != after_turn.sha):
                self._changed_during_reply = True

    async def _turn(self, message: str, snapshot: bool = False) -> None:
        """One reply, streamed by `render.TurnRenderer`, stoppable with Esc or Ctrl+C.
        ``snapshot`` takes the /undo snapshot first, with the spinner already up."""
        import signal

        from .render import TurnRenderer, error_lines, events_from_chunk

        self.messages.append({"role": "user", "content": message})
        self.console.print()
        renderer = TurnRenderer(self.console, theme=self.theme, explain_error=self._explain_failure)
        self._turn_renderer = renderer
        # The turn's tool rounds, kept with it so the next message does not make
        # the model list and read everything again (`turn_history.py`).
        from ..turn_history import TurnRecorder, history_for_model

        recorder = TurnRecorder()
        from rich.live import Live

        async def stream() -> None:
            # The person at the terminal is the agent's owner (`access.caller`).
            from webagents.access import run_as_local_owner

            run_as_local_owner(self.built.agent)
            if not self.streaming:
                # `--no-streaming`: the reply appears whole, when it is done.
                if snapshot:
                    await self.snapshot_before_turn_async(message)
                async for chunk in self.built.agent.run_streaming(history_for_model(self.messages)):
                    if isinstance(chunk, dict):
                        for event in events_from_chunk(chunk):
                            recorder.observe(event)
                            renderer.feed(event)
                renderer.flush(final=True)
                return
            with Live(console=self.console, refresh_per_second=12.5, transient=True, get_renderable=renderer.live_view) as live:
                # Kept where a mid-turn question can pause it (`_confirm_control_write`).
                self._turn_live = live
                try:
                    if snapshot:
                        await self.snapshot_before_turn_async(message)
                    async for chunk in self.built.agent.run_streaming(history_for_model(self.messages)):
                        if isinstance(chunk, dict):
                            for event in events_from_chunk(chunk):
                                recorder.observe(event)
                                renderer.feed(event)
                            renderer.flush()
                    renderer.flush(final=True)
                finally:
                    self._turn_live = None

        # CTRL+C STOPS THE REPLY, NOT THE CHAT: SIGINT cancels only the stream;
        # what already arrived stays on screen and the prompt comes back.
        task = asyncio.ensure_future(stream())
        interrupted = False
        failed = False

        def interrupt() -> None:
            nonlocal interrupted
            interrupted = True
            task.cancel()

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, interrupt)
            installed = True
            self._turn_interrupt = interrupt
        except (NotImplementedError, RuntimeError, ValueError):
            installed = False
        # The Esc listener, on a stack a mid-turn question can close and
        # re-enter (`_confirm_control_write`): the terminal must be cooked to
        # read its answer as a line.
        from contextlib import ExitStack

        keys = ExitStack()
        keys.enter_context(stream_keys(interrupt, loop))
        self._turn_keys = keys
        self._turn_reenter_keys = lambda: keys.enter_context(stream_keys(interrupt, loop))
        try:
            await task
        except asyncio.CancelledError:
            if not interrupted:
                raise
        except Exception as error:  # noqa: BLE001 - shown in the turn, and the chat goes on
            from webagents.utils.errors import describe_exception

            detail = describe_exception(error)
            failed = True
            explained = self._explain_failure(detail)
            self.console.print()
            for line in error_lines(self.theme, explained.headline, explained.hint, self.console.width):
                self.console.print(line)
        finally:
            keys.close()
            self._turn_keys = None
            self._turn_reenter_keys = None
            self._turn_interrupt = None
            self._turn_renderer = None
            if installed:
                loop.remove_signal_handler(signal.SIGINT)
        if interrupted:
            renderer.flush(final=True)
            p = self.theme.palette
            self.console.print(Text.assemble(("  ⎿  ", p.faint), ("Interrupted", p.warning)))
        answer = renderer.plain_text()
        failed = failed or renderer.failed
        finish = renderer.finish
        # A turn the agent stopped for repeating one tool call says so, answer
        # or not (2026-09-28, `core/tool_budget.py`); a turn that spent its
        # rounds is said below, by the question the chat asks.
        looped = finish is not None and finish.reason == "tool_loop"
        if (not answer.strip() or looped) and not failed and not interrupted:
            # NOTHING SAID IS SAID (2026-09-27): a turn that ended with no
            # text and no error drew nothing, or the agent's canned apology.
            # One truthful line names the provider's reason
            # (`failures.present_empty_reply`, the TypeScript chat's twin).
            from .failures import present_empty_reply

            explained = present_empty_reply(
                finish.reason if finish else None,
                blocked=bool(finish and finish.blocked),
                retried=bool(finish and finish.retried),
                thinking=renderer.saw_thinking(),
                rounds=finish.rounds if finish else None,
                tool=finish.tool if finish else None,
            )
            self.console.print()
            for line in error_lines(self.theme, explained.headline, explained.hint, self.console.width):
                self.console.print(line)
        stats = renderer.stats()
        if stats is not None:
            self.console.print()
            self.console.print(stats)
        # A REPLY IS SOMETHING SAID (2026-09-25, narrowed 2026-09-27): the
        # session's last line counted every turn, so a chat whose only message
        # was refused ended "1 reply", and then every turn that ended quietly,
        # so a chat of seven empty completions said "7 replies". A turn counts
        # only when it said something; the TypeScript chat counts the same way.
        if answer.strip():
            self.turns += 1
        self.input_tokens += renderer.usage.prompt_tokens
        self.output_tokens += renderer.usage.completion_tokens
        self.session_tokens += renderer.usage.prompt_tokens + renderer.usage.completion_tokens
        # The turn's cost: reported by the platform, else estimated for the
        # model that answered (plan item 2.4), only when the turn ran through
        # Robutler (`turn_cost_model`, 2026-09-29): a key's turn shows tokens alone.
        usage = renderer.usage
        if usage.prompt_tokens or usage.completion_tokens or usage.cost_credits is not None:
            model = self.turn_cost_model()
            self.cost = add_turn_cost(self.cost, model, usage.prompt_tokens, usage.completion_tokens, usage.cost_credits)
            self.session_cost = add_turn_cost(self.session_cost, model, usage.prompt_tokens, usage.completion_tokens, usage.cost_credits)

        if answer:
            self.messages.extend(recorder.messages())
            self.messages.append({"role": "assistant", "content": answer})
        elif self.messages and self.messages[-1].get("role") == "user":
            # A turn that said nothing leaves no trace, so the next message is
            # not sent after an unanswered one.
            self.messages.pop()
        self.console.print()
        # THE CAP ASKS (2026-09-28, the owner): a turn that spent its tool
        # rounds ends with the answer its last, tool-less call gave, and the
        # interactive chat offers another budget. Yes continues from the
        # conversation with a fresh budget (a turn that brought no answer is
        # sent again); no, or anything else, ends the turn. Only the
        # interactive chat asks: `-p`, `serve`, the daemon and ACP end with
        # the answer and the finish reason.
        if finish is not None and finish.reason == "tool_round_limit" and not failed and not interrupted:
            if await self._keep_going(finish.rounds):
                from webagents.agents.core.tool_budget import CONTINUE_MESSAGE

                await self._turn(CONTINUE_MESSAGE if answer else message)
            elif not self._at_terminal() and answer.strip():
                from webagents.agents.core.tool_budget import tool_round_limit_sentence

                self.notice("warn", tool_round_limit_sentence(finish.rounds, answered=True))

    async def _keep_going(self, rounds: Optional[int]) -> bool:
        """The chat's question after a turn that spent its tool rounds
        (`core/tool_budget.py`, `continue_question`): yes is the default, and
        Ctrl+C, Ctrl+D or Esc answer no. Asked only at a terminal."""
        if not self._at_terminal():
            return False
        from webagents.agents.core.tool_budget import continue_question

        used = rounds if rounds is not None else getattr(self.built.agent, "max_tool_iterations", 0)
        answer = await self._ask(f"  {continue_question(used)} ")
        if answer is None:
            return False
        return answer.strip().lower() in ("", "y", "yes")

    def _expand_file_references(self, text: str) -> str:
        """`@path/to/file` includes that file, when it exists; anything else is left as typed."""

        def replace(match: "re.Match[str]") -> str:
            ref = match.group(1)
            path = Path(ref).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                try:
                    return f'\n\n<file path="{ref}">\n{path.read_text()}\n</file>\n\n'
                except (OSError, UnicodeDecodeError) as error:
                    return f"@{ref} (could not read it: {error})"
            return match.group(0)

        return re.sub(r"(?<![\w.])@([A-Za-z0-9_\-./~]+)", replace, text)

    # -- the loop -----------------------------------------------------------------

    def _goodbye(self) -> None:
        from .render import duration

        if not self.turns:
            self.console.print()
            return
        # This chat's own replies and tokens: a resumed conversation's earlier
        # turns are its own, not this chat's (2026-09-26).
        tokens = self.session_tokens
        parts = [f"{self.turns} {'reply' if self.turns == 1 else 'replies'}"]
        if tokens:
            parts.append(f"{tokens:,} tokens")
        if self.session_cost.known:
            parts.append(cost_words(self.session_cost.credits, self.session_cost.estimated))
        parts.append(duration(time.time() - self.session_started))
        self.console.print(Text("✦ " + " · ".join(parts), style=self.theme.palette.faint))
        self.console.print()

    async def run(self) -> None:
        """At a terminal: the wordmark, the offer when there is no model, the card,
        then the box for every message. Anything else (a pipe, a script) gets a
        plain prompt and plain output."""
        # The interactive loop, the one place a control-file write may ask
        # (S-314), and the one place a refused host is asked about
        # (`_attach_host_asker`).
        self._chatting = True
        await self.initialize()
        tty = sys.stdin.isatty() and sys.stdout.isatty()
        stop_recording: Optional[Callable[[], None]] = None
        if tty:
            # Everything the chat writes from here on, recorded (`ui/screen.py`)
            # so the `/` menu can open over the conversation instead of
            # scrolling it, and tied to the screen by where the cursor starts.
            screen, stop_recording = record_screen(lambda: _terminal_size().columns, lambda: _terminal_size().lines)
            start_row = query_cursor_row()
            if start_row is not None:
                screen.anchor(start_row)
            self.prompt_box.screen = screen
            self.prompt_box.console = self.console
            background = query_background()
            if background:
                self.theme = theme_for(self.console, background=background)
                self.console.pop_theme()
                self.console.push_theme(RichTheme(markdown_styles(self.theme)))
                self.prompt_box.theme = self.theme
            play_wordmark(self.console, self.theme)
            # Before the card, so the card shows what the agent will run on.
            if self.model_problem:
                await self.offer_model_access()
            self.console.print()
            self.print_card()
            self._say_new_agent_tip()
        else:
            print(f"\nWebAgents CLI - Connected to {self.agent_name}")
            if self.model_problem:
                print(self.model_problem)
            print("Type /help for available commands, or start chatting.\n")
            self._say_new_agent_tip()

        while self.running:
            try:
                self.say_recording_problem()
                # The agent's file may have changed since the chat loaded it
                # (S-283): say so, once per version, and never reload unasked.
                self._say_file_changed()
                if tty:
                    await self._refresh_completion_data()
                    line = await self.prompt_box.ask(f"Message {self.agent_name}, or type / for commands")
                else:
                    line = await self._ask("> ")
                if line is None:
                    break
                if line.strip() and tty:
                    # The sent line only: what follows brings its own leading
                    # blank line (a notice, /status, a reply), as in the
                    # TypeScript chat, where this printed a second one.
                    for sent in sent_message(self.theme, self.console.width, line):
                        self.console.print(sent)
                await self.handle_input(line)
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
        # The last turn's recording, before the process ends with it.
        await self.settle_recording()
        self.say_recording_problem()
        self._goodbye()
        if stop_recording is not None:
            stop_recording()
        # The agent's skills closed, MCP servers included, by the task that
        # holds them, so `asyncio.run` ends cleanly (2026-09-26).
        if self.built is not None:
            await self._cleanup_agent(self.built)


def _terminal_size() -> os.terminal_size:
    """The terminal's real size (the screen record's), whatever COLUMNS says."""
    try:
        return os.get_terminal_size(sys.__stdout__.fileno())
    except (AttributeError, OSError, ValueError):
        return os.terminal_size((80, 24))


def repl_log_path() -> Path:
    """`logs/repl.log` in the profile's folder (`~/.webagents-<profile>` under
    `--profile`, 2026-09-27). The log went to `~/.webagents/logs` whatever the
    profile, beside the wrong config. An agent's signing keys stay in
    `~/.webagents/keys` on purpose: the key is the agent's identity on this
    machine, not a platform setting (`docs/cli/configuration.md`)."""
    from ..config_store import global_dir

    return global_dir() / "logs" / "repl.log"


def start_repl(
    agent_path: Optional[Path] = None,
    model: Optional[str] = None,
    streaming: bool = True,
    chosen: bool = False,
) -> None:
    """Open the chat with the agent at `agent_path`.

    None finds this folder's agent, or the built-in one; with `chosen` (`-a`),
    `agent_path` is final and None means the built-in agent.
    """
    from webagents.utils.logging import setup_logging

    log_file = repl_log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(log_file), console_output=False)

    session = WebAgentsSession(agent_path=agent_path, model=model, streaming=streaming, chosen=chosen)
    asyncio.run(session.run())
