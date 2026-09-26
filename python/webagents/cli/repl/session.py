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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.text import Text
from rich.theme import Theme as RichTheme

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
from ..ui.banner import WelcomeInfo, play_wordmark, welcome_card
from ..ui.prompt_box import PromptBox, sent_message
from ..ui.screen import record_screen
from ..ui.terminal import query_background, query_cursor_row, stream_keys
from ..ui.theme import markdown_styles, theme_for
from .commands import CHAT_COMMANDS, chat_command, help_lines, notice
from .failures import FailureText

#: The embedded agent's name, offered by /agent in every folder.
BUILT_IN_AGENT = "robutler"

#: "Find the agent file the usual way" as opposed to None, the built-in agent.
_UNSET: Any = object()


def _short_path(path: Path) -> str:
    text = str(path)
    home = str(Path.home())
    if text == home or text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _truncate_start(text: str, width: int) -> str:
    return text if len(text) <= width else "…" + text[-(width - 1):]


class WebAgentsSession:
    """One chat with one agent at a time, in this process."""

    def __init__(
        self,
        agent_path: Optional[Path] = None,
        model: Optional[str] = None,
        streaming: bool = True,
        chosen: bool = False,
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

        history_dir = Path.home() / ".webagents"
        history_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_box = PromptBox(
            self.theme,
            commands=[(f"/{c.name}", c.description) for c in CHAT_COMMANDS],
            footer=self._footer_parts,
            history=FileHistory(str(history_dir / "history")),
        )

        self._handlers: Dict[str, Callable[[str], Any]] = {
            "help": self.cmd_help,
            "new": lambda _args: self.start_new_conversation(),
            "clear": lambda _args: self.clear_screen(),
            "resume": self.cmd_resume,
            "undo": lambda _args: self.cmd_undo(),
            "rewind": self.cmd_rewind,
            "model": self.cmd_model,
            "agent": self.cmd_agent,
            "tools": lambda _args: self.cmd_tools(),
            "status": lambda _args: self.cmd_status(),
            "login": lambda _args: self.sign_in(),
            "logout": lambda _args: self.cmd_logout(),
            "keys": self.cmd_keys,
            "sandbox": lambda _args: self.cmd_sandbox(),
            "publish": lambda _args: self.cmd_publish(),
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

    async def initialize(self) -> None:
        """Build the agent, the way the daemon would (`cli/agent_builder.py`)."""
        from ..agent_builder import build_agent
        from ..credentials import get_token

        # `get_token`, read per search: `discovery` searches as the person at
        # this terminal when the agent has no platform credential of its own.
        self.built = await build_agent(
            self._resolve_agent_file(), working_dir=Path.cwd(), model=self.explicit_model, person_token=get_token
        )

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
        if not self.built:
            return []
        out = []
        for tool in self.built.agent.get_all_tools():
            description = (tool.get("description") or "").strip().split("\n")[0]
            out.append((str(tool.get("name")), description))
        # By name, as the TypeScript chat lists them: registration order is
        # each SDK's own business, and the list reads the same in both.
        return sorted(out)

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
        tokens = self.input_tokens + self.output_tokens
        if tokens:
            parts.append(f"{compact_number(tokens)} tokens")
        parts.append(_truncate_start(_short_path(Path.cwd()), 28))
        return parts

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

    def snapshot_before_turn(self, message: str) -> None:
        """Before a message: a snapshot of the folder, for /undo, when the agent
        can change files. Never in the home folder or above, and quiet about
        it until /undo is asked for. A snapshot that cannot be taken is said
        once, and the message goes ahead."""
        if not self.can_change_files:
            return
        folder = self.agent_folder()
        if checkpoints.snapshots_off_reason(folder):
            return
        try:
            self.turn_snapshots.append(checkpoints.take_snapshot(folder, checkpoints.turn_label(message))["id"])
        except OSError as error:
            if not self._snapshots_noticed:
                self.notice("warn", checkpoints.snapshot_failed(str(error)))
            self._snapshots_noticed = True

    async def _confirm(self, question: str) -> bool:
        """A yes to ``question``, asked only at a terminal (as /publish asks)."""
        if not sys.stdin.isatty():
            return False
        answer = await self._ask(f"  {question}")
        return (answer or "").strip().lower() in ("y", "yes")

    async def _confirm_and_restore(
        self, folder: Path, target: Dict[str, Any], plan: Any, header: str, before_label: str, done: Any = None
    ) -> None:
        """Show what a restore would do, ask, and do it."""
        p = self.theme.palette
        self.console.print()
        self.console.print(Text(f"  {header}", style=p.text))
        for line in checkpoints.plan_lines(plan):
            self.console.print(Text(line, style=p.muted))
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
            f"Continuing the conversation from {when_label(chosen.updated_at)} ({len(self.messages)} messages).",
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
                self.notice("error", f"Unknown command /{asked.lstrip('/')}.", "Type /help for the list.")
                return
            self.notice("info", spec.usage, spec.description)
            return
        self._print_lines(help_lines(self.theme))

    async def cmd_model(self, args: str) -> None:
        if not args.strip():
            self.notice("info", f"Model: {self.model_label() or '(none)'}", "Switch with /model <provider/model>.")
            return
        # REBUILD THE AGENT, do not just set the field: the built LLM skill
        # keeps its own model.
        previous = self.explicit_model
        self.explicit_model = args.strip()
        try:
            await self.initialize()
            # A model with no way to run here (no key, not signed in) is not a
            # switch; keep the one that works.
            if self.model_problem:
                raise RuntimeError(self.model_problem)
            self.notice("ok", f"Model set to {self.model_label()}")
        except Exception as error:  # noqa: BLE001 - said, and the working agent put back
            self.explicit_model = previous
            self.notice("error", f"Could not switch model: {error}")
            try:
                await self.initialize()
            except Exception:  # noqa: BLE001
                pass

    def folder_agents(self) -> List[Tuple[str, Path, str]]:
        """The agent files in this folder, with their names: AGENT.md and AGENT-<name>.md."""
        from ..agent_files import folder_agents

        return list(folder_agents(Path.cwd()))

    async def cmd_agent(self, args: str) -> None:
        p = self.theme.palette
        agents = self.folder_agents()
        current = self.agent_name
        wanted = args.strip()
        if not wanted:
            width = max([10, len(BUILT_IN_AGENT)] + [len(a[0]) for a in agents]) + 2
            room = max(10, self.console.width - width - 24)

            def row(name: str, where: str, description: str) -> Text:
                mark = ("●", p.accent) if name == current else ("○", p.faint)
                return Text.assemble(
                    "  ",
                    mark,
                    " ",
                    (name.ljust(width), f"bold {p.text}"),
                    (where.ljust(18), p.faint),
                    (_truncate(description, room), p.muted),
                )

            lines = [Text("Agents", style=f"bold {p.text}")]
            for name, file, description in agents:
                lines.append(row(name, file.name, description))
            lines.append(row(BUILT_IN_AGENT, "built in", "The general assistant"))
            lines.append(Text(""))
            lines.append(Text("  Switch with /agent <name>.", style=p.faint))
            self._print_lines(lines)
            return
        target = next((a for a in agents if a[0] == wanted), None)
        if target is None and wanted != BUILT_IN_AGENT:
            self.notice("error", f"There is no agent called {wanted} in this folder.", "Type /agent to see the list.")
            return
        if wanted == current:
            self.notice("info", f"Already talking to {current}.")
            return
        self.selected_file = target[1] if target else None
        self.explicit_model = None
        await self.initialize()
        self.start_new_conversation(False)
        if self.console.is_terminal:
            self.console.print()
            for line in welcome_card(self.theme, self.console.width, self.welcome_info()):
                self.console.print(line)
        self.notice("ok", f"Now talking to {self.agent_name}.", self.model_problem)

    def cmd_tools(self) -> None:
        p = self.theme.palette
        tools = self.tool_list()
        if not tools:
            self.notice("info", "This agent has no tools.", "Add skills to its AGENT.md, for example `filesystem`.")
            return
        width = min(28, max(len(name) for name, _ in tools)) + 2
        room = self.console.width - width - 6
        lines = [Text(f"Tools ({len(tools)})", style=f"bold {p.text}")]
        for name, description in tools:
            lines.append(
                Text.assemble(
                    "  ",
                    ("●", p.success),
                    " ",
                    (_truncate(name, width - 2).ljust(width), f"bold {p.text}"),
                    (_truncate(description, max(10, room)), p.muted),
                )
            )
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
        from .render import compact_number

        tokens = self.input_tokens + self.output_tokens
        file = self.built.file if self.built else None
        rows = [
            ("Account", account),
            ("Agent", f"{self.agent_name} ({file.name})" if file else f"{self.agent_name} (built in)"),
            ("Model", self.model_route()),
            ("Sandbox", self.sandbox_summary()[1]),
            ("Folder", _short_path(self.agent_folder())),
            (
                "Conversation",
                f"{len(self.messages)} messages{f', {compact_number(tokens)} tokens' if tokens else ''}"
                + (", also on Robutler" if self.platform_chat_id else ""),
            ),
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

    async def cmd_logout(self) -> None:
        from ..config_store import platform_url
        from ..credentials import TOKEN_ENV_VAR, get_token
        from ..platform.auth import logout

        host = re.sub(r"^https?://", "", platform_url().rstrip("/"))
        if not get_token():
            self.notice("info", "Not signed in.")
            return
        logout()
        await self.initialize()
        if os.environ.get(TOKEN_ENV_VAR):
            self.notice("warn", f"{TOKEN_ENV_VAR} is set in this shell, and it keeps you signed in.", "Unset it to sign out completely.")
            return
        self.notice("ok", f"Signed out of {host}.", self.model_problem or f"{self.agent_name} runs on {self.model_label()}.")

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

    def sandbox_summary(self) -> Tuple[str, str, Optional[str]]:
        """(kind, headline, detail): what the agent's commands may do, for /sandbox and /status."""
        shell = self.built.agent.skills.get("shell") if self.built else None
        if shell is None:
            return ("info", "Not needed: this agent cannot run commands.", None)
        policy = getattr(shell, "policy", None)
        if policy is None:
            return (
                "warn",
                "Off: commands run with your permissions.",
                "Add a `sandbox:` section to the agent file to confine them.",
            )
        preset = getattr(self.built.sandbox, "preset", None) or "custom"
        network = "on" if getattr(policy, "network", False) else "off"
        return (
            "ok",
            f"On ({preset}): writes stay in {_short_path(self.agent_folder())} and a scratch folder; network {network}; secrets removed.",
            None,
        )

    def cmd_sandbox(self) -> None:
        kind, headline, detail = self.sandbox_summary()
        self.notice(kind, f"Sandbox: {headline}", detail)

    async def cmd_publish(self) -> None:
        from ..publish import PublishIO, publish_agent

        p = self.theme.palette
        file = self.built.file if self.built else None
        if file is None:
            self.notice("warn", "Publishing needs an AGENT.md in this folder.", "Create one with `webagents init`, then /publish.")
            return

        async def confirm(question: str) -> bool:
            if not sys.stdin.isatty():
                return False
            answer = await asyncio.to_thread(input, f"  {question} [y/N] ")
            return answer.strip().lower() in ("y", "yes")

        self.console.print()
        result = await publish_agent(
            file,
            PublishIO(
                ok=lambda line: self.notice("ok", line),
                print=lambda line: self.console.print(Text(f"  {line}", style=p.muted)),
                error=lambda line: self.notice("error", line),
                confirm=confirm,
            ),
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
        try:
            return await asyncio.to_thread(input, question)
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
            kept += f"; `webagents secrets unset {env_var}` removes it."
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
        )

    async def handle_input(self, user_input: str) -> None:
        text = user_input.strip()
        if not text:
            return
        if text.startswith("/"):
            name, _, args = text[1:].partition(" ")
            handler = self._handlers.get(name.lower())
            if handler is None:
                self.notice("error", f"Unknown command /{name.lower()}.", "Type / to see the commands, or /help.")
                return
            result = handler(args.strip())
            if asyncio.iscoroutine(result):
                await result
            return
        if self.model_problem:
            self.notice("warn", self.model_problem, "Type /login to sign in without leaving the chat.")
            return
        message = self._expand_file_references(text)
        self.snapshot_before_turn(message)
        try:
            await self._turn(message)
        finally:
            self.save_conversation()
            self.record_on_robutler()

    async def _turn(self, message: str) -> None:
        """One reply, streamed by `render.TurnRenderer`, stoppable with Esc or Ctrl+C."""
        import signal

        from .render import TurnRenderer, error_lines, events_from_chunk

        self.messages.append({"role": "user", "content": message})
        self.console.print()
        renderer = TurnRenderer(self.console, theme=self.theme, explain_error=self._explain_failure)
        from rich.live import Live

        async def stream() -> None:
            # The person at the terminal is the agent's owner (`access.caller`).
            from webagents.access import run_as_local_owner

            run_as_local_owner(self.built.agent)
            if not self.streaming:
                # `--no-streaming`: the reply appears whole, when it is done.
                async for chunk in self.built.agent.run_streaming(list(self.messages)):
                    if isinstance(chunk, dict):
                        for event in events_from_chunk(chunk):
                            renderer.feed(event)
                renderer.flush(final=True)
                return
            with Live(console=self.console, refresh_per_second=12.5, transient=True, get_renderable=renderer.live_view):
                async for chunk in self.built.agent.run_streaming(list(self.messages)):
                    if isinstance(chunk, dict):
                        for event in events_from_chunk(chunk):
                            renderer.feed(event)
                        renderer.flush()
                renderer.flush(final=True)

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
        except (NotImplementedError, RuntimeError, ValueError):
            installed = False
        try:
            with stream_keys(interrupt, loop):
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
            if installed:
                loop.remove_signal_handler(signal.SIGINT)
        if interrupted:
            renderer.flush(final=True)
            p = self.theme.palette
            self.console.print(Text.assemble(("  ⎿  ", p.faint), ("Interrupted", p.warning)))
        stats = renderer.stats()
        if stats is not None:
            self.console.print()
            self.console.print(stats)
        answer = renderer.plain_text()
        # A REPLY IS SOMETHING SAID (2026-09-25): the session's last line
        # counted every turn, so a chat whose only message was refused ended
        # "1 reply". A turn counts when it said something, or ended without
        # failing or being stopped; the TypeScript chat counts the same way.
        failed = failed or renderer.failed
        if answer.strip() or not (failed or interrupted):
            self.turns += 1
        self.input_tokens += renderer.usage.prompt_tokens
        self.output_tokens += renderer.usage.completion_tokens

        if answer:
            self.messages.append({"role": "assistant", "content": answer})
        elif self.messages and self.messages[-1].get("role") == "user":
            # A turn that said nothing leaves no trace, so the next message is
            # not sent after an unanswered one.
            self.messages.pop()
        self.console.print()

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
        tokens = self.input_tokens + self.output_tokens
        parts = [f"{self.turns} {'reply' if self.turns == 1 else 'replies'}"]
        if tokens:
            parts.append(f"{tokens:,} tokens")
        parts.append(duration(time.time() - self.session_started))
        self.console.print(Text("✦ " + " · ".join(parts), style=self.theme.palette.faint))
        self.console.print()

    async def run(self) -> None:
        """At a terminal: the wordmark, the offer when there is no model, the card,
        then the box for every message. Anything else (a pipe, a script) gets a
        plain prompt and plain output."""
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
        else:
            print(f"\nWebAgents CLI - Connected to {self.agent_name}")
            if self.model_problem:
                print(self.model_problem)
            print("Type /help for available commands, or start chatting.\n")

        while self.running:
            try:
                self.say_recording_problem()
                if tty:
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


def _terminal_size() -> os.terminal_size:
    """The terminal's real size (the screen record's), whatever COLUMNS says."""
    try:
        return os.get_terminal_size(sys.__stdout__.fileno())
    except (AttributeError, OSError, ValueError):
        return os.terminal_size((80, 24))


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

    log_dir = Path.home() / ".webagents" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(log_dir / "repl.log"), console_output=False)

    session = WebAgentsSession(agent_path=agent_path, model=model, streaming=streaming, chosen=chosen)
    asyncio.run(session.run())
