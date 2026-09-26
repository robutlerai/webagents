"""
The chat's commands and keys (2026-09-24), the same in both SDKs.

The TypeScript chat's list (`typescript/src/cli/chat-commands.ts`) is the
reference; this one must match it word for word, and
`tests/cli/test_chat_command_parity.py` reads that file and fails when they
differ. Also here: how a command's one-line result is drawn (`notice`), the
same markers and colours the TypeScript chat uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..ui.theme import ChatTheme


@dataclass(frozen=True)
class ChatCommandSpec:
    #: Typed after the slash.
    name: str
    #: How it is typed, for /help.
    usage: str
    #: One line, for the menu and /help.
    description: str


CHAT_COMMANDS: Tuple[ChatCommandSpec, ...] = (
    ChatCommandSpec("help", "/help [command]", "Show the commands and keys"),
    ChatCommandSpec("new", "/new", "Start a new conversation"),
    ChatCommandSpec("clear", "/clear", "Start a new conversation and clear the screen"),
    ChatCommandSpec("resume", "/resume [number]", "Continue an earlier conversation in this folder"),
    ChatCommandSpec("undo", "/undo", "Put back the files your last message changed"),
    ChatCommandSpec("rewind", "/rewind [number]", "Put the folder back as it was before an earlier message"),
    ChatCommandSpec("model", "/model [provider/model]", "Show or switch the model"),
    ChatCommandSpec("agent", "/agent [name]", "List this folder's agents, or switch to one"),
    ChatCommandSpec("tools", "/tools", "List what the agent can use"),
    ChatCommandSpec("status", "/status", "Account, agent, model, sandbox and folder"),
    ChatCommandSpec("login", "/login", "Sign in to Robutler"),
    ChatCommandSpec("logout", "/logout", "Sign out of Robutler"),
    ChatCommandSpec("keys", "/keys [set|unset NAME]", "Model provider keys, and where each comes from"),
    ChatCommandSpec("sandbox", "/sandbox", "What the agent's commands are allowed to do"),
    ChatCommandSpec("publish", "/publish", "Publish this agent to Robutler, or update it"),
    ChatCommandSpec("exit", "/exit", "Leave the chat"),
)

CHAT_KEYS: Tuple[Tuple[str, str], ...] = (
    ("enter", "send"),
    ("alt+enter", "new line (or end a line with \\)"),
    ("↑ ↓", "earlier messages"),
    ("tab", "complete a command"),
    ("esc", "stop a reply; twice in the box, clear it"),
    ("ctrl+c", "clear the box; twice, leave"),
)


def chat_command(name: str) -> Optional[ChatCommandSpec]:
    """The spec for `name`, with or without its slash."""
    bare = name.lstrip("/").lower()
    return next((c for c in CHAT_COMMANDS if c.name == bare), None)


_MARKERS = {
    "ok": "✓",
    "info": "✦",
    "warn": "▲",
    "error": "✗",
}


def notice(console: Console, theme: ChatTheme, kind: str, text: str, detail: Optional[str] = None) -> None:
    """A command's one-line result: ✓ done, ✦ information, ▲ warning, ✗ failure.

    Drawn as the TypeScript chat's `notice`: a blank line, the marker and the
    text (wrapped under itself), the detail faint beneath, a blank line.
    """
    p = theme.palette
    marker_style = {"ok": p.success, "info": p.agent, "warn": p.warning, "error": f"bold {p.error}"}[kind]
    text_style = p.error if kind == "error" else p.warning if kind == "warn" else p.text
    grid = Table.grid(padding=0)
    # One column short of the terminal, as the TypeScript `notice` wraps
    # (`terminalColumns() - 1`): the last column makes some terminals wrap
    # on their own, and the two chats broke the same sentence differently.
    grid.width = max(20, console.width - 1)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row(Text(_MARKERS[kind], style=marker_style), Text(text, style=text_style))
    if detail:
        grid.add_row(Text(""), Text(detail, style=p.faint))
    console.print()
    console.print(grid)
    console.print()


def help_lines(theme: ChatTheme) -> List[Text]:
    """/help: every command with its usage, then the keys."""
    p = theme.palette
    width = max(len(c.usage) for c in CHAT_COMMANDS) + 2
    lines = [Text("Commands", style=f"bold {p.text}")]
    for c in CHAT_COMMANDS:
        lines.append(Text.assemble("  ", (c.usage.ljust(width), p.accent), (c.description, p.muted)))
    lines.append(Text(""))
    lines.append(Text("Keys", style=f"bold {p.text}"))
    for key, what in CHAT_KEYS:
        lines.append(Text.assemble("  ", (key.ljust(width), p.text), (what, p.muted)))
    return lines
