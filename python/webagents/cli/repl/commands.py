"""
The chat's commands and keys (2026-09-24), the same in both SDKs.

The shared fixture `python/tests/fixtures/cli/chat_commands.json` is the
reference (names, usage, wording, groups, details): the TypeScript chat keeps
the same list in `src/cli/chat-commands.ts`, and both suites compare their
table with the fixture (`test_chat_commands_fixture_interactive.py`,
`chat-commands-fixture-interactive.test.ts`), so a command cannot be added,
renamed or reworded in one chat only. Also here: how a command's one-line
result is drawn (`notice`), the same markers and colours as the TypeScript chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..ui.theme import ChatTheme

#: The /help headings, in order (2026-09-29): the conversation first, as the
#: commands used most; what the agent is (Agent) apart from what it may do
#: (Limits); /status beside /help and /exit.
CHAT_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("conversation", "Conversation"),
    ("agent", "Agent"),
    ("limits", "Limits"),
    ("account", "Account"),
    ("chat", "Chat"),
)


@dataclass(frozen=True)
class ChatCommandSpec:
    #: Typed after the slash.
    name: str
    #: How it is typed, for /help.
    usage: str
    #: One line, for the menu and /help.
    description: str
    #: Which /help heading it sits under.
    group: str
    #: Its forms, for `/help <command>`: one line each.
    details: Tuple[str, ...] = field(default_factory=tuple)


#: ONE LEVEL OF SUBCOMMANDS (2026-09-29, the owner's question "should we do
#: subcommands, eg /agent model?"). A bare noun shows the thing (`/model`,
#: `/skills`, `/mcp`); a verb after it changes it (`add`, `remove`, `set`,
#: `run`, `delete`), and `list` shows what could be added. Actions on the
#: conversation are plain verbs (`/new`, `/resume`, `/undo`). A command that
#: takes a free name (an agent, a model) keeps its verbs few and fixed:
#: `/agent new|edit` and nothing more, since every verb there is a name an
#: agent could not be switched to by.
CHAT_COMMANDS: Tuple[ChatCommandSpec, ...] = (
    ChatCommandSpec("new", "/new", "Start a new conversation", "conversation"),
    ChatCommandSpec(
        "resume",
        "/resume [number | delete <number>]",
        "Continue an earlier conversation, or delete one",
        "conversation",
        (
            "/resume                    this folder's earlier conversations, newest first",
            "/resume <number>           continue one",
            "/resume delete <number>    delete one, after asking",
        ),
    ),
    ChatCommandSpec(
        "compact",
        "/compact [focus]",
        "Summarize the conversation so far, to make room",
        "conversation",
        (
            "/compact            everything before the latest exchange becomes a summary",
            "/compact <focus>    and the summary keeps what you name",
        ),
    ),
    ChatCommandSpec("context", "/context", "How full the model's context is", "conversation"),
    ChatCommandSpec("undo", "/undo", "Put back the files your last message or command changed", "conversation"),
    ChatCommandSpec("rewind", "/rewind [number]", "Put the folder back as it was before an earlier message", "conversation"),
    ChatCommandSpec("clear", "/clear", "Start a new conversation and clear the screen", "conversation"),
    ChatCommandSpec(
        "agent",
        "/agent [name]",
        "List this folder's agents, switch to one, or make one",
        "agent",
        (
            "/agent                                    the agents here",
            "/agent <name>                             switch to one",
            "/agent new <name> [chatbot|tool-agent]    make one here",
            "/agent edit [name]                        open its file in your editor, then use it",
        ),
    ),
    ChatCommandSpec(
        "model",
        "/model [provider/model] [--save]",
        "Show or switch the model",
        "agent",
        (
            "/model                            which model, and how it is reached",
            "/model <provider/model>           switch, for this chat",
            "/model <provider/model> --save    switch, and keep it in the agent file",
        ),
    ),
    ChatCommandSpec(
        "skills",
        "/skills [list|add|remove]",
        "This agent's skills, and adding or removing one",
        "agent",
        (
            "/skills list",
            "/skills add <name>... | <owner/repo | git URL | folder> [--skill <name>]",
            "/skills remove <name>...",
        ),
    ),
    ChatCommandSpec("tools", "/tools", "List what the agent can use, and who else may", "agent"),
    ChatCommandSpec(
        "mcp",
        "/mcp [list|add|remove]",
        "The MCP servers this agent uses; add or remove one",
        "agent",
        (
            "/mcp                  this agent's servers, connected or not",
            "/mcp list             the servers Claude Desktop, Claude Code, Cursor, VS Code and Windsurf use",
            "/mcp add <name>       copy one into this agent (--from <app> when more than one has it)",
            "/mcp remove <name>    take one out of this agent; its secrets stay stored",
        ),
    ),
    ChatCommandSpec(
        "memory",
        "/memory [forget <key>]",
        "What this agent remembers",
        "agent",
        ("/memory                  the notes it keeps, and where", "/memory forget <key>     remove one of your notes"),
    ),
    ChatCommandSpec(
        "cron",
        "/cron [run <name>]",
        "This agent's schedules; run one now",
        "agent",
        ("/cron                 the schedules, as the daemon runs them", "/cron run <name>      run one now, and deliver its result"),
    ),
    ChatCommandSpec("reload", "/reload", "Read the agent file again and use it", "agent"),
    ChatCommandSpec("access", "/access", "Who may call this agent, and what each caller gets", "limits"),
    ChatCommandSpec("sandbox", "/sandbox", "What the agent's commands are allowed to do", "limits"),
    ChatCommandSpec(
        "rounds",
        "/rounds [n] [--save]",
        "Show or set the tool rounds one turn may run",
        "limits",
        (
            "/rounds               how many, and where that comes from",
            "/rounds <n>           set it, for this chat",
            "/rounds <n> --save    set it, and keep it in the agent file",
        ),
    ),
    ChatCommandSpec("login", "/login", "Sign in to Robutler", "account"),
    ChatCommandSpec("logout", "/logout", "Sign out of Robutler", "account"),
    ChatCommandSpec(
        "keys",
        "/keys [set|remove NAME]",
        "Model provider keys, and where each comes from",
        "account",
        (
            "/keys                  every provider key, and where each comes from",
            "/keys set <NAME>       store one, asked for with echo off",
            "/keys remove <NAME>    remove a stored one",
        ),
    ),
    ChatCommandSpec(
        "secrets",
        "/secrets [set|remove NAME]",
        "Secrets for MCP servers, and adding or removing one",
        "account",
        (
            "/secrets                  the names stored on this machine",
            "/secrets set <NAME>       store one, asked for with echo off; an MCP server uses it as ${secret:NAME}",
            "/secrets remove <NAME>    remove a stored one",
        ),
    ),
    ChatCommandSpec(
        "publish",
        "/publish [--dry-run]",
        "Publish this agent to Robutler, or update it",
        "account",
        ("/publish              create it, or update the linked one after asking", "/publish --dry-run    show what would be sent, and send nothing"),
    ),
    ChatCommandSpec("status", "/status", "Account, agent, model, sandbox, folder and Robutler", "chat"),
    ChatCommandSpec("help", "/help [command]", "Show the commands and keys", "chat"),
    ChatCommandSpec("exit", "/exit", "Leave the chat", "chat"),
)

#: The commands whose arguments the box completes (interactive-mode spec 3.8,
#: 2026-09-26; `rewind` and `model` since 2026-09-30): after `/<command> ` the
#: menu stays open with what the chat offers for the argument being typed,
#: found by what is typed (`ui/prompt_box.py`, "SEARCH"). The same list in the
#: TypeScript box (`cli/chat-commands.ts`), pinned by the fixture's `completion`.
COMPLETED_COMMANDS: Tuple[str, ...] = ("resume", "rewind", "agent", "model", "skills", "help", "keys", "cron", "memory", "mcp")

#: The commands enter opens as a list to choose from, when there is something
#: to choose (2026-09-30, the owner: "/resume ... should have search/filter on
#: typing and up/down arrow selection"): their bare form only prints that list.
PICKER_COMMANDS: Tuple[str, ...] = ("resume", "rewind")

#: The model tiers `/model` offers when the agent can run any provider's models
#: (no provider skill, or `proxy`): Robutler maps each to its current model.
MODEL_TIERS: Tuple[str, ...] = ("auto/fastest", "auto/balanced", "auto/smartest")

#: The typed-line history both chats keep (owner decision D1, S-291,
#: 2026-09-26): one file per profile, in the profile's own folder, readable by
#: its owner only, in prompt_toolkit's `FileHistory` format. The TypeScript chat
#: reads and writes the same file (`cli/chat-history.ts`).
CHAT_HISTORY = {"file": "history", "dir_mode": 0o700, "file_mode": 0o600, "keep": 1000}

CHAT_KEYS: Tuple[Tuple[str, str], ...] = (
    ("enter", "send"),
    ("alt+enter", "new line (or end a line with \\)"),
    ("↑ ↓", "earlier messages typed in this folder"),
    ("→", "take the grey suggestion from history"),
    ("tab", "complete a command, or take the suggestion"),
    ("esc", "stop a reply; twice in the box, clear it"),
    ("ctrl+c", "clear the box; twice, leave"),
)

#: Names a person may type that are not commands, and the command each became.
MOVED_COMMANDS: Dict[str, str] = {"edit": "agent edit"}

#: The last line of /help: the ways to run an agent outside the chat.
OUTSIDE_THE_CHAT = (
    "Outside the chat: webagents serve (HTTP), webagents mcp serve (MCP clients), "
    "webagents acp (code editors), webagents daemon (every agent here, with schedules)."
)


def chat_command(name: str) -> Optional[ChatCommandSpec]:
    """The spec for `name`, with or without its slash."""
    bare = name.lstrip("/").lower()
    return next((c for c in CHAT_COMMANDS if c.name == bare), None)


def takes_no_arguments(spec: ChatCommandSpec) -> bool:
    """Whether a command's usage names no argument, so an argument is a usage error."""
    return "[" not in spec.usage and "<" not in spec.usage


_MARKERS = {
    "ok": "✓",
    "info": "✦",
    "warn": "▲",
    "error": "✗",
}


def notice(console: Console, theme: ChatTheme, kind: str, text: str, detail: Optional[str] = None) -> None:
    """A command's one-line result: ✓ done, ✦ information, ▲ warning, ✗ failure.

    Drawn as the TypeScript chat's `notice`: a blank line, the marker and the
    text (wrapped under itself), the detail faint beneath, a blank line. A
    detail with newlines becomes several faint rows (the /help subcommand forms).
    """
    p = theme.palette
    marker_style = {"ok": p.success, "info": p.agent, "warn": p.warning, "error": f"bold {p.error}"}[kind]
    text_style = p.error if kind == "error" else p.warning if kind == "warn" else p.text
    grid = Table.grid(padding=0)
    grid.width = max(20, console.width - 1)
    grid.add_column(width=2, no_wrap=True)
    # Fold, never truncate: a long message (a path, a loader sentence) wraps
    # under itself, as the TypeScript chat's `wrapStyled` does, rather than
    # losing its tail to an ellipsis.
    grid.add_column(ratio=1, overflow="fold")
    grid.add_row(Text(_MARKERS[kind], style=marker_style), Text(text, style=text_style))
    if detail:
        for line in detail.split("\n"):
            grid.add_row(Text(""), Text(line, style=p.faint))
    console.print()
    console.print(grid)
    console.print()


def help_lines(theme: ChatTheme) -> List[Text]:
    """/help: the commands by group, then the keys, then the outside line."""
    p = theme.palette
    width = max(len(c.usage) for c in CHAT_COMMANDS) + 2
    lines: List[Text] = []
    for group, heading in CHAT_GROUPS:
        lines.append(Text(heading, style=f"bold {p.text}"))
        for c in CHAT_COMMANDS:
            if c.group != group:
                continue
            lines.append(Text.assemble("  ", (c.usage.ljust(width), p.accent), (c.description, p.muted)))
        lines.append(Text(""))
    lines.append(Text("Keys", style=f"bold {p.text}"))
    for key, what in CHAT_KEYS:
        lines.append(Text.assemble("  ", (key.ljust(width), p.text), (what, p.muted)))
    lines.append(Text(""))
    lines.append(Text(OUTSIDE_THE_CHAT, style=p.faint))
    return lines
