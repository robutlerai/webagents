"""
How one assistant turn is drawn in the line-by-line chat (2026-09-24).

WHAT WAS WRONG. The chat re-rendered the whole turn (text, tool calls,
thinking) inside one Rich `Live` region on every chunk. Four things followed:
  - the tool calls it could see were drawn ABOVE the text or at the very end,
    never where they happened, and the text before and after a tool call was
    fused into one paragraph;
  - a tool's RESULT was never shown: the daemon sends `{"type": "tool_call"}`
    and `{"type": "tool_result"}` events, and the chat only understood an older
    `{"object": "metadata"}` shape that nothing emits any more;
  - streamed tool-call arguments arrive with the call's `id` on the first
    chunk only, and the rest were filed under their index as a second,
    nameless call ("Ran ...");
  - a reply taller than the terminal was clipped while it streamed, and the
    cost of each chunk grew with the length of the reply.

THE DESIGN, from what the terminal agents people use settled on:
  - the turn is a sequence of SEGMENTS in arrival order: text, tool calls,
    thinking;
  - finished work is printed ONCE, to ordinary scrollback: a markdown block is
    finished at a blank line outside a code fence (tables, lists and code
    blocks are only ever printed whole, never re-flowed after the fact); a
    tool call is finished when its result arrives;
  - only the unfinished tail stays in the live region: the block being
    written, any running tool, and a status line with the elapsed time;
  - a tool call is one line (a status dot, the name, its key argument) with
    the result summarised under it, and the output itself under that when
    tool details are on (Ctrl+T, then T).

THE LOOK (same day, after the TypeScript chat's redesign): the TypeScript
chat's visual language (`typescript/src/cli/render.ts`), so the SDKs match. A
two-column gutter carries the markers: ✦ the agent speaking, once per block of
text; ● a tool call (a spinner while it runs, green or red after, its result
under ⎿ with the time it took); ∴ a finished thought; ✗ an error with a hint
under it. Markdown is `ui/markdown.py` (left headings, a gradient H1, code as a
shaded band, rounded tables). The status line is an animated star and a verb
with a band of light crossing it ("Thinking", "Writing", "Running read_file"),
the time, a running token count and how to stop. The line builders below
(`tool_lines`, `thought_lines`, `error_lines`, `status_line`, `stats_line`)
are shared with the full-screen chat, so both draw a tool call the same way.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from rich.console import Console, Group, RenderableType
from rich.segment import Segment, Segments
from rich.text import Text

from ..ui.markdown import ChatMarkdown
from ..ui.motion import DOT_FRAMES, DOT_INTERVAL, STAR_FRAMES, STAR_INTERVAL, frame_at, shimmer, star_colour
from ..ui.theme import ChatTheme, theme_for

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
#: Lines of the block being written that stay visible while it streams.
LIVE_TAIL_LINES = 12
#: Output lines shown under a tool call when tool details are on.
DETAIL_LINES = 12
#: Two columns of gutter: the marker, then a space.
GUTTER = 2

# ---------------------------------------------------------------------------
# Stream events, from every shape the daemon and the in-process agent send
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    index: int
    id: Optional[str]
    name: Optional[str]
    arguments: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ToolResult:
    id: str
    status: str
    result: str


@dataclass
class ThinkingStart:
    pass


@dataclass
class ThinkingEnd:
    pass


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class StreamError:
    """The turn failed: the model could not be reached, refused the key, ...."""
    message: str


Event = Union[TextDelta, ToolCallDelta, ToolCall, ToolResult, ThinkingStart, ThinkingEnd, Usage, StreamError]


def events_from_chunk(chunk: Dict[str, Any]) -> List[Event]:
    """Every event one streamed chunk carries, whatever its shape."""
    events: List[Event] = []
    kind = chunk.get("type")
    error = chunk.get("error")
    if error and not chunk.get("choices"):
        # `{"error": "..."}` is what the servers stream when the run fails, and
        # `{"error": {"message": ...}}` is the OpenAI shape. Both were dropped,
        # so a failed turn drew an empty reply and said nothing (2026-09-24).
        message = error.get("message") if isinstance(error, dict) else error
        return [StreamError(str(message or error))]
    if kind == "tool_call":
        events.append(ToolCall(str(chunk.get("call_id") or chunk.get("id") or ""), str(chunk.get("name") or ""),
                               _as_text(chunk.get("arguments"))))
    elif kind == "tool_result":
        events.append(ToolResult(str(chunk.get("id") or chunk.get("call_id") or ""),
                                 str(chunk.get("status") or "success"), _as_text(chunk.get("result"))))
    elif chunk.get("object") == "metadata":
        payload = chunk.get("payload") or {}
        mtype = chunk.get("type")
        if mtype == "tool_start":
            events.append(ToolCall(str(payload.get("id") or ""), str(payload.get("name") or ""),
                                   _as_text(payload.get("arguments"))))
        elif mtype == "tool_result":
            events.append(ToolResult(str(payload.get("id") or ""), str(payload.get("status") or "success"),
                                     _as_text(payload.get("result"))))
        elif mtype == "thought_start":
            events.append(ThinkingStart())
        elif mtype == "thought_end":
            events.append(ThinkingEnd())
        return events

    choices = chunk.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            events.append(TextDelta(str(content)))
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            events.append(ToolCallDelta(int(tc.get("index") or 0), tc.get("id"), fn.get("name"),
                                        _as_text(fn.get("arguments"))))
    usage = chunk.get("usage")
    if isinstance(usage, dict) and usage:
        events.append(Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)))
    return events


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Tool call presentation
# ---------------------------------------------------------------------------

_KEY_ARGS = ("path", "file_path", "filename", "dir_path", "directory", "command", "cmd", "query", "pattern",
             "url", "name", "task")


def tool_key_argument(arguments: str) -> str:
    """The argument that says what a call was about: a path, a command, a query."""
    try:
        args = json.loads(arguments) if arguments else {}
    except ValueError:
        return arguments[:60]
    if not isinstance(args, dict):
        return str(args)[:60]
    for key in _KEY_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value if len(value) <= 60 else value[:59] + "…"
    for value in args.values():
        if isinstance(value, str) and value:
            return value if len(value) <= 60 else value[:59] + "…"
    return ""


#: How the local tools say they failed: they RETURN the message with a success
#: status ("File not found: x", "Access denied: ..."), so the status alone
#: painted a failed read green and summarised it as "Read 1 lines".
_FAILURE_TEXT = re.compile(
    r"^(error\b|access denied|permission denied|file not found|directory not found|no such file|"
    r"command not found|failed\b|traceback)",
    re.IGNORECASE,
)


def tool_failed(status: str, result: str) -> bool:
    """Whether a call failed, by its status or by the message it returned."""
    if status not in ("success", "ok", "completed"):
        return True
    first = (result or "").strip().splitlines()[0] if (result or "").strip() else ""
    return bool(_FAILURE_TEXT.match(first))


def tool_result_summary(name: str, result: str, status: str) -> str:
    """One line that says what the call produced."""
    text = (result or "").strip()
    if tool_failed(status, text):
        first = text.splitlines()[0] if text else "failed"
        return first if len(first) <= 100 else first[:99] + "…"
    if not text:
        return "(no output)"
    lines = [line for line in text.splitlines() if line.strip()]
    # A directory listing: a "Directory listing for ...:" header, then one
    # entry per line, directories marked `[DIR]` (files unmarked).
    if lines and lines[0].lower().startswith("directory listing"):
        entries = lines[1:]
        if not entries:
            return "empty directory"
        names = []
        for entry in entries:
            is_dir = re.match(r"^\s*\[(DIR|dir)\]", entry) is not None
            name = re.sub(r"^\s*\[(DIR|FILE|dir|file)\]\s*", "", entry).strip()
            names.append(name + ("/" if is_dir else ""))
        shown = ", ".join(names[:5]) + ("…" if len(names) > 5 else "")
        return f"{len(names)} {'entry' if len(names) == 1 else 'entries'}: {shown}"
    if name in ("read_file", "read"):
        return f"Read {len(text.splitlines())} lines"
    first = lines[0] if lines else text
    more = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
    first = first if len(first) <= 80 else first[:79] + "…"
    return first + more


@dataclass
class ToolSegment:
    key: str
    id: Optional[str]
    name: str = ""
    arguments: str = ""
    status: str = "running"
    result: str = ""
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    #: Why a call that never returned was drawn as finished ("interrupted").
    unfinished: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.status != "running"


@dataclass
class TextSegment:
    text: str = ""
    committed: int = 0  # characters of `text` already printed to scrollback
    closed: bool = False


@dataclass
class ThoughtSegment:
    text: str = ""
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None


@dataclass
class ErrorSegment:
    message: str


Segment_ = Union[TextSegment, ToolSegment, ThoughtSegment, ErrorSegment]


def complete_blocks_end(text: str, start: int) -> int:
    """How far into `text` (from `start`) the markdown is made of FINISHED blocks.

    A block ends at a blank line outside a fenced code block. Everything up to
    the returned offset can be printed once and never re-flowed; the rest is
    still being written.
    """
    in_fence = False
    fence = ""
    boundary = start
    pos = start
    for line in text[start:].splitlines(keepends=True):
        if not line.endswith("\n"):
            break  # the line being written
        stripped = line.strip()
        marker = re.match(r"^(```+|~~~+)", stripped)
        if marker:
            if not in_fence:
                in_fence, fence = True, marker.group(1)[:3]
            elif stripped.startswith(fence):
                in_fence = False
        pos += len(line)
        if not in_fence and not stripped:
            boundary = pos
    return boundary


class TurnRenderer:
    """Draws one assistant turn: printed once where finished, live where not."""

    def __init__(
        self,
        console: Console,
        show_tool_details: bool = False,
        expand_thinking: bool = False,
        theme: Optional[ChatTheme] = None,
        error_hint: Optional[Callable[[str], Optional[str]]] = None,
        explain_error: Optional[Callable[[str], Tuple[Any, ...]]] = None,
    ):
        self.console = console
        self.theme = theme or theme_for(console)
        #: A line of advice under an error (the chat knows which key the provider reads).
        self.error_hint = error_hint
        #: The error's headline and advice together (`failures.py`); the chat
        #: passes this, and `error_hint` is used only without it.
        self.explain_error = explain_error
        self.show_tool_details = show_tool_details
        self.expand_thinking = expand_thinking
        self._last_text = 0.0
        self.segments: List[Segment_] = []
        self.started = time.time()
        self.printed_segments = 0  # segments fully printed to scrollback
        self._index_to_key: Dict[int, str] = {}
        self._in_think_tag = False
        self.usage = Usage(0, 0)
        self.todos: Optional[list] = None
        self._printer = console  # replaced by the Live's console while live
        self._printed_any = False  # one blank line between printed blocks

    # -- events -------------------------------------------------------------

    def feed(self, event: Event) -> None:
        if isinstance(event, TextDelta):
            self._text(event.text)
        elif isinstance(event, ToolCallDelta):
            self._tool_delta(event)
        elif isinstance(event, ToolCall):
            self._tool_call(event)
        elif isinstance(event, ToolResult):
            self._tool_result(event)
        elif isinstance(event, ThinkingStart):
            self._close_text()
            self.segments.append(ThoughtSegment())
        elif isinstance(event, ThinkingEnd):
            thought = self._open_thought()
            if thought:
                thought.ended = time.time()
        elif isinstance(event, Usage):
            self.usage = Usage(self.usage.prompt_tokens + event.prompt_tokens,
                               self.usage.completion_tokens + event.completion_tokens)
        elif isinstance(event, StreamError):
            self._close_text()
            self.segments.append(ErrorSegment(event.message))

    @property
    def failed(self) -> bool:
        """Whether the turn reported an error."""
        return any(isinstance(segment, ErrorSegment) for segment in self.segments)

    def _text(self, text: str) -> None:
        self._last_text = time.time()
        # `<think>` inside the text stream: some models send reasoning that way.
        while text:
            if self._in_think_tag:
                end = text.find("</think>")
                thought = self._open_thought() or self._new_thought()
                if end == -1:
                    thought.text += text
                    return
                thought.text += text[:end]
                thought.ended = time.time()
                self._in_think_tag = False
                text = text[end + len("</think>"):]
                continue
            start = text.find("<think>")
            if start == -1:
                self._current_text().text += text
                return
            if start:
                self._current_text().text += text[:start]
            self._close_text()
            self._new_thought()
            self._in_think_tag = True
            text = text[start + len("<think>"):]

    def _current_text(self) -> TextSegment:
        last = self.segments[-1] if self.segments else None
        if isinstance(last, TextSegment) and not last.closed:
            return last
        segment = TextSegment()
        self.segments.append(segment)
        return segment

    def _close_text(self) -> None:
        last = self.segments[-1] if self.segments else None
        if isinstance(last, TextSegment):
            last.closed = True

    def _new_thought(self) -> ThoughtSegment:
        thought = ThoughtSegment()
        self.segments.append(thought)
        return thought

    def _open_thought(self) -> Optional[ThoughtSegment]:
        for segment in reversed(self.segments):
            if isinstance(segment, ThoughtSegment) and segment.ended is None:
                return segment
        return None

    def _find_tool(self, tool_id: str, unfinished_only: bool = True) -> Optional[ToolSegment]:
        for segment in self.segments:
            if isinstance(segment, ToolSegment) and segment.id == tool_id:
                if not unfinished_only or not segment.finished:
                    return segment
        return None

    def _tool_delta(self, event: ToolCallDelta) -> None:
        if event.id:
            tool = self._find_tool(event.id)
            if tool is None:
                self._close_text()
                tool = ToolSegment(key=f"{event.id}#{len(self.segments)}", id=event.id)
                self.segments.append(tool)
            self._index_to_key[event.index] = tool.key
        else:
            key = self._index_to_key.get(event.index)
            tool = next((s for s in self.segments if isinstance(s, ToolSegment) and s.key == key), None)
            if tool is None:
                # Arguments for a call whose first chunk never arrived: file
                # them under the index rather than invent a second call later.
                self._close_text()
                tool = ToolSegment(key=f"index{event.index}#{len(self.segments)}", id=None)
                self.segments.append(tool)
                self._index_to_key[event.index] = tool.key
        if event.name:
            tool.name = event.name
        if event.arguments:
            tool.arguments += event.arguments

    def _tool_call(self, event: ToolCall) -> None:
        tool = self._find_tool(event.id)
        if tool is None:
            self._close_text()
            tool = ToolSegment(key=f"{event.id}#{len(self.segments)}", id=event.id)
            self.segments.append(tool)
        tool.name = event.name or tool.name
        if event.arguments:
            tool.arguments = event.arguments
        if tool.name == "write_todos":
            try:
                args = json.loads(tool.arguments or "{}")
                todos = args.get("todos")
                self.todos = json.loads(todos) if isinstance(todos, str) else todos
            except (ValueError, AttributeError):
                pass

    def _tool_result(self, event: ToolResult) -> None:
        tool = self._find_tool(event.id)
        if tool is None:
            tool = ToolSegment(key=f"{event.id}#{len(self.segments)}", id=event.id, name="tool")
            self._close_text()
            self.segments.append(tool)
        tool.status = event.status or "success"
        tool.result = event.result
        tool.ended = time.time()

    # -- drawing --------------------------------------------------------------

    def _markdown_lines(self, text: str) -> List[List[Segment]]:
        width = max(20, self.console.width - GUTTER - 1)
        options = self.console.options.update(width=width)
        lines = self.console.render_lines(ChatMarkdown(text, self.theme), options, pad=False)
        return _trim_blank(lines)

    def _print(self, renderable: RenderableType) -> None:
        # Blocks (a markdown chunk, a tool call, a thought) are separated by
        # one blank line, the way the chat reads when it is all finished.
        if self._printed_any:
            self._printer.print("")
        self._printer.print(renderable)
        self._printed_any = True

    def _gutter(self, lines: List[List[Segment]], marked: bool) -> List[Segment]:
        """Lines of a text block behind the gutter: the ✦ on the block's first."""
        marker = Segment("✦ ", self.console.get_style(self.theme.palette.agent))
        blank = Segment(" " * GUTTER)
        out: List[Segment] = []
        for index, line in enumerate(lines):
            out.append(marker if marked and index == 0 else blank)
            out.extend(line)
            out.append(Segment.line())
        return out

    def _print_markdown(self, text: str, marked: bool) -> None:
        if text.strip():
            self._print(Segments(self._gutter(self._markdown_lines(text), marked)))

    def flush(self, final: bool = False) -> None:
        """Print everything that is finished, in order, exactly once."""
        width = self.console.width
        while self.printed_segments < len(self.segments):
            segment = self.segments[self.printed_segments]
            is_last = self.printed_segments == len(self.segments) - 1
            if isinstance(segment, TextSegment):
                done = segment.closed or not is_last or final
                end = len(segment.text) if done else complete_blocks_end(segment.text, segment.committed)
                if end > segment.committed:
                    first = not segment.text[: segment.committed].strip()
                    self._print_markdown(segment.text[segment.committed:end], marked=first)
                    segment.committed = end
                if not done:
                    return
            elif isinstance(segment, ToolSegment):
                if not segment.finished and not final:
                    return
                if not segment.finished:
                    segment.unfinished = "interrupted"
                self._print(Group(*tool_lines(self.theme, segment, time.time(), width, self.show_tool_details)))
            elif isinstance(segment, ThoughtSegment):
                if segment.ended is None and not final:
                    return
                if segment.ended is None:
                    segment.ended = time.time()
                self._print(Group(*thought_lines(self.theme, segment, time.time(), self.expand_thinking)))
            elif isinstance(segment, ErrorSegment):
                if self.explain_error is not None:
                    explained = self.explain_error(segment.message)
                    headline, hint = explained[0], explained[1]
                else:
                    headline = segment.message
                    hint = self.error_hint(segment.message) if self.error_hint else None
                self._print(Group(*error_lines(self.theme, headline, hint, width)))
            self.printed_segments += 1

    def verb(self, now: float) -> str:
        """What the status line says the agent is doing now."""
        running = [s for s in self.segments if isinstance(s, ToolSegment) and not s.finished]
        if running:
            more = f" +{len(running) - 1}" if len(running) > 1 else ""
            return f"Running {running[0].name or 'a tool'}{more}"
        if self._open_thought() is not None:
            return "Thinking"
        tail = self.segments[-1] if self.segments else None
        pending = isinstance(tail, TextSegment) and bool(tail.text[tail.committed:].strip())
        if pending or now - self._last_text < 1.2:
            return "Writing"
        return "Thinking"

    def streamed_characters(self) -> int:
        return sum(len(s.text) for s in self.segments if isinstance(s, TextSegment))

    def live_view(self) -> RenderableType:
        """What is still being written, plus the status line."""
        now = time.time()
        width = self.console.width
        parts: List[RenderableType] = []
        blocks: List[List[RenderableType]] = []
        for segment in self.segments[self.printed_segments:]:
            if isinstance(segment, TextSegment):
                pending = segment.text[segment.committed:]
                if pending.strip():
                    lines = self._markdown_lines(pending)
                    marked = not segment.text[: segment.committed].strip()
                    tail = lines[-LIVE_TAIL_LINES:]
                    blocks.append([Segments(self._gutter(tail, marked and len(tail) == len(lines)))])
            elif isinstance(segment, ToolSegment):
                blocks.append(tool_lines(self.theme, segment, now, width, self.show_tool_details))
            elif isinstance(segment, ThoughtSegment):
                blocks.append(thought_lines(self.theme, segment, now, self.expand_thinking))
        # The same spacing the printed blocks will have, so nothing jumps when
        # a block moves from here to scrollback.
        for index, block in enumerate(blocks):
            if index or self._printed_any:
                parts.append(Text(""))
            parts.extend(block)
        if parts:
            parts.append(Text(""))
        parts.append(status_line(self.theme, self.verb(now), self.started, self.streamed_characters(), now))
        return Group(*parts)

    def produced_something(self) -> bool:
        return any(
            (isinstance(s, TextSegment) and s.text.strip()) or isinstance(s, ToolSegment) for s in self.segments
        )

    def stats(self) -> Optional[Text]:
        """The turn's last line, when it produced something: how long, how many tokens."""
        if not self.produced_something():
            return None
        tokens = self.usage.prompt_tokens + self.usage.completion_tokens
        return stats_line(self.theme, time.time() - self.started, tokens)

    @property
    def failed(self) -> bool:
        """Whether the turn reported an error."""
        return any(isinstance(segment, ErrorSegment) for segment in self.segments)

    def plain_text(self) -> str:
        """The turn's text for the conversation history: no tool calls, no thinking.

        Stretches of text either side of a tool call are separate paragraphs:
        joined with one newline they rendered as one fused paragraph when a
        restored conversation was shown again (2026-09-24).
        """
        stretches = [s.text.strip() for s in self.segments if isinstance(s, TextSegment) and s.text.strip()]
        return "\n\n".join(stretches)


def duration(seconds: float) -> str:
    """3.24 -> "3.2s", 75 -> "1m 15s" (the TypeScript chat's format)."""
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{round(seconds)}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {round(seconds - minutes * 60):02d}s"


def compact_number(n: int) -> str:
    """1234 -> "1.2k", for token counts in a status line."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k" if n < 10_000 else f"{round(n / 1000)}k"
    return f"{n / 1_000_000:.1f}M"


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def tool_lines(theme: ChatTheme, tool: ToolSegment, now: float, width: int, expanded: bool = False) -> List[Text]:
    """A tool call: `● name(argument)`, then `⎿  what it produced` and how long it took."""
    p = theme.palette
    bad = tool.unfinished == "interrupted" or (tool.finished and tool_failed(tool.status, tool.result))
    if not tool.finished and tool.unfinished is None:
        dot = Text(frame_at(DOT_FRAMES, DOT_INTERVAL, now), style=p.accent)
    else:
        dot = Text("●", style=p.error if bad else p.success)
    header = Text.assemble(dot, " ", (tool.name or "tool", f"bold {p.text}"))
    argument = tool_key_argument(tool.arguments)
    if argument:
        room = max(8, width - GUTTER - len(tool.name) - 4)
        header.append(f"({_clip(argument, room)})", style=p.muted)
    lines = [header]
    if not tool.finished and tool.unfinished is None:
        return lines
    ended = tool.ended or now
    seconds = ended - tool.started
    timing = f"  {duration(seconds)}" if seconds >= 0.5 and tool.unfinished is None else ""
    summary = tool.unfinished or tool_result_summary(tool.name, tool.result, tool.status)
    summary = _clip(summary, max(10, width - 7 - len(timing)))
    lines.append(Text.assemble(("  ⎿  ", p.faint), (summary, p.error if bad else p.muted), (timing, p.faint)))
    if expanded and tool.unfinished is None and tool.result.strip():
        body = tool.result.rstrip().splitlines()
        for line in body[:DETAIL_LINES]:
            lines.append(Text("     " + _clip(line, max(20, width - 6)), style=p.faint))
        if len(body) > DETAIL_LINES:
            lines.append(Text(f"     … +{len(body) - DETAIL_LINES} lines", style=f"italic {p.faint}"))
    return lines


def thought_lines(theme: ChatTheme, thought: ThoughtSegment, now: float, expanded: bool = False) -> List[Text]:
    p = theme.palette
    if thought.ended is None:
        return [Text(f"{frame_at(DOT_FRAMES, DOT_INTERVAL, now)} Thinking…", style=f"italic {p.faint}")]
    lines = [Text(f"∴ Thought for {duration(thought.ended - thought.started)}", style=f"italic {p.faint}")]
    if expanded and thought.text.strip():
        for line in thought.text.strip().splitlines():
            lines.append(Text("  │ " + line, style=p.faint))
    return lines


def error_lines(theme: ChatTheme, message: str, hint: Optional[str], width: int) -> List[Text]:
    """`✗ what failed`, and under it what to do about it."""
    p = theme.palette
    first = (message or "The model returned an error.").splitlines()[0]
    lines = [Text.assemble(("✗ ", f"bold {p.error}"), (first, p.error))]
    if hint:
        lines.append(Text.assemble(("  ⎿  ", p.faint), (hint, p.faint)))
    return lines


def error_hint(model: Optional[str], message: str) -> Optional[str]:
    """What to do about a failed turn, for the errors a first run meets: a key
    the provider refused, a server that is not there, a rate limit, a model
    that does not exist. Names environment variables, never their values (the
    TypeScript chat's `errorHint` gives the same advice)."""
    import os

    from webagents.agents.skills.core.llm.providers import provider_for_model

    provider = provider_for_model(model)
    key = provider.env_vars[0] if provider is not None and provider.env_vars else None
    if re.search(r"\b401\b|unauthori[sz]ed|invalid.{0,10}api.?key|incorrect api key|api key not valid|authentication", message, re.I):
        return f"The provider refused the key. Check {key}, then start the chat again." if key else "The provider refused the key."
    if re.search(r"\b429\b|rate.?limit|quota", message, re.I):
        return "The provider is limiting requests. Wait a moment, then try again."
    if re.search(r"could not reach|connection error|connection refused|connect refused|ECONNREFUSED|ENOTFOUND|timed out|all connection attempts failed", message, re.I):
        base = key.replace("_API_KEY", "_BASE_URL") if key and key.endswith("_API_KEY") else None
        if base and os.environ.get(base):
            return f"{base} is set; check that it points at a running server."
        return "Check the network connection."
    if re.search(r"\b404\b|model.{0,40}(not found|does not exist)|unknown model", message, re.I):
        # The TypeScript chat's words: /model switches it in either chat.
        return "Check the model name; /model switches it."
    return None


def status_line(theme: ChatTheme, verb: str, started: float, streamed_chars: int, now: float) -> Text:
    """Claude Code's star in the brand gradient, the verb with Codex's band of light,
    the time, a running estimate of what has streamed back, and how to stop."""
    p = theme.palette
    seconds = int(now - started)
    elapsed = f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"
    streamed = f" · ↓ {compact_number(max(1, round(streamed_chars / 4)))} tokens" if streamed_chars else ""
    line = Text()
    line.append(frame_at(STAR_FRAMES, STAR_INTERVAL, now), style=f"bold {star_colour(theme, now)}")
    line.append(" ")
    line.append_text(shimmer(theme, f"{verb}…", p.agent, now))
    line.append(f" ({elapsed}{streamed} · ", style=p.faint)
    line.append("esc", style=p.muted)
    line.append(" to interrupt)", style=p.faint)
    return line


def stats_line(theme: ChatTheme, seconds: float, tokens: int) -> Text:
    parts = [f"Worked for {duration(seconds)}"]
    if tokens:
        parts.append(f"{tokens:,} tokens")
    return Text("✻ " + " · ".join(parts), style=theme.palette.faint)


def _is_blank(line: List[Segment]) -> bool:
    """An empty line: whitespace only, and nothing painted (a code block's
    shaded padding row has a background, and is kept)."""
    for segment in line:
        if segment.text.strip():
            return False
        if segment.style is not None and segment.style.bgcolor is not None:
            return False
    return True


def _trim_blank(lines: List[List[Segment]]) -> List[List[Segment]]:
    """Drop the blank lines Rich puts around some blocks: the renderer adds
    exactly one between blocks itself, and both together doubled every gap."""
    start, end = 0, len(lines)
    while start < end and _is_blank(lines[start]):
        start += 1
    while end > start and _is_blank(lines[end - 1]):
        end -= 1
    return lines[start:end]


def _join_lines(lines: Iterable[List[Segment]]) -> List[Segment]:
    out: List[Segment] = []
    for line in lines:
        out.extend(line)
        out.append(Segment.line())
    return out
