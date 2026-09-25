"""
Markdown the way the chat draws it (2026-09-24).

Rich's defaults were made for documents: a centred H1, code in a padded
Monokai box, borderless SIMPLE tables, a quote bar in the quote's own style.
These subclasses give the chat the TypeScript chat's look
(`typescript/src/cli/render.ts`):

  * headings left-aligned; H1 in the agent's lime, H2 in the text colour;
  * code as a shaded BAND with the language faint at its right, half-block
    rows padding it by half a line (Gemini CLI's trick), syntax colour in the
    chat palette. A band, not a box: a border character on every line is what
    a person copying the code gets in their clipboard, which is why neither
    Gemini CLI nor Codex draws side borders on code. Below 256 colours there
    is no band, so thin rules mark the start and end;
  * tables with rounded borders and a bold header in the agent's colour, cells wrapping to
    fit (Rich wraps them);
  * a bar in the agent's colour beside a quote, the quote itself italic and
    muted;
  * a bare URL as a link, found by the TypeScript chat's pattern
    (`_link_bare_urls`, below).

`ChatMarkdown(text, theme)` is otherwise Rich's `Markdown`; the style names it
reads (inline code, links, bullets) come from `theme.markdown_styles`, pushed
onto the console by the chat.
"""

from __future__ import annotations

import re

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import BlockQuote, CodeBlock, Heading, ListItem, Markdown, TableElement
from rich.segment import Segment
from rich.syntax import PygmentsSyntaxTheme, Syntax
from rich.table import Table
from rich.text import Text

from .theme import ChatTheme, pygments_style

# The chat theme rides on a per-instance subclass of each element, as
# `chat_theme` (Rich's CodeBlock already uses `theme` for its Pygments theme).


class _ChatHeading(Heading):
    chat_theme: ChatTheme

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        # H1 takes its style, the agent's lime ("Signal", `theme.py`); it used
        # to be drawn letter by letter through the brand gradient.
        yield text


class _ChatCodeBlock(CodeBlock):
    chat_theme: ChatTheme

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        theme = self.chat_theme
        palette = theme.palette
        code = str(self.text).rstrip("\n")
        width = options.max_width
        lang = "" if self.lexer_name == "text" else self.lexer_name
        if not theme.rich_colour:
            label = f" {lang} " if lang else ""
            yield Text(f"──{label}{'─' * max(0, width - 2 - len(label))}", style=palette.border)
            for line in code.split("\n"):
                yield Text(f" {line}")
            yield Text("─" * width, style=palette.border)
            return
        band = palette.code_surface
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=PygmentsSyntaxTheme(pygments_style(theme)),
            background_color=band,
            word_wrap=True,
            padding=(0, 1),
        )
        lines = console.render_lines(syntax, options.update(width=width), pad=True)
        yield Text("▄" * width, style=band)
        for index, line in enumerate(lines):
            row = Text()
            for segment in line:
                if segment.text:
                    row.append(segment.text, style=segment.style)
            if index == 0 and lang:
                used = len(row.plain.rstrip())
                if used + len(lang) + 3 <= width:
                    row = row[: width - len(lang) - 1]
                    row.append(lang, style=f"{palette.faint} on {band}")
                    row.append(" ", style=f"on {band}")
            yield row
        yield Text("▀" * width, style=band)


class _ChatBlockQuote(BlockQuote):
    chat_theme: ChatTheme

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 2)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bar = Segment("▎ ", console.get_style(self.chat_theme.palette.agent))
        for line in lines:
            yield bar
            yield from line
            yield Segment.line()


class _ChatListItem(ListItem):
    """`• item` and `1. item` in violet (Rich wrote ` • ` and a number with no dot)."""

    chat_theme: ChatTheme

    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 2)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bullet = Segment("• ", console.get_style(self.chat_theme.palette.agent))
        padding = Segment("  ")
        for index, line in enumerate(lines):
            yield bullet if index == 0 else padding
            yield from line
            yield Segment.line()

    def render_number(self, console: Console, options: ConsoleOptions, number: int, last_number: int) -> RenderResult:
        width = len(str(last_number)) + 2
        render_options = options.update(width=options.max_width - width)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        numeral = Segment(f"{number}.".ljust(width), console.get_style(self.chat_theme.palette.agent))
        padding = Segment(" " * width)
        for index, line in enumerate(lines):
            yield numeral if index == 0 else padding
            yield from line
            yield Segment.line()


class _ChatTable(TableElement):
    chat_theme: ChatTheme

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        palette = self.chat_theme.palette
        table = Table(box=box.ROUNDED, border_style=palette.border, show_edge=True, padding=(0, 1))
        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize(f"bold {palette.agent}")
                table.add_column(heading)
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*[element.content for element in row.cells])
        yield table


_ELEMENTS: dict = {}


def _elements_for(theme: ChatTheme) -> dict:
    """The element classes for one theme, made once (the theme is held, so its id is not reused)."""
    cached = _ELEMENTS.get(id(theme))
    if cached is not None and cached[0] is theme:
        return cached[1]
    attrs = {"chat_theme": theme}
    elements = {
        **Markdown.elements,
        "heading_open": type("Heading", (_ChatHeading,), attrs),
        "fence": type("CodeBlock", (_ChatCodeBlock,), attrs),
        "code_block": type("CodeBlock", (_ChatCodeBlock,), attrs),
        "blockquote_open": type("BlockQuote", (_ChatBlockQuote,), attrs),
        "table_open": type("TableElement", (_ChatTable,), attrs),
        "list_item_open": type("ListItem", (_ChatListItem,), attrs),
    }
    _ELEMENTS[id(theme)] = (theme, elements)
    return elements


class ChatMarkdown(Markdown):
    """Rich's Markdown with the chat's headings, code bands, tables and quotes."""

    def __init__(self, markup: str, theme: ChatTheme, **kwargs) -> None:
        self.elements = _elements_for(theme)
        super().__init__(_link_bare_urls(_tasks(markup)), hyperlinks=True, **kwargs)


def preview_lines(console: Console, markup: str, theme: ChatTheme, width: int, budget: int = 8):
    """The first whole blocks of `markup` that fit in `budget` rendered lines.

    For a restored conversation's short previews. Cutting the rendered lines
    at a fixed count left a table or a code band without its bottom edge;
    this stops before the block that does not fit. A first block taller than
    the budget is cut as before, having nothing else to show. Returns
    `(lines, more)`, `more` when anything was left out.
    """
    from markdown_it import MarkdownIt

    options = console.options.update(width=width)

    def render(text: str):
        return console.render_lines(ChatMarkdown(text, theme), options, pad=False)

    source = markup.split("\n")
    tokens = MarkdownIt().enable("strikethrough").enable("table").parse(markup)
    ends = sorted({token.map[1] for token in tokens if token.level == 0 and token.map})
    best = None
    for end in ends:
        lines = render("\n".join(source[:end]))
        if len(lines) > budget:
            break
        best = (lines, end)
    if best is None:
        lines = render(markup)
        return lines[:budget], len(lines) > budget
    lines, end = best
    return lines, any(line.strip() for line in source[end:])


#: The TypeScript chat's inline pattern (`typescript/src/cli/render.ts`,
#: `INLINE_PATTERN`), alternative for alternative. Only the last one, a bare
#: URL, is acted on here; the others are there so a URL inside inline code, a
#: link or emphasis is left alone exactly where the TypeScript chat leaves it.
INLINE_PATTERN = re.compile(
    r"`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|\*([^*\s][^*]*)\*|(?<![\w])_([^_\s][^_]*)_(?![\w])"
    r"|\[([^\]]+)\]\(([^)\s]+)\)|(https?://[^\s)<>\"'`]+[^\s)<>.,;:!?\"'`])"
)
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s*(#{1,6})\s")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _link_line(line: str) -> str:
    out = []
    rest = line
    while rest:
        m = INLINE_PATTERN.search(rest)
        if not m:
            out.append(rest)
            break
        out.append(rest[: m.start()])
        out.append(f"<{m.group(9)}>" if m.group(9) is not None else m.group(0))
        rest = rest[m.end():]
    return "".join(out)


def _link_bare_urls(markup: str) -> str:
    """Bare URLs as autolinks (`<https://...>`), so Rich draws them as the
    links they are: underlined, in the link colour, and clickable (2026-09-25).

    The TypeScript chat always linked them and this one never did. Only where
    the TypeScript chat reads inline markdown: paragraphs, list items, H3 and
    table cells, not code, quotes, other headings or a table's header row.
    """
    lines = markup.split("\n")
    header_rows = {i - 1 for i, line in enumerate(lines) if i and _TABLE_RULE.match(line) and "|" in lines[i - 1]}
    fenced = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        heading = _HEADING.match(line)
        if (
            fenced
            or line.startswith("    ")
            or line.lstrip().startswith(">")
            or (heading and len(heading.group(1)) != 3)
            or i in header_rows
            or _TABLE_RULE.match(line)
            or re.match(r"^\s*[-*+]\s+✔ ", line)
        ):
            continue
        lines[i] = _link_line(line)
    return "\n".join(lines)


def _tasks(markup: str) -> str:
    """`- [ ] x` and `- [x] x` as the ballot boxes a terminal can show."""
    import re

    markup = re.sub(r"(?m)^(\s*[-*+]\s+)\[ \]\s+", r"\1☐ ", markup)
    return re.sub(r"(?m)^(\s*[-*+]\s+)\[[xX]\]\s+", r"\1✔ ", markup)
