"""
What the terminal shows, kept from what the chat writes (2026-09-25).

WHY. The `/` menu needs rows under the input box, and at the bottom of the
terminal there are none. Growing into them scrolls the terminal: lines go up
into the scrollback, and no escape sequence brings a line back down from
there. The box used to follow the menu up and scroll the screen back down when
it closed, which left blank rows in the history where those lines had been
(the owner's "gap in history after the command menu disappears", 2026-09-25).
Now the menu opens upward over the last rows of the conversation and puts them
back when it closes (`prompt_box.py`), so nothing scrolls. Putting them back
needs to know exactly what they hold, and a terminal will not say, so the chat
keeps this record: every write to stdout and stderr passes through `feed`,
which follows what a terminal does with it (text and its wrapping, colour,
links, cursor moves, erases) closely enough to draw a row again as it was.

Rows are numbered from wherever the record started, not from the top of the
screen, which it cannot see. A cursor position report ties the two together
(`anchor`: the chat asks once as it starts and the box once per prompt), so a
move stops at the top and bottom where the terminal stops it, and a report
that disagrees with the record means something wrote to the terminal behind
its back, so it forgets. A row is KNOWN only when the record saw all of it: a
fresh row the cursor entered below everything written (blank on any
terminal), or a row erased whole. Text printed on a row it never saw leaves
that row unknown, since the rest of it may hold anything. After a resize
(which reflows the screen) or a sequence it cannot follow it forgets
everything, and after an absolute move or a scroll region it also stops
trusting that fresh rows are blank, until the screen scrolls or is cleared.
The menu never covers a row that is not known: it opens downward instead.
While the input box draws itself it is `paused`; the box knows its own rows.

The TypeScript chat keeps the same record (`typescript/src/cli/ui/screen.ts`);
both are held to `tests/fixtures/cli/screen_record.json`.
"""

from __future__ import annotations

import codecs
import math
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.cells import get_character_cell_size

#: Rows kept above the cursor; older ones are forgotten (the menu needs eight).
KEEP_ROWS = 400
#: An escape sequence left open this long is not one; the record gives up on it.
MAX_PENDING = 4096
FLAG_ORDER = (1, 2, 3, 4, 5, 7, 8, 9, 53)
_PARAMS = re.compile(r"^([0-9:;]*)([ -/]*)$")
_ALTERNATE = re.compile(r"(^|;)(1049|1047|47)(;|$)")
INFINITY = math.inf


class _Cell:
    """One column: the character (with any zero-width marks after it; '' for the
    right half of a wide one), its SGR parameters in one canonical order, and
    the OSC 8 link it was written under."""

    __slots__ = ("ch", "sgr", "link")

    def __init__(self, ch: str, sgr: str, link: str) -> None:
        self.ch = ch
        self.sgr = sgr
        self.link = link


Row = List[Optional[_Cell]]
_BLANK = _Cell(" ", "", "")


def _colon_colour(head: int, sub: List[str]) -> str:
    if len(sub) > 1 and sub[1] == "5":
        return f"{head};5;{int(sub[2] or 0) if len(sub) > 2 else 0}"
    if len(sub) > 1 and sub[1] == "2":
        # 38:2:r:g:b, or with a colour space id first: 38:2:id:r:g:b.
        rgb = sub[2:][-3:]
        if len(rgb) == 3:
            return f"{head};2;" + ";".join(str(int(v or 0)) for v in rgb)
    return ""


class _Pen:
    """The graphic rendition in force: what `ESC [ ... m` sets."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.flags: set = set()
        self.fg = self.bg = self.underline_colour = ""
        self._code: Optional[str] = ""

    @property
    def code(self) -> str:
        if self._code is None:
            parts = [str(flag) for flag in FLAG_ORDER if flag in self.flags]
            parts += [value for value in (self.fg, self.bg, self.underline_colour) if value]
            self._code = ";".join(parts)
        return self._code

    def _flag(self, n: int, on: bool) -> None:
        if on:
            self.flags.add(n)
        else:
            self.flags.discard(n)

    def _colour(self, which: int, value: str) -> None:
        if not value:
            return
        if which == 38:
            self.fg = value
        elif which == 48:
            self.bg = value
        else:
            self.underline_colour = value

    def apply(self, params: str) -> None:
        self._code = None
        parts = params.split(";") if params else ["0"]
        i = 0
        while i < len(parts):
            part = parts[i]
            if ":" in part:
                # Sub-parameters: 38:2::r:g:b, 4:3 (an underline style).
                sub = part.split(":")
                head = int(sub[0] or 0)
                if head in (38, 48, 58):
                    self._colour(head, _colon_colour(head, sub))
                elif head == 4:
                    self._flag(4, int(sub[1] or 1) != 0 if len(sub) > 1 else True)
                i += 1
                continue
            n = int(part) if part else 0
            if n in (38, 48, 58):
                if i + 2 < len(parts) and parts[i + 1] == "5":
                    self._colour(n, f"{n};5;{int(parts[i + 2] or 0)}")
                    i += 3
                elif i + 4 < len(parts) and parts[i + 1] == "2":
                    self._colour(n, f"{n};2;{int(parts[i + 2] or 0)};{int(parts[i + 3] or 0)};{int(parts[i + 4] or 0)}")
                    i += 5
                else:
                    break  # malformed: the rest is not read, as terminals do
                continue
            if n == 0:
                self.reset()
                self._code = None
            elif n in FLAG_ORDER:
                self._flag(n, True)
            elif n == 6:
                self._flag(5, True)
            elif n == 21:
                self._flag(4, True)
            elif n == 22:
                self._flag(1, False)
                self._flag(2, False)
            elif n == 23:
                self._flag(3, False)
            elif n == 24:
                self._flag(4, False)
            elif n == 25:
                self._flag(5, False)
            elif n == 27:
                self._flag(7, False)
            elif n == 28:
                self._flag(8, False)
            elif n == 29:
                self._flag(9, False)
            elif n == 55:
                self._flag(53, False)
            elif 30 <= n <= 37 or 90 <= n <= 97:
                self.fg = str(n)
            elif n == 39:
                self.fg = ""
            elif 40 <= n <= 47 or 100 <= n <= 107:
                self.bg = str(n)
            elif n == 49:
                self.bg = ""
            elif n == 59:
                self.underline_colour = ""
            i += 1
        self._code = None


class ScreenRecord:
    """The rows the terminal shows around the cursor (module docstring)."""

    def __init__(self, columns: Callable[[], int], height: Callable[[], int] = lambda: 24) -> None:
        #: Set while something else (the input box) draws; nothing is recorded.
        self.paused = False
        self._columns = columns
        self._height = height
        self._rows: Dict[int, Row] = {0: []}  # the chat starts on a fresh line
        #: Rows at or after this one that hold nothing are blank; before it, unknown.
        self._blank_from = INFINITY
        self._row = 0
        self._col = 0
        #: The lowest row the cursor has been on. A row entered below it is fresh,
        #: so blank; infinite while that is not true (the cursor may be anywhere).
        self._lowest: float = 0
        #: The row at the top of the screen, once a cursor position report said.
        self._screen_top: Optional[int] = None
        self._saved: Optional[Tuple[int, int]] = None
        self._pen = _Pen()
        self._link = ""
        self._pending = ""
        #: An erase of the whole screen came last, so a move home lands on a known row.
        self._cleared = False
        #: The alternate screen is up; the main one, and this record of it, wait underneath.
        self._alternate = False
        self._size = self._size_key()

    # -- what the chat asks -------------------------------------------------------

    def feed(self, text: str) -> None:
        """Follow one write to the terminal."""
        if self.paused or not text:
            return
        self._check_size()
        data = self._pending + text
        self._pending = ""
        i, n = 0, len(data)
        while i < n:
            ch = data[i]
            code = ord(ch)
            if code == 0x1B:
                used = self._escape(data, i)
                if used < 0:
                    rest = data[i:]
                    if len(rest) > MAX_PENDING:
                        self.forget()
                    else:
                        self._pending = rest
                    return
                i += used
                continue
            if code < 0x20 or code == 0x7F:
                if not self._alternate:
                    self._control(code)
                i += 1
                continue
            j = i + 1
            while j < n:
                nxt = ord(data[j])
                if nxt == 0x1B or nxt < 0x20 or nxt == 0x7F:
                    break
                j += 1
            if not self._alternate:
                self._print(data[i:j])
            i = j

    def forget(self) -> None:
        """Forget every row. The cursor is still after everything written (a
        resize, or writes the record missed, leave it there), so rows entered
        below it are still fresh."""
        self._rows.clear()
        self._blank_from = INFINITY
        self._lowest = self._row
        self._screen_top = None
        self._saved = None
        self._cleared = False

    def clear_screen(self) -> None:
        """The screen was cleared and the cursor put at its top (ctrl+l in the box)."""
        self._rows.clear()
        self._row = self._col = 0
        self._lowest = 0
        self._blank_from = 0
        self._screen_top = 0
        self._saved = None
        self._cleared = False

    def anchor(self, screen_row: int) -> None:
        """The terminal reported the cursor on `screen_row` (1-based). A report
        that disagrees with the record means writes it did not see: it forgets."""
        self._check_size()
        if self._screen_top is not None and self._row - self._screen_top + 1 != screen_row:
            self.forget()
        self._screen_top = self._row - (screen_row - 1)

    def scrolled(self, rows: int) -> None:
        """The terminal scrolled by `rows` while the record was paused: the
        input box grew past the bottom row. Without this, the next report would
        disagree."""
        if self._screen_top is not None and rows > 0:
            self._screen_top += rows

    @property
    def cursor(self) -> Tuple[int, int]:
        """The cursor's row and column, in the record's own numbering (tests)."""
        return self._row, self._col

    def rows_above(self, count: int) -> Optional[List[str]]:
        """The `count` rows directly above the cursor, oldest first, drawn so
        that writing one at the start of a blank row shows it as it was; None
        when any of them is unknown."""
        return self._above(count, _render)

    def plain_rows_above(self, count: int) -> Optional[List[str]]:
        """The same rows as plain text (tests, and the fixture both SDKs share)."""
        return self._above(count, _plain)

    def _above(self, count: int, draw: Callable[[Row], str]) -> Optional[List[str]]:
        self._check_size()
        out: List[str] = []
        for r in range(self._row - count, self._row):
            row = self._rows.get(r)
            if row is not None:
                out.append(draw(row))
            elif r >= self._blank_from:
                out.append("")
            else:
                return None
        return out

    # -- the stream -----------------------------------------------------------------

    def _drift(self) -> None:
        """Forget every row, and where the cursor is: after an absolute move, or a scroll region."""
        self.forget()
        self._lowest = INFINITY

    def _escape(self, data: str, i: int) -> int:
        """One escape sequence at `i`: how many characters it took, or -1 when it is not complete yet."""
        n = len(data)
        if i + 1 >= n:
            return -1
        kind = data[i + 1]
        if kind == "[":
            j = i + 2
            while j < n and 0x20 <= ord(data[j]) <= 0x3F:
                j += 1
            if j >= n:
                return -1
            if 0x40 <= ord(data[j]) <= 0x7E:
                self._csi(data[i + 2:j], data[j])
            return j + 1 - i
        if kind in "]PX^_":
            # A string, ended by BEL or ST (ESC \).
            j = i + 2
            while j < n:
                c = data[j]
                if c == "\x07":
                    if kind == "]":
                        self._osc(data[i + 2:j])
                    return j + 1 - i
                if c == "\x1b":
                    if j + 1 >= n:
                        return -1
                    if data[j + 1] == "\\":
                        if kind == "]":
                            self._osc(data[i + 2:j])
                        return j + 2 - i
                j += 1
            return -1
        if kind in "()*+-./#%":
            return 3 if i + 2 < n else -1
        if self._alternate:
            return 2
        if kind == "7":
            self._saved = (self._row, self._col)
        elif kind == "8":
            if self._saved is not None:
                self._row, self._col = self._saved
        elif kind == "D":
            self._line_feed(False)
        elif kind == "E":
            self._line_feed(True)
        elif kind == "M":  # reverse index: at the top of the screen it scrolls the screen down
            self._drift()
        elif kind == "c":  # a full reset
            self._pen.reset()
            self._link = ""
            self._drift()
        self._cleared = False
        return 2

    def _csi(self, body: str, final: str) -> None:
        private = body[0] if body[:1] in ("<", "=", ">", "?") else ""
        rest = body[1:] if private else body
        match = _PARAMS.match(rest)
        if not match:
            return
        params, intermediates = match.group(1), match.group(2)
        if private:
            if private == "?" and final in ("h", "l") and _ALTERNATE.search(params):
                self._alternate = final == "h"
            return  # other private modes change nothing drawn
        if intermediates or self._alternate:
            return
        if final == "m":
            self._pen.apply(params)
            return
        nums = [int(p.split(":")[0] or 0) if p else 0 for p in params.split(";")]
        n = nums[0] if nums else 0
        was_cleared = self._cleared
        self._cleared = False
        columns = self._columns()
        at = min(self._col, columns - 1)
        if final == "A":
            self._move_up(n or 1)
        elif final == "B":
            self._move_down(n or 1)
        elif final == "C":
            self._col = min(columns - 1, at + (n or 1))
        elif final == "D":
            self._col = max(0, at - (n or 1))
        elif final == "E":
            self._move_down(n or 1)
            self._col = 0
        elif final == "F":
            self._move_up(n or 1)
            self._col = 0
        elif final in ("G", "`"):
            self._col = min(columns - 1, max(1, n or 1) - 1)
        elif final in ("H", "f"):
            second = nums[1] if len(nums) > 1 else 0
            if (n or 1) == 1 and (second or 1) == 1 and was_cleared:
                self.clear_screen()
            else:
                self._drift()
                self._col = min(columns - 1, max(1, second or 1) - 1)
        elif final == "J":
            if n == 0:
                self._erase_below()
            elif n == 2:
                # The screen is blank, and a move home next puts the cursor on its first row.
                self._drift()
                self._cleared = True
            elif n == 3:
                self._cleared = was_cleared
            else:
                self.forget()
        elif final == "K":
            if n == 0:
                self._clear_cells(self._row, at, columns)
            elif n == 1:
                self._clear_cells(self._row, 0, at + 1)
            else:
                self._clear_cells(self._row, 0, columns)
        elif final == "X":
            self._clear_cells(self._row, at, at + (n or 1))
        elif final == "P":
            row = self._rows.get(self._row)
            if row is not None:
                del row[at:at + (n or 1)]
        elif final == "@":
            row = self._rows.get(self._row)
            if row is not None and at <= len(row):
                row[at:at] = [None] * (n or 1)
                del row[columns:]
        elif final == "s":
            self._saved = (self._row, self._col)
        elif final == "u":
            if self._saved is not None:
                self._row, self._col = self._saved
        elif final in ("L", "M", "S", "T", "d", "r"):
            self._drift()
        # Reports and queries (n, c, t, q): nothing drawn.

    def _osc(self, body: str) -> None:
        # OSC 8 ; params ; target: a link, or its end when the target is empty.
        if body.startswith("8;"):
            second = body.find(";", 2)
            self._link = body[second + 1:] if second >= 0 else ""

    def _control(self, code: int) -> None:
        self._cleared = False
        if code == 0x0D:
            self._col = 0
        elif code in (0x0A, 0x0B, 0x0C):
            # A terminal's output translation makes a line feed a new line.
            self._line_feed(True)
        elif code == 0x08:
            self._col = max(0, min(self._col, self._columns() - 1) - 1)
        elif code == 0x09:
            self._col = min(self._columns() - 1, (self._col // 8 + 1) * 8)

    def _print(self, text: str) -> None:
        self._cleared = False
        columns = self._columns()
        for ch in text:
            width = get_character_cell_size(ch)
            if width == 0:
                row = self._rows.get(self._row)
                if row is not None:
                    before = _cell(row, self._col - 1)
                    if before is not None and before.ch == "":
                        before = _cell(row, self._col - 2)
                    if before is not None:
                        before.ch += ch
                continue
            if self._col + width > columns:
                self._line_feed(True)
            # Text on a row the record never saw leaves it unknown (module docstring).
            row = self._rows.get(self._row)
            if row is None and self._row >= self._blank_from:
                row = self._row_at(self._row)
            if row is not None:
                self._split(row, self._col)
                self._split(row, self._col + width - 1)
                _put(row, self._col, _Cell(ch, self._pen.code, self._link))
                if width == 2:
                    _put(row, self._col + 1, _Cell("", self._pen.code, self._link))
            self._col += width

    # -- rows -----------------------------------------------------------------------

    @staticmethod
    def _split(row: Row, col: int) -> None:
        """Writing over half of a wide character blanks its other half."""
        cell = _cell(row, col)
        if cell is None:
            return
        if cell.ch == "" and _cell(row, col - 1) is not None:
            row[col - 1] = None
        else:
            after = _cell(row, col + 1)
            if after is not None and after.ch == "":
                row[col + 1] = None

    def _row_at(self, index: int) -> Row:
        row = self._rows.get(index)
        if row is None:
            row = []
            self._rows[index] = row
        return row

    def _line_feed(self, carriage_return: bool) -> None:
        self._row += 1
        if carriage_return:
            self._col = 0
        # Past the last row the screen scrolls: the top moves down with it, and
        # the row that appears is blank.
        scrolled = self._screen_top is not None and self._row > self._screen_top + self._height() - 1
        if scrolled:
            self._screen_top = self._row - self._height() + 1
        self._enter(self._row, scrolled)
        if len(self._rows) > KEEP_ROWS + 50:
            for index in sorted(self._rows)[: len(self._rows) - KEEP_ROWS]:
                del self._rows[index]

    def _enter(self, index: int, blank: bool = False) -> None:
        """The cursor came onto `index`: a row below everything it has been on is fresh, so blank."""
        if index not in self._rows and (blank or index > self._lowest or index >= self._blank_from):
            self._rows[index] = []
        if math.isfinite(self._lowest) and index > self._lowest:
            self._lowest = index

    def _move_up(self, n: int) -> None:
        # A known top stops the cursor where the terminal stops it.
        self._row = max(self._screen_top, self._row - n) if self._screen_top is not None else self._row - n

    def _move_down(self, n: int) -> None:
        if self._screen_top is None:
            if self._row + n > self._lowest:
                # Past everything written, with the bottom of the screen not
                # known: the terminal may have stopped the cursor at its edge.
                self._row += n
                self.forget()
                return
            self._row += n
            return
        target = min(self._screen_top + self._height() - 1, self._row + n)
        while self._row < target:
            self._row += 1
            self._enter(self._row)

    def _erase_below(self) -> None:
        columns = self._columns()
        self._clear_cells(self._row, columns - 1 if self._col >= columns else self._col, columns)
        for index in [index for index in self._rows if index > self._row]:
            del self._rows[index]
        # Everything below the cursor is blank now.
        self._lowest = self._row

    def _clear_cells(self, index: int, start: int, end: int) -> None:
        row = self._rows.get(index)
        if row is None:
            # A row not seen before is known once all of it is blank.
            if start == 0 and end >= self._columns():
                self._row_at(index)
            return
        first = _cell(row, start)
        if first is not None and first.ch == "" and start > 0:
            row[start - 1] = None
        last = _cell(row, end)
        if last is not None and last.ch == "":
            row[end] = None
        for c in range(start, min(end, len(row))):
            row[c] = None

    def _size_key(self) -> Tuple[int, int]:
        return self._columns(), self._height()

    def _check_size(self) -> None:
        """A resize reflows what the terminal shows, in ways that differ by terminal."""
        size = self._size_key()
        if size != self._size:
            self._size = size
            self.forget()


def _cell(row: Row, col: int) -> Optional[_Cell]:
    return row[col] if 0 <= col < len(row) else None


def _put(row: Row, col: int, cell: _Cell) -> None:
    if col >= len(row):
        row.extend([None] * (col + 1 - len(row)))
    row[col] = cell


def _visible_end(row: Row) -> int:
    end = len(row)
    while end > 0:
        cell = row[end - 1]
        if cell is not None and (cell.sgr or cell.link or cell.ch != " "):
            break
        end -= 1
    return end


def _plain(row: Row) -> str:
    return "".join((row[c] or _BLANK).ch for c in range(_visible_end(row)))


def _render(row: Row) -> str:
    """A row as text and escapes that draw it again from its first column."""
    out = "\x1b[0m"
    sgr = link = ""
    for c in range(_visible_end(row)):
        cell = row[c] or _BLANK
        if cell.ch == "":
            continue
        if cell.link != link:
            if link:
                out += "\x1b]8;;\x1b\\"
            if cell.link:
                out += f"\x1b]8;;{cell.link}\x1b\\"
            link = cell.link
        if cell.sgr != sgr:
            out += f"\x1b[0;{cell.sgr}m" if cell.sgr else "\x1b[0m"
            sgr = cell.sgr
        out += cell.ch
    if link:
        out += "\x1b]8;;\x1b\\"
    if sgr:
        out += "\x1b[0m"
    return out


# -- recording the chat's streams --------------------------------------------------------


class _BinaryTap:
    """A stream's `buffer`, recorded: prompt_toolkit writes its bytes there."""

    def __init__(self, buffer: Any, screen: ScreenRecord) -> None:
        self._buffer = buffer
        self._screen = screen
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def write(self, data: bytes) -> int:
        try:
            self._screen.feed(self._decoder.decode(bytes(data)))
        except Exception:
            self._screen.forget()
        return self._buffer.write(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._buffer, name)


class _Tap:
    """A text stream whose writes the record sees first. Everything else is the
    stream's own, so rich, prompt_toolkit and print() treat it as the terminal."""

    def __init__(self, stream: Any, screen: ScreenRecord) -> None:
        self._stream = stream
        self._screen = screen
        buffer = getattr(stream, "buffer", None)
        self._binary = _BinaryTap(buffer, screen) if buffer is not None else None

    def write(self, text: str) -> int:
        try:
            self._screen.feed(text)
        except Exception:
            self._screen.forget()
        return self._stream.write(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    @property
    def buffer(self) -> Any:
        if self._binary is None:
            raise AttributeError("buffer")
        return self._binary

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def record_screen(columns: Callable[[], int], height: Callable[[], int]) -> Tuple[ScreenRecord, Callable[[], None]]:
    """Record everything written to stdout and stderr (which share the
    terminal) until the returned `stop()`."""
    screen = ScreenRecord(columns, height)
    saved = (sys.stdout, sys.stderr)
    sys.stdout = _Tap(sys.stdout, screen)  # type: ignore[assignment]
    sys.stderr = _Tap(sys.stderr, screen)  # type: ignore[assignment]

    def stop() -> None:
        sys.stdout, sys.stderr = saved

    return screen, stop
