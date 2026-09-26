/**
 * What the terminal shows, kept from what the chat writes (2026-09-25).
 *
 * WHY. The `/` menu needs rows under the input box, and at the bottom of the
 * terminal there are none. Growing into them scrolls the terminal: lines go
 * up into the scrollback, and no escape sequence brings a line back down from
 * there. The box used to follow the menu up and scroll the screen back down
 * when it closed, which left blank rows in the history where those lines had
 * been (the owner's "gap in history after the command menu disappears",
 * 2026-09-25). Now the menu opens upward over the last rows of the
 * conversation and puts them back when it closes (`input.ts`), so nothing
 * scrolls. Putting them back needs to know exactly what they hold, and a
 * terminal will not say, so the chat keeps this record: every write to
 * stdout and stderr passes through `feed`, which follows what a terminal does
 * with it (text and its wrapping, colour, links, cursor moves, erases)
 * closely enough to draw a row again as it was.
 *
 * Rows are numbered from wherever the record started, not from the top of the
 * screen, which it cannot see. A cursor position report ties the two together
 * (`anchor`: the chat asks once as it starts and the box once per prompt), so
 * a move stops at the top and bottom where the terminal stops it, and a
 * report that disagrees with the record means something wrote to the terminal
 * behind its back, so it forgets. A row is KNOWN only when the record saw all
 * of it: a fresh row the cursor entered below everything written (blank on
 * any terminal), or a row erased whole. Text printed on a row it never saw
 * leaves that row unknown, since the rest of it may hold anything. After a
 * resize (which reflows the screen) or a sequence it cannot follow it forgets
 * everything, and after an absolute move or a scroll region it also stops
 * trusting that fresh rows are blank, until the screen scrolls or is cleared.
 * The menu never covers a row that is not known: it opens downward instead.
 * While the input box draws itself it is `paused`; the box knows its own rows.
 *
 * The Python chat keeps the same record (`python/webagents/cli/ui/screen.py`);
 * both are held to `python/tests/fixtures/cli/screen_record.json`.
 */

import { charWidth } from './ansi';

interface Cell {
  /** The character, with any zero-width marks after it; '' for the right half of a wide one. */
  ch: string;
  /** SGR parameters it was written with, in one canonical order ('' = plain). */
  sgr: string;
  /** The OSC 8 link it was written under ('' = none). */
  link: string;
}

type Row = Array<Cell | undefined>;

/** Rows kept above the cursor; older ones are forgotten (the menu needs eight). */
const KEEP_ROWS = 400;
/** An escape sequence left open this long is not one; the record gives up on it. */
const MAX_PENDING = 4096;
const FLAG_ORDER = [1, 2, 3, 4, 5, 7, 8, 9, 53];

/** The graphic rendition in force: what `\x1b[...m` sets. */
class Pen {
  private flags = new Set<number>();
  private fg = '';
  private bg = '';
  private underlineColour = '';
  private cached: string | null = '';

  get code(): string {
    if (this.cached === null) {
      const parts = FLAG_ORDER.filter((flag) => this.flags.has(flag)).map(String);
      if (this.fg) parts.push(this.fg);
      if (this.bg) parts.push(this.bg);
      if (this.underlineColour) parts.push(this.underlineColour);
      this.cached = parts.join(';');
    }
    return this.cached;
  }

  reset(): void {
    this.flags.clear();
    this.fg = this.bg = this.underlineColour = '';
    this.cached = '';
  }

  apply(params: string): void {
    this.cached = null;
    const parts = params === '' ? ['0'] : params.split(';');
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i];
      if (part.includes(':')) {
        // Sub-parameters: 38:2::r:g:b, 4:3 (an underline style).
        const sub = part.split(':');
        const head = Number(sub[0] || 0);
        if (head === 38 || head === 48 || head === 58) this.colour(head, colonColour(head, sub));
        else if (head === 4) this.flag(4, Number(sub[1] || 1) !== 0);
        continue;
      }
      const n = part === '' ? 0 : Number(part);
      if (n === 38 || n === 48 || n === 58) {
        if (parts[i + 1] === '5' && parts[i + 2] !== undefined) {
          this.colour(n, `${n};5;${Number(parts[i + 2])}`);
          i += 2;
        } else if (parts[i + 1] === '2' && i + 4 < parts.length) {
          this.colour(n, `${n};2;${Number(parts[i + 2])};${Number(parts[i + 3])};${Number(parts[i + 4])}`);
          i += 4;
        } else {
          break; // malformed: the rest is not read, as terminals do
        }
        continue;
      }
      if (n === 0) this.reset();
      else if (FLAG_ORDER.includes(n)) this.flag(n, true);
      else if (n === 6) this.flag(5, true);
      else if (n === 21) this.flag(4, true);
      else if (n === 22) {
        this.flag(1, false);
        this.flag(2, false);
      } else if (n === 23) this.flag(3, false);
      else if (n === 24) this.flag(4, false);
      else if (n === 25) this.flag(5, false);
      else if (n === 27) this.flag(7, false);
      else if (n === 28) this.flag(8, false);
      else if (n === 29) this.flag(9, false);
      else if (n === 55) this.flag(53, false);
      else if ((n >= 30 && n <= 37) || (n >= 90 && n <= 97)) this.fg = String(n);
      else if (n === 39) this.fg = '';
      else if ((n >= 40 && n <= 47) || (n >= 100 && n <= 107)) this.bg = String(n);
      else if (n === 49) this.bg = '';
      else if (n === 59) this.underlineColour = '';
    }
    this.cached = null;
  }

  private flag(n: number, on: boolean): void {
    if (on) this.flags.add(n);
    else this.flags.delete(n);
  }

  private colour(which: number, value: string): void {
    if (!value) return;
    if (which === 38) this.fg = value;
    else if (which === 48) this.bg = value;
    else this.underlineColour = value;
  }
}

function colonColour(head: number, sub: string[]): string {
  if (sub[1] === '5') return `${head};5;${Number(sub[2] || 0)}`;
  if (sub[1] === '2') {
    // 38:2:r:g:b, or with a colour space id first: 38:2:id:r:g:b.
    const rgb = sub.slice(2).slice(-3);
    if (rgb.length === 3) return `${head};2;${rgb.map((v) => Number(v || 0)).join(';')}`;
  }
  return '';
}

export class ScreenRecord {
  /** Set while something else (the input box) draws; nothing is recorded. */
  paused = false;

  private rows = new Map<number, Row>();
  /** Rows at or after this one that hold nothing are blank; before it, unknown. */
  private blankFrom = Number.POSITIVE_INFINITY;
  private row = 0;
  private col = 0;
  /**
   * The lowest row the cursor has been on. A row entered below it is fresh,
   * so blank; infinite while that is not true (the cursor may be anywhere).
   */
  private lowest = 0;
  /** The row at the top of the screen, once a cursor position report said. */
  private screenTop: number | null = null;
  private saved: { row: number; col: number } | null = null;
  private pen = new Pen();
  private link = '';
  private pending = '';
  /** An erase of the whole screen came last, so a move home lands on a known row. */
  private cleared = false;
  /** The alternate screen is up; the main one, and this record of it, wait underneath. */
  private alternate = false;
  private size: string;

  constructor(
    private readonly columns: () => number,
    private readonly height: () => number = () => 24,
  ) {
    this.size = this.sizeKey();
    // The chat starts on a fresh line.
    this.rows.set(0, []);
  }

  /** Follow one write to the terminal. */
  feed(text: string): void {
    if (this.paused || !text) return;
    this.checkSize();
    const data = this.pending + text;
    this.pending = '';
    let i = 0;
    while (i < data.length) {
      const code = data.charCodeAt(i);
      if (code === 0x1b) {
        const used = this.escape(data, i);
        if (used < 0) {
          const rest = data.slice(i);
          if (rest.length > MAX_PENDING) this.forget();
          else this.pending = rest;
          return;
        }
        i += used;
        continue;
      }
      if (code < 0x20 || code === 0x7f) {
        if (!this.alternate) this.control(code);
        i += 1;
        continue;
      }
      let j = i + 1;
      while (j < data.length) {
        const next = data.charCodeAt(j);
        if (next === 0x1b || next < 0x20 || next === 0x7f) break;
        j += 1;
      }
      if (!this.alternate) this.print(data.slice(i, j));
      i = j;
    }
  }

  /**
   * Forget every row. The cursor is still after everything written (a resize,
   * or writes the record missed, leave it there), so rows entered below it
   * are still fresh.
   */
  forget(): void {
    this.rows.clear();
    this.blankFrom = Number.POSITIVE_INFINITY;
    this.lowest = this.row;
    this.screenTop = null;
    this.saved = null;
    this.cleared = false;
  }

  /** Forget every row, and where the cursor is: after an absolute move, or a scroll region. */
  private drift(): void {
    this.forget();
    this.lowest = Number.POSITIVE_INFINITY;
  }

  /** The screen was cleared and the cursor put at its top (what the input box's ctrl+l does). */
  clearScreen(): void {
    this.rows.clear();
    this.row = this.col = 0;
    this.lowest = 0;
    this.blankFrom = 0;
    this.screenTop = 0;
    this.saved = null;
    this.cleared = false;
  }

  /**
   * The terminal reported the cursor on `screenRow` (1-based). A report that
   * disagrees with the record means writes it did not see: it forgets.
   */
  anchor(screenRow: number): void {
    this.checkSize();
    if (this.screenTop !== null && this.row - this.screenTop + 1 !== screenRow) this.forget();
    this.screenTop = this.row - (screenRow - 1);
  }

  /**
   * The terminal scrolled by `rows` while the record was paused: the input box
   * grew past the bottom row. Without this, the next report would disagree.
   */
  scrolled(rows: number): void {
    if (this.screenTop !== null && rows > 0) this.screenTop += rows;
  }

  /** The cursor's row and column, in the record's own numbering (tests). */
  get cursor(): { row: number; col: number } {
    return { row: this.row, col: this.col };
  }

  /**
   * The `count` rows directly above the cursor, oldest first, drawn so that
   * writing one at the start of a blank row shows it as it was; null when any
   * of them is unknown.
   */
  rowsAbove(count: number): string[] | null {
    this.checkSize();
    const out: string[] = [];
    for (let r = this.row - count; r < this.row; r += 1) {
      const row = this.rows.get(r);
      if (row) out.push(render(row));
      else if (r >= this.blankFrom) out.push('');
      else return null;
    }
    return out;
  }

  /** The same rows as plain text (tests, and the fixture both SDKs share). */
  plainRowsAbove(count: number): string[] | null {
    this.checkSize();
    const out: string[] = [];
    for (let r = this.row - count; r < this.row; r += 1) {
      const row = this.rows.get(r);
      if (row) out.push(plain(row));
      else if (r >= this.blankFrom) out.push('');
      else return null;
    }
    return out;
  }

  // -- the stream -------------------------------------------------------------

  /** One escape sequence at `i`: how many characters it took, or -1 when it is not complete yet. */
  private escape(data: string, i: number): number {
    if (i + 1 >= data.length) return -1;
    const kind = data[i + 1];
    if (kind === '[') {
      let j = i + 2;
      while (j < data.length) {
        const c = data.charCodeAt(j);
        if (c < 0x20 || c > 0x3f) break;
        j += 1;
      }
      if (j >= data.length) return -1;
      const final = data.charCodeAt(j);
      if (final >= 0x40 && final <= 0x7e) this.csi(data.slice(i + 2, j), data[j]);
      return j + 1 - i;
    }
    if (kind === ']' || kind === 'P' || kind === 'X' || kind === '^' || kind === '_') {
      // A string, ended by BEL or ST (ESC \).
      for (let j = i + 2; j < data.length; j += 1) {
        const c = data.charCodeAt(j);
        if (c === 0x07) {
          if (kind === ']') this.osc(data.slice(i + 2, j));
          return j + 1 - i;
        }
        if (c === 0x1b) {
          if (j + 1 >= data.length) return -1;
          if (data[j + 1] === '\\') {
            if (kind === ']') this.osc(data.slice(i + 2, j));
            return j + 2 - i;
          }
        }
      }
      return -1;
    }
    if ('()*+-./#%'.includes(kind)) return i + 2 < data.length ? 3 : -1;
    if (this.alternate) return 2;
    switch (kind) {
      case '7':
        this.saved = { row: this.row, col: this.col };
        break;
      case '8':
        if (this.saved) ({ row: this.row, col: this.col } = this.saved);
        break;
      case 'D':
        this.lineFeed(false);
        break;
      case 'E':
        this.lineFeed(true);
        break;
      case 'M': // reverse index: at the top of the screen it scrolls the screen down
        this.drift();
        break;
      case 'c': // a full reset
        this.pen.reset();
        this.link = '';
        this.drift();
        break;
      default:
        break;
    }
    this.cleared = false;
    return 2;
  }

  private csi(body: string, final: string): void {
    const privateMarker = /^[<=>?]/.test(body) ? body[0] : '';
    const rest = privateMarker ? body.slice(1) : body;
    const match = /^([0-9:;]*)([ -/]*)$/.exec(rest);
    if (!match) return;
    const [, params, intermediates] = match;
    if (privateMarker) {
      if (privateMarker === '?' && (final === 'h' || final === 'l') && /(^|;)(1049|1047|47)(;|$)/.test(params)) {
        this.alternate = final === 'h';
      }
      return; // other private modes change nothing drawn
    }
    if (intermediates || this.alternate) return;
    if (final === 'm') {
      this.pen.apply(params);
      return;
    }
    const nums = params.split(';').map((p) => (p === '' ? 0 : Number(p.split(':')[0])));
    const n = nums[0] || 0;
    const wasCleared = this.cleared;
    this.cleared = false;
    const columns = this.columns();
    const at = Math.min(this.col, columns - 1);
    switch (final) {
      case 'A':
        this.moveUp(n || 1);
        break;
      case 'B':
        this.moveDown(n || 1);
        break;
      case 'C':
        this.col = Math.min(columns - 1, at + (n || 1));
        break;
      case 'D':
        this.col = Math.max(0, at - (n || 1));
        break;
      case 'E':
        this.moveDown(n || 1);
        this.col = 0;
        break;
      case 'F':
        this.moveUp(n || 1);
        this.col = 0;
        break;
      case 'G':
      case '`':
        this.col = Math.min(columns - 1, Math.max(1, n || 1) - 1);
        break;
      case 'H':
      case 'f':
        if ((nums[0] || 1) === 1 && (nums[1] || 1) === 1 && wasCleared) this.clearScreen();
        else {
          this.drift();
          this.col = Math.min(columns - 1, Math.max(1, nums[1] || 1) - 1);
        }
        break;
      case 'J':
        if (n === 0) this.eraseBelow();
        else if (n === 2) {
          // The screen is blank, and a move home next puts the cursor on its first row.
          this.drift();
          this.cleared = true;
        } else if (n === 3) this.cleared = wasCleared;
        else this.forget();
        break;
      case 'K':
        if (n === 0) this.clearCells(this.row, at, columns);
        else if (n === 1) this.clearCells(this.row, 0, at + 1);
        else this.clearCells(this.row, 0, columns);
        break;
      case 'X':
        this.clearCells(this.row, at, at + (n || 1));
        break;
      case 'P': {
        const row = this.rows.get(this.row);
        if (row) row.splice(at, n || 1);
        break;
      }
      case '@': {
        const row = this.rows.get(this.row);
        if (row) {
          row.splice(at, 0, ...new Array<Cell | undefined>(n || 1).fill(undefined));
          row.length = Math.min(row.length, columns);
        }
        break;
      }
      case 's':
        this.saved = { row: this.row, col: this.col };
        break;
      case 'u':
        if (this.saved) ({ row: this.row, col: this.col } = this.saved);
        break;
      case 'L':
      case 'M':
      case 'S':
      case 'T':
      case 'd':
      case 'r':
        this.drift();
        break;
      default:
        // Reports and queries (n, c, t, q): nothing drawn.
        break;
    }
  }

  private osc(body: string): void {
    // OSC 8 ; params ; target: a link, or its end when the target is empty.
    if (body.startsWith('8;')) {
      const target = body.slice(body.indexOf(';', 2) + 1);
      this.link = body.indexOf(';', 2) >= 0 ? target : '';
    }
  }

  private control(code: number): void {
    this.cleared = false;
    switch (code) {
      case 0x0d:
        this.col = 0;
        break;
      case 0x0a:
      case 0x0b:
      case 0x0c:
        // A terminal's output translation makes a line feed a new line.
        this.lineFeed(true);
        break;
      case 0x08:
        this.col = Math.max(0, Math.min(this.col, this.columns() - 1) - 1);
        break;
      case 0x09:
        this.col = Math.min(this.columns() - 1, (Math.floor(this.col / 8) + 1) * 8);
        break;
      default:
        break;
    }
  }

  private print(text: string): void {
    this.cleared = false;
    const columns = this.columns();
    for (const ch of text) {
      const width = charWidth(ch.codePointAt(0) ?? 0);
      if (width === 0) {
        const row = this.rows.get(this.row);
        const before = row?.[this.col - 1]?.ch === '' ? row?.[this.col - 2] : row?.[this.col - 1];
        if (before) before.ch += ch;
        continue;
      }
      if (this.col + width > columns) this.lineFeed(true);
      // Text on a row the record never saw leaves it unknown (file comment).
      const row = this.rows.get(this.row) ?? (this.row >= this.blankFrom ? this.rowAt(this.row) : undefined);
      if (row) {
        this.split(row, this.col);
        this.split(row, this.col + width - 1);
        row[this.col] = { ch, sgr: this.pen.code, link: this.link };
        if (width === 2) row[this.col + 1] = { ch: '', sgr: this.pen.code, link: this.link };
      }
      this.col += width;
    }
  }

  // -- rows -------------------------------------------------------------------

  /** Writing over half of a wide character blanks its other half. */
  private split(row: Row, col: number): void {
    const cell = row[col];
    if (!cell) return;
    if (cell.ch === '' && row[col - 1]) row[col - 1] = undefined;
    else if (row[col + 1]?.ch === '') row[col + 1] = undefined;
  }

  private rowAt(index: number): Row {
    let row = this.rows.get(index);
    if (!row) {
      row = [];
      this.rows.set(index, row);
    }
    return row;
  }

  private lineFeed(carriageReturn: boolean): void {
    this.row += 1;
    if (carriageReturn) this.col = 0;
    // Past the last row the screen scrolls: the top moves down with it, and
    // the row that appears is blank.
    const scrolled = this.screenTop !== null && this.row > this.screenTop + this.height() - 1;
    if (scrolled) this.screenTop = this.row - this.height() + 1;
    this.enter(this.row, scrolled);
    if (this.rows.size > KEEP_ROWS + 50) {
      for (const index of [...this.rows.keys()].sort((a, b) => a - b).slice(0, this.rows.size - KEEP_ROWS)) {
        this.rows.delete(index);
      }
    }
  }

  /** The cursor came onto `index`: a row below everything it has been on is fresh, so blank. */
  private enter(index: number, blank = false): void {
    if (!this.rows.has(index) && (blank || index > this.lowest || index >= this.blankFrom)) this.rows.set(index, []);
    if (Number.isFinite(this.lowest) && index > this.lowest) this.lowest = index;
  }

  private moveUp(n: number): void {
    // A known top stops the cursor where the terminal stops it.
    this.row = this.screenTop !== null ? Math.max(this.screenTop, this.row - n) : this.row - n;
  }

  private moveDown(n: number): void {
    if (this.screenTop === null) {
      if (this.row + n > this.lowest) {
        // Past everything written, with the bottom of the screen not known: the
        // terminal may have stopped the cursor at its edge.
        this.row += n;
        this.forget();
        return;
      }
      this.row += n;
      return;
    }
    const target = Math.min(this.screenTop + this.height() - 1, this.row + n);
    while (this.row < target) {
      this.row += 1;
      this.enter(this.row);
    }
  }

  private eraseBelow(): void {
    this.clearCells(this.row, this.col >= this.columns() ? this.columns() - 1 : this.col, this.columns());
    for (const index of [...this.rows.keys()]) if (index > this.row) this.rows.delete(index);
    // Everything below the cursor is blank now.
    this.lowest = this.row;
  }

  private clearCells(index: number, from: number, to: number): void {
    const row = this.rows.get(index);
    if (!row) {
      // A row not seen before is known once all of it is blank.
      if (from === 0 && to >= this.columns()) this.rowAt(index);
      return;
    }
    if (row[from]?.ch === '' && from > 0) row[from - 1] = undefined;
    if (row[to]?.ch === '') row[to] = undefined;
    for (let c = from; c < Math.min(to, row.length); c += 1) row[c] = undefined;
  }

  private sizeKey(): string {
    return `${this.columns()}x${this.height()}`;
  }

  /** A resize reflows what the terminal shows, in ways that differ by terminal. */
  private checkSize(): void {
    const size = this.sizeKey();
    if (size !== this.size) {
      this.size = size;
      this.forget();
    }
  }
}

const BLANK: Cell = { ch: ' ', sgr: '', link: '' };

function visibleEnd(row: Row): number {
  let end = row.length;
  while (end > 0) {
    const cell = row[end - 1];
    if (cell && (cell.sgr || cell.link || cell.ch !== ' ')) break;
    end -= 1;
  }
  return end;
}

function plain(row: Row): string {
  let out = '';
  for (let c = 0; c < visibleEnd(row); c += 1) out += (row[c] ?? BLANK).ch;
  return out;
}

/** A row as text and escapes that draw it again from its first column. */
function render(row: Row): string {
  let out = '\x1b[0m';
  let sgr = '';
  let link = '';
  for (let c = 0; c < visibleEnd(row); c += 1) {
    const cell = row[c] ?? BLANK;
    if (cell.ch === '') continue;
    if (cell.link !== link) {
      if (link) out += '\x1b]8;;\x1b\\';
      if (cell.link) out += `\x1b]8;;${cell.link}\x1b\\`;
      link = cell.link;
    }
    if (cell.sgr !== sgr) {
      out += cell.sgr ? `\x1b[0;${cell.sgr}m` : '\x1b[0m';
      sgr = cell.sgr;
    }
    out += cell.ch;
  }
  if (link) out += '\x1b]8;;\x1b\\';
  if (sgr) out += '\x1b[0m';
  return out;
}

/**
 * Record everything written to these streams (stdout and stderr, which share
 * the terminal) until `stop()`.
 */
export function recordScreen(
  streams: NodeJS.WriteStream[],
  columns: () => number,
  height: () => number,
): { screen: ScreenRecord; stop: () => void } {
  const screen = new ScreenRecord(columns, height);
  const restore: Array<() => void> = [];
  for (const stream of streams) {
    const original = stream.write;
    const decoder = new TextDecoder();
    const patched = function (this: NodeJS.WriteStream, chunk: unknown, ...rest: unknown[]): boolean {
      try {
        if (typeof chunk === 'string') screen.feed(chunk);
        else if (chunk instanceof Uint8Array) screen.feed(decoder.decode(chunk, { stream: true }));
      } catch {
        screen.forget();
      }
      return (original as (...args: unknown[]) => boolean).call(this, chunk, ...rest);
    };
    stream.write = patched as typeof stream.write;
    restore.push(() => {
      stream.write = original;
    });
  }
  return { screen, stop: () => restore.forEach((undo) => undo()) };
}
