/**
 * How one assistant turn is drawn in the TypeScript chat (2026-09-24).
 *
 * WHAT WAS WRONG. The chat wrote `Assistant: ` and then every text delta raw,
 * so markdown arrived as its source (`## `, `**`, table pipes, code fences),
 * and it DROPPED every other chunk the agent streams: a tool call and its
 * result were invisible, a model error answered with nothing at all, and the
 * screen simply paused while a tool ran. A first rework made it readable; the
 * owner's verdict on that one was "it sucks", so this is the second, built
 * from what the terminal agents people use actually do.
 *
 * THE LAYOUT. Two columns of gutter carry the markers, the content sits to
 * their right:
 *   ✦  the agent speaking (Gemini CLI's marker), once per block of text;
 *   ●  a tool call: a spinner while it runs, green or red when it returns,
 *      its result summarised under it after ⎿ (Claude Code's shape);
 *   ∴  a finished thought;
 *   ✗  an error.
 * Code blocks are held until their closing fence and drawn as a panel with
 * syntax colour (a growing preview shows while they stream), tables are held
 * until their last row and drawn with rounded borders, cells wrapped to fit
 * (Codex holds structures the same way). H1 takes the brand gradient.
 *
 * THE LIVE REGION. Finished lines go to ordinary scrollback once; below them
 * a small region is redrawn in place: the line being written, a code block in
 * progress, running tools, and the status line, an animated star and a verb
 * with light running across it ("Thinking", "Writing", "Running read_file"),
 * the elapsed time and how to stop. Every update is ONE write inside
 * synchronized output (DEC 2026), so it never flickers.
 *
 * Not a terminal (a pipe, a file): no live region and no colour, the text as
 * it arrives. NO_COLOR on a terminal keeps the layout and drops the colour.
 * No dependencies: see `ui/ansi.ts` for why.
 */

import type { StreamChunk } from '../core/types';
import { ESC, compactNumber, duration, padEnd, truncate, visibleWidth, wrapStyled } from './ui/ansi';
import { highlightLine, type HighlightState } from './ui/highlight';
import { DOT_FRAMES, DOT_INTERVAL_MS, STAR_FRAMES, STAR_INTERVAL_MS, frameAt, shimmer, starColour } from './ui/motion';
import { themeFor, type Theme } from './ui/theme';

export { visibleWidth, wrapStyled } from './ui/ansi';

// ---------------------------------------------------------------------------
// Inline markdown
// ---------------------------------------------------------------------------

/**
 * `code`, **bold**, *italic*, ~~strike~~, [links](url) and bare URLs in one
 * line; the markers go either way.
 *
 * A bare URL ends at a space, a closing bracket, an angle bracket, a quote or
 * a backtick, and not on punctuation (2026-09-25). Quotes and backticks used
 * to be part of it, so a URL inside a JSON answer (`"url":"https://…","n":1`)
 * ran to the next space and a line of JSON became one blue link. The Python
 * chat links bare URLs by this pattern, alternative for alternative
 * (`python/webagents/cli/ui/markdown.py`), both held to
 * `python/tests/fixtures/cli/bare_urls.json`.
 */
export const INLINE_PATTERN =
  /`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|\*([^*\s][^*]*)\*|(?<![\w])_([^_\s][^_]*)_(?![\w])|\[([^\]]+)\]\(([^)\s]+)\)|(https?:\/\/[^\s)<>"'`]+[^\s)<>.,;:!?"'`])/;

export function inline(theme: Theme, text: string): string {
  const { paint, palette } = theme;
  const parts: string[] = [];
  let rest = text;
  const pattern = INLINE_PATTERN;
  while (rest) {
    const m = pattern.exec(rest);
    if (!m) {
      parts.push(paint.fg(palette.text, rest));
      break;
    }
    if (m.index) parts.push(paint.fg(palette.text, rest.slice(0, m.index)));
    if (m[1] !== undefined) parts.push(paint.fg(palette.inlineCode, m[1]));
    else if (m[2] !== undefined || m[3] !== undefined) parts.push(paint.bold(paint.fg(palette.text, (m[2] ?? m[3]) as string)));
    else if (m[4] !== undefined) parts.push(paint.strike(paint.fg(palette.faint, m[4])));
    else if (m[5] !== undefined || m[6] !== undefined) parts.push(paint.italic(paint.fg(palette.text, (m[5] ?? m[6]) as string)));
    else if (m[7] !== undefined) {
      parts.push(paint.on ? paint.link(m[8], paint.underline(paint.fg(palette.info, m[7]))) : `${m[7]} (${m[8]})`);
    } else if (m[9] !== undefined) parts.push(paint.link(m[9], paint.underline(paint.fg(palette.info, m[9]))));
    rest = rest.slice(m.index + m[0].length);
  }
  return parts.join('');
}

// ---------------------------------------------------------------------------
// Block markdown
// ---------------------------------------------------------------------------

type Align = 'left' | 'right' | 'center';

function tableCells(row: string): string[] {
  const inner = row.trim().replace(/^\|/, '').replace(/\|$/, '');
  // `\|` is a pipe inside a cell, not a column break.
  return inner.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, '|'));
}

function pad(text: string, width: number, align: Align): string {
  const gap = Math.max(0, width - visibleWidth(text));
  if (align === 'right') return ' '.repeat(gap) + text;
  if (align === 'center') return ' '.repeat(Math.floor(gap / 2)) + text + ' '.repeat(Math.ceil(gap / 2));
  return text + ' '.repeat(gap);
}

/**
 * Styled text cut into rows of `width` columns by character, escapes kept
 * whole and re-applied at the start of each row (the border between rows
 * resets the colour). For code, which must not be re-flowed by words.
 */
function sliceStyled(text: string, width: number): string[] {
  const rows: string[] = [];
  let row = '';
  let used = 0;
  let active = '';
  for (const token of text.match(/\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;[^\x1b]*\x1b\\|[\s\S]/gu) ?? []) {
    if (token.startsWith('\x1b')) {
      row += token;
      if (token.endsWith('m')) active += token;
      continue;
    }
    const w = visibleWidth(token);
    if (used + w > width) {
      rows.push(row);
      row = active;
      used = 0;
    }
    row += token;
    used += w;
  }
  rows.push(row);
  return rows;
}

/**
 * Turns complete markdown lines into styled terminal lines, one line at a
 * time. Holds a code block until its closing fence and a table until its last
 * row, because neither can be laid out before then.
 */
export class MarkdownLines {
  private fence: { marker: string; lang: string; lines: string[] } | null = null;
  private table: string[] = [];

  constructor(
    private readonly theme: Theme,
    /** Columns available to the content (the gutter is not included). */
    private readonly width: () => number = () => 78,
  ) {}

  /** The finished lines one complete markdown line produces (none while a block is held). */
  render(line: string): string[] {
    const out: string[] = [];
    if (this.fence) {
      if (line.trim().startsWith(this.fence.marker) && /^\s*(```+|~~~+)\s*$/.test(line)) {
        out.push(...this.drawCode(this.fence.lang, this.fence.lines));
        this.fence = null;
      } else {
        this.fence.lines.push(line);
      }
      return out;
    }
    const tableRow = /^\s*\|.*\|\s*$/.test(line);
    if (this.table.length && !tableRow) out.push(...this.drawTable(this.table.splice(0)));
    if (tableRow) {
      this.table.push(line);
      return out;
    }
    const open = /^\s*(```+|~~~+)\s*([\w+#.-]*)/.exec(line);
    if (open) {
      this.fence = { marker: open[1], lang: open[2] ?? '', lines: [] };
      return out;
    }
    out.push(...this.block(line));
    return out;
  }

  /** Whatever is held back, drawn now: a table that ended the answer, a fence never closed. */
  flush(): string[] {
    const out: string[] = [];
    if (this.table.length) out.push(...this.drawTable(this.table.splice(0)));
    if (this.fence) {
      out.push(...this.drawCode(this.fence.lang, this.fence.lines));
      this.fence = null;
    }
    return out;
  }

  /** Whether a code block or a table is being held. */
  get holding(): boolean {
    return Boolean(this.fence) || this.table.length > 0;
  }

  /** What the held block looks like so far, for the live region: its last few lines. */
  pendingPreview(maxLines = 8): string[] {
    const { paint, palette } = this.theme;
    const fence = this.fence;
    if (fence) {
      return this.codeRows(fence.lang, fence.lines.slice(-maxLines), false);
    }
    return this.table.slice(-maxLines).map((row) => paint.fg(palette.faint, truncate(row, this.width())));
  }

  /** A line still being written, styled as far as it can be without committing to it. */
  preview(line: string): string {
    const { paint, palette } = this.theme;
    if (this.fence) {
      const rows = this.codeRows(this.fence.lang, [line], false, false);
      return rows[rows.length - 1];
    }
    if (/^\s*\|/.test(line) || /^\s*(```|~~~)/.test(line)) return paint.fg(palette.faint, line);
    return this.block(line, false).join('\n');
  }

  private block(line: string, wrap = true): string[] {
    const { paint, palette } = this.theme;
    const width = Math.max(20, this.width());
    const fit = (text: string, first = '', rest = first) => (wrap ? wrapStyled(text, width, first, rest) : [first + text]);

    const heading = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const plain = heading[2];
      // H1 in the agent's lime, H2 in the text colour (theme.ts, "Signal").
      if (level === 1) return fit(paint.bold(paint.fg(palette.agent, plain)));
      if (level === 2) return fit(paint.bold(paint.fg(palette.text, plain)));
      if (level === 3) return fit(paint.bold(inline(this.theme, plain)));
      return fit(paint.bold(paint.fg(palette.muted, plain)));
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) return [paint.fg(palette.border, '─'.repeat(Math.min(width, 72)))];
    const quote = /^\s*>\s?(.*)$/.exec(line);
    if (quote) {
      const bar = `${paint.fg(palette.agent, '▎')} `;
      return fit(paint.italic(paint.fg(palette.muted, quote[1])), bar, bar);
    }
    const task = /^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$/.exec(line);
    if (task) {
      const done = task[2] !== ' ';
      const lead = ' '.repeat(task[1].length);
      const box = done ? paint.fg(palette.success, '✔') : paint.fg(palette.muted, '☐');
      const text = done ? paint.strike(paint.fg(palette.faint, task[3])) : inline(this.theme, task[3]);
      return fit(text, `${lead}${box} `, `${lead}  `);
    }
    const bullet = /^(\s*)[-*+]\s+(.*)$/.exec(line);
    if (bullet) {
      const level = Math.floor(bullet[1].length / 2);
      const lead = '  '.repeat(level);
      const glyph = ['•', '◦', '▪'][level % 3];
      return fit(inline(this.theme, bullet[2]), `${lead}${paint.fg(palette.agent, glyph)} `, `${lead}  `);
    }
    const ordered = /^(\s*)(\d+)[.)]\s+(.*)$/.exec(line);
    if (ordered) {
      const lead = '  '.repeat(Math.floor(ordered[1].length / 2));
      const marker = `${ordered[2]}.`;
      return fit(inline(this.theme, ordered[3]), `${lead}${paint.fg(palette.agent, marker)} `, `${lead}${' '.repeat(marker.length + 1)}`);
    }
    if (!line.trim()) return [''];
    return fit(inline(this.theme, line));
  }

  /**
   * A fenced block as a shaded band, the language faint at its right, syntax
   * colour inside. A band, not a box: a `│` on every line is what a person
   * copying the code gets in their clipboard, which is why neither Gemini CLI
   * nor Codex draws code with side borders. Half-block rows above and below
   * pad the band by half a line (Gemini's trick). Below 256 colours there is
   * no band to draw, so thin rules mark where the code starts and ends.
   */
  private drawCode(lang: string, lines: string[]): string[] {
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    return this.codeRows(lang, lines, true);
  }

  /**
   * The rows of a code block; `closed` false for the one still streaming,
   * `labelled` false for the line being written under it (the label is on
   * the block's first row, once).
   */
  private codeRows(lang: string, lines: string[], closed: boolean, labelled = true): string[] {
    const { paint, palette } = this.theme;
    const width = Math.max(24, this.width());
    const inner = width - 2;
    const text = lines.map((line) => line.replace(/\t/g, '    '));
    const state: HighlightState = {};
    const rows: string[] = [];
    for (const line of text) rows.push(...sliceStyled(highlightLine(this.theme, lang, line, state), inner));
    if (!rows.length) rows.push('');

    const banded = paint.on && paint.depth >= 256;
    if (!banded) {
      const label = lang ? ` ${lang} ` : '';
      const out = [paint.fg(palette.border, `──${label}${'─'.repeat(Math.max(0, width - 2 - label.length))}`)];
      out.push(...rows.map((row) => ` ${row}`));
      if (closed) out.push(paint.fg(palette.border, '─'.repeat(width)));
      return out;
    }
    const band = palette.codeSurface;
    const reset = `${ESC}39m${ESC}23m`;
    const out = [paint.fg(band, '▄'.repeat(width))];
    rows.forEach((row, i) => {
      let body = padEnd(row, inner);
      const label = lang && labelled ? paint.fg(palette.faint, lang) : '';
      // The language on the first row, right-aligned, when the code leaves room.
      if (i === 0 && label && visibleWidth(row) + visibleWidth(label) + 2 <= inner) {
        body = `${row}${reset}${' '.repeat(inner - visibleWidth(row) - visibleWidth(label))}${label}`;
      }
      out.push(paint.bg(band, ` ${body}${reset} `));
    });
    if (closed) out.push(paint.fg(band, '▀'.repeat(width)));
    return out;
  }

  /**
   * A table with rounded borders and a bold header. Columns keep their natural
   * width when it fits; otherwise the widest give way and their cells wrap.
   */
  private drawTable(rows: string[]): string[] {
    const { paint, palette } = this.theme;
    let cells = rows.map(tableCells);
    const separator = (row: string[]) => row.length > 0 && row.every((cell) => /^:?-+:?$/.test(cell));
    let header: string[] | null = null;
    let aligns: Align[] = [];
    if (cells.length >= 2 && separator(cells[1])) {
      header = cells[0];
      aligns = cells[1].map((cell) =>
        cell.startsWith(':') && cell.endsWith(':') ? 'center' : cell.endsWith(':') ? 'right' : 'left',
      );
      cells = cells.slice(2);
    }
    const styledHeader = header?.map((cell) => paint.bold(paint.fg(palette.agent, cell))) ?? null;
    const body = cells.map((row) => row.map((cell) => inline(this.theme, cell)));
    const all = styledHeader ? [styledHeader, ...body] : body;
    const count = Math.max(...all.map((row) => row.length));
    const natural = Array.from({ length: count }, (_, i) => Math.max(1, ...all.map((row) => visibleWidth(row[i] ?? ''))));
    const chrome = 3 * count + 1;
    const room = this.width() - chrome;
    if (room < count * 3) return rows.map((row) => paint.fg(palette.faint, row));
    // Water-fill: the widest columns shrink to a common cap until the table fits.
    let widths = natural;
    if (natural.reduce((a, b) => a + b, 0) > room) {
      let cap = Math.max(...natural);
      while (cap > 3 && natural.reduce((sum, w) => sum + Math.min(w, cap), 0) > room) cap -= 1;
      widths = natural.map((w) => Math.min(w, cap));
    }
    const border = (s: string) => paint.fg(palette.border, s);
    const rule = (left: string, mid: string, right: string) => border(left + widths.map((w) => '─'.repeat(w + 2)).join(mid) + right);
    const draw = (row: string[]) => {
      const wrapped = widths.map((w, i) => wrapStyled(row[i] ?? '', w));
      const height = Math.max(...wrapped.map((lines) => lines.length));
      const out: string[] = [];
      for (let r = 0; r < height; r += 1) {
        out.push(
          border('│') +
            widths.map((w, i) => ` ${pad(wrapped[i][r] ?? '', w, aligns[i] ?? 'left')} `).join(border('│')) +
            border('│'),
        );
      }
      return out;
    };
    const out = [rule('╭', '┬', '╮')];
    if (styledHeader) {
      out.push(...draw(styledHeader));
      out.push(rule('├', '┼', '┤'));
    }
    for (const row of body) out.push(...draw(row));
    out.push(rule('╰', '┴', '╯'));
    return out;
  }
}

// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

const KEY_ARGS = ['path', 'file_path', 'filename', 'dir_path', 'directory', 'command', 'cmd', 'query', 'pattern', 'url', 'name'];

/** The argument that says what a call was about: a path, a command, a query. */
export function toolKeyArgument(argumentsJson: string): string {
  let args: unknown;
  try {
    args = argumentsJson ? JSON.parse(argumentsJson) : {};
  } catch {
    return argumentsJson.slice(0, 60);
  }
  if (!args || typeof args !== 'object') return String(args ?? '').slice(0, 60);
  const record = args as Record<string, unknown>;
  const clip = (v: string) => (v.length <= 60 ? v : `${v.slice(0, 59)}…`);
  for (const key of KEY_ARGS) {
    const value = record[key];
    if (typeof value === 'string' && value) return clip(value);
  }
  for (const value of Object.values(record)) if (typeof value === 'string' && value) return clip(value);
  return '';
}

const FAILURE_TEXT = /^(error\b|access denied|permission denied|file not found|directory not found|no such file|command not found|failed\b|traceback)/i;

/** Whether a call failed, by its flag or by the message it returned. */
export function toolFailed(isError: boolean | undefined, result: string): boolean {
  if (isError) return true;
  const first = (result ?? '').trim().split('\n')[0] ?? '';
  return FAILURE_TEXT.test(first);
}

/** One line that says what a call produced. */
export function toolResultSummary(name: string, result: string, failed: boolean): string {
  const text = (result ?? '').trim();
  if (failed) {
    const first = text.split('\n')[0] || 'failed';
    return first.length <= 100 ? first : `${first.slice(0, 99)}…`;
  }
  if (!text) return '(no output)';
  const lines = text.split('\n').filter((l) => l.trim());
  if (/^directory listing/i.test(lines[0] ?? '') || (name === 'list_directory' && lines.length)) {
    const entries = /^directory listing/i.test(lines[0] ?? '') ? lines.slice(1) : lines;
    if (!entries.length) return 'empty directory';
    const names = entries.map((e) => {
      const isDir = /^\s*\[(DIR|dir)\]|\/$/.test(e);
      const n = e.replace(/^\s*\[(DIR|FILE|dir|file)\]\s*/, '').replace(/^\s*[-*]\s*/, '').trim();
      return isDir && !n.endsWith('/') ? `${n}/` : n;
    });
    return `${names.length} ${names.length === 1 ? 'entry' : 'entries'}: ${names.slice(0, 5).join(', ')}${names.length > 5 ? '…' : ''}`;
  }
  if (name === 'read_file' || name === 'read') return `Read ${text.split('\n').length} lines`;
  const first = lines[0] ?? text;
  const clipped = first.length <= 80 ? first : `${first.slice(0, 79)}…`;
  return lines.length > 1 ? `${clipped}  (+${lines.length - 1} lines)` : clipped;
}

interface ToolState {
  id: string;
  name: string;
  args: string;
  done: boolean;
  failed: boolean;
  result: string;
  /** The latest line a running tool reported (delegation streams its reply). */
  progress: string;
  startedAt: number;
  endedAt: number;
  /** Why a call that never returned was drawn as finished. */
  unfinished?: 'interrupted' | 'no result';
}

// ---------------------------------------------------------------------------
// The turn
// ---------------------------------------------------------------------------

export interface TurnPrinterOptions {
  /** Where output goes. Defaults to process.stdout. */
  out?: NodeJS.WriteStream;
  /** Show each tool's output under its summary. */
  toolDetails?: boolean;
  /**
   * Lay the answer out and keep a live status at the bottom (default: when
   * `out` is a terminal). Off, the text is written as it arrives, markdown
   * and all, which is what a pipe wants.
   */
  live?: boolean;
  /** Colour (default: live, unless NO_COLOR is set). */
  color?: boolean;
  /** The theme, when the caller has one (the chat does, for the input box). */
  theme?: Theme;
  /** A line of advice under an error (the chat knows which key the provider reads). */
  errorHint?: (message: string) => string | undefined;
  /**
   * The error's headline and advice together (`failures.ts`); the chat passes
   * this, and `errorHint` is used only without it.
   */
  explainError?: (message: string) => { headline: string; hint?: string };
}

/** How many running tools the live region lists before it summarises the rest. */
const MAX_LIVE_TOOLS = 6;
/** The live region is redrawn at most this often while text streams (aider: 20 fps). */
const REDRAW_MS = 40;
/** The animation clock: the star, the shimmer and the tool spinners. */
const TICK_MS = 80;
/** Two columns of gutter: the marker, then a space. */
const GUTTER = 2;

export class TurnPrinter {
  private readonly out: NodeJS.WriteStream;
  private readonly live: boolean;
  private readonly theme: Theme;
  private readonly md: MarkdownLines;
  private readonly toolDetails: boolean;
  private readonly errorHint?: (message: string) => string | undefined;
  private readonly explainError?: (message: string) => { headline: string; hint?: string };
  private partial = '';
  private readonly tools: ToolState[] = [];
  private thinkingSince: number | null = null;
  private lastText = 0;
  private liveRows = 0;
  private lastDraw = 0;
  private started = Date.now();
  private timer: NodeJS.Timeout | null = null;
  /** Something was written this turn, so a blank line before the next block reads as a break. */
  private wroteSomething = false;
  /** A blank line owed, written only if more follows: an answer never ends in blank lines. */
  private pendingBlank = false;
  /** The next text line starts a block, and carries the ✦. */
  private blockStart = true;
  private text = '';
  private errors = 0;
  private tokens: number | null = null;
  private inputTokens = 0;
  private outputTokens = 0;
  /** What one update writes: erase, finished lines, the live region, sent as ONE write. */
  private pending = '';
  /** A live region is drawn (start() was called), so updates move the cursor. */
  private animated = false;

  constructor(options: TurnPrinterOptions = {}) {
    this.out = options.out ?? process.stdout;
    this.live = options.live ?? Boolean(this.out.isTTY);
    this.theme =
      options.theme ??
      themeFor(this.out, process.env, {
        interactive: this.live,
        ...(options.color === false ? { depth: 0 as const } : options.color === true ? { depth: 16777216 as const } : {}),
      });
    this.md = new MarkdownLines(this.theme, () => this.columns() - GUTTER - 1);
    this.toolDetails = options.toolDetails ?? false;
    this.errorHint = options.errorHint;
    this.explainError = options.explainError;
  }

  /** Start the status line (and its animation) for this turn. */
  start(): void {
    this.started = Date.now();
    if (!this.live) return;
    this.animated = true;
    this.timer = setInterval(() => {
      this.redrawLive();
      this.flushOut();
    }, TICK_MS);
    this.timer.unref?.();
    this.redrawLive();
    this.flushOut();
  }

  /** Everything the answer's text said, for the conversation history. */
  get plainText(): string {
    return this.text;
  }

  /** Whether an error chunk arrived this turn. */
  get failed(): boolean {
    return this.errors > 0;
  }

  /** Tokens the turn reported, when the provider said. */
  get usageTokens(): number | null {
    return this.tokens;
  }

  /** The same, split into what was sent and what came back, for the saved conversation. */
  get usageSplit(): { input: number; output: number } {
    return { input: this.inputTokens, output: this.outputTokens };
  }

  feed(chunk: StreamChunk): void {
    switch (chunk.type) {
      case 'delta':
        if (chunk.delta) this.onText(chunk.delta);
        break;
      case 'thinking':
        if (this.thinkingSince === null) this.thinkingSince = Date.now();
        this.redrawLive();
        break;
      case 'tool_call':
        if (chunk.tool_call) this.onToolCall(chunk.tool_call.id, chunk.tool_call.name, chunk.tool_call.arguments);
        break;
      case 'tool_result':
        if (chunk.tool_result) {
          this.onToolResult(chunk.tool_result.call_id, chunk.tool_result.result, chunk.tool_result.is_error);
        }
        break;
      case 'tool_progress':
        if (chunk.tool_progress) this.onToolProgress(chunk.tool_progress.call_id, chunk.tool_progress.text);
        break;
      case 'error':
        this.onError(chunk.error?.message ?? 'The model returned an error.');
        break;
      case 'done': {
        const usage = chunk.response?.usage;
        if (usage) {
          const total = usage.total_tokens || (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0);
          if (total) this.tokens = (this.tokens ?? 0) + total;
          this.inputTokens += usage.input_tokens ?? 0;
          this.outputTokens += usage.output_tokens ?? 0;
        }
        break;
      }
      default:
        break;
    }
    this.flushOut();
  }

  /** Print what is left, take the live region down, and close the turn with its stats. */
  finish(options: { interrupted?: boolean } = {}): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.endThinking();
    this.commitPartial();
    for (const tool of this.tools.filter((t) => !t.done)) {
      tool.done = true;
      tool.endedAt = Date.now();
      tool.unfinished = options.interrupted ? 'interrupted' : 'no result';
      if (this.live) this.commitBlock(this.toolLines(tool));
      else this.emit(`  -> ${tool.unfinished}\n`);
    }
    this.clearLive();
    const { paint, palette } = this.theme;
    if (!this.live && this.text && !this.text.endsWith('\n')) this.emit('\n');
    if (options.interrupted) this.emit(`${paint.fg(palette.faint, '  ⎿  ')}${paint.fg(palette.warning, 'Interrupted')}\n`);
    // The turn's stats, when it produced something; under a bare error they
    // would only say how quickly it failed.
    if (this.animated && (this.text || this.tools.length)) {
      const seconds = (Date.now() - this.started) / 1000;
      const stats = [`Worked for ${duration(seconds)}`, ...(this.tokens ? [`${this.tokens.toLocaleString('en-US')} tokens`] : [])];
      this.emit(`\n${paint.fg(palette.faint, `✻ ${stats.join(' · ')}`)}\n`);
    }
    this.flushOut();
  }

  // -- output ---------------------------------------------------------------

  private emit(text: string): void {
    this.pending += text;
  }

  /**
   * One write per update. Erasing the live region, printing what finished and
   * drawing the region again used to be three writes, and a terminal that
   * painted between them showed the status line blinking out. Wrapped in
   * synchronized output (DEC mode 2026), so a terminal that supports it
   * paints the update whole; the others ignore the two sequences.
   */
  private flushOut(): void {
    if (!this.pending) return;
    const chunk = this.pending;
    this.pending = '';
    this.out.write(this.animated ? `${ESC}?2026h${chunk}${ESC}?2026l` : chunk);
  }

  // -- text -----------------------------------------------------------------

  private onText(delta: string): void {
    this.endThinking();
    this.text += delta;
    this.lastText = Date.now();
    if (!this.live) {
      this.emit(delta);
      return;
    }
    let rest = delta;
    let newline = rest.indexOf('\n');
    while (newline !== -1) {
      const line = this.partial + rest.slice(0, newline);
      this.partial = '';
      this.commitText(this.md.render(line));
      rest = rest.slice(newline + 1);
      newline = rest.indexOf('\n');
    }
    this.partial += rest;
    if (Date.now() - this.lastDraw >= REDRAW_MS) this.redrawLive();
  }

  /** The line being written, and a block being held, go out before a tool or the end. */
  private commitPartial(): void {
    if (this.partial) {
      const line = this.partial;
      this.partial = '';
      this.commitText(this.md.render(line));
    }
    this.commitText(this.md.flush());
  }

  private endThinking(): void {
    if (this.thinkingSince === null) return;
    const seconds = (Date.now() - this.thinkingSince) / 1000;
    this.thinkingSince = null;
    const { paint, palette } = this.theme;
    if (this.live) this.commitBlock([paint.italic(paint.fg(palette.faint, `∴ Thought for ${duration(seconds)}`))]);
  }

  // -- tools ----------------------------------------------------------------

  private onToolCall(id: string, name: string, args: string): void {
    this.endThinking();
    // The same call can be announced more than once (as the model streams it,
    // then again as it runs); an id seen before updates that call and never
    // draws a second one.
    const existing = id ? this.tools.find((t) => t.id === id) : undefined;
    if (existing) {
      if (!existing.done) {
        existing.name = name || existing.name;
        existing.args = args || existing.args;
        this.redrawLive();
      }
      return;
    }
    this.commitPartial();
    const now = Date.now();
    this.tools.push({ id, name, args, done: false, failed: false, result: '', progress: '', startedAt: now, endedAt: 0 });
    if (!this.live) {
      this.emit(`${this.text && !this.text.endsWith('\n') ? '\n' : ''}[${name}] ${toolKeyArgument(args)}\n`);
      return;
    }
    this.redrawLive();
  }

  private onToolProgress(id: string, text: string): void {
    const tool = this.tools.find((t) => t.id === id && !t.done);
    if (!tool || !text) return;
    const last = text.trimEnd().split('\n').pop() ?? '';
    tool.progress = last.length <= 100 ? last : `${last.slice(0, 99)}…`;
    this.redrawLive();
  }

  private onToolResult(id: string, result: string, isError?: boolean): void {
    const tool = this.tools.find((t) => t.id === id && !t.done) ?? this.tools.find((t) => !t.done);
    if (!tool) return;
    tool.done = true;
    tool.endedAt = Date.now();
    tool.result = typeof result === 'string' ? result : JSON.stringify(result);
    tool.failed = toolFailed(isError, tool.result);
    if (!this.live) {
      this.emit(`  -> ${toolResultSummary(tool.name, tool.result, tool.failed)}\n`);
      return;
    }
    this.commitBlock(this.toolLines(tool));
  }

  private onError(message: string): void {
    this.errors += 1;
    this.endThinking();
    this.commitPartial();
    const explained = this.explainError?.(message) ?? {
      headline: message.split('\n')[0] ?? message,
      hint: this.errorHint?.(message),
    };
    const first = explained.headline;
    if (!this.live) {
      this.emit(`\nError: ${first}\n`);
      return;
    }
    const { paint, palette } = this.theme;
    const marker = `${paint.bold(paint.fg(palette.error, '✗'))} `;
    const lines = wrapStyled(paint.fg(palette.error, first), this.columns() - 1, marker, '  ');
    const hint = explained.hint;
    if (hint) lines.push(...wrapStyled(paint.fg(palette.faint, hint), this.columns() - 1, paint.fg(palette.faint, '  ⎿  '), '     '));
    this.commitBlock(lines);
  }

  private toolLines(tool: ToolState): string[] {
    const { paint, palette } = this.theme;
    const bad = tool.failed || tool.unfinished === 'interrupted';
    const dot = !tool.done
      ? paint.fg(palette.accent, frameAt(DOT_FRAMES, DOT_INTERVAL_MS, Date.now()))
      : paint.fg(bad ? palette.error : palette.success, '●');
    const argument = toolKeyArgument(tool.args);
    const room = this.columns() - GUTTER - visibleWidth(tool.name) - 4;
    const call = `${paint.bold(paint.fg(palette.text, tool.name || 'tool'))}${
      argument ? paint.fg(palette.muted, `(${truncate(argument, Math.max(8, room))})`) : ''
    }`;
    const lines = [`${dot} ${call}`];
    const elbow = paint.fg(palette.faint, '  ⎿  ');
    if (!tool.done) {
      if (tool.progress) lines.push(`${elbow}${paint.fg(palette.faint, truncate(tool.progress, this.columns() - 7))}`);
      return lines;
    }
    const seconds = (tool.endedAt - tool.startedAt) / 1000;
    const timing = seconds >= 0.5 ? paint.fg(palette.faint, `  ${duration(seconds)}`) : '';
    const summary = tool.unfinished ?? toolResultSummary(tool.name, tool.result, tool.failed);
    const width = this.columns() - 7 - visibleWidth(timing);
    lines.push(`${elbow}${paint.fg(bad ? palette.error : palette.muted, truncate(summary, Math.max(10, width)))}${timing}`);
    if (this.toolDetails && !tool.unfinished && tool.result.trim()) {
      const body = tool.result.trimEnd().split('\n');
      for (const line of body.slice(0, 12)) lines.push(paint.fg(palette.faint, `     ${truncate(line, this.columns() - 6)}`));
      if (body.length > 12) lines.push(paint.fg(palette.faint, `     … +${body.length - 12} lines`));
    }
    return lines;
  }

  // -- the live region ------------------------------------------------------

  private columns(): number {
    return Math.max(30, this.out.columns ?? 80);
  }

  private rowsOf(line: string): number {
    return Math.max(1, Math.ceil(visibleWidth(line) / this.columns()));
  }

  private clearLive(): void {
    if (!this.live || this.liveRows === 0) return;
    this.emit(`${this.liveRows > 1 ? `${ESC}${this.liveRows - 1}A` : ''}\r${ESC}J`);
    this.liveRows = 0;
  }

  /** What the status verb says the agent is doing now. */
  private verb(now: number): string {
    const running = this.tools.filter((t) => !t.done);
    if (running.length) return `Running ${running[0].name || 'a tool'}${running.length > 1 ? ` +${running.length - 1}` : ''}`;
    if (this.thinkingSince !== null) return 'Thinking';
    if (this.partial || this.md.holding || now - this.lastText < 1200) return 'Writing';
    return 'Thinking';
  }

  /**
   * The bottom of the screen: the line being written, a block in progress,
   * running tools, the status. Kept shorter than the screen, because the
   * cursor cannot move above the top of it to erase: a paragraph with no line
   * break yet shows its tail.
   */
  private liveLines(): string[] {
    const { paint, palette } = this.theme;
    const now = Date.now();
    const room = Math.max(4, (this.out.rows ?? 24) - 6);
    const body: string[] = [];
    // The blank line the next finished block will get, so nothing jumps when it lands.
    if ((this.md.holding || this.partial) && this.pendingBlank) body.push('');
    if (this.md.holding) body.push(...this.md.pendingPreview(Math.max(1, Math.min(8, room - 3))).map((line) => this.gutter(line, false)));
    if (this.partial) {
      const budget = (this.columns() - GUTTER) * Math.max(1, room - body.length - 2) - 2;
      let tail = this.partial;
      if (visibleWidth(tail) > budget) {
        let width = 0;
        let i = tail.length;
        while (i > 0 && width < budget) {
          i -= 1;
          width += visibleWidth(tail[i]);
        }
        tail = `…${tail.slice(i + 1)}`;
      }
      body.push(this.gutter(this.md.preview(tail), this.blockStart && !this.md.holding));
    }
    const lines: string[] = [];
    if (body.length) lines.push(...body, '');
    const running = this.tools.filter((t) => !t.done);
    for (const tool of running.slice(0, MAX_LIVE_TOOLS)) lines.push(...this.toolLines(tool));
    if (running.length > MAX_LIVE_TOOLS) lines.push(paint.fg(palette.faint, `  … ${running.length - MAX_LIVE_TOOLS} more running`));
    if (running.length) lines.push('');
    // Claude Code's star, in the brand gradient; the verb in the agent's colour
    // with Codex's band of light; then time, a running estimate of what has
    // streamed back (Claude Code shows the same arrow), and how to stop.
    const star = paint.bold(paint.fg(starColour(this.theme, now), frameAt(STAR_FRAMES, STAR_INTERVAL_MS, now)));
    const verb = shimmer(this.theme, `${this.verb(now)}…`, palette.agent, now);
    const seconds = Math.floor((now - this.started) / 1000);
    const elapsed = seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`;
    const streamed = this.text.length ? ` · ↓ ${compactNumber(Math.max(1, Math.round(this.text.length / 4)))} tokens` : '';
    const meta = `${paint.fg(palette.faint, `(${elapsed}${streamed} · `)}${paint.fg(palette.muted, 'esc')}${paint.fg(palette.faint, ' to interrupt)')}`;
    lines.push(`${star} ${verb} ${meta}`);
    return lines;
  }

  private redrawLive(): void {
    if (!this.live || this.timer === null) return;
    this.clearLive();
    const lines = this.liveLines();
    this.emit(lines.join('\n'));
    this.liveRows = lines.reduce((sum, line) => sum + this.rowsOf(line), 0);
    this.lastDraw = Date.now();
  }

  /** A content line with its gutter: the ✦ when it starts a block of text. */
  private gutter(line: string, marker: boolean): string {
    const { paint, palette } = this.theme;
    return `${marker ? paint.fg(palette.agent, '✦') : ' '} ${line}`;
  }

  /** Finished lines of the answer, above the live region. */
  private commitText(lines: string[]): void {
    const out: string[] = [];
    for (const line of lines) {
      if (!line.trim()) {
        if (this.wroteSomething || out.length) this.pendingBlank = true;
        continue;
      }
      if (this.pendingBlank) out.push('');
      this.pendingBlank = false;
      out.push(this.gutter(line, this.blockStart));
      this.blockStart = false;
    }
    if (!out.length) return;
    this.clearLive();
    this.emit(`${out.join('\n')}\n`);
    this.wroteSomething = true;
    this.redrawLive();
  }

  /** A finished block (a tool call, a thought, an error), set apart by blank lines. */
  private commitBlock(lines: string[]): void {
    this.clearLive();
    this.emit(`${this.wroteSomething ? '\n' : ''}${lines.join('\n')}\n`);
    this.wroteSomething = true;
    this.pendingBlank = true;
    this.blockStart = true;
    this.redrawLive();
  }
}

// ---------------------------------------------------------------------------
// -p --output-format stream-json
// ---------------------------------------------------------------------------

/**
 * One line of `--output-format stream-json`, or null for a chunk it does not
 * carry. Text deltas, tool calls and their results, thinking, then `done`; an
 * `error` line means the turn failed. It used to carry text deltas alone, so
 * a script saw neither the tools run nor the failure: a rejected key ended in
 * a `done` with an empty answer (2026-09-24).
 */
export function streamJsonEvent(chunk: StreamChunk): Record<string, unknown> | null {
  switch (chunk.type) {
    case 'delta':
      return chunk.delta ? { type: 'delta', delta: chunk.delta } : null;
    case 'tool_call':
      return chunk.tool_call
        ? { type: 'tool_call', tool_call: { id: chunk.tool_call.id, name: chunk.tool_call.name, arguments: chunk.tool_call.arguments } }
        : null;
    case 'tool_result':
      return chunk.tool_result
        ? {
            type: 'tool_result',
            tool_result: {
              call_id: chunk.tool_result.call_id,
              result: chunk.tool_result.result,
              is_error: Boolean(chunk.tool_result.is_error),
            },
          }
        : null;
    case 'thinking':
      return chunk.thinking ? { type: 'thinking', thinking: chunk.thinking } : null;
    case 'done':
      return { type: 'done', response: chunk.response ?? null };
    case 'error': {
      const code = (chunk.error as (Error & { code?: string }) | undefined)?.code;
      return { type: 'error', error: { message: chunk.error?.message ?? 'The model returned an error.', ...(code ? { code } : {}) } };
    }
    default:
      return null;
  }
}
