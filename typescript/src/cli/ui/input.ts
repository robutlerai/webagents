/**
 * The chat's input box (2026-09-24).
 *
 * Node's readline draws a bare prompt on one line and redraws over anything
 * below it, so a box, a placeholder, a footer and a command menu were not
 * possible with it. This is a small editor instead: `InputEditor` is the state
 * (text, cursor, history, the `/` menu) and changes only in `handleKey`, so it
 * is tested without a terminal; `layoutPrompt` turns it into lines; and
 * `promptBox` owns the terminal while the person types: raw mode, bracketed
 * paste, and one redraw per change, erased in place before the next.
 *
 * Keys, the way the terminal agents people use have settled them:
 *   enter sends; alt+enter or ctrl+j, or a line ending in `\`, adds a line;
 *   ↑/↓ walk the history (or the menu, or the lines of a multi-line message);
 *   / opens the command menu, tab completes, enter runs the highlighted one;
 *   esc closes the menu, twice clears the box; ctrl+c clears the box, twice on
 *   an empty box leaves; ctrl+d on an empty box leaves; ctrl+a/e/u/k/w, alt+b/f
 *   and ctrl+←/→ edit as in a shell.
 *
 * THE BOX GOES BACK DOWN WHEN THE MENU CLOSES (2026-09-25). A box near the
 * bottom of the terminal has no room below it for the menu, so opening the
 * menu scrolls the terminal and the box rises; closing it left the box up
 * there with empty rows under it. The box now knows where it started (one
 * cursor position report as it appears, `queryCursorRow`), counts the rows
 * its growth scrolled, and when it shrinks scrolls the screen back down by at
 * most that many (reverse index at the top row, which every terminal has), so
 * it returns to where it was. The lines the menu pushed into the scrollback
 * stay there; the rows they leave at the top of the screen are blank. The
 * Python box does the same (`python/webagents/cli/ui/prompt_box.py`).
 */

import * as readline from 'node:readline';

import { ESC, hardWrap, padEnd, truncate, visibleWidth } from './ansi';
import { SPARKLE_MS, sparkAt } from './motion';
import { queryCursorRow } from './terminal';
import type { Theme } from './theme';

export interface Command {
  name: string;
  description: string;
}

export interface Key {
  name?: string;
  ctrl?: boolean;
  meta?: boolean;
  shift?: boolean;
  sequence?: string;
}

export type EditorAction =
  | { kind: 'none' }
  | { kind: 'render' }
  | { kind: 'submit'; text: string }
  | { kind: 'exit' }
  | { kind: 'clear-screen' };

/** How long a first ctrl+c (exit) or esc (clear) waits for its second. */
export const CONFIRM_MS = 2000;
const MAX_MENU_ITEMS = 6;

export class InputEditor {
  chars: string[] = [];
  cursor = 0;
  menuIndex = 0;
  menuDismissed = false;
  pasting = false;
  exitArmedAt = 0;
  escArmedAt = 0;
  private historyIndex: number;
  private draft: string[] | null = null;

  constructor(
    private readonly history: string[],
    private readonly commands: Command[],
  ) {
    this.historyIndex = history.length;
  }

  get value(): string {
    return this.chars.join('');
  }

  set(value: string): void {
    this.chars = Array.from(value);
    this.cursor = this.chars.length;
    this.changed();
  }

  /** The commands the menu offers now; empty when it is closed. */
  menu(): Command[] {
    const value = this.value;
    if (this.menuDismissed || !value.startsWith('/') || /\s/.test(value)) return [];
    const query = value.slice(1).toLowerCase();
    const prefix = this.commands.filter((c) => c.name.toLowerCase().startsWith(query));
    const inside = this.commands.filter((c) => !prefix.includes(c) && c.name.toLowerCase().includes(query));
    return [...prefix, ...inside];
  }

  /** A first ctrl+c on an empty box, recent enough that a second one leaves. */
  exitArmed(now: number): boolean {
    return this.exitArmedAt > 0 && now - this.exitArmedAt < CONFIRM_MS;
  }

  /** A first esc on a box with text in it, recent enough that a second one clears it. */
  escArmed(now: number): boolean {
    return this.escArmedAt > 0 && now - this.escArmedAt < CONFIRM_MS;
  }

  handleKey(str: string | undefined, key: Key, now: number): EditorAction {
    const name = key.name;
    if (name === 'paste-start') {
      this.pasting = true;
      return { kind: 'none' };
    }
    if (name === 'paste-end') {
      this.pasting = false;
      return { kind: 'render' };
    }
    if (this.pasting) {
      // Pasted text is text: a newline in it is a line, not "send".
      if (name === 'return' || name === 'enter') this.insert('\n');
      else if (str) this.insert(str.replace(/\r\n?/g, '\n').replace(/\t/g, '  '));
      return { kind: 'render' };
    }

    if (key.ctrl && name === 'c') {
      if (this.chars.length) {
        this.set('');
        this.exitArmedAt = 0;
        return { kind: 'render' };
      }
      if (this.exitArmed(now)) return { kind: 'exit' };
      this.exitArmedAt = now;
      return { kind: 'render' };
    }
    this.exitArmedAt = 0;
    if (name !== 'escape') this.escArmedAt = 0;

    const menu = this.menu();
    if (key.ctrl) {
      switch (name) {
        case 'd':
          if (!this.chars.length) return { kind: 'exit' };
          this.deleteForward();
          return { kind: 'render' };
        case 'a':
          this.cursor = this.lineStart();
          return { kind: 'render' };
        case 'e':
          this.cursor = this.lineEnd();
          return { kind: 'render' };
        case 'b':
          this.cursor = Math.max(0, this.cursor - 1);
          return { kind: 'render' };
        case 'f':
          this.cursor = Math.min(this.chars.length, this.cursor + 1);
          return { kind: 'render' };
        case 'u':
          this.chars.splice(this.lineStart(), this.cursor - this.lineStart());
          this.cursor = this.lineStart();
          this.changed();
          return { kind: 'render' };
        case 'k':
          this.chars.splice(this.cursor, this.lineEnd() - this.cursor);
          this.changed();
          return { kind: 'render' };
        case 'w':
        case 'backspace':
          this.deleteWordBack();
          return { kind: 'render' };
        case 'left':
          this.cursor = this.wordLeft();
          return { kind: 'render' };
        case 'right':
          this.cursor = this.wordRight();
          return { kind: 'render' };
        case 'j':
          this.insert('\n');
          return { kind: 'render' };
        case 'l':
          return { kind: 'clear-screen' };
        default:
          return { kind: 'none' };
      }
    }

    if (key.meta) {
      switch (name) {
        case 'return':
        case 'enter':
          this.insert('\n');
          return { kind: 'render' };
        case 'b':
        case 'left':
          this.cursor = this.wordLeft();
          return { kind: 'render' };
        case 'f':
        case 'right':
          this.cursor = this.wordRight();
          return { kind: 'render' };
        case 'backspace':
          this.deleteWordBack();
          return { kind: 'render' };
        case 'd':
        case 'delete':
          this.chars.splice(this.cursor, this.wordRight() - this.cursor);
          this.changed();
          return { kind: 'render' };
        default:
          break;
      }
    }

    switch (name) {
      case 'return': {
        if (menu.length) {
          const command = menu[Math.min(this.menuIndex, menu.length - 1)];
          return { kind: 'submit', text: `/${command.name}` };
        }
        if (this.cursor === this.chars.length && this.chars[this.cursor - 1] === '\\') {
          // A line ending in a backslash continues, as in a shell.
          this.chars[this.cursor - 1] = '\n';
          this.changed();
          return { kind: 'render' };
        }
        return { kind: 'submit', text: this.value };
      }
      case 'enter':
        this.insert('\n');
        return { kind: 'render' };
      case 'backspace':
        if (this.cursor > 0) {
          this.chars.splice(this.cursor - 1, 1);
          this.cursor -= 1;
          this.changed();
        }
        return { kind: 'render' };
      case 'delete':
        this.deleteForward();
        return { kind: 'render' };
      case 'left':
        this.cursor = Math.max(0, this.cursor - 1);
        return { kind: 'render' };
      case 'right':
        this.cursor = Math.min(this.chars.length, this.cursor + 1);
        return { kind: 'render' };
      case 'home':
        this.cursor = this.lineStart();
        return { kind: 'render' };
      case 'end':
        this.cursor = this.lineEnd();
        return { kind: 'render' };
      case 'up':
        if (menu.length) {
          this.menuIndex = (this.menuIndex - 1 + menu.length) % menu.length;
        } else if (this.lineStart() > 0) {
          this.verticalMove(-1);
        } else {
          this.historyStep(-1);
        }
        return { kind: 'render' };
      case 'down':
        if (menu.length) {
          this.menuIndex = (this.menuIndex + 1) % menu.length;
        } else if (this.lineEnd() < this.chars.length) {
          this.verticalMove(1);
        } else {
          this.historyStep(1);
        }
        return { kind: 'render' };
      case 'tab':
        if (menu.length) {
          this.set(`/${menu[Math.min(this.menuIndex, menu.length - 1)].name} `);
        }
        return { kind: 'render' };
      case 'escape':
        if (menu.length) {
          this.menuDismissed = true;
          return { kind: 'render' };
        }
        if (this.chars.length) {
          if (this.escArmed(now)) {
            this.set('');
            this.escArmedAt = 0;
          } else {
            this.escArmedAt = now;
          }
        }
        return { kind: 'render' };
      default:
        break;
    }

    if (str && !key.ctrl && !key.meta && !/[\x00-\x08\x0b-\x1f\x7f]/.test(str)) {
      this.insert(str);
      return { kind: 'render' };
    }
    return { kind: 'none' };
  }

  // -- editing --------------------------------------------------------------

  private changed(): void {
    this.menuIndex = 0;
    this.menuDismissed = false;
  }

  private insert(text: string): void {
    const chars = Array.from(text);
    this.chars.splice(this.cursor, 0, ...chars);
    this.cursor += chars.length;
    this.changed();
  }

  private deleteForward(): void {
    if (this.cursor < this.chars.length) {
      this.chars.splice(this.cursor, 1);
      this.changed();
    }
  }

  private deleteWordBack(): void {
    const start = this.wordLeft();
    this.chars.splice(start, this.cursor - start);
    this.cursor = start;
    this.changed();
  }

  private lineStart(): number {
    let i = this.cursor;
    while (i > 0 && this.chars[i - 1] !== '\n') i -= 1;
    return i;
  }

  private lineEnd(): number {
    let i = this.cursor;
    while (i < this.chars.length && this.chars[i] !== '\n') i += 1;
    return i;
  }

  private wordLeft(): number {
    let i = this.cursor;
    while (i > 0 && /\s/.test(this.chars[i - 1])) i -= 1;
    while (i > 0 && !/\s/.test(this.chars[i - 1])) i -= 1;
    return i;
  }

  private wordRight(): number {
    let i = this.cursor;
    while (i < this.chars.length && /\s/.test(this.chars[i])) i += 1;
    while (i < this.chars.length && !/\s/.test(this.chars[i])) i += 1;
    return i;
  }

  private verticalMove(direction: -1 | 1): void {
    const column = this.cursor - this.lineStart();
    if (direction < 0) {
      const previousEnd = this.lineStart() - 1;
      let previousStart = previousEnd;
      while (previousStart > 0 && this.chars[previousStart - 1] !== '\n') previousStart -= 1;
      this.cursor = Math.min(previousStart + column, previousEnd);
    } else {
      const nextStart = this.lineEnd() + 1;
      let nextEnd = nextStart;
      while (nextEnd < this.chars.length && this.chars[nextEnd] !== '\n') nextEnd += 1;
      this.cursor = Math.min(nextStart + column, nextEnd);
    }
  }

  private historyStep(direction: -1 | 1): void {
    if (!this.history.length) return;
    const next = this.historyIndex + direction;
    if (next < 0 || next > this.history.length) return;
    if (this.historyIndex === this.history.length) this.draft = [...this.chars];
    this.historyIndex = next;
    this.chars = next === this.history.length ? [...(this.draft ?? [])] : Array.from(this.history[next]);
    this.cursor = this.chars.length;
    this.menuDismissed = true;
  }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export interface PromptFooter {
  /**
   * Left side, most important first: agent, model, tokens, folder. What does
   * not fit is dropped from the end, whole; a name cut in the middle says less
   * than no name.
   */
  left: string[];
}

export interface PromptFrame {
  lines: string[];
  /** Where the terminal cursor goes, relative to the first line. */
  cursorRow: number;
  cursorCol: number;
}

/** The most rows of text the box shows before it scrolls. */
const MAX_TEXT_ROWS = 10;

export function layoutPrompt(
  theme: Theme,
  editor: InputEditor,
  columns: number,
  footer: PromptFooter,
  placeholder: string,
  now: number,
  /** When the box appeared: the idle starfield shows for its first few seconds. */
  shownAt = now,
): PromptFrame {
  const { paint, palette } = theme;
  const width = Math.max(24, columns - 1);
  const inner = width - 4;
  const textWidth = inner - 2;

  // The text as rows of the box, and where the cursor falls.
  const rows: string[] = [''];
  let rowWidth = 0;
  let cursorRow = 0;
  let cursorCol = 0;
  editor.chars.forEach((ch, i) => {
    if (i === editor.cursor) {
      cursorRow = rows.length - 1;
      cursorCol = rowWidth;
    }
    if (ch === '\n') {
      rows.push('');
      rowWidth = 0;
      return;
    }
    const w = visibleWidth(ch);
    if (rowWidth + w > textWidth) {
      rows.push('');
      rowWidth = 0;
      if (i === editor.cursor) {
        cursorRow = rows.length - 1;
        cursorCol = 0;
      }
    }
    rows[rows.length - 1] += ch;
    rowWidth += w;
  });
  if (editor.cursor === editor.chars.length) {
    cursorRow = rows.length - 1;
    cursorCol = rowWidth;
  }
  const top = Math.max(0, Math.min(cursorRow - MAX_TEXT_ROWS + 1, rows.length - MAX_TEXT_ROWS));
  const visible = rows.slice(top, top + MAX_TEXT_ROWS);

  // The box in the plain border colour, the whole way round (theme.ts,
  // "Signal"); its edge used to run through the brand gradient.
  const horizontal = (left: string, right: string) => paint.fg(palette.border, `${left}${'─'.repeat(width - 2)}${right}`);
  const leftEdge = paint.fg(palette.border, '│');
  const rightEdge = paint.fg(palette.border, '│');

  const lines = [horizontal('╭', '╮')];
  visible.forEach((row, i) => {
    const first = top + i === 0;
    const prefix = first ? `${paint.bold(paint.fg(palette.accent, '❯'))} ` : '  ';
    let text: string;
    if (!editor.chars.length && first) {
      const hint = truncate(placeholder, textWidth);
      text = paint.italic(paint.fg(palette.faint, hint));
      // Codex's idle starfield: faint dots twinkling in the empty part of the
      // box for its first seconds, then gone. Truecolour or 256 only.
      if (theme.animate && paint.depth >= 256 && now - shownAt < SPARKLE_MS) {
        const start = visibleWidth(hint) + 3;
        let field = '   ';
        for (let column = start; column < textWidth; column += 1) field += sparkAt(theme, column, now);
        text += field;
      }
    } else {
      text = paint.fg(palette.text, row);
    }
    lines.push(`${leftEdge} ${prefix}${padEnd(text, textWidth)} ${rightEdge}`);
  });
  lines.push(horizontal('╰', '╯'));

  const menu = editor.menu();
  if (menu.length) {
    const shown = menu.slice(0, MAX_MENU_ITEMS);
    const selected = Math.min(editor.menuIndex, menu.length - 1);
    const offset = Math.max(0, Math.min(selected - MAX_MENU_ITEMS + 1, menu.length - MAX_MENU_ITEMS));
    const window = menu.slice(offset, offset + MAX_MENU_ITEMS);
    const nameWidth = Math.max(...shown.map((c) => c.name.length)) + 2;
    for (const [i, command] of window.entries()) {
      const active = offset + i === selected;
      const marker = active ? paint.fg(palette.accent, '❯') : ' ';
      const name = `/${command.name}`.padEnd(nameWidth + 1);
      const description = truncate(command.description, Math.max(10, width - nameWidth - 8));
      lines.push(
        active
          ? ` ${marker} ${paint.bold(paint.fg(palette.accent, name))}${paint.fg(palette.text, description)}`
          : ` ${marker} ${paint.fg(palette.muted, name)}${paint.fg(palette.faint, description)}`,
      );
    }
    if (menu.length > MAX_MENU_ITEMS) {
      lines.push(paint.fg(palette.faint, `   ${menu.length - MAX_MENU_ITEMS} more, keep typing to narrow`));
    }
  } else {
    lines.push(footerLine(theme, editor, width, footer, now));
  }

  return { lines, cursorRow: 1 + cursorRow - top, cursorCol: 4 + cursorCol };
}

function footerLine(theme: Theme, editor: InputEditor, width: number, footer: PromptFooter, now: number): string {
  const { paint, palette } = theme;
  const hint = (key: string, what: string) => `${paint.fg(palette.muted, key)} ${paint.fg(palette.faint, what)}`;
  const sep = paint.fg(palette.faint, ' · ');
  // The right side in decreasing length: all the hints, the first one, none.
  // A "press again" warning is never shortened away.
  let variants: string[];
  if (editor.exitArmed(now)) variants = [paint.fg(palette.warning, 'press ctrl+c again to exit')];
  else if (editor.escArmed(now)) variants = [paint.fg(palette.warning, 'press esc again to clear')];
  else {
    const hints = editor.chars.length
      ? [hint('enter', 'send'), hint('alt+enter', 'new line')]
      : [hint('/', 'commands'), hint('↑', 'history'), hint('ctrl+c', 'exit')];
    variants = [hints.join(sep), hints[0], ''];
  }
  // The status on the left matters more than the help on the right: take the
  // longest right side that still leaves room for the first status item.
  for (const right of variants) {
    const room = width - 2 - visibleWidth(right) - (right ? 2 : 0);
    const parts = [...footer.left];
    while (parts.length > 1 && visibleWidth(parts.join(' · ')) > room) parts.pop();
    const fits = !parts.length || visibleWidth(parts[0]) <= room;
    if (!fits && right !== variants[variants.length - 1]) continue;
    const left = parts.length && room > 3 ? paint.fg(palette.faint, truncate(parts.join(' · '), room)) : '';
    const gap = Math.max(1, width - 1 - visibleWidth(left) - visibleWidth(right));
    return ` ${left}${' '.repeat(gap)}${right}`;
  }
  return '';
}

/** The message as it stays in the conversation once sent. */
export function sentMessage(theme: Theme, columns: number, text: string): string[] {
  const { paint, palette } = theme;
  const width = Math.max(24, columns - 1);
  const rows = text.split('\n').flatMap((line) => hardWrap(line, width - 4));
  return rows.map((row, i) => {
    const prefix = i === 0 ? `${paint.bold(paint.fg(palette.accent, '❯'))} ` : '  ';
    const line = ` ${prefix}${paint.fg(palette.text, row)}`;
    // The band only where it is drawn; padding a plain line is trailing
    // spaces in whatever the person copies.
    return paint.on && paint.depth >= 256 ? paint.bg(palette.surface, padEnd(line, width)) : line;
  });
}

// ---------------------------------------------------------------------------
// The terminal
// ---------------------------------------------------------------------------

export interface PromptBoxOptions {
  theme: Theme;
  commands: Command[];
  /** Earlier messages, oldest first. The caller adds the new one. */
  history: string[];
  placeholder: string;
  footer: () => PromptFooter;
  input?: NodeJS.ReadStream;
  output?: NodeJS.WriteStream;
}

export type PromptResult = { kind: 'submit'; text: string } | { kind: 'exit' };

/**
 * Shows the box and resolves with what was sent (or `exit`). The box is erased
 * when it resolves, and a sent message is written in its place.
 */
export async function promptBox(options: PromptBoxOptions): Promise<PromptResult> {
  const input = options.input ?? process.stdin;
  const out = options.output ?? process.stdout;
  const { theme } = options;
  const editor = new InputEditor(options.history, options.commands);
  // Where the box starts (file comment); null when the terminal will not say,
  // and then the box never moves itself.
  const startRow = await queryCursorRow(input, out);

  return new Promise((resolve) => {
    let drawnCursorRow = -1;
    /** The 1-based screen row of the box's first line, while it is known. */
    let top: number | null = startRow;
    /** Rows the box's growth has scrolled the terminal, and so may take back. */
    let lifted = 0;
    let drawnHeight = 0;
    let scheduled = false;
    let finished = false;
    let hintTimer: NodeJS.Timeout | null = null;
    const shownAt = Date.now();
    // Redraws for the idle starfield, for as long as it lasts and the box is empty.
    let sparkleTimer: NodeJS.Timeout | null = theme.animate
      ? setInterval(() => {
          if (editor.chars.length || Date.now() - shownAt > SPARKLE_MS + 200) {
            if (sparkleTimer) clearInterval(sparkleTimer);
            sparkleTimer = null;
            schedule();
            return;
          }
          schedule();
        }, 150)
      : null;

    const erase = () => (drawnCursorRow < 0 ? '' : `${drawnCursorRow > 0 ? `${ESC}${drawnCursorRow}A` : ''}\r${ESC}J`);

    const draw = () => {
      scheduled = false;
      if (finished) return;
      const frame = layoutPrompt(theme, editor, out.columns ?? 80, options.footer(), options.placeholder, Date.now(), shownAt);
      const height = frame.lines.length;
      let lower = '';
      if (top !== null) {
        const rows = out.rows ?? 24;
        if (height < drawnHeight && lifted > 0) {
          // Shrinking: scroll the screen down by the rows growing took, as far
          // as there are empty rows below, then follow it down (file comment).
          const down = Math.min(lifted, Math.max(0, rows - (top + height - 1)));
          if (down > 0) {
            lower = `\x1b7${ESC}1;1H${'\x1bM'.repeat(down)}\x1b8${ESC}${down}B`;
            top += down;
            lifted -= down;
          }
        }
        // Growing past the last row scrolls the terminal by the overflow.
        const overflow = top + height - 1 - rows;
        if (overflow > 0) {
          top -= overflow;
          lifted += overflow;
        }
      }
      const up = height - 1 - frame.cursorRow;
      out.write(
        `${ESC}?2026h${erase()}${lower}${frame.lines.join('\n')}${up > 0 ? `${ESC}${up}A` : ''}\r${ESC}${frame.cursorCol}C${ESC}?2026l`,
      );
      drawnCursorRow = frame.cursorRow;
      drawnHeight = height;
    };
    const schedule = () => {
      if (scheduled) return;
      scheduled = true;
      setImmediate(draw);
    };

    const cleanup = () => {
      finished = true;
      if (hintTimer) clearTimeout(hintTimer);
      if (sparkleTimer) clearInterval(sparkleTimer);
      input.removeListener('keypress', onKey);
      out.removeListener('resize', onResize);
      out.write(`${ESC}?2004l`);
      if (input.isTTY) input.setRawMode(false);
      input.pause();
    };

    const onKey = (str: string | undefined, key: Key | undefined) => {
      const action = editor.handleKey(str, key ?? { sequence: str }, Date.now());
      switch (action.kind) {
        case 'submit': {
          cleanup();
          const echo = action.text.trim() ? `${sentMessage(theme, out.columns ?? 80, action.text).join('\n')}\n` : '';
          out.write(`${ESC}?2026h${erase()}${echo}${ESC}?2026l`);
          resolve({ kind: 'submit', text: action.text });
          return;
        }
        case 'exit':
          cleanup();
          out.write(`${ESC}?2026h${erase()}${ESC}?2026l`);
          resolve({ kind: 'exit' });
          return;
        case 'clear-screen':
          out.write(`${ESC}2J${ESC}H`);
          drawnCursorRow = -1;
          top = 1;
          lifted = 0;
          schedule();
          return;
        case 'render':
          schedule();
          // The "press again" hints fade on their own.
          if (editor.exitArmedAt || editor.escArmedAt) {
            if (hintTimer) clearTimeout(hintTimer);
            hintTimer = setTimeout(schedule, CONFIRM_MS + 50);
          }
          return;
        default:
          return;
      }
    };

    // A resize reflows the screen, so where the box sits is no longer known.
    const onResize = () => {
      top = null;
      schedule();
    };

    readline.emitKeypressEvents(input);
    if (input.isTTY) input.setRawMode(true);
    input.on('keypress', onKey);
    input.resume();
    out.on('resize', onResize);
    out.write(`${ESC}?2004h`);
    draw();
  });
}
