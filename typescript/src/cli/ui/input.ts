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
 * THE MENU NEVER SCROLLS THE TERMINAL (2026-09-25). A box at the bottom of the
 * terminal has no room under it for the menu. Growing into it scrolled the
 * terminal, which pushed conversation lines into the scrollback, and nothing
 * brings a line back down from there. The box used to scroll the screen back
 * down when the menu closed, so that it sat at the bottom again, and that left
 * blank rows in the history where those lines had been (the owner's "gap in
 * history after the command menu disappears"). So the menu opens over the last
 * rows of the conversation, above the box, when those rows are on screen and
 * the chat's record of them (`screen.ts`) is sure of them. They are drawn
 * again when the menu closes, so the box never moves and the scrollback is
 * never touched.
 *
 * THE MENU ALWAYS OPENS UPWARD WHEN IT CAN (2026-09-28). Until then it opened
 * under the box whenever it fitted there, so the room left under the box
 * decided the direction: at the bottom of the terminal the `/` menu opened
 * above while `/agent `'s four values, which fitted, opened below, and a box
 * half way down the screen opened everything below (the owner: "completion
 * menu should always open up"). Now it opens above whenever at least
 * MIN_MENU_REACH known rows are there to cover, with fewer items (and "N
 * more") when there are fewer than it wants. It opens under the box only when
 * it cannot: the rows above are not known, or there are too few of them (the
 * top of a cleared screen). The terminal may scroll then, and the box stays
 * where that leaves it until the next message. That is a box a little higher,
 * never a gap.
 * The box asks the terminal once where it starts (`queryCursorRow`); that one
 * answer also checks the record against the screen. The Python box does the
 * same (`python/webagents/cli/ui/prompt_box.py`).
 *
 * SEARCH, AND PICKERS (2026-09-30, the owner: "/resume and other commands and
 * subcommands should have search/filter on typing and up/down arrow
 * selection"). A command's values used to be matched on their names alone,
 * one word at a time, and `/resume` offered only `delete`, so a conversation
 * could not be chosen from the menu at all. Now:
 *   - A command's `complete` is told everything typed after the command and
 *     answers a `Slot`: the rows for the argument being typed, and the typed
 *     text they are matched against (`query`, a suffix of the line), which may
 *     be several words: `/resume launch plan`.
 *   - `rankRows` keeps the rows the query finds: the names it starts, then the
 *     names it is inside, then the rows where every word of it starts a word
 *     of the name, the description or the row's `search` text (a
 *     conversation's whole first message). Commands match descriptions from
 *     two characters on, so `/h` still lists what starts with h. Fixture:
 *     `python/tests/fixtures/cli/chat_commands.json` `menu_search`.
 *   - A row that completes the command (`runs`: a conversation, a snapshot, a
 *     model, a command to explain) is chosen with return: the line is sent. A
 *     row the line goes on after (a verb, one of several skills), or one whose
 *     command acts without asking (`/mcp remove`), is inserted, as before; tab
 *     always inserts. A row may insert other text than it shows (`insert`: a
 *     conversation's id for its number, so the line means the conversation
 *     highlighted whatever the list is when it runs).
 *   - `/resume` and `/rewind` (`Command.picker`): return on the command in the
 *     menu opens its list instead of printing it, when there is one to choose.
 * The Python box does the same (`prompt_box.py`, `Slot`, `rank_rows`).
 */

import * as readline from 'node:readline';

import { ESC, hardWrap, padEnd, truncate, visibleWidth } from './ansi';
import { SPARKLE_MS, sparkAt } from './motion';
import type { ScreenRecord } from './screen';
import { queryCursorRow } from './terminal';
import type { Theme } from './theme';

/** One value a command offers for the argument being typed. */
export interface SlotRow {
  value: string;
  description: string;
  /** More text a query matches, not shown (a conversation's whole first message). */
  search?: string;
  /** Choosing it completes the command: return sends the line (file comment, "SEARCH"). */
  runs?: boolean;
  /** What goes in the box instead of `value` (a conversation's id, for its number). */
  insert?: string;
}

/** What a command offers for the argument being typed: the rows, and the typed text they are matched against, a suffix of the line. */
export interface Slot {
  rows: SlotRow[];
  query: string;
}

export interface Command {
  name: string;
  description: string;
  /**
   * What the box offers after `/<name> ` (2026-09-26, interactive-mode spec
   * 3.8; a `Slot` since 2026-09-30, file comment "SEARCH"): given everything
   * typed after the command, the rows for the argument being typed and the
   * text they are matched against. With rows, the menu stays open after the
   * command; with none, or null, it closes and return sends the line.
   */
  complete?: (args: string) => Slot | null;
  /** Return on the command in the menu opens its list instead of running it (`/resume`, `/rewind`), when the list has a row to choose. */
  picker?: boolean;
}

/** One row of the menu: a command (`name` without its `/`), or an argument value. */
export interface MenuItem {
  name: string;
  description: string;
  /** True for an argument value: tab inserts it; return inserts it, or sends the line when it `runs`. */
  argument?: boolean;
  runs?: boolean;
  insert?: string;
}

/**
 * The fewest characters a query needs before it matches command descriptions
 * (file comment, "SEARCH"): below it, `/h` would match every command whose
 * description has an h-word.
 */
export const COMMAND_SEARCH_MIN = 2;

/** The complete words before the one being typed, and that word ('' after a space). */
export function argumentWords(args: string): { before: string[]; partial: string } {
  const partial = !args || /\s$/.test(args) ? '' : (args.trim().split(/\s+/).pop() ?? '');
  return { before: args.slice(0, args.length - partial.length).split(/\s+/).filter(Boolean), partial };
}

/** The text typed after the first `words` words, as typed: a search's query. */
export function restAfter(args: string, words: number): string {
  let rest = args.trimStart();
  for (let i = 0; i < words; i += 1) {
    const first = rest.split(/\s+/)[0] ?? '';
    rest = first ? rest.slice(first.length).trimStart() : '';
  }
  return rest;
}

/** The words a query word may start: split at spaces and slashes, and into letters and digits. */
function tokens(text: string): string[] {
  const lowered = text.toLowerCase();
  return [...lowered.split(/[\s/]+/).filter(Boolean), ...(lowered.match(/[a-z0-9]+/g) ?? [])];
}

/**
 * The rows `query` finds, best first (file comment, "SEARCH"): the names it
 * starts, then the names it is inside, then the rows where every word of it
 * starts a word of the name, the description or `search`. A command's name is
 * matched without its `/`, and its description only from
 * `COMMAND_SEARCH_MIN` characters. Order is kept within each group. The Python
 * twin is `prompt_box.py` `rank_rows`.
 */
export function rankRows<T extends { value: string; description: string; search?: string }>(rows: readonly T[], query: string, commands = false): T[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...rows];
  const words = q.split(/\s+/);
  const name = (row: T) => {
    const value = row.value.toLowerCase();
    return commands && value.startsWith('/') ? value.slice(1) : value;
  };
  const starts: T[] = [];
  const inside: T[] = [];
  const found: T[] = [];
  for (const row of rows) {
    const n = name(row);
    if (n.startsWith(q)) starts.push(row);
    else if (n.includes(q)) inside.push(row);
    else if (!commands || q.length >= COMMAND_SEARCH_MIN) {
      const haystack = tokens([n, row.description, row.search ?? ''].join(' '));
      if (words.every((w) => haystack.some((t) => t.startsWith(w)))) found.push(row);
    }
  }
  return [...starts, ...inside, ...found];
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
/** The most rows the menu covers above the box: a blank row, the commands, and "N more". */
const MENU_REACH = MAX_MENU_ITEMS + 2;
/** The fewest it opens above with: a blank row and two of the menu's (file comment). */
const MIN_MENU_REACH = 3;
/**
 * How long an esc waits to be the start of a key sequence rather than a key
 * of its own. Node's readline waits half a second, so an esc that closed the
 * menu turned the next key typed within that time into alt+key, and a `/`
 * typed straight after it was lost. The Python box waits the same 0.1 s
 * (`ttimeoutlen`).
 */
export const ESCAPE_TIMEOUT_MS = 100;

export class InputEditor {
  chars: string[] = [];
  menuIndex = 0;
  menuDismissed = false;
  /**
   * The text is a line ↑ or ↓ brought back from history (2026-09-29): the menu
   * stays closed, so the arrows keep walking the history past a `/command`,
   * until the line is edited or the cursor moves. Until then a recalled `/mcp`
   * reopened the menu at the first edit only; the Python box reopened it at
   * once and the next ↑ moved in the menu (the owner: "up/down history stops
   * when there is /command"). The Python twin is `prompt_box.py` `recalled`.
   */
  recalled = false;
  private cursorAt = 0;
  private navigating = false;
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

  get cursor(): number {
    return this.cursorAt;
  }

  /** A move of the cursor opens a recalled line's menu again (`recalled`); history's own moves do not. */
  set cursor(value: number) {
    if (!this.navigating && value !== this.cursorAt) this.recalled = false;
    this.cursorAt = value;
  }

  set(value: string): void {
    this.chars = Array.from(value);
    this.cursor = this.chars.length;
    this.changed();
  }

  /**
   * The rows the menu offers now; empty when it is closed. While the command
   * name is being typed, the commands it finds (`rankRows`); after
   * `/<command> `, the rows that command's `complete` offers, found by what is
   * typed for them (file comment, "SEARCH").
   */
  menu(): MenuItem[] {
    return this.menuState().items;
  }

  /** The menu's rows, and the query they answer: null for the command list, else the text typed for the argument (a suffix of the line). */
  private menuState(): { items: MenuItem[]; query: string | null } {
    const value = this.value;
    if (this.menuDismissed || this.recalled || !value.startsWith('/') || value.includes('\n')) return { items: [], query: null };
    if (!/\s/.test(value)) {
      const rows = rankRows(this.commands.map((c) => ({ value: `/${c.name}`, description: c.description })), value.slice(1), true);
      return { items: rows.map((r) => ({ name: r.value.slice(1), description: r.description })), query: null };
    }
    const command = value.slice(1).split(/\s/)[0];
    const complete = this.commands.find((c) => c.name === command)?.complete;
    const slot = complete ? complete(value.slice(1 + command.length)) : null;
    if (!slot || !slot.rows.length) return { items: [], query: null };
    const items = rankRows(slot.rows, slot.query).map((r) => ({
      name: r.value,
      description: r.description,
      argument: true as const,
      ...(r.runs ? { runs: true } : {}),
      ...(r.insert !== undefined ? { insert: r.insert } : {}),
    }));
    return { items, query: slot.query };
  }

  /** Return on `/<name>` opens its list: a picker with a row to choose. */
  private opensPicker(name: string): boolean {
    const command = this.commands.find((c) => c.name === name);
    if (!command?.picker || !command.complete) return false;
    return (command.complete(' ')?.rows ?? []).some((r) => r.runs);
  }

  /**
   * The rest of the newest history line that starts with the last line typed
   * (2026-09-29, the Python box's suggestions from history, prompt_toolkit's
   * `AutoSuggestFromHistory` rule): drawn faint after the cursor, and taken by
   * tab, →, ctrl+e or ctrl+f, alt+f for one word. Empty while the menu is
   * open, when the cursor is not at the end, or when that line is blank. The
   * history is this folder's (`chat-history.ts`), oldest first.
   */
  suggestion(): string {
    if (this.cursor !== this.chars.length || this.menu().length) return '';
    const value = this.value;
    const text = value.slice(value.lastIndexOf('\n') + 1);
    if (!text.trim()) return '';
    for (let i = this.history.length - 1; i >= 0; i -= 1) {
      const lines = this.history[i].split('\n');
      for (let j = lines.length - 1; j >= 0; j -= 1) {
        if (lines[j].startsWith(text)) return lines[j].slice(text.length);
      }
    }
    return '';
  }

  /** Take the suggestion, or its first word (prompt_toolkit's alt+f split); false when there is none. */
  private acceptSuggestion(wordOnly = false): boolean {
    const rest = this.suggestion();
    if (!rest) return false;
    this.insert(wordOnly ? rest.split(/([^\s/]+(?:\s+|\/))/).find((part) => part) ?? rest : rest);
    return true;
  }

  /** Put an offered value in place of what is typed for it, and a space after it. */
  private insertArgument(text: string): void {
    const query = this.menuState().query ?? '';
    const head = this.value.slice(0, this.value.length - query.length);
    this.set(`${head}${text} `);
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
          if (!this.acceptSuggestion()) this.cursor = this.lineEnd();
          return { kind: 'render' };
        case 'b':
          this.cursor = Math.max(0, this.cursor - 1);
          return { kind: 'render' };
        case 'f':
          if (!this.acceptSuggestion()) this.cursor = Math.min(this.chars.length, this.cursor + 1);
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
          if (!this.acceptSuggestion(true)) this.cursor = this.wordRight();
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
          // An esc and a `/` typed within the esc timeout arrive as alt+/,
          // which means nothing here, so the `/` was lost. It is what was
          // meant, and it opens the menu (2026-09-28). The Python box does the same.
          if (key.sequence === '\x1b/') {
            this.insert('/');
            return { kind: 'render' };
          }
          break;
      }
    }

    switch (name) {
      case 'return': {
        if (menu.length) {
          const item = menu[Math.min(this.menuIndex, menu.length - 1)];
          if (item.argument) {
            // A row the line goes on after is inserted, never run: the person
            // sends the line once the menu has nothing more to offer. A row
            // that completes the command sends the line with it. A row that
            // IS the text already typed sends the line as typed: a fully
            // typed `/agent edit` still offered `edit`, and return put a space
            // after it instead of sending (2026-09-27). The Python prompt
            // decides the same way (`enter_choice`).
            const query = this.menuState().query ?? '';
            const typed = query.trim().toLowerCase();
            const text = item.insert ?? item.name;
            if (typed && (typed === item.name.toLowerCase() || typed === text.toLowerCase())) return { kind: 'submit', text: this.value };
            if (item.runs) return { kind: 'submit', text: this.value.slice(0, this.value.length - query.length) + text };
            this.insertArgument(text);
            return { kind: 'render' };
          }
          if (this.opensPicker(item.name)) {
            this.set(`/${item.name} `);
            return { kind: 'render' };
          }
          return { kind: 'submit', text: `/${item.name}` };
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
        if (!this.acceptSuggestion()) this.cursor = Math.min(this.chars.length, this.cursor + 1);
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
          const item = menu[Math.min(this.menuIndex, menu.length - 1)];
          if (item.argument) this.insertArgument(item.insert ?? item.name);
          else this.set(`/${item.name} `);
        } else {
          this.acceptSuggestion();
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
    this.recalled = false;
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
    this.navigating = true;
    try {
      this.cursor = this.chars.length;
    } finally {
      this.navigating = false;
    }
    this.recalled = true;
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
  /** The box, then under it the menu while it is open, else the status line. */
  lines: string[];
  /** Where the terminal cursor goes, relative to the first line. */
  cursorRow: number;
  cursorCol: number;
  /** The box alone, and the menu's and status line's rows, for a menu drawn above the box. */
  box: string[];
  menu: string[];
  status: string;
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
  /** The most rows the menu may take: fewer when it opens above over fewer rows (file comment). */
  menuRows = MAX_MENU_ITEMS + 1,
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
  // The suggestion from history, faint after the cursor on the last row, cut
  // to the room left there (`InputEditor.suggestion`).
  const rest = editor.suggestion();
  const room = textWidth - rowWidth;
  const ghost = rest && room > 1 ? truncate(rest, room) : '';

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
      if (ghost && top + i === rows.length - 1) text += paint.fg(palette.faint, ghost);
    }
    lines.push(`${leftEdge} ${prefix}${padEnd(text, textWidth)} ${rightEdge}`);
  });
  lines.push(horizontal('╰', '╯'));

  const menu = menuLines(theme, editor, width, menuRows);
  const status = footerLine(theme, editor, width, footer, now);
  return {
    lines: [...lines, ...(menu.length ? menu : [status])],
    cursorRow: 1 + cursorRow - top,
    cursorCol: 4 + cursorCol,
    box: lines,
    menu,
    status,
  };
}

/**
 * The command menu's rows while a command is being typed; none when it is
 * closed. At most `maxRows` of them: every match when they fit, else as many
 * as fit with "N more" under them, the window following the highlighted row.
 */
function menuLines(theme: Theme, editor: InputEditor, width: number, maxRows = MAX_MENU_ITEMS + 1): string[] {
  const { paint, palette } = theme;
  const menu = editor.menu();
  if (!menu.length) return [];
  const lines: string[] = [];
  const count = menu.length <= Math.min(MAX_MENU_ITEMS, maxRows) ? menu.length : Math.max(1, Math.min(MAX_MENU_ITEMS, maxRows - 1));
  const selected = Math.min(editor.menuIndex, menu.length - 1);
  const offset = Math.max(0, Math.min(selected - count + 1, menu.length - count));
  const window = menu.slice(offset, offset + count);
  // As shown: a command with its `/`, an argument value as it is. The column
  // is two wider than the longest, for commands and values alike (argument
  // rows were a column wider than the Python box's until 2026-09-28).
  const label = (item: MenuItem) => (item.argument ? item.name : `/${item.name}`);
  const nameWidth = Math.max(...[...menu.slice(0, count), ...window].map((c) => label(c).length)) + 2;
  for (const [i, command] of window.entries()) {
    const active = offset + i === selected;
    const marker = active ? paint.fg(palette.accent, '❯') : ' ';
    const name = label(command).padEnd(nameWidth);
    const description = truncate(command.description, Math.max(10, width - nameWidth - 7));
    lines.push(
      active
        ? ` ${marker} ${paint.bold(paint.fg(palette.accent, name))}${paint.fg(palette.text, description)}`
        : ` ${marker} ${paint.fg(palette.muted, name)}${paint.fg(palette.faint, description)}`,
    );
  }
  if (menu.length > count) {
    lines.push(paint.fg(palette.faint, `   ${menu.length - count} more, keep typing to narrow`));
  }
  return lines;
}

/**
 * The rows directly above the box that the menu may cover, oldest first: as
 * many as `reach` allows and the record knows, and at least MIN_MENU_REACH;
 * null when fewer are known (file comment).
 */
function knownRowsAbove(screen: ScreenRecord, reach: number): string[] | null {
  for (let count = reach; count >= MIN_MENU_REACH; count -= 1) {
    const rows = screen.rowsAbove(count);
    if (rows) return rows;
  }
  return null;
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
  /**
   * The chat's record of the screen (`screen.ts`), which lets the menu open
   * over the conversation. Without it the menu always opens under the box.
   */
  screen?: ScreenRecord;
}

export type PromptResult = { kind: 'submit'; text: string } | { kind: 'exit' };

/**
 * Shows the box and resolves with what was sent (or `exit`). The box is erased
 * when it resolves, and a sent message is written in its place.
 */
export async function promptBox(options: PromptBoxOptions): Promise<PromptResult> {
  const input = options.input ?? process.stdin;
  const out = options.output ?? process.stdout;
  const { theme, screen } = options;
  const editor = new InputEditor(options.history, options.commands);
  // The box's own drawing is not the conversation: the record waits at the
  // box's first row until the box is gone.
  if (screen) screen.paused = true;
  // Where the box starts (file comment); null when the terminal will not say,
  // and then the menu always opens under the box.
  const startRow = await queryCursorRow(input, out);
  if (screen && startRow !== null) screen.anchor(startRow);

  return new Promise((resolve) => {
    let drawnCursorRow = -1;
    /** The 1-based screen row of the box's first line, while it is known. */
    let top: number | null = startRow;
    /** Where the box's first line was when the record last knew (the start, or the top after ctrl+l). */
    let knownTop: number | null = startRow;
    /** The open menu went under the box, and stays there until it closes. */
    let menuBelow = false;
    /**
     * The menu drawn over the conversation: the rows it may cover, as they
     * were (oldest first), and whether it is on screen yet.
     */
    let over: { saved: string[]; drawn: boolean } | null = null;
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

    /** Back to the first row drawn (`extra` rows higher still), and clear from there down. */
    const erase = (extra = 0) => {
      if (drawnCursorRow < 0) return '';
      const up = drawnCursorRow + extra;
      return `${up > 0 ? `${ESC}${up}A` : ''}\r${ESC}J`;
    };
    /** Everything drawn taken down, and what the menu covered put back: the cursor ends on the box's first row. */
    const takeDown = () => {
      const covered = over?.drawn ? over.saved.map((row) => `${row}\n`).join('') : '';
      over = null;
      return `${erase()}${covered}`;
    };

    /** The record takes over again, told how far the box's growth scrolled the terminal. */
    const resume = () => {
      if (!screen) return;
      if (knownTop !== null && top !== null) screen.scrolled(knownTop - top);
      screen.paused = false;
    };

    const draw = () => {
      scheduled = false;
      if (finished) return;
      const rows = out.rows ?? 24;
      const menuOpen = editor.menu().length > 0;
      if (!menuOpen) menuBelow = false;
      else if (!over && !menuBelow) {
        // Where the menu opens (file comment): over the conversation, above
        // the box, whenever the rows it covers are on screen and known, even
        // when it would fit under the box; else under the box, and the
        // terminal may scroll.
        const saved = screen && top !== null && drawnCursorRow >= 0 ? knownRowsAbove(screen, Math.min(MENU_REACH, top - 1)) : null;
        if (saved) over = { saved, drawn: false };
        else menuBelow = true;
      }
      // Above the box, the menu has the rows it covers less the blank one over it.
      const frame = layoutPrompt(
        theme,
        editor,
        out.columns ?? 80,
        options.footer(),
        options.placeholder,
        Date.now(),
        shownAt,
        over ? over.saved.length - 1 : undefined,
      );

      let head = '';
      let above = 0;
      let lift = 0;
      let lines = frame.lines;
      if (over) {
        if (!over.drawn) lift = over.saved.length;
        if (menuOpen) {
          // The menu at the bottom of the rows it may cover, a blank row over
          // it, and above that the conversation's own rows, drawn again.
          const cover = ['', ...frame.menu].slice(-over.saved.length);
          head = [...over.saved.slice(0, over.saved.length - cover.length), ...cover].map((row) => `${row}\n`).join('');
          above = over.saved.length;
          lines = [...frame.box, frame.status];
          over.drawn = true;
        } else {
          head = over.saved.map((row) => `${row}\n`).join('');
          over = null;
        }
      }
      const height = lines.length;
      if (top !== null) {
        // Growing past the last row scrolls the terminal by the overflow.
        const overflow = top + height - 1 - rows;
        if (overflow > 0) top -= overflow;
      }
      const up = height - 1 - frame.cursorRow;
      out.write(
        `${ESC}?2026h${erase(lift)}${head}${lines.join('\n')}${up > 0 ? `${ESC}${up}A` : ''}\r${ESC}${frame.cursorCol}C${ESC}?2026l`,
      );
      drawnCursorRow = above + frame.cursorRow;
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
          out.write(`${ESC}?2026h${takeDown()}`);
          resume();
          out.write(`${echo}${ESC}?2026l`);
          resolve({ kind: 'submit', text: action.text });
          return;
        }
        case 'exit':
          cleanup();
          out.write(`${ESC}?2026h${takeDown()}`);
          resume();
          out.write(`${ESC}?2026l`);
          resolve({ kind: 'exit' });
          return;
        case 'clear-screen':
          out.write(`${ESC}2J${ESC}H`);
          screen?.clearScreen();
          drawnCursorRow = -1;
          top = 1;
          knownTop = 1;
          over = null;
          menuBelow = false;
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

    // The first reader of a stream sets its esc timeout for good: the box, or
    // a question asked before it (`promptLine`, which passes the same).
    readline.emitKeypressEvents(input, { escapeCodeTimeout: ESCAPE_TIMEOUT_MS } as unknown as readline.Interface);
    if (input.isTTY) input.setRawMode(true);
    input.on('keypress', onKey);
    input.resume();
    out.on('resize', onResize);
    out.write(`${ESC}?2004h`);
    draw();
  });
}
