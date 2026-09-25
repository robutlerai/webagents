/**
 * Terminal primitives for the chat UI: what the terminal can show, colour at
 * that depth, and text measured the way a terminal lays it out (2026-09-24).
 *
 * WHY NOT A LIBRARY. The chat is part of the `webagents` package, which people
 * install as a LIBRARY; a UI framework (Ink pulls in React) or a styling stack
 * would ride along into every agent that never opens a terminal. What the chat
 * needs fits in this folder: colour at three depths, gradients, width,
 * wrapping. Nothing here writes to the terminal; the callers do.
 *
 * Colour depth, in order: NO_COLOR (none), FORCE_COLOR (0-3), COLORTERM
 * truecolor/24bit, terminals known to render 24-bit, a 256-colour TERM, else
 * the 16 basic colours. Every colour is given as #rrggbb and degrades to the
 * nearest 256-colour or basic entry, so one palette serves all three.
 */

export const ESC = '\x1b[';

export type ColorDepth = 0 | 16 | 256 | 16777216;

const TRUECOLOR_PROGRAMS = new Set(['iTerm.app', 'WezTerm', 'vscode', 'ghostty', 'Hyper', 'WarpTerminal', 'Tabby', 'rio']);

/** How many colours this terminal shows, from the environment. */
export function detectColorDepth(env: NodeJS.ProcessEnv = process.env, isTTY = true): ColorDepth {
  if (env.NO_COLOR !== undefined && env.NO_COLOR !== '') return 0;
  const force = env.FORCE_COLOR;
  if (force !== undefined) {
    if (force === '0' || force === 'false') return 0;
    if (force === '2') return 256;
    if (force === '3') return 16777216;
    return 16;
  }
  if (!isTTY) return 0;
  if (env.TERM === 'dumb') return 0;
  if (env.COLORTERM === 'truecolor' || env.COLORTERM === '24bit') return 16777216;
  if (env.TERM_PROGRAM && TRUECOLOR_PROGRAMS.has(env.TERM_PROGRAM)) return 16777216;
  if (env.WT_SESSION || env.TERM === 'xterm-kitty' || env.TERM === 'alacritty' || env.TERM === 'xterm-ghostty') {
    return 16777216;
  }
  if (env.TERM && /256col/.test(env.TERM)) return 256;
  return 16;
}

/**
 * Whether the terminal has a LIGHT background, from COLORFGBG (xterm, rxvt,
 * iTerm, Konsole set it: "fg;bg", with bg 7 or 15 meaning light). Unknown is
 * treated as dark, which is what almost every developer terminal is.
 */
export function detectLightBackground(env: NodeJS.ProcessEnv = process.env): boolean {
  const value = env.COLORFGBG;
  if (!value) return false;
  const bg = Number(value.split(';').pop());
  return bg === 7 || bg === 15;
}

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

export function hexToRgb(hex: string): Rgb {
  const h = hex.replace('#', '');
  return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
}

export function rgbToHex({ r, g, b }: Rgb): string {
  const part = (n: number) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
  return `#${part(r)}${part(g)}${part(b)}`;
}

/** The 256-colour index nearest to an RGB colour (the 6x6x6 cube or the grey ramp). */
export function rgbTo256({ r, g, b }: Rgb): number {
  if (Math.abs(r - g) < 10 && Math.abs(g - b) < 10) {
    if (r < 8) return 16;
    if (r > 248) return 231;
    return Math.round(((r - 8) / 247) * 24) + 232;
  }
  const step = (v: number) => (v < 48 ? 0 : v < 115 ? 1 : Math.floor((v - 35) / 40));
  return 16 + 36 * step(r) + 6 * step(g) + step(b);
}

/** The basic (30-37, 90-97) colour nearest to an RGB colour. */
export function rgbTo16({ r, g, b }: Rgb): number {
  const max = Math.max(r, g, b);
  if (max < 60) return 30;
  const bright = max > 180 ? 60 : 0;
  const bit = (v: number) => (v > max * 0.55 ? 1 : 0);
  const code = bit(r) + bit(g) * 2 + bit(b) * 4;
  if (code === 0) return 90;
  return 30 + code + bright;
}

/** Linear mix of two colours, `t` from 0 (a) to 1 (b). */
export function mix(a: string, b: string, t: number): string {
  const x = hexToRgb(a);
  const y = hexToRgb(b);
  const k = Math.max(0, Math.min(1, t));
  return rgbToHex({ r: x.r + (y.r - x.r) * k, g: x.g + (y.g - x.g) * k, b: x.b + (y.b - x.b) * k });
}

/** The colour at position `t` (0..1) along a gradient through `stops`. */
export function gradientAt(stops: string[], t: number): string {
  if (stops.length === 1) return stops[0];
  const k = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(k));
  return mix(stops[i], stops[i + 1], k - i);
}

/** Escape sequences at a colour depth. Every method returns plain text at depth 0. */
export class Paint {
  constructor(readonly depth: ColorDepth) {}

  get on(): boolean {
    return this.depth > 0;
  }

  /** The SGR parameters for a colour at this depth ("38;2;r;g;b", "38;5;n" or "3x"). */
  sgr(hex: string, background = false): string {
    const rgb = hexToRgb(hex);
    if (this.depth === 16777216) return `${background ? 48 : 38};2;${rgb.r};${rgb.g};${rgb.b}`;
    if (this.depth === 256) return `${background ? 48 : 38};5;${rgbTo256(rgb)}`;
    const basic = rgbTo16(rgb);
    return String(background ? basic + 10 : basic);
  }

  fg(hex: string, text: string): string {
    return this.on ? `${ESC}${this.sgr(hex, false)}m${text}${ESC}39m` : text;
  }

  bg(hex: string, text: string): string {
    // A background below 256 colours is a blunt instrument; leave it out.
    return this.on && this.depth >= 256 ? `${ESC}${this.sgr(hex, true)}m${text}${ESC}49m` : text;
  }

  bold(text: string): string {
    return this.on ? `${ESC}1m${text}${ESC}22m` : text;
  }

  dim(text: string): string {
    return this.on ? `${ESC}2m${text}${ESC}22m` : text;
  }

  italic(text: string): string {
    return this.on ? `${ESC}3m${text}${ESC}23m` : text;
  }

  underline(text: string): string {
    return this.on ? `${ESC}4m${text}${ESC}24m` : text;
  }

  strike(text: string): string {
    return this.on ? `${ESC}9m${text}${ESC}29m` : text;
  }

  /** Each character coloured along a gradient; spaces are left alone. */
  gradient(text: string, stops: string[], offset = 0, span?: number): string {
    if (!this.on) return text;
    const chars = Array.from(text);
    const width = span ?? Math.max(1, chars.length - 1);
    return chars
      .map((ch, i) => (ch === ' ' ? ch : `${ESC}${this.sgr(gradientAt(stops, (i + offset) / width))}m${ch}`))
      .join('')
      .concat(`${ESC}39m`);
  }

  /** A clickable link (OSC 8) where the terminal supports it; the text alone otherwise. */
  link(url: string, text: string): string {
    return this.on ? `\x1b]8;;${url}\x1b\\${text}\x1b]8;;\x1b\\` : text;
  }
}

// ---------------------------------------------------------------------------
// Width and wrapping
// ---------------------------------------------------------------------------

// Escapes (SGR, cursor movement) and OSC 8 link wrappers take no columns.
const ESCAPES = /\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;[^\x1b]*\x1b\\/g;

export function stripAnsi(text: string): string {
  return text.replace(ESCAPES, '');
}

function charWidth(code: number): number {
  if (code === 0 || code < 32 || (code >= 0x7f && code < 0xa0)) return 0;
  // Combining marks, zero-width joiner and variation selectors take no column.
  if ((code >= 0x300 && code <= 0x36f) || code === 0x200d || (code >= 0xfe00 && code <= 0xfe0f)) return 0;
  return (code >= 0x1100 && code <= 0x115f) ||
    (code >= 0x2e80 && code <= 0xa4cf && code !== 0x303f) ||
    (code >= 0xac00 && code <= 0xd7a3) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0xfe30 && code <= 0xfe4f) ||
    (code >= 0xff00 && code <= 0xff60) ||
    (code >= 0xffe0 && code <= 0xffe6) ||
    (code >= 0x1f300 && code <= 0x1f64f) ||
    (code >= 0x1f900 && code <= 0x1f9ff) ||
    (code >= 0x20000 && code <= 0x3fffd)
    ? 2
    : 1;
}

/** Columns a string takes on screen: escapes are free, wide characters count two. */
export function visibleWidth(text: string): number {
  let width = 0;
  for (const ch of stripAnsi(text)) width += charWidth(ch.codePointAt(0) ?? 0);
  return width;
}

/** Plain text cut to `width` columns, with an ellipsis when it was cut. */
export function truncate(text: string, width: number): string {
  if (visibleWidth(text) <= width) return text;
  let out = '';
  let used = 0;
  for (const ch of text) {
    const w = charWidth(ch.codePointAt(0) ?? 0);
    if (used + w > width - 1) break;
    out += ch;
    used += w;
  }
  return `${out}…`;
}

/** Plain text cut to `width` columns from the START, keeping the end (paths: the tail is what matters). */
export function truncateStart(text: string, width: number): string {
  if (visibleWidth(text) <= width) return text;
  const chars = Array.from(text);
  let out = '';
  let used = 0;
  for (let i = chars.length - 1; i >= 0; i -= 1) {
    const w = charWidth(chars[i].codePointAt(0) ?? 0);
    if (used + w > width - 1) break;
    out = chars[i] + out;
    used += w;
  }
  return `…${out}`;
}

/** Pads (styled) text with spaces to `width` columns. */
export function padEnd(text: string, width: number): string {
  return text + ' '.repeat(Math.max(0, width - visibleWidth(text)));
}

/**
 * Word-wraps already styled text at `width` columns. Escapes do not count,
 * and a style open across a break carries on (terminals keep it). A word
 * wider than the line is broken by characters, so nothing runs past the edge.
 */
export function wrapStyled(text: string, width: number, first = '', rest = first): string[] {
  const lines: string[] = [];
  let line = first;
  let lineWidth = visibleWidth(first);
  let empty = true;
  const restWidth = visibleWidth(rest);
  for (const word of text.split(' ')) {
    const w = visibleWidth(word);
    if (!empty && w > 0 && lineWidth + 1 + w > width) {
      lines.push(line);
      line = rest;
      lineWidth = restWidth;
      empty = true;
    }
    if (w > width - lineWidth && w > 0) {
      // Too long for any line: break it by characters, escapes kept whole.
      let piece = '';
      let pieceWidth = 0;
      for (const token of word.match(/\x1b\[[0-9;?]*[A-Za-z]|\x1b\]8;;[^\x1b]*\x1b\\|[\s\S]/gu) ?? []) {
        const tw = visibleWidth(token);
        if (tw > 0 && lineWidth + (empty ? 0 : 1) + pieceWidth + tw > width) {
          lines.push(line + (empty ? '' : ' ') + piece);
          line = rest;
          lineWidth = restWidth;
          empty = true;
          piece = '';
          pieceWidth = 0;
        }
        piece += token;
        pieceWidth += tw;
      }
      line += (empty ? '' : ' ') + piece;
      lineWidth += (empty ? 0 : 1) + pieceWidth;
      empty = false;
      continue;
    }
    line += (empty ? '' : ' ') + word;
    lineWidth += (empty ? 0 : 1) + w;
    empty = false;
  }
  lines.push(line);
  return lines;
}

/** Plain text broken into lines of at most `width` columns, by character (for input). */
export function hardWrap(text: string, width: number): string[] {
  const lines: string[] = [];
  let line = '';
  let used = 0;
  for (const ch of text) {
    const w = charWidth(ch.codePointAt(0) ?? 0);
    if (used + w > width) {
      lines.push(line);
      line = '';
      used = 0;
    }
    line += ch;
    used += w;
  }
  lines.push(line);
  return lines;
}

/** `~/…` for paths under the home directory. */
/**
 * How wide to draw: the terminal's width, or `COLUMNS` when the output is not
 * a terminal (a pipe, a test), as the Python chat's Rich console reads it,
 * else 80. Without `COLUMNS` a piped chat drew at 80 while the Python one drew
 * at whatever `COLUMNS` said.
 */
export function terminalColumns(stream: { columns?: number } = process.stdout): number {
  if (stream.columns) return stream.columns;
  const fromEnv = Number(process.env.COLUMNS);
  return Number.isFinite(fromEnv) && fromEnv > 0 ? fromEnv : 80;
}

export function shortPath(path: string, home = process.env.HOME ?? ''): string {
  return home && (path === home || path.startsWith(`${home}/`)) ? `~${path.slice(home.length)}` : path;
}

/** 1234 -> "1.2k", for token counts in a status line. */
export function compactNumber(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

/** 3.24 -> "3.2s", 75 -> "1m 15s". */
export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${Math.round(seconds - m * 60)}s`;
}
