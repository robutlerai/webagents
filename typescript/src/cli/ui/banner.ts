/**
 * The chat's opening screen: the WEBAGENTS wordmark and a card saying which
 * agent this is, which model it runs and what it can do (2026-09-24).
 *
 * The wordmark is the Python CLI's (`python/webagents/cli/ui/splash.py`), so
 * the two SDKs open the same way. Its letters are the palette's
 * `wordmarkFace` and its shadow strokes `wordmarkShadow` (theme.ts, "Signal");
 * on a terminal that animates, a band of the agent's lime crosses it once as
 * it appears (about half a second, never again in the session).
 *
 * ONE COLUMN IN, 79 WIDE (2026-09-25). It was indented two columns, so at 80
 * columns the art ran into the last column and the full wordmark was kept for
 * 82 and up; a standard 80-column terminal got the three-line one. One column
 * of margin each side centres the 78-column art at 80 columns and never
 * writes the last column. Narrower terminals get the three-line wordmark,
 * drawn all in the letters' colour (it has no shadow strokes).
 */

import { ESC, mix, padEnd, truncate, truncateStart, visibleWidth, wrapStyled } from './ansi';
import type { Theme } from './theme';

const WORDMARK = [
  '██╗    ██╗███████╗██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗',
  '██║    ██║██╔════╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝',
  '██║ █╗ ██║█████╗  ██████╔╝███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗',
  '██║███╗██║██╔══╝  ██╔══██╗██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║',
  '╚███╔███╔╝███████╗██████╔╝██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║',
  ' ╚══╝╚══╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝',
];

const WORDMARK_SMALL = ['╦ ╦╔═╗╔╗ ╔═╗╔═╗╔═╗╔╗╔╔╦╗╔═╗', '║║║║╣ ╠╩╗╠═╣║ ╦║╣ ║║║ ║ ╚═╗', '╚╩╝╚═╝╚═╝╩ ╩╚═╝╚═╝╝╚╝ ╩ ╚═╝'];

/** The columns the full wordmark needs: one of margin, 78 of art, and the last column left empty. */
export const FULL_WORDMARK_COLUMNS = 80;
const INDENT = ' ';

function artFor(columns: number): string[] {
  return columns >= FULL_WORDMARK_COLUMNS ? WORDMARK : WORDMARK_SMALL;
}

/** The wordmark lines for this width, with the band of light at `sweep` (-1 = none). */
export function wordmark(theme: Theme, columns: number, sweep = -1): string[] {
  const art = artFor(columns);
  const { paint, palette } = theme;
  if (!paint.on) return art.map((line) => `${INDENT}${line}`);
  // The full art's letters are its `█` cells and the rest its shadow; the
  // three-line art is all letters (file comment).
  const allFace = art === WORDMARK_SMALL;
  return art.map((line, row) => {
    let out = INDENT;
    Array.from(line).forEach((ch, col) => {
      if (ch === ' ') {
        out += ' ';
        return;
      }
      const position = col + row * 2;
      let colour = allFace || ch === '█' ? palette.wordmarkFace : palette.wordmarkShadow;
      if (sweep >= 0) {
        const distance = Math.abs(position - sweep);
        if (distance < 6) colour = mix(colour, palette.agent, ((Math.cos((distance / 6) * Math.PI) + 1) / 2) * 0.85);
      }
      out += `${ESC}${paint.sgr(colour)}m${ch}`;
    });
    return `${out}${ESC}39m`;
  });
}

/** Draws the wordmark, with the light crossing it once when the terminal animates. */
export async function playWordmark(out: NodeJS.WriteStream, theme: Theme): Promise<void> {
  const columns = out.columns ?? 80;
  const lines = wordmark(theme, columns);
  out.write(`\n${lines.join('\n')}\n`);
  if (!theme.animate || !theme.paint.on) return;
  const art = artFor(columns);
  const span = Math.max(...art.map((line) => line.length)) + art.length * 2;
  const frames = 18;
  out.write(`${ESC}?25l`);
  try {
    for (let frame = 0; frame <= frames; frame += 1) {
      const sweep = frame === frames ? -1 : -6 + (frame / (frames - 1)) * (span + 12);
      out.write(`${ESC}?2026h${ESC}${lines.length}A\r${wordmark(theme, columns, sweep).join('\n')}\n${ESC}?2026l`);
      await new Promise((resolve) => setTimeout(resolve, 28));
    }
  } finally {
    out.write(`${ESC}?25h`);
  }
}

export interface WelcomeInfo {
  agent: string;
  description?: string;
  model?: string;
  tools: string[];
  folder: string;
  /** Shown in the card, in amber: a missing key, a skill that failed to load. */
  warnings: string[];
  /** The SDK's version, at the right of the card's top edge. */
  version?: string;
}

/** The card under the wordmark. */
export function welcomeCard(theme: Theme, columns: number, info: WelcomeInfo): string[] {
  const { paint, palette } = theme;
  const width = Math.max(40, Math.min(columns - 1, 80));
  const inner = width - 4;
  const border = (text: string) => paint.fg(palette.border, text);
  const title = ` ${paint.fg(palette.agent, '✦')} ${paint.bold(paint.fg(palette.text, info.agent))} `;
  const version = info.version && info.version !== 'unknown' ? ` ${paint.fg(palette.faint, `webagents ${info.version}`)} ` : '';
  const fill = Math.max(0, width - 4 - visibleWidth(title) - visibleWidth(version));
  const top = `${border('╭─')}${title}${border('─'.repeat(fill))}${version}${border('─╮')}`;
  const row = (content = '') => `${border('│')} ${padEnd(content, inner)} ${border('│')}`;
  const field = (name: string, value: string, keepEnd = false) =>
    row(`${paint.fg(palette.faint, name.padEnd(7))}${paint.fg(palette.text, (keepEnd ? truncateStart : truncate)(value, inner - 7))}`);

  const lines = [top];
  if (info.description) lines.push(row(paint.fg(palette.muted, truncate(info.description, inner))));
  lines.push(row());
  if (info.model) lines.push(field('model', info.model));
  if (info.tools.length) {
    const shown: string[] = [];
    let used = 0;
    for (const tool of info.tools) {
      if (used + tool.length + 2 > inner - 7 - 10) break;
      shown.push(tool);
      used += tool.length + 2;
    }
    const more = info.tools.length - shown.length;
    lines.push(field('tools', `${shown.join(', ')}${more > 0 ? `  +${more} more` : ''}`));
  }
  lines.push(field('folder', info.folder, true));
  for (const warning of info.warnings) {
    // Wrapped, not cut: the advice is usually the end of the sentence.
    lines.push(row());
    for (const line of wrapStyled(paint.fg(palette.warning, warning), inner, `${paint.fg(palette.warning, '▲')} `, '  ')) {
      lines.push(row(line));
    }
  }
  lines.push(border(`╰${'─'.repeat(width - 2)}╯`));
  const hint = (key: string, what: string) => `${paint.fg(palette.muted, key)} ${paint.fg(palette.faint, what)}`;
  const sep = paint.fg(palette.faint, '  ·  ');
  lines.push(
    ` ${[hint('enter', 'send'), hint('/', 'commands'), hint('↑', 'history'), hint('esc', 'stop a reply')].join(sep)}`,
  );
  return lines;
}
