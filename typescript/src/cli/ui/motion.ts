/**
 * The moving parts of the chat: the spinner glyph and the shimmer that runs
 * across the status verb while the agent works (2026-09-24).
 *
 * Both are pure functions of time, so a redraw at any moment shows the right
 * frame and a test can ask for frame N without waiting for it. The live region
 * redraws on a timer (render.ts); these only say what a frame looks like.
 *
 * The spinner is a star that grows and shrinks (the one Claude Code made
 * familiar), the shimmer a soft band of light sweeping left to right across
 * the verb (as Codex does), in the accent colour.
 */

import { mix } from './ansi';
import type { Theme } from './theme';

/** A star that grows, then shrinks back: Claude Code's frames, played forward then back. */
const STAR = ['·', '✢', '✳', '✶', '✻', '✽'];
export const STAR_FRAMES = [...STAR, ...STAR.slice(1, -1).reverse()];
export const STAR_INTERVAL_MS = 120;

/** Braille dots, for a tool that is running (smaller than the star). */
export const DOT_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
export const DOT_INTERVAL_MS = 80;

export function frameAt(frames: string[], intervalMs: number, now: number): string {
  return frames[Math.floor(now / intervalMs) % frames.length];
}

/**
 * The star's colour: the agent's lime, breathing (full strength to 45% toward
 * the background and back every few seconds). One colour, pulsing: until
 * 2026-09-25 it cycled through a cyan-to-pink gradient, as Gemini CLI's
 * spinner does (theme.ts, "Signal").
 */
export function starColour(theme: Theme, now: number): string {
  return mix(theme.palette.agent, theme.background, 0.45 * (1 - (Math.sin(now / 650) + 1) / 2));
}

/** One sweep of the shimmer across the verb, in milliseconds. */
export const SHIMMER_PERIOD_MS = 2000;

/**
 * `text` with a band of light sweeping across it, Codex's formula
 * (`codex-rs/tui/src/summary_shimmer.rs`): the text rests half-faded toward
 * the background and a full-strength band, `max(10% of the width, 3)`
 * characters either side of its centre, crosses it every two seconds. Here
 * the peak also lifts a little toward white, so the band reads on violet.
 * Without 256 colours or animation, the text is simply in its colour.
 */
export function shimmer(theme: Theme, text: string, colour: string, now: number): string {
  const { paint } = theme;
  const chars = Array.from(text);
  if (!paint.on || !theme.animate || paint.depth < 256) return paint.fg(colour, text);
  const half = Math.max(0.1 * chars.length, 3);
  const position = ((now % SHIMMER_PERIOD_MS) / SHIMMER_PERIOD_MS) * (chars.length + 2 * half) - half;
  return chars
    .map((ch, i) => {
      const distance = Math.min(Math.abs(i + 0.5 - position) / half, 1);
      const intensity = 0.5 * (1 + Math.cos(Math.PI * distance));
      const lit = mix(colour, '#ffffff', 0.22 * intensity);
      return paint.fg(mix(theme.background, lit, 0.5 + 0.5 * intensity), ch);
    })
    .join('');
}

/** The glyphs of the idle starfield in an empty input box (Codex's `sparkle_field.rs`). */
export const SPARKS = ['⠁', '⠂', '⠄', '⠈', '⠐', '⠠', '⡀', '⢀'];
/** How long the starfield lasts after the box appears. */
export const SPARKLE_MS = 15000;

/**
 * What one cell of the idle starfield shows at `now`: a braille dot that
 * twinkles on its own 4 to 7 second cycle (brightness sin^12, at most 0.55),
 * in one cell out of five, or nothing, in the agent's lime. Deterministic
 * per column, so a redraw never makes the field jump.
 */
export function sparkAt(theme: Theme, column: number, now: number): string {
  const hash = (n: number) => {
    let x = (n + 1) * 2654435761;
    x ^= x >>> 15;
    x = Math.imul(x, 2246822519);
    x ^= x >>> 13;
    return (x >>> 0) / 4294967296;
  };
  if (hash(column) >= 0.2) return ' ';
  const period = 4000 + hash(column + 101) * 3000;
  const phase = ((now + hash(column + 211) * period) % period) / period;
  const brightness = Math.sin(Math.PI * phase) ** 12 * 0.55;
  if (brightness < 0.04) return ' ';
  const colour = mix(theme.background, theme.palette.agent, brightness);
  return theme.paint.fg(colour, SPARKS[Math.floor(hash(column + 307) * SPARKS.length)]);
}
