/**
 * The chat's colours, in one place (2026-09-24).
 *
 * SIGNAL (2026-09-25, the operator's choice): Robutler's black and white, and
 * one lime that means the agent. The WEBAGENTS wordmark is white letters with
 * a grey shadow (black and grey on a light terminal); lime is the agent's
 * mark, its H1 headings, the spinner, the idle sparkles and the light that
 * crosses the wordmark once. Until that day the wordmark, H1, the input box
 * border and a hue-cycling spinner all ran through a cyan-to-pink gradient,
 * which with the gradient-edged input box read as Gemini CLI's look. The
 * person's side is plain white, and the input box border is the plain border.
 *
 * The rest is a small semantic palette, so a colour means one thing
 * everywhere: green (emerald, so it never reads as the agent's lime) is a
 * tool that worked, red one that failed, amber a warning, blue a link.
 *
 * Both CLIs hold this palette, value for value, against
 * `python/tests/fixtures/cli/theme.json`; the Python twin is
 * `python/webagents/cli/ui/theme.py`. Two variants, because the same #hex
 * that reads well on a dark terminal is too pale on a light one
 * (detectLightBackground).
 */

import { Paint, detectColorDepth, detectLightBackground, hexToRgb, mix, type ColorDepth } from './ansi';

export interface Palette {
  /** The wordmark's letters (the `█` cells). */
  wordmarkFace: string;
  /** The wordmark's shadow (its box-drawing strokes). */
  wordmarkShadow: string;
  /** The person: prompt character, their messages, a running tool's dots. */
  accent: string;
  /** The agent, and the one colour of the brand: reply marker, bullets, H1, the spinner. */
  agent: string;
  success: string;
  error: string;
  warning: string;
  info: string;
  text: string;
  muted: string;
  faint: string;
  border: string;
  /** The band behind a sent message; only drawn at 256 colours and up. */
  surface: string;
  /** The band behind a code block, a shade quieter than `surface`. */
  codeSurface: string;
  inlineCode: string;
  code: {
    keyword: string;
    string: string;
    number: string;
    comment: string;
    func: string;
    type: string;
    property: string;
    punctuation: string;
    added: string;
    removed: string;
  };
}

export const DARK: Palette = {
  wordmarkFace: '#f4f4f5',
  wordmarkShadow: '#52525b',
  accent: '#fafafa',
  agent: '#bef264',
  success: '#34d399',
  error: '#f87171',
  warning: '#fbbf24',
  info: '#60a5fa',
  text: '#e4e4e7',
  muted: '#a1a1aa',
  faint: '#71717a',
  border: '#52525b',
  surface: '#26272b',
  codeSurface: '#1f2023',
  inlineCode: '#fdba74',
  code: {
    keyword: '#c084fc',
    string: '#86efac',
    number: '#fdba74',
    comment: '#6b7280',
    func: '#93c5fd',
    type: '#67e8f9',
    property: '#f9a8d4',
    punctuation: '#9ca3af',
    added: '#4ade80',
    removed: '#f87171',
  },
};

export const LIGHT: Palette = {
  wordmarkFace: '#18181b',
  wordmarkShadow: '#a1a1aa',
  accent: '#18181b',
  agent: '#4d7c0f',
  success: '#047857',
  error: '#dc2626',
  warning: '#b45309',
  info: '#2563eb',
  text: '#18181b',
  muted: '#52525b',
  faint: '#71717a',
  border: '#a1a1aa',
  surface: '#f1f1f2',
  codeSurface: '#f6f6f7',
  inlineCode: '#c2410c',
  code: {
    keyword: '#7c3aed',
    string: '#15803d',
    number: '#c2410c',
    comment: '#6b7280',
    func: '#1d4ed8',
    type: '#0e7490',
    property: '#be185d',
    punctuation: '#4b5563',
    added: '#15803d',
    removed: '#b91c1c',
  },
};

export interface Theme {
  paint: Paint;
  palette: Palette;
  /** The terminal's background: asked for (ui/terminal.ts), or assumed from `light`. */
  background: string;
  /** Cursor movement and a live region are possible (a terminal, not a pipe). */
  interactive: boolean;
  /** Animations may run (a terminal, and not switched off). */
  animate: boolean;
  light: boolean;
}

/** Relative luminance, 0 (black) to 1 (white). */
function luminance(hex: string): number {
  const { r, g, b } = hexToRgb(hex);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

/**
 * The theme for `out`, from the environment. With the terminal's own
 * `background` (queryBackground), light or dark is read from it rather than
 * guessed, and the shaded bands are mixed from it, the way Gemini CLI and
 * Codex do: a lift toward white on a dark terminal, toward black on a light one.
 */
export function themeFor(
  out: { isTTY?: boolean } = process.stdout,
  env: NodeJS.ProcessEnv = process.env,
  overrides: { depth?: ColorDepth; interactive?: boolean; animate?: boolean; background?: string | null } = {},
): Theme {
  const interactive = overrides.interactive ?? Boolean(out.isTTY);
  const depth = overrides.depth ?? detectColorDepth(env, interactive);
  const known = overrides.background ?? null;
  const light = known ? luminance(known) > 0.55 : detectLightBackground(env);
  const animate =
    overrides.animate ?? (interactive && !env.CI && env.WEBAGENTS_NO_ANIMATION === undefined && env.TERM !== 'dumb');
  const base = light ? LIGHT : DARK;
  const palette: Palette = known
    ? {
        ...base,
        surface: light ? mix(known, '#000000', 0.06) : mix(known, '#ffffff', 0.12),
        codeSurface: light ? mix(known, '#000000', 0.035) : mix(known, '#ffffff', 0.065),
      }
    : base;
  const background = known ?? (light ? '#ffffff' : '#16181d');
  return { paint: new Paint(depth), palette, background, interactive, animate, light };
}
