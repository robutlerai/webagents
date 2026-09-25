/**
 * Asking the terminal what it looks like (2026-09-24).
 *
 * The background colour is the one fact a palette cannot guess: a shaded band
 * that reads as a gentle lift on one dark theme is a grey slab on another, and
 * is wrong on every light one. Gemini CLI and Codex both ask (OSC 11) and mix
 * their surfaces from the answer; so does this.
 *
 * The query is sent together with Primary Device Attributes (DA1), which every
 * terminal answers, and terminals answer in order: DA1 arriving first means
 * the terminal ignores OSC 11, and the wait ends there instead of at a
 * timeout. The wait always ends at DA1, never at the colour, so neither reply
 * can land in the input box later as stray characters. tmux answers DA1 itself and forwards OSC 11 only with
 * passthrough on, so under tmux this usually reports "unknown"; the palette's
 * own values are the fallback.
 */

const OSC11 = /\x1b\]11;rgb:([0-9a-fA-F]{1,4})\/([0-9a-fA-F]{1,4})\/([0-9a-fA-F]{1,4})/;
const DA1 = /\x1b\[\?[0-9;]*c/;

function channel(hex: string): string {
  const value = parseInt(hex, 16) / (16 ** hex.length - 1);
  return Math.round(value * 255)
    .toString(16)
    .padStart(2, '0');
}

/** Parses a terminal's reply; `#rrggbb` for an OSC 11 answer, `null` when there is none (yet). */
export function parseBackgroundReply(reply: string): { background: string | null; complete: boolean } {
  const osc = OSC11.exec(reply);
  const background = osc ? `#${channel(osc[1])}${channel(osc[2])}${channel(osc[3])}` : null;
  // Complete only once DA1 has come: it follows any OSC 11 answer, and
  // stopping at the colour left the DA1 reply to land in the input box.
  return { background, complete: DA1.test(reply) };
}

/** The terminal's background colour as `#rrggbb`, or null when it will not say. */
export function queryBackground(
  input: NodeJS.ReadStream = process.stdin,
  output: NodeJS.WriteStream = process.stdout,
  timeoutMs = 400,
): Promise<string | null> {
  if (!input.isTTY || !output.isTTY || typeof input.setRawMode !== 'function') return Promise.resolve(null);
  return new Promise((resolve) => {
    let reply = '';
    const wasRaw = input.isRaw;
    const finish = (value: string | null) => {
      clearTimeout(timer);
      input.removeListener('data', onData);
      if (!wasRaw) input.setRawMode(false);
      input.pause();
      resolve(value);
    };
    const onData = (data: Buffer | string) => {
      reply += typeof data === 'string' ? data : data.toString('latin1');
      const parsed = parseBackgroundReply(reply);
      if (parsed.complete) finish(parsed.background);
    };
    const timer = setTimeout(() => finish(parseBackgroundReply(reply).background), timeoutMs);
    input.setRawMode(true);
    input.on('data', onData);
    input.resume();
    output.write('\x1b]11;?\x1b\\\x1b[c');
  });
}
