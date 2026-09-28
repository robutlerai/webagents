/**
 * A few lines the CLI adds after an agent's own instructions (2026-09-28).
 *
 * WHY. An agent with `filesystem` or `shell` listed its folder and read its
 * files on every message, "hi" included: nothing told it not to (the
 * small-talk rule went into the embedded ROBUTLER.md only), and nothing told
 * it that the `AGENT.md` it saw in the listing was its own definition, so it
 * opened that too. For an agent loaded from a file that has one of those
 * skills, the CLI adds where its instructions come from, its working folder,
 * and the small-talk rule. Agent-facing words; the person never sees them.
 *
 * ONE TEXT IN BOTH SDKS: `python/webagents/cli/preamble.py`, pinned by
 * `python/tests/fixtures/chat/turn_history.json` (`preamble`).
 */

import * as path from 'node:path';

/** The skills that let an agent look around its folder. */
export const EXPLORING_SKILLS: readonly string[] = ['filesystem', 'shell'];

export function cliPreamble(agentFile: string, folder: string): string {
  return (
    'Notes from the webagents CLI:\n' +
    `- Your instructions above come from ${agentFile}; you already have them, so there is no need to open that file to learn who you are.\n` +
    `- Your working folder is ${folder}.\n` +
    '- Answer greetings, thanks and small talk directly, without tools. Use a tool only when the request needs one, ' +
    'and do not explore the folder unless you are asked to.'
  );
}

function skillNames(skills: ReadonlyArray<unknown> | undefined): string[] {
  const out: string[] = [];
  for (const entry of skills ?? []) {
    if (typeof entry === 'string') out.push(entry.trim().toLowerCase());
    else if (entry && typeof entry === 'object') out.push(...Object.keys(entry).map((k) => k.trim().toLowerCase()));
  }
  return out;
}

/**
 * `instructions` with the preamble after them, for an agent loaded from a file
 * that can explore its folder; unchanged otherwise (the embedded agent has its
 * own rule).
 */
export function withCliPreamble<T extends string | undefined>(instructions: T, agentFile: string | undefined, skills: ReadonlyArray<unknown> | undefined): T | string {
  if (!agentFile || !skillNames(skills).some((n) => EXPLORING_SKILLS.includes(n))) return instructions;
  const file = path.resolve(agentFile);
  const note = cliPreamble(file, path.dirname(file));
  return instructions && instructions.trim() ? `${instructions.trimEnd()}\n\n${note}` : note;
}
