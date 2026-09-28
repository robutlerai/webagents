/**
 * Which agent a command means (2026-09-24), by the rules and in the words of
 * the Python CLI (`python/webagents/cli/agent_files.py`):
 *
 *  - no name: this folder's AGENT.md, else its only AGENT-<name>.md, else the
 *    built-in assistant (`findAgentFile`);
 *  - `-a <name>`: the agent in this folder called that (its `name:`, or the
 *    `<name>` of AGENT-<name>.md), or the built-in one by its own name.
 *
 * A name that matches nothing is refused, naming the agents that are here.
 * `findAgentFile(dir, name)` falls back to AGENT.md when AGENT-<name>.md is
 * missing, so `-a <typo>` opened this folder's other agent under the name
 * that was typed.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { parseAgentMarkdown, readAgentFile } from '../agents/index';

/** The assistant that runs where there is no agent file, in both CLIs. */
export const BUILT_IN_AGENT = 'robutler';

export interface FolderAgent {
  name: string;
  file: string;
  description: string;
  /**
   * Why the file does not load, when it does not (2026-09-26, D2 of the
   * interactive-mode review): the chat lists such a file with its sentence
   * and refuses to switch to it, rather than hiding it or ending. Its name
   * is the file's (`AGENT-bad.md` is `bad`), as the loaders name a file
   * with no `name:`.
   */
  problem?: string;
}

/** `-a <name>` named no agent in this folder. */
export class AgentNotFound extends Error {}

/** The name a file goes by when its front matter cannot say: `AGENT.md` is `default`, `AGENT-<name>.md` is `<name>`. */
export function nameForFile(file: string): string {
  const base = path.basename(file).replace(/\.md$/i, '');
  if (base === 'AGENT') return 'default';
  return base.startsWith('AGENT-') ? base.slice('AGENT-'.length) : base;
}

/** The agent files in `folder` (AGENT.md and AGENT-<name>.md), with their names; a broken one carries its `problem`. */
export function folderAgents(folder: string): FolderAgent[] {
  let names: string[];
  try {
    names = fs.readdirSync(folder).filter((f) => f === 'AGENT.md' || /^AGENT-.+\.md$/.test(f)).sort();
  } catch {
    return [];
  }
  const out: FolderAgent[] = [];
  for (const f of names) {
    const file = path.join(folder, f);
    try {
      const parsed = parseAgentMarkdown(readAgentFile(file), file);
      out.push({ name: parsed.name, file, description: parsed.description ?? '' });
    } catch (error) {
      // A file that does not load is still listed, with why (file comment).
      out.push({ name: nameForFile(file), file, description: '', problem: (error as Error).message });
    }
  }
  return out;
}

/**
 * `-a <name>`: that agent's file, or null for the built-in one.
 * Throws `AgentNotFound`, whose message names the agents that are here.
 */
export function agentFileFor(folder: string, name: string): string | null {
  const wanted = name.trim();
  const agents = folderAgents(folder);
  const match = agents.find((a) => a.name === wanted || path.basename(a.file) === `AGENT-${wanted}.md`);
  if (match) return match.file;
  if (wanted === BUILT_IN_AGENT) return null;
  const here = agents.map((a) => a.name).join(', ');
  throw new AgentNotFound(
    `There is no agent called ${wanted} in this folder. ` +
      (here ? `Agents here: ${here}, and the built-in ${BUILT_IN_AGENT}.` : `The built-in ${BUILT_IN_AGENT} runs without -a.`),
  );
}
