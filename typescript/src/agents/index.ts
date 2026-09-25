/**
 * Builtin embedded agents.
 * 
 * These agents are bundled with the webagents package and serve as defaults
 * when no local agent files are present.
 */

import { existsSync, readFileSync, readdirSync } from 'fs';
import { parse as parseYaml } from 'yaml';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Get the path to the embedded ROBUTLER.md agent.
 */
export function getRobutlerPath(): string {
  return join(__dirname, 'ROBUTLER.md');
}

/**
 * Get the content of the embedded ROBUTLER.md agent.
 */
export function getRobutlerContent(): string {
  return readFileSync(getRobutlerPath(), 'utf-8');
}

/**
 * An agent file that cannot be used as written, with a sentence for its author
 * (`<file>: <what to fix>`). The CLI prints the message alone and exits 1, as
 * the Python CLI does for its `AgentFormatError`.
 */
export class AgentFileError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AgentFileError';
  }
}

/** A skill as written: a bare name, or a single-key object carrying its config. */
export type SkillEntry = string | Record<string, unknown>;

/** What an AGENT.md declares. */
export interface ParsedAgent {
  name: string;
  description: string;
  /** Skill names, as written. A dict-form entry (`- mcp: {...}`) yields its key. */
  skills: string[];
  /**
   * The same list UNFLATTENED, so a caller that can honour a skill's config
   * still has it. `skills` above is the convenience view.
   */
  skillEntries: SkillEntry[];
  /** `provider/model`, when the file declares one. */
  model?: string;
  namespace?: string;
  intents: string[];
  /**
   * The `access:` block as written (ADR-0045), when the file has one.
   * `access/policy.ts` `parseAccess` is its one reader; the loaders run it and
   * refuse the file with its sentence when the block is malformed.
   */
  access?: unknown;
  /** Frontmatter keys this interface does not model, kept rather than dropped. */
  extra: Record<string, unknown>;
  instructions: string;
}

/**
 * Infer an agent name from its filename, matching the Python loader's rule
 * (`python/webagents/cli/loader/agent_md.py`): `AGENT.md` is `default`, and
 * `AGENT-<name>.md` is `<name>`.
 */
function nameFromFilename(filePath: string): string {
  const base = filePath.replace(/^.*[\\/]/, '').replace(/\.md$/i, '');
  if (/^AGENT$/i.test(base)) return 'default';
  const named = base.replace(/^AGENT[-_.]?/i, '');
  return named || 'agent';
}

/**
 * Parse agent frontmatter from markdown content.
 *
 * Uses a real YAML parser (2026-09-23). This was a hand-rolled line scanner
 * that matched `name:`, `description:` and a `skills:` block with a
 * `startsWith('-')` heuristic. It handled exactly those three keys, could not
 * see `model:` at all, and produced nothing for a dict-form skill entry like
 * `- mcp: {command: ...}`, which is a documented form. `yaml` is already a
 * runtime dependency of this package, so there was never a reason to hand-roll
 * it.
 *
 * Malformed frontmatter falls back to treating the whole file as instructions
 * rather than throwing: a REPL that refuses to start because a comment broke
 * the YAML is worse than one that runs with a plain prompt and says so.
 *
 * THIS IS THE ONLY AGENT.md PARSER IN THE PACKAGE (2026-09-23). There were
 * three, and the other two were line scanners that disagreed with each other
 * and with the documented format:
 *
 *   * `daemon/watcher.ts` matched `^(\w+):\s*(.*)$`, so a block-style
 *     `skills:` list produced `['']` rather than the skills, and it could see
 *     only four keys.
 *   * `core/extensions/local-dev.ts` split `skills` on commas, so the same
 *     block list produced `[]`, and swept every other key into a `config`
 *     object that nothing ever read.
 *
 * Pass `filePath` to get the Python loader's name-from-filename fallback;
 * omit it and an unnamed agent stays `unknown`.
 */
export function parseAgentMarkdown(content: string, filePath?: string): ParsedAgent {
  const fallbackName = filePath ? nameFromFilename(filePath) : 'unknown';
  const bare = (instructions: string): ParsedAgent => ({
    name: fallbackName,
    description: '',
    skills: [],
    skillEntries: [],
    intents: [],
    extra: {},
    instructions,
  });

  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!frontmatterMatch) return bare(content);

  const [, frontmatter, body] = frontmatterMatch;

  let data: Record<string, unknown> = {};
  try {
    data = (parseYaml(frontmatter) ?? {}) as Record<string, unknown>;
  } catch {
    return bare(content);
  }

  // Skills may be bare strings or single-key objects carrying config. Both
  // resolve to the skill NAME in `skills`; `skillEntries` keeps the original
  // so a caller that can use the config still has it.
  const skillEntries = (Array.isArray(data.skills) ? data.skills : []).filter(
    (entry): entry is SkillEntry =>
      typeof entry === 'string' || (!!entry && typeof entry === 'object'),
  );
  const skills = skillEntries
    .map((entry) => {
      if (typeof entry === 'string') return entry;
      const keys = Object.keys(entry);
      return keys.length === 1 ? keys[0] : '';
    })
    .filter(Boolean);

  const modelled = new Set([
    'name',
    'description',
    'skills',
    'model',
    'namespace',
    'intents',
    'access',
  ]);
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (!modelled.has(key)) extra[key] = value;
  }

  return {
    name: typeof data.name === 'string' ? data.name : fallbackName,
    description: typeof data.description === 'string' ? data.description : '',
    skills,
    skillEntries,
    model: typeof data.model === 'string' ? data.model : undefined,
    namespace: typeof data.namespace === 'string' ? data.namespace : undefined,
    intents: (Array.isArray(data.intents) ? data.intents : []).filter(
      (i): i is string => typeof i === 'string',
    ),
    ...('access' in data ? { access: data.access } : {}),
    extra,
    instructions: body.trim(),
  };
}

/**
 * Find the agent file to use, mirroring the Python CLI's resolution rules.
 *
 * 1. an explicit `AGENT-<name>.md` when a name is given
 * 2. `AGENT.md`
 * 3. the single `AGENT-*.md`, if there is exactly one (ambiguous otherwise)
 *
 * Returns null when nothing matches, leaving the caller to fall back to the
 * embedded agent. Kept in step with `python/webagents/cli/commands/agent.py`.
 */
export function findAgentFile(dir: string, name?: string): string | null {
  const candidates: string[] = [];
  if (name) candidates.push(join(dir, `AGENT-${name}.md`));
  candidates.push(join(dir, 'AGENT.md'));

  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }

  if (!name) {
    const named = readdirSync(dir).filter((f) => /^AGENT-.+\.md$/.test(f));
    // More than one is ambiguous, and guessing is worse than falling back.
    if (named.length === 1) return join(dir, named[0]);
  }
  return null;
}
