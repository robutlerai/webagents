/**
 * Builtin embedded agents.
 * 
 * These agents are bundled with the webagents package and serve as defaults
 * when no local agent files are present.
 */

import { existsSync, lstatSync, readFileSync, readdirSync } from 'fs';
import { parse as parseYaml } from 'yaml';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { SandboxDeclarationError, parseSandboxDeclaration, unknownKeysMessage, type SandboxDeclaration } from '../sandbox/policy';
import { ObservabilityConfigError, parseObservability, type ObservabilityConfig } from '../observability/otel';
import { parseMaxToolRounds } from '../core/tool-budget';
import { parseCronBlock } from './schedules';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * The keys an agent file's front matter may carry: the Python loader's schema
 * (`cli/loader/schema.py`, `AgentMetadata`), sorted as its sentence lists
 * them. Anything else is refused with that sentence (2026-09-26, D7 of the
 * interactive-mode review): `skils:` loaded here silently, with no skills,
 * while the Python CLI refused it. Keys this parser does not model (`tools`,
 * `visibility`, ...) are kept in `extra`, as before.
 */
export const AGENT_FILE_KEYS: readonly string[] = [
  'access',
  'agent_skills',
  'author',
  'cron',
  'description',
  'fallback_models',
  'intents',
  'mcp_servers',
  'model',
  'name',
  'namespace',
  'max_tool_rounds',
  'observability',
  'sandbox',
  'scopes',
  'skills',
  'tags',
  'tools',
  'version',
  'visibility',
  'watch',
];

/**
 * The bytes of an agent file, refusing a symbolic link (2026-09-26, S-290).
 * A cloned repository can carry `AGENT-x.md -> ../somewhere/else`, and every
 * loader read through it; the editors then wrote through it. Every loader
 * in the CLI reads through this, so a link is never an agent here. The
 * Python loader (`cli/loader/agent_md.py`) refuses the same way, with the
 * same sentence (fixture `chat_edits.json`, `loader.linked`).
 */
export function readAgentFile(filePath: string): string {
  if (lstatSync(filePath).isSymbolicLink()) {
    throw new AgentFileError(`${filePath}: is a symbolic link; an agent file must be a regular file in its folder.`);
  }
  return readFileSync(filePath, 'utf-8');
}

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
  /**
   * The `sandbox:` block, checked (`sandbox/policy.ts`), when the file has
   * one (2026-09-26, gap-closure plan item 1.2). It used to land in `extra`
   * and nothing read it, so the same AGENT.md that was confined under the
   * Python CLI ran its shell unconfined here (S-248 addendum). A block with
   * an unknown key is refused at parse with the Python loader's sentence
   * (S-270), not dropped.
   */
  sandbox?: SandboxDeclaration;
  /**
   * The `cron:` block as written (plan item 1.7, 2026-09-26), when the file
   * has one. `agents/schedules.ts` `parseCronBlock` is its one reader; the
   * daemon runs it and refuses the file with its sentence when the block is
   * malformed, as it does for `access`.
   */
  cron?: unknown;
  /**
   * `agent_skills:` (plan item 1.4, 2026-09-26): folders holding SKILL.md
   * skills kept outside `.agents/skills`, each a skill folder or a folder of
   * skill folders, relative to the agent file. `skills:` names CODED skills;
   * this names the other kind (`skills/skillmd/skillmd-loader.ts`). Absent
   * when the file has none; the Python loader reads the same key.
   */
  agentSkills?: string[];
  /**
   * `fallback_models:` (plan item 2.8, 2026-09-26): the models to try, in
   * order, when the agent's model fails with a provider error (a 5xx, a
   * 429, no answer), each `provider/model`. Refused with the fixture's
   * sentence when it is not a list of strings
   * (`python/tests/fixtures/w2ops/models.json`, `failover`).
   */
  fallbackModels?: string[];
  /**
   * `observability:` (plan item 2.4, 2026-09-26): `{otel: true}` records the
   * run as OpenTelemetry spans (`observability/otel.ts`). Refused at parse
   * with the Python schema's sentence when it says something else.
   */
  observability?: ObservabilityConfig;
  /**
   * `max_tool_rounds:` (2026-09-28, `core/tool-budget.ts`): the tool rounds
   * one turn may run before its last, tool-less answer. Refused at parse
   * with the shared sentence unless a whole number from 1 to 1000.
   */
  maxToolRounds?: number;
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
 * Front matter that is not YAML is REFUSED with a sentence naming the file
 * (2026-09-26, D7 of the interactive-mode review). It used to fall back to
 * treating the whole file as instructions, so a comment that broke the YAML
 * sent the front matter to the model as prose, under the name `default`,
 * with no tools; the Python loader refused the same file. CRLF line ends
 * parse; an unknown top-level key and the old string form of `cron:` are
 * refused with the Python loader's sentences, so a file means one thing to
 * both SDKs. `serve`, the daemon and the chat all read through here.
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
  const where = filePath ?? 'agent file';
  const bare = (instructions: string): ParsedAgent => ({
    name: fallbackName,
    description: '',
    skills: [],
    skillEntries: [],
    intents: [],
    extra: {},
    instructions,
  });

  // A BOM and CRLF line ends are the editor's business, not the file's
  // meaning: the Python loader reads the file with universal newlines.
  const text = content.replace(/^﻿/, '').replace(/\r\n?/g, '\n');
  const frontmatterMatch = text.match(/^---[ \t]*\n([\s\S]*?)\n---[ \t]*(?:\n([\s\S]*))?$/);
  if (!frontmatterMatch) return bare(text);

  const [, frontmatter, body = ''] = frontmatterMatch;

  // With the line and column when the parser knows them (2026-09-26, the e2e
  // run): the file's line, the opening fence being line 1, so the number
  // points at the file the person opens. The Python loader does the same
  // from PyYAML's mark; both sentences are `chat_edits.json` `loader`.
  const notYaml = (position?: { line: number; col: number }) =>
    new AgentFileError(
      position
        ? `The front matter of ${where} is not valid YAML (line ${position.line + 1}, column ${position.col}). Fix it, then try again.`
        : `The front matter of ${where} is not valid YAML. Fix it, then try again.`,
    );
  let data: Record<string, unknown> = {};
  try {
    data = (parseYaml(frontmatter) ?? {}) as Record<string, unknown>;
  } catch (error) {
    const linePos = (error as { linePos?: Array<{ line: number; col: number }> } | null)?.linePos;
    throw notYaml(Array.isArray(linePos) && linePos[0] ? linePos[0] : undefined);
  }
  if (typeof data !== 'object' || Array.isArray(data)) throw notYaml();

  // The keys the Python schema knows, and nothing else (file comment).
  const unknown = Object.keys(data).filter((key) => !AGENT_FILE_KEYS.includes(key));
  if (unknown.length) {
    throw new AgentFileError(`${where}: ${unknownKeysMessage(unknown, AGENT_FILE_KEYS, 'AGENT.md frontmatter')}`);
  }

  // `cron:` is checked here as the Python schema checks it: the string form
  // and a malformed schedule stop the file with the shared sentence rather
  // than loading an agent whose schedules silently do not exist.
  if (data.cron !== undefined && data.cron !== null) {
    try {
      parseCronBlock(data.cron);
    } catch (err) {
      if (err instanceof AgentFileError) throw new AgentFileError(`${where}: ${err.message}`);
      throw err;
    }
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
    'sandbox',
    'cron',
    'agent_skills',
    'fallback_models',
    'observability',
    'max_tool_rounds',
  ]);
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (!modelled.has(key)) extra[key] = value;
  }

  // A `sandbox:` block is checked HERE, not left for the shell to find: a
  // mistyped key must stop the file with its sentence (S-270), and a file
  // that declares a sandbox and could not be enforced must never load as an
  // agent that runs commands unconfined (S-248 addendum). A YAML error still
  // falls back above; this is a well-formed file saying something wrong.
  let sandbox: SandboxDeclaration | undefined;
  if (data.sandbox !== undefined && data.sandbox !== null) {
    try {
      sandbox = parseSandboxDeclaration(data.sandbox);
    } catch (err) {
      if (err instanceof SandboxDeclarationError) {
        throw new AgentFileError(`${where}: ${err.message}`);
      }
      throw err;
    }
  }

  // `agent_skills:` is a list of folder paths or nothing; anything else is
  // refused with the Python loader's sentence (fixture `skillmd.json`).
  let agentSkills: string[] | undefined;
  if (data.agent_skills !== undefined && data.agent_skills !== null) {
    const entries = data.agent_skills;
    if (!Array.isArray(entries) || !entries.every((entry) => typeof entry === 'string' && entry.trim())) {
      throw new AgentFileError(`${where}: agent_skills: must be a list of folder paths`);
    }
    agentSkills = entries as string[];
  }

  // `fallback_models:` is a list of provider/model strings or nothing;
  // anything else is refused with the shared sentence (plan item 2.8).
  let fallbackModels: string[] | undefined;
  if (data.fallback_models !== undefined && data.fallback_models !== null) {
    const entries = data.fallback_models;
    if (!Array.isArray(entries) || !entries.every((entry) => typeof entry === 'string' && entry.trim())) {
      throw new AgentFileError(`${where}: fallback_models: must be a list of provider/model strings`);
    }
    fallbackModels = (entries as string[]).map((entry) => entry.trim());
  }

  // `observability:` is checked here as the Python schema checks it (plan
  // item 2.4): an unknown key stops the file with the shared sentence.
  let observability: ObservabilityConfig | undefined;
  if (data.observability !== undefined && data.observability !== null) {
    try {
      observability = parseObservability(data.observability);
    } catch (err) {
      if (err instanceof ObservabilityConfigError) throw new AgentFileError(`${where}: ${err.message}`);
      throw err;
    }
  }

  // `max_tool_rounds:` is checked here as the Python schema checks it
  // (2026-09-28): the shared sentence stops the file.
  let maxToolRounds: number | undefined;
  if (data.max_tool_rounds !== undefined && data.max_tool_rounds !== null) {
    try {
      maxToolRounds = parseMaxToolRounds(data.max_tool_rounds);
    } catch (err) {
      throw new AgentFileError(`${where}: ${(err as Error).message}`);
    }
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
    ...(sandbox ? { sandbox } : {}),
    ...('cron' in data ? { cron: data.cron } : {}),
    ...(agentSkills ? { agentSkills } : {}),
    ...(fallbackModels ? { fallbackModels } : {}),
    ...(observability ? { observability } : {}),
    ...(maxToolRounds !== undefined ? { maxToolRounds } : {}),
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
