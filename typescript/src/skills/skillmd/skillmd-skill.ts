/**
 * The skill an agent carries for its SKILL.md skills (gap-closure plan item
 * 1.4, 2026-09-26). The Python twin is
 * `python/webagents/agents/skills/local/skillmd/skillmd_skill.py`; both run
 * `python/tests/fixtures/skillmd/skillmd.json`.
 *
 * PROGRESSIVE DISCLOSURE, three tiers, as agentskills.io describes it:
 *
 *   1. The catalog. A system-prompt section listing each skill's name,
 *      description and location in the `<available_skills>` format, about a
 *      hundred tokens per skill, omitted when there are none. It is shown to
 *      a caller only when that caller may call `activate_skill`, so a
 *      stranger an `access:` block keeps away from the tools does not read a
 *      menu it cannot order from.
 *   2. Activation. `activate_skill(name)`, `name` an enum of the loaded
 *      skills, returns the body of SKILL.md wrapped in
 *      `<skill_content name="...">`, then the skill's folder and the files it
 *      bundles, LISTED and not read. It never injects twice: an earlier tool
 *      result in the conversation that opens the same tag is what "already
 *      active" means, so the check needs no state and survives a restart of
 *      the daemon.
 *   3. Resources. `read_skill_file` reads one bundled text file, confined to
 *      the skill's folder by real path; `run_skill_script` runs one bundled
 *      script through the kernel sandbox (`src/sandbox/`, srt), with the
 *      skill's folder readable and write-denied, the agent's own `sandbox:`
 *      folders and `network:` list, and nothing else. An agent that declares
 *      no sandbox gets a synthesised `strict` policy with no writable folder
 *      but the private scratch; one that declares `unrestricted` is refused,
 *      because a skill fetched from a git repository is third-party code and
 *      runs confined or not at all.
 *
 * OWNER-ONLY BY DEFAULT (S-248, ADR-0045). All three tools carry
 * `scopes: ['owner']` until the agent file hands `agent_skills` (the whole
 * skill) or a tool's name to a group with `access: tools:`. The tools are
 * registered per instance rather than with `@tool`, because the `name` enum
 * is this agent's list of skills. The catalog prompt is registered by hand
 * too (`registerPrompt`), never with `@prompt`: `BaseAgent.addSkill` pushes
 * a skill's `prompts` AND its decorated prompts, so a decorated one lands in
 * the system message twice.
 *
 * `!`cmd`` substitutions and `$ARGUMENTS` in a body are text: nothing here
 * runs or expands them.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { Skill } from '../../core/skill';
import { callerScopes, scopeAllows } from '../../core/scopes';
import type { AgenticMessage, Context, Tool } from '../../core/types';

type JSONSchema = NonNullable<Tool['parameters']>;
import {
  INTERRUPTED_RESULT,
  SandboxUnavailable,
  parseSandboxDeclaration,
  policyFromDeclaration,
  refusalHint,
  runSandboxed,
  sandboxState,
  type SandboxDeclaration,
  type SandboxPolicy,
} from '../../sandbox/index';
import { activationMarker, activationText, catalogText, type SkillMd, type SkippedSkill } from './skillmd-loader';

/** The name the agent's skill map (and `access: tools:`) knows this skill by. */
export const SKILL_KEY = 'agent_skills';

export const TOOL_NAMES = ['activate_skill', 'read_skill_file', 'run_skill_script'] as const;

/** How a bundled script is run, by its extension; an executable file with another extension runs on its own. */
export const INTERPRETERS: Readonly<Record<string, string>> = {
  '.py': 'python3',
  '.sh': 'bash',
  '.js': 'node',
  '.mjs': 'node',
  '.cjs': 'node',
};

export const DEFAULT_TIMEOUT = 60;
export const MAX_TIMEOUT = 300;

/** The largest bundled file `read_skill_file` returns. */
export const READ_MAX_BYTES = 256 * 1024;

export const ACTIVATE_DESCRIPTION =
  'Load the full instructions of one available skill (see <available_skills>) into this conversation. ' +
  'Call it once per skill, before doing what the skill covers; the result also lists the files the skill bundles.';
export const READ_DESCRIPTION =
  "Read a text file bundled with an activated skill, by its path inside the skill's folder " +
  '(for example references/api.md). Only files inside that folder can be read.';
export const RUN_DESCRIPTION =
  'Run a script bundled with an activated skill inside the sandbox and return its output. ' +
  "script is the path inside the skill's folder (for example scripts/fill_form.py); " +
  "args are its command-line arguments; a relative path in args resolves against the agent's folder.";

export const UNRESTRICTED_REFUSAL =
  "Access denied: skill scripts run only in a sandbox, and this agent's sandbox is unrestricted, which is no sandbox";

/** The three tools' JSON schemas for a list of skill names (pinned by the fixture). */
export function toolParameters(names: readonly string[]): Record<(typeof TOOL_NAMES)[number], JSONSchema> {
  const enumeration = [...names].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return {
    activate_skill: {
      type: 'object',
      properties: {
        name: { type: 'string', enum: enumeration, description: "The skill's name, as listed in <available_skills>." },
      },
      required: ['name'],
    },
    read_skill_file: {
      type: 'object',
      properties: {
        skill: { type: 'string', enum: enumeration, description: "The skill's name." },
        path: { type: 'string', description: "The file's path inside the skill's folder." },
      },
      required: ['skill', 'path'],
    },
    run_skill_script: {
      type: 'object',
      properties: {
        skill: { type: 'string', enum: enumeration, description: "The skill's name." },
        script: { type: 'string', description: "The script's path inside the skill's folder." },
        args: { type: 'array', items: { type: 'string' }, description: 'Command-line arguments for the script.' },
        timeout: { type: 'number', description: 'Seconds before the script is stopped (default 60, at most 300).' },
      },
      required: ['skill', 'script'],
    },
  } as Record<(typeof TOOL_NAMES)[number], JSONSchema>;
}

export interface SkillMdSkillConfig {
  skills: readonly SkillMd[];
  skipped?: readonly SkippedSkill[];
  warnings?: readonly string[];
  /** The agent's folder: where scripts run and what a relative argument resolves against. */
  agentDir: string;
  /** The agent file's `sandbox:` block (checked, or as written), or none. */
  sandbox?: SandboxDeclaration | Record<string, unknown> | null;
  maxTimeout?: number;
}

/** `'...'` with single quotes escaped, as `shlex.quote` does. */
function shellQuote(value: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'"'"'`)}'`;
}

function inside(target: string, root: string): boolean {
  const trimmed = root.replace(/\/+$/, '');
  return target === trimmed || target.startsWith(trimmed + '/');
}

function looksBinary(head: Buffer): boolean {
  return head.includes(0);
}

function realOrNull(p: string): string | null {
  try {
    return fs.realpathSync(p);
  } catch {
    return null;
  }
}

/**
 * SKILL.md skills: a catalog in the prompt, activation on demand, and the
 * skill's files and scripts behind two confined tools (file comment).
 */
export class SkillMdSkill extends Skill {
  readonly skills: Map<string, SkillMd>;
  readonly skipped: SkippedSkill[];
  readonly warnings: string[];
  readonly agentDir: string;
  /** The agent file's `sandbox:` declaration, or null. */
  readonly sandboxDeclaration: SandboxDeclaration | Record<string, unknown> | null;
  private readonly timeoutLimit: number;

  constructor(config: SkillMdSkillConfig) {
    super({ name: 'SkillMdSkill' });
    const sorted = [...config.skills].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    this.skills = new Map(sorted.map((skill) => [skill.name, skill]));
    this.skipped = [...(config.skipped ?? [])];
    this.warnings = [...(config.warnings ?? [])];
    this.agentDir = realOrNull(config.agentDir) ?? path.resolve(config.agentDir);
    this.sandboxDeclaration = config.sandbox ?? null;
    this.timeoutLimit = config.maxTimeout ?? MAX_TIMEOUT;

    const parameters = toolParameters([...this.skills.keys()]);
    const owned = (tool: Omit<Tool, 'enabled' | 'scopes'>): Tool => ({ ...tool, scopes: ['owner'], enabled: true });
    this.registerTool(owned({ name: 'activate_skill', description: ACTIVATE_DESCRIPTION, parameters: parameters.activate_skill, handler: (params, context) => this.activate(params as { name?: unknown }, context) }));
    this.registerTool(owned({ name: 'read_skill_file', description: READ_DESCRIPTION, parameters: parameters.read_skill_file, handler: (params) => this.read(params as { skill?: unknown; path?: unknown }) }));
    this.registerTool(owned({ name: 'run_skill_script', description: RUN_DESCRIPTION, parameters: parameters.run_skill_script, handler: (params, context) => this.run(params as { skill?: unknown; script?: unknown; args?: unknown; timeout?: unknown }, context?.signal) }));
    this.registerPrompt({ name: 'skillsCatalog', priority: 60, scope: 'all', handler: (context) => this.skillsCatalog(context) });
  }

  // --------------------------------------------------------------------------
  // Tier 1: the catalog
  // --------------------------------------------------------------------------

  /** The scopes `activate_skill` carries now, which `access: tools:` may have rewritten. */
  private activateScopes(): string[] | undefined {
    return this.tools.find((tool) => tool.name === 'activate_skill')?.scopes;
  }

  /** Outside any run there is no caller: that is the process itself. */
  private callerMayActivate(context: Context | undefined): boolean {
    const auth = (context as { auth?: unknown } | undefined)?.auth;
    if (auth === undefined || auth === null) return true;
    return scopeAllows(this.activateScopes(), callerScopes(auth as Parameters<typeof callerScopes>[0]));
  }

  /** The `<available_skills>` section, for a caller who may activate one. */
  skillsCatalog(context?: Context): string {
    if (!this.skills.size || !this.callerMayActivate(context)) return '';
    return catalogText([...this.skills.values()]);
  }

  // --------------------------------------------------------------------------
  // Tier 2: activation
  // --------------------------------------------------------------------------

  private unknown(name: unknown): string {
    const available = this.skills.size ? [...this.skills.keys()].join(', ') : 'none';
    return `No skill called "${String(name)}". Available: ${available}`;
  }

  /** Whether an earlier tool result in this conversation opens the skill's tag: the conversation is the state. */
  private alreadyActive(name: string, context: Context | undefined): boolean {
    const marker = activationMarker(name);
    let messages: unknown;
    try {
      messages = context?.get?.('_agentic_messages');
    } catch {
      messages = undefined;
    }
    if (!Array.isArray(messages)) return false;
    for (const message of messages as Array<Partial<AgenticMessage> & { content?: unknown }>) {
      if (!message || message.role !== 'tool') continue;
      const content = message.content;
      if (typeof content === 'string' && content.includes(marker)) return true;
      const parts = Array.isArray(content) ? content : Array.isArray(message.content_items) ? message.content_items : [];
      for (const part of parts as Array<{ text?: unknown }>) {
        if (part && typeof part.text === 'string' && part.text.includes(marker)) return true;
      }
    }
    return false;
  }

  private async activate(params: { name?: unknown }, context: Context | undefined): Promise<string> {
    const skill = this.skills.get(String(params.name ?? ''));
    if (!skill) return this.unknown(params.name ?? '');
    if (this.alreadyActive(skill.name, context)) {
      return `Skill "${skill.name}" is already active in this conversation; its instructions are above.`;
    }
    return activationText(skill);
  }

  // --------------------------------------------------------------------------
  // Tier 3: resources
  // --------------------------------------------------------------------------

  /**
   * The real path of `relative` inside the skill's folder, or null when it is
   * not a regular file there (a symbolic link out of the folder resolves
   * outside it and is refused the same way).
   */
  private resolveInside(skill: SkillMd, relative: string): string | null {
    const candidate = realOrNull(path.join(skill.directory, relative));
    if (!candidate || !inside(candidate, skill.directory)) return null;
    try {
      if (!fs.statSync(candidate).isFile()) return null;
    } catch {
      return null;
    }
    return candidate;
  }

  private async read(params: { skill?: unknown; path?: unknown }): Promise<string> {
    const skill = this.skills.get(String(params.skill ?? ''));
    if (!skill) return this.unknown(params.skill ?? '');
    const relative = String(params.path ?? '');
    const target = this.resolveInside(skill, relative);
    if (!target) return `Access denied: ${relative} is not a file inside the skill folder`;
    let data: Buffer;
    try {
      if (fs.statSync(target).size > READ_MAX_BYTES) return `Access denied: ${relative} is larger than 256 KiB`;
      data = fs.readFileSync(target);
    } catch (err) {
      return `Error reading ${relative}: ${(err as Error).message}`;
    }
    if (looksBinary(data.subarray(0, 8192))) return `Access denied: ${relative} is not a text file`;
    return data.toString('utf8');
  }

  /**
   * The policy a script of `skill` runs under, or the refusal text.
   *
   * The agent's own `sandbox:` when it declares one; a synthesised `strict`
   * policy with no writable folder (only the private scratch) when it
   * declares none; a refusal for `unrestricted`. In every case the running
   * skill's folder is readable, and EVERY skill's folder is write-denied
   * (S-283, 2026-09-26): only the running skill's folder was, and the
   * review's `pdf` script wrote `.agents/skills/other/SKILL.md`, instructions
   * the next activation of `other` would have trusted. The agent's own files
   * (`AGENT.md`, `WEBAGENTS.md`, `mcp.json`, `.agents/skills`) are denied by
   * the policy's escalation set on top.
   */
  scriptPolicy(skill: SkillMd): SandboxPolicy | string {
    let base: SandboxPolicy;
    try {
      const declared = this.sandboxDeclaration ?? { preset: 'strict', allowed_folders: [] };
      base = policyFromDeclaration(parseSandboxDeclaration(declared), { cwd: this.agentDir });
    } catch (err) {
      return `Access denied: invalid sandbox declaration: ${(err as Error).message}`;
    }
    if (!base.confined) return UNRESTRICTED_REFUSAL;
    const readRoots = [...base.readRoots];
    if (base.scopedReads && !readRoots.includes(skill.directory)) readRoots.push(skill.directory);
    const readOnly = [...(base.readOnly ?? [])];
    for (const each of [skill, ...this.skills.values()]) {
      if (!readOnly.includes(each.directory)) readOnly.push(each.directory);
    }
    return { ...base, readRoots, readOnly };
  }

  /**
   * The state scripts run under, as the status row prints it: the agent
   * file's preset (or `off`, which refuses them) with `(agent file)`, or
   * `strict (default)`. For `doctor` and `/sandbox`: an agent that runs
   * SKILL.md scripts is not one that cannot run commands. `--no-sandbox`
   * does not apply to scripts.
   */
  scriptState(): string {
    if (!this.skills.size) return '';
    const declared = this.sandboxDeclaration !== null && this.sandboxDeclaration !== undefined;
    try {
      const policy = policyFromDeclaration(parseSandboxDeclaration(this.sandboxDeclaration ?? { preset: 'strict', allowed_folders: [] }), { cwd: this.agentDir });
      return sandboxState(policy, declared ? 'agent file' : 'default');
    } catch {
      return 'invalid (agent file)';
    }
  }

  /** The shell line for a bundled script, or null when it is not runnable. */
  scriptCommand(target: string, args: readonly string[]): string | null {
    const extension = path.extname(target).toLowerCase();
    const interpreter = INTERPRETERS[extension];
    let argv: string[];
    if (!interpreter) {
      try {
        fs.accessSync(target, fs.constants.X_OK);
      } catch {
        return null;
      }
      argv = [target, ...args];
    } else {
      argv = [interpreter, target, ...args];
    }
    return argv.map((part) => shellQuote(String(part))).join(' ');
  }

  private async run(params: { skill?: unknown; script?: unknown; args?: unknown; timeout?: unknown }, signal?: AbortSignal): Promise<string> {
    const skill = this.skills.get(String(params.skill ?? ''));
    if (!skill) return this.unknown(params.skill ?? '');
    const script = String(params.script ?? '');
    const target = this.resolveInside(skill, script);
    if (!target) return `Access denied: ${script} is not a file inside the skill folder`;
    const args = Array.isArray(params.args) ? (params.args as unknown[]).map((a) => String(a)) : [];
    const command = this.scriptCommand(target, args);
    if (command === null) {
      return `Access denied: ${script} is not a script this agent can run (.py, .sh, .js, .mjs, .cjs, or an executable file)`;
    }
    const policy = this.scriptPolicy(skill);
    if (typeof policy === 'string') return policy;
    let seconds = Number(params.timeout);
    if (!Number.isFinite(seconds) || params.timeout === undefined || params.timeout === null) seconds = DEFAULT_TIMEOUT;
    seconds = Math.max(1, Math.min(Math.trunc(seconds), this.timeoutLimit));

    let result;
    try {
      // The turn's Esc or Ctrl+C kills the script's process group (`signal`).
      result = await runSandboxed(command, policy, { timeout: seconds, maxBuffer: 1024 * 1024, signal });
    } catch (err) {
      if (err instanceof SandboxUnavailable) return `Access denied: ${err.message}`;
      return `Error running ${script}: ${(err as Error).message}`;
    }
    if (result.interrupted) return INTERRUPTED_RESULT;
    if (result.timedOut) return `Script timed out after ${seconds}s`;
    let output = result.stdout || '';
    if (result.stderr) output += `\nStderr: ${result.stderr}`;
    if (result.exitCode !== 0) output += `\nExit code: ${result.exitCode}`;
    // A script is a confined command too: the same one sentence names the switch (fixture `hints`).
    const hint = refusalHint(command, `${result.stdout}\n${result.stderr}`);
    if (hint) output += `\n${hint}`;
    return output || '(No output)';
  }

  // --------------------------------------------------------------------------
  // For doctor and `skills list`
  // --------------------------------------------------------------------------

  report(): { skills: string[]; skipped: SkippedSkill[]; warnings: string[] } {
    return {
      skills: [...this.skills.keys()],
      skipped: [...this.skipped],
      warnings: [...this.warnings, ...[...this.skills.values()].flatMap((skill) => skill.warnings.map((w) => `${skill.name}: ${w}`))],
    };
  }
}
