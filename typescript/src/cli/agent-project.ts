/**
 * The agent a directory describes, read ONE way for every command
 * (2026-09-24).
 *
 * Found walking the TypeScript onboarding path as a first-time developer.
 * `webagents init` wrote `agent.json` plus `instructions.md`, and nothing read
 * `instructions.md`: `serve` took `instructions` from `agent.json` (the scaffold
 * leaves it out), so a newcomer's first agent ran with no instructions at all,
 * and `publish` posted `agent.json` verbatim, which the portal refused because
 * `skills` was a list where its schema wants a mapping. The docs, meanwhile,
 * describe `AGENT.md`, the one format both SDKs parse (`parseAgentMarkdown`,
 * mirroring `python/webagents/cli/loader/`).
 *
 * So, in order: `AGENT.md` (what `init` now writes), then `agent.json` with
 * `instructions.md` beside it when the JSON carries none (every project made
 * before this). `serve` and `publish` both come through here.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { parseAgentMarkdown, type SkillEntry } from '../agents/index.js';

export interface AgentProject {
  name: string;
  description?: string;
  instructions?: string;
  model?: string;
  /** Skill names, for code that only needs names (`serve`). */
  skills: string[];
  /** The skills as written, config included, for the platform payload. */
  skillEntries: SkillEntry[];
  intents?: string[];
  /** The `access:` block as written (ADR-0045), when the file has one. */
  access?: unknown;
  /** The file it was read from, for messages. */
  source: string;
}

/** Thrown when a file exists and cannot be used; absence is not an error here. */
export class AgentProjectError extends Error {}

function readText(filePath: string): string {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (err) {
    throw new AgentProjectError(`Cannot read ${filePath}: ${(err as Error).message}`);
  }
}

function fromMarkdown(filePath: string): AgentProject {
  const parsed = parseAgentMarkdown(readText(filePath), filePath);
  return {
    name: parsed.name,
    description: parsed.description || undefined,
    instructions: parsed.instructions || undefined,
    model: parsed.model,
    skills: parsed.skills,
    skillEntries: parsed.skillEntries,
    intents: parsed.intents.length ? parsed.intents : undefined,
    ...(parsed.access !== undefined ? { access: parsed.access } : {}),
    source: filePath,
  };
}

function fromJson(filePath: string): AgentProject {
  let config: Record<string, unknown>;
  try {
    config = JSON.parse(readText(filePath)) as Record<string, unknown>;
  } catch (err) {
    if (err instanceof AgentProjectError) throw err;
    throw new AgentProjectError(`${filePath} is not valid JSON: ${(err as Error).message}`);
  }

  // The scaffold keeps instructions in their own file; `agent.json` never had them.
  let instructions = typeof config.instructions === 'string' ? config.instructions : undefined;
  const beside = path.join(path.dirname(filePath), 'instructions.md');
  if (!instructions && fs.existsSync(beside)) instructions = readText(beside).trim();

  const entries = Array.isArray(config.skills) ? (config.skills as SkillEntry[]) : [];
  return {
    name: typeof config.name === 'string' && config.name ? config.name : 'agent',
    description: typeof config.description === 'string' ? config.description : undefined,
    instructions,
    model: typeof config.model === 'string' ? config.model : undefined,
    skills: entries.map((e) => (typeof e === 'string' ? e : Object.keys(e)[0] ?? '')).filter(Boolean),
    skillEntries: entries,
    intents: Array.isArray(config.intents) ? (config.intents as unknown[]).map(String) : undefined,
    source: filePath,
  };
}

/**
 * The project at `target` (a directory, an `AGENT*.md`, or an `agent.json`),
 * or `null` when there is none. A file that exists and cannot be read or
 * parsed throws `AgentProjectError`: absent and broken are not the same thing.
 */
export function loadAgentProject(target: string): AgentProject | null {
  const resolved = path.resolve(target);
  let stat: fs.Stats;
  try {
    stat = fs.statSync(resolved);
  } catch {
    return null;
  }

  if (stat.isFile()) {
    return resolved.toLowerCase().endsWith('.md') ? fromMarkdown(resolved) : fromJson(resolved);
  }

  const markdown = path.join(resolved, 'AGENT.md');
  if (fs.existsSync(markdown)) return fromMarkdown(markdown);
  const json = path.join(resolved, 'agent.json');
  if (fs.existsSync(json)) return fromJson(json);
  return null;
}

/**
 * What `POST /api/agents` accepts, built from a project.
 *
 * `skills` is the one shape that changes: files write a LIST (`- openai`,
 * `- mcp: {...}`) and the portal's schema is a mapping (`z.record`), so a bare
 * name becomes an empty config. The same rule as the Python CLI's
 * `commands/deploy.py:_payload`, so both CLIs publish a file the same way.
 * Keys the portal does not model (the scaffold's old `template`) are not sent.
 */
export function toPlatformPayload(project: AgentProject): Record<string, unknown> {
  const skills: Record<string, unknown> = {};
  for (const entry of project.skillEntries) {
    if (typeof entry === 'string') skills[entry] = {};
    else {
      const [key] = Object.keys(entry);
      if (key) skills[key] = entry[key] ?? {};
    }
  }
  const payload: Record<string, unknown> = { name: project.name };
  if (project.description) payload.description = project.description;
  if (project.instructions) payload.instructions = project.instructions;
  if (project.model) payload.model = project.model;
  if (project.intents?.length) payload.intents = project.intents;
  if (Object.keys(skills).length) payload.skills = skills;
  return payload;
}
