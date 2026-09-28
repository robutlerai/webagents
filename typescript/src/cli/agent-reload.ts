/**
 * The reload engine's fingerprint and diff (2026-09-26, interactive-mode
 * spec 3.1): what "the loaded version" is, and what "changed since the chat
 * loaded it" reports.
 *
 * S-283 is the reason a reload is never automatic: the agent's files are
 * control files, so the chat reads them again only when the person types
 * `/reload` or `/agent edit`, and asks before it takes a version it did not
 * write whose policy-bearing parts changed. This module holds the pure parts;
 * the sentences are the shared fixture's (`chat_edits.json`). The Python chat
 * mirrors it in `cli/repl/agent_reload.py`.
 */

import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';

import type { ParsedAgent } from '../agents/index';
import { discoverSkills } from '../skills/skillmd/skillmd-loader';

/** The parts of a loaded agent whose change the chat reports and, for some, asks about. */
export interface LoadedAgent {
  /** SHA-256 of the agent file's bytes. */
  sha: string;
  name: string;
  description: string;
  model: string;
  /** The instruction body, kept to count changed lines. */
  instructions: string;
  /** `skills:` names, in the file's order. */
  skills: string[];
  /** `agent_skills:` folders. */
  agentSkills: string[];
  /** `access:`, `sandbox:`, `cron:` as canonical JSON, `''` when absent. */
  access: string;
  sandbox: string;
  cron: string;
  /** The SKILL.md skill names the folder resolves, sorted. */
  skillmd: string[];
  /** Digests of `.webagents/skills.lock` and `mcp.json`, `''` when absent. */
  lock: string;
  mcp: string;
}

function digest(file: string): string {
  try {
    return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  } catch {
    return '';
  }
}

function canonical(value: unknown): string {
  return value === undefined || value === null ? '' : JSON.stringify(value);
}

/** The fingerprint of the agent `file` declares, in the folder `folder` (file comment). */
export function loadedAgentOf(file: string, folder: string, parsed: ParsedAgent): LoadedAgent {
  const skillmd = discoverSkills(folder, parsed.agentSkills ?? []).skills.map((s) => s.name).sort();
  return {
    sha: digest(file),
    name: parsed.name,
    description: parsed.description,
    model: parsed.model ?? '',
    instructions: parsed.instructions,
    skills: [...parsed.skills],
    agentSkills: [...(parsed.agentSkills ?? [])],
    access: canonical(parsed.access),
    sandbox: canonical(parsed.sandbox),
    cron: canonical(parsed.cron),
    skillmd,
    lock: digest(path.join(folder, '.webagents', 'skills.lock')),
    mcp: digest(path.join(folder, 'mcp.json')),
  };
}

/** The parts whose change makes the chat ASK before it takes a version it did not write. */
const POLICY_PARTS = new Set(['access', 'sandbox', 'cron', 'skills', 'agent_skills', 'SKILL.md skills', 'mcp.json']);

export interface ReloadDiff {
  /** One line per changed part, in the spec's order; empty when nothing changed. */
  parts: string[];
  /** Whether a policy-bearing part changed (the reload asks then). */
  policyChanged: boolean;
}

/** What changed between `before` and `after`, as the lines `/reload` lists (spec 3.1). */
export function reloadDiff(before: LoadedAgent, after: LoadedAgent): ReloadDiff {
  const parts: string[] = [];
  let policyChanged = false;
  const note = (label: string, changed: boolean) => {
    if (!changed) return;
    parts.push(label);
    if (POLICY_PARTS.has(label)) policyChanged = true;
  };
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  note('model', before.model !== after.model);
  note('skills', !same(before.skills, after.skills));
  note('agent_skills', !same(before.agentSkills, after.agentSkills));
  note('access', before.access !== after.access);
  note('sandbox', before.sandbox !== after.sandbox);
  note('cron', before.cron !== after.cron);
  note('description', before.description !== after.description);
  note('name', before.name !== after.name);
  if (before.instructions !== after.instructions) {
    parts.push(`instructions (${after.instructions.split('\n').length} lines)`);
  }
  if (!same(before.skillmd, after.skillmd)) {
    const added = after.skillmd.filter((n) => !before.skillmd.includes(n)).length;
    const gone = before.skillmd.filter((n) => !after.skillmd.includes(n)).length;
    parts.push(`SKILL.md skills: +${added} -${gone}`);
    policyChanged = true;
  }
  note('mcp.json', before.mcp !== after.mcp);
  return { parts, policyChanged };
}

/** Whether two fingerprints are the same version (the file bytes and everything the folder adds). */
export function sameVersion(a: LoadedAgent, b: LoadedAgent): boolean {
  return a.sha === b.sha && a.lock === b.lock && a.mcp === b.mcp && JSON.stringify(a.skillmd) === JSON.stringify(b.skillmd);
}

/** `Tools added: a, b.` / `Tools gone: c.`, empty parts omitted; the tool sets are sorted. */
export function toolChangeLines(before: readonly string[], after: readonly string[]): string[] {
  const added = after.filter((t) => !before.includes(t)).sort();
  const gone = before.filter((t) => !after.includes(t)).sort();
  const lines: string[] = [];
  if (added.length) lines.push(`Tools added: ${added.join(', ')}.`);
  if (gone.length) lines.push(`Tools gone: ${gone.join(', ')}.`);
  return lines;
}
