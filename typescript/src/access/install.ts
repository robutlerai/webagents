/**
 * Putting an agent file's `access:` block into effect (ADR-0045).
 *
 * The loaders (`cli/app.ts` for the chat, `cli/serve-action.ts` for `serve`)
 * call `accessSkillFor` with the block and the agent file, add the skill it
 * returns, and after the agent is built call `applyAccessTools`, which gives
 * every tool `access.tools` names the scopes of the groups that name it. The
 * Python twin is `python/webagents/access/install.py`.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { AccessConfigError, parseAccess, type AccessPolicy } from './policy';
import { AccessSkill } from '../skills/access/skill';
import type { ISkill } from '../core/types';

/** Each group's instructions file, read now, relative to the agent file. */
export function instructionTexts(policy: AccessPolicy, agentDir: string): Map<string, string> {
  const texts = new Map<string, string>();
  for (const [group, relative] of policy.instructions) {
    const file = path.resolve(agentDir, relative);
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      throw new AccessConfigError(`access.instructions.${group}: ${relative} was not found next to the agent file.`);
    }
    texts.set(group, fs.readFileSync(file, 'utf8').trim());
  }
  return texts;
}

/** The skill and policy for a block, or an `AccessConfigError`. */
export function accessSkillFor(raw: unknown, agentFile: string | undefined): { skill: AccessSkill; policy: AccessPolicy } {
  const policy = parseAccess(raw);
  const agentDir = agentFile ? path.dirname(agentFile) : process.cwd();
  return { skill: new AccessSkill({ policy, instructionTexts: instructionTexts(policy, agentDir) }), policy };
}

/**
 * Give every tool the block names (by its skill's `skills:` name, or its own
 * name) exactly the scopes of the groups naming it. A name that is neither is
 * refused, with the same sentence as Python.
 */
export function applyAccessTools(policy: AccessPolicy, skillsByName: ReadonlyMap<string, ISkill>): void {
  const grants = new Map<string, string[]>();
  for (const [group, names] of policy.tools) {
    for (const name of names) {
      const scopes = grants.get(name) ?? [];
      if (!scopes.includes(`group:${group}`)) scopes.push(`group:${group}`);
      grants.set(name, scopes);
    }
  }
  if (grants.size === 0) return;
  const toolsOf = (skill: ISkill): Array<{ name: string; scopes?: string[] }> =>
    ((skill as unknown as { tools?: Array<{ name: string; scopes?: string[] }> }).tools ?? []);
  const allTools = new Map<string, { name: string; scopes?: string[] }>();
  for (const skill of skillsByName.values()) for (const tool of toolsOf(skill)) allTools.set(tool.name, tool);
  for (const [group, names] of policy.tools) {
    for (const name of names) {
      if (!skillsByName.has(name) && !allTools.has(name)) {
        throw new AccessConfigError(`access.tools.${group}: "${name}" is not a skill in this agent file or one of its tools.`);
      }
    }
  }
  for (const [skillName, skill] of skillsByName) {
    for (const tool of toolsOf(skill)) {
      const scopes = grants.get(tool.name) ?? grants.get(skillName);
      if (scopes) tool.scopes = [...scopes];
    }
  }
}
