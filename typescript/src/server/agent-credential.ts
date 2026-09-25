/**
 * The agent's own platform credential, found rather than configured (2026-09-24).
 *
 * The twin of `python/webagents/utils/agent_credential.py`; keep the two in step.
 *
 * WHY. One credential had four names, and a developer had to copy it by hand:
 * `webagents publish` stored the agent's key in the keystore, and the docs then
 * told them to read it back out and export it as `WEBAGENTS_AGENT_TOKEN` (the
 * bridge, the heartbeat, the inbox), while discovery read `WEBAGENTS_API_KEY`
 * and payments read `ROBUTLER_API_KEY`, the name registration uses for the
 * OWNER's key. The CLI already knows everything needed to find it: the
 * directory is linked to its platform agent (`link.agentName` in
 * `./.webagents/config.json`) and the key is stored under that name.
 *
 * THE ORDER:
 *   1. what the code passes explicitly;
 *   2. `WEBAGENTS_AGENT_TOKEN` (and, for callers that accept them, the older
 *      names): the environment is how containers and CI receive secrets;
 *   3. the key `publish` (or the Python `deploy`) stored, when this directory is
 *      linked to THIS agent: the link names one agent, and a second agent file
 *      in the same directory must not authenticate as the first.
 * Nothing here signs. An agent served at a public https URL needs no credential
 * at all where the caller can sign with its identity, and that stays the
 * preferred path.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

/** The one environment name for the agent's own credential. */
export const AGENT_TOKEN_ENV = 'WEBAGENTS_AGENT_TOKEN';

export interface AgentCredential {
  token: string;
  /** `explicit`, `env:<NAME>` or `keystore:<NAME>`: for messages, never the value. */
  source: string;
}

export interface ResolveAgentCredentialOptions {
  /** A credential passed in code. Wins. */
  explicit?: string;
  /**
   * Older environment names this caller still accepts, after
   * `WEBAGENTS_AGENT_TOKEN`. Leave empty where the key must be AGENT-BOUND
   * (the bridge, the heartbeat): those names have long held owners' keys.
   */
  legacyEnv?: string[];
  /** Where to look for the directory link. Defaults to the working directory. */
  cwd?: string;
}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] || undefined : undefined;
}

/** The keystore name `publish`/`deploy` store an agent's key under. */
export function agentKeyName(platformName: string): string {
  return `AGENT_KEY_${platformName.toUpperCase().replace(/[.-]/g, '_')}`;
}

/** `link.agentName` from `./.webagents/config.json`, the project file only. */
export function linkedAgentName(cwd: string = process.cwd()): string | undefined {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(cwd, '.webagents', 'config.json'), 'utf-8'));
    const name = data && typeof data === 'object' ? data['link.agentName'] : undefined;
    return name ? String(name) : undefined;
  } catch {
    return undefined;
  }
}

/** Whether the directory's link names THIS agent (`alice.helper` matches `helper`). */
export function linkMatches(linked: string, agentName: string | undefined): boolean {
  if (!agentName) return false;
  return linked === agentName || linked.endsWith(`.${agentName}`);
}

async function storedKey(keyName: string): Promise<string | undefined> {
  try {
    const { openSecretStore } = await import('../skills/secrets/store.js');
    const { globalDir, profileName, scopedNamespace } = await import('../cli/config-store.js');
    const profile = profileName();
    const store = await openSecretStore({
      namespace: scopedNamespace('providers', profile),
      secretsDir: path.join(globalDir(profile), 'secrets'),
      quiet: true,
    });
    return (await store.get(keyName)) ?? undefined;
  } catch {
    // A keystore that cannot be read is the same as no stored key here: the
    // caller reports the missing credential with its own fix.
    return undefined;
  }
}

/** The agent's credential and where it came from, or `undefined`. See the file comment for the order. */
export async function resolveAgentCredential(
  agentName: string | undefined,
  options: ResolveAgentCredentialOptions = {},
): Promise<AgentCredential | undefined> {
  if (options.explicit) return { token: options.explicit, source: 'explicit' };
  for (const name of [AGENT_TOKEN_ENV, ...(options.legacyEnv ?? [])]) {
    const value = envVar(name);
    if (value) return { token: value, source: `env:${name}` };
  }
  const linked = linkedAgentName(options.cwd);
  if (linked && linkMatches(linked, agentName)) {
    const keyName = agentKeyName(linked);
    const value = await storedKey(keyName);
    if (value) return { token: value, source: `keystore:${keyName}` };
  }
  return undefined;
}
