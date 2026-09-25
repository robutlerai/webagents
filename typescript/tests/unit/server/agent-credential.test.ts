/**
 * The agent's own credential is found, not exported by hand (2026-09-24).
 *
 * `webagents publish` stored the agent's key in the keystore, and then the docs
 * told the developer to read it back and export it as WEBAGENTS_AGENT_TOKEN,
 * while discovery read WEBAGENTS_API_KEY and payments ROBUTLER_API_KEY (the
 * name registration uses for the OWNER's key). One resolver now: explicit,
 * then the environment, then the stored key for the agent this directory is
 * linked to.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import {
  agentKeyName,
  linkMatches,
  resolveAgentCredential,
} from '../../../src/server/agent-credential';
import { openSecretStore } from '../../../src/skills/secrets/store';
import { PaymentSkill } from '../../../src/skills/payments/skill';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const SAVED = { ...process.env };

function restoreEnv() {
  for (const key of Object.keys(process.env)) if (!(key in SAVED)) delete process.env[key];
  for (const [key, value] of Object.entries(SAVED)) process.env[key] = value;
}

let home: string;
let project: string;

async function storeKey(name: string, value: string) {
  const store = await openSecretStore({
    namespace: 'providers',
    secretsDir: path.join(home, '.webagents', 'secrets'),
    quiet: true,
    backend: 'file',
  });
  await store.set(name, value);
}

function link(dir: string, platformName: string) {
  fs.mkdirSync(path.join(dir, '.webagents'), { recursive: true });
  fs.writeFileSync(
    path.join(dir, '.webagents', 'config.json'),
    JSON.stringify({ 'link.agentId': 'a-1', 'link.agentName': platformName }),
  );
}

beforeEach(() => {
  home = tempDir('wa-cred-home-');
  project = tempDir('wa-cred-proj-');
  process.env.HOME = home;
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  delete process.env.WEBAGENTS_PROFILE;
  delete process.env.WEBAGENTS_AGENT_TOKEN;
  delete process.env.WEBAGENTS_API_KEY;
  delete process.env.ROBUTLER_API_KEY;
});
afterEach(restoreEnv);

describe('resolveAgentCredential', () => {
  it('names the stored key the way publish and deploy do', () => {
    expect(agentKeyName('alice.my-agent')).toBe('AGENT_KEY_ALICE_MY_AGENT');
  });

  it('finds the key publish stored for the linked agent, with nothing exported', async () => {
    link(project, 'alice.helper');
    await storeKey('AGENT_KEY_ALICE_HELPER', 'rok_STORED');
    // THE FRICTION: this took `secrets get ... --show` and an export.
    expect(await resolveAgentCredential('helper', { cwd: project })).toEqual({
      token: 'rok_STORED',
      source: 'keystore:AGENT_KEY_ALICE_HELPER',
    });
  });

  it('does not let a second agent in the directory authenticate as the linked one', async () => {
    link(project, 'alice.helper');
    await storeKey('AGENT_KEY_ALICE_HELPER', 'rok_STORED');
    expect(linkMatches('alice.helper', 'planner')).toBe(false);
    expect(await resolveAgentCredential('planner', { cwd: project })).toBeUndefined();
  });

  it('lets the environment override the store, and explicit override both', async () => {
    link(project, 'alice.helper');
    await storeKey('AGENT_KEY_ALICE_HELPER', 'rok_STORED');
    process.env.WEBAGENTS_AGENT_TOKEN = 'from-env';
    expect((await resolveAgentCredential('helper', { cwd: project }))?.source).toBe('env:WEBAGENTS_AGENT_TOKEN');
    expect((await resolveAgentCredential('helper', { cwd: project, explicit: 'in-code' }))?.token).toBe('in-code');
  });

  it('reads older names only for callers that accept them', async () => {
    process.env.WEBAGENTS_API_KEY = 'owner-key';
    // The bridge and the heartbeat need an agent-bound key; this name has long
    // held owners' keys, which the bridge refuses.
    expect(await resolveAgentCredential('helper', { cwd: project })).toBeUndefined();
    expect((await resolveAgentCredential('helper', { cwd: project, legacyEnv: ['WEBAGENTS_API_KEY'] }))?.token).toBe(
      'owner-key',
    );
  });
});

describe('PaymentSkill', () => {
  it('uses the stored key for its agent when none is configured', async () => {
    link(project, 'alice.helper');
    await storeKey('AGENT_KEY_ALICE_HELPER', 'rok_STORED');
    const cwd = process.cwd();
    process.chdir(project);
    try {
      const skill = new PaymentSkill({});
      skill.setAgent({ name: 'helper' });
      await skill.initialize();
      expect((skill as unknown as { apiKey?: string }).apiKey).toBe('rok_STORED');
    } finally {
      process.chdir(cwd);
    }
  });
});
