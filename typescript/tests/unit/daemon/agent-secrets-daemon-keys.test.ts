/**
 * The daemon keeps its agents' keys where `serve()` keeps every key (S-309,
 * 2026-09-27, the agent-secrets lane): a key an earlier daemon left in the
 * agent folder's `.webagents/keys` moves into the store on load with its
 * thumbprint unchanged, a key that git ever tracked is said to need
 * rotation, two folders sharing an agent name are told they share a key, a
 * conflicting or unusable legacy file is never moved. The paths and lines
 * are the shared fixture's (`python/tests/fixtures/daemon/signedhooks.json`,
 * `store`); the Python daemon runs the same in
 * `tests/daemon/test_agent_secrets_daemon_keys.py`.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgentKeyConflictError, AgentKeyFileError, loadOrCreateAgentIdentity } from '../../../src/crypto/identity-store';
import {
  DAEMON_KEY_LINES,
  agentKeyOriginFileName,
  daemonAgentIdentity,
  daemonKeyLine,
  daemonKeysDir,
  legacyDaemonKeysDir,
} from '../../../src/daemon/agent-identity';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/daemon/signedhooks.json'), 'utf8')) as {
  store: {
    env: string;
    default: string;
    legacy_dir: string;
    origin_file: string;
    origin_keys: string[];
    moved_line: string;
    tracked_line: string;
    shared_line: string;
  };
  key_file: string;
  agent_file: string;
};

const tempDir = tempDirs();
const PUBLIC_URL = 'https://agent.example';
const ISOLATED = ['HOME', 'WEBAGENTS_KEYS_DIR', 'WEBAGENTS_PUBLIC_URL'];
const saved: Record<string, string | undefined> = {};
const said: string[] = [];
const originalLog = console.log;

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-agent-secrets-home-');
  said.length = 0;
  console.log = (...args: unknown[]) => said.push(args.map(String).join(' '));
});

afterEach(() => {
  console.log = originalLog;
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
});

function project(name = 'reporter'): { dir: string; filePath: string } {
  const dir = tempDir('wa-agent-secrets-project-');
  const filePath = path.join(dir, 'AGENT.md');
  writeFileSync(filePath, FIXTURE.agent_file.replace('{url}', 'https://hook.example/hook').replace('name: reporter', `name: ${name}`));
  return { dir, filePath };
}

function storeFile(agent = 'reporter'): string {
  return path.join(process.env.HOME!, FIXTURE.store.default, FIXTURE.key_file.replace('{agent}', agent));
}

function legacyFile(dir: string, agent = 'reporter'): string {
  return path.join(dir, FIXTURE.store.legacy_dir, FIXTURE.key_file.replace('{agent}', agent));
}

/** A key at the old place, minted by the store itself so its bytes are exactly what a daemon wrote. */
async function mintLegacy(dir: string, agent = 'reporter'): Promise<string> {
  const identity = await loadOrCreateAgentIdentity(agent, { issuer: `${PUBLIC_URL}/agents/${agent}`, keysDir: legacyDaemonKeysDir(dir) });
  return identity.kid;
}

function gitAvailable(): boolean {
  try {
    execFileSync('git', ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

describe('the words and the places are the fixture’s', () => {
  it('lines, store and sidecar', () => {
    expect(DAEMON_KEY_LINES.moved).toBe(FIXTURE.store.moved_line);
    expect(DAEMON_KEY_LINES.tracked).toBe(FIXTURE.store.tracked_line);
    expect(DAEMON_KEY_LINES.shared).toBe(FIXTURE.store.shared_line);
    expect(daemonKeysDir()).toBe(path.join(process.env.HOME!, FIXTURE.store.default));
    process.env[FIXTURE.store.env] = '/mnt/agent-keys';
    expect(daemonKeysDir()).toBe('/mnt/agent-keys');
    expect(legacyDaemonKeysDir('/x')).toBe(path.join('/x', FIXTURE.store.legacy_dir));
    expect(agentKeyOriginFileName('reporter')).toBe(FIXTURE.store.origin_file.replace('{agent}', 'reporter'));
  });
});

describe('a key an earlier daemon left in the agent folder', () => {
  it('moves into the store on load, same thumbprint, and the folder is left without a key', async () => {
    const { dir, filePath } = project();
    const kid = await mintLegacy(dir);
    const legacy = legacyFile(dir);
    expect(existsSync(legacy)).toBe(true);

    const identity = await daemonAgentIdentity({ name: 'reporter', filePath }, PUBLIC_URL);
    expect(identity.kid).toBe(kid);
    expect(existsSync(storeFile())).toBe(true);
    expect(existsSync(legacy)).toBe(false);
    expect(existsSync(path.join(dir, FIXTURE.store.legacy_dir))).toBe(false);
    expect(said).toContain(daemonKeyLine('moved', { agent: 'reporter', legacy, store: storeFile() }));
    expect(said.some((line) => line.includes('rotate the key'))).toBe(false);

    // The sidecar names this folder; a second load from it says nothing more.
    const origin = JSON.parse(readFileSync(path.join(daemonKeysDir(), agentKeyOriginFileName('reporter')), 'utf8')) as Record<string, unknown>;
    expect(Object.keys(origin).sort()).toEqual([...FIXTURE.store.origin_keys].sort());
    expect(origin.folder).toBe(realpathSync(dir));
    said.length = 0;
    expect((await daemonAgentIdentity({ name: 'reporter', filePath }, PUBLIC_URL)).kid).toBe(kid);
    expect(said).toEqual([]);
  });

  it('a key git ever tracked is said to need rotation', async () => {
    if (!gitAvailable()) return;
    const { dir, filePath } = project();
    await mintLegacy(dir);
    const legacy = legacyFile(dir);
    const git = (args: string[]) =>
      execFileSync('git', ['-C', dir, ...args], {
        stdio: 'ignore',
        env: { ...process.env, GIT_AUTHOR_NAME: 't', GIT_AUTHOR_EMAIL: 't@example', GIT_COMMITTER_NAME: 't', GIT_COMMITTER_EMAIL: 't@example' },
      });
    git(['init', '-q']);
    git(['add', '-f', path.relative(dir, legacy)]);
    git(['commit', '-q', '-m', 'oops']);

    await daemonAgentIdentity({ name: 'reporter', filePath }, PUBLIC_URL);
    expect(said).toContain(daemonKeyLine('tracked', { legacy, folder: dir, store: storeFile() }));
  });

  it('a legacy file that is not a key is said and never moved', async () => {
    const { dir, filePath } = project();
    const legacy = legacyFile(dir);
    mkdirSync(path.dirname(legacy), { recursive: true });
    writeFileSync(legacy, 'not a key');
    await expect(daemonAgentIdentity({ name: 'reporter', filePath }, PUBLIC_URL)).rejects.toBeInstanceOf(AgentKeyFileError);
    expect(readFileSync(legacy, 'utf8')).toBe('not a key');
    expect(existsSync(storeFile())).toBe(false);
  });

  it('a different key already in the store is a conflict, and neither file moves', async () => {
    const { dir, filePath } = project();
    await mintLegacy(dir);
    await loadOrCreateAgentIdentity('reporter', { issuer: `${PUBLIC_URL}/agents/reporter` });
    const legacy = legacyFile(dir);
    const before = readFileSync(storeFile(), 'utf8');
    await expect(daemonAgentIdentity({ name: 'reporter', filePath }, PUBLIC_URL)).rejects.toBeInstanceOf(AgentKeyConflictError);
    expect(existsSync(legacy)).toBe(true);
    expect(readFileSync(storeFile(), 'utf8')).toBe(before);
  });

  it('the store pointed at the folder itself moves nothing', async () => {
    const { dir, filePath } = project();
    process.env[FIXTURE.store.env] = legacyDaemonKeysDir(dir);
    const kid = await mintLegacy(dir);
    expect((await daemonAgentIdentity({ name: 'reporter', filePath }, PUBLIC_URL)).kid).toBe(kid);
    expect(existsSync(legacyFile(dir))).toBe(true);
    expect(said.some((line) => line.includes('moved the signing key'))).toBe(false);
  });
});

describe('two folders with one agent name', () => {
  it('are told they share a key, once', async () => {
    const first = project();
    const second = project();
    const kid = (await daemonAgentIdentity({ name: 'reporter', filePath: first.filePath }, PUBLIC_URL)).kid;
    said.length = 0;
    expect((await daemonAgentIdentity({ name: 'reporter', filePath: second.filePath }, PUBLIC_URL)).kid).toBe(kid);
    const expected = daemonKeyLine('shared', {
      agent: 'reporter',
      store: storeFile(),
      origin: realpathSync(first.dir),
      folder: realpathSync(second.dir),
    });
    expect(said).toEqual([expected]);
    said.length = 0;
    await daemonAgentIdentity({ name: 'reporter', filePath: second.filePath }, PUBLIC_URL);
    expect(said).toEqual([]);
    // The sidecar still names the first folder.
    const origin = JSON.parse(readFileSync(path.join(daemonKeysDir(), agentKeyOriginFileName('reporter')), 'utf8')) as { folder: string };
    expect(origin.folder).toBe(realpathSync(first.dir));
  });
});
