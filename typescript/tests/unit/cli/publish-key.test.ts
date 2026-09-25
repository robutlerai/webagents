/**
 * S-223: `webagents publish` printed the new agent's API key (2026-09-24).
 *
 * `POST /api/agents` answers `{agent, rawApiKey}`, and `publish` printed the
 * whole body, so the key went to stdout and from there into scrollback, screen
 * shares and CI logs. The platform returns it once. These run the real command
 * against a loopback stub that answers the route's real shape with a DUMMY key.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';

import { SecretStore } from '../../../src/skills/secrets/store';
import { tempDirs, TSX_CLI } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const CLI_SOURCE = path.resolve(__dirname, '../../../src/cli/index.ts');
const DUMMY_KEY = 'rok_DUMMY_NOT_A_REAL_KEY';

async function publishAgainstStub(): Promise<{ code: number | null; stdout: string; stderr: string; home: string }> {
  const server = http.createServer((req, res) => {
    req.resume();
    req.on('end', () => {
      res.statusCode = 201;
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ agent: { id: 'a-1', username: 'owner.probe' }, rawApiKey: DUMMY_KEY }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as { port: number }).port;

  const home = tempDir('wa-publish-home-');
  const project = tempDir('wa-publish-proj-');
  fs.writeFileSync(
    path.join(project, 'agent.json'),
    JSON.stringify({ name: 'probe', description: 'p', instructions: 'p' }),
  );

  try {
    // spawn, not spawnSync: the stub answers on THIS event loop.
    return await new Promise((resolve) => {
      const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, 'publish', project, '--yes'], {
        env: {
          ...process.env,
          HOME: home,
          WEBAGENTS_TOKEN: 'stub-token',
          ROBUTLER_API_URL: `http://127.0.0.1:${port}`,
          WEBAGENTS_PROFILE: '',
          // The file backend, so the test can prove the key was STORED without
          // ever touching the developer's real keychain.
          WEBAGENTS_SECRETS_BACKEND: 'file',
        },
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (d) => (stdout += d));
      child.stderr.on('data', (d) => (stderr += d));
      child.on('close', (code) => resolve({ code, stdout, stderr, home }));
    });
  } finally {
    server.close();
  }
}

describe('S-223: publish never prints the agent key', () => {
  it('keeps the key off stdout and stderr, and stores it instead', async () => {
    const out = await publishAgainstStub();

    expect(out.code).toBe(0);
    // THE BUG: the key appeared in stdout.
    expect(out.stdout).not.toContain(DUMMY_KEY);
    expect(out.stderr).not.toContain(DUMMY_KEY);

    // It says who was published and where the key went...
    expect(out.stdout).toContain('owner.probe');
    expect(out.stdout).toContain('AGENT_KEY_OWNER_PROBE');

    // ...and the key is really there, under the name the Python CLI uses, so
    // `webagents secrets get` finds it.
    const secrets = path.join(out.home, '.webagents', 'secrets', 'providers.json');
    const stored = JSON.parse(fs.readFileSync(secrets, 'utf-8')) as Record<string, string>;
    expect(stored.AGENT_KEY_OWNER_PROBE).toBe(DUMMY_KEY);
  }, 60_000);

  it('can be read back with THIS CLI, redacted unless --show (2026-09-24)', async () => {
    // `publish` used to point a TypeScript-only developer at the Python CLI to
    // read the key it had just stored.
    const out = await publishAgainstStub();
    expect(out.code).toBe(0);
    expect(out.stdout).not.toContain('Python CLI');

    const run = (args: string[]) =>
      spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, 'secrets', 'get', ...args], {
        env: { ...process.env, HOME: out.home, WEBAGENTS_PROFILE: '', WEBAGENTS_SECRETS_BACKEND: 'file' },
        encoding: 'utf-8',
        timeout: 60_000,
      });

    const redacted = run(['AGENT_KEY_OWNER_PROBE']);
    expect(redacted.status).toBe(0);
    expect(redacted.stdout).toContain('AGENT_KEY_OWNER_PROBE is stored');
    expect(redacted.stdout + redacted.stderr).not.toContain(DUMMY_KEY);

    const shown = run(['AGENT_KEY_OWNER_PROBE', '--show']);
    expect(shown.status).toBe(0);
    expect(shown.stdout).toBe(`${DUMMY_KEY}\n`);

    const absent = run(['AGENT_KEY_NOBODY']);
    expect(absent.status).toBe(1);
    expect(absent.stderr).toContain('AGENT_KEY_NOBODY is not stored');
  }, 90_000);
});

describe('the keystore-mode index is maintained by the store itself', () => {
  function fakeKeyring() {
    const items = new Map<string, string>();
    return {
      Entry: class {
        constructor(private service: string, private name: string) {}
        getPassword() {
          return items.get(`${this.service}/${this.name}`) ?? null;
        }
        setPassword(value: string) {
          items.set(`${this.service}/${this.name}`, value);
        }
        deletePassword() {
          return items.delete(`${this.service}/${this.name}`);
        }
      },
    };
  }

  function keystoreStore(dir: string) {
    return new SecretStore({
      namespace: 'providers',
      keyring: fakeKeyring() as unknown as ConstructorParameters<typeof SecretStore>[0]['keyring'],
      unavailableReason: 'tests.fakeKeyring',
      filePath: path.join(dir, 'providers.json'),
      quiet: true,
    });
  }

  it('lists a key written through set, without any caller calling noteIndex', async () => {
    const dir = tempDir('wa-index-');
    const store = keystoreStore(dir);

    await store.set('AGENT_KEY_ONE', DUMMY_KEY);
    const { names, complete } = await store.list();
    expect(names).toEqual(['AGENT_KEY_ONE']);
    // A keychain cannot be enumerated, and the store says so.
    expect(complete).toBe(false);
  });

  it('drops the name again on delete', async () => {
    const dir = tempDir('wa-index-');
    const store = keystoreStore(dir);

    await store.set('AGENT_KEY_ONE', DUMMY_KEY);
    await store.delete('AGENT_KEY_ONE');
    expect((await store.list()).names).toEqual([]);
  });

  it('writes no value into the index', async () => {
    const dir = tempDir('wa-index-');
    await keystoreStore(dir).set('AGENT_KEY_ONE', DUMMY_KEY);
    const index = fs.readFileSync(path.join(dir, 'providers.index.json'), 'utf-8');
    expect(index).not.toContain(DUMMY_KEY);
  });
});
