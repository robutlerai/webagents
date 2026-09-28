/**
 * Keychain dialogs on macOS, the CLI's half (the keychain-ux lane,
 * 2026-09-27), the TypeScript twin of `python/tests/cli/test_keychain_ux_cli.py`:
 * what `logout`, `whoami`, `secrets list`, `doctor` and the chat's /status say,
 * that `serve` and the daemon forbid dialogs before they read anything, and
 * that this CLI never writes the Python CLI's `credentials.json`. The words
 * come from `python/tests/fixtures/keychain_ux/keychain_ux.json`.
 *
 * Every test runs with HOME pointed at a scratch folder and
 * WEBAGENTS_SECRETS_BACKEND=file; the keychain itself is a fake where needed.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as kx from '../../../src/skills/secrets/keychain-ux';
import { CLI_SOURCE, TSX_CLI, tempDirs } from '../../helpers/cli';
import { spawnSync } from 'node:child_process';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const FIXTURE: any = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/keychain_ux/keychain_ux.json'), 'utf8'));
const tempDir = tempDirs();
const DUMMY = 'dummy-value-not-a-real-secret';

const saved: Record<string, string | undefined> = {};
const VARS = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'ROBUTLER_API_URL', 'WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'WEBAGENTS_SECRETS_DIR'];
let home: string;

beforeEach(() => {
  for (const name of VARS) saved[name] = process.env[name];
  home = tempDir('wa-kx-cli-');
  process.env.HOME = home;
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  delete process.env.WEBAGENTS_PROFILE;
  delete process.env.WEBAGENTS_TOKEN;
  delete process.env.WEBAGENTS_SECRETS_DIR;
  kx.resetKeychainForTests({ interactive: false });
});

afterEach(() => {
  for (const name of VARS) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  kx.resetKeychainForTests();
  vi.doUnmock('../../../src/cli/credentials');
  vi.resetModules();
});

function writeRecord(runtimes: Record<string, unknown>): void {
  fs.mkdirSync(path.join(home, '.webagents'), { recursive: true });
  fs.writeFileSync(path.join(home, '.webagents', 'keychain.json'), JSON.stringify({ runtimes }));
}

describe('credentials.json', () => {
  it('this CLI never writes one: its whoami asks the platform, and the record it does write is 0600', () => {
    expect(FIXTURE.credentials_file.name).toBe('credentials.json');
    expect(FIXTURE.credentials_file.mode).toBe('0600');
    const env = { ...process.env, HOME: home, WEBAGENTS_SECRETS_BACKEND: 'file', ROBUTLER_API_URL: 'http://127.0.0.1:9' } as NodeJS.ProcessEnv;
    const cli = (args: string[], input = '') => spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, ...args], { cwd: home, env, encoding: 'utf-8', input, timeout: 60_000 });
    const set = cli(['secrets', 'set', 'OPENAI_API_KEY'], `${DUMMY}\n`);
    expect(set.status, set.stderr).toBe(0);
    expect(fs.existsSync(path.join(home, '.webagents', 'secrets', 'providers.json'))).toBe(true);
    expect(cli(['whoami']).status).toBe(1);
    const out = cli(['logout']);
    expect(out.status, out.stderr).toBe(0);
    expect(out.stdout).toContain('Signed out of 127.0.0.1:9.');
    const found: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else found.push(full);
      }
    };
    walk(home);
    expect(found.filter((file) => path.basename(file) === 'credentials.json')).toEqual([]);
  }, 180_000);
});

describe('logout', () => {
  it('names an old item only a dialog could remove', async () => {
    vi.doMock('../../../src/cli/credentials', async (original) => ({
      ...(await original<typeof import('../../../src/cli/credentials')>()),
      clearToken: async () => true,
      leftBehind: () => [{ item: 'webagents:cli', account: 'platform_token' }],
    }));
    const { logoutCommand } = await import('../../../src/cli/account');
    const said: string[] = [];
    expect(await logoutCommand((line) => said.push(line), (line) => said.push(`ERR ${line}`))).toBe(0);
    expect(said).toEqual(['Signed out of 127.0.0.1:9.', FIXTURE.left_behind.replace(/\{item\}/g, 'webagents:cli').replace('{account}', 'platform_token')]);
  });

  it('a token only a dialog could remove is not signed out', async () => {
    const sentence = FIXTURE.blocked.replace(/\{program\}/g, 'node').replace('{item}', 'webagents (TypeScript) cli').replace('{command}', 'webagents whoami');
    vi.doMock('../../../src/cli/credentials', async (original) => ({
      ...(await original<typeof import('../../../src/cli/credentials')>()),
      clearToken: async () => {
        const { KeychainDialogBlocked } = await import('../../../src/skills/secrets/keychain-ux');
        throw new KeychainDialogBlocked(sentence, 'webagents (TypeScript) cli', 'platform_token');
      },
    }));
    const { logoutCommand } = await import('../../../src/cli/account');
    const said: string[] = [];
    const errors: string[] = [];
    expect(await logoutCommand((line) => said.push(line), (line) => errors.push(line))).toBe(1);
    expect(said).toEqual([]);
    expect(errors).toEqual([sentence]);
  });
});

describe('whoami', () => {
  it('a sign-in this run could not read is the one sentence', async () => {
    const keychain = await import('../../../src/skills/secrets/keychain-ux');
    const said: string[] = [];
    keychain.resetKeychainForTests({ interactive: false, writer: (text) => said.push(text) });
    await keychain.noteBlocked('webagents (TypeScript) cli', 'platform_token');
    expect(said).toHaveLength(1);
    const { whoAmI } = await import('../../../src/cli/account');
    const result = await whoAmI();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe('keychain_needs_terminal');
    const me = await keychain.currentProgram();
    expect(result.message).toBe(FIXTURE.blocked.replace(/\{program\}/g, me.program).replace('{item}', 'webagents (TypeScript) cli').replace('{command}', 'webagents whoami'));
  });

  it('where the other CLI is signed in, it says each keeps its own', async () => {
    writeRecord({ python: { items: { 'webagents (Python) cli': { platform_token: { program: 'Python' } } } } });
    const { whoAmI } = await import('../../../src/cli/account');
    const result = await whoAmI();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.code).toBe('not_signed_in');
    expect(result.message).toBe(`Not signed in to 127.0.0.1:9. ${FIXTURE.other_runtime.signed_in.replace('{other}', 'Python CLI')}`);
  });

  it('settles what a background run could not read, in a terminal only', async () => {
    const keychain = await import('../../../src/skills/secrets/keychain-ux');
    const secrets = path.join(home, '.webagents', 'secrets');
    const record = new keychain.KeychainRecord(path.join(home, '.webagents', 'keychain.json'));
    await record.notePending('webagents (TypeScript) cli', 'platform_token', 'cli', secrets, false);
    const { settleKeychain } = await import('../../../src/cli/account');
    keychain.resetKeychainForTests({ interactive: false });
    expect(await settleKeychain()).toBe(0);
    expect(await record.pending()).toHaveLength(1);
    keychain.resetKeychainForTests({ interactive: true });
    // Nothing is stored under the file backend here: the entry has said its piece.
    expect(await settleKeychain()).toBe(0);
    expect(await record.pending()).toEqual([]);
  });

  it('the whoami command settles before it answers', () => {
    const source = fs.readFileSync(CLI_SOURCE, 'utf8');
    const start = source.indexOf(".command('whoami')");
    const body = source.slice(start, source.indexOf('program', start + 20));
    expect(body.indexOf('await settleKeychain()')).toBeGreaterThan(-1);
    expect(body.indexOf('await settleKeychain()')).toBeLessThan(body.indexOf('await whoAmI()'));
  });
});

describe('secrets list', () => {
  it("mentions the other CLI's keys where this one has none, in the keychain only", async () => {
    writeRecord({ python: { items: { 'webagents (Python) providers': { OPENAI_API_KEY: { program: 'Python' } } } } });
    const { otherRuntimeKeysLine } = await import('../../../src/cli/account');
    expect(await otherRuntimeKeysLine(async () => ({ status: () => ({ backend: 'keystore' }) }))).toBe(
      FIXTURE.other_runtime.keys.replace('{other}', 'Python CLI').replace('{command}', 'webagents secrets set NAME'),
    );
    // The file is shared by both CLIs, so nothing is said there.
    expect(await otherRuntimeKeysLine(async () => ({ status: () => ({ backend: 'file' }) }))).toBe('');
  });
});

describe('doctor and /status', () => {
  it("doctor's keychain line, in the file backend's words", async () => {
    const { keychainCheck } = await import('../../../src/cli/doctor');
    const line = await keychainCheck();
    expect(line.name).toBe('keychain');
    expect(line.status).toBe('ok');
    expect(line.detail).toBe(FIXTURE.doctor.file.replace('{dir}', '~/.webagents/secrets'));
  });

  it('the line never prints a value', async () => {
    process.env.WEBAGENTS_TOKEN = DUMMY;
    const { keychainCheck } = await import('../../../src/cli/doctor');
    const line = await keychainCheck();
    expect(line.detail).not.toContain(DUMMY);
    expect(line.fix ?? '').not.toContain(DUMMY);
  });

  it("the chat's /status Keychain row is the doctor detail", async () => {
    const { InteractiveREPL } = await import('../../../src/cli/app');
    const { keychainCheck } = await import('../../../src/cli/doctor');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const repl: any = new InteractiveREPL({});
    expect(FIXTURE.status_row).toBe('Keychain');
    expect(await repl.keychainRow()).toBe((await keychainCheck()).detail);
  });
});

describe('serve and the daemon are never asked', () => {
  it('serve forbids dialogs before the agent reads its keys', async () => {
    const keychain = await import('../../../src/skills/secrets/keychain-ux');
    keychain.resetKeychainForTests({ interactive: true });
    expect(keychain.dialogsAllowed()).toBe(true);
    const { serveAction } = await import('../../../src/cli/serve-action');
    const seen: boolean[] = [];
    await expect(
      serveAction('AGENT.md', { port: '1' } as never, {
        loadConfig: () => {
          seen.push(keychain.dialogsAllowed());
          throw new Error('stop here');
        },
      } as never),
    ).rejects.toThrow('stop here');
    expect(seen).toEqual([false]);
  });

  it('the daemon forbids dialogs before it builds an agent', async () => {
    const keychain = await import('../../../src/skills/secrets/keychain-ux');
    keychain.resetKeychainForTests({ interactive: true });
    const { WebAgentsDaemon } = await import('../../../src/daemon/server');
    const daemon = new WebAgentsDaemon({ port: 0, watch: false, cron: false });
    const seen: boolean[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (daemon as any).discover = async () => {
      seen.push(keychain.dialogsAllowed());
      throw new Error('stop here');
    };
    await expect(daemon.start()).rejects.toThrow('stop here');
    expect(seen).toEqual([false]);
  });
});
