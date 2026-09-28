/**
 * Keychain dialogs on macOS, on the REAL keychain (the keychain-ux lane,
 * 2026-09-27), the TypeScript twin of
 * `python/tests/skills/local/test_keychain_ux_macos_probe.py`.
 *
 * This SDK has no switch that turns a dialog into an error, so what is proved
 * here is the other half: with nobody to answer, an item another program made
 * is NOT READ AT ALL (the store decides from the record and an attribute-only
 * lookup first), and an item this node made is read, bounded, with no dialog.
 *
 * SAFETY, because this touches the person's login keychain:
 *   - Only items this file creates, named `webagents (TypeScript) keychain-ux-probe-<hex>`
 *     or `webagents:keychain-ux-probe-<hex>` for a fresh hex each test. Nothing
 *     else is read, listed or touched.
 *   - Every read this file causes is either refused before it is made, or of
 *     an item node itself made, which cannot ask. Every keychain call runs in a
 *     CHILD process under a timeout anyway.
 *   - Every item is removed in a `finally` by the program that made it
 *     (`/usr/bin/security` for the foreign ones, node for its own), and its
 *     absence is checked with an attribute-only lookup, which cannot ask.
 *   - The children run with the REAL home: with HOME pointed at a scratch
 *     folder macOS finds no default keychain, and an add would show its
 *     "keychain cannot be found" prompt (a suite hung on that on 2026-09-27).
 *     The file skips unless `security default-keychain` answers for it.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(HERE, '../../../src');
const SECURITY = '/usr/bin/security';
const REAL_HOME = os.userInfo().homedir;
const tempDir = tempDirs();

function childEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env, HOME: REAL_HOME } as Record<string, string | undefined>;
  delete env.WEBAGENTS_SECRETS_BACKEND;
  delete env.WEBAGENTS_PROFILE;
  return env as NodeJS.ProcessEnv;
}

function realKeychainHere(): boolean {
  if (process.platform !== 'darwin' || !fs.existsSync(SECURITY)) return false;
  try {
    createRequire(path.join(SRC, 'skills/secrets/store.ts'))('@napi-rs/keyring');
  } catch {
    return false;
  }
  return spawnSync(SECURITY, ['default-keychain', '-d', 'user'], { env: childEnv(), timeout: 10_000 }).status === 0;
}

const CHILD = `
import { createRequire } from 'node:module';
import * as path from 'node:path';
import * as kx from ${JSON.stringify(path.join(SRC, 'skills/secrets/keychain-ux.ts'))};
import { SecretStore } from ${JSON.stringify(path.join(SRC, 'skills/secrets/store.ts'))};

const [op, namespace, folder, interactive] = process.argv.slice(2);
const said: string[] = [];
kx.resetKeychainForTests({ interactive: interactive === '1', writer: (text) => said.push(text) });
const keyring = createRequire(${JSON.stringify(path.join(SRC, 'skills/secrets/store.ts'))})('@napi-rs/keyring');
const value = process.env.PROBE_VALUE ?? '';
const started = Date.now();
const out: Record<string, unknown> = { op };
if (op === 'get' || op === 'set' || op === 'delete') {
  const secrets = path.join(folder, 'secrets');
  // The macOS guard, explicitly: this module came from createRequire, not from
  // the store's own loader, and the store only guards the module it loaded.
  const keychain = new kx.KeychainAccess(keyring, namespace, secrets);
  if (!keychain.mac) throw new Error('no macOS guard: refusing to touch the keychain unguarded');
  const store = new SecretStore({ namespace, keyring, unavailableReason: 'probe', filePath: path.join(secrets, namespace + '.json'), quiet: true, keychain });
  if (op === 'get') {
    const got = await store.get('probe');
    out.found = got !== null;
    out.value_matches = got === value;
    out.blocked = kx.wasBlocked(kx.serviceName(namespace), 'probe') || kx.wasBlocked(kx.legacyServiceName(namespace), 'probe');
  } else if (op === 'set') {
    out.backend = await store.set('probe', value);
  } else {
    out.removed = await store.delete('probe');
    out.left_behind = store.leftBehind;
  }
} else if (op === 'raw-set') {
  new keyring.Entry(namespace, 'probe').setPassword(value);
} else if (op === 'raw-delete') {
  try { out.deleted = new keyring.Entry(namespace, 'probe').deletePassword(); } catch { out.deleted = false; }
}
out.said = said;
out.ms = Date.now() - started;
console.log(JSON.stringify(out));
`;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function child(op: string, namespace: string, folder: string, options: { interactive?: boolean; value?: string } = {}): any {
  const script = path.join(folder, 'keychain-ux-probe-child.mts');
  fs.writeFileSync(script, CHILD);
  const done = spawnSync(process.execPath, [TSX_CLI, script, op, namespace, folder, options.interactive ? '1' : '0'], {
    env: { ...childEnv(), PROBE_VALUE: options.value ?? '' },
    encoding: 'utf-8',
    timeout: 30_000,
  });
  if (done.error) throw new Error(`${op} on ${namespace} did not return within 30s: a keychain call waited, which is the bug (${done.error.message})`);
  expect(done.status, done.stderr).toBe(0);
  return JSON.parse(done.stdout.trim().split('\n').at(-1)!);
}

function exists(service: string): boolean {
  // Attribute-only lookup: no secret requested, so it cannot ask.
  const code = spawnSync(SECURITY, ['find-generic-password', '-s', service, '-a', 'probe'], { env: childEnv(), timeout: 10_000 }).status;
  expect([0, 44]).toContain(code);
  return code === 0;
}

function securityAdd(service: string, value: string): void {
  const done = spawnSync(SECURITY, ['add-generic-password', '-s', service, '-a', 'probe', '-l', `${service} (a test probe, safe to Deny)`, '-w', value], { env: childEnv(), timeout: 10_000 });
  expect(done.status, String(done.stderr)).toBe(0);
}

function securityDelete(service: string): void {
  spawnSync(SECURITY, ['delete-generic-password', '-s', service, '-a', 'probe'], { env: childEnv(), timeout: 10_000 });
}

function probe() {
  const tag = `keychain-ux-probe-${randomBytes(6).toString('hex')}`;
  return { namespace: tag, own: `webagents (TypeScript) ${tag}`, old: `webagents:${tag}`, value: `probe-${randomBytes(12).toString('hex')}` };
}

const onMac = realKeychainHere() ? describe : describe.skip;

onMac('the real macOS keychain', () => {
  it('an item another program made is not read at all with nobody to answer', () => {
    const p = probe();
    const folder = tempDir('wa-kx-probe-');
    securityAdd(p.own, p.value);
    try {
      const got = child('get', p.namespace, folder, { value: p.value });
      expect(got.found).toBe(false);
      expect(got.blocked).toBe(true);
      expect(got.said).toHaveLength(1);
      expect(got.said[0]).toMatch(/^macOS may ask before /);
      expect(got.ms).toBeLessThan(5000);
    } finally {
      securityDelete(p.own);
    }
    expect(exists(p.own)).toBe(false);
  }, 120_000);

  it('an item this node made is read with nobody to answer, bounded, with no dialog', () => {
    const p = probe();
    const folder = tempDir('wa-kx-probe-');
    try {
      expect(child('set', p.namespace, folder, { value: p.value }).backend).toBe('keystore');
      const got = child('get', p.namespace, folder, { value: p.value });
      expect(got.value_matches).toBe(true);
      expect(got.said).toEqual([]);
      const gone = child('delete', p.namespace, folder);
      expect(gone.removed).toBe(true);
      expect(gone.left_behind).toEqual([]);
    } finally {
      child('raw-delete', p.own, folder);
    }
    expect(exists(p.own)).toBe(false);
  }, 120_000);

  it('an old item another program made is never read, and removing names it', () => {
    const p = probe();
    const folder = tempDir('wa-kx-probe-');
    securityAdd(p.old, p.value);
    try {
      const got = child('get', p.namespace, folder, { value: p.value });
      expect(got.found).toBe(false);
      expect(got.blocked).toBe(true);
      expect(exists(p.own)).toBe(false);
      const gone = child('delete', p.namespace, folder);
      expect(gone.left_behind).toEqual([{ item: p.old, account: 'probe' }]);
      expect(exists(p.old)).toBe(true);
    } finally {
      securityDelete(p.old);
      child('raw-delete', p.own, folder);
    }
    expect(exists(p.old)).toBe(false);
    expect(exists(p.own)).toBe(false);
  }, 120_000);

  it('an old item this node made is copied in a terminal, after the explanation', () => {
    const p = probe();
    const folder = tempDir('wa-kx-probe-');
    try {
      child('raw-set', p.old, folder, { value: p.value });
      // A terminal: the read happens, after the four lines. Node made the old
      // item, so macOS has nothing to ask.
      const got = child('get', p.namespace, folder, { interactive: true, value: p.value });
      expect(got.value_matches).toBe(true);
      expect(got.said).toHaveLength(1);
      expect(got.said[0].split('\n')).toHaveLength(4);
      expect(exists(p.own)).toBe(true);
      expect(exists(p.old)).toBe(true);
      const gone = child('delete', p.namespace, folder, { interactive: true });
      expect(gone.removed).toBe(true);
      // This SDK never removes an old item: it names it.
      expect(gone.left_behind).toEqual([{ item: p.old, account: 'probe' }]);
    } finally {
      child('raw-delete', p.old, folder);
      child('raw-delete', p.own, folder);
    }
    expect(exists(p.old)).toBe(false);
    expect(exists(p.own)).toBe(false);
  }, 120_000);
});

it('the probe file skips cleanly where there is no real keychain', () => {
  // Always runs, so this file is never an empty suite.
  expect(typeof realKeychainHere()).toBe('boolean');
});
