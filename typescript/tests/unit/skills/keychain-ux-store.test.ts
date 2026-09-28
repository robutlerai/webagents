/**
 * Keychain dialogs on macOS, the store's half (the keychain-ux lane,
 * 2026-09-27), the TypeScript twin of
 * `python/tests/skills/local/test_keychain_ux_store.py`.
 *
 * A FAKE macOS keychain drives these. An item trusts the program that made it;
 * any other program gets a dialog. Unlike the Python SDK, this one has no switch
 * that turns a dialog into an error (`@napi-rs/keyring` offers none), so the
 * fake records every dialog, and the tests assert that none is ever raised in a
 * run with nobody to answer: the store must decide from the record and an
 * attribute-only lookup BEFORE it reads. The real keychain is exercised only by
 * `keychain-ux-macos-probe.test.ts`, with probe items of its own. Every name and
 * sentence comes from `python/tests/fixtures/keychain_ux/keychain_ux.json`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as kx from '../../../src/skills/secrets/keychain-ux';
import { SecretStore, openSecretStore, serviceKey } from '../../../src/skills/secrets/store';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const FIXTURE: any = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/keychain_ux/keychain_ux.json'), 'utf8'));
const tempDir = tempDirs();

// Obvious dummies. Nothing here is or resembles a real credential.
const DUMMY = 'dummy-value-not-a-real-secret';
const OTHER_DUMMY = 'another-dummy-value-not-a-real-secret';
const OWN = 'webagents (TypeScript) cli';
const OLD = 'webagents:cli';

interface Item {
  value: string;
  creator: string;
  trusted: Set<string>;
}

/** A fake macOS keychain, as `@napi-rs/keyring` meets it. */
class MacWorld {
  items = new Map<string, Item>();
  program = 'node 24.7.0';
  answer: 'always' | 'deny' = 'always';
  /** An `AsyncEntry` call that needs a dialog never answers (a dialog nobody answers). */
  hang = true;
  events: Array<[string, string, string?, string?]> = [];
  reads: Array<[string, string]> = [];

  key(service: string, account: string): string {
    return `${service}\u0000${account}`;
  }

  add(service: string, account: string, value: string, creator = this.program): void {
    this.items.set(this.key(service, account), { value, creator, trusted: new Set() });
  }

  get(service: string, account: string): Item | undefined {
    return this.items.get(this.key(service, account));
  }

  dialogs() {
    return this.events.filter((event) => event[0] === 'dialog');
  }

  said() {
    return this.events.filter((event) => event[0] === 'say').map((event) => event[1]);
  }

  private trusted(item: Item): boolean {
    return item.creator === this.program || item.trusted.has(this.program);
  }

  /** macOS asks (always: this SDK has no way to stop it). */
  private ask(op: string, service: string, account: string, item: Item): void {
    this.events.push(['dialog', op, service, account]);
    if (this.answer === 'deny') throw new Error('User interaction is not allowed.');
    item.trusted.add(this.program);
  }

  readItem(service: string, account: string): string | null {
    this.reads.push([service, account]);
    const item = this.get(service, account);
    if (!item) return null;
    if (!this.trusted(item)) this.ask('read', service, account, item);
    return item.value;
  }

  /** The Rust keyring finds (reads) an existing item before it changes it. */
  writeItem(service: string, account: string, value: string): void {
    const item = this.get(service, account);
    if (item) {
      if (!this.trusted(item)) this.ask('read', service, account, item);
      item.value = value;
      return;
    }
    this.add(service, account, value);
  }

  deleteItem(service: string, account: string): boolean {
    const item = this.get(service, account);
    if (!item) return false;
    if (!this.trusted(item)) this.ask('read', service, account, item);
    this.items.delete(this.key(service, account));
    return true;
  }

  wouldAsk(service: string, account: string): boolean {
    const item = this.get(service, account);
    return Boolean(item && !this.trusted(item));
  }

  keyring(): kx.KeyringModuleLike {
    const world = this;
    class Entry {
      constructor(
        private service: string,
        private account: string,
      ) {}
      getPassword() {
        return world.readItem(this.service, this.account);
      }
      setPassword(value: string) {
        world.writeItem(this.service, this.account, value);
      }
      deletePassword() {
        return world.deleteItem(this.service, this.account);
      }
    }
    class AsyncEntry {
      constructor(
        private service: string,
        private account: string,
      ) {}
      private bounded<T>(signal: AbortSignal | null | undefined, run: () => T): Promise<T> {
        if (world.hang && world.wouldAsk(this.service, this.account)) {
          world.events.push(['hung', this.service, this.account]);
          return new Promise((_, reject) => signal?.addEventListener('abort', () => reject(Object.assign(new Error('The operation was aborted'), { name: 'AbortError' }))));
        }
        return Promise.resolve().then(run);
      }
      getPassword(signal?: AbortSignal | null) {
        return this.bounded(signal, () => world.readItem(this.service, this.account));
      }
      setPassword(value: string, signal?: AbortSignal | null) {
        return this.bounded(signal, () => world.writeItem(this.service, this.account, value));
      }
      deletePassword(signal?: AbortSignal | null) {
        return this.bounded(signal, () => world.deleteItem(this.service, this.account));
      }
    }
    return { Entry, AsyncEntry };
  }

  mac(): kx.MacKeychain {
    const world = this;
    return new (class extends kx.MacKeychain {
      override async exists(service: string, account?: string): Promise<boolean | null> {
        for (const key of world.items.keys()) {
          const [s, a] = key.split('\u0000');
          if (s === service && (account === undefined || a === account)) return true;
        }
        return false;
      }
    })();
  }
}

function run(world: MacWorld, interactive: boolean): void {
  kx.resetKeychainForTests({ interactive, writer: (text) => world.events.push(['say', text]) });
}

function macStore(dir: string, world: MacWorld, namespace = 'cli', timeoutMs = 50): SecretStore {
  const secrets = path.join(dir, 'secrets');
  const keyring = world.keyring();
  const keychain = new kx.KeychainAccess(keyring, namespace, secrets, { mac: world.mac(), timeoutMs });
  return new SecretStore({ namespace, keyring: keyring as never, unavailableReason: 'fake macOS', filePath: path.join(secrets, `${namespace}.json`), quiet: true, keychain });
}

function record(dir: string): kx.KeychainRecord {
  return new kx.KeychainRecord(path.join(dir, 'keychain.json'));
}

/** Make the record say this node last used the item. */
async function recordThisNode(dir: string, service: string, account: string): Promise<void> {
  await record(dir).noteUsed(service, account);
}

/** Make the record say an older node last used the item. */
async function recordOlderNode(dir: string, service: string, account: string): Promise<void> {
  const rec = record(dir);
  await rec.noteUsed(service, account);
  const data = await rec.read();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (data as any).runtimes.typescript.items[service][account].version = '24.6.0';
  fs.writeFileSync(path.join(dir, 'keychain.json'), JSON.stringify(data));
}

beforeEach(() => {
  delete process.env.WEBAGENTS_PROFILE;
  kx.resetKeychainForTests({ interactive: false });
});
afterEach(() => kx.resetKeychainForTests());

describe('names and words, from the fixture both SDKs read', () => {
  it('each SDK files its items under its own name', () => {
    expect(FIXTURE.service.template).toBe(kx.SERVICE_TEMPLATE);
    expect(FIXTURE.service.legacy_template).toBe(kx.LEGACY_SERVICE_TEMPLATE);
    expect(FIXTURE.service.runtimes).toEqual(kx.RUNTIME_LABELS);
    for (const example of FIXTURE.service.examples) {
      expect(kx.serviceName(example.namespace, example.runtime)).toBe(example.service);
      expect(kx.legacyServiceName(example.namespace)).toBe(example.legacy);
    }
    expect(serviceKey('cli')).toBe(OWN);
  });

  it('the program macOS names', () => {
    for (const c of FIXTURE.program_name.cases) expect(kx.programName(c.path)).toBe(c.program);
  });

  it('what a recorded program means for the next read', () => {
    for (const c of FIXTURE.record.predictions) expect(kx.prediction(c.recorded, c.current), c.about).toBe(c.prediction);
  });

  it('the words', () => {
    expect([...kx.EXPLANATION]).toEqual(FIXTURE.explanation);
    expect(kx.EXPLANATION).toHaveLength(4);
    expect(kx.BLOCKED).toBe(FIXTURE.blocked);
    expect(kx.BLOCKED_COMMAND).toBe(FIXTURE.blocked_command);
    expect(kx.LEFT_BEHIND).toBe(FIXTURE.left_behind);
    expect(kx.DENIED).toBe(FIXTURE.denied);
    expect(kx.NO_DEFAULT_KEYCHAIN).toBe(FIXTURE.no_default_keychain);
    expect(kx.OTHER_SIGNED_IN).toBe(FIXTURE.other_runtime.signed_in);
    expect(kx.OTHER_KEYS).toBe(FIXTURE.other_runtime.keys);
    expect(kx.OTHER_KEYS_COMMAND).toBe(FIXTURE.other_runtime.keys_command);
    expect(kx.INTERPRETER_NAMES).toEqual(FIXTURE.names.interpreter);
    expect(kx.CLI_NAMES).toEqual(FIXTURE.names.cli);
    expect(kx.RECORD_FILE).toBe(FIXTURE.record.file);
    expect(kx.RECORD_FILE_ELSEWHERE).toBe(FIXTURE.record.file_elsewhere);
    expect(kx.RECORD_ABOUT).toBe(FIXTURE.record.about);
    expect(kx.STATUS_ROW).toBe(FIXTURE.status_row);
    expect(kx.DIALOG_STATUSES).toEqual(FIXTURE.no_quiet_read.dialog_statuses);
    expect(kx.ITEM_NOT_FOUND).toBe(FIXTURE.no_quiet_read.item_not_found);
    expect(kx.READ_TIMEOUT_MS).toBe(FIXTURE.no_quiet_read.read_timeout_ms);
    const { cases: _cases, ...words } = FIXTURE.doctor;
    expect(kx.DOCTOR_WORDS).toEqual(words);
  });
});

describe('the record', () => {
  it('lives beside the items, and never above a test folder', () => {
    expect(kx.recordPath('/Users/x/.webagents-local/secrets')).toBe('/Users/x/.webagents-local/keychain.json');
    expect(kx.recordPath('/tmp/anything')).toBe('/tmp/anything/keychain.record.json');
  });

  it('is 0600 and holds no value', async () => {
    const dir = tempDir('wa-kx-record-');
    const world = new MacWorld();
    await macStore(dir, world).set('platform_token', DUMMY);
    const file = path.join(dir, 'keychain.json');
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);
    expect(fs.readFileSync(file, 'utf8')).not.toContain(DUMMY);
    const me = await kx.currentProgram();
    const entry = (await record(dir).item(OWN, 'platform_token'))!;
    expect(Object.keys(entry).sort()).toEqual([...FIXTURE.record.fields].sort());
    expect(entry.program).toBe(me.program);
    expect(entry.version).toBe(me.version);
  });

  it('a looser record is repaired when read', async () => {
    const dir = tempDir('wa-kx-record-');
    const file = path.join(dir, 'keychain.json');
    fs.writeFileSync(file, JSON.stringify(FIXTURE.record.sample), { mode: 0o644 });
    fs.chmodSync(file, 0o644);
    await record(dir).items();
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);
  });

  it('the sample reads the same in both SDKs', async () => {
    const dir = tempDir('wa-kx-record-');
    fs.writeFileSync(path.join(dir, 'keychain.json'), JSON.stringify(FIXTURE.record.sample));
    const reads = FIXTURE.record.sample_reads;
    const rec = record(dir);
    expect('platform_token' in ((await rec.items('python'))['webagents (Python) cli'] ?? {})).toBe(reads.python_signed_in);
    expect(Object.keys((await rec.items('python'))['webagents (Python) providers'] ?? {}).sort()).toEqual(reads.python_key_names);
    expect('platform_token' in ((await rec.items())['webagents (TypeScript) cli'] ?? {})).toBe(reads.typescript_signed_in);
    expect((await rec.pending()).map(({ service, account, namespace, legacy }) => ({ service, account, namespace, legacy }))).toEqual(reads.typescript_pending);
  });
});

describe('reads: said before, never waited on', () => {
  it('an item this node last used is read, bounded, with no dialog', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY);
    await recordThisNode(dir, OWN, 'platform_token');
    run(world, false);
    expect(await macStore(dir, world).get('platform_token')).toBe(DUMMY);
    expect(world.dialogs()).toEqual([]);
    expect(world.said()).toEqual([]);
  });

  it('nobody to answer is never asked: an item with no record is not read at all', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    world.add(OWN, 'OTHER', DUMMY, 'node 24.6.0');
    run(world, false);
    const store = macStore(dir, world);
    expect(await store.get('platform_token')).toBeNull();
    expect(await store.get('OTHER')).toBeNull();
    expect(world.reads).toEqual([]);
    expect(world.dialogs()).toEqual([]);
    const me = await kx.currentProgram();
    expect(world.said()).toEqual([FIXTURE.blocked.replace(/\{program\}/g, me.program).replace('{item}', OWN).replace('{command}', 'webagents whoami')]);
    const pending = await record(dir).pending();
    expect(pending.map((p) => [p.service, p.account, p.legacy]).sort()).toEqual([
      [OWN, 'OTHER', false],
      [OWN, 'platform_token', false],
    ]);
  });

  it('a copy in the file stands in, and nothing is said', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    fs.mkdirSync(path.join(dir, 'secrets'), { recursive: true });
    fs.writeFileSync(path.join(dir, 'secrets', 'cli.json'), JSON.stringify({ platform_token: OTHER_DUMMY }));
    run(world, false);
    expect(await macStore(dir, world).get('platform_token')).toBe(OTHER_DUMMY);
    expect(world.said()).toEqual([]);
    expect(world.dialogs()).toEqual([]);
  });

  it('a terminal hears the four lines before the dialog, once per run', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.program = 'node 24.7.0';
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    world.add(OWN, 'OTHER', OTHER_DUMMY, 'node 24.6.0');
    await recordOlderNode(dir, OWN, 'platform_token');
    run(world, true);
    const store = macStore(dir, world);
    expect(await store.get('platform_token')).toBe(DUMMY);
    expect(await store.get('OTHER')).toBe(OTHER_DUMMY);
    const me = await kx.currentProgram();
    const lines = FIXTURE.explanation.map((line: string) => line.replace(/\{program\}/g, me.program).replace('{item}', OWN));
    expect(world.events[0]).toEqual(['say', lines.join('\n')]);
    expect(world.events.map((event) => event[0])).toEqual(['say', 'dialog', 'dialog']);
    // Always Allow, and the record now names this node: the next run with
    // nobody to answer reads it without asking.
    world.events = [];
    run(world, false);
    expect(await macStore(dir, world).get('platform_token')).toBe(DUMMY);
    expect(world.events).toEqual([]);
  });

  it('serve and the daemon are never asked, even at a terminal', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    run(world, true);
    kx.forbidKeychainDialogs('serve');
    expect(await macStore(dir, world).get('platform_token')).toBeNull();
    expect(world.dialogs()).toEqual([]);
    expect(kx.wasBlocked(OWN, 'platform_token')).toBe(true);
  });

  it('deny carries on without the item', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    world.answer = 'deny';
    run(world, true);
    expect(await macStore(dir, world).get('platform_token')).toBeNull();
    const me = await kx.currentProgram();
    expect(world.said().at(-1)).toBe(FIXTURE.denied.replace('{program}', me.program).replace('{item}', OWN));
  });

  it('an item the record calls safe that asks anyway (Allow, not Always Allow) is bounded by the timeout', async () => {
    const dir = tempDir('wa-kx-read-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    await recordThisNode(dir, OWN, 'platform_token');
    run(world, false);
    const started = Date.now();
    expect(await macStore(dir, world, 'cli', 50).get('platform_token')).toBeNull();
    expect(Date.now() - started).toBeLessThan(5000);
    expect(world.events.some((event) => event[0] === 'hung')).toBe(true);
    // The record was wrong: it is dropped, so the next check predicts a dialog.
    expect(await record(dir).item(OWN, 'platform_token')).toBeUndefined();
    expect(kx.wasBlocked(OWN, 'platform_token')).toBe(true);
  });
});

describe('the old shared name', () => {
  it('in a terminal an old item is read after the explanation, copied, and left in place', async () => {
    const dir = tempDir('wa-kx-old-');
    const world = new MacWorld();
    world.add(OLD, 'platform_token', DUMMY, 'Python 3.14.2');
    run(world, true);
    const store = macStore(dir, world);
    expect(await store.get('platform_token')).toBe(DUMMY);
    expect(world.events.map((event) => event[0])).toEqual(['say', 'dialog']);
    expect(world.get(OWN, 'platform_token')?.value).toBe(DUMMY);
    expect(world.get(OLD, 'platform_token')).toBeDefined();
    expect(await record(dir).legacyDone(OLD, 'platform_token')).toBe(true);
    expect(await store.ownIndexNames()).toEqual(['platform_token']);
  });

  it('a run with nobody to answer never reads an old item', async () => {
    const dir = tempDir('wa-kx-old-');
    const world = new MacWorld();
    world.add(OLD, 'platform_token', DUMMY);
    run(world, false);
    expect(await macStore(dir, world).get('platform_token')).toBeNull();
    expect(world.reads).toEqual([]);
    expect(world.get(OWN, 'platform_token')).toBeUndefined();
    expect((await record(dir).pending()).map((p) => [p.service, p.account, p.legacy])).toEqual([[OWN, 'platform_token', true]]);
    const me = await kx.currentProgram();
    expect(world.said()).toEqual([FIXTURE.blocked.replace(/\{program\}/g, me.program).replace('{item}', OLD).replace('{command}', 'webagents whoami')]);
  });

  it('removing names an old item (this SDK never removes one) and never reads it again', async () => {
    const dir = tempDir('wa-kx-old-');
    const world = new MacWorld();
    world.add(OLD, 'platform_token', DUMMY, 'Python 3.14.2');
    run(world, true);
    const store = macStore(dir, world);
    expect(await store.delete('platform_token')).toBe(true);
    expect(store.leftBehind).toEqual([{ item: OLD, account: 'platform_token' }]);
    expect(world.get(OLD, 'platform_token')).toBeDefined();
    expect(world.dialogs()).toEqual([]);
    expect(kx.leftBehindSentence(OLD, 'platform_token')).toBe(FIXTURE.left_behind.replace(/\{item\}/g, OLD).replace('{account}', 'platform_token'));
    // Removed means removed: the next read does not copy it back.
    expect(await store.get('platform_token')).toBeNull();
    expect(world.dialogs()).toEqual([]);
  });

  it('list names what the old index holds until it is copied or removed', async () => {
    const dir = tempDir('wa-kx-old-');
    const world = new MacWorld();
    world.add('webagents:providers', 'OPENAI_API_KEY', DUMMY);
    const store = macStore(dir, world, 'providers');
    fs.mkdirSync(path.join(dir, 'secrets'), { recursive: true });
    fs.writeFileSync(path.join(dir, 'secrets', 'providers.index.json'), JSON.stringify(['OPENAI_API_KEY']));
    expect(await store.list()).toEqual({ names: ['OPENAI_API_KEY'], complete: false });
    run(world, true);
    await store.delete('OPENAI_API_KEY');
    expect(await store.list()).toEqual({ names: [], complete: false });
  });

  it('elsewhere the old item is copied in any run', async () => {
    const items = new Map<string, string>([[`${OLD}\u0000platform_token`, DUMMY]]);
    class Entry {
      constructor(
        private s: string,
        private a: string,
      ) {}
      getPassword() {
        return items.get(`${this.s}\u0000${this.a}`) ?? null;
      }
      setPassword(value: string) {
        items.set(`${this.s}\u0000${this.a}`, value);
      }
      deletePassword() {
        return items.delete(`${this.s}\u0000${this.a}`);
      }
    }
    const dir = tempDir('wa-kx-old-');
    const keychain = new kx.KeychainAccess({ Entry }, 'cli', path.join(dir, 'secrets'), { mac: null });
    const store = new SecretStore({ namespace: 'cli', keyring: { Entry } as never, unavailableReason: 'plain', filePath: path.join(dir, 'secrets', 'cli.json'), quiet: true, keychain });
    expect(await store.get('platform_token')).toBe(DUMMY);
    expect(items.get(`${OWN}\u0000platform_token`)).toBe(DUMMY);
  });
});

describe('writes and removals after an upgrade', () => {
  it('replacing an item an older node made', async () => {
    const dir = tempDir('wa-kx-write-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    await recordOlderNode(dir, OWN, 'platform_token');
    run(world, false);
    await expect(macStore(dir, world).set('platform_token', OTHER_DUMMY)).rejects.toBeInstanceOf(kx.KeychainDialogBlocked);
    expect(world.dialogs()).toEqual([]);
    run(world, true);
    expect(await macStore(dir, world).set('platform_token', OTHER_DUMMY)).toBe('keystore');
    expect(world.events.map((event) => event[0])).toEqual(['say', 'dialog']);
    expect(world.get(OWN, 'platform_token')?.value).toBe(OTHER_DUMMY);
  });

  it('removing an item an older node made, with nobody to answer', async () => {
    const dir = tempDir('wa-kx-write-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
    run(world, false);
    await expect(macStore(dir, world).delete('platform_token')).rejects.toBeInstanceOf(kx.KeychainDialogBlocked);
    expect(world.get(OWN, 'platform_token')).toBeDefined();
    expect(world.dialogs()).toEqual([]);
  });

  it('a new item is added with nobody to answer, and read back', async () => {
    const dir = tempDir('wa-kx-write-');
    const world = new MacWorld();
    run(world, false);
    const store = macStore(dir, world);
    expect(await store.set('platform_token', DUMMY)).toBe('keystore');
    expect(await store.get('platform_token')).toBe(DUMMY);
    expect(world.dialogs()).toEqual([]);
  });
});

describe('where the keychain is not used at all', () => {
  it('the file backend makes no keychain call', async () => {
    const dir = tempDir('wa-kx-file-');
    const store = await openSecretStore({ namespace: 'cli', secretsDir: dir, quiet: true, backend: 'file' });
    expect(store.status().backend).toBe('file');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((store as any).access).toBeNull();
  });

  it('no default keychain means the file', async () => {
    vi.resetModules();
    vi.doMock('../../../src/skills/secrets/keychain-ux', async (original) => ({
      ...(await original<typeof import('../../../src/skills/secrets/keychain-ux')>()),
      defaultKeychainAvailable: async () => false,
    }));
    try {
      const fresh = await import('../../../src/skills/secrets/store');
      const dir = tempDir('wa-kx-file-');
      const status = (await fresh.openSecretStore({ namespace: 'cli', secretsDir: dir, quiet: true, backend: 'auto' })).status();
      expect(status.backend).toBe('file');
      // Where @napi-rs/keyring itself cannot load, that is the reason given instead.
      if (!/not installed|failed to load|not running under Node/.test(status.reason ?? '')) {
        expect(status.reason).toBe(FIXTURE.no_default_keychain.replace('{home}', process.env.HOME ?? '~'));
      }
    } finally {
      vi.doUnmock('../../../src/skills/secrets/keychain-ux');
      vi.resetModules();
    }
  });
});

it('whoami in a terminal reads what a background run could not', async () => {
  const dir = tempDir('wa-kx-settle-');
  const world = new MacWorld();
  world.add(OWN, 'platform_token', DUMMY, 'node 24.6.0');
  run(world, false);
  expect(await macStore(dir, world).get('platform_token')).toBeNull();
  expect(await record(dir).pending()).toHaveLength(1);

  world.events = [];
  run(world, true);
  const settled = await kx.settlePending([path.join(dir, 'keychain.json')], async (namespace) => macStore(dir, world, namespace));
  expect(settled).toBe(1);
  expect(await record(dir).pending()).toEqual([]);
  expect(world.events.map((event) => event[0])).toEqual(['say', 'dialog']);
  world.events = [];
  run(world, false);
  expect(await macStore(dir, world).get('platform_token')).toBe(DUMMY);
  expect(world.events).toEqual([]);
});

describe("doctor's keychain line", () => {
  it('the fixture cases', () => {
    for (const c of FIXTURE.doctor.cases) {
      const line = kx.doctorLine({ ...c.facts });
      expect(line.name).toBe('keychain');
      expect(line.status, c.about).toBe(c.status);
      expect(line.detail, c.about).toBe(c.detail);
      expect(line.fix, c.about).toBe(c.fix);
    }
  });

  it('facts are gathered without reading a value', async () => {
    const dir = tempDir('wa-kx-doctor-');
    const world = new MacWorld();
    world.add(OWN, 'platform_token', DUMMY);
    world.add(OLD, 'platform_token', DUMMY);
    const store = macStore(dir, world);
    await recordThisNode(dir, OWN, 'platform_token');
    const me = await kx.currentProgram();
    const facts = await kx.doctorFacts([store], record(dir), { [OLD]: ['platform_token'] }, { mac: world.mac() });
    expect(world.reads).toEqual([]);
    expect(facts).toEqual({ backend: 'keychain', runtime: 'typescript', count: 1, interpreter_version: me.version, prediction: 'quiet', legacy: true, pending: 0, other: false, env_token: false });
    await recordOlderNode(dir, OWN, 'platform_token');
    expect((await kx.doctorFacts([store], record(dir), {}, { mac: world.mac() })).prediction).toBe('upgraded');
  });

  it('the other CLI is named when this one has nothing', async () => {
    const dir = tempDir('wa-kx-doctor-');
    const world = new MacWorld();
    const store = macStore(dir, world);
    fs.writeFileSync(path.join(dir, 'keychain.json'), JSON.stringify({ runtimes: { python: { items: { 'webagents (Python) cli': { platform_token: { program: 'Python' } } } } } }));
    const facts = await kx.doctorFacts([store], record(dir), {}, { mac: world.mac() });
    expect(facts.count).toBe(0);
    expect(facts.other).toBe(true);
    expect(kx.doctorLine(facts).detail).toBe('macOS keychain: nothing stored by the TypeScript CLI yet; the Python CLI keeps its own items here');
  });
});
