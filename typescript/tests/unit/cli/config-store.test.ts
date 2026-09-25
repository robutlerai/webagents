/**
 * Phase 3: the TypeScript half of one config contract (2026-09-23).
 *
 * The two CLIs COLLIDED on `~/.webagents/config.json` with incompatible
 * schemas (Python wrote nested objects, this one wrote flat strings) and
 * DIVERGED on the credential filename, so logging in with one SDK left the
 * other logged out. These pin the shared contract from this side.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  ConfigStore,
  DEFAULTS,
  envChain,
  expand,
  globalDir,
  profileName,
} from '../../../src/cli/config-store';

let dir: string;
const savedEnv = { ...process.env };

beforeEach(() => {
  dir = mkdtempSync(path.join(tmpdir(), 'webagents-cfg-'));
  delete process.env.WEBAGENTS_PROFILE;
  delete process.env.WEBAGENTS_TOKEN;
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  process.env = { ...savedEnv };
});

function writeProjectConfig(data: Record<string, unknown>) {
  mkdirSync(path.join(dir, '.webagents'), { recursive: true });
  writeFileSync(path.join(dir, '.webagents', 'config.json'), JSON.stringify(data));
}

describe('config precedence', () => {
  it('orders layers highest first', () => {
    const store = new ConfigStore({ cwd: dir });
    expect(store.layers().map(([name]) => name)).toEqual(['flag', 'project', 'global', 'default']);
  });

  it('lets a flag beat project, and project beat default', () => {
    writeProjectConfig({ 'daemon.port': 4321, model: 'from-project' });
    const store = new ConfigStore({ cwd: dir, overrides: { model: 'from-flag' } });

    expect(store.get('model')).toBe('from-flag');
    expect(store.sourceOf('model')).toBe('flag');
    expect(store.get('daemon.port')).toBe(4321);
    expect(store.sourceOf('daemon.port')).toBe('project');
    expect(store.get('daemon.host')).toBe(DEFAULTS['daemon.host']);
    expect(store.sourceOf('daemon.host')).toBe('default');
  });

  it('creates nothing merely by being read', () => {
    // The Python CLI used to scaffold eight directories in whatever folder you
    // were standing in, reached from `is_authenticated()`.
    new ConfigStore({ cwd: dir }).get('daemon.port');
    expect(require('node:fs').readdirSync(dir)).toEqual([]);
  });
});

describe('dotenv and ${VAR} expansion', () => {
  it('reads .env but never overrides the process environment', () => {
    writeFileSync(path.join(dir, '.env'), 'FROM_FILE=file\nSHARED=file\n# c\nQUOTED="q"\n');
    process.env.SHARED = 'process';

    const env = envChain(undefined, dir);
    expect(env.FROM_FILE).toBe('file');
    expect(env.QUOTED).toBe('q');
    // A developer who exports something in their shell means it.
    expect(env.SHARED).toBe('process');
  });

  it('expands, falls back, and leaves an unresolvable reference as written', () => {
    const env = { SET: 'value' };
    expect(expand('${SET}', env)).toBe('value');
    expect(expand('${MISSING:-fallback}', env)).toBe('fallback');
    // Not "". A silent empty string turns a misconfigured URL into an
    // inscrutable request to nowhere.
    expect(expand('${MISSING}', env)).toBe('${MISSING}');
  });

  it('resolves a reference stored in config', () => {
    writeProjectConfig({ 'platform.url': '${PORTAL_URL}' });
    process.env.PORTAL_URL = 'https://staging.example';
    expect(new ConfigStore({ cwd: dir }).get('platform.url')).toBe('https://staging.example');
  });
});

describe('config validation', () => {
  it('reports an unknown key rather than accepting it', () => {
    writeProjectConfig({ 'daemon.prot': 1 });
    const problems = new ConfigStore({ cwd: dir }).validate();
    expect(problems.some((p) => p.includes('unknown key') && p.includes('daemon.prot'))).toBe(true);
  });

  it('reports an unresolvable reference', () => {
    writeProjectConfig({ 'platform.url': '${NOT_SET_ANYWHERE}' });
    const problems = new ConfigStore({ cwd: dir }).validate();
    expect(problems.some((p) => p.includes('NOT_SET_ANYWHERE'))).toBe(true);
  });

  it('reports malformed JSON without breaking every other command', () => {
    mkdirSync(path.join(dir, '.webagents'), { recursive: true });
    writeFileSync(path.join(dir, '.webagents', 'config.json'), '{not json');
    const store = new ConfigStore({ cwd: dir });
    expect(store.validate().some((p) => p.includes('not valid JSON'))).toBe(true);
    expect(store.get('daemon.host')).toBe(DEFAULTS['daemon.host']);
  });
});

describe('profiles', () => {
  it('resolves the profile from the environment, not just an argument', () => {
    expect(path.basename(globalDir())).toBe('.webagents');
    process.env.WEBAGENTS_PROFILE = 'staging';
    // Taking the argument literally is what let the token go to the profile
    // keystore while its metadata landed in the shared directory.
    expect(path.basename(globalDir())).toBe('.webagents-staging');
    expect(profileName()).toBe('staging');
  });

  it('lets an explicit profile beat the environment', () => {
    process.env.WEBAGENTS_PROFILE = 'from-env';
    expect(path.basename(globalDir('from-arg'))).toBe('.webagents-from-arg');
  });
});

describe('matching the Python contract', () => {
  it('knows exactly the same config keys', async () => {
    // The two SDKs share these files. A key one side knows and the other
    // rejects would make `config set` fail depending on which CLI you reached
    // for, which is the collision this phase exists to end.
    const pythonSource = await import('node:fs').then((fs) =>
      fs.readFileSync(
        path.join(__dirname, '../../../../python/webagents/cli/config_store.py'),
        'utf-8',
      ),
    );
    const block = pythonSource.slice(
      pythonSource.indexOf('DEFAULTS: Dict[str, Any] = {'),
      pythonSource.indexOf('}', pythonSource.indexOf('DEFAULTS: Dict[str, Any] = {')),
    );
    const pythonKeys = [...block.matchAll(/^\s*"([^"]+)":/gm)].map((m) => m[1]).sort();
    expect(Object.keys(DEFAULTS).sort()).toEqual(pythonKeys);
  });
});
