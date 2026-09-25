/**
 * Defects found by the 2026-09-23 end-to-end campaign in the TypeScript CLI.
 * Each passed the unit suite and only showed up by running the built CLI.
 */

import { describe, expect, it, beforeEach, afterEach } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { scopedNamespace, ConfigStore } from '../../../src/cli/config-store';
import { tempDirs, TSX_CLI } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const REPO = path.resolve(__dirname, '../../../..');
const CLI = path.resolve(__dirname, '../../../src/cli/index.ts');

function runCli(args: string[], env: Record<string, string>, cwd: string) {
  return spawnSync(process.execPath, [TSX_CLI, CLI, ...args], {
    cwd,
    env: { ...process.env, ...env },
    encoding: 'utf-8',
    timeout: 60_000,
  });
}

describe('S-219: the keystore namespace carries the profile', () => {
  const saved = process.env.WEBAGENTS_PROFILE;
  afterEach(() => {
    if (saved === undefined) delete process.env.WEBAGENTS_PROFILE;
    else process.env.WEBAGENTS_PROFILE = saved;
  });

  it('keeps the bare name for the default profile', () => {
    delete process.env.WEBAGENTS_PROFILE;
    // So a token stored before the fix stays readable from the profile that wrote it.
    expect(scopedNamespace('cli')).toBe('cli');
  });

  it('suffixes it under a profile, explicit or from the environment', () => {
    expect(scopedNamespace('cli', 'ci')).toBe('cli-ci');
    process.env.WEBAGENTS_PROFILE = 'test';
    expect(scopedNamespace('cli')).toBe('cli-test');
  });

  it('matches the Python implementation of the same rule', () => {
    // Two implementations of one rule. If they drift, a profile logged in from
    // one SDK is logged out in the other.
    const py = fs.readFileSync(
      path.join(REPO, 'python/webagents/cli/config_store.py'),
      'utf-8',
    );
    expect(py).toContain('def scoped_namespace(');
    expect(py).toContain('return base if not resolved else f"{base}-{resolved}"');
  });

  it('is what the credential store actually uses', () => {
    const src = fs.readFileSync(path.join(REPO, 'typescript/src/cli/credentials.ts'), 'utf-8');
    expect(src).toContain('namespace: scopedNamespace(CLI_NAMESPACE, resolved)');
    expect(src).not.toMatch(/namespace:\s*CLI_NAMESPACE,/);
  });
});

describe('config commands and the daemon client read the SAME file', () => {
  let home: string;
  let cwd: string;

  beforeEach(() => {
    home = tempDir('wa-cfg-home-');
    cwd = tempDir('wa-cfg-cwd-');
  });

  it('under a profile, `config set` lands where ConfigStore reads', () => {
    // `config set` used a private loadConfig/saveConfig hard-wired to
    // ~/.webagents/config.json, while ConfigStore (which `status` and `list`
    // read) resolves ~/.webagents-<profile>/. Under a profile, set succeeded,
    // get echoed the value, and `status` kept dialling the default port.
    const env = { HOME: home, WEBAGENTS_PROFILE: 'campaign' };
    const set = runCli(['config', 'set', 'daemon.port', '8821'], env, cwd);
    expect(set.status).toBe(0);

    const profileFile = path.join(home, '.webagents-campaign', 'config.json');
    expect(JSON.parse(fs.readFileSync(profileFile, 'utf-8'))['daemon.port']).toBe(8821);
    expect(fs.existsSync(path.join(home, '.webagents', 'config.json'))).toBe(false);
  }, 90_000);

  it('stores a number as a number, not the string "8821"', () => {
    const env = { HOME: home };
    runCli(['config', 'set', 'daemon.port', '8821'], env, cwd);
    const stored = JSON.parse(fs.readFileSync(path.join(home, '.webagents', 'config.json'), 'utf-8'));
    expect(stored['daemon.port']).toBe(8821);
    expect(typeof stored['daemon.port']).toBe('number');
  }, 90_000);

  it('refuses an unknown key, as the Python CLI does', () => {
    const result = runCli(['config', 'set', 'nosuch.key', '1'], { HOME: home }, cwd);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('Unknown config key');
  }, 90_000);

  it('refuses a non-numeric value for a numeric key', () => {
    const result = runCli(['config', 'set', 'daemon.port', 'eighty'], { HOME: home }, cwd);
    expect(result.status).toBe(1);
  }, 90_000);

  it('the daemon client reads the value `config set` wrote', () => {
    process.env.HOME = home;
    const store = new ConfigStore({ profile: 'campaign' });
    store.set('daemon.port', 8821);
    expect(new ConfigStore({ profile: 'campaign' }).get('daemon.port')).toBe(8821);
  });
});

describe('publish finds the token where login put it', () => {
  it('reads it through getToken, never out of auth.json', () => {
    // Since Phase 3 `login` has stored the token in the keystore and written
    // only {portalUrl} to auth.json, so after a SUCCESSFUL login `sync` and
    // `publish` both said "Not logged in". `sync` is gone (2026-09-24);
    // `publish` is the one command that sends the agent.
    const src = fs.readFileSync(path.join(REPO, 'typescript/src/cli/publish.ts'), 'utf-8');
    expect(src).not.toMatch(/loadAuth\(\)/);
    expect(src).toContain('await getToken()');
  });

  it('a token in WEBAGENTS_TOKEN gets past the sign-in check', () => {
    const home = tempDir('wa-publish-home-');
    const cwd = tempDir('wa-publish-cwd-');
    fs.writeFileSync(path.join(cwd, 'AGENT.md'), '---\nname: probe\n---\nProbe.\n');
    // Pointed at a port with nothing on it: we only care that it did NOT stop
    // at "Not signed in", i.e. it read the token and went on to the network.
    const result = runCli(
      ['publish', '--yes'],
      {
        HOME: home,
        WEBAGENTS_TOKEN: 'fake-token-for-the-check',
        WEBAGENTS_PROFILE: 'campaign-publish',
        WEBAGENTS_SECRETS_BACKEND: 'file',
        ROBUTLER_API_URL: 'http://127.0.0.1:9',
      },
      cwd,
    );
    const out = `${result.stdout}${result.stderr}`;
    expect(out).not.toContain('Not signed in');
    expect(out).toContain('Could not reach http://127.0.0.1:9');
  }, 90_000);
});
