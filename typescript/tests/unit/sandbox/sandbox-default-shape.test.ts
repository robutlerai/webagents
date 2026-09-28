/**
 * The granular `sandbox:` shape, its defaults and aliases, the host groups,
 * the built-in denies as settings, the state strings, the refusal hints and
 * srt's refused-host log line (the sandbox-default lane, 2026-09-27),
 * against the shared fixture `python/tests/fixtures/sandbox/srt.json` that
 * `python/tests/sandbox/test_sandbox_default_shape.py` reads too. Nothing
 * here runs srt; the real-srt proofs are in `sandbox-default-denies.test.ts`.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  CREDENTIAL_DIRS,
  HOST_GROUPS,
  PROFILE_DIR_PATTERN,
  REFUSAL_HINTS,
  ROOT_READ_DENY,
  ROOT_READ_DENY_PATTERNS,
  SANDBOX_ACCEPTED_KEYS,
  SANDBOX_FILES_KEYS,
  SANDBOX_KEYS,
  SANDBOX_NETWORK_KEYS,
  SandboxDeclarationError,
  UNAVAILABLE_TAIL,
  buildSettings,
  envFromDotenv,
  expandHosts,
  isSandboxOff,
  noSandboxRequested,
  parseSandboxDeclaration,
  policyFromDeclaration,
  profileDirDenies,
  refusalKind,
  refusedHostsFromSrtLog,
  rootReadDenies,
  sandboxState,
  wrappedCommand,
} from '../../../src/sandbox/index';
import { NO_SANDBOX_WARNING, ShellSkill, UNRESTRICTED_WARNING } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8'));
const tempDir = tempDirs();

describe('the shape', () => {
  it('has the fixture keys, aliases and nested keys', () => {
    expect([...SANDBOX_KEYS]).toEqual(FIXTURE.schema.keys);
    expect([...SANDBOX_ACCEPTED_KEYS]).toEqual(FIXTURE.schema.accepted_keys);
    expect([...SANDBOX_FILES_KEYS]).toEqual(FIXTURE.schema.files_keys);
    expect([...SANDBOX_NETWORK_KEYS]).toEqual(FIXTURE.schema.network_keys);
    expect(HOST_GROUPS).toEqual(Object.fromEntries(Object.entries(FIXTURE.host_groups).filter(([key]) => key !== 'about')));
  });

  it('normalises the defaults, the aliases and the opt-out as the fixture pins them', () => {
    expect(parseSandboxDeclaration({})).toEqual(FIXTURE.schema.defaults.declaration);
    expect(parseSandboxDeclaration({ preset: 'strict' }).files.read).toEqual(FIXTURE.schema.defaults.strict_read);
    for (const kase of FIXTURE.schema.alias_cases as Array<{ declared: unknown; normalised: unknown }>) {
      expect(parseSandboxDeclaration(kase.declared)).toEqual(kase.normalised);
    }
    expect(isSandboxOff(false) && isSandboxOff('off') && isSandboxOff('OFF')).toBe(true);
    expect(isSandboxOff({}) || isSandboxOff('on') || isSandboxOff(true)).toBe(false);
    expect(parseSandboxDeclaration(true)).toEqual(FIXTURE.schema.defaults.declaration);
    expect(parseSandboxDeclaration('on')).toEqual(FIXTURE.schema.defaults.declaration);
    expect(policyFromDeclaration(parseSandboxDeclaration('off'), { cwd: tempDir('wa-shape-off-') }).confined).toBe(false);
  });

  it('refuses an unknown key at every level with the fixture sentence, and both spellings together', () => {
    for (const kase of FIXTURE.unknown_key.cases as Array<{ declared: unknown; message: string }>) {
      expect(() => parseSandboxDeclaration(kase.declared)).toThrow(SandboxDeclarationError);
      expect(() => parseSandboxDeclaration(kase.declared)).toThrow(kase.message);
    }
    expect(() => parseSandboxDeclaration({ allowed_folders: ['.'], files: { write: ['.'] } })).toThrow(/old spelling of files.write/);
    expect(() => parseSandboxDeclaration({ env_passthrough: ['A'], env: ['B'] })).toThrow(/old spelling of env/);
    expect(() => parseSandboxDeclaration({ network: { local: 'yes' } })).toThrow(/network.local must be true or false/);
    expect(() => parseSandboxDeclaration([])).toThrow(SandboxDeclarationError);
  });

  it('expands host groups to exactly the fixture hosts and keeps checking the rest', () => {
    for (const [group, hosts] of Object.entries(HOST_GROUPS)) expect(expandHosts([group])).toEqual(hosts);
    expect(expandHosts(['npm', 'example.com', 'registry.npmjs.org'])).toEqual(['registry.npmjs.org', 'example.com']);
    expect(() => expandHosts(['*'])).toThrow(/network entry/);
    const built = policyFromDeclaration(parseSandboxDeclaration({ network: { hosts: ['github', 'pypi'] } }), { cwd: tempDir('wa-shape-groups-') });
    expect(built.networkDomains).toEqual([...HOST_GROUPS.github, ...HOST_GROUPS.pypi]);
  });
});

describe('the built-in denies', () => {
  it('lists the keychain and the profile pattern, and enumerates on Linux', () => {
    expect([...CREDENTIAL_DIRS]).toEqual(FIXTURE.credential_dirs_unreadable_under_development);
    expect(PROFILE_DIR_PATTERN).toBe(FIXTURE.builtin_denies.profile_dirs.pattern);
    expect([...ROOT_READ_DENY]).toEqual(FIXTURE.builtin_denies.root_read_deny.literal);
    expect([...ROOT_READ_DENY_PATTERNS]).toEqual(FIXTURE.builtin_denies.root_read_deny.patterns);
    const home = tempDir('wa-shape-home-');
    fs.mkdirSync(path.join(home, '.webagents-local'));
    fs.mkdirSync(path.join(home, '.webagents-team'));
    fs.writeFileSync(path.join(home, '.webagents-note'), 'a file, not a profile');
    expect(profileDirDenies(home, 'darwin')).toEqual([path.join(home, '.webagents-*')]);
    expect(profileDirDenies(home, 'linux')).toEqual([path.join(home, '.webagents-local'), path.join(home, '.webagents-team')]);
    const root = tempDir('wa-shape-root-');
    fs.writeFileSync(path.join(root, '.env.local'), 'A=1');
    fs.writeFileSync(path.join(root, '.envelope'), 'not env');
    expect(rootReadDenies(root, 'darwin')).toEqual([path.join(root, '.env'), path.join(root, '.webagents'), path.join(root, '.env.*')]);
    expect(rootReadDenies(root, 'linux')).toEqual([path.join(root, '.env.local')]);
    fs.writeFileSync(path.join(root, '.env'), 'B=2');
    expect(rootReadDenies(root, 'linux')).toEqual([path.join(root, '.env'), path.join(root, '.env.local')]);
  });

  it('puts the granular keys into the settings as the fixture pins them', () => {
    const base = tempDir('wa-shape-settings-');
    const home = path.join(base, 'home');
    const work = path.join(base, 'work');
    const cwd = path.join(base, 'cwd');
    for (const dir of [home, work, cwd, path.join(home, 'notes')]) fs.mkdirSync(dir);
    const savedHome = process.env.HOME;
    process.env.HOME = home;
    try {
      const granular = FIXTURE.settings_cases_granular;
      const sub = (value: unknown): unknown =>
        JSON.parse(
          JSON.stringify(value)
            .replace(/\{work\}/g, fs.realpathSync(work))
            .replace(/\{home\}/g, fs.realpathSync(home))
            .replace(/\{cwd\}/g, fs.realpathSync(cwd)),
        );
      const built = policyFromDeclaration(parseSandboxDeclaration(sub(granular.declared)), { cwd, tmpdir: path.join(base, 'tmp') });
      const settings = buildSettings(built, {}) as Record<string, Record<string, unknown>>;
      const fill = (value: unknown): unknown => JSON.parse(JSON.stringify(sub(value)).replace(/\{scratch\}/g, built.scratch!));
      const check = (expectations: Record<string, unknown>) => {
        for (const [key, expected] of Object.entries(expectations)) {
          const [section, field] = key.split('.');
          const includes = field.endsWith('_includes');
          const actual = settings[section][includes ? field.slice(0, -'_includes'.length) : field] as unknown[];
          if (includes) for (const item of fill(expected) as unknown[]) expect(actual, key).toContain(item);
          else expect(actual, key).toEqual(fill(expected));
        }
      };
      check(granular.expect);
      if (process.platform === 'darwin') check(granular.darwin);
      else {
        expect(built.unenforceable).toContain('network.sockets');
        expect('allowUnixSockets' in settings.network).toBe(false);
      }
      for (const key of FIXTURE.never_set as string[]) {
        expect(key in settings.network).toBe(false);
        expect(key in settings.filesystem).toBe(false);
      }
      // The defaults, on macOS: the keychain, the profile glob, the root denies.
      if (process.platform === 'darwin') {
        const dflt = buildSettings(policyFromDeclaration(parseSandboxDeclaration({}), { cwd, tmpdir: path.join(base, 'tmp') }), {}) as Record<string, Record<string, unknown[]>>;
        for (const entry of fill(granular.default_denies_darwin) as string[]) expect(dflt.filesystem.denyRead).toContain(entry);
      }
    } finally {
      if (savedHome === undefined) delete process.env.HOME;
      else process.env.HOME = savedHome;
    }
  });
});

describe('the state, the opt-outs and the fail-closed sentence', () => {
  it('spells the state as the fixture does, from the origin', () => {
    const dir = tempDir('wa-shape-state-');
    const status = FIXTURE.status;
    const byDefault = new ShellSkill({ baseDir: dir, env: {} });
    expect(byDefault.sandboxStateLine()).toBe(status.default);
    expect(sandboxState(byDefault.policy, 'default')).toBe(status.default);
    const declared = new ShellSkill({ baseDir: dir, sandbox: { preset: 'strict' }, env: {} });
    expect(declared.sandboxStateLine()).toBe(status.agent_file.replace('{preset}', 'strict'));
    const warned: string[] = [];
    const original = console.warn;
    console.warn = (line: string) => {
      warned.push(String(line));
    };
    try {
      const off = new ShellSkill({ baseDir: dir, sandbox: false, env: {} });
      expect(off.sandboxStateLine()).toBe(status.off_agent_file);
      const flag = new ShellSkill({ baseDir: dir, sandbox: { preset: 'strict' }, env: { [status.env_flag]: '1' } });
      expect(flag.sandboxStateLine()).toBe(status.off_flag);
      expect(flag.sandboxOrigin).toBe('--no-sandbox');
    } finally {
      console.warn = original;
    }
    expect(UNRESTRICTED_WARNING).toBe(status.warnings.off_agent_file);
    expect(UNRESTRICTED_WARNING).toBe(FIXTURE.unrestricted.warning);
    expect(NO_SANDBOX_WARNING).toBe(status.warnings.off_flag);
    expect(warned).toEqual([UNRESTRICTED_WARNING, NO_SANDBOX_WARNING]);
    expect(noSandboxRequested({ WEBAGENTS_NO_SANDBOX: 'yes' }) && noSandboxRequested({ WEBAGENTS_NO_SANDBOX: '1' })).toBe(true);
    expect(noSandboxRequested({ WEBAGENTS_NO_SANDBOX: '0' }) || noSandboxRequested({})).toBe(false);
  });

  it('names the opt-out when srt cannot run here', () => {
    expect(UNAVAILABLE_TAIL).toBe(FIXTURE.refusals.unavailable_tail);
    // The sandbox-engine lane (2026-09-27): the reason, this machine's fix, then the tail.
    expect(FIXTURE.refusals.unavailable).toBe(`{reason}: {fix}. ${UNAVAILABLE_TAIL}`);
  });
});

describe('the hints and the refused-host log', () => {
  it('says the fixture sentences and recognises the fixture outputs', () => {
    expect(REFUSAL_HINTS).toEqual(FIXTURE.hints.sentences);
    for (const kase of FIXTURE.hints.cases as Array<{ command: string; output: string; kind: string | null }>) {
      expect(refusalKind(kase.command, kase.output) ?? null, `${kase.command}: ${kase.output}`).toBe(kase.kind);
    }
  });

  it("reads refused hosts from srt's own log lines only", () => {
    for (const kase of FIXTURE.refused_hosts.cases as Array<{ stderr: string; hosts: string[] }>) {
      expect(refusedHostsFromSrtLog(kase.stderr)).toEqual(kase.hosts);
    }
  });

  it('wraps the command with the Node proxy switch, and merges stderr when capturing', () => {
    const { command, path: userPath, network, text, merged_text } = FIXTURE.no_proxy.wrapped;
    expect(wrappedCommand(command, userPath, network)).toBe(text);
    expect(wrappedCommand(command, userPath, network, true)).toBe(merged_text);
  });

  it('reads a listed env name from the .env files the CLI loads, and nothing else', () => {
    const cwd = tempDir('wa-shape-dotenv-');
    const home = tempDir('wa-shape-dotenv-home-');
    fs.writeFileSync(path.join(cwd, '.env'), '# keys\nPROBE_API_KEY="from-project"\nPLAIN=plain\nOTHER_TOKEN=other\n');
    fs.mkdirSync(path.join(home, '.webagents-local'));
    fs.writeFileSync(path.join(home, '.webagents-local', '.env'), 'PROFILE_SECRET=from-profile\nPROBE_API_KEY=shadowed\n');
    const savedHome = process.env.HOME;
    process.env.HOME = home;
    try {
      const env = { WEBAGENTS_PROFILE: 'local', OTHER_TOKEN: 'from-process' };
      expect(envFromDotenv(['PROBE_API_KEY', 'PROFILE_SECRET', 'OTHER_TOKEN', 'MISSING'], cwd, env)).toEqual({
        PROBE_API_KEY: 'from-project',
        PROFILE_SECRET: 'from-profile',
      });
      expect(envFromDotenv([], cwd, env)).toEqual({});
      expect(envFromDotenv(['probe_api_key'], cwd, env)).toEqual({ probe_api_key: 'from-project' });
    } finally {
      if (savedHome === undefined) delete process.env.HOME;
      else process.env.HOME = savedHome;
      void os;
    }
  });
});
