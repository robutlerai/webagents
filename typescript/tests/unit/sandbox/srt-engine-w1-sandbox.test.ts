/**
 * The srt engine's contract (gap-closure plan item 1.2, 2026-09-26), against
 * the shared fixture `python/tests/fixtures/sandbox/srt.json` that the Python
 * suite reads too (`python/tests/sandbox/test_srt_engine_w1_sandbox.py`): the
 * settings a declaration becomes (validated against srt's own zod schema, which
 * strips unknown keys silently), the environment srt runs with, the private
 * settings file, timeouts, the `network:` list against a real local server,
 * failing closed on a missing or wrong-version srt, the strict schema (S-270)
 * and the owner-only rule for unconfined commands.
 *
 * Enforcement tests run srt for real and skip, with the reason, where it
 * cannot run (Linux without bubblewrap, socat and ripgrep).
 */

import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  CREDENTIAL_DIRS,
  DROPPED_ENV,
  ESCALATION_DENY,
  ENV_CLI,
  INERT_SANDBOX_FIELDS,
  LINUX_DEPS,
  PRESETS,
  SANDBOX_KEYS,
  SECRET_NAME_PARTS,
  SRT_PACKAGE,
  SRT_PATH,
  SRT_VERSION,
  SandboxDeclarationError,
  SandboxUnavailable,
  backendStatus,
  buildSettings,
  checkNetworkEntry,
  denyReads,
  parseSandboxDeclaration,
  policyFromDeclaration,
  resetBackendStatus,
  runSandboxed,
  sandboxRequiredReason,
  settingsBase,
  srtEnvironment,
  wrappedCommand,
  writeSettings,
} from '../../../src/sandbox/index';
import { ShellSkill, UNRESTRICTED_WARNING } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8'));
const tempDir = tempDirs();

const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`srt enforcement tests skipped: ${status.reason}`);

function policy(declared: Record<string, unknown>, cwd: string, tmpdir?: string) {
  return policyFromDeclaration(parseSandboxDeclaration(declared), { cwd, tmpdir });
}

function tree(): { work: string; outside: string } {
  const base = tempDir('wa-srt-');
  const work = path.join(base, 'work');
  const outside = path.join(base, 'outside');
  fs.mkdirSync(work);
  fs.mkdirSync(outside);
  fs.writeFileSync(path.join(outside, 'token'), 'TOP-SECRET\n');
  return { work: fs.realpathSync(work), outside: fs.realpathSync(outside) };
}

describe('the fixture is the contract', () => {
  it('pins the engine', () => {
    expect(SRT_PACKAGE).toBe(FIXTURE.engine.package);
    expect(SRT_VERSION).toBe(FIXTURE.engine.version);
    expect(SRT_PATH).toBe(FIXTURE.engine.srt_path);
    expect([...LINUX_DEPS]).toEqual(FIXTURE.engine.linux_deps);
  });

  it('has the schema keys, presets, escalation set and credential folders', () => {
    expect([...SANDBOX_KEYS]).toEqual(FIXTURE.schema.keys);
    expect(Object.keys(PRESETS).sort()).toEqual([...FIXTURE.schema.presets].sort());
    expect(parseSandboxDeclaration({}).preset).toBe(FIXTURE.schema.default_preset);
    expect([...INERT_SANDBOX_FIELDS]).toEqual(FIXTURE.schema.inert);
    expect([...ESCALATION_DENY]).toEqual(FIXTURE.escalation_deny);
    expect([...CREDENTIAL_DIRS]).toEqual(FIXTURE.credential_dirs_unreadable_under_development);
    for (const [name, preset] of Object.entries(FIXTURE.presets as Record<string, { confined: boolean }>)) {
      expect(PRESETS[name].confined).toBe(preset.confined);
    }
  });

  it('runs srt with the environment the fixture describes', () => {
    expect([...SECRET_NAME_PARTS]).toEqual(FIXTURE.env.secret_name_parts);
    expect([...DROPPED_ENV]).toEqual(FIXTURE.env.dropped_from_srt);
    const given: Record<string, string> = { PATH: '/home/u/bin:/usr/bin', HOME: '/home/u', LANG: 'C' };
    for (const name of FIXTURE.env.dropped_from_srt as string[]) given[name.toLowerCase()] = 'x';
    const env = srtEnvironment(given, '/tmp/scratch');
    const dropped = new Set((FIXTURE.env.dropped_from_srt as string[]).map((n) => n.toUpperCase()));
    for (const name of Object.keys(env)) if (name !== 'CLAUDE_CODE_TMPDIR') expect(dropped.has(name.toUpperCase())).toBe(false);
    expect(env.PATH).toBe(FIXTURE.engine.srt_path);
    expect(env.CLAUDE_CODE_TMPDIR).toBe('/tmp/scratch');
    expect(env.TMPDIR).toBe('/tmp/scratch');
    expect(env.HOME).toBe('/home/u');
    expect(wrappedCommand('echo hi', '/home/u/bin:/usr/bin').startsWith('export PATH=/home/u/bin:/usr/bin\n')).toBe(true);
  });

  it('builds the settings of every fixture case, and srt accepts them as written', async () => {
    const base = tempDir('wa-srt-cases-');
    const home = path.join(base, 'home');
    const work = path.join(base, 'work');
    const cwd = path.join(base, 'cwd');
    for (const dir of [home, work, cwd]) fs.mkdirSync(dir);
    const savedHome = process.env.HOME;
    process.env.HOME = home;
    // srt's own schema: unknown keys are STRIPPED silently, so a settings
    // shape it would not honour must be caught here, not at run time.
    const srt = await import(/* @vite-ignore */ '@anthropic-ai/sandbox-runtime' as string);
    try {
      for (const kase of FIXTURE.settings_cases) {
        const declared = JSON.parse(JSON.stringify(kase.declared).replace(/\{work\}/g, fs.realpathSync(work)));
        const built = policy(declared, cwd, path.join(base, 'tmp'));
        const settings = buildSettings(built, {}) as Record<string, Record<string, unknown>>;
        const fill = (value: unknown): unknown =>
          JSON.parse(
            JSON.stringify(value)
              .replace(/\{work\}/g, fs.realpathSync(work))
              .replace(/\{scratch\}/g, built.scratch!)
              .replace(/\{home\}/g, fs.realpathSync(home))
              .replace(/\{cwd\}/g, fs.realpathSync(cwd)),
          );
        for (const [key, expected] of Object.entries(kase.expect as Record<string, unknown>)) {
          const [section, field] = key.split('.');
          const includes = field.endsWith('_includes');
          const actual = settings[section][includes ? field.slice(0, -'_includes'.length) : field] as unknown[];
          let wanted = fill(expected) as unknown[];
          // On Linux, bubblewrap binds concrete paths only, so `buildSettings`
          // denies writes to the paths that EXIST (the documented Linux rule,
          // fixture `agent_file_deny.linux`). The fixture lists the macOS set;
          // on a Linux runner the expected write-denies are the ones present.
          // Found 2026-09-28, the first time this suite ran on Linux (CI).
          if (includes && key === 'filesystem.denyWrite_includes' && process.platform === 'linux') {
            wanted = wanted.filter((item) => typeof item === 'string' && fs.existsSync(item));
          }
          if (includes) for (const item of wanted) expect(actual, `${kase.name}: ${key}`).toContain(item);
          else expect(actual, `${kase.name}: ${key}`).toEqual(wanted);
        }
        for (const key of FIXTURE.never_set as string[]) {
          expect(key in settings).toBe(false);
          expect(key in settings.network).toBe(false);
          expect(key in settings.filesystem).toBe(false);
        }
        const parsed = srt.SandboxRuntimeConfigSchema.safeParse(settings);
        expect(parsed.success, `${kase.name}: ${JSON.stringify(parsed.error?.issues)}`).toBe(true);
        // Nothing stripped: the shape srt keeps is the shape we wrote.
        expect(parsed.data.network.allowedDomains).toEqual(settings.network.allowedDomains);
        expect(parsed.data.filesystem.allowWrite).toEqual(settings.filesystem.allowWrite);
        expect(parsed.data.filesystem.denyWrite).toEqual(settings.filesystem.denyWrite);
        expect(parsed.data.filesystem.denyRead).toEqual(settings.filesystem.denyRead);
        if (settings.filesystem.allowRead) expect(parsed.data.filesystem.allowRead).toEqual(settings.filesystem.allowRead);
      }
    } finally {
      if (savedHome === undefined) delete process.env.HOME;
      else process.env.HOME = savedHome;
    }
  });

  it('refuses network entries that are not hosts, at load', () => {
    for (const kase of FIXTURE.network_rejected as Array<{ entry: string }>) {
      expect(() => checkNetworkEntry(kase.entry)).toThrow(/sandbox: network entry/);
    }
    expect(checkNetworkEntry('github.com')).toBe('github.com');
    expect(checkNetworkEntry('*.pypi.org')).toBe('*.pypi.org');
    expect(checkNetworkEntry('127.0.0.1:8080')).toBe('127.0.0.1:8080');
  });

  it('rejects a misspelled key with the fixture sentence (S-270)', () => {
    for (const kase of FIXTURE.unknown_key.cases as Array<{ declared: unknown; message: string }>) {
      expect(() => parseSandboxDeclaration(kase.declared)).toThrow(SandboxDeclarationError);
      expect(() => parseSandboxDeclaration(kase.declared)).toThrow(kase.message);
    }
  });

  it('says unrestricted is an opt-out, once, where the agent loads', () => {
    expect(UNRESTRICTED_WARNING).toBe(FIXTURE.unrestricted.warning);
    const warned: string[] = [];
    const original = console.warn;
    console.warn = (line: string) => {
      warned.push(String(line));
    };
    try {
      const skill = new ShellSkill({ baseDir: tempDir('wa-srt-unr-'), sandbox: { preset: 'unrestricted' } });
      expect(skill.policy?.confined).toBe(false);
    } finally {
      console.warn = original;
    }
    expect(warned).toContain(FIXTURE.unrestricted.warning);
  });
});

describe('the settings file is private', () => {
  it('is 0600 in a 0700 folder outside every write root', () => {
    const { work } = tree();
    const built = policy({ preset: 'development', allowed_folders: [work] }, work);
    const file = writeSettings(built);
    try {
      expect(fs.statSync(file).mode & 0o777).toBe(0o600);
      expect(fs.statSync(path.dirname(file)).mode & 0o777).toBe(0o700);
      const real = fs.realpathSync(file);
      for (const root of built.writeRoots) expect(real.startsWith(root.replace(/\/+$/, '') + '/')).toBe(false);
      expect(JSON.parse(fs.readFileSync(file, 'utf8')).filesystem.allowWrite).toEqual(built.writeRoots);
    } finally {
      fs.rmSync(path.dirname(file), { recursive: true, force: true });
    }
  });

  it('refuses when every place would be writable from inside', () => {
    const built = policy({ preset: 'development', allowed_folders: ['/'] }, tempDir('wa-srt-root-'));
    expect(() => settingsBase(built)).toThrow(SandboxUnavailable);
    expect(() => settingsBase(built)).toThrow(/no place for the settings file outside the writable folders/);
  });
});

describe('it fails closed', () => {
  afterEach(() => {
    delete process.env[ENV_CLI];
    resetBackendStatus();
  });

  it('reports a missing srt and runs nothing', async () => {
    process.env[ENV_CLI] = path.join(tempDir('wa-srt-missing-'), 'nowhere', 'cli.js');
    resetBackendStatus();
    const missing = backendStatus();
    expect(missing.available).toBe(false);
    expect(missing.reason).toContain(`${SRT_PACKAGE}@${SRT_VERSION} was not found`);
    expect(missing.reason).toContain(ENV_CLI);
    const cwd = tempDir('wa-srt-missing-cwd-');
    await expect(runSandboxed('echo hi', policy({}, cwd))).rejects.toThrow(/was not run/);
  });

  it('refuses a wrong version', () => {
    const pkg = path.join(tempDir('wa-srt-version-'), 'node_modules', '@anthropic-ai', 'sandbox-runtime');
    fs.mkdirSync(path.join(pkg, 'dist'), { recursive: true });
    fs.writeFileSync(path.join(pkg, 'dist', 'cli.js'), 'process.exit(0)\n');
    fs.writeFileSync(path.join(pkg, 'package.json'), JSON.stringify({ name: SRT_PACKAGE, version: '0.0.1' }));
    process.env[ENV_CLI] = path.join(pkg, 'dist', 'cli.js');
    resetBackendStatus();
    const wrong = backendStatus();
    expect(wrong.available).toBe(false);
    expect(wrong.reason).toContain(`is ${SRT_PACKAGE} 0.0.1; this SDK requires exactly ${SRT_VERSION}`);
  });

  it('says where srt came from when it is here', () => {
    if (!status.available) {
      expect(status.reason).not.toBe('');
      return;
    }
    expect(status.backend).toBe('srt');
    expect(status.version).toBe(SRT_VERSION);
    expect(status.found).toContain('cli.js from');
    expect(status.found).toContain('node from');
    expect(path.isAbsolute(status.node!) && path.isAbsolute(status.path!)).toBe(true);
  });
});

describe('a caller other than the owner runs commands only in a sandbox', () => {
  const owner = { auth: { authenticated: true, scope: 'owner', provider: 'local' } } as never;
  const stranger = { auth: { authenticated: true, scope: 'user' } } as never;
  const anonymous = { auth: { authenticated: false } } as never;

  it('matches the fixture reasons', () => {
    const reasons = FIXTURE.refusals.not_owner_reasons;
    expect(sandboxRequiredReason(null)).toBe(reasons.undeclared);
    expect(sandboxRequiredReason(policy({ preset: 'unrestricted' }, tempDir('wa-srt-unr2-')))).toBe(reasons.unrestricted);
  });

  it('serves a stranger confined by default, and refuses one when the agent opted out', async () => {
    // The sandbox is on by default (2026-09-27): a file with no `sandbox:`
    // confines, so a stranger is served confined where srt runs, and refused
    // with the engine's reason where it does not. The opt-outs refuse them.
    const dir = tempDir('wa-srt-owner-');
    const prefix = (FIXTURE.refusals.not_owner as string).split('{reason}')[0];
    const byDefault = new ShellSkill({ baseDir: dir, env: {} });
    expect(byDefault.sandboxOrigin).toBe('default');
    if (status.available) expect(await byDefault.runCommand({ command: 'echo hi', timeout: 20 }, stranger)).toContain('hi');
    else expect(await byDefault.runCommand({ command: 'echo hi' }, stranger)).toBe(prefix + `the sandbox is unavailable: ${status.reason}`);

    const off = new ShellSkill({ baseDir: dir, sandbox: 'off', env: {} });
    expect(off.sandboxOrigin).toBe('agent file');
    expect(await off.runCommand({ command: 'echo hi' }, stranger)).toBe(prefix + FIXTURE.refusals.not_owner_reasons.unrestricted);
    expect((await off.runCommand({ command: 'echo hi' }, anonymous)).startsWith(prefix)).toBe(true);
    expect(await off.runCommand({ command: 'echo hi' }, owner)).toContain('hi');
    // No context at all is code calling the skill directly: the process itself.
    expect(await off.runCommand({ command: 'echo hi' }, {} as never)).toContain('hi');

    const flag = new ShellSkill({ baseDir: dir, env: { WEBAGENTS_NO_SANDBOX: '1' } });
    expect(flag.sandboxOrigin).toBe('--no-sandbox');
    expect(await flag.runCommand({ command: 'echo hi' }, stranger)).toBe(prefix + FIXTURE.refusals.not_owner_reasons.unrestricted);
    expect(await flag.runCommand({ command: 'echo hi' }, owner)).toContain('hi');
  }, 60_000);

  forReal('serves a stranger when the agent declares a sandbox, confined', async () => {
    const { work, outside } = tree();
    const skill = new ShellSkill({ baseDir: work, sandbox: { preset: 'strict', allowed_folders: [work] } });
    expect(await skill.runCommand({ command: 'echo hi', timeout: 20 }, stranger)).toContain('hi');
    expect(await skill.runCommand({ command: `cat $(echo ${outside})/token`, timeout: 20 }, stranger)).not.toContain('TOP-SECRET');
  }, 60_000);
});

describe('the engine for real', () => {
  forReal('write_inside_ok and write_outside_denied', async () => {
    const { work, outside } = tree();
    const built = policy({ preset: 'strict', allowed_folders: [work] }, work);
    const inside = await runSandboxed(`echo ok > ${work}/f && cat ${work}/f`, built, { timeout: 20 });
    expect(inside.stdout.trim()).toBe('ok');
    const escaped = await runSandboxed(`echo pwned > ${outside}/escaped && echo WROTE`, built, { timeout: 20 });
    expect(escaped.stdout).not.toContain('WROTE');
    expect(fs.existsSync(path.join(outside, 'escaped'))).toBe(false);
  }, 60_000);

  forReal('deny_read_unreadable under development, through $(...)', async () => {
    const { work } = tree();
    const home = tempDir('wa-srt-home-');
    fs.mkdirSync(path.join(home, '.ssh'));
    fs.writeFileSync(path.join(home, '.ssh', 'id_test'), 'PRIVATE-KEY-CANARY\n');
    const savedHome = process.env.HOME;
    process.env.HOME = home;
    try {
      const built = policy({ preset: 'development', allowed_folders: [work] }, work);
      expect(denyReads(built)).toContain(path.join(fs.realpathSync(home), '.ssh'));
      const result = await runSandboxed(`cat $(echo ${home}/.ssh)/id_test`, built, { timeout: 20 });
      expect(result.stdout).not.toContain('PRIVATE-KEY-CANARY');
    } finally {
      if (savedHome === undefined) delete process.env.HOME;
      else process.env.HOME = savedHome;
    }
  }, 60_000);

  describe('the network list against a local server', () => {
    let server: http.Server;
    let port: number;
    beforeAll(async () => {
      server = http.createServer((_req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('HELLO-FROM-SITE\n');
      });
      await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
      port = (server.address() as AddressInfo).port;
    });
    afterAll(async () => {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    });

    forReal('is denied without an allowlist and reachable with one', async () => {
      const { work } = tree();
      const denied = policy({ preset: 'strict', allowed_folders: [work] }, work);
      const direct = await runSandboxed(`curl -sf -m 4 http://127.0.0.1:${port}/ && echo NET`, denied, { timeout: 30 });
      expect(direct.stdout).not.toContain('NET');
      // Through srt's proxy too (NO_PROXY sends loopback direct otherwise).
      const viaProxy = await runSandboxed(`curl -sf -m 4 --noproxy '' http://127.0.0.1:${port}/ && echo NET`, denied, { timeout: 30 });
      expect(viaProxy.stdout).not.toContain('NET');
      const allowed = policy({ preset: 'development', allowed_folders: [work], network: [`127.0.0.1:${port}`] }, work);
      const served = await runSandboxed(`curl -sf -m 4 --noproxy '' http://127.0.0.1:${port}/`, allowed, { timeout: 30 });
      expect(served.stdout.trim(), served.stderr).toBe('HELLO-FROM-SITE');
    }, 90_000);
  });

  forReal('secret_env_absent, passthrough present', async () => {
    const { work } = tree();
    const built = policy({ preset: 'strict', allowed_folders: [work], env_passthrough: ['GH_TOKEN'] }, work);
    const env = { ...process.env, OPENAI_API_KEY: 'sk-FAKE-canary', GH_TOKEN: 'ghp-FAKE-canary' };
    const result = await runSandboxed('echo "k=[$OPENAI_API_KEY] g=[$GH_TOKEN] s=[$SANDBOX_RUNTIME]"', built, { timeout: 20, env });
    expect(result.stdout).toContain('k=[]');
    expect(result.stdout).toContain('g=[ghp-FAKE-canary]');
    expect(result.stdout).toContain('s=[1]');
  }, 60_000);

  forReal('timeout_kills_tree and reports a timeout, not exit 0', async () => {
    const { work } = tree();
    const built = policy({ preset: 'development', allowed_folders: [work] }, work);
    const started = Date.now();
    const result = await runSandboxed('sleep 3139 & echo DONE; wait', built, { timeout: 2 });
    expect(result.timedOut).toBe(true);
    expect(Date.now() - started).toBeLessThan(15_000);
    await new Promise((resolve) => setTimeout(resolve, 500));
    let left = '';
    try {
      // `[s]leep`: Linux's pgrep matches the `sh -c` that runs it, whose own
      // command line holds the pattern; macOS's leaves its ancestors out.
      left = execSync('pgrep -f "[s]leep 3139"', { encoding: 'utf8' }).trim();
    } catch {
      left = '';
    }
    expect(left, `the command's children survived the timeout: ${left}`).toBe('');
    // `sleep` is not in the shell's allow-list; the timeout is what is under test.
    const skill = new ShellSkill({ baseDir: work, allowedCommands: ['sleep'], sandbox: { preset: 'development', allowed_folders: [work] } });
    expect(await skill.runCommand({ command: 'sleep 3140', timeout: 1 }, {} as never)).toBe('Command timed out after 1s');
  }, 60_000);

  forReal('removes the settings file after the command', async () => {
    const { work } = tree();
    const built = policy({ preset: 'development', allowed_folders: [work] }, work);
    const base = settingsBase(built);
    // OURS ONLY (the ptypass-fixes lane, 2026-09-27). Other test files run
    // srt at the same time (vitest runs files in parallel), and a command of
    // theirs keeps its settings folder for as long as it runs: the interrupt
    // tests hold one for seconds, which outlasted the short wait this used
    // to allow any new folder. A folder is this command's when its settings
    // name this test's own write root, which no other test uses.
    const ours = () =>
      fs.readdirSync(base).filter((name) => {
        if (!name.startsWith('webagents-srt-')) return false;
        try {
          return fs.readFileSync(path.join(base, name, 'settings.json'), 'utf8').includes(JSON.stringify(work));
        } catch {
          return false;
        }
      });
    expect(ours()).toEqual([]);
    await runSandboxed('true', built, { timeout: 20 });
    expect(ours()).toEqual([]);
  }, 60_000);

  forReal('cannot widen itself from inside (macOS)', async () => {
    if (process.platform !== 'darwin') return;
    const { work, outside } = tree();
    const built = policy({ preset: 'strict', allowed_folders: [work] }, work);
    const result = await runSandboxed(`/usr/bin/sandbox-exec -p '(version 1)(allow default)' -- cat ${outside}/token`, built, { timeout: 20 });
    expect(result.stdout).not.toContain('TOP-SECRET');
  }, 60_000);
});

describe('the docker skill parity waiver is recorded', () => {
  it('names why the TypeScript Docker skill stays code-only', () => {
    expect(FIXTURE.docker_skill.typescript_nameable).toBe(false);
    expect(FIXTURE.docker_skill.python_nameable).toBe(true);
    expect(FIXTURE.docker_skill.reason).toContain('code-only');
  });
});

// Unused-import guard for the environment helper: os is used by tempDirs indirectly.
void os;
