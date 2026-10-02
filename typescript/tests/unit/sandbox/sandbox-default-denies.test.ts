/**
 * The sandbox-default lane's proofs against REAL srt (2026-09-27): the
 * built-in denies (S-311 keychain, S-315 profile folders, S-309 `.env` and
 * `.webagents/` in the agent's folder), the defaults an agent with no
 * `sandbox:` gets, `network.local`, `files.deny`, `env:` from `.env`, the
 * refusal hints, the refused-host capture, the ask-on-first-use loop, and
 * that everyday commands and the common tools still work. The Python twin
 * is `python/tests/sandbox/test_sandbox_default_denies.py`; the fixture
 * `python/tests/fixtures/sandbox/srt.json` (`builtin_denies`, `hints`,
 * `scenarios`) names what both prove.
 *
 * Every test skips, with the reason, where srt cannot run. The keychain
 * probe runs on macOS only, creates ONE item under a service name of its own
 * with this process's own keyring binding, reads it inside srt, expects
 * nothing back, and deletes it in a finally; it never reads, lists or
 * touches any other keychain item.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { execSync } from 'node:child_process';
import * as fs from 'node:fs';
import http from 'node:http';
import { createRequire } from 'node:module';
import type { AddressInfo } from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { REFUSAL_HINTS, backendStatus, defaultPolicy, parseSandboxDeclaration, policyFromDeclaration, runSandboxed } from '../../../src/sandbox/index';
import { ShellSkill, type HostAnswer } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8'));
const tempDir = tempDirs();
const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`sandbox-default real-srt tests skipped: ${status.reason}`);
const OWNER = { auth: { authenticated: true, scope: 'owner', provider: 'local' } } as never;

function policy(declared: unknown, cwd: string) {
  return policyFromDeclaration(parseSandboxDeclaration(declared), { cwd });
}

async function out(command: string, built: ReturnType<typeof policy>): Promise<string> {
  const result = await runSandboxed(command, built, { timeout: 60 });
  return `${result.stdout}\n${result.stderr}`;
}

function which(tool: string): boolean {
  try {
    execSync(`command -v ${tool}`, { stdio: 'ignore', shell: '/bin/sh' });
    return true;
  } catch {
    return false;
  }
}

describe('the built-in denies, for real', () => {
  forReal('keeps .env, .env.* and .webagents/ of the agent folder unreadable under development and strict, and .envelope readable', async () => {
    const work = fs.realpathSync(tempDir('wa-denies-work-'));
    fs.writeFileSync(path.join(work, '.env'), 'PROBE_SECRET_KEY=from-dotenv\n');
    fs.writeFileSync(path.join(work, '.env.local'), 'LOCAL=1\n');
    fs.writeFileSync(path.join(work, '.envelope'), 'not-env\n');
    fs.mkdirSync(path.join(work, '.webagents'));
    fs.writeFileSync(path.join(work, '.webagents', 'k.json'), 'KEYDATA\n');
    for (const built of [defaultPolicy({ cwd: work }), policy({ preset: 'strict' }, work)]) {
      expect(await out('cat .env', built)).not.toContain('from-dotenv');
      expect(await out(`cat $(echo ${work})/.env`, built)).not.toContain('from-dotenv');
      expect(await out('cat .env.local', built)).not.toContain('LOCAL=1');
      expect(await out('cat .webagents/k.json', built)).not.toContain('KEYDATA');
      expect(await out('cat .envelope', built)).toContain('not-env');
    }
  }, 120_000);

  forReal('keeps a profile folder and a files.deny path unreadable under development, through $(...)', async () => {
    const home = fs.realpathSync(tempDir('wa-denies-home-'));
    const work = fs.realpathSync(tempDir('wa-denies-work2-'));
    fs.mkdirSync(path.join(home, '.webagents-local'));
    fs.writeFileSync(path.join(home, '.webagents-local', 'history'), 'HIST-CANARY\n');
    fs.mkdirSync(path.join(home, 'notes'));
    fs.writeFileSync(path.join(home, 'notes', 'n.txt'), 'NOTE-CANARY\n');
    fs.writeFileSync(path.join(home, 'plain.txt'), 'PLAIN-OK\n');
    const savedHome = process.env.HOME;
    process.env.HOME = home;
    try {
      const built = policy({ preset: 'development', files: { deny: ['~/notes'] } }, work);
      expect(await out(`cat $(echo ${home})/.webagents-local/history`, built)).not.toContain('HIST-CANARY');
      expect(await out(`cat $(echo ${home})/notes/n.txt`, built)).not.toContain('NOTE-CANARY');
      // `development` still reads broadly: the rest of the home folder is readable.
      expect(await out(`cat $(echo ${home})/plain.txt`, built)).toContain('PLAIN-OK');
    } finally {
      if (savedHome === undefined) delete process.env.HOME;
      else process.env.HOME = savedHome;
    }
  }, 120_000);

  // S-343 (2026-09-29): the same probe with the first eleven credential
  // folders read every one of these. Fake files in a scratch HOME only.
  forReal('keeps the credential files outside the first list, and every .env under $HOME, unreadable under development (S-343)', async () => {
    const home = fs.realpathSync(tempDir('wa-denies-s343-home-'));
    const work = path.join(home, 'work', 'agent');
    const canaries: Record<string, string> = {
      '.config/gh/hosts.yml': 'GH-CANARY',
      '.git-credentials': 'GITCRED-CANARY',
      '.zsh_history': 'HISTORY-CANARY',
      '.codex/auth.json': 'CODEX-CANARY',
      '.config/op/config': 'OP-CANARY',
      'work/other-project/.env': 'OTHER-ENV-CANARY',
      'work/other-project/.env.production': 'OTHER-ENV-PROD-CANARY',
      'work/agent/packages/api/.env': 'NESTED-ENV-CANARY',
    };
    for (const [relative, canary] of Object.entries(canaries)) {
      fs.mkdirSync(path.dirname(path.join(home, relative)), { recursive: true });
      fs.writeFileSync(path.join(home, relative), `${canary}\n`);
    }
    fs.writeFileSync(path.join(home, 'work', 'other-project', 'README.md'), 'README-OK\n');
    const savedHome = process.env.HOME;
    process.env.HOME = home;
    try {
      const built = defaultPolicy({ cwd: work });
      for (const [relative, canary] of Object.entries(canaries)) {
        expect(await out(`cat $(echo ${home})/${relative}`, built), relative).not.toContain(canary);
      }
      // The rest of another project stays readable: only its secrets are denied.
      expect(await out(`cat $(echo ${home})/work/other-project/README.md`, built)).toContain('README-OK');
    } finally {
      if (savedHome === undefined) delete process.env.HOME;
      else process.env.HOME = savedHome;
    }
  }, 180_000);

  // With HOME pointed at a folder with no keychain (a scratch HOME), macOS has
  // no default keychain, and the synchronous add below shows its "keychain
  // cannot be found" prompt and waits, where no test timeout can stop it (the
  // Python twin hung a suite on exactly that on 2026-09-27, the keychain-ux
  // lane). Skip rather than ask.
  const defaultKeychain = (() => {
    try {
      execSync('/usr/bin/security default-keychain -d user', { stdio: 'ignore', timeout: 10_000 });
      return true;
    } catch {
      return false;
    }
  })();
  const keychainProbe = process.platform === 'darwin' && status.available && defaultKeychain ? it : it.skip;
  keychainProbe('cannot read a keychain item this interpreter created (S-311), and no dialog is needed', async () => {
    const require = createRequire(import.meta.url);
    let keyringPath: string;
    let Entry: new (service: string, user: string) => { setPassword(v: string): void; getPassword(): string | null; deletePassword(): boolean };
    try {
      keyringPath = require.resolve('@napi-rs/keyring');
      ({ Entry } = require(keyringPath));
    } catch (error) {
      console.warn(`keychain probe skipped: @napi-rs/keyring cannot load: ${(error as Error).message}`);
      return;
    }
    const service = `webagents-sandbox-probe-${Date.now().toString(36)}`;
    const entry = new Entry(service, 'probe');
    try {
      entry.setPassword('PROBE-VALUE-42');
    } catch (error) {
      console.warn(`keychain probe skipped: no keystore here: ${(error as Error).message}`);
      return;
    }
    try {
      expect(entry.getPassword()).toBe('PROBE-VALUE-42');
      const work = fs.realpathSync(tempDir('wa-denies-keychain-'));
      const built = defaultPolicy({ cwd: work });
      const script = `const { Entry } = require(${JSON.stringify(keyringPath)}); let v = null; try { v = new Entry(${JSON.stringify(service)}, 'probe').getPassword(); } catch (e) { v = 'ERR ' + e.message; } console.log('INSIDE=' + JSON.stringify(v));`;
      const result = await runSandboxed(`${JSON.stringify(process.execPath)} -e ${JSON.stringify(script)}`, built, { timeout: 60 });
      expect(result.stdout).toContain('INSIDE=');
      expect(result.stdout).not.toContain('PROBE-VALUE-42');
    } finally {
      entry.deletePassword();
      expect(entry.getPassword()).toBeNull();
    }
  }, 120_000);
});

describe('the defaults and the switches, for real', () => {
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

  forReal('an agent with no sandbox block gets no network, no local servers, no listening, and writes only in its folder', async () => {
    const base = fs.realpathSync(tempDir('wa-denies-default-'));
    const work = path.join(base, 'work');
    const outside = path.join(base, 'outside');
    fs.mkdirSync(work);
    fs.mkdirSync(outside);
    const skill = new ShellSkill({ baseDir: work, env: {} });
    expect(skill.sandboxStateLine()).toBe('development (default)');
    // Spelled so the shell's own string check (no absolute path outside the folder) lets it through: the kernel is what refuses it.
    expect(await skill.runCommand({ command: `echo written > "$(dirname "$PWD")/outside/leak.txt"; echo DONE`, timeout: 30 }, OWNER)).toContain('DONE');
    expect(fs.existsSync(path.join(outside, 'leak.txt'))).toBe(false);
    expect(await skill.runCommand({ command: 'echo inside > here.txt && cat here.txt', timeout: 30 }, OWNER)).toContain('inside');
    const local = await skill.runCommand({ command: `curl -sS -m 4 http://127.0.0.1:${port}/`, timeout: 30 }, OWNER);
    expect(local).not.toContain('HELLO-FROM-SITE');
    expect(local).toContain(REFUSAL_HINTS.local);
    const listen = await skill.runCommand({ command: `node -e "require('net').createServer().listen(0, '127.0.0.1', function () { console.log('LISTEN' + '-OK'); process.exit(0) }).on('error', function (e) { console.log('ERR ' + e.code); process.exit(1) })"`, timeout: 30 }, OWNER);
    // On Linux the command has its own network namespace: it may listen
    // there, and nothing outside can reach the port. macOS refuses the bind.
    if (process.platform !== 'linux') expect(listen).not.toContain('LISTEN-OK');
  }, 180_000);

  forReal('network.local opens local servers and listening; the hint is not appended then', async () => {
    const work = fs.realpathSync(tempDir('wa-denies-local-'));
    const skill = new ShellSkill({ baseDir: work, sandbox: { network: { local: true } }, env: {} });
    // The retries: on Linux this goes through srt's proxy, whose bridge can still be starting under a loaded runner.
    const local = await skill.runCommand({ command: `curl -sS -m 4 --retry 3 --retry-connrefused --retry-delay 1 http://127.0.0.1:${port}/`, timeout: 30 }, OWNER);
    expect(local).toContain('HELLO-FROM-SITE');
    expect(local).not.toContain(REFUSAL_HINTS.local);
    const listen = await skill.runCommand({ command: `node -e "require('net').createServer().listen(0, '127.0.0.1', function () { console.log('LISTEN' + '-OK'); process.exit(0) })"`, timeout: 30 }, OWNER);
    expect(listen).toContain('LISTEN-OK');
  }, 120_000);

  forReal('a listed host is reached by curl, Python urllib, Node fetch, pip and npm through the proxy', async () => {
    const work = fs.realpathSync(tempDir('wa-denies-listed-'));
    const built = policy({ network: { hosts: [`127.0.0.1:${port}`] } }, work);
    expect(await out(`curl -sS -m 4 http://127.0.0.1:${port}/`, built)).toContain('HELLO-FROM-SITE');
    expect(await out(`node -e "fetch('http://127.0.0.1:${port}/').then(r => r.text()).then(t => console.log(t)).catch(e => console.log('ERR', e.cause || e))"`, built)).toContain('HELLO-FROM-SITE');
    if (which('python3')) {
      expect(await out(`python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:${port}/', timeout=4).read())"`, built)).toContain('HELLO-FROM-SITE');
      // pip reaches the index (our server answers text/plain, which pip skips by name) rather than a proxy error.
      const pip = await out(`python3 -m pip download --no-deps --index-url http://127.0.0.1:${port}/simple/ -d "$TMPDIR/pipdl" nonexistent-probe-pkg 2>&1 | tail -3`, built);
      if (!/No module named pip/.test(pip)) expect(pip).toMatch(/No matching distribution|Skipping page http:\/\/127\.0\.0\.1/);
    }
    if (which('npm')) {
      // npm reaches the registry (and complains about the body, which is not JSON) rather than failing on its cache or the network.
      const npm = await out(`npm view nonexistent-probe-pkg --registry http://127.0.0.1:${port}/ 2>&1 | tail -3`, built);
      expect(npm).toMatch(/HELLO-FROM-SITE|not valid JSON|invalid json/i);
    }
  }, 240_000);

  forReal('a listed env name comes from .env, which the command itself cannot read', async () => {
    const work = fs.realpathSync(tempDir('wa-denies-env-'));
    fs.writeFileSync(path.join(work, '.env'), 'PROBE_SECRET_KEY=from-dotenv\nPLAIN=plain-value\n');
    const built = policy({ env: ['PROBE_SECRET_KEY'] }, work);
    const shown = await out('echo "[$PROBE_SECRET_KEY] [$PLAIN]"; cat .env', built);
    expect(shown).toContain('[from-dotenv] []');
    expect(shown).not.toContain('plain-value');
    const skill = new ShellSkill({ baseDir: work, env: {} });
    expect(await skill.runCommand({ command: 'cat .env', timeout: 30 }, OWNER)).toContain(REFUSAL_HINTS.env);
  }, 120_000);

  forReal('everyday commands run under the defaults: git status, ls -la, rg, node', async () => {
    const work = fs.realpathSync(tempDir('wa-denies-everyday-'));
    execSync('git init -q . && git -c user.email=a@b -c user.name=a commit -q --allow-empty -m init', { cwd: work, stdio: 'ignore' });
    fs.writeFileSync(path.join(work, 'a.txt'), 'assert 1\n');
    const built = defaultPolicy({ cwd: work });
    expect(await out('git status --short && echo GIT-OK', built)).toContain('GIT-OK');
    expect(await out('ls -la', built)).toContain('a.txt');
    if (which('rg')) expect(await out('rg -n assert a.txt', built)).toContain('assert 1');
    expect(await out('node -e "console.log(21*2)"', built)).toContain('42');
  }, 120_000);

  forReal('an unlisted host gets the hosts hint, the refused host comes from srt, and the ask loop re-runs with it', async () => {
    const work = fs.realpathSync(tempDir('wa-denies-ask-'));
    const asked: Array<{ host: string; command: string }> = [];
    const written: string[] = [];
    let answer: HostAnswer = 'no';
    const asker = {
      askHost: async (question: { host: string; command: string }) => {
        asked.push(question);
        return answer;
      },
      allowHostAlways: async (host: string) => {
        written.push(host);
      },
    };
    const skill = new ShellSkill({ baseDir: work, env: {}, asker });
    // Through srt's proxy (`--noproxy ''`): a direct loopback socket is refused by the
    // kernel and never reaches the proxy, so it is a `local` refusal with no host to ask about.
    const command = `curl -sS -m 4 --noproxy '' http://127.0.0.1:${port}/`;
    // `no`: the output carries the hint, the host was read from srt's log, nothing written.
    const refused = await skill.runCommand({ command, timeout: 30 }, OWNER);
    expect(asked).toEqual([{ host: '127.0.0.1', command }]);
    expect(refused).not.toContain('HELLO-FROM-SITE');
    expect(refused).toContain(REFUSAL_HINTS.hosts);
    // `once`: re-run with the host for this run.
    answer = 'once';
    const served = await skill.runCommand({ command, timeout: 30 }, OWNER);
    expect(served).toContain('HELLO-FROM-SITE');
    expect(written).toEqual([]);
    // `always`: the file write hook runs first, then the re-run.
    answer = 'always';
    expect(await skill.runCommand({ command, timeout: 30 }, OWNER)).toContain('HELLO-FROM-SITE');
    expect(written).toEqual(['127.0.0.1']);
    // Nobody but the owner is asked.
    const before = asked.length;
    await skill.runCommand({ command, timeout: 30 }, { auth: { authenticated: true, scope: 'user' } } as never);
    expect(asked.length).toBe(before);
    void os;
  }, 240_000);
});
