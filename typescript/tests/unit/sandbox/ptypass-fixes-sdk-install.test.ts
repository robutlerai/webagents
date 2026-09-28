/**
 * S-316, pinned and proved (the ptypass-fixes lane, 2026-09-27). The Python
 * twin is `python/tests/sandbox/test_ptypass_fixes_sdk_install.py`; both read
 * `python/tests/fixtures/sandbox/srt.json` (`sdk_install_deny`).
 *
 * WHY. Under `development` the agent's folder is a write root. When the SDK
 * was installed there (a project-local `node_modules`), a confined command
 * could rewrite the SDK's own code and srt itself, which runs OUTSIDE the
 * sandbox for every command: the next command or the next `webagents` start
 * ran what was planted, unconfined. The install is now write-denied
 * whenever it lies inside a write root, and nothing else is: a command may
 * still write a different `node_modules` the project keeps.
 *
 * Held here: the rule case by case, where an npm or pnpm layout puts the
 * install (the outermost `node_modules`), where this process's install is,
 * the settings, the `install` line `doctor` and `sandbox setup` print, and,
 * with real srt, that a confined command cannot write this process's own
 * package (a source checkout here) while it can write another folder's
 * `node_modules`.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  backendStatus,
  buildSettings,
  defaultPolicy,
  installInsideCheck,
  installWriteDenies,
  parseSandboxDeclaration,
  policyFromDeclaration,
  runSandboxed,
  sdkInstallPaths,
  setupChecks,
} from '../../../src/sandbox/index';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8')).sdk_install_deny;
const PACKAGE_ROOT = fs.realpathSync(path.resolve(HERE, '../../..'));
const tempDir = tempDirs();
const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`ptypass-fixes S-316 real-srt test skipped: ${status.reason}`);

describe('the rule, case by case', () => {
  for (const c of FIXTURE.cases as Array<{ name: string; write_roots: string[]; installs: string[]; denied: string[] }>) {
    it(c.name, () => {
      expect(installWriteDenies(c.write_roots, c.installs)).toEqual(c.denied);
    });
  }
});

describe('where the install is', () => {
  it('is the outermost node_modules for a package installed with npm or pnpm', () => {
    const agent = fs.realpathSync(tempDir('wa-s316-layout-'));
    const npmRoot = path.join(agent, 'node_modules', 'webagents');
    const pnpmRoot = path.join(agent, 'pnpm', 'node_modules', '.pnpm', 'webagents@0.3.6', 'node_modules', 'webagents');
    for (const root of [npmRoot, pnpmRoot]) {
      fs.mkdirSync(path.join(root, 'dist', 'sandbox'), { recursive: true });
      fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'webagents' }));
    }
    expect(sdkInstallPaths(process.env, path.join(npmRoot, 'dist', 'sandbox', 'srt.js'))).toContain(path.join(agent, 'node_modules'));
    expect(sdkInstallPaths(process.env, path.join(pnpmRoot, 'dist', 'sandbox', 'srt.js'))).toContain(path.join(agent, 'pnpm', 'node_modules'));
  });

  it('ignores a package.json that is not webagents (a bundled portal file, say)', () => {
    const other = fs.realpathSync(tempDir('wa-s316-other-'));
    fs.mkdirSync(path.join(other, 'lib'), { recursive: true });
    fs.writeFileSync(path.join(other, 'package.json'), JSON.stringify({ name: 'portal' }));
    expect(sdkInstallPaths(process.env, path.join(other, 'lib', 'srt.js'))).not.toContain(other);
  });

  it('holds this process: its package root (a source checkout here) and the folder of the node running it', () => {
    const installs = sdkInstallPaths();
    expect(installs.some((p) => PACKAGE_ROOT === p || PACKAGE_ROOT.startsWith(p + '/'))).toBe(true);
    const nodeDir = path.dirname(fs.realpathSync(process.execPath));
    expect(installs.some((p) => nodeDir === p || nodeDir.startsWith(p + '/'))).toBe(true);
  });
});

describe('the settings and the install line', () => {
  it('deny the install only when the folder holds it', () => {
    expect((buildSettings(defaultPolicy({ cwd: PACKAGE_ROOT })).filesystem as { denyWrite: string[] }).denyWrite).toContain(PACKAGE_ROOT);
    const elsewhere = fs.realpathSync(tempDir('wa-s316-elsewhere-'));
    const denied = (buildSettings(defaultPolicy({ cwd: elsewhere })).filesystem as { denyWrite: string[] }).denyWrite;
    expect(denied.some((entry) => entry === PACKAGE_ROOT || entry.startsWith(PACKAGE_ROOT + '/'))).toBe(false);
  });

  it('say it in the fixture words, from doctor and from sandbox setup', () => {
    const report = FIXTURE.report as { name: string; status: string; detail: string; fix: string };
    const parent = path.dirname(PACKAGE_ROOT);
    const found = installInsideCheck([parent]);
    expect(found).toBeDefined();
    expect(found).toMatchObject({ name: report.name, status: report.status, fix: report.fix });
    const [head, tail] = report.detail.split('{where}');
    expect(found!.detail.startsWith(head) && found!.detail.endsWith(tail)).toBe(true);
    expect(found!.detail.slice(head.length, found!.detail.length - tail.length).split(', ')).toContain(path.basename(PACKAGE_ROOT));
    expect(installInsideCheck([fs.realpathSync(tempDir('wa-s316-none-'))])).toBeUndefined();
    expect(installInsideCheck([PACKAGE_ROOT])?.detail).toContain(`(${FIXTURE.report.is_the_folder})`);

    const before = process.cwd();
    try {
      process.chdir(PACKAGE_ROOT);
      const names = setupChecks().map((c) => c.name);
      expect(names.indexOf('install')).toBeGreaterThanOrEqual(0);
      expect(names.indexOf('install')).toBeLessThan(names.indexOf('confined'));
      process.chdir(fs.realpathSync(tempDir('wa-s316-setup-')));
      expect(setupChecks().map((c) => c.name)).not.toContain('install');
    } finally {
      process.chdir(before);
    }
  }, 60_000);
});

describe('doctor', () => {
  it('prints the install line after the sandbox line when the agent\'s commands may write the install', async () => {
    const { runChecks } = await import('../../../src/cli/doctor');
    const keep = { HOME: process.env.HOME, WEBAGENTS_SECRETS_BACKEND: process.env.WEBAGENTS_SECRETS_BACKEND, ROBUTLER_API_URL: process.env.ROBUTLER_API_URL, OPENAI_API_KEY: process.env.OPENAI_API_KEY };
    const before = process.cwd();
    const project = fs.realpathSync(tempDir('wa-s316-doctor-'));
    fs.writeFileSync(
      path.join(project, 'AGENT.md'),
      `---\nname: bot\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - shell\nsandbox:\n  files:\n    write: [".", ${JSON.stringify(PACKAGE_ROOT)}]\n---\nBody\n`,
    );
    const quiet = ['log', 'warn', 'error', 'info'].map((method) => {
      const original = (console as unknown as Record<string, (...args: unknown[]) => void>)[method];
      (console as unknown as Record<string, (...args: unknown[]) => void>)[method] = () => {};
      return () => {
        (console as unknown as Record<string, (...args: unknown[]) => void>)[method] = original;
      };
    });
    try {
      process.env.HOME = fs.realpathSync(tempDir('wa-s316-doctor-home-'));
      process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
      process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
      process.env.OPENAI_API_KEY = 'sk-dummy';
      process.chdir(project);
      const checks = await runChecks();
      const names = checks.map((c) => c.name);
      const install = checks.find((c) => c.name === 'install');
      expect(install).toMatchObject({ status: 'warn', fix: FIXTURE.report.fix });
      expect(install!.detail).toContain(PACKAGE_ROOT);
      expect(names.indexOf('install')).toBe(names.indexOf('sandbox') + 1);
    } finally {
      process.chdir(before);
      for (const restore of quiet) restore();
      for (const [name, value] of Object.entries(keep)) {
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
      }
    }
  }, 120_000);
});

describe('with real srt', () => {
  forReal('a confined command cannot write the SDK\'s own install, and can write another node_modules', async () => {
    const agent = fs.realpathSync(tempDir('wa-s316-agent-'));
    const policy = policyFromDeclaration(parseSandboxDeclaration({ files: { write: ['.', PACKAGE_ROOT] } }), { cwd: agent });
    const probes = [path.join(PACKAGE_ROOT, 'node_modules', '.ptypass-fixes-s316-probe'), path.join(PACKAGE_ROOT, '.ptypass-fixes-s316-probe')];
    try {
      for (const probe of probes) {
        const result = await runSandboxed(`echo planted > ${probe} && echo WROTE`, policy, { timeout: 60 });
        expect(`${result.stdout}${result.stderr}`).not.toContain('WROTE');
        expect(fs.existsSync(probe)).toBe(false);
      }
      const other = await runSandboxed('mkdir -p node_modules/lib && echo x > node_modules/lib/ok.js && echo WROTE', policy, { timeout: 60 });
      expect(other.stdout).toContain('WROTE');
      expect(fs.existsSync(path.join(agent, 'node_modules', 'lib', 'ok.js'))).toBe(true);
    } finally {
      for (const probe of probes) fs.rmSync(probe, { force: true });
    }
  }, 120_000);
});
