/**
 * `webagents sandbox setup`, the refusal sentence and doctor's `sandbox` fix
 * (the sandbox-engine lane, 2026-09-27), against
 * `python/tests/fixtures/sandbox/sandbox_engine.json`, which the Python suite
 * (`tests/sandbox/sandbox_engine_setup_test.py`) reads too.
 *
 * The engine ships with the package, so what can still be missing is the
 * machine's: the Linux programs, named with this distribution's install
 * line; what a container must allow; WSL 2 on native Windows; a macOS TMPDIR
 * too long for srt's socket. `setupChecks` says which and runs a real
 * confined `true`; a refusal carries the same reason and fix, then the
 * opt-out. Platform cases the helpers cannot take as arguments are simulated
 * by redefining `process.platform`; the confined `true` runs for real where
 * srt can.
 */

import { afterEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  CLI_FROM_BUNDLED,
  ENV_CLI,
  ENV_NODE,
  FIX_CONTAINER,
  FIX_NAMESPACES,
  FIX_TMPDIR,
  FIX_WINDOWS,
  LINUX_PACKAGES,
  NODE_MINIMUM,
  SRT_PACKAGE,
  SRT_VERSION,
  UNAVAILABLE_TAIL,
  backendStatus,
  chooseNode,
  inContainer,
  linuxInstallFix,
  locateCli,
  namespaceRestriction,
  parseNodeVersion,
  resetBackendStatus,
  setupChecks,
  srtSocketProbe,
  tmpdirTooLong,
  unavailableFix,
  unavailableMessage,
} from '../../../src/sandbox/index';
import { programsReason } from '../../../src/sandbox/srt';
import { CHAT_WORDS } from '../../../src/cli/chat-words';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'sandbox/sandbox_engine.json'), 'utf8'));
const SRT_FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'sandbox/srt.json'), 'utf8'));
const CHAT_FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'cli/chat_edits.json'), 'utf8'));
const tmp = tempDirs();

const status = backendStatus();
const itWithSrt = status.available ? it : it.skip;

function fakeNode(dir: string, version: string): string {
  fs.mkdirSync(dir, { recursive: true });
  const node = path.join(dir, 'node');
  fs.writeFileSync(node, `#!/bin/sh\necho ${version}\n`, { mode: 0o755 });
  return node;
}

function asPlatform<T>(platform: string, run: () => T): T {
  const original = Object.getOwnPropertyDescriptor(process, 'platform')!;
  Object.defineProperty(process, 'platform', { value: platform });
  resetBackendStatus();
  try {
    return run();
  } finally {
    Object.defineProperty(process, 'platform', original);
    resetBackendStatus();
  }
}

afterEach(() => resetBackendStatus());

describe('the words', () => {
  it('ends a refusal with the check and the opt-out', () => {
    expect(UNAVAILABLE_TAIL).toBe(SRT_FIXTURE.refusals.unavailable_tail);
    expect(UNAVAILABLE_TAIL).toContain('webagents sandbox setup');
    expect(UNAVAILABLE_TAIL).not.toContain('npm install');
    expect(SRT_FIXTURE.refusals.unavailable).toBe(FIXTURE.unavailable.template.replace('{tail}', UNAVAILABLE_TAIL));
  });

  it('puts the reason, this machine\'s fix, then the tail', () => {
    const { reason, fix } = FIXTURE.unavailable.example;
    expect(unavailableMessage({ reason, fix })).toBe(FIXTURE.unavailable.template.replace('{reason}', reason).replace('{fix}', fix).replace('{tail}', UNAVAILABLE_TAIL));
    expect(unavailableMessage({ reason, fix: '' })).toBe(FIXTURE.unavailable.template_no_fix.replace('{reason}', reason).replace('{tail}', UNAVAILABLE_TAIL));
  });

  it('gives doctor and the chat the same fix line', () => {
    expect(unavailableFix({ fix: '`sudo apt-get install socat`' })).toBe(FIXTURE.doctor.fix.replace('{fix}', '`sudo apt-get install socat`'));
    expect(unavailableFix({})).toBe(FIXTURE.doctor.fix_no_fix);
    expect(CHAT_WORDS.sandboxFix).toBe(FIXTURE.doctor.fix);
    expect(CHAT_FIXTURE.words.sandboxFix).toBe(FIXTURE.doctor.fix);
  });

  it('names the missing Linux programs as the fixture does', () => {
    expect(programsReason(['bwrap', 'socat'])).toBe(FIXTURE.reasons.programs.replace('{programs}', 'bwrap, socat'));
    expect(LINUX_PACKAGES).toEqual(FIXTURE.install.packages);
  });
});

describe('what this machine needs', () => {
  it('writes this distribution\'s install line', () => {
    for (const kase of FIXTURE.install.cases as Array<{ os_release: string; missing: string[]; fix: string }>) {
      expect(linuxInstallFix(kase.os_release, kase.missing), kase.os_release).toBe(kase.fix);
    }
  });

  it('recognises a container', () => {
    const root = tmp('sandbox-engine-root-');
    expect(inContainer(root, {})).toBe(false);
    fs.writeFileSync(path.join(root, '.dockerenv'), '');
    expect(inContainer(root, {})).toBe(true);
    const k8s = tmp('sandbox-engine-k8s-');
    fs.mkdirSync(path.join(k8s, 'proc', '1'), { recursive: true });
    fs.writeFileSync(path.join(k8s, 'proc', '1', 'cgroup'), '0::/kubepods/besteffort/pod1234\n');
    expect(inContainer(k8s, {})).toBe(true);
    expect(inContainer(path.join(root, 'none'), { container: 'podman' })).toBe(true);
  });

  it('names the namespace restriction and its fix', () => {
    for (const check of FIXTURE.namespaces.checks as Array<{ file: string; restricts_when: string; fix: keyof typeof FIX_NAMESPACES }>) {
      const root = tmp('sandbox-engine-ns-');
      fs.mkdirSync(path.dirname(path.join(root, 'proc', 'sys', check.file)), { recursive: true });
      fs.writeFileSync(path.join(root, 'proc', 'sys', check.file), `${check.restricts_when}\n`);
      expect(namespaceRestriction(root)).toEqual({ file: `/proc/sys/${check.file}`, value: check.restricts_when, fix: FIXTURE.fixes[check.fix] });
    }
    expect(FIX_NAMESPACES).toEqual({ userns_apparmor: FIXTURE.fixes.userns_apparmor, userns_clone: FIXTURE.fixes.userns_clone, userns_max: FIXTURE.fixes.userns_max });
    expect(namespaceRestriction(tmp('sandbox-engine-none-'))).toBeUndefined();
  });

  it('catches a TMPDIR too long for srt\'s socket', () => {
    expect(tmpdirTooLong('/tmp/wa')).toBe(false);
    const deep = path.join(tmp('sandbox-engine-deep-'), 'x'.repeat(60));
    fs.mkdirSync(deep);
    expect(tmpdirTooLong(deep)).toBe(true);
    // The path srt builds, measured (the ptypass-fixes lane, 2026-09-27;
    // fixture `tmpdir`): the preflight folder as spelled, 5-digit pid.
    expect(srtSocketProbe('/tmp/wa')).toBe(`/tmp/wa/webagents-srt-preflight-XXXXXX/${FIXTURE.tmpdir.socket}`);
    for (const c of FIXTURE.tmpdir.cases as Array<{ name: string; tmpdir: string; too_long: boolean }>) {
      expect(tmpdirTooLong(c.tmpdir), c.name).toBe(c.too_long);
    }
    expect(FIX_TMPDIR).toBe(FIXTURE.fixes.tmpdir);
    expect(FIX_CONTAINER).toBe(FIXTURE.fixes.container);
  });
});

describe('the engine and its node', () => {
  it('reads node versions as the fixture does, with srt\'s own minimum', () => {
    for (const kase of FIXTURE.node.version_cases as Array<{ output: string; version: number[] | null; enough: boolean }>) {
      expect(parseNodeVersion(kase.output) ?? null).toEqual(kase.version);
    }
    expect(NODE_MINIMUM.join('.')).toBe(FIXTURE.node.minimum);
  });

  it('takes the dependency\'s cli.js unless WEBAGENTS_SRT_CLI names another', () => {
    const own = locateCli({});
    expect(own.ok && own.cliFrom).toBe(CLI_FROM_BUNDLED);
    expect(FIXTURE.engine.typescript.cli_from.bundled).toBe(CLI_FROM_BUNDLED);
    const missing = locateCli({ [ENV_CLI]: '/nowhere/cli.js' });
    expect(missing.ok).toBe(false);
    if (!missing.ok) {
      expect(missing.reason).toBe(FIXTURE.reasons.engine_env_missing.replace('{package}', SRT_PACKAGE).replace('{version}', SRT_VERSION).replace('{value}', '/nowhere/cli.js'));
      expect(missing.fix).toBe(FIXTURE.fixes.engine_env.replace('{version}', SRT_VERSION));
    }
  });

  it('refuses an explicit node older than srt\'s minimum, with the fixture\'s words', () => {
    const old = fakeNode(tmp('sandbox-engine-node-'), 'v18.19.1');
    const chosen = chooseNode({ [ENV_NODE]: old });
    expect(chosen.ok).toBe(false);
    if (!chosen.ok) {
      const detail = FIXTURE.reasons.node_too_old.replace('{path}', old).replace('{found}', 'v18.19.1').replace('{minimum}', FIXTURE.node.minimum);
      expect(chosen.reason).toBe(FIXTURE.reasons.node_env.replace('{detail}', detail));
      expect(chosen.fix).toBe(FIXTURE.fixes.node_env.replace('{minimum}', FIXTURE.node.minimum));
    }
    const recent = fakeNode(tmp('sandbox-engine-node-'), 'v24.19.0');
    const ok = chooseNode({ [ENV_NODE]: recent });
    expect(ok.ok && ok.nodeFrom).toBe(ENV_NODE);
  });
});

describe('the refusal on each machine', () => {
  it('points native Windows at WSL 2', () => {
    asPlatform('win32', () => {
      const refused = backendStatus();
      expect([refused.available, refused.reason, refused.fix]).toEqual([false, FIXTURE.reasons.windows, FIXTURE.fixes.windows]);
      expect(FIX_WINDOWS).toBe(FIXTURE.fixes.windows);
      expect(unavailableMessage(refused).startsWith(`${FIXTURE.reasons.windows}: ${FIXTURE.fixes.windows}. `)).toBe(true);
      expect(setupChecks()).toEqual([
        { name: 'platform', status: 'fail', detail: FIXTURE.reasons.windows, fix: FIXTURE.fixes.windows },
        { name: 'confined', status: 'fail', detail: FIXTURE.setup.not_run },
      ]);
    });
  });

  it('says which other platform has none', () => {
    asPlatform('freebsd', () => {
      const refused = backendStatus();
      expect(refused.reason).toBe(FIXTURE.reasons.platform.replace('{platform}', 'freebsd'));
      expect(refused.fix).toBe(FIXTURE.fixes.platform);
    });
  });
});

describe('webagents sandbox setup', () => {
  it('lists its checks in the fixture\'s order and fails on a missing named install', () => {
    const previous = process.env[ENV_CLI];
    process.env[ENV_CLI] = '/nowhere/cli.js';
    try {
      const checks = setupChecks();
      const names = checks.map((c) => c.name);
      expect(names).toEqual((FIXTURE.setup.checks as string[]).filter((n) => names.includes(n)));
      expect(checks.find((c) => c.name === 'engine')?.status).toBe('fail');
      expect(checks[checks.length - 1]).toEqual({ name: 'confined', status: 'fail', detail: FIXTURE.setup.not_run });
    } finally {
      if (previous === undefined) delete process.env[ENV_CLI];
      else process.env[ENV_CLI] = previous;
    }
  });

  it('is declared in the CLI with the fixture\'s words', () => {
    const source = fs.readFileSync(path.resolve(HERE, '../../../src/cli/index.ts'), 'utf8');
    expect(source).toContain(`program.command('${FIXTURE.setup.group}').description('${FIXTURE.setup.group_description}')`);
    expect(source).toContain(`.command('${FIXTURE.setup.command}')\n  .description('${FIXTURE.setup.description}')`);
  });

  itWithSrt('runs a confined `true` here', () => {
    // From a folder that holds no part of the install: run from this
    // checkout, the SDK's own package root, `install` warns (S-316, the
    // ptypass-fixes lane, 2026-09-27).
    const before = process.cwd();
    process.chdir(tmp('wa-setup-here-'));
    let checks: ReturnType<typeof setupChecks>;
    try {
      checks = setupChecks();
    } finally {
      process.chdir(before);
    }
    expect(checks[checks.length - 1]).toEqual({ name: 'confined', status: 'ok', detail: FIXTURE.setup.ok.confined });
    expect(checks.find((c) => c.name === 'engine')?.detail).toBe(
      FIXTURE.setup.ok.engine.replace('{version}', SRT_VERSION).replace('{from}', FIXTURE.engine.typescript.cli_from.bundled),
    );
    expect(checks.every((c) => c.status === 'ok')).toBe(true);
  });
});
