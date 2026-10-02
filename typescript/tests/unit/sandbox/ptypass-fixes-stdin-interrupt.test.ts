/**
 * The stdin rule and the interrupt, pinned (the ptypass-fixes lane,
 * 2026-09-27). The Python twin is
 * `python/tests/sandbox/test_ptypass_fixes_stdin_interrupt.py`; both read
 * `python/tests/fixtures/sandbox/srt.json` (`interrupt`, scenarios
 * `stdin_is_devnull` and `interrupt_kills_tree`).
 *
 * WHY. The real-terminal PTY pass found that the Python chat handed every
 * command the owner's terminal as its stdin (S-317): a confined command read
 * what the owner typed and drew on the screen. This SDK always passed
 * `stdio: ['ignore', ...]`; the first test keeps it that way, behaviourally:
 * a child process whose own stdin is a pipe full of text runs `cat` through
 * `runSandboxed`, and `cat` must read nothing. The pass also found that Esc
 * stopped the reply while the command ran on for 13 to 18 seconds; the rest
 * hold that an aborted signal (the turn's, through `context.signal`) kills
 * the command's whole process group, confined and not, in the shell tool
 * too, and that the tool then answers `interrupt.result`.
 *
 * The confined cases need real srt and skip, with the reason, where it
 * cannot run.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { INTERRUPTED_RESULT, backendStatus, defaultPolicy, parseSandboxDeclaration, policyFromDeclaration, runSandboxed } from '../../../src/sandbox/index';
import { ShellSkill } from '../../../src/skills/shell/skill';
import { SDK_TSCONFIG, TSX_CLI, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8'));
const SANDBOX_INDEX = path.resolve(HERE, '../../../src/sandbox/index.ts');
const tempDir = tempDirs();
const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`ptypass-fixes real-srt tests skipped: ${status.reason}`);
const withTsx = TSX_PROBLEM === null ? it : it.skip;
const forRealWithTsx = status.available && TSX_PROBLEM === null ? it : it.skip;
const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };
const CANARY = 'CANARY-TYPED-BY-THE-OWNER';

/** A command that writes its own pid, starts a child that writes its pid, then sleeps: the interrupt must take both. */
const SLEEPER =
  `node -e "const fs=require('fs');const cp=require('child_process');fs.writeFileSync('parent.pid',String(process.pid));` +
  `cp.spawn(process.execPath,['-e','require(\\'fs\\').writeFileSync(\\'child.pid\\',String(process.pid));setTimeout(()=>{},60000)'],{stdio:'ignore'});` +
  `setTimeout(()=>{},60000)"`;

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

async function waitForFile(file: string, ms = 30_000): Promise<number> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (fs.existsSync(file)) {
      const text = fs.readFileSync(file, 'utf8').trim();
      if (text) return Number(text);
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`${file} never appeared`);
}

async function goneWithin(pid: number, ms: number): Promise<boolean> {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (!alive(pid)) return true;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return !alive(pid);
}

function policy(declared: unknown, cwd: string) {
  return policyFromDeclaration(parseSandboxDeclaration(declared), { cwd });
}

/** `cat` through `runSandboxed`, in a child process whose own stdin is a pipe holding CANARY. */
function catInAChildWithTextOnStdin(work: string, declared: unknown): string {
  const script = path.join(work, 'ptypass-fixes-stdin-probe.ts');
  fs.writeFileSync(
    script,
    [
      `import { parseSandboxDeclaration, policyFromDeclaration, runSandboxed } from ${JSON.stringify(SANDBOX_INDEX)};`,
      `const policy = policyFromDeclaration(parseSandboxDeclaration(${JSON.stringify(declared)}), { cwd: ${JSON.stringify(work)} });`,
      `runSandboxed('cat; echo END-OF-CAT', policy, { timeout: 60 }).then((r) => console.log(JSON.stringify(r)));`,
    ].join('\n'),
  );
  const done = spawnSync(process.execPath, [TSX_CLI, '--tsconfig', SDK_TSCONFIG, script], {
    cwd: work,
    input: `${CANARY}\n`,
    encoding: 'utf8',
    timeout: 120_000,
  });
  expect(done.status, done.stderr).toBe(0);
  const lines = done.stdout.trim().split('\n');
  return (JSON.parse(lines[lines.length - 1]) as { stdout: string }).stdout;
}

describe('the fixture', () => {
  it('names the sentence, the stdin rule and the scenarios', () => {
    expect(FIXTURE.interrupt.result).toBe(INTERRUPTED_RESULT);
    expect(FIXTURE.interrupt.stdin.startsWith('/dev/null')).toBe(true);
    const names = (FIXTURE.scenarios as Array<{ name: string }>).map((s) => s.name);
    expect(names).toEqual(expect.arrayContaining(['stdin_is_devnull', 'interrupt_kills_tree']));
  });
});

describe('stdin is /dev/null (S-317 twin)', () => {
  forRealWithTsx('a confined command reads EOF, not its caller\'s stdin', () => {
    const out = catInAChildWithTextOnStdin(fs.realpathSync(tempDir('wa-ptypass-stdin-')), {});
    expect(out).toContain('END-OF-CAT');
    expect(out).not.toContain(CANARY);
  }, 150_000);

  withTsx('an unconfined command reads EOF, not its caller\'s stdin', () => {
    const out = catInAChildWithTextOnStdin(fs.realpathSync(tempDir('wa-ptypass-stdin-off-')), 'off');
    expect(out).toContain('END-OF-CAT');
    expect(out).not.toContain(CANARY);
  }, 150_000);
});

/**
 * The pid this test can signal for a pid a confined command wrote. On Linux
 * srt gives the command its own pid namespace, so the number in the file is
 * the pid INSIDE it (2, 3): the process is found from outside by its
 * namespace pids and its command line. Elsewhere the pid is the pid.
 */
function outsidePid(inside: number, marker: string): number {
  if (process.platform !== 'linux') return inside;
  for (const entry of fs.readdirSync('/proc')) {
    if (!/^\d+$/.test(entry)) continue;
    try {
      const status = fs.readFileSync(`/proc/${entry}/status`, 'utf8');
      const cmdline = fs.readFileSync(`/proc/${entry}/cmdline`, 'utf8').replace(/\0/g, ' ');
      const pids = (status.split('\n').find((line) => line.startsWith('NSpid:')) ?? '').split(/\s+/).slice(1).filter(Boolean);
      if (pids.length > 1 && pids[pids.length - 1] === String(inside) && cmdline.includes(marker)) return Number(entry);
    } catch {
      // The process went away while it was read.
    }
  }
  return inside;
}

async function interruptRun(work: string, built: ReturnType<typeof policy>): Promise<number> {
  const controller = new AbortController();
  const running = runSandboxed(SLEEPER, built, { timeout: 120, signal: controller.signal });
  const parent = outsidePid(await waitForFile(path.join(work, 'parent.pid')), 'parent.pid');
  const child = outsidePid(await waitForFile(path.join(work, 'child.pid')), 'child.pid');
  expect(alive(parent) && alive(child)).toBe(true);
  const started = Date.now();
  controller.abort();
  const result = await running;
  const took = Date.now() - started;
  expect(result.interrupted).toBe(true);
  expect(result.timedOut).toBe(false);
  expect(await goneWithin(parent, 3000)).toBe(true);
  expect(await goneWithin(child, 3000)).toBe(true);
  return took;
}

describe('the interrupt kills the tree', () => {
  forReal('confined', async () => {
    const work = fs.realpathSync(tempDir('wa-ptypass-int-'));
    expect(await interruptRun(work, defaultPolicy({ cwd: work }))).toBeLessThan(5000);
  }, 120_000);

  it('unconfined', async () => {
    const work = fs.realpathSync(tempDir('wa-ptypass-int-off-'));
    expect(await interruptRun(work, policy('off', work))).toBeLessThan(5000);
  }, 120_000);

  it('an already aborted signal runs nothing', async () => {
    const work = fs.realpathSync(tempDir('wa-ptypass-int-pre-'));
    const controller = new AbortController();
    controller.abort();
    const result = await runSandboxed('touch ran', policy('off', work), { timeout: 10, signal: controller.signal });
    expect(result.interrupted).toBe(true);
    expect(fs.existsSync(path.join(work, 'ran'))).toBe(false);
  });
});

async function cancelTheTool(work: string, skill: ShellSkill): Promise<void> {
  const controller = new AbortController();
  // What the chat's Esc and Ctrl+C do: abort the turn's signal, which the
  // agent hands every tool as `context.signal`.
  const answer = skill.runCommand({ command: SLEEPER, timeout: 120 }, { auth: OWNER, signal: controller.signal } as never);
  const parent = outsidePid(await waitForFile(path.join(work, 'parent.pid')), 'parent.pid');
  const child = outsidePid(await waitForFile(path.join(work, 'child.pid')), 'child.pid');
  controller.abort();
  expect(await answer).toBe(INTERRUPTED_RESULT);
  expect(await goneWithin(parent, 3000)).toBe(true);
  expect(await goneWithin(child, 3000)).toBe(true);
}

describe('the shell tool stops with its turn', () => {
  forReal('confined by default', async () => {
    const work = fs.realpathSync(tempDir('wa-ptypass-tool-'));
    await cancelTheTool(work, new ShellSkill({ baseDir: work }));
  }, 120_000);

  it('sandbox: off', async () => {
    const work = fs.realpathSync(tempDir('wa-ptypass-tool-off-'));
    await cancelTheTool(work, new ShellSkill({ baseDir: work, sandbox: 'off' }));
  }, 120_000);
});
