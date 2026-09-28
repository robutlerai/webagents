/**
 * The command list gates unconfined commands only (the ptypass-fixes lane,
 * 2026-09-27; fixture `python/tests/fixtures/sandbox/srt.json`
 * `shell_allowlist`). The Python twin is
 * `python/tests/skills/local/test_ptypass_fixes_shell_allowlist.py`.
 *
 * WHY. The real-terminal PTY pass found `sleep` and `mkdir` refused in both
 * SDKs, `python3` here only and `rg` in Python only, before the sandbox was
 * ever reached, while `docs/cli/sandbox.md` says the kernel is the boundary.
 * Confined, a command is now refused only for a name the agent file blocks;
 * unconfined, one list and one wording hold in both SDKs. The model's guide
 * says only what holds.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { backendStatus } from '../../../src/sandbox/index';
import { DEFAULT_ALLOWED, DEFAULT_BLOCKED, IS_BLOCKED, NOT_ALLOWED, ShellSkill } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8')).shell_allowlist;
const tempDir = tempDirs();
const forReal = backendStatus().available ? it : it.skip;
const OWNER = { auth: { authenticated: true, scope: 'owner', provider: 'local' } } as never;

type Case = { command: string; declared_blocked?: string[]; declared_allowed?: string[]; confined: string | null; unconfined: string | null };
type Gate = { _checkCommand(command: string, confined?: boolean): { allowed: boolean; reason: string }; shellGuide(ctx: unknown): string };

describe('one list and one wording', () => {
  it('matches the fixture', () => {
    expect(DEFAULT_ALLOWED).toEqual(FIXTURE.allowed);
    expect(DEFAULT_BLOCKED).toEqual(FIXTURE.blocked);
    expect(NOT_ALLOWED).toBe(FIXTURE.not_allowed);
    expect(IS_BLOCKED).toBe(FIXTURE.is_blocked);
  });

  for (const c of FIXTURE.cases as Case[]) {
    it(`gates ${c.command}`, () => {
      const skill = new ShellSkill({
        ...(c.declared_blocked ? { blocked_commands: c.declared_blocked } : {}),
        ...(c.declared_allowed ? { allowed_commands: c.declared_allowed } : {}),
      }) as unknown as Gate;
      for (const mode of ['confined', 'unconfined'] as const) {
        const expected = c[mode];
        expect(skill._checkCommand(c.command, mode === 'confined'), mode).toEqual(expected === null ? { allowed: true, reason: '' } : { allowed: false, reason: expected });
      }
    });
  }
});

describe('the tool and the guide', () => {
  it('refuses an unconfined command for its name, with the prefix', async () => {
    const skill = new ShellSkill({ baseDir: tempDir('wa-ptf-allow-off-'), sandbox: 'off' });
    expect(await skill.runCommand({ command: 'sleep 0' }, OWNER)).toBe(FIXTURE.prefix + FIXTURE.not_allowed.replace('{name}', 'sleep'));
  });

  it('tells the model the list only when it applies', () => {
    const confined = new ShellSkill({ baseDir: tempDir('wa-ptf-guide-'), blocked_commands: ['nc'] }) as unknown as Gate;
    const text = confined.shellGuide({});
    expect(text).toContain('none is refused for its name, except these, which the agent file blocks: nc');
    expect(text).not.toContain('the only ones that will execute');
    const off = new ShellSkill({ baseDir: tempDir('wa-ptf-guide-off-'), sandbox: 'off' }) as unknown as Gate;
    expect(off.shellGuide({})).toContain('Allowed commands (the only ones that will execute)');
  });

  forReal('runs a confined sleep and mkdir, and the kernel still holds', async () => {
    const work = fs.realpathSync(tempDir('wa-ptf-allow-'));
    const skill = new ShellSkill({ baseDir: work });
    const made = await skill.runCommand({ command: 'sleep 0.1 && mkdir made && echo MADE' }, OWNER);
    expect(made).toContain('MADE');
    expect(fs.statSync(path.join(work, 'made')).isDirectory()).toBe(true);
    const outside = path.join(path.dirname(work), `ptypass-fixes-outside-${path.basename(work)}`);
    const refused = await skill.runCommand({ command: `mkdir ${outside} && echo MADE` }, OWNER);
    expect(refused).not.toContain('MADE');
    expect(fs.existsSync(outside)).toBe(false);
  }, 120_000);
});
