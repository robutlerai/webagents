/**
 * A `sandbox:` block in an agent file reaches the shell and is enforced
 * (2026-09-26, gap-closure plan item 1.2; S-248 addendum: this SDK used to
 * drop the block without a word). The loader keeps it as a modelled key,
 * refuses a mistyped key with the Python loader's sentence (S-270), and the
 * resolver hands it to `ShellSkill`, whose policy is what holds. The chat
 * and `doctor` say which engine, in the Python CLI's words.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgentFileError, parseAgentMarkdown } from '../../../src/agents/index';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { loadAgentProject } from '../../../src/cli/agent-project';
import { loadAgentConfigFile } from '../../../src/cli/serve-action';
import type { ShellSkill } from '../../../src/skills/shell/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sandbox/srt.json'), 'utf8'));
const tempDir = tempDirs();

const DECLARED = `---
name: boxed
skills:
  - shell
sandbox:
  preset: strict
  allowed_folders:
    - ./workspace
  network:
    - github.com
---

Body.
`;

describe('the loader models sandbox:', () => {
  it('keeps the block as a checked declaration, not in extra', () => {
    const parsed = parseAgentMarkdown(DECLARED, '/x/AGENT.md');
    // The normalised shape (2026-09-27): the flat keys fold into files and network.
    expect(parsed.sandbox).toEqual({
      preset: 'strict',
      files: { write: ['./workspace'], read: [], deny: [] },
      network: { hosts: ['github.com'], local: false, sockets: [] },
      env: [],
      allowed_commands: [],
      allowed_imports: [],
    });
    expect(parsed.extra).toEqual({});
  });

  it('leaves a file without one alone', () => {
    const parsed = parseAgentMarkdown('---\nname: a\nskills:\n  - shell\n---\nBody.\n', '/x/AGENT.md');
    expect(parsed.sandbox).toBeUndefined();
  });

  it('refuses a misspelled key with the fixture sentence (S-270)', () => {
    for (const kase of FIXTURE.unknown_key.cases as Array<{ declared: Record<string, unknown>; message: string }>) {
      // A nested block (`files: {writes: [...]}`) is written as a YAML flow mapping.
      const lines = Object.entries(kase.declared).map(([key, value]) =>
        Array.isArray(value)
          ? `  ${key}:\n${value.map((v) => `    - ${v}`).join('\n')}`
          : value !== null && typeof value === 'object'
            ? `  ${key}: ${JSON.stringify(value)}`
            : `  ${key}: ${value}`,
      );
      const content = `---\nname: a\nskills:\n  - shell\nsandbox:\n${lines.join('\n')}\n---\nBody.\n`;
      expect(() => parseAgentMarkdown(content, '/x/AGENT.md')).toThrow(AgentFileError);
      expect(() => parseAgentMarkdown(content, '/x/AGENT.md')).toThrow(`/x/AGENT.md: ${kase.message}`);
    }
  });

  it('is carried by the project loader that serve and mcp serve use', () => {
    const dir = tempDir('wa-sandbox-project-');
    fs.writeFileSync(path.join(dir, 'AGENT.md'), DECLARED);
    const project = loadAgentProject(dir)!;
    expect(project.sandbox?.preset).toBe('strict');
    const config = loadAgentConfigFile(dir);
    expect((config.sandbox as { network: { hosts: string[] } }).network.hosts).toEqual(['github.com']);
  });
});

describe('the resolver hands it to the shell', () => {
  it('builds a ShellSkill whose policy holds the declaration', async () => {
    const dir = tempDir('wa-sandbox-resolve-');
    fs.mkdirSync(path.join(dir, 'workspace'));
    const parsed = parseAgentMarkdown(DECLARED, path.join(dir, 'AGENT.md'));
    const { byName, failed } = await resolveSkillsByName(parsed.skillEntries, { agentDir: dir, sandbox: parsed.sandbox });
    expect(failed).toEqual([]);
    const shell = byName.get('shell') as unknown as ShellSkill;
    expect(shell.policy?.preset).toBe('strict');
    expect(shell.policy?.scopedReads).toBe(true);
    expect(shell.policy?.networkDomains).toEqual(['github.com']);
    expect(shell.policy?.writeRoots).toContain(fs.realpathSync(path.join(dir, 'workspace')));
    expect(shell.sandboxError).toBeNull();
  });

  it('confines the shell by default when the file declares nothing', async () => {
    // On by default (2026-09-27): the defaults, from the agent's folder, no network.
    const dir = tempDir('wa-sandbox-none-');
    const { byName } = await resolveSkillsByName(['shell'], { agentDir: dir });
    const shell = byName.get('shell') as unknown as ShellSkill;
    expect(shell.sandboxOrigin).toBe('default');
    expect(shell.policy?.confined).toBe(true);
    expect(shell.policy?.preset).toBe('development');
    expect(shell.policy?.networkDomains).toEqual([]);
    expect(shell.policy?.writeRoots).toContain(fs.realpathSync(dir));
    expect(shell.sandboxStateLine()).toBe('development (default)');
  });

  it('refuses every command under a declaration it cannot resolve, rather than running free', async () => {
    const dir = tempDir('wa-sandbox-typo-');
    const { byName } = await resolveSkillsByName(['shell'], {
      agentDir: dir,
      sandbox: { preset: 'stirct', allowed_folders: ['.'], allowed_commands: [], allowed_imports: [], env_passthrough: [], network: [] },
    });
    const shell = byName.get('shell') as unknown as ShellSkill;
    expect(shell.sandboxError).toContain("unknown sandbox preset 'stirct'");
    expect(await shell.runCommand({ command: 'echo hi' }, {} as never)).toBe(
      `Access denied: invalid sandbox declaration: ${shell.sandboxError}`,
    );
  });
});
