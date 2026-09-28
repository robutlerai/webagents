/**
 * `webagents acp` (gap-closure plan item 1.6, 2026-09-26): the command's
 * words, as the fixture and the Python CLI have them, and the action:
 * stdout is reserved before the agent file is read, the agent `serve` builds
 * is served through the ACP skill, and sessions go under the profile
 * directory unless the file names its own folder. The transcripts a real
 * client would see are `acp-stdio.test.ts`.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { acpAction, defaultSessionsDir } from '../../../src/cli/acp-action';
import { BaseAgent } from '../../../src/core/agent';
import type { IAgent } from '../../../src/core/types';
import { ACPTransportSkill } from '../../../src/skills/transport/acp/skill';
import { CLI_SOURCE, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROTOCOL = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/acp/acp_protocol.json'), 'utf8')) as {
  cli: { command: string; description: string; arguments: [string, string, string][] };
};

const tempDir = tempDirs();

describe('the words, as the fixture and the Python CLI have them', () => {
  it('declares the command, its description and its argument, and no option', () => {
    const cli = readFileSync(CLI_SOURCE, 'utf-8');
    const start = cli.indexOf(`  .command('${PROTOCOL.cli.command}')`);
    expect(start, 'the acp command moved; fix this scan').toBeGreaterThan(-1);
    const block = cli.slice(start, cli.indexOf('.action(', start));
    expect(block).toContain(`.description('${PROTOCOL.cli.description}')`);
    for (const [term, help, fallback] of PROTOCOL.cli.arguments) {
      expect(block).toContain(`.argument('${term}', '${help}', '${fallback}')`);
    }
    expect(block).not.toContain('.option(');
  });
});

describe('the action', () => {
  const originalLog = console.log;
  afterEach(() => {
    console.log = originalLog;
  });

  it('reserves stdout, serves the built agent through an ACP skill it adds, and keeps sessions under the profile', async () => {
    const home = tempDir('wa-acp-home-');
    const agent = new BaseAgent({ name: 'fixture', instructions: '' }) as unknown as IAgent;
    const seen: { skill?: ACPTransportSkill; agent?: IAgent; logs: string[] } = { logs: [] };
    const errors: unknown[][] = [];
    const originalError = console.error;
    console.error = (...args: unknown[]) => void errors.push(args);
    try {
      await acpAction('.', {
        createAgent: () => agent,
        sessionsDir: () => path.join(home, '.webagents', 'acp', 'sessions'),
        serve: async (skill, built) => {
          seen.skill = skill;
          seen.agent = built;
          console.log('a skill printing at startup');
        },
      });
    } finally {
      console.error = originalError;
    }
    expect(seen.agent).toBe(agent);
    expect(seen.skill).toBeInstanceOf(ACPTransportSkill);
    expect((agent as unknown as { skills: unknown[] }).skills).toContain(seen.skill);
    expect(seen.skill?.settings.sessions_dir).toBe(path.join(home, '.webagents', 'acp', 'sessions'));
    // `console.log` went to stderr: the wire is stdout, and nothing else may write to it.
    expect(errors).toEqual([['a skill printing at startup']]);
  });

  it('keeps the file’s own sessions_dir and the ACP skill the file names', async () => {
    const dir = tempDir('wa-acp-project-');
    writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: fixture\nskills:\n  - acp:\n      sessions_dir: /tmp/acp-here\n---\n\nBody.\n');
    const seen: { skill?: ACPTransportSkill } = {};
    await acpAction(dir, { serve: async (skill) => void (seen.skill = skill) });
    expect(seen.skill?.settings.sessions_dir).toBe('/tmp/acp-here');
    // A real served agent is built here (the file's skills, the stored keys),
    // which is slow under a parallel run; the mcp-serve tests allow the same.
  }, 60_000);

  it('defaults sessions to the profile directory', async () => {
    expect(await defaultSessionsDir()).toMatch(/[\\/]\.webagents(-[^\\/]+)?[\\/]acp[\\/]sessions$/);
  });
});
