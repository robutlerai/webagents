/**
 * Phase 5b: one AGENT.md parser for the whole package (2026-09-23).
 *
 * There were three, and the two that were not this one were line scanners
 * that disagreed with each other and with the documented format. Both got the
 * single most common declaration wrong -- a block-style `skills:` list -- and
 * both did so SILENTLY, producing an agent with no skills rather than an
 * error.
 *
 * These tests pin the shared parser's behaviour and then assert that the two
 * former offenders now route through it.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { parseAgentMarkdown } from '../../../src/agents/index.js';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const BLOCK_FORM = `---
name: ops
description: Handles the pager
namespace: ai.myorg
model: openai/gpt-4o-mini
skills:
  - memory
  - mcp
intents:
  - triage alerts
  - page the on-call
version: 2
---

Body instructions.
`;

describe('parseAgentMarkdown', () => {
  it('reads a block-style skills list', () => {
    // The old watcher matched \`^(\w+):\s*(.*)\$\`, so \`skills:\` matched with an
    // EMPTY value and produced [''] -- one unnamed skill, and the real two
    // gone. The old local-dev parser split on commas and produced [].
    expect(parseAgentMarkdown(BLOCK_FORM).skills).toEqual(['memory', 'mcp']);
  });

  it('reads every modelled key, not just four', () => {
    const parsed = parseAgentMarkdown(BLOCK_FORM);
    expect(parsed.name).toBe('ops');
    expect(parsed.description).toBe('Handles the pager');
    expect(parsed.model).toBe('openai/gpt-4o-mini');
    expect(parsed.namespace).toBe('ai.myorg');
    expect(parsed.intents).toEqual(['triage alerts', 'page the on-call']);
  });

  it('keeps unmodelled keys instead of dropping them', () => {
    expect(parseAgentMarkdown(BLOCK_FORM).extra).toEqual({ version: 2 });
  });

  it('keeps a valid cron block as written for the daemon to check (plan item 1.7)', () => {
    // Modelled since 2026-09-26, so it no longer lands in `extra`; the loader
    // runs `parseCronBlock` (D7), so a valid list is kept and a bad one is
    // refused at parse.
    const withCron = BLOCK_FORM.replace(
      'version: 2\n',
      'version: 2\ncron:\n  - name: daily\n    schedule: "0 9 * * 1-5"\n    prompt: Report.\n    deliver:\n      chat: owner\n',
    );
    const parsed = parseAgentMarkdown(withCron);
    expect(Array.isArray(parsed.cron)).toBe(true);
    expect((parsed.cron as Array<{ name: string }>)[0].name).toBe('daily');
  });

  it('refuses the old string cron form at parse (D7, 2026-09-26)', () => {
    expect(() => parseAgentMarkdown('---\nname: a\ncron: "0 9 * * *"\n---\n', '/x/AGENT.md')).toThrow(
      'cron: must be a list of schedules',
    );
  });

  it('keeps a dict-form skill entry whole, and also exposes its name', () => {
    const parsed = parseAgentMarkdown(
      '---\nname: a\nskills:\n  - mcp:\n      servers: [fs]\n  - memory\n---\n\nBody.\n',
    );
    expect(parsed.skills).toEqual(['mcp', 'memory']);
    expect(parsed.skillEntries[0]).toEqual({ mcp: { servers: ['fs'] } });
  });

  it('infers the name from the filename the way the Python loader does', () => {
    expect(parseAgentMarkdown('No frontmatter.', '/x/AGENT.md').name).toBe('default');
    expect(parseAgentMarkdown('No frontmatter.', '/x/AGENT-planner.md').name).toBe('planner');
  });

  it('stays "unknown" when no filename is offered', () => {
    expect(parseAgentMarkdown('No frontmatter.').name).toBe('unknown');
  });

  it('refuses broken YAML front matter, naming the file (D7, 2026-09-26)', () => {
    // The loader used to fall back to the whole file; it now refuses invalid
    // YAML as the Python loader does, so a file means one thing to both SDKs.
    expect(() => parseAgentMarkdown('---\nname: [unclosed\n---\n\nBody.\n', '/x/AGENT.md')).toThrow(
      'The front matter of /x/AGENT.md is not valid YAML',
    );
  });
});

describe('the daemon watcher uses it', () => {
  it('registers the skills a block-style file declares', async () => {
    const dir = tempDir('wa-watcher-');
    fs.writeFileSync(path.join(dir, 'AGENT.md'), BLOCK_FORM);

    const { AgentWatcher } = await import('../../../src/daemon/watcher.js');
    const watcher = new AgentWatcher(dir);
    watcher.start();
    const agents = watcher.getAgents();
    watcher.stop();

    expect(agents).toHaveLength(1);
    expect(agents[0].name).toBe('ops');
    expect(agents[0].skills).toEqual(['memory', 'mcp']);
    expect(agents[0].model).toBe('openai/gpt-4o-mini');
  });

  it('does not register the cross-vendor AGENTS.md as an agent', async () => {
    // Its pattern is case-insensitive and `.*` matches `S`, so the Agentic AI
    // Foundation's coding-agent file used to register as an agent named `S`.
    const dir = tempDir('wa-watcher-');
    fs.writeFileSync(path.join(dir, 'AGENTS.md'), '# Build with pnpm test\n');

    const { AgentWatcher } = await import('../../../src/daemon/watcher.js');
    const watcher = new AgentWatcher(dir);
    watcher.start();
    const agents = watcher.getAgents();
    watcher.stop();

    expect(agents).toEqual([]);
  });
});
