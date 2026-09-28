/**
 * What a CLI says on stderr about an MCP server that did not load or connect,
 * or carries a literal that looks like a key (2026-09-26, the e2e run: the
 * Python `-p` was silent about MCP problems while this SDK said them). The
 * three sentences are `problem_lines` in
 * `python/tests/fixtures/mcp_tool/config_shapes.json`, which the Python suite
 * runs too (`tests/cli/test_prompt_mcp_problems_e2efix.py`): here the
 * templates, the lines a report yields, and the skill printing the `failed`
 * line itself for a server that cannot start.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { MCP_PROBLEM_LINES, mcpProblemLines } from '../../../src/skills/mcp/config';
import { MCPSkill } from '../../../src/skills/mcp/skill';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LINES = (JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/config_shapes.json'), 'utf8')) as {
  secret_refs: { problem_lines: { failed: string; rejected: string; warning: string; cases: { name: string; report: Report; lines: string[] }[] } };
}).secret_refs.problem_lines;
type Report = { name: string; rejected?: string; error?: string; warnings: string[] }[];

describe('the MCP problem lines', () => {
  it('the templates are the fixture’s', () => {
    expect(MCP_PROBLEM_LINES).toEqual({ failed: LINES.failed, rejected: LINES.rejected, warning: LINES.warning });
  });

  it.each(LINES.cases)('$name', ({ report, lines }) => {
    expect(mcpProblemLines(report)).toEqual(lines);
  });

  describe('the skill says a failed connection on the console', () => {
    let printed: string[] = [];
    beforeEach(() => {
      printed = [];
      for (const method of ['log', 'warn', 'error', 'info', 'debug'] as const) {
        vi.spyOn(console, method).mockImplementation((...args: unknown[]) => void printed.push(args.map(String).join(' ')));
      }
    });
    afterEach(() => vi.restoreAllMocks());

    it('prints the fixture’s failed line for a server that cannot start', async () => {
      const skill = new MCPSkill({ mcp: { ghost: { command: '/nonexistent/mcp-server' } } as never });
      const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
      await agent.initialize();
      try {
        const expected = mcpProblemLines(skill.serverReport());
        expect(expected).toHaveLength(1);
        expect(expected[0]).toMatch(/^\[MCPSkill\] Server "ghost" failed to connect: /);
        expect(printed).toContain(expected[0]);
      } finally {
        await agent.cleanup();
      }
    });
  });
});
