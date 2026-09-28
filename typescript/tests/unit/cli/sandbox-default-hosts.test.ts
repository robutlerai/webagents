/**
 * The agent-file line edit behind "always" in the chat's ask-on-first-use,
 * and the words it uses (the sandbox-default lane, 2026-09-27), against the
 * shared fixture `python/tests/fixtures/cli/sandbox_default_hosts.json`
 * that `python/tests/cli/test_sandbox_default_hosts.py` reads too. Every
 * edited file is read back through the loader, so the host really lands in
 * `sandbox.network.hosts`.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HOST_WORDS, addNetworkHost, hostAnswer } from '../../../src/cli/sandbox-default-hosts';
import { parseAgentMarkdown } from '../../../src/agents/index';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/sandbox_default_hosts.json'), 'utf8'));

describe('adding a host to the agent file', () => {
  it('says the fixture words and reads the fixture answers', () => {
    expect(HOST_WORDS).toEqual(FIXTURE.words);
    for (const [kind, answers] of Object.entries(FIXTURE.answers as Record<string, string[]>)) {
      for (const typed of answers) expect(hostAnswer(typed === 'anything else' ? 'xyz' : typed)).toBe(kind);
    }
  });

  it.each(FIXTURE.cases as Array<{ case: string; before: string; after?: string; problem?: string }>)('$case', (kase) => {
    const result = addNetworkHost(kase.before, FIXTURE.host);
    if (kase.after !== undefined) {
      expect(result).toEqual({ text: kase.after });
      const parsed = parseAgentMarkdown(kase.after, '/x/AGENT.md');
      expect(parsed.sandbox?.network.hosts).toContain(FIXTURE.host);
    } else {
      expect(result).toEqual({ problem: kase.problem });
    }
  });

  it('keeps CRLF files CRLF', () => {
    const before = '---\r\nname: a\r\n---\r\nBody.\r\n';
    const result = addNetworkHost(before, 'example.com');
    expect('text' in result && result.text.includes('\r\nsandbox:\r\n  network:\r\n    hosts:\r\n      - example.com\r\n---')).toBe(true);
  });
});
