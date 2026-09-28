/**
 * Small things the CLI e2e pass found (2026-09-28), in the TypeScript CLI,
 * against the shared fixture `cli/final_sdk_low_items.json` the Python suite
 * reads too: tool lines cut at a word boundary, the control-file diff's
 * order, `secrets set NAME VALUE`, and `-p`'s sign-in hint and its failures
 * in JSON.
 */

import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FAILURE_LINE_LIMIT, clipWords, toolResultSummary } from '../../../src/cli/render';
import { EXPIRED_HINT, forPrompt } from '../../../src/cli/failures';
import { unifiedDiff } from '../../../src/skills/filesystem/agent-secrets-guard';
import { CLI_SOURCE, TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/final_sdk_low_items.json'), 'utf8'),
) as {
  failure_line_limit: number;
  clip: Array<{ text: string; limit: number; clipped: string }>;
  secrets_value_as_argument: string;
  prompt_expired_hint: string;
  prompt_failure: { no_model_code: string; json_keys: string[]; stream_json_type: string };
};
const tempDir = tempDirs();

/** The CLI, run for real under a scratch HOME with no key and no sign-in. */
function cli(args: string[], cwd: string): { status: number | null; stdout: string; stderr: string } {
  const env: Record<string, string | undefined> = { ...process.env, HOME: tempDir('wa-final-sdk-low-home-'), WEBAGENTS_SECRETS_BACKEND: 'file', ROBUTLER_API_URL: 'http://127.0.0.1:9' };
  for (const name of ['WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'WEBAGENTS_AGENT_TOKEN', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY']) delete env[name];
  const out = spawnSync(process.execPath, [TSX_CLI, CLI_SOURCE, ...args], { cwd, env: env as NodeJS.ProcessEnv, encoding: 'utf-8', timeout: 60_000 });
  return { status: out.status, stdout: out.stdout, stderr: out.stderr };
}

describe('a tool line', () => {
  for (const c of FIXTURE.clip) {
    it(`is cut at a word boundary (${c.limit}: ${c.text.slice(0, 20)})`, () => {
      expect(clipWords(c.text, c.limit)).toBe(c.clipped);
    });
  }

  it("shows a file tool's refusal whole", () => {
    expect(FAILURE_LINE_LIMIT).toBe(FIXTURE.failure_line_limit);
    expect(toolResultSummary('read_file', FIXTURE.clip[2].text, true)).toBe(FIXTURE.clip[2].text);
  });
});

describe("a control file's diff", () => {
  it('shows the removed line before the one that replaces it', () => {
    const lines = unifiedDiff('a\nb\nc\n', 'a\nB\nc\n', 'x.md').split('\n');
    expect(lines.indexOf('-b')).toBeGreaterThan(-1);
    expect(lines.indexOf('-b')).toBeLessThan(lines.indexOf('+B'));
  });
});

describe('-p', () => {
  it('names the login command, not the chat command', () => {
    expect(forPrompt(EXPIRED_HINT)).toBe(FIXTURE.prompt_expired_hint);
    expect(forPrompt('something else')).toBe('something else');
  });

  it('answers a failure in JSON when asked for JSON', () => {
    const cwd = tempDir('wa-final-sdk-low-');
    fs.writeFileSync(path.join(cwd, 'AGENT.md'), '---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n');
    const json = cli(['-p', 'hi', '--output-format', 'json'], cwd);
    expect(json.status, json.stderr).toBe(1);
    const document = JSON.parse(json.stdout) as { error: Record<string, unknown> };
    expect(Object.keys(document.error)).toEqual(FIXTURE.prompt_failure.json_keys);
    expect(document.error.code).toBe(FIXTURE.prompt_failure.no_model_code);
    const stream = cli(['-p', 'hi', '--output-format', 'stream-json'], cwd);
    expect(stream.status).toBe(1);
    const line = JSON.parse(stream.stdout.trim().split('\n').at(-1)!) as { type: string; error: { code: string } };
    expect(line.type).toBe(FIXTURE.prompt_failure.stream_json_type);
    expect(line.error.code).toBe(FIXTURE.prompt_failure.no_model_code);
  }, 60_000);
});

describe('secrets set NAME VALUE', () => {
  it('says where values come from, and stores nothing', () => {
    const out = cli(['secrets', 'set', 'SOME_KEY', 'some-value-as-arg'], tempDir('wa-final-sdk-low-'));
    expect(out.status).toBe(1);
    expect(out.stderr.trim()).toBe(FIXTURE.secrets_value_as_argument);
  }, 60_000);
});
