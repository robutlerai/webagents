/**
 * CLI contract tests.
 *
 * WHY THIS FILE MOVED (2026-09-23). It lived at `tests/e2e/cli.test.ts` and ran
 * in NEITHER runner: `vitest.config.ts` excludes `tests/e2e/**`, and playwright
 * matches only `**\/*.spec.ts`. So the CLI's only tests had never executed, while
 * `vitest.config.ts` simultaneously excluded `src/cli/**` from coverage with the
 * comment "CLI tested via E2E". Both halves of that were false.
 *
 * What it asserted was also thin: two of its four "unit tests" checked that
 * `'/help'.startsWith('/')` and `'  x  '.trim()` behave as JavaScript specifies,
 * which cannot fail and says nothing about this CLI. Those are gone.
 *
 * What is here now pins the defects that were actually shipping, so they cannot
 * come back silently. The spawn-based checks still need a build and stay behind
 * RUN_E2E; everything else imports the modules directly and always runs.
 */

import { describe, it, expect } from 'vitest';
import { spawn } from 'child_process';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

import { LLM_PROVIDERS, findProvider, configuredProviders } from '../../../src/skills/llm/providers';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const here = dirname(fileURLToPath(import.meta.url));
const packageRoot = join(here, '..', '..', '..');

describe('provider registry', () => {
  it('every provider needing a key names the env var the skill really reads', () => {
    // The registry is the source of truth for the API-key preflight, so a
    // wrong env var here means the preflight lies about what to export.
    // Verified against each skill's own `process.env` read.
    const expected: Record<string, string> = {
      openai: 'OPENAI_API_KEY',
      anthropic: 'ANTHROPIC_API_KEY',
      google: 'GOOGLE_API_KEY',
      xai: 'XAI_API_KEY',
      fireworks: 'FIREWORKS_API_KEY',
    };
    for (const [id, envVar] of Object.entries(expected)) {
      expect(findProvider(id), `provider ${id} missing`).toBeDefined();
      expect(findProvider(id)!.envVar).toBe(envVar);
    }
  });

  it('resolves the aliases an agent file may use', () => {
    expect(findProvider('claude')?.id).toBe('anthropic');
    expect(findProvider('gemini')?.id).toBe('google');
    expect(findProvider('grok')?.id).toBe('xai');
    // Case-insensitive: agent files are hand-written.
    expect(findProvider('OpenAI')?.id).toBe('openai');
    expect(findProvider('nope')).toBeUndefined();
  });

  it('treats local providers as always available and keyed ones as env-gated', () => {
    const { available, missing } = configuredProviders({});
    expect(available.map((p) => p.id).sort()).toEqual(['transformers', 'webllm']);
    expect(missing.map((p) => p.id)).toContain('openai');

    const withKey = configuredProviders({ OPENAI_API_KEY: 'sk-test' });
    expect(withKey.available.map((p) => p.id)).toContain('openai');
  });

  it('carries no bare model id outside the registry', () => {
    // The `models` command used to print four hardcoded ids that all went
    // stale. Every concrete id now lives in one field, so this asserts the
    // field exists for cloud providers rather than the ids themselves, which
    // are expected to change.
    for (const p of LLM_PROVIDERS) {
      if (p.credential === 'local') continue;
      expect(p.defaultModel, `${p.id} has no defaultModel`).toBeTruthy();
      expect(p.modelFormat).toBe(`${p.id}/<model>`);
    }
  });
});

describe('skill resolution', () => {
  it('resolves provider aliases and non-LLM skills to instances', async () => {
    const r = await resolveSkillsByName(['claude', 'filesystem', 'shell']);
    expect(r.skills).toHaveLength(3);
    expect(r.unknown).toEqual([]);
  });

  it('reports unknown names instead of dropping them', async () => {
    // `serve` turns this into a hard error. Silently ignoring a typo'd skill
    // is how an agent starts up missing the capability its file declares.
    const r = await resolveSkillsByName(['openai', 'filesystm']);
    expect(r.skills).toHaveLength(1);
    expect(r.unknown).toEqual(['filesystm']);
  });
});

describe('agent file parsing', () => {
  it('reads the model, which the hand-rolled parser could not see at all', async () => {
    const { parseAgentMarkdown } = await import('../../../src/agents/index');
    const parsed = parseAgentMarkdown(
      '---\nname: bot\nmodel: anthropic/claude-haiku-4-5-20251001\n---\n\nBody.\n',
    );
    expect(parsed.name).toBe('bot');
    expect(parsed.model).toBe('anthropic/claude-haiku-4-5-20251001');
    expect(parsed.instructions).toBe('Body.');
  });

  it('yields the key of a dict-form skill entry', async () => {
    // `- mcp: {...}` is a documented form. The previous line-scanner matched
    // `line.trim().startsWith('-')` and produced nothing usable for it.
    const { parseAgentMarkdown } = await import('../../../src/agents/index');
    const parsed = parseAgentMarkdown(
      '---\nname: bot\nskills:\n  - openai\n  - mcp:\n      command: npx\n---\n\nBody.\n',
    );
    expect(parsed.skills).toEqual(['openai', 'mcp']);
  });

  it('falls back to plain instructions rather than throwing on bad YAML', async () => {
    // A REPL that refuses to start because a stray colon broke the frontmatter
    // is worse than one that runs with a plain prompt.
    const { parseAgentMarkdown } = await import('../../../src/agents/index');
    const parsed = parseAgentMarkdown('---\nname: [unclosed\n---\n\nBody.\n');
    expect(parsed.name).toBe('unknown');
    expect(parsed.instructions).toContain('Body.');
  });

  it('resolves AGENT-<name>.md, then AGENT.md, and refuses to guess when ambiguous', async () => {
    const { findAgentFile } = await import('../../../src/agents/index');
    const { writeFileSync } = await import('node:fs');
    const path = await import('node:path');

    const dir = tempDir('webagents-find-');
    expect(findAgentFile(dir)).toBeNull();

    writeFileSync(path.join(dir, 'AGENT-one.md'), '---\nname: one\n---\n');
    // Exactly one named agent: unambiguous, so use it.
    expect(findAgentFile(dir)).toBe(path.join(dir, 'AGENT-one.md'));

    writeFileSync(path.join(dir, 'AGENT-two.md'), '---\nname: two\n---\n');
    // Two named agents and no AGENT.md: ambiguous, so do not guess.
    expect(findAgentFile(dir)).toBeNull();
    // Unless one is named explicitly.
    expect(findAgentFile(dir, 'two')).toBe(path.join(dir, 'AGENT-two.md'));

    writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: default\n---\n');
    // AGENT.md wins over the ambiguity.
    expect(findAgentFile(dir)).toBe(path.join(dir, 'AGENT.md'));
  });
});

describe('package version', () => {
  it('is not hardcoded in the CLI source', () => {
    // `const version = '0.1.0'` sat here while the package was 0.3.6, so
    // `--version` lied, `update` told everyone to upgrade forever, and `init`
    // scaffolded an uninstallable dependency range.
    const src = readFileSync(join(packageRoot, 'src', 'cli', 'index.ts'), 'utf-8');
    expect(src).not.toMatch(/^const version\s*=\s*['"]\d+\.\d+\.\d+['"]/m);
    expect(src).toContain('package.json');
  });

  it('scaffolds no package.json, so there is no dependency range to go stale', () => {
    // It pinned `^${version}` once, which a CLI built ahead of the registry
    // could not satisfy (ETARGET), then `latest`. `init` writes AGENT.md only
    // now, as the Python CLI does (2026-09-25).
    const src = readFileSync(join(packageRoot, 'src', 'cli', 'index.ts'), 'utf-8');
    expect(src).not.toMatch(/writeFileSync\(path\.join\(dir, 'package\.json'\)/);
  });
});

// Spawn-based checks need `pnpm build` first, so they stay opt-in.
const runE2E = process.env.RUN_E2E === 'true';

describe.skipIf(!runE2E)('CLI process behaviour (RUN_E2E=true)', () => {
  const cliPath = join(packageRoot, 'dist', 'cli', 'index.js');

  async function runCli(args: string[]): Promise<{ stdout: string; stderr: string; code: number | null }> {
    return new Promise((resolve) => {
      const proc = spawn('node', [cliPath, ...args]);
      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', (d) => { stdout += d.toString(); });
      proc.stderr.on('data', (d) => { stderr += d.toString(); });
      proc.on('close', (code) => resolve({ stdout, stderr, code }));
    });
  }

  it('reports the real package version', async () => {
    const pkg = JSON.parse(readFileSync(join(packageRoot, 'package.json'), 'utf-8'));
    const { stdout } = await runCli(['--version']);
    expect(stdout.trim()).toBe(pkg.version);
  });

  it('displays help', async () => {
    const { stdout } = await runCli(['--help']);
    expect(stdout).toContain('Usage:');
  });

  it('rejects an unknown --output-format rather than silently using text', async () => {
    const { stderr, code } = await runCli(['chat', '-p', 'hi', '--output-format', 'bogus']);
    expect(code).toBe(2);
    expect(stderr).toContain('Unknown --output-format');
  });

  it('lists providers, not a hardcoded set of model ids', async () => {
    const { stdout } = await runCli(['models']);
    expect(stdout).toContain('openai/<model>');
    expect(stdout).not.toContain('claude-3-5-sonnet');
  });
});
