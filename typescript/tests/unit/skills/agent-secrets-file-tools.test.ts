/**
 * The file tools guard the agent's secrets and its control files (2026-09-27,
 * the agent-secrets lane; S-312 and S-314). The names, sentences and cases
 * are the shared fixture's (`python/tests/fixtures/agent_secrets/file_tools.json`);
 * the Python suite runs the same in `tests/skills/local/test_agent_secrets_file_tools.py`.
 * The hole is proved closed the way the real-model pass exercised it: in
 * process, `write_file` asked to rewrite `AGENT.md` to `preset: unrestricted`.
 */

import { describe, expect, it } from 'vitest';
import { existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { Context } from '../../../src/core/types';
import { scopeAllows } from '../../../src/core/scopes';
import { FilesystemSkill } from '../../../src/skills/filesystem/skill';
import {
  CONTROL_HEADER,
  CONTROL_QUESTION,
  controlDeclined,
  controlEntries,
  controlPatterns,
  controlRefusal,
  CONTROL_PREFIXES,
  isControlPath,
  isSecretPath,
  SECRET_FOLDERS,
  SECRET_NAMES,
  SECRET_PREFIXES,
  secretRefusal,
  unifiedDiff,
} from '../../../src/skills/filesystem/agent-secrets-guard';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/agent_secrets/file_tools.json'), 'utf8')) as {
  secrets: {
    names: string[];
    prefixes: string[];
    folders: string[];
    refusal: string;
    cases: Array<{ path: string; refused: boolean }>;
    symlink: { link: string; target: string; refused: boolean };
  };
  control: {
    entries: string[];
    patterns: string[];
    prefixes: string[];
    refusal: string;
    header: string;
    question: string;
    declined: string;
    cases: Array<{ path: string; home?: boolean; control: boolean }>;
    named_agent_file: string;
  };
  list_directory: { path_description: string };
};

const tempDir = tempDirs();

function contextWith(scope: string | null): Context {
  const held = scope ? [scope] : [];
  return { hasScope: (required: string) => scopeAllows(required, held), get: () => undefined } as unknown as Context;
}
const OWNER = contextWith('owner');
const FRIEND = contextWith('group:friends');
const NOBODY = contextWith(null);

describe('the names and the sentences are the fixture’s', () => {
  it('secrets', () => {
    expect([...SECRET_NAMES]).toEqual(FIXTURE.secrets.names);
    expect([...SECRET_PREFIXES]).toEqual(FIXTURE.secrets.prefixes);
    expect([...SECRET_FOLDERS]).toEqual(FIXTURE.secrets.folders);
    expect(secretRefusal('{path}')).toBe(FIXTURE.secrets.refusal);
  });

  it('control files: the sandbox’s deny set, its agent-file patterns, and .env*', () => {
    expect(controlEntries()).toEqual(FIXTURE.control.entries);
    expect(controlPatterns()).toEqual(FIXTURE.control.patterns);
    expect([...CONTROL_PREFIXES]).toEqual(FIXTURE.control.prefixes);
    expect(controlRefusal('{path}')).toBe(FIXTURE.control.refusal);
    expect(CONTROL_HEADER).toBe(FIXTURE.control.header);
    expect(CONTROL_QUESTION).toBe(FIXTURE.control.question);
    expect(controlDeclined('{path}')).toBe(FIXTURE.control.declined);
  });

  it('list_directory needs no argument and says what its path defaults to', () => {
    const tool = new FilesystemSkill().tools.find((t) => t.name === 'list_directory')!;
    const parameters = tool.parameters as { properties: { path: { description: string } }; required?: string[] };
    expect(parameters.required ?? []).toEqual([]);
    expect(parameters.properties.path.description).toBe(FIXTURE.list_directory.path_description);
  });
});

describe('which paths are secrets, and which are control files (the fixture’s cases)', () => {
  it.each(FIXTURE.secrets.cases.map((c) => [c.path, c.refused] as const))('secret? %s -> %s', (relative, refused) => {
    const dir = tempDir('wa-agent-secrets-');
    expect(isSecretPath(path.join(dir, relative))).toBe(refused);
  });

  it.each(FIXTURE.control.cases.map((c) => [`${c.home ? '~/' : ''}${c.path}`, c] as const))('control? %s', (_shown, c) => {
    const dir = tempDir('wa-agent-secrets-');
    const home = tempDir('wa-agent-secrets-home-');
    expect(isControlPath(path.join(c.home ? home : dir, c.path))).toBe(c.control);
  });

  it('a symbolic link to a secret is a secret', () => {
    const dir = tempDir('wa-agent-secrets-');
    writeFileSync(path.join(dir, FIXTURE.secrets.symlink.target), 'KEY=1\n');
    symlinkSync(path.join(dir, FIXTURE.secrets.symlink.target), path.join(dir, FIXTURE.secrets.symlink.link));
    expect(isSecretPath(path.join(dir, FIXTURE.secrets.symlink.link))).toBe(FIXTURE.secrets.symlink.refused);
  });

  it('the running agent’s own file is a control file under any name', () => {
    const dir = tempDir('wa-agent-secrets-');
    const own = path.join(dir, FIXTURE.control.named_agent_file);
    expect(isControlPath(own)).toBe(false);
    expect(isControlPath(own, own)).toBe(true);
    expect(isControlPath(path.join(dir, 'notes.md'), own)).toBe(false);
  });
});

describe('the file tools refuse the secrets set for everyone (S-312)', () => {
  function folder(): { dir: string; skill: FilesystemSkill } {
    const dir = tempDir('wa-agent-secrets-');
    writeFileSync(path.join(dir, '.env'), 'OPENAI_API_KEY=sk-live-fixture\n');
    mkdirSync(path.join(dir, '.webagents', 'keys'), { recursive: true });
    writeFileSync(path.join(dir, '.webagents', 'keys', 'reporter.ed25519.jwk.json'), '{"d":"secret"}');
    writeFileSync(path.join(dir, 'README.md'), 'sk-live-fixture is not here\nhello\n');
    return { dir, skill: new FilesystemSkill({ baseDir: dir }) };
  }

  it('read_file, write_file and replace answer the sentence, and the bytes stay', async () => {
    const { dir, skill } = folder();
    expect(await skill.readFile({ path: '.env' }, OWNER)).toBe(secretRefusal('.env'));
    expect(await skill.readFile({ path: '.webagents/keys/reporter.ed25519.jwk.json' }, OWNER)).toBe(
      secretRefusal('.webagents/keys/reporter.ed25519.jwk.json'),
    );
    expect(await skill.writeFile({ file_path: '.env.local', content: 'X=1' }, OWNER)).toBe(secretRefusal('.env.local'));
    expect(existsSync(path.join(dir, '.env.local'))).toBe(false);
    expect(await skill.replace({ file_path: '.env', old_string: 'sk-live', new_string: 'gone' }, OWNER)).toBe(secretRefusal('.env'));
    expect(readFileSync(path.join(dir, '.env'), 'utf8')).toBe('OPENAI_API_KEY=sk-live-fixture\n');
  });

  it('search_file_content never reads them, and a link to one is refused too', async () => {
    const { dir, skill } = folder();
    const found = await skill.searchFileContent({ pattern: 'sk-live' }, OWNER);
    expect(found).toContain('README.md');
    expect(found).not.toContain('.env');
    expect(found).not.toContain('OPENAI_API_KEY');
    symlinkSync(path.join(dir, '.env'), path.join(dir, 'config.txt'));
    expect(await skill.readFile({ path: 'config.txt' }, OWNER)).toBe(secretRefusal('config.txt'));
  });

  it('a listing still shows the names, with no argument at all', async () => {
    const { skill } = folder();
    const listing = await skill.listDirectory({}, OWNER);
    expect(listing).toContain('.env');
    expect(listing).toContain('[DIR] .webagents');
    expect(await skill.listDirectory({ path: undefined as unknown as string }, OWNER)).toBe(listing);
  });
});

describe('the file tools write a control file only with the owner’s yes in the chat (S-314)', () => {
  const BEFORE = '---\nname: helper\nsandbox:\n  preset: development\n---\n\nYou help.\n';
  const AFTER = '---\nname: helper\nsandbox:\n  preset: unrestricted\n---\n\nYou help.\n';

  function folder(confirm?: (file: string, diff: string) => Promise<boolean>): { dir: string; skill: FilesystemSkill } {
    const dir = tempDir('wa-agent-secrets-');
    writeFileSync(path.join(dir, 'AGENT.md'), BEFORE);
    return { dir, skill: new FilesystemSkill({ baseDir: dir, ...(confirm ? { confirmControlWrite: confirm } : {}) }) };
  }

  it('with no chat to ask (serve, the daemon, -p) the write is refused and AGENT.md keeps its preset', async () => {
    const { dir, skill } = folder();
    expect(await skill.writeFile({ file_path: 'AGENT.md', content: AFTER }, OWNER)).toBe(controlRefusal('AGENT.md'));
    expect(await skill.replace({ file_path: 'AGENT.md', old_string: 'development', new_string: 'unrestricted' }, OWNER)).toBe(
      controlRefusal('AGENT.md'),
    );
    expect(readFileSync(path.join(dir, 'AGENT.md'), 'utf8')).toBe(BEFORE);
    for (const planted of ['AGENT-evil.md', 'WEBAGENTS.md', 'mcp.json', '.mcp.json', '.agents/skills/x/SKILL.md', '.git/hooks/pre-commit', '.envrc']) {
      expect(await skill.writeFile({ file_path: planted, content: 'planted' }, OWNER)).toBe(controlRefusal(planted));
      expect(existsSync(path.join(dir, planted))).toBe(false);
    }
    expect(await skill.replace({ file_path: 'AGENT-evil.md', old_string: '', new_string: 'planted' }, OWNER)).toBe(
      controlRefusal('AGENT-evil.md'),
    );
    expect(existsSync(path.join(dir, 'AGENT-evil.md'))).toBe(false);
  });

  it('with the chat asking, the owner sees the diff and a yes writes it', async () => {
    const asked: Array<{ file: string; diff: string }> = [];
    const { dir, skill } = folder(async (file, diff) => {
      asked.push({ file, diff });
      return true;
    });
    expect(await skill.writeFile({ file_path: 'AGENT.md', content: AFTER }, OWNER)).toContain('Successfully overwrote');
    expect(readFileSync(path.join(dir, 'AGENT.md'), 'utf8')).toBe(AFTER);
    expect(asked).toHaveLength(1);
    expect(asked[0].file).toBe('AGENT.md');
    expect(asked[0].diff).toContain('-  preset: development');
    expect(asked[0].diff).toContain('+  preset: unrestricted');
    // A new control file shows as all additions.
    expect(await skill.writeFile({ file_path: 'WEBAGENTS.md', content: 'Shared context.\n' }, OWNER)).toContain('Successfully created');
    expect(asked[1].diff).toContain('+Shared context.');
  });

  it('a no leaves the file as it was, and says the owner declined', async () => {
    const { dir, skill } = folder(async () => false);
    expect(await skill.writeFile({ file_path: 'AGENT.md', content: AFTER }, OWNER)).toBe(controlDeclined('AGENT.md'));
    expect(await skill.replace({ file_path: 'AGENT.md', old_string: 'development', new_string: 'unrestricted' }, OWNER)).toBe(
      controlDeclined('AGENT.md'),
    );
    expect(readFileSync(path.join(dir, 'AGENT.md'), 'utf8')).toBe(BEFORE);
  });

  it('a caller who is not the owner is refused even when the chat could ask', async () => {
    let asked = 0;
    const { dir, skill } = folder(async () => {
      asked += 1;
      return true;
    });
    expect(await skill.writeFile({ file_path: 'AGENT.md', content: AFTER }, FRIEND)).toBe(controlRefusal('AGENT.md'));
    expect(await skill.writeFile({ file_path: 'AGENT.md', content: AFTER }, NOBODY)).toBe(controlRefusal('AGENT.md'));
    expect(asked).toBe(0);
    expect(readFileSync(path.join(dir, 'AGENT.md'), 'utf8')).toBe(BEFORE);
  });

  it('an ordinary file never asks', async () => {
    let asked = 0;
    const { dir, skill } = folder(async () => {
      asked += 1;
      return false;
    });
    expect(await skill.writeFile({ file_path: 'notes.md', content: 'plain' }, OWNER)).toContain('Successfully created');
    expect(readFileSync(path.join(dir, 'notes.md'), 'utf8')).toBe('plain');
    expect(asked).toBe(0);
  });

  it('the diff is a unified diff, and a huge file is summarized', () => {
    const diff = unifiedDiff('a\nb\nc\n', 'a\nB\nc\n', 'x.md');
    expect(diff.split('\n').slice(0, 2)).toEqual(['--- x.md', '+++ x.md']);
    expect(diff).toContain('-b');
    expect(diff).toContain('+B');
    const big = Array.from({ length: 5000 }, (_, i) => `line ${i}`).join('\n');
    expect(unifiedDiff(big, `${big}\nmore`, 'big.md')).toContain('too large to show');
  });
});
