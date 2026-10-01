/**
 * `webagents mcp list` / `mcp add` and the chat's `/mcp list` / `/mcp add`
 * (`src/cli/mcp-import.ts`, 2026-09-29), against the shared fixture
 * `python/tests/fixtures/mcp_tool/import.json`, which the Python suite reads
 * too (`tests/cli/test_mcp_import.py`).
 *
 * The owner asked whether the MCP servers already set up in other apps could
 * be listed and used. What is pinned: the settings files read and how each
 * app's entry is read; that a listing never prints a key (a key-looking value,
 * the value after a flag named like a key, a key in an address's query,
 * variable and header values); that `add` moves keys into this profile's
 * secret store and the entry reads `${secret:NAME}`, refuses a key on the
 * command line, turns VS Code's inputs into secrets to set, and writes the
 * entry where the agent reads its servers without touching any other byte of
 * the agent file. Then the real command and the chat command, in a temporary
 * home with the file secret store, so nothing reaches the owner's own
 * settings or keystore.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  AddRefused,
  agentFileFor,
  agentNameOf,
  APPS,
  CannotEdit,
  choose,
  convertEntry,
  describe as describeEntry,
  discover,
  type FoundServer,
  insertIntoAgentFile,
  listLines,
  maskedArgs,
  type McpEntry,
  normalizeEntry,
  planAdd,
  RemoveRefused,
  removeFromAgent,
  removeFromAgentFile,
  resultLines,
  type SecretStoreLike,
  settingsFiles,
  stripJsonc,
  WORDS,
  writePlan,
} from '../../../src/cli/mcp-import';
import { CLI_ARGS, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
type RawFound = { app_id: string; app: string; file: string; name: string; entry: McpEntry };
type Fixture = {
  words: Record<string, string>;
  apps: Array<[string, string]>;
  settings_files: { folder: string } & Record<'darwin' | 'linux', { home: string; files: Array<[string, string, string, string]> }>;
  jsonc: Array<{ about: string; text: string; data: unknown }>;
  normalize: Array<{ about: string; raw: unknown; entry: McpEntry | null }>;
  masked_args: Array<{ args: string[]; masked: string[]; carried: boolean }>;
  describe: Array<{ entry: McpEntry; says: string }>;
  list: { home: string; cases: Array<{ about: string; found: RawFound[]; unreadable: Array<[string, string, string]>; lines: string[] }> };
  convert: {
    folder: string;
    cases: Array<{
      about: string;
      found: RawFound;
      store: Record<string, string>;
      refused?: string;
      entry?: McpEntry;
      stored?: string[];
      to_set?: string[];
      store_after?: Record<string, string>;
      writes?: number;
    }>;
  };
  insert: Array<{ about: string; text: string; name: string; entry: McpEntry; after?: string; cannot_edit?: boolean }>;
  choose: { found: RawFound[]; list_command: string; cases: Array<{ name: string; from: string | null; picks?: number; refused?: string }> };
  plan: Array<{
    about: string;
    agent_file: string;
    agent: string;
    mcp_json: string | null;
    found: RawFound;
    store: Record<string, string>;
    where?: string;
    writes?: Record<string, string>;
    stored?: string[];
    refused?: string;
  }>;
  agent_name: Array<{ file: string; text: string; name: string }>;
  remove: { cases: Array<{ about: string; text: string; name: string; after?: string; cannot_edit?: boolean }> };
  discover: {
    system: string;
    files: Record<string, string>;
    found: Array<[string, string, string, string, McpEntry]>;
    unreadable: Array<[string, string, string]>;
  };
};
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/import.json'), 'utf8')) as Fixture;

const found = (raw: RawFound): FoundServer => ({ appId: raw.app_id, app: raw.app, file: raw.file, name: raw.name, entry: raw.entry });
const cliCommand = (rest: string): string => `webagents ${rest}`;

/** The store `convertEntry` writes keys to, in memory. */
class FakeStore implements SecretStoreLike {
  held: Record<string, string>;
  writes = 0;
  constructor(held: Record<string, string> = {}) {
    this.held = { ...held };
  }
  async get(name: string): Promise<string | null> {
    return this.held[name] ?? null;
  }
  async set(name: string, value: string): Promise<void> {
    this.held[name] = value;
    this.writes += 1;
  }
}

const tempDir = tempDirs();

describe('the words', () => {
  it('are the fixture words and apps', () => {
    expect(WORDS).toEqual(FIXTURE.words);
    expect(APPS).toEqual(FIXTURE.apps);
  });

  it.each(['darwin', 'linux'] as const)('the settings files read on %s', (system) => {
    const c = FIXTURE.settings_files[system];
    expect(settingsFiles(c.home, FIXTURE.settings_files.folder, system)).toEqual(c.files);
  });

  it('reads APPDATA on Windows', () => {
    const files = settingsFiles('C:\\Users\\u', 'C:\\w', 'win32', 'C:\\Users\\u\\AppData\\Roaming');
    expect(files[0][2]).toBe('C:\\Users\\u\\AppData\\Roaming\\Claude\\claude_desktop_config.json');
    expect(files[5][2]).toBe('C:\\Users\\u\\AppData\\Roaming\\Code\\User\\mcp.json');
  });
});

describe('reading', () => {
  it.each(FIXTURE.jsonc.map((c) => [c.about, c] as const))('jsonc: %s', (_about, c) => {
    expect(JSON.parse(stripJsonc(c.text))).toEqual(c.data);
  });

  it.each(FIXTURE.normalize.map((c) => [c.about, c] as const))('normalize: %s', (_about, c) => {
    expect(normalizeEntry(c.raw)).toEqual(c.entry);
  });

  it('does not split a command that is a file with spaces in its path', () => {
    const dir = path.join(tempDir('wa-mcp-import-cmd-'), 'My Server');
    fs.mkdirSync(dir);
    const server = path.join(dir, 'run server');
    fs.writeFileSync(server, '#!/bin/sh\n');
    expect(normalizeEntry({ command: server })).toEqual({ command: server, args: [] });
  });

  it('discovers every app and says which file it could not read', () => {
    const c = FIXTURE.discover;
    const root = tempDir('wa-mcp-import-discover-');
    const home = path.join(root, 'home');
    const folder = path.join(root, 'folder');
    fs.mkdirSync(home);
    fs.mkdirSync(folder);
    const real = fs.realpathSync(folder);
    for (const [relative, text] of Object.entries(c.files)) {
      const file = path.join(root, relative);
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, text.split('{folder}').join(real));
    }
    const shown = (file: string): string => file.split(real).join('folder').split(home).join('home');
    const result = discover(home, folder, c.system, '');
    expect(result.found.map((f) => [f.appId, f.app, shown(f.file), f.name, f.entry])).toEqual(c.found);
    expect(result.unreadable.map(([a, f, r]) => [a, shown(f), r])).toEqual(c.unreadable);
  });
});

describe('reading again', () => {
  it('reads a settings file again when it changes, drops a byte-order mark and replaces bytes that are not UTF-8', () => {
    const root = tempDir('wa-mcp-import-reread-');
    const home = path.join(root, 'home');
    const folder = path.join(root, 'folder');
    fs.mkdirSync(path.join(home, '.cursor'), { recursive: true });
    fs.mkdirSync(folder);
    const file = path.join(home, '.cursor', 'mcp.json');
    fs.writeFileSync(file, `\ufeff${JSON.stringify({ mcpServers: { a: { command: 'x' } } })}`);
    expect(discover(home, folder, 'linux').found.map((f) => f.name)).toEqual(['a']);
    fs.writeFileSync(file, JSON.stringify({ mcpServers: { a: { command: 'x' }, bb: { command: 'y' } } }));
    expect(discover(home, folder, 'linux').found.map((f) => f.name)).toEqual(['a', 'bb']);
    fs.writeFileSync(file, Buffer.from('{"mcpServers": {"c": {"command": "caf\xe9"}}}', 'latin1'));
    expect(discover(home, folder, 'linux').found.map((f) => f.entry)).toEqual([{ command: 'caf\ufffd', args: [] }]);
  });
});

describe('showing', () => {
  it.each(FIXTURE.masked_args.map((c) => [c.args.join(' '), c] as const))('masks the keys in: %s', (_args, c) => {
    expect(maskedArgs(c.args)).toEqual([c.masked, c.carried]);
  });

  it.each(FIXTURE.describe.map((c) => [c.says, c] as const))('one line: %s', (_says, c) => {
    expect(describeEntry(c.entry)).toBe(c.says);
  });

  it.each(FIXTURE.list.cases.map((c) => [c.about, c] as const))('the listing: %s', (_about, c) => {
    expect(listLines({ found: c.found.map(found), unreadable: c.unreadable }, FIXTURE.list.home)).toEqual(c.lines);
  });
});

describe('converting', () => {
  it.each(FIXTURE.convert.cases.map((c) => [c.about, c] as const))('%s', async (_about, c) => {
    const store = new FakeStore(c.store);
    const server = found(c.found);
    if (c.refused !== undefined) {
      await expect(convertEntry(server, FIXTURE.convert.folder, store)).rejects.toThrow(new AddRefused(c.refused));
      expect(store.held).toEqual(c.store);
      return;
    }
    const converted = await convertEntry(server, FIXTURE.convert.folder, store);
    expect(converted.entry).toEqual(c.entry);
    expect(converted.stored).toEqual(c.stored);
    expect(converted.toSet).toEqual(c.to_set);
    expect(store.held).toEqual(c.store_after);
    if (c.writes !== undefined) expect(store.writes).toBe(c.writes);
    expect(server.entry).toEqual(c.found.entry);
  });
});

describe('writing', () => {
  it.each(FIXTURE.insert.map((c) => [c.about, c] as const))('insert: %s', (_about, c) => {
    if (c.cannot_edit) {
      expect(() => insertIntoAgentFile(c.text, c.name, c.entry, 'AGENT.md')).toThrow(CannotEdit);
      return;
    }
    expect(insertIntoAgentFile(c.text, c.name, c.entry, 'AGENT.md')).toBe(c.after);
  });

  it.each(FIXTURE.choose.cases.map((c) => [`${c.name} from ${c.from}`, c] as const))('choose: %s', (_label, c) => {
    const discovery = { found: FIXTURE.choose.found.map(found), unreadable: [] };
    const from = c.from ?? undefined;
    if (c.refused !== undefined) {
      expect(() => choose(discovery, c.name, from, FIXTURE.choose.list_command)).toThrow(new AddRefused(c.refused));
      return;
    }
    expect(discovery.found.indexOf(choose(discovery, c.name, from, FIXTURE.choose.list_command))).toBe(c.picks);
  });

  it.each(FIXTURE.plan.map((c) => [c.about, c] as const))('plan: %s', async (_about, c) => {
    const dir = tempDir('wa-mcp-import-plan-');
    const agentFile = path.join(dir, c.agent_file);
    fs.writeFileSync(agentFile, c.agent);
    if (c.mcp_json !== null) fs.writeFileSync(path.join(dir, 'mcp.json'), c.mcp_json);
    const before = Object.fromEntries(fs.readdirSync(dir).map((n) => [n, fs.readFileSync(path.join(dir, n), 'utf8')]));
    const name = agentNameOf(agentFile, c.agent);
    if (c.refused !== undefined) {
      await expect(planAdd(found(c.found), agentFile, new FakeStore(c.store), name)).rejects.toThrow(new AddRefused(c.refused));
    } else {
      const plan = await planAdd(found(c.found), agentFile, new FakeStore(c.store), name);
      expect(plan.where).toBe(c.where);
      expect(Object.fromEntries(plan.writes.map(([file, text]) => [path.basename(file), text]))).toEqual(c.writes);
      expect(plan.converted.stored).toEqual(c.stored);
      writePlan(plan);
      for (const [file, text] of Object.entries(c.writes ?? {})) expect(fs.readFileSync(path.join(dir, file), 'utf8')).toBe(text);
    }
    for (const [file, text] of Object.entries(before)) {
      if (!(c.writes && file in c.writes)) expect(fs.readFileSync(path.join(dir, file), 'utf8')).toBe(text);
    }
  });

  it.each(FIXTURE.agent_name.map((c) => [c.file, c] as const))("the agent's name: %s", (_file, c) => {
    expect(agentNameOf(c.file, c.text)).toBe(c.name);
  });

  it('finds the agent file a path names', () => {
    const root = tempDir('wa-mcp-import-agent-');
    fs.mkdirSync(path.join(root, 'one'));
    fs.writeFileSync(path.join(root, 'one', 'AGENT.md'), 'x');
    expect(agentFileFor(path.join(root, 'one'), cliCommand)).toBe(path.join(root, 'one', 'AGENT.md'));
    expect(agentFileFor(path.join(root, 'one', 'AGENT.md'), cliCommand)).toBe(path.join(root, 'one', 'AGENT.md'));
    fs.mkdirSync(path.join(root, 'named'));
    fs.writeFileSync(path.join(root, 'named', 'AGENT-bob.md'), 'x');
    expect(agentFileFor(path.join(root, 'named'), cliCommand)).toBe(path.join(root, 'named', 'AGENT-bob.md'));
    fs.writeFileSync(path.join(root, 'named', 'AGENT-amy.md'), 'x');
    expect(() => agentFileFor(path.join(root, 'named'), cliCommand)).toThrow(
      new AddRefused(`More than one agent file in ${path.join(root, 'named')}; name one: AGENT-amy.md, AGENT-bob.md.`),
    );
    fs.mkdirSync(path.join(root, 'empty'));
    expect(() => agentFileFor(path.join(root, 'empty'), cliCommand)).toThrow(
      new AddRefused(`No agent file in ${path.join(root, 'empty')}; webagents init makes one.`),
    );
  });

  it('says what it did', () => {
    const plan = { writes: [], where: 'mcp.json', converted: { entry: {}, stored: ['REMOTE_TOKEN'], toSet: ['API_KEY', 'GITHUB_TOKEN'] } };
    expect(resultLines(plan, 'remote', 'helper', true, cliCommand)).toEqual([
      'Added remote to helper (mcp.json).',
      "Stored REMOTE_TOKEN in this profile's secrets; the entry reads them as ${secret:NAME}.",
      'Set API_KEY, GITHUB_TOKEN before it connects: `webagents secrets set API_KEY`, `webagents secrets set GITHUB_TOKEN`.',
      '/reload connects it.',
    ]);
    const bare = { writes: [], where: 'AGENT.md', converted: { entry: {}, stored: [], toSet: [] } };
    expect(resultLines(bare, 'x', 'yoyo', false, cliCommand)).toEqual(['Added x to yoyo (AGENT.md).', 'The agent connects it the next time it starts.']);
  });
});

describe('removing', () => {
  it.each(FIXTURE.remove.cases.map((c) => [c.about, c] as const))('remove: %s', (_about, c) => {
    if (c.cannot_edit) {
      expect(() => removeFromAgentFile(c.text, c.name, 'AGENT.md')).toThrow(CannotEdit);
      return;
    }
    expect(removeFromAgentFile(c.text, c.name, 'AGENT.md')).toBe(c.after);
  });

  it('says where, and which secrets stay', () => {
    const dir = tempDir('wa-mcp-remove-');
    fs.writeFileSync(path.join(dir, 'AGENT.md'), '---\nname: helper\nskills:\n  - mcp\n---\nBe brief.\n');
    fs.writeFileSync(
      path.join(dir, 'mcp.json'),
      JSON.stringify({ mcpServers: { remote: { url: 'https://x.example/mcp', headers: { Authorization: 'Bearer ${secret:REMOTE_TOKEN}' } }, time: { command: 'uvx' } } }),
    );
    expect(removeFromAgent('remote', dir, { inChat: false, cliCommand })).toEqual([
      'Removed remote from helper (mcp.json).',
      "It read REMOTE_TOKEN from this profile's secrets; they stay stored (`webagents secrets remove <NAME>` removes one).",
      'The agent stops using it the next time it starts.',
    ]);
    expect(JSON.parse(fs.readFileSync(path.join(dir, 'mcp.json'), 'utf8'))).toEqual({ mcpServers: { time: { command: 'uvx' } } });
    expect(() => removeFromAgent('remote', dir, { inChat: true, cliCommand })).toThrow(new RemoveRefused('helper has no MCP server named remote.'));
    expect(removeFromAgent('time', dir, { inChat: true, cliCommand }).at(-1)).toBe('/reload stops it in this chat.');
  });
});

describe('the commands', () => {
  const ISOLATED = [
    'HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_DIR', 'APPDATA', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
    'WEBAGENTS_TOKEN', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
  ];
  const saved: Record<string, string | undefined> = {};
  const cwd = process.cwd();
  let home = '';
  let project = '';
  let printed: string[] = [];

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    const root = tempDir('wa-mcp-import-cmd-');
    home = path.join(root, 'home');
    project = path.join(root, 'project');
    fs.mkdirSync(path.join(home, '.cursor'), { recursive: true });
    fs.mkdirSync(project);
    fs.writeFileSync(
      path.join(home, '.cursor', 'mcp.json'),
      JSON.stringify({
        mcpServers: {
          'chrome-devtools': { command: 'npx -y chrome-devtools-mcp@latest' },
          remote: { url: 'https://mcp.example.com/mcp', headers: { Authorization: 'Bearer abc123-not-a-key' } },
        },
      }),
    );
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: helper\nskills:\n  - memory\n---\nBe brief.\n');
    process.env.HOME = home;
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
    process.env.NO_COLOR = '1';
    process.env.COLUMNS = '200';
    process.chdir(project);
    printed = [];
    for (const method of ['log', 'error'] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        printed.push(args.map(String).join(' '));
      });
    }
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    process.chdir(cwd);
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
    vi.restoreAllMocks();
  });

  function runCli(args: string[]): Promise<{ code: number | null; out: string; err: string }> {
    return new Promise((resolve) => {
      const child = spawn(process.execPath, [...CLI_ARGS, ...args], {
        cwd: project,
        env: { ...process.env, HOME: home, WEBAGENTS_PROFILE: '', WEBAGENTS_SECRETS_BACKEND: 'file' },
      });
      let out = '';
      let err = '';
      child.stdout.on('data', (d) => (out += d));
      child.stderr.on('data', (d) => (err += d));
      child.on('close', (code) => resolve({ code, out, err }));
    });
  }

  async function secret(name: string): Promise<string | null> {
    const { providerKeyStore } = await import('../../../src/cli/provider-keys');
    return (await providerKeyStore()).get(name);
  }

  it.skipIf(TSX_PROBLEM !== null)('the CLI lists and adds', async () => {
    const listed = await runCli(['mcp', 'list']);
    expect(listed.code, listed.err).toBe(0);
    expect(listed.out.trimEnd().split('\n')).toEqual([
      'MCP servers other apps on this machine use',
      'Cursor  ~/.cursor/mcp.json',
      '  chrome-devtools  npx -y chrome-devtools-mcp@latest',
      '  remote  https://mcp.example.com/mcp  headers Authorization',
      'Add one to an agent: webagents mcp add <name>, or /mcp add <name> in the chat.',
    ]);
    expect(listed.out).not.toContain('abc123');

    const added = await runCli(['mcp', 'add', 'remote']);
    expect(added.code, added.err).toBe(0);
    expect(added.out.trimEnd().split('\n')).toEqual([
      'Added remote to helper (mcp.json).',
      "Stored REMOTE_TOKEN in this profile's secrets; the entry reads them as ${secret:NAME}.",
      'The agent connects it the next time it starts.',
    ]);
    expect(await secret('REMOTE_TOKEN')).toBe('abc123-not-a-key');
    const mcpJson = fs.readFileSync(path.join(project, 'mcp.json'), 'utf8');
    expect(JSON.parse(mcpJson)).toEqual({
      mcpServers: { remote: { url: 'https://mcp.example.com/mcp', headers: { Authorization: 'Bearer ${secret:REMOTE_TOKEN}' } } },
    });
    const agent = fs.readFileSync(path.join(project, 'AGENT.md'), 'utf8');
    expect(mcpJson + agent).not.toContain('abc123');
    expect(agent).toBe('---\nname: helper\nskills:\n  - memory\n  - mcp\n---\nBe brief.\n');

    const again = await runCli(['mcp', 'add', 'remote']);
    expect(again.code).toBe(1);
    expect(again.err).toContain('helper already has an MCP server named remote.');

    const missing = await runCli(['mcp', 'add', 'nope']);
    expect(missing.code).toBe(1);
    expect(missing.err).toContain("No MCP server named nope in other apps' settings; webagents mcp list shows them.");
  }, 60_000);

  // In a loaded chat: the add's own edit of the agent file (the `- mcp` it
  // adds) is not announced as a change, since its own last line says /reload
  // connects it; an edit from outside after it still is.
  it('the chat lists and adds', async () => {
    process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
    const agentFile = path.join(project, 'AGENT.md');
    fs.writeFileSync(agentFile, '---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBe brief.\n');
    const { InteractiveREPL } = await import('../../../src/cli/app');
    const repl = new InteractiveREPL({ interactive: true }) as unknown as {
      initialize(): Promise<void>;
      handleInput(line: string): Promise<void>;
      sayFileChanged(): void;
      cleanup?(): Promise<void>;
    };
    await repl.initialize();
    await repl.handleInput('/mcp list');
    await repl.handleInput('/mcp add chrome-devtools');
    repl.sayFileChanged();
    expect(printed.join('\n')).not.toContain('changed since the chat loaded it');
    await repl.handleInput('/mcp add chrome-devtools');
    await repl.handleInput('/mcp add');
    fs.appendFileSync(agentFile, 'Be kind.\n');
    repl.sayFileChanged();
    const out = printed.join('\n');
    expect(out).toContain('  chrome-devtools  npx -y chrome-devtools-mcp@latest');
    expect(out).toContain('Added chrome-devtools to helper (mcp.json).');
    expect(out).toContain('/reload connects it.');
    expect(out).toContain('helper already has an MCP server named chrome-devtools.');
    expect(out).toContain('/mcp [list | add <name> [--from <app>] | remove <name>]');
    expect(out).toContain('AGENT.md changed since the chat loaded it. /reload uses the new version.');
    const written = JSON.parse(fs.readFileSync(path.join(project, 'mcp.json'), 'utf8')) as { mcpServers: Record<string, unknown> };
    expect(written.mcpServers['chrome-devtools']).toEqual({ command: 'npx', args: ['-y', 'chrome-devtools-mcp@latest'] });
    expect(fs.readFileSync(agentFile, 'utf8').startsWith('---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n  - mcp\n---\n')).toBe(true);
  });
});
