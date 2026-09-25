/**
 * `webagents serve` initialises the agent EXACTLY ONCE.
 *
 * The CLI used to call `agent.initialize()` and then `serve()`, which calls it
 * again. Removing the CLI's call was previously claimed to be "covered by the
 * existing CLI suite"; it was not. At the time the only file referencing
 * `cli/index` was `tests/e2e/cli.test.ts`, which ran in NEITHER runner: vitest
 * excluded `tests/e2e/**` and playwright matched only `*.spec.ts`. That file
 * now lives beside this one at `tests/unit/cli/cli.test.ts` and does run
 * (2026-09-23), but it still cannot drive `serve`, because importing
 * `cli/index` executes `program.parse()` against the test runner's own argv.
 * So this remains the coverage for the action itself, driving the real
 * `serveAction` against the real `serve()`.
 *
 * Double-initialisation is idempotent for the stock skills, so a regression
 * here is silent until someone attaches a skill whose `initialize` is not —
 * which is exactly the class of bug a call-count assertion catches and a
 * smoke test does not.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { BaseAgent } from '../../../src/core/agent.js';
import { serveAction, loadAgentConfigFile } from '../../../src/cli/serve-action.js';
import { serve } from '../../../src/server/node.js';
import type { IAgent } from '../../../src/core/types.js';
import type { ServeHandle } from '../../../src/server/node.js';

describe('the serve command declares no option it cannot honour', () => {
  /**
   * `--multi` was declared on the command and read nowhere: `webagents serve
   * --multi` silently served ONE agent while its help text promised to "load
   * all agents in directory". This walks the declaration instead of trusting
   * that nobody adds another one — `cli/index.ts` cannot be imported (it calls
   * `program.parse()` at module scope, against the test runner's own argv), so
   * the source is read.
   */
  it('every long option on `serve` is a field serveAction reads', () => {
    const cli = readFileSync(new URL('../../../src/cli/index.ts', import.meta.url), 'utf-8');
    const start = cli.indexOf("  .command('serve')");
    expect(start, 'the serve command moved; fix this scan').toBeGreaterThan(-1);
    const block = cli.slice(start, cli.indexOf('.action(', start));

    const longFlags = [...block.matchAll(/\.option\('(?:-\w, )?--([a-z-]+)/g)].map((m) => m[1]);
    expect(longFlags.length, 'the option scan found nothing').toBeGreaterThan(0);

    const action = readFileSync(
      new URL('../../../src/cli/serve-action.ts', import.meta.url),
      'utf-8',
    );
    const optionsType = action.slice(
      action.indexOf('export interface ServeCommandOptions'),
      action.indexOf('}', action.indexOf('export interface ServeCommandOptions')),
    );

    const undeclared = longFlags.filter((flag) => {
      const camel = flag.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());
      return !new RegExp(`\\b${camel}\\??:`).test(optionsType);
    });
    expect(
      undeclared,
      'these `serve` options are declared on the CLI but are not fields of ' +
        'ServeCommandOptions, so nothing reads them. Wire them up or drop them — ' +
        'a flag that lies is worse than a missing one.',
    ).toEqual([]);
  });
});

describe('cli serve action', () => {
  let keysDir: string;
  let previousKeysDir: string | undefined;
  const handles: ServeHandle[] = [];

  beforeEach(async () => {
    keysDir = await mkdtemp(path.join(tmpdir(), 'webagents-cli-keys-'));
    previousKeysDir = process.env.WEBAGENTS_KEYS_DIR;
    process.env.WEBAGENTS_KEYS_DIR = keysDir;
  });

  afterEach(async () => {
    for (const handle of handles.splice(0)) await handle.close().catch(() => {});
    if (previousKeysDir === undefined) delete process.env.WEBAGENTS_KEYS_DIR;
    else process.env.WEBAGENTS_KEYS_DIR = previousKeysDir;
    await rm(keysDir, { recursive: true, force: true });
  });

  /** A real agent whose `initialize` counts its calls and still does its job. */
  function spyAgent(): { agent: IAgent; calls: () => number } {
    const agent = new BaseAgent({ name: 'cli-spy', instructions: 'You are helpful.' });
    const original = agent.initialize.bind(agent);
    let calls = 0;
    (agent as unknown as { initialize: () => Promise<void> }).initialize = async () => {
      calls += 1;
      await original();
    };
    return { agent: agent as unknown as IAgent, calls: () => calls };
  }

  it('initialises the agent exactly once, through serve() and not itself', async () => {
    const { agent, calls } = spyAgent();

    await serveAction(
      '.',
      { port: '0', host: '127.0.0.1' },
      {
        loadConfig: () => ({ name: 'cli-spy' }),
        createAgent: () => agent,
        // The REAL serve(), with the port forced ephemeral so the test can
        // bind. Everything else is the production path, including the one
        // `initialize()` call that belongs to it.
        serve: async (a, config) => {
          const handle = await serve(a, { ...config, port: 0, heartbeat: false });
          handles.push(handle);
          return handle;
        },
      },
    );

    expect(calls()).toBe(1);
  });

  it('serves the agent it built, on the host and port it was given', async () => {
    let seen: { agent: IAgent; config: { port: number; hostname: string } } | undefined;

    await serveAction(
      './somewhere',
      { port: '8123', host: '0.0.0.0' },
      {
        loadConfig: () => ({ name: 'from-config', instructions: 'From agent.json.' }),
        serve: async (agent, config) => {
          seen = { agent, config };
        },
      },
    );

    expect(seen?.agent.name).toBe('from-config');
    expect(seen?.config).toEqual({ port: 8123, hostname: '0.0.0.0' });
  });

  it('falls back to a default agent when there is no agent.json', () => {
    const config = loadAgentConfigFile(path.join(tmpdir(), 'webagents-no-such-dir-abc123'));
    expect(config).toEqual({ name: 'agent' });
  });

  it('refuses a malformed agent.json instead of serving a default in its place', async () => {
    // ABSENT and MALFORMED used to share one `catch`, so a trailing comma in
    // agent.json printed "No agent.json found" and served an agent called
    // `agent` with no instructions. The file was right there, the user was told
    // it was not, and the server looked healthy (2026-09-23).
    const dir = await mkdtemp(path.join(tmpdir(), 'webagents-malformed-'));
    try {
      await writeFile(path.join(dir, 'agent.json'), '{ "name": "x", }', 'utf-8');
      expect(() => loadAgentConfigFile(dir)).toThrow(/not valid JSON/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('keeps the model and skills the agent file declares', async () => {
    // Both were declared in the config type, written by `webagents init`, and
    // then dropped: only name/description/instructions reached BaseAgent, so a
    // scaffolded agent served with no LLM skill and never answered.
    const dir = await mkdtemp(path.join(tmpdir(), 'webagents-config-'));
    try {
      await writeFile(
        path.join(dir, 'agent.json'),
        JSON.stringify({ name: 'kept', model: 'openai/some-model', skills: ['openai'] }),
        'utf-8',
      );
      const config = loadAgentConfigFile(dir);
      expect(config.model).toBe('openai/some-model');
      expect(config.skills).toEqual(['openai']);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
