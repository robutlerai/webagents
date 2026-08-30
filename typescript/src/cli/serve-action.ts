/**
 * The `webagents serve` command's action, lifted out of `cli/index.ts`.
 *
 * WHY IT LIVES HERE: `cli/index.ts` calls `program.parse()` at module scope,
 * so importing it to test a command RUNS the CLI against the test runner's own
 * argv. The action was therefore untestable in place — the only file
 * referencing `cli/index` is `tests/e2e/cli.test.ts`, which the vitest config
 * excludes and which never exercises `serve` at all. Moving the action into
 * its own module is the smallest change that makes it a unit under test.
 *
 * What is under test is a single, easily-regressed property: the CLI does NOT
 * call `agent.initialize()` itself. `serve()` owns initialization (it is what
 * starts an attached PortalConnectSkill before the socket binds), and the CLI
 * used to call it a second time. Double-initialization happens to be
 * idempotent for the stock skills, but nothing in the Skill contract promises
 * that for a third-party one.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import type { IAgent } from '../core/types';

export interface ServeCommandOptions {
  /** Port as commander hands it over: a string. */
  port: string;
  host: string;
  // No `multi`: the CLI's `--multi` was removed because nothing here ever read
  // it. See the note at its old declaration in cli/index.ts.
}

/** Seams for the unit test. Every one defaults to what the CLI really does. */
export interface ServeCommandDeps {
  loadConfig?: (agentPath: string) => Record<string, unknown>;
  createAgent?: (config: Record<string, unknown>) => Promise<IAgent> | IAgent;
  serve?: (agent: IAgent, config: { port: number; hostname: string }) => Promise<unknown>;
}

/** `agent.json` next to (or at) the given path; a default agent if absent. */
export function loadAgentConfigFile(agentPath: string): Record<string, unknown> {
  const configPath = path.resolve(agentPath);
  try {
    const raw = fs.readFileSync(
      fs.statSync(configPath).isDirectory() ? path.join(configPath, 'agent.json') : configPath,
      'utf-8',
    );
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    console.log('No agent.json found, serving default agent');
    return { name: 'agent' };
  }
}

export async function serveAction(
  agentPath: string,
  options: ServeCommandOptions,
  deps: ServeCommandDeps = {},
): Promise<void> {
  const loadConfig = deps.loadConfig ?? loadAgentConfigFile;
  const agentConfig = loadConfig(agentPath);

  const createAgent =
    deps.createAgent ??
    (async (config: Record<string, unknown>) => {
      const { BaseAgent } = await import('../core/agent.js');
      return new BaseAgent({
        name: (config.name as string) ?? 'agent',
        description: config.description as string,
        instructions: config.instructions as string,
      }) as unknown as IAgent;
    });

  const serveFn =
    deps.serve ??
    (async (agent: IAgent, config: { port: number; hostname: string }) => {
      const { serve } = await import('../server/node.js');
      return serve(agent, config);
    });

  const agent = await createAgent(agentConfig);

  // No `agent.initialize()` here: `serve()` owns that. See the module
  // docstring; `tests/unit/cli/serve-action.test.ts` pins the call count.
  await serveFn(agent, { port: parseInt(options.port, 10), hostname: options.host });
}
