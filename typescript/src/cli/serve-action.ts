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

import * as path from 'node:path';
import type { IAgent } from '../core/types';
import { loadAgentProject } from './agent-project.js';

export interface ServeCommandOptions {
  /** Port as commander hands it over: a string. */
  port: string;
  /** Unset: `serve()` decides (loopback unless the agent is meant to be reached). */
  host?: string;
  // No `multi`: the CLI's `--multi` was removed because nothing here ever read
  // it. See the note at its old declaration in cli/index.ts.
}

/** Seams for the unit test. Every one defaults to what the CLI really does. */
export interface ServeCommandDeps {
  loadConfig?: (agentPath: string) => Record<string, unknown>;
  createAgent?: (config: Record<string, unknown>) => Promise<IAgent> | IAgent;
  serve?: (agent: IAgent, config: { port: number; hostname?: string }) => Promise<unknown>;
}

/**
 * The agent at the given path (`AGENT.md`, or `agent.json` with its
 * `instructions.md`); a default agent if ABSENT.
 *
 * Absent and malformed are deliberately not the same thing (2026-09-23). One
 * `catch` used to cover the stat, the read and the JSON.parse alike, so a
 * trailing comma in `agent.json` printed "No agent.json found" and served an
 * agent called `agent` with no instructions. The file was right there; the user
 * was told it did not exist, and got a server that looked healthy. A file that
 * exists and cannot be parsed is now fatal.
 */
export function loadAgentConfigFile(agentPath: string): Record<string, unknown> {
  // `AGENT.md` first, then `agent.json` with `instructions.md` beside it
  // (2026-09-24, `agent-project.ts`). This read `agent.json` alone, and the
  // scaffold keeps its instructions in `instructions.md`, so every agent
  // `init` created was served with no instructions at all.
  const project = loadAgentProject(agentPath);
  if (!project) {
    console.log(`No AGENT.md or agent.json at ${path.resolve(agentPath)}, serving default agent`);
    return { name: 'agent' };
  }
  const config: Record<string, unknown> = { name: project.name, skills: project.skills, skillEntries: project.skillEntries };
  if (project.description) config.description = project.description;
  if (project.instructions) config.instructions = project.instructions;
  if (project.model) config.model = project.model;
  if (project.access !== undefined) config.access = project.access;
  config.source = project.source;
  return config;
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

      // `model` and `skills` are read here now. They were declared in the type,
      // written by `webagents init`, and then dropped on the floor: only name,
      // description and instructions ever reached BaseAgent, so a scaffolded
      // agent served with no LLM skill attached and never answered.
      const { resolveSkillsByName } = await import('../skills/resolve.js');
      const declared = Array.isArray(config.skills) ? (config.skills as unknown[]).map(String) : [];
      // The entries as written, config included (`- rest: {sign: always}`);
      // the names above are for the provider checks below.
      const entries = Array.isArray(config.skillEntries)
        ? (config.skillEntries as Array<string | Record<string, unknown>>)
        : declared;
      // Keys kept with `webagents secrets set` count, as they do in the chat
      // and in the Python `serve`. Handed to the model client only, never put
      // in `process.env`, which the shell skill passes to every command.
      const { keyProviders } = await import('./model-access.js');
      const { readStoredProviderKeys } = await import('./provider-keys.js');
      const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
      const apiKeys: Record<string, string> = {};
      const { storedKeyFor } = await import('./provider-keys.js');
      for (const provider of keyProviders()) {
        const value = storedKeyFor(provider, stored);
        if (value) apiKeys[provider.id] = value;
      }
      // The agent's folder, as the Python `serve` roots its skills there.
      const { statSync } = await import('node:fs');
      const { dirname, resolve } = await import('node:path');
      const target = resolve(agentPath);
      let agentDir = target;
      try {
        if (!statSync(target).isDirectory()) agentDir = dirname(target);
      } catch {
        agentDir = process.cwd();
      }
      const { skills, byName, unknown, failed } = await resolveSkillsByName(entries, {
        model: config.model as string | undefined,
        apiKeys,
        agentDir,
      });

      if (unknown.length) {
        // Fatal, unlike the daemon's warn-and-continue: `serve` is a foreground
        // command run by a person who can fix the file, and starting an agent
        // that is missing a skill it declares is worse than refusing.
        throw new Error(
          `Unknown skill(s) in agent config: ${unknown.join(', ')}. ` +
          'Run `webagents skills list` for the available names.',
        );
      }
      for (const f of failed) {
        // Loaded but threw: usually a missing optional peer dependency. Name it
        // and continue, because the rest of the agent is still serviceable.
        console.warn(`Skill "${f.name}" failed to load: ${f.reason}`);
      }

      // Said at startup rather than as a 500 on the first request (2026-09-24).
      // A warning, not a refusal: the key may be supplied some other way.
      const { findProvider, missingProviderKey } = await import('../skills/llm/providers.js');
      const { cliCommand } = await import('./config-store.js');
      if (declared.some((name) => findProvider(name) !== undefined)) {
        const missing = missingProviderKey(declared, { ...stored, ...process.env });
        if (missing) {
          console.warn(
            `${missing.envVar} is not set, so every request will fail until it is. ` +
            `Stop the server, run \`${cliCommand(`secrets set ${missing.envVar}`)}\`, and start it again.`,
          );
        }
      }

      // Who may call it, and what each group gets (ADR-0045). A malformed
      // block stops the server with its own sentence, as Python's does.
      const { AccessConfigError } = await import('../access/policy.js');
      const { AgentFileError } = await import('../agents/index.js');
      const { accessSkillFor, applyAccessTools } = await import('../access/install.js');
      let access: ReturnType<typeof accessSkillFor> | undefined;
      try {
        access = config.access !== undefined ? accessSkillFor(config.access, config.source as string | undefined) : undefined;
      } catch (err) {
        if (err instanceof AccessConfigError) throw new AgentFileError(`${config.source as string}: ${err.message}`);
        throw err;
      }

      const agent = new BaseAgent({
        name: (config.name as string) ?? 'agent',
        description: config.description as string,
        instructions: config.instructions as string,
        model: config.model as string | undefined,
        skills: access ? [...skills, access.skill] : skills,
      }) as unknown as IAgent;
      if (access) {
        try {
          applyAccessTools(access.policy, byName);
        } catch (err) {
          if (err instanceof AccessConfigError) throw new AgentFileError(`${config.source as string}: ${err.message}`);
          throw err;
        }
      }
      return agent;
    });

  const serveFn =
    deps.serve ??
    (async (agent: IAgent, config: { port: number; hostname?: string }) => {
      const { serve } = await import('../server/node.js');
      return serve(agent, config);
    });

  const agent = await createAgent(agentConfig);

  // No `agent.initialize()` here: `serve()` owns that. See the module
  // docstring; `tests/unit/cli/serve-action.test.ts` pins the call count.
  await serveFn(agent, { port: parseInt(options.port, 10), hostname: options.host });
}
