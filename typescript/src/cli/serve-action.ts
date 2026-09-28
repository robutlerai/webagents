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
 *
 * THE MODEL IS DECIDED FOR OTHER CALLERS (S-327, B6; 2026-09-28). `serve`
 * had no model decision at all: a file naming no LLM skill served with no
 * model, silently, even with the key exported, and answered every request
 * 500 "No LLM skill available". It now decides as the chat does, for other
 * callers (`model-access.ts` `attachModelForCallers`): the provider key, or
 * Robutler's models on the agent's own platform credential, never the
 * sign-in. With neither, `serve` refuses to start with the sentence naming
 * both; `mcp serve` (`createServedAgent`), whose tools need no model, says it
 * and starts. A file naming a skill that does not exist is one sentence and
 * exit 1 too (B9), not a stack.
 */

import * as path from 'node:path';
import { effectiveMaxToolRounds } from '../core/tool-budget';
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
  /**
   * An agent with no model it can run for other callers (S-327): `refuse`
   * (`serve`, the default) stops with the sentence; `warn` (`mcp serve`,
   * whose tools need no model) says it on stderr and goes on.
   */
  noModel?: 'refuse' | 'warn';
  /**
   * Whose turns the agent runs (S-327). `true` (the default: `serve`, `mcp
   * serve --http`): other callers', so never the sign-in. `false`: the
   * owner's own (ACP, `mcp serve` over stdio), so the sign-in may pay, as in
   * the chat.
   */
  forCallers?: boolean;
}

/**
 * Why the default builder found no model for an agent's callers, by agent
 * (the sentence naming both ways out). A test's own `createAgent` records
 * nothing, so a seam never refuses.
 */
const MODEL_PROBLEMS = new WeakMap<object, string>();

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
  // The file's `sandbox:`, checked, for the shell to enforce (plan item 1.2).
  if (project.sandbox !== undefined) config.sandbox = project.sandbox;
  if (project.agentSkills !== undefined) config.agentSkills = project.agentSkills;
  // `fallback_models:` and `observability:` (plan items 2.8 and 2.4).
  if (project.fallbackModels !== undefined) config.fallbackModels = project.fallbackModels;
  if (project.observability !== undefined) config.observability = project.observability;
  if (project.maxToolRounds !== undefined) config.maxToolRounds = project.maxToolRounds;
  config.source = project.source;
  return config;
}

/**
 * The agent `serve` would run from `agentPath`, built and not yet started:
 * the file's skills and model, the stored provider keys, the folder, the
 * access block (2026-09-26). `webagents mcp serve` builds its agent through
 * this, so the agent an MCP client reaches is, by construction, the agent
 * `serve` would have put behind HTTP, rather than a second builder that
 * drifts from this one the way `serve` once drifted from the daemon.
 */
export async function createServedAgent(agentPath: string, options: { forCallers?: boolean } = {}): Promise<IAgent> {
  let built: IAgent | undefined;
  await serveAction(agentPath, { port: '0' }, {
    serve: async (agent) => {
      built = agent;
    },
    noModel: 'warn',
    forCallers: options.forCallers ?? true,
  });
  if (!built) throw new Error(`${agentPath}: no agent was built`);
  return built;
}

export async function serveAction(
  agentPath: string,
  options: ServeCommandOptions,
  deps: ServeCommandDeps = {},
): Promise<void> {
  // Before the agent is built, since building it reads its keys: `serve`
  // never waits on a macOS keychain dialog (keychain-ux, 2026-09-27).
  (await import('../skills/secrets/keychain-ux')).forbidKeychainDialogs('serve');
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
      const { readStoredProviderKeys, storedApiKeys } = await import('./provider-keys.js');
      const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
      // The one rule for every server-side builder (`storedApiKeys`): the
      // daemon and `cron run` read the same store since 2026-09-26.
      const apiKeys = await storedApiKeys(stored);
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
      // Robutler's socket for a listed `proxy` skill: for other callers with
      // `callersPay` and no sign-in, each call paid by the caller's own token
      // (S-327); for the owner's own turns (ACP, stdio MCP) the sign-in.
      const { platformLlmUrl } = await import('./model-access');
      const { resolvePlatformUrl } = await import('./config-store');
      const proxyUrl = platformLlmUrl(resolvePlatformUrl()[0]);
      const forCallers = deps.forCallers ?? true;
      const owner = forCallers
        ? undefined
        : await (async () => {
          const { getToken } = await import('./credentials');
          return {
            signedIn: async () => Boolean(await getToken()),
            platformToken: async () => (await getToken()) ?? undefined,
          };
        })();
      const { skills, byName, unknown, failed, skillmd } = await resolveSkillsByName(entries, {
        model: config.model as string | undefined,
        apiKeys,
        proxy: owner ? { proxyUrl, platformToken: owner.platformToken } : { proxyUrl, callersPay: true },
        agentDir,
        ...(config.sandbox !== undefined ? { sandbox: config.sandbox as import('../sandbox/policy.js').SandboxDeclaration } : {}),
        ...(Array.isArray(config.agentSkills) ? { agentSkills: config.agentSkills as string[] } : {}),
      });
      // SKILL.md skills that could not load are said, never fatal (plan item 1.4).
      const { skillmdReportLines } = await import('../skills/resolve.js');
      for (const line of skillmdReportLines(skillmd)) console.warn(line);

      if (unknown.length) {
        // Fatal, unlike the daemon's warn-and-continue: `serve` is a foreground
        // command run by a person who can fix the file, and starting an agent
        // that is missing a skill it declares is worse than refusing. An
        // `AgentFileError`, so the CLI prints the sentence and exits 1 rather
        // than a stack (B9, 2026-09-28), as the Python `serve` answers.
        const { AgentFileError: UnknownSkillError } = await import('../agents/index');
        throw new UnknownSkillError(
          `Unknown skill(s) in agent config: ${unknown.join(', ')}. ` +
          'Run `webagents skills list` for the available names.',
        );
      }
      for (const f of failed) {
        // Loaded but threw: usually a missing optional peer dependency. Name it
        // and continue, because the rest of the agent is still serviceable.
        console.warn(`Skill "${f.name}" failed to load: ${f.reason}`);
      }

      // THE MODEL, FOR OTHER CALLERS (S-327, B6): the key, or Robutler's
      // models on the agent's own credential, never the sign-in. It replaces
      // the "X is not set, so every request will fail" warning, which is now
      // either a model that runs or the refusal below.
      const { attachModelForCallers } = await import('./model-access');
      const agentName = (config.name as string) ?? 'agent';
      const ownCredential = async () => {
        const { resolveAgentCredential } = await import('../server/agent-credential');
        return Boolean(await resolveAgentCredential(agentName));
      };
      const decided = await attachModelForCallers({
        agentName,
        declaredSkills: declared,
        skills,
        byName,
        notBuilt: new Set([...unknown, ...failed.map((f) => f.name)]),
        model: config.model as string | undefined,
        apiKeys,
        env: { ...process.env, ...stored },
        proxyUrl,
        ownCredential,
        ...(owner ? { owner } : {}),
      });

      // `fallback_models:` (plan item 2.8): the model's skill becomes a chain.
      if (!decided.problem && Array.isArray(config.fallbackModels) && config.fallbackModels.length) {
        const { withFallbackModels } = await import('../skills/resolve.js');
        const chained = await withFallbackModels(skills, config.fallbackModels as string[], {
          primaryModel: decided.model ?? (config.model as string | undefined),
          apiKeys,
          env: { ...process.env, ...stored },
          forCallers,
          ...(owner
            ? ((await owner.signedIn()) ? { proxy: { proxyUrl, platformToken: owner.platformToken } } : {})
            : ((await ownCredential()) ? { proxy: { proxyUrl, callersPay: true } } : {})),
        });
        skills.splice(0, skills.length, ...chained.skills);
        for (const f of chained.failed) console.warn(`Skill "${f.name}" failed to load: ${f.reason}`);
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
        name: agentName,
        description: config.description as string,
        instructions: config.instructions as string,
        model: decided.model ?? (config.model as string | undefined),
        skills: access ? [...skills, access.skill] : skills,
        ...(config.observability !== undefined ? { observability: config.observability as { otel?: boolean } } : {}),
        // `--max-tool-rounds`, then the file's `max_tool_rounds`, then 50 (2026-09-28).
        maxToolIterations: effectiveMaxToolRounds(config.maxToolRounds as number | undefined).rounds,
      }) as unknown as IAgent;
      if (access) {
        try {
          applyAccessTools(access.policy, byName);
        } catch (err) {
          if (err instanceof AccessConfigError) throw new AgentFileError(`${config.source as string}: ${err.message}`);
          throw err;
        }
      }
      if (decided.problem) MODEL_PROBLEMS.set(agent, decided.problem);
      return agent;
    });

  const serveFn =
    deps.serve ??
    (async (agent: IAgent, config: { port: number; hostname?: string }) => {
      const { serve } = await import('../server/node.js');
      return serve(agent, config);
    });

  const agent = await createAgent(agentConfig);
  const problem = MODEL_PROBLEMS.get(agent);
  if (problem) {
    // The Python `serve`'s line (`cli/serve.py` `load_served_agent`), on
    // stderr; `serve` stops there, `mcp serve` goes on (S-327).
    const line = `[webagents] ${agent.name}: ${problem}`;
    if ((deps.noModel ?? 'refuse') === 'refuse') {
      const { AgentFileError } = await import('../agents/index');
      throw new AgentFileError(line);
    }
    console.warn(line);
  }

  // No `agent.initialize()` here: `serve()` owns that. See the module
  // docstring; `tests/unit/cli/serve-action.test.ts` pins the call count.
  await serveFn(agent, { port: parseInt(options.port, 10), hostname: options.host });
}
