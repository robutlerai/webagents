/**
 * WebAgents Daemon Server
 *
 * Main daemon that manages agents, file watching, and the agents' `cron:`
 * schedules.
 *
 * SCHEDULES COME FROM AGENT FILES ONLY (S-273, 2026-09-26). This daemon
 * took `POST /agents/cron` (and `/cron`) from any caller with `{id, cron,
 * agentName, task}` and then ran the served agent, with its tools and on
 * the owner's model key, with the caller's `task` as the prompt, every time
 * the job fired; the credential floor gates only the billable paths, so a
 * daemon on `--host 0.0.0.0` let anyone on the network schedule any prompt
 * on any served agent. The add and remove routes are gone. `GET` lists what
 * the files declare, with the runner's state (`schedule-runner.ts`), and a
 * schedule is added by writing it into the agent file the daemon watches.
 */

import { Hono } from 'hono';
import { effectiveMaxToolRounds } from '../core/tool-budget';
import { cors } from 'hono/cors';
import * as path from 'node:path';
import { credentialFloor, hasCredential, unauthorizedResponse } from '../server/credential-floor';
import { isLoopbackAddress, replyText } from '../server/error-reply';
import { refusalResponse, servedRunOptions } from '../server/handler';
import { AgentRegistry } from './registry';
import { AgentWatcher, type AgentDefinition } from './watcher';
import { ScheduleRunner } from './schedule-runner';
import { daemonAgentIdentity, daemonPublicUrl } from './agent-identity';
import type { IAgent } from '../core/types';
import { BaseAgent } from '../core/agent';
import type { CronSchedule } from '../agents/schedules';
import type { SigningIdentity } from '../crypto/http-signature';

/**
 * Daemon configuration
 */
export interface DaemonConfig {
  /** Port to listen on */
  port?: number;
  /** Hostname to bind to */
  hostname?: string;
  /**
   * The folder whose agents are served, and kept current as their files
   * change (`watcher.ts`). Default: the working directory, as the Python
   * daemon (`create_server(watch_dirs=None)`) and `webagents daemon` without
   * `-w` in both CLIs.
   */
  watchDir?: string;
  /** Serve and watch `watchDir`'s agents (default true). */
  watch?: boolean;
  /** Run the served agents' `cron:` schedules (default true). */
  cron?: boolean;
  /**
   * The address this daemon publishes for its agents (`agent-identity.ts`):
   * each served agent signs as `{publicUrl}/agents/{name}` and its key set
   * is served at `/agents/{name}/.well-known/jwks.json`. Default:
   * `WEBAGENTS_PUBLIC_URL`, else the daemon's own bind address, which the
   * signer refuses (loopback, plain http), so webhooks then go out unsigned
   * and the run's record says why.
   */
  publicUrl?: string;
  /** Enable health checks */
  healthChecks?: boolean;
  /** Health check interval (ms) */
  healthCheckInterval?: number;
  /**
   * Browser origins allowed to call this daemon. EMPTY BY DEFAULT.
   *
   * The daemon is a local control plane: it lists, registers and deregisters
   * agents and lists schedules. It ran with `cors()` defaults, i.e.
   * `Access-Control-Allow-Origin: *`, so any page the developer happened to
   * have open could drive it. Opt in explicitly instead.
   */
  allowedOrigins?: string[];
}

/**
 * WebAgents Daemon
 */
export class WebAgentsDaemon {
  private config: DaemonConfig;
  private registry: AgentRegistry;
  private watcher: AgentWatcher | null = null;
  /** The served agents' `cron:` schedules and their state (`schedule-runner.ts`). */
  private runner: ScheduleRunner;
  private app: Hono;
  /** Which file each served agent came from, by name (two files may declare one name). */
  private servedFrom: Map<string, string> = new Map();
  /** Agents being built from their files, awaited before the daemon answers. */
  private building: Set<Promise<void>> = new Set();

  constructor(config: DaemonConfig = {}) {
    this.config = {
      port: 8080,
      // Loopback by default, as the Python daemon binds (S-284, 2026-09-26).
      // It defaulted to `0.0.0.0`, publishing a control plane that could
      // deregister the owner's agents and add remote ones to the whole
      // network; exposing it is now one `--host` away, and no longer the
      // default. The `webagents daemon` command already passes `daemon.host`
      // (127.0.0.1), so this only changes an embedder that constructs the
      // daemon directly with no hostname.
      hostname: '127.0.0.1',
      watch: true,
      cron: true,
      healthChecks: true,
      healthCheckInterval: 30000,
      ...config,
    };

    this.registry = new AgentRegistry();
    // A schedule runs the agent this daemon serves under the name, and is
    // "not served" while the file is gone or failed to build.
    this.runner = new ScheduleRunner({ agentFor: (name) => this.registry.get(name)?.agent });
    this.app = this.createApp();

    // The agents under the folder (`watchDir`, else the working directory).
    if (this.config.watch) {
      this.watcher = new AgentWatcher(this.config.watchDir ?? process.cwd());
      this.setupWatcher();
    }
  }
  
  /**
   * Create the Hono app
   */
  private createApp(): Hono {
    const app = new Hono();

    // ==========================================================================
    // THE CREDENTIAL FLOOR, FIRST (2026-09-23, logged as S-214).
    //
    // This daemon had no floor at all, unlike `server/node.ts:126-132`. That was
    // survivable only because it served nothing billable: the routes were
    // health, agent CRUD and cron. The inference route below changes that, so
    // the floor lands in the same commit rather than after it.
    //
    // It decides from method and path alone, above route dispatch and above any
    // `await c.req.json()`, so an anonymous caller cannot make the daemon parse
    // arbitrary bytes.
    // ==========================================================================
    app.use('*', async (c, next) => {
      const refusal = credentialFloor(c.req.raw);
      if (refusal) return refusal;
      await next();
      return undefined;
    });

    // CORS is scoped, not wide open. `cors()` with defaults answers
    // `Access-Control-Allow-Origin: *`, which let any page in the developer's
    // browser enumerate and deregister their local agents and edit cron. The
    // daemon is a local control plane; a browser origin has no business calling
    // it unless the operator says so.
    app.use('*', cors({
      origin: this.config.allowedOrigins ?? [],
      allowMethods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
    }));

    // Health check
    app.get('/health', (c) => {
      return c.json({ status: 'ok', stats: this.registry.getStats() });
    });
    
    // List agents
    app.get('/agents', (c) => {
      return c.json({
        agents: this.registry.getAll().map(a => ({
          name: a.name,
          source: a.source,
          url: a.url,
          capabilities: a.capabilities,
          healthy: a.healthy,
        })),
      });
    });

    // Cron lives under /agents to match the Python daemon (2026-09-23).
    //
    // The Python daemon builds its router with `url_prefix="/agents"`, so its
    // cron route is `/agents/cron`. These were `/cron`, which meant a client
    // could not be written against both daemons without first asking which one
    // had answered. `/cron` is kept as an alias so anything already calling it
    // keeps working.
    //
    // BEFORE `/agents/:name`, which otherwise answers `/agents/cron` with
    // "Agent not found" (Hono dispatches to the first route that matches;
    // the route sat below it until 2026-09-26 and was unreachable).
    //
    // READ-ONLY (S-273, file comment): the schedules of every served agent as
    // their files declare them, each with its next fire and last run. There
    // is no route that adds or removes one.
    for (const route of ['/agents/cron', '/cron']) app.get(route, (c) => {
      return c.json({ schedules: this.runner.list() });
    });

    // ==========================================================================
    // Inference. The daemon could list a locally registered agent and never let
    // anyone talk to it: there was no chat/completions, no streaming, no UAMP
    // route of any kind (2026-09-23). A registry you cannot address is a
    // directory, not a daemon.
    //
    // Path matches the Python daemon so one client library works against both.
    // `chat/completions` is already in BILLABLE_PATHS, so the floor above
    // gates this by suffix match without further wiring.
    // ==========================================================================
    app.post('/agents/:name/chat/completions', async (c) => {
      const name = c.req.param('name');
      const entry = this.registry.get(name);

      if (!entry?.agent) {
        // A remote entry has a `url` and no instance; say which case this is
        // rather than a bare 404.
        return c.json(
          { error: entry ? 'Agent is registered remotely; call its url directly' : 'Agent not found' },
          404,
        );
      }

      // The bytes first, then the JSON: an agent file's access block checks a
      // signed request's Content-Digest against exactly what arrived (ADR-0045).
      let raw: Uint8Array;
      let body: { messages?: unknown; stream?: boolean; model?: string; metadata?: unknown };
      try {
        raw = new Uint8Array(await c.req.arrayBuffer());
        body = JSON.parse(new TextDecoder().decode(raw));
      } catch {
        return c.json({ error: 'Body must be JSON' }, 400);
      }
      if (!Array.isArray(body.messages)) {
        return c.json({ error: '`messages` must be an array' }, 400);
      }

      this.registry.updateActivity(name);
      const messages = body.messages as Parameters<IAgent['run']>[0];
      // A failed run's own text only while this daemon listens on loopback,
      // where the caller is the developer's own CLI; the fixed sentence and a
      // logged reference otherwise, as every served agent answers (S-228).
      const detail = isLoopbackAddress(this.config.hostname);
      // The request on the run (S-345, 2026-09-29): the credential headers on
      // its metadata, where the auth skill reads them, the body's `metadata`
      // under them, and the request itself in session data only this route
      // writes (ADR-0045). This passed the session data alone, so an auth
      // skill on a served agent saw no credential and refused every caller,
      // the owner included, while an agent without one ran for any credential
      // string the floor let through. The same builder as every served route.
      const runOptions = servedRunOptions(c.req.raw, raw, body.metadata);

      if (!body.stream) {
        try {
          const response = await entry.agent.run(messages, runOptions);
          return c.json(response);
        } catch (err) {
          // With the bearer challenge on a 401 (2026-09-29, `refusalResponse`).
          const refusal = refusalResponse(err, c.req.raw);
          if (refusal) return c.json(refusal.body, refusal.status, refusal.headers);
          return c.json({ error: replyText(err, `${name} chat/completions`, { detail }) }, 500);
        }
      }

      // SSE, in the same shape the Python daemon emits, so `DaemonClient`
      // consumes both identically.
      const encoder = new TextEncoder();
      const agent = entry.agent;
      // The first chunk is pulled here, so a refusal is a status and not a
      // 200 whose stream happens to carry an error, as `serve` answers.
      const chunks = agent.runStreaming(messages, runOptions);
      let first: IteratorResult<unknown> | undefined;
      let firstError: unknown;
      try {
        first = await chunks.next();
      } catch (err) {
        const refusal = refusalResponse(err, c.req.raw);
        if (refusal) return c.json(refusal.body, refusal.status, refusal.headers);
        // Anything else fails inside the stream, as it always has.
        firstError = err;
      }
      const stream = new ReadableStream({
        async start(controller) {
          try {
            if (firstError !== undefined) throw firstError;
            if (first && !first.done) controller.enqueue(encoder.encode(`data: ${JSON.stringify(first.value)}\n\n`));
            for await (const chunk of chunks) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            }
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          } catch (err) {
            const message = replyText(err, `${name} chat/completions`, { detail });
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: message })}\n\n`));
          } finally {
            controller.close();
          }
        },
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
      });
    });

    // The served agent's key set, where `serve()` serves one (2026-09-27,
    // `agent-identity.ts`): `{agent URL}/.well-known/jwks.json`, so a webhook
    // signed as `{publicUrl}/agents/{name}` verifies against a key set
    // fetched from this very path. The same headers as `server/handler.ts`.
    // Public, as every key set is; an agent that holds no key answers 404.
    app.get('/agents/:name/.well-known/jwks.json', (c) => {
      const entry = this.registry.get(c.req.param('name'));
      const identity = entry?.agent?.identity as { getJwks?: () => unknown } | undefined;
      if (!identity || typeof identity.getJwks !== 'function') {
        return c.json({ error: 'No signing identity' }, 404);
      }
      return c.json(identity.getJwks() as Record<string, unknown>, 200, { 'Cache-Control': 'public, max-age=3600' });
    });

    // Get agent info
    app.get('/agents/:name', async (c) => {
      const name = c.req.param('name');
      const agent = this.registry.get(name);

      if (!agent) {
        return c.json({ error: 'Agent not found' }, 404);
      }

      const { describeSchedule } = await import('../agents/schedules.js');
      return c.json({
        name: agent.name,
        source: agent.source,
        url: agent.url,
        capabilities: agent.capabilities,
        healthy: agent.healthy,
        registeredAt: agent.registeredAt,
        lastActivity: agent.lastActivity,
        // The file's `cron:` schedules, as the shared fixture spells them.
        schedules: this.schedulesOf(name).map(describeSchedule),
      });
    });
    
    // Register remote agent. These two routes mutate the registry, so they
    // take the credential the floor already demands for the billable route
    // (S-284, 2026-09-26): the floor gates only `chat/completions`, so
    // register and remove answered anyone who could reach the port, which on
    // the old `0.0.0.0` default was the whole network. `hasCredential` is the
    // floor's own check; the local CLI reaches neither route (its
    // `DaemonClient` reads `list`/`status`/`logs` only), so gating them breaks
    // no CLI flow.
    app.post('/agents/register', async (c) => {
      if (!hasCredential(c.req.raw)) return unauthorizedResponse();
      const body = await c.req.json();

      if (!body.name || !body.url || !body.capabilities) {
        return c.json({ error: 'Missing required fields: name, url, capabilities' }, 400);
      }

      this.registry.registerRemote(body.name, body.url, body.capabilities, 'api');

      return c.json({ success: true });
    });

    // Unregister agent
    app.delete('/agents/:name', (c) => {
      if (!hasCredential(c.req.raw)) return unauthorizedResponse();
      const name = c.req.param('name');
      const removed = this.registry.unregister(name);

      if (!removed) {
        return c.json({ error: 'Agent not found' }, 404);
      }

      return c.json({ success: true });
    });

    return app;
  }
  
  /**
   * Serve what the watcher finds (`watcher.ts`): an agent per file, rebuilt
   * when its file changes and let go when the file goes. Two files declaring
   * one name: the later one is served, and the daemon says so, as the Python
   * registry does.
   */
  private setupWatcher(): void {
    if (!this.watcher) return;

    const track = (job: Promise<void>) => {
      this.building.add(job);
      void job.finally(() => this.building.delete(job));
    };
    const letGo = (previous: AgentDefinition) => {
      if (this.servedFrom.get(previous.name) !== previous.filePath) return;
      this.registry.unregister(previous.name);
      this.servedFrom.delete(previous.name);
      this.runner.removeAgent(previous.name);
    };

    this.watcher.on('agent:added', (definition: AgentDefinition) => track(this.serveDefinition(definition)));
    this.watcher.on('agent:updated', (definition: AgentDefinition, previous: AgentDefinition) => {
      if (previous.name !== definition.name) letGo(previous);
      track(this.serveDefinition(definition));
    });
    this.watcher.on('agent:removed', (_filePath: string, previous: AgentDefinition) => letGo(previous));
    this.watcher.on('error', (error) => {
      console.error('Watcher error:', error);
    });
  }

  /** Build the agent a file declares and serve it under its name. */
  private async serveDefinition(definition: AgentDefinition): Promise<void> {
    // The file's `cron:` block, checked before the agent is built (plan item
    // 1.7, 2026-09-26): a malformed block stops the file with its sentence,
    // as a malformed access block does, rather than serving an agent whose
    // schedules silently do not exist. The Python loader refuses the same
    // file at parse (`cli/loader/schema.py`).
    const { parseCronBlock } = await import('../agents/schedules.js');
    const { AgentFileError } = await import('../agents/index.js');
    let schedules: CronSchedule[] = [];
    try {
      schedules = definition.cron === undefined ? [] : parseCronBlock(definition.cron);
    } catch (err) {
      if (err instanceof AgentFileError) {
        console.error(`[daemon] ${definition.filePath}: ${err.message} The agent is not served.`);
        return;
      }
      throw err;
    }
    // The agent's signing identity (`agent-identity.ts`), found or created in
    // the agent folder's `.webagents/keys` before the agent is built, so a
    // skill that signs on boot can. A key file that exists and cannot be used
    // is said and the agent is served unsigned; it is never replaced.
    let identity: SigningIdentity | undefined;
    try {
      identity = await daemonAgentIdentity(definition, this.publicUrl());
    } catch (err) {
      console.error(`[daemon] ${definition.filePath}: ${(err as Error).message} The agent is served without a signing identity.`);
    }
    const agent = await this.buildAgent(definition, identity);
    if (!agent) return;
    const before = this.servedFrom.get(definition.name);
    if (before !== undefined && before !== definition.filePath) {
      // The Python registry's words (`cli/daemon/registry.py`).
      console.warn(`agent '${definition.name}' is declared by two files; ${definition.filePath} now replaces ${before}`);
    }
    this.registry.unregister(definition.name);
    this.registry.registerLocal(agent);
    this.servedFrom.set(definition.name, definition.filePath);
    // The runner keeps a schedule's state across a rebuild of its file: an
    // edit to the instructions does not reset the next fire.
    this.runner.setSchedules(definition.name, path.dirname(definition.filePath), schedules);
  }

  /** An agent from its file, as `buildDefinedAgent` builds one, holding `identity` when it has one. */
  private async buildAgent(definition: AgentDefinition, identity?: SigningIdentity): Promise<BaseAgent | null> {
    return buildDefinedAgent(definition, { identity });
  }

  /** The address this daemon publishes for its agents (`DaemonConfig.publicUrl`). */
  publicUrl(): string {
    return daemonPublicUrl({ hostname: this.config.hostname!, port: this.config.port!, publicUrl: this.config.publicUrl });
  }

  /** The `cron:` schedules of a served agent, as its file declares them; none for an unknown name. */
  schedulesOf(name: string): readonly CronSchedule[] {
    return this.runner
      .entries()
      .filter((entry) => entry.agentName === name)
      .map((entry) => entry.schedule);
  }

  /**
   * Find the folder's agents and build them, and keep watching: what
   * `start()` does before it answers, so the first request finds them.
   */
  async discover(): Promise<void> {
    this.watcher?.start();
    while (this.building.size) await Promise.allSettled([...this.building]);
  }

  /**
   * Register an agent
   */
  registerAgent(agent: IAgent): void {
    this.registry.registerLocal(agent);
  }

  /**
   * Get the registry
   */
  getRegistry(): AgentRegistry {
    return this.registry;
  }

  /** The runner of the served agents' `cron:` schedules. */
  getScheduleRunner(): ScheduleRunner {
    return this.runner;
  }

  /**
   * Start the daemon
   */
  async start(): Promise<void> {
    // A daemon never waits on a macOS keychain dialog: nobody may be there to
    // answer, and it would be on the screen at a random later moment. A read
    // that may ask is refused with the one sentence instead (keychain-ux,
    // 2026-09-27). Before the agents are built, since building reads keys.
    (await import('../skills/secrets/keychain-ux')).forbidKeychainDialogs('daemon');
    // The folder's agents, built before the first request can arrive.
    await this.discover();

    // The schedules, ticking from now (`--no-cron` leaves them listed and idle).
    if (this.config.cron) {
      this.runner.start();
    }
    
    // Start health checks
    if (this.config.healthChecks) {
      this.registry.startHealthChecks(this.config.healthCheckInterval);
    }
    
    // Start HTTP server
    const port = this.config.port!;
    const hostname = this.config.hostname!;
    
    console.log(`WebAgents daemon starting on http://${hostname}:${port}`);
    
    // Use Bun or Node.js server
    if (typeof Bun !== 'undefined') {
      Bun.serve({
        port,
        hostname,
        fetch: this.app.fetch,
      });
    } else {
      let nodeServe: (
        options: { fetch: unknown; port: number; hostname: string },
        onListening?: (info: { port: number }) => void,
      ) => { on?: (event: string, listener: (error: unknown) => void) => void };
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ({ serve: nodeServe } = await import('@hono/node-server' as any));
      } catch {
        console.error('Failed to start daemon. Install @hono/node-server for Node.js support.');
        throw new Error('No compatible server runtime found');
      }
      // Bound, or one sentence for a port already in use (`server/listen-error.ts`,
      // 2026-09-26): the daemon died with an unhandled 'error' event and a
      // stack trace, as `serve` did.
      const { listenError } = await import('../server/listen-error');
      await new Promise<void>((resolve, reject) => {
        const listener = nodeServe({ fetch: this.app.fetch, port, hostname }, () => resolve());
        listener.on?.('error', (error: unknown) => reject(listenError(error, hostname, port)));
      });
    }
    
    console.log('WebAgents daemon started');
  }
  
  /**
   * Stop the daemon
   */
  stop(): void {
    if (this.watcher) {
      this.watcher.stop();
    }

    this.runner.stop();
    this.registry.stopHealthChecks();

    console.log('WebAgents daemon stopped');
  }
}

/**
 * An agent from its file, the way `serve` builds one (2026-09-25): the
 * shared skill resolver, so `filesystem`, `shell`, `rest` and skill config
 * load here too, and the file's `access:` block installed (ADR-0045), so a
 * block that keeps callers out keeps them out of the daemon as well. A file
 * whose block is malformed is not served, and the daemon says why.
 *
 * The agent's `model` reaches its LLM skill (2026-09-24): each was built
 * with no config and fell back to its own hardcoded model, so `model:` in
 * the file never applied.
 *
 * Exported for `webagents cron run` (`cli/cron-action.ts`), which builds
 * the one agent a schedule names exactly as this daemon would.
 *
 * `options.identity` is the agent's signing identity (`agent-identity.ts`),
 * set on the agent BEFORE `initialize()` as `serve()` sets it, so a skill
 * that signs on boot can, and the webhook deliverer and the A2A card find
 * it where a served agent keeps it (`agent.identity`).
 *
 * THE MODEL IS DECIDED FOR OTHER CALLERS (S-327, 2026-09-28), as `serve`
 * decides it (`cli/model-access.ts` `attachModelForCallers`): the provider
 * key, or Robutler's models on the agent's own platform credential with no
 * sign-in, each call paid by the caller's payment token. There was no
 * decision here at all, so a file naming no LLM skill had no model even with
 * its key stored. An agent with neither is not served, and the daemon says
 * why, naming both ways out, as it does for a malformed `access:` block.
 */
export async function buildDefinedAgent(definition: AgentDefinition, options: { identity?: SigningIdentity } = {}): Promise<BaseAgent | null> {
  const { resolveSkillsByName } = await import('../skills/resolve.js');
  const { AccessConfigError } = await import('../access/policy.js');
  const { accessSkillFor, applyAccessTools } = await import('../access/install.js');
  // Keys kept with `webagents secrets set` reach the model client here as
  // they do in the chat and in `serve` (2026-09-26, the e2e run): without
  // them an agent that answered in the chat failed under the daemon and
  // `cron run` with "OpenAI API key not configured". Handed to the client
  // only, never put in `process.env` (`cli/provider-keys.ts` says why).
  const { storedApiKeys } = await import('../cli/provider-keys');
  const apiKeys = await storedApiKeys();
  const entries = definition.skillEntries ?? definition.skills ?? [];
  // Robutler's socket, for a listed `proxy` skill and the decision below:
  // `callersPay`, never a sign-in (S-327).
  const { attachModelForCallers, platformLlmUrl } = await import('../cli/model-access');
  const { resolvePlatformUrl } = await import('../cli/config-store');
  const { readStoredProviderKeys } = await import('../cli/provider-keys');
  const proxyUrl = platformLlmUrl(resolvePlatformUrl()[0]);
  const { skills, byName, unknown, failed, skillmd } = await resolveSkillsByName(entries, {
    model: definition.model,
    apiKeys,
    proxy: { proxyUrl, callersPay: true },
    agentDir: path.dirname(definition.filePath),
    ...(definition.sandbox !== undefined ? { sandbox: definition.sandbox } : {}),
    ...(definition.agentSkills !== undefined ? { agentSkills: definition.agentSkills } : {}),
  });
  for (const name of unknown) console.warn(`[daemon] Unknown skill "${name}" in ${definition.filePath}; skipping.`);
  for (const f of failed) console.warn(`[daemon] Skill "${f.name}" failed to load: ${f.reason}`);
  const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
  const ownCredential = async () => {
    const { resolveAgentCredential } = await import('../server/agent-credential');
    return Boolean(await resolveAgentCredential(definition.name));
  };
  const decided = await attachModelForCallers({
    agentName: definition.name,
    declaredSkills: (definition.skills ?? []).map(String),
    skills,
    byName,
    notBuilt: new Set([...unknown, ...failed.map((f) => f.name)]),
    model: definition.model,
    apiKeys,
    env: { ...process.env, ...stored },
    proxyUrl,
    ownCredential,
  });
  if (decided.problem) {
    console.error(`[daemon] ${definition.filePath}: ${decided.problem} The agent is not served.`);
    return null;
  }
  // `fallback_models:` (plan item 2.8): the model's skill becomes a chain.
  if (definition.fallbackModels?.length) {
    const { withFallbackModels } = await import('../skills/resolve.js');
    const chained = await withFallbackModels(skills, definition.fallbackModels, {
      primaryModel: decided.model ?? definition.model,
      apiKeys,
      env: { ...process.env, ...stored },
      forCallers: true,
      ...((await ownCredential()) ? { proxy: { proxyUrl, callersPay: true } } : {}),
    });
    skills.splice(0, skills.length, ...chained.skills);
    for (const f of chained.failed) console.warn(`[daemon] Skill "${f.name}" failed to load: ${f.reason}`);
  }
  // SKILL.md skills that could not load are said, never fatal (plan item 1.4).
  const { skillmdReportLines } = await import('../skills/resolve.js');
  for (const line of skillmdReportLines(skillmd)) console.warn(`[daemon] ${line}`);
  let access: ReturnType<typeof accessSkillFor> | undefined;
  try {
    access = definition.access !== undefined ? accessSkillFor(definition.access, definition.filePath) : undefined;
    const agent = new BaseAgent({
      name: definition.name,
      description: definition.description,
      instructions: definition.instructions,
      model: decided.model ?? definition.model,
      skills: access ? [...skills, access.skill] : skills,
      ...(definition.observability !== undefined ? { observability: definition.observability } : {}),
      // `--max-tool-rounds`, then the file's `max_tool_rounds`, then 50 (2026-09-28).
      maxToolIterations: effectiveMaxToolRounds(definition.maxToolRounds).rounds,
    });
    if (options.identity) agent.identity = options.identity;
    agent.applyCompactionPolicy(definition.compaction);
    if (access) applyAccessTools(access.policy, byName);
    await agent.initialize();
    return agent;
  } catch (err) {
    if (err instanceof AccessConfigError) {
      console.error(`[daemon] ${definition.filePath}: ${err.message} The agent is not served.`);
      return null;
    }
    throw err;
  }
}

// Bun type declaration
declare const Bun: {
  serve(options: { port: number; hostname: string; fetch: (request: Request) => Response | Promise<Response> }): void;
} | undefined;
