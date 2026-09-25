/**
 * WebAgents Daemon Server
 * 
 * Main daemon that manages agents, file watching, and cron jobs.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { credentialFloor } from '../server/credential-floor';
import { isLoopbackAddress, replyText } from '../server/error-reply';
import { inboundRequest, refusalResponse } from '../server/handler';
import { AgentRegistry } from './registry';
import { AgentWatcher } from './watcher';
import { CronScheduler } from './cron';
import type { IAgent } from '../core/types';
import { BaseAgent } from '../core/agent';

/**
 * Daemon configuration
 */
export interface DaemonConfig {
  /** Port to listen on */
  port?: number;
  /** Hostname to bind to */
  hostname?: string;
  /** Directory to watch for agent files */
  watchDir?: string;
  /** Enable file watching */
  watch?: boolean;
  /** Enable cron scheduler */
  cron?: boolean;
  /** Enable health checks */
  healthChecks?: boolean;
  /** Health check interval (ms) */
  healthCheckInterval?: number;
  /**
   * Browser origins allowed to call this daemon. EMPTY BY DEFAULT.
   *
   * The daemon is a local control plane: it lists, registers and deregisters
   * agents and edits cron. It ran with `cors()` defaults, i.e.
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
  private scheduler: CronScheduler;
  private app: Hono;
  
  constructor(config: DaemonConfig = {}) {
    this.config = {
      port: 8080,
      hostname: '0.0.0.0',
      watch: true,
      cron: true,
      healthChecks: true,
      healthCheckInterval: 30000,
      ...config,
    };
    
    this.registry = new AgentRegistry();
    this.scheduler = new CronScheduler();
    this.app = this.createApp();
    
    // Set up file watcher
    if (this.config.watch && this.config.watchDir) {
      this.watcher = new AgentWatcher(this.config.watchDir);
      this.setupWatcher();
    }
    
    // Set up cron scheduler
    if (this.config.cron) {
      this.setupScheduler();
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
      let body: { messages?: unknown; stream?: boolean; model?: string };
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
      // The request, in session data only this route writes (ADR-0045).
      const runOptions = { sessionData: { _inboundRequest: inboundRequest(c.req.raw, raw) } };

      if (!body.stream) {
        try {
          const response = await entry.agent.run(messages, runOptions);
          return c.json(response);
        } catch (err) {
          const refusal = refusalResponse(err);
          if (refusal) return c.json(refusal.body, refusal.status);
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
        const refusal = refusalResponse(err);
        if (refusal) return c.json(refusal.body, refusal.status);
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

    // Get agent info
    app.get('/agents/:name', (c) => {
      const name = c.req.param('name');
      const agent = this.registry.get(name);
      
      if (!agent) {
        return c.json({ error: 'Agent not found' }, 404);
      }
      
      return c.json({
        name: agent.name,
        source: agent.source,
        url: agent.url,
        capabilities: agent.capabilities,
        healthy: agent.healthy,
        registeredAt: agent.registeredAt,
        lastActivity: agent.lastActivity,
      });
    });
    
    // Register remote agent
    app.post('/agents/register', async (c) => {
      const body = await c.req.json();
      
      if (!body.name || !body.url || !body.capabilities) {
        return c.json({ error: 'Missing required fields: name, url, capabilities' }, 400);
      }
      
      this.registry.registerRemote(body.name, body.url, body.capabilities, 'api');
      
      return c.json({ success: true });
    });
    
    // Unregister agent
    app.delete('/agents/:name', (c) => {
      const name = c.req.param('name');
      const removed = this.registry.unregister(name);
      
      if (!removed) {
        return c.json({ error: 'Agent not found' }, 404);
      }
      
      return c.json({ success: true });
    });
    
    // Cron lives under /agents to match `webagentsd` (2026-09-23).
    //
    // The Python daemon builds its router with `url_prefix="/agents"`, so its
    // cron routes are `/agents/cron`. These were `/cron`, which meant a client
    // could not be written against both daemons without first asking which one
    // had answered. `/cron` is kept as an alias so anything already calling it
    // keeps working.
    const cronRoutes = ['/agents/cron', '/cron'];

    // List cron jobs
    for (const route of cronRoutes) app.get(route, (c) => {
      return c.json({ jobs: this.scheduler.getJobs() });
    });
    
    // Add cron job
    for (const route of cronRoutes) app.post(route, async (c) => {
      const body = await c.req.json();
      
      if (!body.id || !body.cron || !body.agentName || !body.task) {
        return c.json({ error: 'Missing required fields: id, cron, agentName, task' }, 400);
      }
      
      this.scheduler.addJob({
        id: body.id,
        cron: body.cron,
        agentName: body.agentName,
        task: body.task,
        params: body.params,
        enabled: body.enabled ?? true,
      });
      
      return c.json({ success: true });
    });
    
    // Delete cron job
    for (const route of cronRoutes) app.delete(`${route}/:id`, (c) => {
      const id = c.req.param('id');
      const removed = this.scheduler.removeJob(id);
      
      if (!removed) {
        return c.json({ error: 'Job not found' }, 404);
      }
      
      return c.json({ success: true });
    });
    
    return app;
  }
  
  /**
   * Set up file watcher
   */
  private setupWatcher(): void {
    if (!this.watcher) return;
    
    this.watcher.on('agent:added', async (definition) => {
      console.log(`Agent discovered: ${definition.name}`);
      const agent = await this.buildAgent(definition);
      if (agent) this.registry.registerLocal(agent);
    });
    
    this.watcher.on('agent:updated', async (definition) => {
      console.log(`Agent updated: ${definition.name}`);
      this.registry.unregister(definition.name);
      const agent = await this.buildAgent(definition);
      if (agent) this.registry.registerLocal(agent);
    });
    
    this.watcher.on('agent:removed', (filePath) => {
      console.log(`Agent file removed: ${filePath}`);
      // In a real implementation, unregister the agent
    });
    
    this.watcher.on('error', (error) => {
      console.error('Watcher error:', error);
    });
  }
  
  /**
   * Set up cron scheduler
   */
  private setupScheduler(): void {
    this.scheduler.on('job:execute', async (job) => {
      console.log(`Executing cron job: ${job.id} for agent ${job.agentName}`);
      
      const agent = this.registry.get(job.agentName);
      if (!agent) {
        console.error(`Agent not found for job ${job.id}: ${job.agentName}`);
        return;
      }

      try {
        const localAgent = agent as { run?: Function };
        if (typeof localAgent.run === 'function') {
          const taskMessage = job.task
            ? `Execute task: ${job.task}${job.params ? ' with params: ' + JSON.stringify(job.params) : ''}`
            : 'Run scheduled task';
          const result = await localAgent.run([{ role: 'user', content: taskMessage }]);
          console.log(`Cron job ${job.id} completed:`, result?.content?.slice(0, 200));
        }
      } catch (err) {
        console.error(`Cron job ${job.id} failed:`, (err as Error).message);
      }
    });
  }
  
  /**
   * Resolve skill names from AGENT.md frontmatter into skill instances.
   *
   * The agent's `model` now reaches its LLM skill (2026-09-24): each was built
   * with no config and fell back to its own hardcoded model, so `model:` in
   * the file never applied. Same fix as `skills/resolve.ts`, kept to the same
   * set of names as before on purpose: this daemon still binds every interface
   * by default (S-214), so teaching it `shell` is not a side effect to take.
   */
  /**
   * An agent from its file, the way `serve` builds one (2026-09-25): the
   * shared skill resolver, so `filesystem`, `shell`, `rest` and skill config
   * load here too, and the file's `access:` block installed (ADR-0045), so a
   * block that keeps callers out keeps them out of the daemon as well. A file
   * whose block is malformed is not served, and the daemon says why.
   */
  private async buildAgent(definition: import('./watcher.js').AgentDefinition): Promise<BaseAgent | null> {
    const { resolveSkillsByName } = await import('../skills/resolve.js');
    const { AccessConfigError } = await import('../access/policy.js');
    const { accessSkillFor, applyAccessTools } = await import('../access/install.js');
    const entries = definition.skillEntries ?? definition.skills ?? [];
    const { dirname } = await import('node:path');
    const { skills, byName, unknown, failed } = await resolveSkillsByName(entries, {
      model: definition.model,
      agentDir: dirname(definition.filePath),
    });
    for (const name of unknown) console.warn(`[daemon] Unknown skill "${name}" in ${definition.filePath}; skipping.`);
    for (const f of failed) console.warn(`[daemon] Skill "${f.name}" failed to load: ${f.reason}`);
    let access: ReturnType<typeof accessSkillFor> | undefined;
    try {
      access = definition.access !== undefined ? accessSkillFor(definition.access, definition.filePath) : undefined;
      const agent = new BaseAgent({
        name: definition.name,
        description: definition.description,
        instructions: definition.instructions,
        model: definition.model,
        skills: access ? [...skills, access.skill] : skills,
      });
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
  
  /**
   * Get the scheduler
   */
  getScheduler(): CronScheduler {
    return this.scheduler;
  }
  
  /**
   * Start the daemon
   */
  async start(): Promise<void> {
    // Start file watcher
    if (this.watcher) {
      this.watcher.start();
    }
    
    // Start cron scheduler
    if (this.config.cron) {
      this.scheduler.start();
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
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const { serve } = await import('@hono/node-server' as any);
        serve({
          fetch: this.app.fetch,
          port,
          hostname,
        });
      } catch {
        console.error('Failed to start daemon. Install @hono/node-server for Node.js support.');
        throw new Error('No compatible server runtime found');
      }
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
    
    this.scheduler.stop();
    this.registry.stopHealthChecks();
    
    console.log('WebAgents daemon stopped');
  }
}

// Bun type declaration
declare const Bun: {
  serve(options: { port: number; hostname: string; fetch: (request: Request) => Response | Promise<Response> }): void;
} | undefined;
