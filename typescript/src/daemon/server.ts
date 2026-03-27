/**
 * WebAgents Daemon Server
 * 
 * Main daemon that manages agents, file watching, and cron jobs.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
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
    
    app.use('*', cors());
    
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
    
    // List cron jobs
    app.get('/cron', (c) => {
      return c.json({ jobs: this.scheduler.getJobs() });
    });
    
    // Add cron job
    app.post('/cron', async (c) => {
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
    app.delete('/cron/:id', (c) => {
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
      
      const skills = await this.resolveSkills(definition.skills || []);
      const agent = new BaseAgent({
        name: definition.name,
        description: definition.description,
        instructions: definition.instructions,
        model: definition.model,
        skills,
      });
      await agent.initialize();
      
      this.registry.registerLocal(agent);
    });
    
    this.watcher.on('agent:updated', async (definition) => {
      console.log(`Agent updated: ${definition.name}`);
      this.registry.unregister(definition.name);
      const skills = await this.resolveSkills(definition.skills || []);
      const agent = new BaseAgent({
        name: definition.name,
        description: definition.description,
        instructions: definition.instructions,
        model: definition.model,
        skills,
      });
      await agent.initialize();
      this.registry.registerLocal(agent);
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
   */
  private async resolveSkills(skillNames: string[]): Promise<import('../core/types.js').ISkill[]> {
    const skills: import('../core/types.js').ISkill[] = [];
    for (const name of skillNames) {
      try {
        const lower = name.toLowerCase();
        if (lower === 'openai') {
          const { OpenAISkill } = await import('../skills/llm/openai/skill.js');
          skills.push(new OpenAISkill());
        } else if (lower === 'anthropic' || lower === 'claude') {
          const { AnthropicSkill } = await import('../skills/llm/anthropic/skill.js');
          skills.push(new AnthropicSkill());
        } else if (lower === 'google' || lower === 'gemini') {
          const { GoogleSkill } = await import('../skills/llm/google/skill.js');
          skills.push(new GoogleSkill({}));
        } else if (lower === 'xai' || lower === 'grok') {
          const { XAISkill } = await import('../skills/llm/xai/skill.js');
          skills.push(new XAISkill());
        } else if (lower === 'discovery') {
          const { PortalDiscoverySkill } = await import('../skills/discovery/skill.js');
          skills.push(new PortalDiscoverySkill());
        } else {
          console.warn(`[daemon] Unknown skill: ${name}`);
        }
      } catch (err) {
        console.warn(`[daemon] Failed to load skill ${name}:`, (err as Error).message);
      }
    }
    return skills;
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
