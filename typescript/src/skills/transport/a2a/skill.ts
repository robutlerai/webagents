/**
 * A2A (Agent-to-Agent) Transport Skill
 *
 * Implements the Agent-to-Agent protocol for structured inter-agent
 * communication. Unlike NLI (natural language), A2A uses typed
 * task objects with explicit schemas for reliable machine-to-machine
 * agent interaction.
 *
 * Supports:
 * - Task creation and delegation
 * - Task status tracking and updates
 * - Structured input/output schemas
 * - Agent card (.well-known/agent.json) serving
 * - Push notification delivery
 */

import { Skill } from '../../../core/skill.js';
import { tool, hook } from '../../../core/decorators.js';
import type { Context, HookData } from '../../../core/types.js';

export interface A2ATransportConfig {
  name?: string;
  enabled?: boolean;
  /** Agent card info */
  agentCard?: AgentCard;
  /** Base URL for this agent */
  baseUrl?: string;
  /** API key for outbound requests */
  apiKey?: string;
  /** Timeout for A2A requests (ms) */
  timeout?: number;
  /** Push notification URL */
  pushUrl?: string;
}

export interface AgentCard {
  name: string;
  description: string;
  url: string;
  version?: string;
  capabilities?: {
    streaming?: boolean;
    pushNotifications?: boolean;
    stateTransitionHistory?: boolean;
  };
  skills?: Array<{
    id: string;
    name: string;
    description: string;
    inputSchema?: Record<string, unknown>;
    outputSchema?: Record<string, unknown>;
  }>;
  authentication?: {
    schemes: string[];
  };
}

export type A2ATaskStatus = 'submitted' | 'working' | 'input-required' | 'completed' | 'failed' | 'canceled';

export interface A2ATask {
  id: string;
  sessionId?: string;
  status: { state: A2ATaskStatus; message?: string; timestamp: string };
  artifacts?: Array<{
    name: string;
    parts: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
  }>;
  history?: Array<{ role: string; parts: Array<{ type: string; text?: string }> }>;
}

interface HttpResponse {
  writeHead(status: number, headers?: Record<string, string>): void;
  end(body?: string): void;
}

export class A2ATransportSkill extends Skill {
  private agentCard?: AgentCard;
  private apiKey?: string;
  private timeout: number;
  private tasks = new Map<string, A2ATask>();

  constructor(config: A2ATransportConfig = {}) {
    super({ ...config, name: config.name || 'a2a-transport' });
    this.agentCard = config.agentCard;
    this.apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30_000;
  }

  @hook({ lifecycle: 'on_connection', priority: 10 })
  async handleA2ARequest(data: HookData, context: Context): Promise<void> {
    const req = data.request as { method?: string } | undefined;
    const res = data.metadata?.response as HttpResponse | undefined;
    if (!req || !res) return;

    const path = (data.metadata?.path as string) ?? '';

    if (path === '/.well-known/agent.json' && this.agentCard) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(this.agentCard));
      return;
    }

    if (path === '/a2a' && req.method === 'POST') {
      const body = await this.readBody(req);
      const jsonrpc = JSON.parse(body);
      const result = await this.handleJsonRpc(jsonrpc, context);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    }
  }

  private async handleJsonRpc(
    msg: { id: string | number; method: string; params?: Record<string, unknown> },
    _context: Context,
  ): Promise<Record<string, unknown>> {
    switch (msg.method) {
      case 'tasks/send':
        return { jsonrpc: '2.0', id: msg.id, result: await this.handleTaskSend(msg.params ?? {}) };
      case 'tasks/get':
        return { jsonrpc: '2.0', id: msg.id, result: await this.handleTaskGet(msg.params ?? {}) };
      case 'tasks/cancel':
        return { jsonrpc: '2.0', id: msg.id, result: await this.handleTaskCancel(msg.params ?? {}) };
      default:
        return {
          jsonrpc: '2.0', id: msg.id,
          error: { code: -32601, message: `Method not found: ${msg.method}` },
        };
    }
  }

  private async handleTaskSend(params: Record<string, unknown>): Promise<A2ATask> {
    const taskId = (params.id as string) ?? crypto.randomUUID();
    const now = new Date().toISOString();

    const task: A2ATask = {
      id: taskId,
      sessionId: params.sessionId as string | undefined,
      status: { state: 'submitted', timestamp: now },
      history: params.message ? [params.message as NonNullable<A2ATask['history']>[number]] : [],
    };
    this.tasks.set(taskId, task);

    // Simulate processing — in production this would dispatch to the agentic loop
    task.status = { state: 'working', timestamp: new Date().toISOString() };

    return task;
  }

  private async handleTaskGet(params: Record<string, unknown>): Promise<A2ATask | { error: string }> {
    const task = this.tasks.get(params.id as string);
    if (!task) return { error: 'Task not found' };
    return task;
  }

  private async handleTaskCancel(params: Record<string, unknown>): Promise<A2ATask | { error: string }> {
    const task = this.tasks.get(params.id as string);
    if (!task) return { error: 'Task not found' };
    task.status = { state: 'canceled', timestamp: new Date().toISOString() };
    return task;
  }

  // ============================================================================
  // Client-side tools for sending A2A requests to other agents
  // ============================================================================

  @tool({
    name: 'a2a_send_task',
    description: 'Send a task to another agent via the A2A protocol.',
    parameters: {
      type: 'object',
      properties: {
        agent_url: { type: 'string', description: 'Target agent URL' },
        message: { type: 'string', description: 'Task message' },
        session_id: { type: 'string', description: 'Session ID for multi-turn tasks' },
      },
      required: ['agent_url', 'message'],
    },
  })
  async a2aSendTask(
    params: { agent_url: string; message: string; session_id?: string },
    context: Context,
  ): Promise<unknown> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.apiKey) headers['Authorization'] = `Bearer ${this.apiKey}`;
    const paymentToken =
      (context as any)?.payment?.token ??
      context?.get?.('payment_token') ??
      (context?.metadata?.paymentToken as string);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;

    const taskId = crypto.randomUUID();
    const body = {
      jsonrpc: '2.0',
      id: 1,
      method: 'tasks/send',
      params: {
        id: taskId,
        sessionId: params.session_id,
        message: {
          role: 'user',
          parts: [{ type: 'text', text: params.message }],
        },
      },
    };

    const res = await fetch(`${params.agent_url}/a2a`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!res.ok) throw new Error(`A2A send failed: ${res.status}`);
    return res.json();
  }

  @tool({
    name: 'a2a_get_task',
    description: 'Get the status and result of a task from another agent.',
    parameters: {
      type: 'object',
      properties: {
        agent_url: { type: 'string', description: 'Target agent URL' },
        task_id: { type: 'string', description: 'Task ID' },
      },
      required: ['agent_url', 'task_id'],
    },
  })
  async a2aGetTask(
    params: { agent_url: string; task_id: string },
    context: Context,
  ): Promise<unknown> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.apiKey) headers['Authorization'] = `Bearer ${this.apiKey}`;
    const paymentToken =
      (context as any)?.payment?.token ??
      context?.get?.('payment_token') ??
      (context?.metadata?.paymentToken as string);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;

    const body = {
      jsonrpc: '2.0',
      id: 1,
      method: 'tasks/get',
      params: { id: params.task_id },
    };

    const res = await fetch(`${params.agent_url}/a2a`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!res.ok) throw new Error(`A2A get failed: ${res.status}`);
    return res.json();
  }

  @tool({
    name: 'a2a_get_agent_card',
    description: 'Fetch the agent card (.well-known/agent.json) from another agent.',
    parameters: {
      type: 'object',
      properties: {
        agent_url: { type: 'string', description: 'Agent base URL' },
      },
      required: ['agent_url'],
    },
  })
  async a2aGetAgentCard(
    params: { agent_url: string },
    _context: Context,
  ): Promise<unknown> {
    const res = await fetch(`${params.agent_url}/.well-known/agent.json`, {
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!res.ok) throw new Error(`Agent card fetch failed: ${res.status}`);
    return res.json();
  }

  private readBody(req: unknown): Promise<string> {
    return new Promise((resolve, reject) => {
      let body = '';
      const r = req as { on: (event: string, cb: (chunk?: unknown) => void) => void };
      r.on('data', (chunk: unknown) => { body += String(chunk); });
      r.on('end', () => resolve(body));
      r.on('error', reject);
    });
  }

  override async cleanup(): Promise<void> {
    this.tasks.clear();
  }
}
