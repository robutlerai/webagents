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

import { Skill } from '../../../core/skill';
import { http } from '../../../core/decorators';
import type { Context, IAgent } from '../../../core/types';
import type { ClientEvent, ResponseDelta } from '../../../uamp/events';
import { generateEventId } from '../../../uamp/events';

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

export class A2ATransportSkill extends Skill {
  private agent: IAgent | null = null;
  private agentCard?: AgentCard;
  private tasks = new Map<string, A2ATask>();

  constructor(config: A2ATransportConfig = {}) {
    super({ ...config, name: config.name || 'a2a-transport' });
    this.agentCard = config.agentCard;
  }

  setAgent(agent: IAgent): void {
    this.agent = agent;

    if (!this.agentCard) {
      const caps = agent.getCapabilities();
      this.agentCard = {
        name: agent.name,
        description: agent.description ?? '',
        url: '',
        version: '1.0',
        capabilities: {
          streaming: caps.supports_streaming ?? true,
          pushNotifications: false,
          stateTransitionHistory: true,
        },
      };
    }
  }

  // ===========================================================================
  // Server-side HTTP endpoints (registered in httpRegistry via @http)
  // ===========================================================================

  @http({ path: '/.well-known/agent.json', method: 'GET' })
  async handleAgentCard(_request: Request, _context: Context): Promise<Response> {
    if (!this.agentCard) {
      return new Response(JSON.stringify({ error: 'No agent card configured' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response(JSON.stringify(this.agentCard), {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  @http({ path: '/a2a', method: 'POST' })
  async handleA2ARequest(request: Request, context: Context): Promise<Response> {
    try {
      const body = await request.json();
      const result = await this.handleJsonRpc(body, context);
      return new Response(JSON.stringify(result), {
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (err) {
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id: null,
        error: { code: -32700, message: `Parse error: ${(err as Error).message}` },
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  }

  private async handleJsonRpc(
    msg: { id: string | number; method: string; params?: Record<string, unknown> },
    context: Context,
  ): Promise<Record<string, unknown>> {
    switch (msg.method) {
      case 'tasks/send':
        return { jsonrpc: '2.0', id: msg.id, result: await this.handleTaskSend(msg.params ?? {}, context) };
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

  private async handleTaskSend(params: Record<string, unknown>, _context: Context): Promise<A2ATask> {
    const taskId = (params.id as string) ?? crypto.randomUUID();
    const now = new Date().toISOString();

    const task: A2ATask = {
      id: taskId,
      sessionId: params.sessionId as string | undefined,
      status: { state: 'submitted', timestamp: now },
      history: params.message ? [params.message as NonNullable<A2ATask['history']>[number]] : [],
    };
    this.tasks.set(taskId, task);

    task.status = { state: 'working', timestamp: new Date().toISOString() };

    if (!this.agent) {
      task.status = { state: 'failed', message: 'No agent attached', timestamp: new Date().toISOString() };
      return task;
    }

    try {
      const message = params.message as { role?: string; parts?: Array<{ type?: string; text?: string; data?: string; mimeType?: string; file?: { uri?: string; name?: string; mimeType?: string } }> } | undefined;

      const clientEvents: ClientEvent[] = [
        {
          type: 'session.create',
          event_id: generateEventId(),
          uamp_version: '1.0',
          session: { modalities: ['text'] },
        } as ClientEvent,
      ];

      for (const part of message?.parts ?? []) {
        if (part.type === 'text' && part.text) {
          clientEvents.push({ type: 'input.text', event_id: generateEventId(), text: part.text, role: 'user' } as ClientEvent);
        } else if (part.type === 'data' && part.data && part.mimeType?.startsWith('image/')) {
          clientEvents.push({ type: 'input.image', event_id: generateEventId(), image: `data:${part.mimeType};base64,${part.data}` } as ClientEvent);
        } else if (part.type === 'data' && part.data && part.mimeType?.startsWith('audio/')) {
          clientEvents.push({ type: 'input.audio', event_id: generateEventId(), audio: part.data, format: part.mimeType.split('/')[1] ?? 'webm' } as ClientEvent);
        } else if (part.type === 'file' && part.file?.uri) {
          clientEvents.push({ type: 'input.file', event_id: generateEventId(), file: { url: part.file.uri }, filename: part.file.name ?? 'document', mime_type: part.file.mimeType ?? 'application/octet-stream' } as ClientEvent);
        }
      }

      if (clientEvents.length === 1) {
        clientEvents.push({ type: 'input.text', event_id: generateEventId(), text: '', role: 'user' } as ClientEvent);
      }

      clientEvents.push({ type: 'response.create', event_id: generateEventId() } as ClientEvent);

      const resultParts: Array<{ type: string; text?: string; data?: string; mimeType?: string }> = [];
      for await (const serverEvent of this.agent.processUAMP(clientEvents)) {
        if (serverEvent.type === 'response.delta') {
          const delta = (serverEvent as unknown as { delta: ResponseDelta }).delta;
          if (delta.text) {
            resultParts.push({ type: 'text', text: delta.text });
          }
          if ((delta as unknown as Record<string, unknown>).image) {
            resultParts.push({ type: 'data', data: (delta as unknown as Record<string, unknown>).image as string, mimeType: 'image/png' });
          }
          if ((delta as unknown as Record<string, unknown>).video) {
            resultParts.push({ type: 'data', data: (delta as unknown as Record<string, unknown>).video as string, mimeType: 'video/mp4' });
          }
        }
      }

      task.artifacts = [{
        name: 'response',
        parts: resultParts.length > 0 ? resultParts : [{ type: 'text', text: '' }],
      }];
      task.status = { state: 'completed', timestamp: new Date().toISOString() };
    } catch (err) {
      task.status = {
        state: 'failed',
        message: (err as Error).message,
        timestamp: new Date().toISOString(),
      };
    }

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

  override async cleanup(): Promise<void> {
    this.tasks.clear();
  }
}
