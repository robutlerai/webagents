/**
 * NLI (Natural Language Interface) Skill
 *
 * Agent-to-agent communication via natural language. Provides a single
 * `delegate` tool that sends a message to another agent and returns
 * its response. Supports UAMP (WebSocket) and HTTP transports with
 * automatic fallback.
 *
 * Advanced capabilities (assertion minting, response signing, trust
 * verification, progress forwarding) are retained internally but
 * removed from the default tool surface to reduce LLM confusion.
 */

import { Skill } from '../../core/skill.js';
import { tool, hook } from '../../core/decorators.js';
import type { ClientEvent, ServerEvent } from '../../uamp/events.js';
import type { Context, HookData, HookResult, Handoff as HandoffType } from '../../core/types.js';
import { UAMPClient, type UAMPClientConfig } from '../../uamp/client.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NLIConfig {
  /** Portal base URL for agent discovery */
  baseUrl?: string;
  /** Specific agent URL to connect to */
  agentUrl?: string;
  /** Custom capability name (e.g., 'analyze_emotion') */
  capability?: string;
  /** Request timeout in ms */
  timeout?: number;
  /** Maximum retry attempts */
  maxRetries?: number;
  /** API key for authentication */
  apiKey?: string;
  /** Signing key (base64-encoded Ed25519 private key) for response signing */
  signingKey?: string;
  /** Agent ID for assertion minting */
  agentId?: string;
  /** Trusted agent IDs or public key fingerprints */
  trustedAgents?: string[];
  /** Trust level: 'strict' = verify signatures, 'permissive' = trust all */
  trustLevel?: 'strict' | 'permissive';
  /** Transport: 'uamp' for WebSocket, 'http' for HTTP, 'auto' tries UAMP then HTTP */
  transport?: 'uamp' | 'http' | 'auto';
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface NLIResponseChunk {
  choices?: Array<{
    delta?: { content?: string };
    finish_reason?: string;
  }>;
}

export interface DelegationAssertion {
  iss: string;
  sub: string;
  actions: string[];
  maxSpend?: string;
  currency?: string;
  exp: number;
  iat: number;
  jti: string;
  chain?: string[];
}

export interface SignedResponse {
  content: string;
  agentId: string;
  signature: string;
  alg: string;
  timestamp: number;
}

export interface ProgressUpdate {
  stage: string;
  message?: string;
  percent?: number;
  step?: number;
  totalSteps?: number;
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

export class NLISkill extends Skill {
  private nliConfig: NLIConfig;
  private nliHandoffs: HandoffType[] = [];
  private progressCallbacks = new Map<string, (update: ProgressUpdate) => void>();

  constructor(config: NLIConfig = {}) {
    super({ name: config.capability ? `nli-${config.capability}` : 'nli' });
    this.nliConfig = {
      baseUrl: config.baseUrl || 'https://portal.webagents.ai',
      agentUrl: config.agentUrl,
      capability: config.capability,
      timeout: config.timeout || 90000,
      maxRetries: config.maxRetries || 3,
      apiKey: config.apiKey,
      signingKey: config.signingKey,
      agentId: config.agentId,
      trustedAgents: config.trustedAgents ?? [],
      trustLevel: config.trustLevel ?? 'permissive',
      transport: config.transport,
    };

    if (this.nliConfig.capability && this.nliConfig.agentUrl) {
      this.nliHandoffs = [{
        name: `nli-${this.nliConfig.capability}`,
        subscribes: [this.nliConfig.capability],
        produces: ['response.delta'],
        priority: 50,
        enabled: true,
        handler: this.routeToAgent.bind(this),
      }];
    }
  }

  override get handoffs(): HandoffType[] {
    return this.nliHandoffs;
  }

  // ============================================================================
  // Consolidated delegate tool
  // ============================================================================

  @tool({
    name: 'delegate',
    description:
      'Send a message to another agent on the Robutler platform and receive ' +
      'its response. Use this to delegate tasks to specialized agents you ' +
      'found via the search tool.\n\n' +
      'The agent is identified by username (e.g., "fundraiser") or full URL. ' +
      'The message should clearly describe what you need the agent to do. ' +
      'Returns the agent\'s text response.\n\n' +
      'If the agent requires payment, a payment token is automatically ' +
      'attached from your owner\'s balance.',
    parameters: {
      type: 'object',
      properties: {
        agent: {
          type: 'string',
          description: 'Agent username or URL (e.g., "fundraiser" or "https://robutler.ai/agents/fundraiser")',
        },
        message: {
          type: 'string',
          description: 'The task or question for the agent',
        },
      },
      required: ['agent', 'message'],
    },
  })
  async delegate(
    params: { agent: string; message: string },
    context: Context,
  ): Promise<string> {
    const agentRef = params.agent.startsWith('@') ? params.agent : params.agent.includes('/') ? params.agent : `@${params.agent}`;
    const fullUrl = this.normalizeUrl(agentRef);

    const emitProgress = context.get<(callId: string, text: string) => void>('_toolProgressFn');
    const toolCall = context.get<{ id?: string }>('tool_call');
    const callId = toolCall?.id;

    let result = '';
    for await (const chunk of this.streamMessage(fullUrl, [{ role: 'user', content: params.message }], context)) {
      result += chunk;
      if (emitProgress && callId) emitProgress(callId, chunk);
    }
    return result || '(no response)';
  }

  // ============================================================================
  // Sleep tool
  // ============================================================================

  @tool({
    name: 'sleep',
    description:
      'Pause execution for a specified duration. Useful for waiting before ' +
      'retrying a failed delegation or letting an external process complete.',
    parameters: {
      type: 'object',
      properties: {
        seconds: {
          type: 'number',
          description: 'Duration to sleep in seconds (max 30)',
        },
        reason: {
          type: 'string',
          description: 'Why the agent is waiting (shown to user)',
        },
      },
      required: ['seconds'],
    },
  })
  async sleep(
    params: { seconds: number; reason?: string },
    _context: Context,
  ): Promise<string> {
    const duration = Math.min(Math.max(params.seconds, 0), 30);
    await new Promise((r) => setTimeout(r, duration * 1000));
    return `Waited ${duration} seconds${params.reason ? `: ${params.reason}` : ''}`;
  }

  // ============================================================================
  // Progress Management (internal)
  // ============================================================================

  onProgress(assertionId: string, callback: (update: ProgressUpdate) => void): void {
    this.progressCallbacks.set(assertionId, callback);
  }

  removeProgressCallback(assertionId: string): void {
    this.progressCallbacks.delete(assertionId);
  }

  // ============================================================================
  // Handoff Handler (internal)
  // ============================================================================

  private async *routeToAgent(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    for (const event of events) {
      const payload = event as unknown as { text?: string; content?: string };
      const message = payload.text || payload.content || '';

      if (message && this.nliConfig.agentUrl) {
        try {
          for await (const chunk of this.streamMessage(
            this.nliConfig.agentUrl,
            [{ role: 'user', content: message }],
            context,
          )) {
            yield {
              type: 'response.delta',
              event_id: `nli-${Date.now()}`,
              delta: { text: chunk },
            } as unknown as ServerEvent;
          }

          yield {
            type: 'response.done',
            event_id: `nli-done-${Date.now()}`,
            response: { output: [] },
          } as unknown as ServerEvent;
        } catch (error) {
          yield {
            type: 'response.error',
            event_id: `error-${Date.now()}`,
            error: { type: 'nli_error', message: (error as Error).message },
          } as unknown as ServerEvent;
        }
      }
    }
  }

  // ============================================================================
  // Internal: Assertion Minting
  // ============================================================================

  mintAssertion(
    target: string,
    actions: string[],
    maxSpend?: string,
    currency?: string,
    ttlSeconds = 300,
  ): DelegationAssertion {
    const now = Math.floor(Date.now() / 1000);
    return {
      iss: this.nliConfig.agentId ?? 'unknown',
      sub: target,
      actions,
      maxSpend,
      currency: currency ?? 'USD',
      exp: now + ttlSeconds,
      iat: now,
      jti: `da_${crypto.randomUUID()}`,
    };
  }

  encodeAssertion(assertion: DelegationAssertion): string {
    return btoa(JSON.stringify(assertion));
  }

  // ============================================================================
  // Hooks: Auto-sign responses
  // ============================================================================

  @hook({ lifecycle: 'after_run', priority: 90 })
  async autoSignResponse(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.nliConfig.signingKey || !this.nliConfig.agentId) return;
    const response = _data.response;
    if (!response || typeof response !== 'string') return;
    const signed = await this.signResponse(response);
    if (signed) {
      context.set('_nli_signature', signed);
    }
  }

  // ============================================================================
  // Internal: Signing
  // ============================================================================

  async signResponse(content: string): Promise<SignedResponse | null> {
    if (!this.nliConfig.signingKey || !this.nliConfig.agentId) return null;

    try {
      const timestamp = Date.now();
      const payload = new TextEncoder().encode(
        `${this.nliConfig.agentId}:${timestamp}:${content}`,
      );

      const keyBytes = Uint8Array.from(atob(this.nliConfig.signingKey), (c) => c.charCodeAt(0));
      const key = await crypto.subtle.importKey(
        'raw',
        keyBytes,
        { name: 'Ed25519' },
        false,
        ['sign'],
      );

      const sig = await crypto.subtle.sign('Ed25519', key, payload);
      const signature = btoa(String.fromCharCode(...new Uint8Array(sig)));

      return {
        content,
        agentId: this.nliConfig.agentId,
        signature,
        alg: 'Ed25519',
        timestamp,
      };
    } catch {
      return null;
    }
  }

  // ============================================================================
  // Internal: Communication
  // ============================================================================

  async *streamMessage(
    agentUrl: string,
    messages: ChatMessage[],
    context?: Context,
  ): AsyncGenerator<string, void, unknown> {
    const transport = this.getTransport();

    if (transport === 'uamp') {
      yield* this.streamMessageUAMP(agentUrl, messages, context);
      return;
    }

    if (transport === 'auto') {
      try {
        let gotData = false;
        for await (const chunk of this.streamMessageUAMP(agentUrl, messages, context)) {
          gotData = true;
          yield chunk;
        }
        if (gotData) return;
      } catch {
        // Fall through to HTTP
      }
    }

    const headers = this.buildHeaders(context);

    const response = await fetch(`${agentUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ messages, stream: true }),
      signal: AbortSignal.timeout(this.nliConfig.timeout!),
    });

    if (!response.ok) {
      throw new Error(`NLI request failed: ${response.status} ${response.statusText}`);
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed: NLIResponseChunk = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) yield content;
            } catch {
              // Skip malformed
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ============================================================================
  // Internal: UAMP Transport
  // ============================================================================

  private getTransport(): 'uamp' | 'http' | 'auto' {
    return this.nliConfig.transport ?? 'auto';
  }

  private getUAMPUrl(agentUrl: string): string {
    const normalized = this.normalizeUrl(agentUrl);
    const u = new URL(normalized);
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    const basePath = u.pathname.replace(/\/+$/, '');
    u.pathname = `${basePath}/uamp`;
    u.search = '';
    return u.toString();
  }

  async *streamMessageUAMP(
    agentUrl: string,
    messages: ChatMessage[],
    context?: Context,
  ): AsyncGenerator<string, void, unknown> {
    const wsUrl = this.getUAMPUrl(agentUrl);
    const paymentToken =
      (context as any)?.payment?.token ??
      context?.get?.<string>('payment_token') ??
      (context?.metadata?.paymentToken as string) ??
      undefined;
    const apiKey = this.nliConfig.apiKey || (context?.metadata?.apiKey as string);

    const headers: Record<string, string> = {};
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const config: UAMPClientConfig = {
      url: wsUrl,
      paymentToken,
      connectTimeout: this.nliConfig.timeout,
      ...(Object.keys(headers).length > 0 ? { headers } : {}),
    };

    const client = new UAMPClient(config);

    let done = false;
    let error: Error | null = null;
    const chunks: string[] = [];
    let resolveChunk: (() => void) | null = null;

    client.on('delta', (text) => {
      chunks.push(text);
      resolveChunk?.();
    });

    client.on('done', () => {
      done = true;
      resolveChunk?.();
    });

    client.on('error', (err) => {
      error = err;
      done = true;
      resolveChunk?.();
    });

    client.on('paymentRequired', (req) => {
      if (paymentToken) {
        client.sendPayment({
          scheme: 'token',
          amount: req.amount,
          token: paymentToken,
        }).catch(() => {});
      } else {
        error = new Error(`Payment required: ${req.amount} ${req.currency}`);
        done = true;
        resolveChunk?.();
      }
    });

    try {
      await client.connect();
      const lastMessage = messages[messages.length - 1];
      await client.sendInput(lastMessage?.content ?? '', lastMessage?.role === 'system' ? 'system' : 'user');

      while (!done) {
        if (chunks.length > 0) {
          yield chunks.shift()!;
          continue;
        }
        await new Promise<void>((r) => { resolveChunk = r; });
        resolveChunk = null;
      }

      while (chunks.length > 0) {
        yield chunks.shift()!;
      }

      if (error) throw error;
    } finally {
      client.close();
    }
  }

  normalizeUrl(agentUrl: string): string {
    if (agentUrl.startsWith('@')) {
      return `${this.nliConfig.baseUrl}/agents/${agentUrl.slice(1)}`;
    }
    if (!agentUrl.startsWith('http://') && !agentUrl.startsWith('https://')) {
      return `https://${agentUrl}`;
    }
    return agentUrl;
  }

  private buildHeaders(context?: Context): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const apiKey = this.nliConfig.apiKey || (context?.metadata?.apiKey as string);
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
    if (context?.auth?.authenticated) {
      const token = context.metadata?.authToken as string;
      if (token) headers['X-Forwarded-Auth'] = token;
    }
    const paymentToken =
      (context as any)?.payment?.token ??
      context?.get?.('payment_token') ??
      (context?.metadata?.paymentToken as string);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;
    return headers;
  }

  override async cleanup(): Promise<void> {
    this.progressCallbacks.clear();
  }
}
