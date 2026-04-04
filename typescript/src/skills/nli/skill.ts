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

import { Skill } from '../../core/skill';
import { tool, hook } from '../../core/decorators';
import type { ClientEvent, ServerEvent } from '../../uamp/events';
import type { Context, HookData, HookResult, Handoff as HandoffType, StructuredToolResult, AgenticMessage } from '../../core/types';
import { UAMPClient, type UAMPClientConfig } from '../../uamp/client';
import type { Message, ContentItem, ImageContent, VideoContent, AudioContent } from '../../uamp/types';
import { getContentItemUrl } from '../../uamp/content';

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
  /** Optional callback to sign /api/content/ URLs for external agent access. Returns a full signed URL. */
  signUrl?: (contentId: string) => Promise<string>;
  /** Optional callback to mint a payment token scoped to the target agent. Returns a JWT string. */
  createDelegateToken?: (targetAgent: string, callerUserId: string) => Promise<string | null>;
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
      signUrl: config.signUrl,
      createDelegateToken: config.createDelegateToken,
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
      'Send a message to another agent on the Robutler platform. ' +
      'Use attachments to forward media content.\n\n' +
      'Tool results that produce media include /api/content/ URLs. ' +
      'To forward media to another agent, pass the /api/content/UUID URL ' +
      'in the attachments array.\n\n' +
      'Media content is automatically displayed to the user. ' +
      'Do NOT include /api/content/ URLs or markdown media syntax ' +
      'in your text replies to the user. Just describe what happened.\n\n' +
      'If the agent requires payment, a payment token is automatically ' +
      'attached from your owner\'s balance.',
    parameters: {
      type: 'object',
      properties: {
        agent: {
          type: 'string',
          description: 'Agent username (e.g., "fundraiser") or full URL',
        },
        message: {
          type: 'string',
          description: 'The task or question for the agent',
        },
        attachments: {
          type: 'array',
          items: { type: 'string' },
          description: 'Media URLs (/api/content/UUID) from tool results to forward to the agent',
        },
      },
      required: ['agent', 'message'],
    },
  })
  async delegate(
    params: { agent: string; message: string; attachments?: string[] },
    context: Context,
  ): Promise<string | StructuredToolResult> {
    const agentRef = params.agent.startsWith('@') ? params.agent : params.agent.includes('/') ? params.agent : `@${params.agent}`;
    const fullUrl = this.normalizeUrl(agentRef);

    let message = params.message;

    // --- Resolve media attachments from conversation content_items ---
    const agenticMessages = context.get<AgenticMessage[]>('_agentic_messages') || [];
    const mediaItems: ContentItem[] = [];
    const attached = new Set<string>();

    // Build a lookup of all content_items in the conversation (all roles)
    const convContentMap = new Map<string, ContentItem>();
    for (const m of agenticMessages) {
      for (const ci of m.content_items ?? []) {
        const cid = (ci as { content_id?: string }).content_id;
        if (cid) convContentMap.set(cid, ci);
        // Also index by URL-extracted UUID for items without explicit content_id
        if (!cid) {
          const url = getContentItemUrl(ci);
          if (url) {
            const urlUuid = url.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i);
            if (urlUuid) convContentMap.set(urlUuid[1], ci);
          }
        }
      }
    }

    console.log(`[nli/delegate] convContentMap: ${convContentMap.size} entries, keys=[${[...convContentMap.keys()].join(', ')}], msgRoles=[${agenticMessages.map(m => m.role).join(',')}], msgContentItems=[${agenticMessages.map(m => (m.content_items?.length ?? 0)).join(',')}]`);

    // 1. Resolve explicit attachments (full /api/content/UUID URLs or bare UUIDs)
    let fromConv = 0;
    for (const ref of params.attachments ?? []) {
      const uuidMatch = ref.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i);
      const id = uuidMatch ? uuidMatch[1] : ref;
      const item = convContentMap.get(id);
      if (item) {
        mediaItems.push(item);
        attached.add(id);
        fromConv++;
      } else if (uuidMatch) {
        // Valid UUID but not in convContentMap — create content_item from URL
        console.log(`[nli/delegate] attachment ${id} not in convContentMap, creating from URL`);
        mediaItems.push({ type: 'image', image: { url: ref }, content_id: id } as any);
        attached.add(id);
        fromConv++;
      } else {
        console.log(`[nli/delegate] attachment "${ref}" is not a valid content reference, skipping`);
      }
    }

    // 2. Fallback: scan message text for /api/content/UUID not already attached
    let fromUrl = 0;
    for (const m of message.matchAll(/\/api\/content\/([0-9a-f-]{36})/gi)) {
      const uuid = m[1];
      if (!attached.has(uuid)) {
        const item = convContentMap.get(uuid);
        if (item) {
          mediaItems.push(item);
        } else {
          mediaItems.push({ type: 'image', image: { url: `/api/content/${uuid}` }, content_id: uuid } as ImageContent);
        }
        attached.add(uuid);
        fromUrl++;
      }
    }

    // 3. Also include user's original media if not already covered
    let fromUser = 0;
    const lastUserMsg = [...agenticMessages].reverse().find(m => m.role === 'user');
    for (const ci of lastUserMsg?.content_items ?? []) {
      if (ci.type !== 'text' && ci.type !== 'tool_call' && ci.type !== 'tool_result') {
        const cid = (ci as { content_id?: string }).content_id;
        if (cid && !attached.has(cid)) {
          mediaItems.push(ci);
          attached.add(cid);
          fromUser++;
        }
      }
    }

    console.log(`[nli/delegate] resolving attachments: requested=${params.attachments} fromConvItems=${fromConv} fromUrlScan=${fromUrl} fromUserMedia=${fromUser}`);

    // 4. Sign /api/content/ URLs in mediaItems for the delegate agent
    const signFn = this.nliConfig.signUrl;
    for (let i = 0; i < mediaItems.length; i++) {
      const url = getContentItemUrl(mediaItems[i]);
      if (url && signFn) {
        const idMatch = url.match(/\/api\/content\/([0-9a-f-]{36})/i);
        if (idMatch) {
          try {
            const signedUrl = await signFn(idMatch[1]);
            mediaItems[i] = { ...mediaItems[i] };
            const field = mediaItems[i].type as 'image' | 'video' | 'audio' | 'file';
            (mediaItems[i] as unknown as Record<string, unknown>)[field] = { url: signedUrl };
            console.log(`[nli/delegate] signed content URL for ${idMatch[1]}`);
          } catch (err) { console.log(`[nli/delegate] failed to sign content URL: ${err}`); }
        }
      }
    }

    // Normalize full URLs in message text to relative
    message = message.replace(/https?:\/\/[^\s)]+?(\/api\/content\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi, '$1');

    // Sign /api/content/ URLs in message text
    if (signFn) {
      const contentUrlPattern = /\/api\/content\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi;
      const contentMatches = [...message.matchAll(contentUrlPattern)];
      for (const match of contentMatches) {
        try {
          const signedUrl = await signFn(match[1]);
          message = message.replace(match[0], signedUrl);
        } catch (err) { console.log(`[nli/delegate] failed to sign text URL: ${err}`); }
      }
    }

    const emitProgress = context.get<(callId: string, text: string) => void>('_toolProgressFn');
    const toolCall = context.get<{ id?: string }>('tool_call');
    const callId = toolCall?.id;

    console.log(`[nli/delegate] resolved ${mediaItems.length} items: types=${mediaItems.map(i => i.type)}, ids=${mediaItems.map(i => (i as { content_id?: string }).content_id)}`);
    console.log(`[nli/delegate] → ${agentRef} message=${message.length} chars, attachments=${params.attachments?.length ?? 0}, mediaItems=${mediaItems.length}, first200=${message.slice(0, 200)}`);

    // Mint a payment token scoped to the target agent so audience claims are correct
    let delegatePaymentToken: string | undefined;
    const callerUserId = (context as any)?.auth?.user_id
      ?? context?.get?.('user_id') as string | undefined;
    if (this.nliConfig.createDelegateToken && callerUserId) {
      try {
        delegatePaymentToken = (await this.nliConfig.createDelegateToken(params.agent, callerUserId)) ?? undefined;
        if (delegatePaymentToken) {
          console.log(`[nli/delegate] minted delegate payment token for @${params.agent}`);
        }
      } catch (err) {
        console.warn(`[nli/delegate] failed to mint delegate token for @${params.agent}:`, err);
      }
    }

    let result = '';
    const delegateMsg: Message = {
      role: 'user',
      content: message,
      content_items: mediaItems.length > 0 ? mediaItems : undefined,
    };
    for await (const chunk of this.streamMessage(fullUrl, [delegateMsg], context, delegatePaymentToken)) {
      result += chunk;
      if (emitProgress && callId) emitProgress(callId, chunk);
    }

    // Collect output items from response.done (primary UAMP path)
    const outputItems = context.get<ContentItem[]>('_nli_output_items') ?? [];
    context.delete('_nli_output_items');

    // URL-scan fallback: find /api/content/UUID in response text not already in outputItems
    const outputIds = new Set(outputItems.map(ci => (ci as { content_id?: string }).content_id).filter(Boolean));
    let urlExtracted = 0;
    for (const m of (result || '').matchAll(/\/api\/content\/([0-9a-f-]{36})/gi)) {
      const uuid = m[1];
      if (!outputIds.has(uuid)) {
        outputItems.push({ type: 'image', image: { url: `/api/content/${uuid}` }, content_id: uuid } as ImageContent);
        outputIds.add(uuid);
        urlExtracted++;
      }
    }

    console.log(`[nli/delegate] response processing: outputItems=${outputItems.length - urlExtracted} fromDone, urlExtracted=${urlExtracted} fromText`);

    if (outputItems.length > 0) {
      const urls = outputItems
        .map(ci => (ci as { content_id?: string }).content_id)
        .filter(Boolean)
        .map(id => `/api/content/${id}`);
      const urlSuffix = urls.length > 0 ? '\n' + urls.join('\n') : '';
      const mediaDesc = outputItems.length > 0 && !result
        ? `[${outputItems.length} media item${outputItems.length > 1 ? 's' : ''} returned]`
        : '';
      const text = `${result || mediaDesc}${urlSuffix}`;
      console.log(`[nli/delegate] returning StructuredToolResult: items=${outputItems.length} urls=${urls.join(', ')}`);
      return { text, content_items: outputItems };
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
    messages: Message[],
    context?: Context,
    delegatePaymentToken?: string,
  ): AsyncGenerator<string, void, unknown> {
    const transport = this.getTransport();

    if (transport === 'uamp') {
      yield* this.streamMessageUAMP(agentUrl, messages, context, delegatePaymentToken);
      return;
    }

    if (transport === 'auto') {
      try {
        let gotData = false;
        for await (const chunk of this.streamMessageUAMP(agentUrl, messages, context, delegatePaymentToken)) {
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
    messages: Message[],
    context?: Context,
    delegatePaymentToken?: string,
  ): AsyncGenerator<string, void, unknown> {
    const wsUrl = this.getUAMPUrl(agentUrl);
    const paymentToken = delegatePaymentToken ??
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
    const outputItems: ContentItem[] = [];
    let resolveChunk: (() => void) | null = null;

    client.on('delta', (text) => {
      chunks.push(text);
      resolveChunk?.();
    });

    client.on('toolCall', (tc) => {
      const label = tc.name?.replace(/_/g, ' ') || 'tool';
      chunks.push(`[${label}] `);
      resolveChunk?.();
    });

    client.on('toolProgress', (progress) => {
      if (progress?.text) {
        chunks.push(progress.text);
        resolveChunk?.();
      }
    });

    client.on('done', (response) => {
      console.log(`[nli/stream] response.done: outputItemCount=${response?.output?.length ?? 0} types=${response?.output?.map((i: { type: string }) => i.type)}`);
      if (response?.output) {
        for (const item of response.output) {
          if (item.type === 'image') outputItems.push(item as ImageContent);
          else if (item.type === 'video') outputItems.push(item as VideoContent);
          else if (item.type === 'audio') outputItems.push(item as AudioContent);
        }
      }
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
      await client.sendInput(
        lastMessage?.content ?? '',
        lastMessage?.role === 'system' ? 'system' : 'user',
        lastMessage?.content_items,
      );

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

      if (outputItems.length > 0 && context) {
        context.set('_nli_output_items', outputItems);
      }
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
