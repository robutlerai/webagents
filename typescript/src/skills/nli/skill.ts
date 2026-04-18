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
import type { Message, ContentItem } from '../../uamp/types';
import { isMediaContent } from '../../uamp/content';

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
  /**
   * Optional callback to resolve or create a delegate sub-chat for cross-agent file isolation.
   * The portal-side implementation handles agent username → userId resolution internally.
   */
  resolveDelegateSubChat?: (params: {
    explicitChatId?: string;
    parentChatId?: string;
    callerUserId: string;
    delegatedAgentRef: string;
  }) => Promise<{ chatId: string; created: boolean; type: string } | null>;
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
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a short, single-line label for a sub-agent's tool call so the
 * parent UI can show "→ text_editor(create /unicorn.html)" instead of
 * an opaque tool name (or, worse, the full args blob — which for
 * text_editor.create includes the entire file body).
 */
function formatSubagentToolLabel(name: string, command?: string, path?: string): string {
  const parts: string[] = [];
  if (command) parts.push(command);
  if (path) parts.push(path);
  return parts.length > 0 ? `${name}(${parts.join(' ')})` : name;
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
      resolveDelegateSubChat: config.resolveDelegateSubChat,
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
      'Hand off a task to another agent on the Robutler platform. ' +
      'SYNCHRONOUS — this call blocks until the delegate finishes and returns the full result. ' +
      'Do NOT say "it should be ready shortly" or "I\'ve sent it off" — you already have the result; present it directly.\n\n' +
      'Forward media by listing content_ids in `attachments` (the IDs come from prior tool results or user-uploaded items). ' +
      'When the delegated agent needs to read or edit an existing file or folder, also include its `content_id` in the message — delegates address content outside their own chat by content_id, not by path.\n\n' +
      'Result handling:\n' +
      '- The result may include content_items in addition to text. Call `present(content_id)` for each new id you want the user to see (ids appear in the result as `Media content_ids: <id>` or `content_id=<uuid>`); content not passed through `present` is not rendered.\n' +
      '- Strip raw internals from your reply (UUIDs, JSON, timing/duration metadata, billing, "Media content_ids: …" suffixes) — describe the result in your own words.\n' +
      '- Never include raw content URLs or markdown image/link syntax in your text reply.\n\n' +
      'Payment is handled automatically from your owner\'s balance when required.',
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
          description: 'Content IDs from tool results to forward to the agent',
        },
        chat_id: {
          type: 'string',
          description:
            'Optional UUID. Identifies a conversation thread with the delegated agent. ' +
            'If omitted, a default sub-chat is reused or created for this (caller, target-agent) pair. ' +
            'If supplied and the chat already exists, the call resumes that thread (you must already be a participant). ' +
            'If supplied and the chat does not exist, a new thread is created with that exact id.',
        },
      },
      required: ['agent', 'message'],
    },
  })
  async delegate(
    params: { agent: string; message: string; attachments?: string[]; chat_id?: string },
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
      }
    }

    console.log(`[nli/delegate] convContentMap: ${convContentMap.size} entries, keys=[${[...convContentMap.keys()].join(', ')}], msgRoles=[${agenticMessages.map(m => m.role).join(',')}], msgContentItems=[${agenticMessages.map(m => (m.content_items?.length ?? 0)).join(',')}]`);

    // 1. Resolve explicit attachments by content_id.
    // Order of resolution: (a) in-conversation content_items map, then
    // (b) DB fallback via the `_resolveContentById` hook (wired by
    // lib/agents/runtime.ts). The DB fallback is essential when an attached
    // content_id was created by a sub-agent in a sibling/sub-chat that the
    // current agent has access to via chat-tree ancestry but whose ContentItem
    // never travelled through this agent's own conversation messages.
    // Items returned by _resolveContentById are already fully signed; skip
    // re-signing for those.
    const resolveById = (context as any)?.get?.('_resolveContentById') as
      | ((id: string, callerUserId?: string) => Promise<ContentItem | null>)
      | undefined;
    const callerUserIdEarly = (context as any)?.auth?.user_id
      ?? (context as any)?.get?.('user_id') as string | undefined;
    const preSignedIds = new Set<string>();
    const unresolved: string[] = [];
    let fromConv = 0;
    let fromDb = 0;
    for (const ref of params.attachments ?? []) {
      const fromMap = convContentMap.get(ref);
      if (fromMap) {
        mediaItems.push(fromMap);
        attached.add(ref);
        fromConv++;
        continue;
      }
      if (resolveById) {
        try {
          const resolved = await resolveById(ref, callerUserIdEarly);
          if (resolved) {
            mediaItems.push(resolved);
            attached.add(ref);
            preSignedIds.add(ref);
            fromDb++;
            continue;
          }
        } catch (err) {
          console.warn(`[nli/delegate] _resolveContentById threw for id=${ref}: ${(err as Error).message}`);
        }
      }
      console.log(`[nli/delegate] attachment "${ref}" not found by content_id (convMap miss + DB miss/denied)`);
      unresolved.push(ref);
    }

    // Hard-error if any explicitly requested attachment could not be
    // resolved. Silently dropping them previously caused the LLM to think
    // the recipient agent had received the file when it had not.
    if (unresolved.length > 0) {
      return `Error: ${unresolved.length} attachment id(s) could not be resolved: ${unresolved.join(', ')}. The content may not exist or you may not have access to it. Verify the content_id and that the file was created in this conversation tree.`;
    }

    // 2. Also include user's original media if not already covered
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

    console.log(`[nli/delegate] resolving attachments: requested=${params.attachments} fromConvItems=${fromConv} fromDbItems=${fromDb} fromUserMedia=${fromUser}`);

    // 3. Sign content URLs by content_id (skip items already signed via DB fallback).
    const signFn = this.nliConfig.signUrl;
    for (let i = 0; i < mediaItems.length; i++) {
      const cid = (mediaItems[i] as { content_id?: string }).content_id;
      if (cid && signFn && !preSignedIds.has(cid)) {
        try {
          const signedUrl = await signFn(cid);
          mediaItems[i] = { ...mediaItems[i] };
          const field = mediaItems[i].type as 'image' | 'video' | 'audio' | 'file';
          (mediaItems[i] as unknown as Record<string, unknown>)[field] = { url: signedUrl };
          console.log(`[nli/delegate] signed content URL for ${cid}`);
        } catch (err) { console.log(`[nli/delegate] failed to sign content URL: ${err}`); }
      }
    }

    const emitProgress = context.get<(callId: string, text: string, opts?: { replace?: boolean; media_type?: string; status?: string; progress_percent?: number; estimated_duration_ms?: number }) => void>('_toolProgressFn');
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
        } else {
          // Resolver explicitly returned null: agent does not exist in the
          // registry. Short-circuit with a clear error instead of attempting
          // to connect to a non-existent endpoint (which silently 200s with
          // an empty body via the HTTP fallback and causes the caller to
          // retry indefinitely).
          console.warn(`[nli/delegate] aborting: target agent @${params.agent} not found in registry`);
          return `Error: agent @${params.agent} does not exist. Use the search tool to discover valid agent names before delegating.`;
        }
      } catch (err) {
        console.warn(`[nli/delegate] failed to mint delegate token for @${params.agent}:`, err);
      }
    }

    // --- Resolve delegate sub-chat for file isolation ---
    let delegateChatId: string | undefined;
    if (this.nliConfig.resolveDelegateSubChat && callerUserId) {
      const parentChatId = context?.get?.('chat_id') as string | undefined
        ?? (context?.metadata?.chatId as string | undefined);
      try {
        const subChat = await this.nliConfig.resolveDelegateSubChat({
          explicitChatId: params.chat_id,
          parentChatId,
          callerUserId,
          delegatedAgentRef: params.agent,
        });
        if (subChat) {
          delegateChatId = subChat.chatId;
          console.log(`[nli/delegate] resolved sub-chat: id=${subChat.chatId} type=${subChat.type} created=${subChat.created}`);
        }
      } catch (err) {
        console.warn(`[nli/delegate] failed to resolve sub-chat:`, err);
        if ((err as Error).message?.includes('not a participant')) {
          return `Error: ${(err as Error).message}`;
        }
      }
    }

    // Link every resolved attachment into the sub-chat so its participants
    // (caller + delegated agent) get an explicit user-scoped content_link.
    // Without this, the delegated agent reads via canAccessViaChatTree
    // (ancestor) but findByName/resolvePath inside the sub-chat won't see
    // the file by basename — the sub-agent then races text_editor create on
    // the same name and either collides or silently creates a divergent new
    // content row (the "Created badge after edit" bug).
    if (delegateChatId && callerUserId && mediaItems.length > 0) {
      const linkFn = context?.get?.('_linkContentToSubChat') as
        | ((ids: string[], chatId: string, userId: string) => Promise<void>)
        | undefined;
      if (linkFn) {
        const attachIds = mediaItems
          .map((m) => (m as { content_id?: string }).content_id)
          .filter((cid): cid is string => typeof cid === 'string' && cid.length > 0);
        if (attachIds.length > 0) {
          try {
            await linkFn(attachIds, delegateChatId, callerUserId);
            console.log(`[nli/delegate] linked ${attachIds.length} attachment(s) to sub-chat ${delegateChatId}`);
          } catch (err) {
            console.warn(`[nli/delegate] _linkContentToSubChat failed: ${(err as Error).message}`);
          }
        }
      }
    }

    // --- Loop guard: short-circuit if the same delegate just returned ---
    // --- two consecutive `(no response)` markers (plan §3.2). -----------
    // The `_countConsecutiveDelegateEmpties` hook is wired by the portal
    // runtime (`lib/agents/runtime.ts`) and counts assistant rows tagged
    // `metadata.kind='delegate_empty_result'` at the tail of the supplied
    // sub-chat. Sub-chats are pair-scoped (`resolveDelegateSubChat` creates
    // one row per (caller, callee)), so the count is implicitly per-callee.
    // Skipped when there's no sub-chat (manual chat_id may also work but
    // we conservatively gate on the resolved sub-chat path).
    if (delegateChatId) {
      const countEmpties = context?.get?.('_countConsecutiveDelegateEmpties') as
        | ((subChatId: string) => Promise<number>)
        | undefined;
      if (typeof countEmpties === 'function') {
        try {
          const consecutive = await countEmpties(delegateChatId);
          if (consecutive >= 2) {
            console.warn(
              `[nli/delegate] loop guard tripped: ${consecutive} consecutive empty results from ${params.agent} in sub-chat ${delegateChatId}; short-circuiting`,
            );
            return {
              text:
                `Error: agent ${agentRef} returned empty responses ${consecutive} times in a row. ` +
                `Stop retrying; try a different agent or report to the user.`,
              data: { subChatId: delegateChatId, loopGuard: true, consecutiveEmpties: consecutive },
            };
          }
        } catch (err) {
          console.warn(`[nli/delegate] _countConsecutiveDelegateEmpties failed (non-fatal): ${(err as Error).message}`);
        }
      }
    }

    let result = '';
    const delegateMsg: Message = {
      role: 'user',
      content: message,
      content_items: mediaItems.length > 0 ? mediaItems : undefined,
    };
    for await (const chunk of this.streamMessage(fullUrl, [delegateMsg], context, delegatePaymentToken, delegateChatId)) {
      result += chunk;
      if (emitProgress && callId) emitProgress(callId, chunk);
    }

    // Collect output items from response.done (primary UAMP path)
    const outputItems = context.get<ContentItem[]>('_nli_output_items') ?? [];
    context.delete('_nli_output_items');

    const textOnly = context.get<string>('_nli_text_only') ?? result;
    context.delete('_nli_text_only');

    console.log(`[nli/delegate] response processing: outputItems=${outputItems.length} fromDone`);

    // Always attach `subChatId` (when resolved) to the structured tool
    // result so the parent's <DelegateSubChatPreview /> renderer can find
    // the right sub-chat to subscribe to (plan §4 step 1). The data field
    // is forwarded by `webagents/typescript/src/core/agent.ts` into the
    // `response.delta { tool_result }` envelope and persisted by the
    // portal-side persister into the parent's `tool_result` row metadata.
    const dataMeta = delegateChatId ? { subChatId: delegateChatId } : undefined;

    if (outputItems.length > 0) {
      const cleanText = textOnly || '';
      const ids = outputItems
        .map(ci => (ci as { content_id?: string }).content_id)
        .filter(Boolean);
      const mediaDesc = !cleanText
        ? `[${outputItems.length} media item${outputItems.length > 1 ? 's' : ''} returned. Use present(content_id) to display each one.]`
        : '';
      const text = cleanText || mediaDesc;
      const idSuffix = ids.length > 0 ? `\nMedia content_ids: ${ids.join(', ')}` : '';
      console.log(`[nli/delegate] returning StructuredToolResult: items=${outputItems.length} ids=${ids.join(', ')} subChatId=${delegateChatId ?? '<none>'}`);
      return {
        text: text + idSuffix,
        content_items: outputItems,
        ...(dataMeta ? { data: dataMeta } : {}),
      };
    }

    // Non-UAMP child hint: suggest save_content for external URLs without structured content
    let returnText = textOnly || '(no response)';
    if (result && result.includes('https://')) {
      returnText += '\nExternal media URLs detected. Use save_content to persist them -- saves to user library, enables tool processing, prevents URL expiration.';
    }
    // When a sub-chat exists, return a StructuredToolResult so the renderer
    // can find `subChatId`; otherwise keep the legacy plain-string return.
    return dataMeta
      ? { text: returnText, data: dataMeta }
      : returnText;
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
    chatId?: string,
  ): AsyncGenerator<string, void, unknown> {
    const transport = this.getTransport();

    if (transport === 'uamp') {
      yield* this.streamMessageUAMP(agentUrl, messages, context, delegatePaymentToken, chatId);
      return;
    }

    if (transport === 'auto') {
      try {
        let gotData = false;
        for await (const chunk of this.streamMessageUAMP(agentUrl, messages, context, delegatePaymentToken, chatId)) {
          gotData = true;
          yield chunk;
        }
        if (gotData) return;
      } catch {
        // Fall through to HTTP
      }
    }

    const headers = this.buildHeaders(context, delegatePaymentToken, chatId);

    const httpSignal = context?.signal
      ? AbortSignal.any([context.signal, AbortSignal.timeout(this.nliConfig.timeout!)])
      : AbortSignal.timeout(this.nliConfig.timeout!);

    const response = await fetch(`${agentUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ messages, stream: true }),
      signal: httpSignal,
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
    chatId?: string,
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
    if (paymentToken) {
      headers['X-Payment-Token'] = paymentToken;
    }
    if (chatId) {
      headers['X-Chat-Id'] = chatId;
    }

    const RESPONSE_TIMEOUT = 360_000;
    const combinedSignal = context?.signal
      ? AbortSignal.any([context.signal, AbortSignal.timeout(RESPONSE_TIMEOUT)])
      : AbortSignal.timeout(RESPONSE_TIMEOUT);

    const extensions: Record<string, unknown> = {};
    if (chatId) {
      extensions['X-Chat-Id'] = chatId;
    }

    const config: UAMPClientConfig = {
      url: wsUrl,
      paymentToken,
      signal: combinedSignal,
      connectTimeout: this.nliConfig.timeout,
      responseTimeout: RESPONSE_TIMEOUT,
      ...(Object.keys(headers).length > 0 ? { headers } : {}),
      ...(Object.keys(extensions).length > 0 ? { extensions } : {}),
      // Note: we deliberately do NOT set `supports_rich_display` for
      // delegated sub-chats. `present` / `read_content` are now always
      // registered on the child agent (they're universal content tools),
      // but leaving rich-display unset preserves the **non-browser
      // safety net**: any `collectedContentItems` the child accumulates
      // auto-promote into `response.done.output` even if the sub-agent
      // forgets to call `present`. Browser sessions opt in to selective
      // output by setting `supports_rich_display: true` themselves.
    };

    const client = new UAMPClient(config);

    type ProgressOpts = { replace?: boolean; media_type?: string; status?: string; progress_percent?: number; estimated_duration_ms?: number };
    const parentProgressFn = context?.get?.<(callId: string, text: string, opts?: ProgressOpts) => void>('_toolProgressFn');
    const parentCallId = context?.get?.<{ id?: string }>('tool_call')?.id;

    let done = false;
    let error: Error | null = null;
    const chunks: string[] = [];
    const outputItems: ContentItem[] = [];
    let resolveChunk: (() => void) | null = null;

    const textOnlyChunks: string[] = [];
    const toolResultItems: ContentItem[] = [];

    client.on('delta', (text) => {
      chunks.push(text);
      textOnlyChunks.push(text);
      resolveChunk?.();
    });

    client.on('toolProgress', (progress) => {
      if (!progress?.text) return;
      if (progress.replace && parentProgressFn && parentCallId) {
        const { call_id: _cid, text: _t, ...opts } = progress;
        parentProgressFn(parentCallId, progress.text, opts);
      } else {
        chunks.push(progress.text);
        resolveChunk?.();
      }
    });

    client.on('toolResult', (tr) => {
      if (tr?.content_items && Array.isArray(tr.content_items)) {
        for (const item of tr.content_items) {
          if (isMediaContent(item as ContentItem)) {
            toolResultItems.push(item as ContentItem);
          }
        }
      }
      // Surface a short "✓ done" line so the parent UI shows that the
      // sub-agent finished a step — useful when the next inner call
      // doesn't fire for a while.
      if (parentProgressFn && parentCallId && tr?.tool) {
        const label = formatSubagentToolLabel(tr.tool, tr.command, tr.path);
        parentProgressFn(parentCallId, `✓ ${label}`, { replace: true, kind: 'subagent_tool' } as ProgressOpts & { kind: string });
      }
    });

    // Forward inner sub-agent tool_call events to the parent's progress
    // channel so the user sees what the delegate is currently doing
    // (e.g. "text_editor create /unicorn.html") instead of staring at a
    // silent spinner for tens of seconds while a long platform tool runs.
    // We route through parentProgressFn (NOT chunks) to keep the LLM-
    // visible text result of the delegate clean.
    client.on('toolCall', (tc) => {
      if (!parentProgressFn || !parentCallId) return;
      let command: string | undefined;
      let path: string | undefined;
      try {
        const args = typeof tc.arguments === 'string' && tc.arguments
          ? JSON.parse(tc.arguments)
          : (tc.arguments as unknown);
        if (args && typeof args === 'object') {
          const a = args as Record<string, unknown>;
          if (typeof a.command === 'string') command = a.command;
          if (typeof a.path === 'string') path = a.path;
        }
      } catch {
        // Args may stream incrementally and not yet parse — fall back to
        // tool name only.
      }
      const label = formatSubagentToolLabel(tc.name, command, path);
      parentProgressFn(parentCallId, `→ ${label}`, { replace: true, kind: 'subagent_tool' } as ProgressOpts & { kind: string });
      if (process.env.LOG_LOOP_DEBUG === '1') {
        const argsLen = typeof tc.arguments === 'string' ? tc.arguments.length : 0;
        console.log(`[loop-debug] nli forwarded sub-toolCall name=${tc.name} args.len=${argsLen} parentCallId=${parentCallId}`);
      }
    });

    // Sub-agents emit file/image content created via platform tools
    // (text_editor create, etc.) as streaming `file` deltas — these are
    // NOT guaranteed to appear in the agent's final `response.done.output`
    // because the underlying agent runtime resets per-iteration state
    // and only the last iteration's done output is forwarded. Capture
    // them here so the parent learns the new content_ids and does not
    // hallucinate a missing file when the sub-agent references it by name.
    client.on('file', (delta) => {
      const ci = { ...(delta as object) } as ContentItem;
      const cid = (ci as { content_id?: string }).content_id;
      if (!cid || !isMediaContent(ci)) return;
      const seen = outputItems.some((o) => (o as { content_id?: string }).content_id === cid)
        || toolResultItems.some((o) => (o as { content_id?: string }).content_id === cid);
      if (!seen) {
        toolResultItems.push(ci);
      }
    });

    if (process.env.DELEGATE_FORWARD_THINKING !== '0') {
      client.on('thinking', (thinking: { content?: string; is_delta?: boolean }) => {
        if (thinking?.content) {
          if (parentProgressFn && parentCallId) {
            parentProgressFn(parentCallId, thinking.content, { kind: 'thinking' } as any);
          } else {
            chunks.push(thinking.content);
            resolveChunk?.();
          }
        }
      });
    }

    client.on('done', (response) => {
      console.log(`[nli/stream] response.done: outputItemCount=${response?.output?.length ?? 0} types=${response?.output?.map((i: { type: string }) => i.type)}`);
      if (response?.output) {
        for (const item of response.output) {
          if (isMediaContent(item as ContentItem)) {
            const forwarded = { ...item };
            delete (forwarded as Record<string, unknown>).display_hint;
            outputItems.push(forwarded as ContentItem);
          }
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

    // TODO: Wire refreshToken to get a fresh token on payment exhaustion instead of re-sending the same one
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
        if (context?.signal?.aborted) {
          done = true;
          break;
        }
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

      // Merge content items from `response.done.output` with items captured
      // from streaming `file`/tool_result deltas, deduping by content_id so
      // platform-tool outputs (text_editor create, etc.) propagate even when
      // the sub-agent's final iteration didn't re-include them.
      const finalItems: ContentItem[] = [];
      const seenIds = new Set<string>();
      const pushUnique = (item: ContentItem) => {
        const cid = (item as { content_id?: string }).content_id;
        if (cid) {
          if (seenIds.has(cid)) return;
          seenIds.add(cid);
        }
        finalItems.push(item);
      };
      for (const item of outputItems) pushUnique(item);
      for (const item of toolResultItems) pushUnique(item);
      if (finalItems.length > 0 && context) {
        context.set('_nli_output_items', finalItems);
      }
      if (context && textOnlyChunks.length > 0) {
        context.set('_nli_text_only', textOnlyChunks.join(''));
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

  private buildHeaders(context?: Context, overridePaymentToken?: string, chatId?: string): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const apiKey = this.nliConfig.apiKey || (context?.metadata?.apiKey as string);
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
    if (context?.auth?.authenticated) {
      const token = context.metadata?.authToken as string;
      if (token) headers['X-Forwarded-Auth'] = token;
    }
    const paymentToken = overridePaymentToken ??
      (context as any)?.payment?.token ??
      context?.get?.('payment_token') ??
      (context?.metadata?.paymentToken as string);
    if (paymentToken) headers['X-Payment-Token'] = paymentToken;
    if (chatId) headers['X-Chat-Id'] = chatId;
    return headers;
  }

  override async cleanup(): Promise<void> {
    this.progressCallbacks.clear();
  }
}
