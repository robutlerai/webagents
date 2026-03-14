import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { Context, SkillConfig } from '../../../core/types.js';
import type { ClientEvent, ServerEvent, SessionCreateEvent, InputTextEvent } from '../../../uamp/events.js';
import {
  generateEventId,
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
} from '../../../uamp/events.js';
import { UAMPClient } from '../../../uamp/client.js';
import type { ContentItem, ToolDefinition, UsageStats } from '../../../uamp/types.js';

export interface LLMProxySkillConfig extends SkillConfig {
  /** URL of the UAMP LLM proxy (default: wss://robutler.ai/llm, override via ROBUTLER_LLM_PROXY_URL env) */
  proxyUrl?: string;
  /** Default model to use */
  model?: string;
  /** Temperature */
  temperature?: number;
  /** Max output tokens */
  max_tokens?: number;
}

export class LLMProxySkill extends Skill {
  private proxyUrl: string;
  private modelConfig: LLMProxySkillConfig;

  constructor(config: LLMProxySkillConfig = {}) {
    super({ ...config, name: config.name || 'llm-proxy' });
    this.proxyUrl = config.proxyUrl
      || (typeof process !== 'undefined' && process.env?.ROBUTLER_LLM_PROXY_URL)
      || 'wss://robutler.ai/llm';
    this.modelConfig = config;
  }

  @handoff({ name: 'llm-proxy', priority: 10 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();

    yield {
      type: 'response.created' as const,
      event_id: generateEventId(),
      response_id: responseId,
    };

    const conversation = context.get<Array<{
      role: string;
      content: string | null;
      tool_calls?: unknown[];
      tool_call_id?: string;
    }>>('_agentic_messages') || [];

    const tools: ToolDefinition[] = [...(context.get<ToolDefinition[]>('_agentic_tools') || [])];
    const paymentToken = context.payment.token;

    if (conversation.length === 0) {
      for (const event of events) {
        if (event.type === 'session.create') {
          const createEvent = event as SessionCreateEvent;
          if (createEvent.session.instructions) {
            conversation.push({ role: 'system', content: createEvent.session.instructions });
          }
          if (createEvent.session.tools) {
            tools.push(...createEvent.session.tools);
          }
        } else if (event.type === 'input.text') {
          const inputEvent = event as InputTextEvent;
          conversation.push({ role: inputEvent.role || 'user', content: inputEvent.text });
        }
      }
    }

    console.log(`[llm-proxy-skill] processUAMP: ${conversation.length} messages, ${tools.length} tools, paymentToken=${paymentToken ? 'yes' : 'no'}, url=${this.proxyUrl}`);

    const client = new UAMPClient({
      url: this.proxyUrl,
      paymentToken,
      signal: context.signal,
      extensions: {
        ...(context.metadata?.chatId ? { 'X-Chat-Id': context.metadata.chatId } : {}),
        ...(context.metadata?.agentId ? { 'X-Agent-Id': context.metadata.agentId } : {}),
      },
      session: {
        modalities: ['text'],
      },
    });

    const collectedOutput: ContentItem[] = [];
    let usage: UsageStats | undefined;
    let fullText = '';
    let error: Error | null = null;
    let done = false;

    const pendingEvents: ServerEvent[] = [];
    let notifyPending: (() => void) | null = null;

    client.on('delta', (text) => {
      fullText += text;
      pendingEvents.push(createResponseDeltaEvent(responseId, { type: 'text', text }));
      notifyPending?.();
    });

    client.on('toolCall', (tc: { id: string; name: string; arguments: string }) => {
      pendingEvents.push(createResponseDeltaEvent(responseId, {
        type: 'tool_call',
        tool_call: tc,
      }));
      notifyPending?.();
    });

    client.on('done', (response) => {
      collectedOutput.push(...response.output);
      usage = response.usage;
      done = true;
      notifyPending?.();
    });

    client.on('error', (err) => {
      error = err;
      done = true;
      notifyPending?.();
    });

    client.on('thinking', (data) => {
      pendingEvents.push({
        type: 'thinking' as const,
        event_id: generateEventId(),
        content: data.content,
        stage: data.stage,
        redacted: data.redacted,
        is_delta: data.is_delta,
      } as ServerEvent);
      notifyPending?.();
    });

    let wasCancelled = false;
    client.on('cancelled', () => {
      wasCancelled = true;
      done = true;
      notifyPending?.();
    });

    client.on('paymentRequired', (req) => {
      if (paymentToken) {
        client.sendPayment({ scheme: 'token', amount: req.amount, token: paymentToken });
      }
    });

    try {
      console.log(`[llm-proxy-skill] connecting to proxy…`);
      await client.connect();
      console.log(`[llm-proxy-skill] connected, sending response.create with ${conversation.length} messages`);

      await client.sendResponse({
        messages: conversation,
        model: this.modelConfig.model || 'auto/balanced',
        tools: tools.length > 0 ? tools : undefined,
        temperature: this.modelConfig.temperature ?? 0.7,
        max_tokens: this.modelConfig.max_tokens ?? 4096,
      });
      console.log(`[llm-proxy-skill] response.create sent, waiting for response…`);

      while (!done) {
        while (pendingEvents.length > 0) {
          yield pendingEvents.shift()!;
        }
        if (!done) {
          await new Promise<void>(resolve => { notifyPending = resolve; });
          notifyPending = null;
        }
      }
      while (pendingEvents.length > 0) {
        yield pendingEvents.shift()!;
      }
    } catch (err) {
      error = err instanceof Error ? err : new Error(String(err));
      console.error(`[llm-proxy-skill] error:`, error.message);
    } finally {
      client.close();
    }

    if (error) {
      console.error(`[llm-proxy-skill] yielding proxy_error: ${error.message}`);
      yield createResponseErrorEvent('proxy_error', error.message, responseId);
      return;
    }

    const output: ContentItem[] = collectedOutput.length > 0
      ? collectedOutput
      : fullText
        ? [{ type: 'text' as const, text: fullText }]
        : [];

    const status = wasCancelled ? 'cancelled' : 'completed';
    yield createResponseDoneEvent(responseId, output, status, usage);
  }
}
