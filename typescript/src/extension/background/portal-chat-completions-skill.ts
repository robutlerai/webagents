import { Skill } from '../../core/skill';
import { handoff } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';
import type { ClientEvent, ServerEvent } from '../../uamp/events';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createResponseErrorEvent,
  generateEventId,
} from '../../uamp/events';
import type { ContentItem, ToolDefinition } from '../../uamp/types';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

interface CompletionResponse {
  choices: Array<{
    message: ChatMessage;
    finish_reason: string;
  }>;
}

export interface PortalChatCompletionsSkillConfig extends SkillConfig {
  portalUrl: string;
  tokenProvider: () => string | null;
  modelProvider: () => string;
}

export class PortalChatCompletionsSkill extends Skill {
  private readonly portalUrl: string;
  private readonly tokenProvider: () => string | null;
  private readonly modelProvider: () => string;

  constructor(config: PortalChatCompletionsSkillConfig) {
    super({ ...config, name: config.name || 'portal-chat-completions' });
    this.portalUrl = config.portalUrl.replace(/\/+$/, '');
    this.tokenProvider = config.tokenProvider;
    this.modelProvider = config.modelProvider;
  }

  @handoff({ name: 'portal-chat-completions', priority: 10 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context,
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();
    const conversation = context.get<ChatMessage[]>('_agentic_messages') ?? this.messagesFromEvents(events);
    const tools = context.get<ToolDefinition[]>('_agentic_tools') ?? [];
    const token = this.tokenProvider();
    const paymentToken = context.payment.token;

    if (!token) {
      yield createResponseErrorEvent('unauthorized', 'Extension is not logged in.', responseId);
      return;
    }
    if (!paymentToken) {
      yield createResponseErrorEvent('payment_required', 'Missing payment token for browser-agent LLM call.', responseId);
      return;
    }

    try {
      const resp = await fetch(`${this.portalUrl}/api/llm/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
          'X-Payment-Token': paymentToken,
        },
        body: JSON.stringify({
          model: this.modelProvider(),
          messages: conversation,
          tools: tools.length > 0 ? tools : undefined,
          tool_choice: tools.length > 0 ? 'auto' : undefined,
        }),
      });

      if (!resp.ok) {
        yield createResponseErrorEvent('llm_error', `LLM call failed: ${resp.status} ${resp.statusText}`, responseId);
        return;
      }

      const data = await resp.json() as CompletionResponse;
      const choice = data.choices[0];
      if (!choice) {
        yield createResponseErrorEvent('llm_error', 'LLM returned no choices.', responseId);
        return;
      }

      const output: ContentItem[] = [];
      if (choice.message.content) {
        output.push({ type: 'text', text: choice.message.content });
        yield createResponseDeltaEvent(responseId, { type: 'text', text: choice.message.content });
      }

      for (const call of choice.message.tool_calls ?? []) {
        const toolCall = {
          id: call.id,
          name: call.function.name,
          arguments: call.function.arguments || '{}',
        };
        output.push({ type: 'tool_call', tool_call: toolCall });
        yield createResponseDeltaEvent(responseId, { type: 'tool_call', tool_call: toolCall });
      }

      yield createResponseDoneEvent(responseId, output);
    } catch (err) {
      yield createResponseErrorEvent('llm_error', (err as Error).message, responseId);
    }
  }

  private messagesFromEvents(events: ClientEvent[]): ChatMessage[] {
    const messages: ChatMessage[] = [];
    for (const event of events) {
      if (event.type === 'session.create') {
        const instructions = event.session?.instructions;
        if (instructions) messages.push({ role: 'system', content: instructions });
      }
      if (event.type === 'input.text') {
        messages.push({ role: event.role ?? 'user', content: event.text });
      }
    }
    return messages;
  }
}
