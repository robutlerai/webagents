/**
 * Completions Transport Skill
 * 
 * OpenAI-compatible Chat Completions API transport.
 * Converts OpenAI format to UAMP and back.
 */

import { Skill } from '../../../core/skill.js';
import { http } from '../../../core/decorators.js';
import type { SkillConfig, Context, IAgent } from '../../../core/types.js';
import type { ClientEvent, ServerEvent, ResponseDelta } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';
import { PaymentRequiredError } from '../../payments/x402.js';

/**
 * OpenAI Chat Completion Request
 */
interface ChatCompletionRequest {
  model: string;
  messages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | null;
    name?: string;
    tool_calls?: Array<{
      id: string;
      type: 'function';
      function: { name: string; arguments: string };
    }>;
    tool_call_id?: string;
  }>;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  tools?: Array<{
    type: 'function';
    function: { name: string; description?: string; parameters?: unknown };
  }>;
  stream_options?: { include_usage?: boolean };
}

/**
 * OpenAI Chat Completion Response
 */
interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Completions transport skill configuration
 */
export interface CompletionsTransportConfig extends SkillConfig {
  /** Base path for endpoints (default: '/v1') */
  basePath?: string;
}

/**
 * Completions Transport Skill
 * 
 * Exposes OpenAI-compatible Chat Completions API.
 */
export class CompletionsTransportSkill extends Skill {
  private agent: IAgent | null = null;
  
  constructor(config: CompletionsTransportConfig = {}) {
    super({ ...config, name: config.name || 'completions' });
  }
  
  /**
   * Set the agent to delegate to
   */
  setAgent(agent: IAgent): void {
    this.agent = agent;
  }
  
  /**
   * Convert OpenAI request to UAMP events
   */
  toUAMP(request: ChatCompletionRequest, paymentToken?: string): ClientEvent[] {
    const events: ClientEvent[] = [];
    
    const extensions: Record<string, unknown> = {
      openai: {
        model: request.model,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
      },
    };
    if (paymentToken) {
      extensions['X-Payment-Token'] = paymentToken;
    }

    // Create session with tools if provided
    events.push({
      type: 'session.create',
      event_id: generateEventId(),
      uamp_version: '1.0',
      session: {
        modalities: ['text'],
        tools: request.tools?.map(t => ({
          type: 'function' as const,
          function: {
            name: t.function.name,
            description: t.function.description,
            parameters: t.function.parameters as import('../../../uamp/types.js').JSONSchema,
          },
        })),
        extensions,
      },
    });
    
    // Convert messages to input events
    for (const msg of request.messages) {
      if (msg.role === 'system') {
        events.push({
          type: 'input.text',
          event_id: generateEventId(),
          text: msg.content || '',
          role: 'system',
        });
      } else if (msg.role === 'user') {
        events.push({
          type: 'input.text',
          event_id: generateEventId(),
          text: msg.content || '',
          role: 'user',
        });
      }
      // Note: assistant and tool messages would be part of conversation history
      // In a full implementation, we'd track conversation state
    }
    
    // Request response
    events.push({
      type: 'response.create',
      event_id: generateEventId(),
    });
    
    return events;
  }
  
  /**
   * Convert UAMP events to OpenAI response
   */
  fromUAMP(events: ServerEvent[], model: string): ChatCompletionResponse {
    let content = '';
    let usage: ChatCompletionResponse['usage'];
    const toolCalls: Array<{
      id: string;
      type: 'function';
      function: { name: string; arguments: string };
    }> = [];
    
    for (const event of events) {
      if (event.type === 'response.delta') {
        const delta = (event as { delta: ResponseDelta }).delta;
        if (delta.text) {
          content += delta.text;
        }
        if (delta.tool_call) {
          const existing = toolCalls.find(tc => tc.id === delta.tool_call!.id);
          if (existing) {
            existing.function.arguments += delta.tool_call.arguments;
          } else {
            toolCalls.push({
              id: delta.tool_call.id,
              type: 'function',
              function: {
                name: delta.tool_call.name,
                arguments: delta.tool_call.arguments,
              },
            });
          }
        }
      } else if (event.type === 'response.done') {
        const done = event as { response: { usage?: { input_tokens: number; output_tokens: number; total_tokens: number } } };
        if (done.response.usage) {
          usage = {
            prompt_tokens: done.response.usage.input_tokens,
            completion_tokens: done.response.usage.output_tokens,
            total_tokens: done.response.usage.total_tokens,
          };
        }
      }
    }
    
    const finishReason = toolCalls.length > 0 ? 'tool_calls' : 'stop';
    
    return {
      id: `chatcmpl-${generateEventId()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: content || null,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        finish_reason: finishReason,
      }],
      usage,
    };
  }
  
  /**
   * Convert UAMP event to SSE chunk
   */
  fromUAMPStreaming(event: ServerEvent, model: string): string | null {
    if (event.type === 'response.delta') {
      const delta = (event as { delta: ResponseDelta }).delta;
      const chunk = {
        id: `chatcmpl-${generateEventId()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          delta: {
            content: delta.text || undefined,
            tool_calls: delta.tool_call ? [{
              index: 0,
              id: delta.tool_call.id,
              type: 'function' as const,
              function: {
                name: delta.tool_call.name,
                arguments: delta.tool_call.arguments,
              },
            }] : undefined,
          },
          finish_reason: null,
        }],
      };
      return `data: ${JSON.stringify(chunk)}\n\n`;
    } else if (event.type === 'response.done') {
      const done = event as { response: { usage?: { input_tokens: number; output_tokens: number; total_tokens: number } } };
      const chunk = {
        id: `chatcmpl-${generateEventId()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          delta: {},
          finish_reason: 'stop',
        }],
        usage: done.response.usage ? {
          prompt_tokens: done.response.usage.input_tokens,
          completion_tokens: done.response.usage.output_tokens,
          total_tokens: done.response.usage.total_tokens,
        } : undefined,
      };
      return `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`;
    }
    return null;
  }
  
  /**
   * Handle chat completions endpoint
   */
  @http({ path: '/v1/chat/completions', method: 'POST' })
  async handleCompletions(request: Request, context: Context): Promise<Response> {
    if (!this.agent) {
      return new Response(JSON.stringify({ error: { message: 'No agent configured' } }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Transport-agnostic payment token: set from HTTP headers so PaymentX402Skill can read it
    const paymentHeader = request.headers.get('X-Payment-Token') ?? request.headers.get('x-payment-token')
      ?? request.headers.get('X-PAYMENT') ?? request.headers.get('x-payment');
    if (paymentHeader) {
      context.set('payment_token', paymentHeader);
    }

    // Also check context for payment token (e.g. set by portal transport-context)
    const paymentToken = paymentHeader || context.get?.('payment_token') as string | undefined;

    try {
      const body = await request.json() as ChatCompletionRequest;
      const uampEvents = this.toUAMP(body, paymentToken);

      if (body.stream) {
        // Streaming: pre-flight to avoid committing 200 before payment check (same as Python)
        const encoder = new TextEncoder();
        const iterator = this.agent.processUAMP(uampEvents)[Symbol.asyncIterator]();
        let firstEvent: IteratorResult<ServerEvent> | null = null;
        try {
          firstEvent = await iterator.next();
        } catch (err) {
          if (err instanceof PaymentRequiredError) {
            return new Response(JSON.stringify({
              error: err.message,
              status_code: 402,
              context: { accepts: err.accepts },
            }), {
              status: 402,
              headers: { 'Content-Type': 'application/json' },
            });
          }
          throw err;
        }
        const stream = new ReadableStream({
          start: async (controller) => {
            try {
              if (firstEvent?.done !== true && firstEvent?.value) {
                const chunk = this.fromUAMPStreaming(firstEvent.value, body.model);
                if (chunk) controller.enqueue(encoder.encode(chunk));
              }
              if (firstEvent?.done === true) {
                controller.close();
                return;
              }
              for (;;) {
                const next = await iterator.next();
                if (next.done) break;
                if (next.value) {
                  const chunk = this.fromUAMPStreaming(next.value, body.model);
                  if (chunk) controller.enqueue(encoder.encode(chunk));
                }
              }
              controller.close();
            } catch (error) {
              if (error instanceof PaymentRequiredError) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                  error: error.message,
                  status_code: 402,
                  context: { accepts: error.accepts },
                })}\n\n`));
              }
              controller.error(error);
            }
          },
        });

        return new Response(stream, {
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          },
        });
      }

      // Non-streaming response
      const events: ServerEvent[] = [];
      try {
        for await (const event of this.agent.processUAMP(uampEvents)) {
          events.push(event);
        }
      } catch (err) {
        if (err instanceof PaymentRequiredError) {
          return new Response(JSON.stringify({
            error: err.message,
            status_code: 402,
            context: { accepts: err.accepts },
          }), {
            status: 402,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        throw err;
      }

      const response = this.fromUAMP(events, body.model);
      return new Response(JSON.stringify(response), {
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (error) {
      if (error instanceof PaymentRequiredError) {
        return new Response(JSON.stringify({
          error: error.message,
          status_code: 402,
          context: { accepts: error.accepts },
        }), {
          status: 402,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify({
        error: {
          message: (error as Error).message,
          type: 'invalid_request_error',
        },
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  }
  
  /**
   * Handle models endpoint
   */
  @http({ path: '/v1/models', method: 'GET' })
  async handleModels(_request: Request, _context: Context): Promise<Response> {
    const capabilities = this.agent?.getCapabilities();
    const models = [{
      id: capabilities?.id || 'default',
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: capabilities?.provider || 'webagents',
    }];
    
    return new Response(JSON.stringify({ object: 'list', data: models }), {
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
