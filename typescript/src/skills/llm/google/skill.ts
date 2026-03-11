/**
 * Google AI Skill
 * 
 * Cloud LLM inference using Google's Gemini API.
 * 
 * @see https://ai.google.dev/gemini-api/docs
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

// Google AI types
interface GooglePart {
  text?: string;
  functionCall?: { name: string; args: Record<string, unknown> };
  functionResponse?: { name: string; response: Record<string, unknown> };
}

interface GoogleContent {
  role: string;
  parts: GooglePart[];
}

interface GoogleFunctionDeclaration {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

interface GoogleAI {
  getGenerativeModel(config: { model: string }): GenerativeModel;
}

interface GenerativeModel {
  generateContentStream(request: GenerateContentRequest): Promise<GenerateContentStreamResult>;
}

interface GenerateContentRequest {
  contents: GoogleContent[];
  systemInstruction?: { parts: Array<{ text: string }> };
  generationConfig?: { temperature?: number; maxOutputTokens?: number; topP?: number };
  tools?: Array<{ functionDeclarations: GoogleFunctionDeclaration[] }>;
}

interface StreamChunkCandidate {
  content?: { parts?: GooglePart[] };
  finishReason?: string;
}

interface GenerateContentStreamResult {
  stream: AsyncIterable<{ text(): string; candidates?: StreamChunkCandidate[] }>;
  response: Promise<{
    usageMetadata?: { promptTokenCount: number; candidatesTokenCount: number };
    candidates?: StreamChunkCandidate[];
  }>;
}

type GoogleAIConstructor = new (apiKey: string) => GoogleAI;

/**
 * Google AI skill configuration
 */
export interface GoogleSkillConfig extends SkillConfig {
  /** API key (defaults to GOOGLE_API_KEY env var) */
  apiKey?: string;
  /** Model ID (e.g., 'gemini-1.5-pro') */
  model?: string;
  /** Temperature */
  temperature?: number;
  /** Max tokens */
  max_tokens?: number;
}

/**
 * Google AI Skill for Gemini models
 */
export class GoogleSkill extends Skill {
  private genAI: GoogleAI | null = null;
  private GoogleAIClass: GoogleAIConstructor | null = null;
  private modelConfig: GoogleSkillConfig;
  
  constructor(config: GoogleSkillConfig = {}) {
    super({ ...config, name: config.name || 'google' });
    this.modelConfig = config;
  }
  
  async initialize(): Promise<void> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const googleAI = await import('@google/generative-ai' as any);
      this.GoogleAIClass = googleAI.GoogleGenerativeAI as unknown as GoogleAIConstructor;
      
      const apiKey = this.modelConfig.apiKey || 
        (typeof process !== 'undefined' ? process.env?.GOOGLE_API_KEY : undefined);
      
      if (apiKey) {
        this.genAI = new this.GoogleAIClass(apiKey);
      }
    } catch {
      console.warn('Google AI SDK not available - @google/generative-ai not installed');
    }
  }
  
  getCapabilities(): Capabilities {
    const model = this.modelConfig.model || 'gemini-1.5-pro';
    return {
      id: model,
      provider: 'google',
      modalities: ['text', 'image', 'audio', 'video'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: true,
      tools: {
        supports_tools: true,
        supports_parallel_tools: true,
        supports_streaming_tools: false,
        built_in_tools: ['code_execution'],
      },
      context_window: model.includes('1.5') ? 1000000 : 128000,
    };
  }
  
  /**
   * Extract tool definitions from session.create events and convert to Google format.
   */
  private extractToolDefinitions(events: ClientEvent[]): GoogleFunctionDeclaration[] {
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.tools && createEvent.session.tools.length > 0) {
          return createEvent.session.tools.map(t => ({
            name: t.function.name,
            description: t.function.description,
            parameters: t.function.parameters as Record<string, unknown> | undefined,
          }));
        }
      }
    }
    return [];
  }

  private extractContents(events: ClientEvent[]): {
    systemInstruction?: { parts: Array<{ text: string }> };
    contents: GoogleContent[];
  } {
    let systemText = '';
    const contents: GoogleContent[] = [];
    
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.instructions) {
          systemText = createEvent.session.instructions;
        }
      } else if (event.type === 'input.text') {
        const inputEvent = event as InputTextEvent;
        if (inputEvent.role === 'system') {
          systemText = (systemText ? systemText + '\n\n' : '') + inputEvent.text;
        } else {
          contents.push({ role: 'user', parts: [{ text: inputEvent.text }] });
        }
      }
    }
    
    return {
      systemInstruction: systemText ? { parts: [{ text: systemText }] } : undefined,
      contents,
    };
  }

  /**
   * Resolve tool_call_id back to function name by scanning previous messages.
   */
  private resolveToolName(
    messages: Array<{ role: string; tool_calls?: Array<{ id: string; function: { name: string } }> }>,
    toolCallId: string,
  ): string {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role === 'assistant' && msg.tool_calls) {
        const tc = msg.tool_calls.find(t => t.id === toolCallId);
        if (tc) return tc.function.name;
      }
    }
    return toolCallId;
  }

  /**
   * Convert agentic messages to Google Gemini content format.
   * Handles assistant tool_calls -> functionCall parts,
   * and tool results -> functionResponse parts.
   */
  private agenticToGoogleContents(
    agenticMessages: Array<{
      role: string;
      content: string | null;
      tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
      tool_call_id?: string;
    }>
  ): { systemInstruction?: { parts: Array<{ text: string }> }; contents: GoogleContent[] } {
    let systemText = '';
    const contents: GoogleContent[] = [];

    for (const msg of agenticMessages) {
      if (msg.role === 'system') {
        systemText += (systemText ? '\n\n' : '') + (msg.content || '');
      } else if (msg.role === 'assistant') {
        const parts: GooglePart[] = [];
        if (msg.content) {
          parts.push({ text: msg.content });
        }
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            let args: Record<string, unknown> = {};
            try { args = JSON.parse(tc.function.arguments); } catch { /* use empty */ }
            parts.push({ functionCall: { name: tc.function.name, args } });
          }
        }
        if (parts.length > 0) {
          contents.push({ role: 'model', parts });
        }
      } else if (msg.role === 'tool') {
        const fnName = this.resolveToolName(agenticMessages as any[], msg.tool_call_id || '');
        let resultObj: Record<string, unknown>;
        try {
          resultObj = JSON.parse(msg.content || '""');
          if (typeof resultObj !== 'object' || resultObj === null) {
            resultObj = { result: msg.content || '' };
          }
        } catch {
          resultObj = { result: msg.content || '' };
        }
        contents.push({
          role: 'user',
          parts: [{ functionResponse: { name: fnName, response: resultObj } }],
        });
      } else {
        contents.push({ role: 'user', parts: [{ text: msg.content || '' }] });
      }
    }

    return {
      systemInstruction: systemText ? { parts: [{ text: systemText }] } : undefined,
      contents,
    };
  }
  
  @handoff({ name: 'google', priority: 8 })
  async *processUAMP(
    events: ClientEvent[],
    context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();
    
    yield {
      type: 'response.created',
      event_id: generateEventId(),
      response_id: responseId,
    };
    
    try {
      if (!this.genAI) {
        throw new Error('Google AI client not initialized');
      }
      
      let systemInstruction: { parts: Array<{ text: string }> } | undefined;
      let contents: GoogleContent[];

      const agenticMessages = context?.get ? context.get<Array<{
        role: string;
        content: string | null;
        tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
        tool_call_id?: string;
      }>>('_agentic_messages') : undefined;

      if (agenticMessages && agenticMessages.length > 0) {
        ({ systemInstruction, contents } = this.agenticToGoogleContents(agenticMessages));
      } else {
        ({ systemInstruction, contents } = this.extractContents(events));
      }
      
      if (contents.length === 0) {
        yield {
          type: 'response.error',
          event_id: generateEventId(),
          response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' },
        };
        return;
      }

      // Extract tools from session.create events or context
      const toolDefs = this.extractToolDefinitions(events);
      const contextTools = context?.get ? context.get<Array<{
        type: string; function: { name: string; description?: string; parameters?: Record<string, unknown> };
      }>>('_agentic_tools') : undefined;
      const finalToolDefs = contextTools && contextTools.length > 0
        ? contextTools.map(t => ({
            name: t.function.name,
            description: t.function.description,
            parameters: t.function.parameters,
          }))
        : toolDefs;
      
      const model = this.genAI.getGenerativeModel({
        model: this.modelConfig.model || 'gemini-1.5-pro',
      });

      const request: GenerateContentRequest = {
        contents,
        systemInstruction,
        generationConfig: {
          temperature: this.modelConfig.temperature ?? 0.7,
          maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
        },
      };

      if (finalToolDefs.length > 0) {
        request.tools = [{ functionDeclarations: finalToolDefs }];
      }
      
      const result = await model.generateContentStream(request);
      
      let fullContent = '';
      const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
      
      for await (const chunk of result.stream) {
        // Check for function calls in candidates
        const candidates = chunk.candidates;
        if (candidates?.[0]?.content?.parts) {
          for (const part of candidates[0].content.parts) {
            if (part.functionCall) {
              const callId = `call_${generateEventId().slice(0, 8)}`;
              const argsStr = JSON.stringify(part.functionCall.args || {});
              toolCalls.push({ id: callId, name: part.functionCall.name, arguments: argsStr });
              yield {
                type: 'response.delta',
                event_id: generateEventId(),
                response_id: responseId,
                delta: {
                  type: 'tool_call',
                  tool_call: { id: callId, name: part.functionCall.name, arguments: argsStr },
                },
              };
            }
          }
        }

        // Stream text content
        try {
          const text = chunk.text();
          if (text) {
            fullContent += text;
            yield {
              type: 'response.delta',
              event_id: generateEventId(),
              response_id: responseId,
              delta: { type: 'text', text },
            };
          }
        } catch {
          // chunk.text() throws if there are no text parts (e.g., function call only)
        }
      }
      
      // Get final usage
      const response = await result.response;

      // Also check final response for function calls not caught in stream
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.functionCall) {
            const alreadyHave = toolCalls.some(tc => tc.name === part.functionCall!.name);
            if (!alreadyHave) {
              const callId = `call_${generateEventId().slice(0, 8)}`;
              const argsStr = JSON.stringify(part.functionCall.args || {});
              toolCalls.push({ id: callId, name: part.functionCall.name, arguments: argsStr });
              yield {
                type: 'response.delta',
                event_id: generateEventId(),
                response_id: responseId,
                delta: {
                  type: 'tool_call',
                  tool_call: { id: callId, name: part.functionCall.name, arguments: argsStr },
                },
              };
            }
          }
        }
      }

      const usage: UsageStats | undefined = response.usageMetadata ? {
        input_tokens: response.usageMetadata.promptTokenCount,
        output_tokens: response.usageMetadata.candidatesTokenCount,
        total_tokens: response.usageMetadata.promptTokenCount + response.usageMetadata.candidatesTokenCount,
      } : undefined;
      
      const output: ContentItem[] = [];
      if (fullContent) {
        output.push({ type: 'text', text: fullContent });
      }
      for (const tc of toolCalls) {
        output.push({
          type: 'tool_call',
          tool_call: { id: tc.id, name: tc.name, arguments: tc.arguments },
        });
      }
      
      yield {
        type: 'response.done',
        event_id: generateEventId(),
        response_id: responseId,
        response: { id: responseId, status: 'completed', output, usage },
      };
    } catch (error) {
      yield {
        type: 'response.error',
        event_id: generateEventId(),
        response_id: responseId,
        error: { code: 'google_error', message: (error as Error).message },
      };
    }
  }
}
