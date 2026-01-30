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

// Google AI types (simplified)
interface GoogleAI {
  getGenerativeModel(config: { model: string }): GenerativeModel;
}

interface GenerativeModel {
  generateContentStream(request: GenerateContentRequest): Promise<GenerateContentStreamResult>;
}

interface GenerateContentRequest {
  contents: Array<{ role: string; parts: Array<{ text: string }> }>;
  systemInstruction?: { parts: Array<{ text: string }> };
  generationConfig?: { temperature?: number; maxOutputTokens?: number; topP?: number };
}

interface GenerateContentStreamResult {
  stream: AsyncIterable<{ text(): string }>;
  response: Promise<{ usageMetadata?: { promptTokenCount: number; candidatesTokenCount: number } }>;
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
  
  private extractContents(events: ClientEvent[]): {
    systemInstruction?: { parts: Array<{ text: string }> };
    contents: Array<{ role: string; parts: Array<{ text: string }> }>;
  } {
    let systemText = '';
    const contents: Array<{ role: string; parts: Array<{ text: string }> }> = [];
    
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
  
  @handoff({ name: 'google', priority: 8 })
  async *processUAMP(
    events: ClientEvent[],
    _context: Context
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
      
      const { systemInstruction, contents } = this.extractContents(events);
      
      if (contents.length === 0) {
        yield {
          type: 'response.error',
          event_id: generateEventId(),
          response_id: responseId,
          error: { code: 'no_input', message: 'No input messages provided' },
        };
        return;
      }
      
      const model = this.genAI.getGenerativeModel({
        model: this.modelConfig.model || 'gemini-1.5-pro',
      });
      
      const result = await model.generateContentStream({
        contents,
        systemInstruction,
        generationConfig: {
          temperature: this.modelConfig.temperature ?? 0.7,
          maxOutputTokens: this.modelConfig.max_tokens ?? 4096,
        },
      });
      
      let fullContent = '';
      
      for await (const chunk of result.stream) {
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
      }
      
      // Get final usage
      const response = await result.response;
      const usage: UsageStats | undefined = response.usageMetadata ? {
        input_tokens: response.usageMetadata.promptTokenCount,
        output_tokens: response.usageMetadata.candidatesTokenCount,
        total_tokens: response.usageMetadata.promptTokenCount + response.usageMetadata.candidatesTokenCount,
      } : undefined;
      
      const output: ContentItem[] = fullContent ? [{ type: 'text', text: fullContent }] : [];
      
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
