/**
 * WebLLM Skill
 * 
 * In-browser LLM inference using WebGPU via @mlc-ai/web-llm.
 * Primary skill for Elaisium browser-based agents.
 * 
 * @see https://github.com/mlc-ai/web-llm
 * @see https://webllm.mlc.ai/docs/user/get_started.html
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, UsageStats, ContentItem } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

// Types for @mlc-ai/web-llm (peer dependency)
interface MLCEngine {
  chat: {
    completions: {
      create(params: ChatCompletionParams): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>;
    };
  };
  getMessage(): Promise<string>;
  resetChat(): Promise<void>;
  unload(): Promise<void>;
}

interface ChatCompletionParams {
  messages: Array<{ role: string; content: string }>;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stream_options?: { include_usage?: boolean };
}

interface ChatCompletion {
  id: string;
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface ChatCompletionChunk {
  id: string;
  choices: Array<{
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface InitProgressCallback {
  (progress: { progress: number; text: string; timeElapsed?: number }): void;
}

type CreateMLCEngineFn = (
  model: string,
  config?: { initProgressCallback?: InitProgressCallback }
) => Promise<MLCEngine>;

/**
 * WebLLM skill configuration
 */
export interface WebLLMSkillConfig extends SkillConfig {
  /** Model ID (e.g., 'Llama-3.1-8B-Instruct-q4f32_1-MLC') */
  model: string;
  /** Temperature for generation */
  temperature?: number;
  /** Max tokens to generate */
  max_tokens?: number;
  /** Top-p sampling */
  top_p?: number;
}

/**
 * WebLLM Skill for in-browser LLM inference
 * 
 * @example
 * ```typescript
 * import { WebLLMSkill } from 'webagents/skills/llm/webllm';
 * 
 * const skill = new WebLLMSkill({
 *   model: 'Llama-3.1-8B-Instruct-q4f32_1-MLC'
 * });
 * 
 * const agent = new BaseAgent({ skills: [skill] });
 * ```
 */
export class WebLLMSkill extends Skill {
  private engine: MLCEngine | null = null;
  private createEngine: CreateMLCEngineFn | null = null;
  private modelConfig: WebLLMSkillConfig;
  private isInitializing = false;
  
  constructor(config: WebLLMSkillConfig) {
    super({ ...config, name: config.name || 'webllm' });
    this.modelConfig = config;
  }
  
  /**
   * Initialize the skill - loads the WebLLM library
   */
  async initialize(): Promise<void> {
    // Dynamically import @mlc-ai/web-llm (peer dependency)
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const webllm = await import('@mlc-ai/web-llm' as any);
      this.createEngine = webllm.CreateMLCEngine;
    } catch {
      console.warn('WebLLM not available - @mlc-ai/web-llm not installed');
    }
  }
  
  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    if (this.engine) {
      await this.engine.unload();
      this.engine = null;
    }
  }
  
  /**
   * Get WebLLM capabilities
   */
  getCapabilities(): Capabilities {
    return {
      id: this.modelConfig.model,
      provider: 'webllm',
      modalities: ['text'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: true,
      extensions: {
        runtime: 'browser',
        engine: 'webgpu',
      },
    };
  }
  
  /**
   * Ensure engine is initialized
   */
  private async ensureEngine(
    onProgress?: (progress: number, text: string) => void
  ): Promise<MLCEngine> {
    if (this.engine) {
      return this.engine;
    }
    
    if (!this.createEngine) {
      throw new Error('WebLLM not available - call initialize() first or install @mlc-ai/web-llm');
    }
    
    if (this.isInitializing) {
      // Wait for initialization to complete
      while (this.isInitializing) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (this.engine) return this.engine;
    }
    
    this.isInitializing = true;
    
    try {
      this.engine = await this.createEngine(this.modelConfig.model, {
        initProgressCallback: (progress) => {
          if (onProgress) {
            onProgress(progress.progress * 100, progress.text);
          }
        },
      });
      return this.engine;
    } finally {
      this.isInitializing = false;
    }
  }
  
  /**
   * Extract messages from UAMP events
   */
  private extractMessages(events: ClientEvent[]): Array<{ role: string; content: string }> {
    const messages: Array<{ role: string; content: string }> = [];
    
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.instructions) {
          messages.push({
            role: 'system',
            content: createEvent.session.instructions,
          });
        }
      } else if (event.type === 'input.text') {
        const inputEvent = event as InputTextEvent;
        messages.push({
          role: inputEvent.role || 'user',
          content: inputEvent.text,
        });
      }
    }
    
    return messages;
  }
  
  /**
   * Process UAMP events with WebLLM
   */
  @handoff({ name: 'webllm', priority: 5 })
  async *processUAMP(
    events: ClientEvent[],
    _context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    const responseId = generateEventId();
    
    // Yield response started
    yield {
      type: 'response.created',
      event_id: generateEventId(),
      response_id: responseId,
    };
    
    try {
      // Ensure engine is loaded, yield progress events
      const engine = await this.ensureEngine((percent, text) => {
        // Could yield progress events here if needed
        console.log(`WebLLM loading: ${percent.toFixed(0)}% - ${text}`);
      });
      
      // Extract messages from events
      const messages = this.extractMessages(events);
      
      if (messages.length === 0) {
        yield {
          type: 'response.error',
          event_id: generateEventId(),
          response_id: responseId,
          error: {
            code: 'no_input',
            message: 'No input messages provided',
          },
        };
        return;
      }
      
      // Call WebLLM with streaming
      const stream = await engine.chat.completions.create({
        messages,
        stream: true,
        temperature: this.modelConfig.temperature ?? 0.7,
        max_tokens: this.modelConfig.max_tokens ?? 2048,
        top_p: this.modelConfig.top_p ?? 0.95,
        stream_options: { include_usage: true },
      });
      
      let fullContent = '';
      let usage: UsageStats | undefined;
      
      // Stream response chunks
      for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) {
          fullContent += delta;
          yield {
            type: 'response.delta',
            event_id: generateEventId(),
            response_id: responseId,
            delta: {
              type: 'text',
              text: delta,
            },
          };
        }
        
        // Capture usage from last chunk
        if (chunk.usage) {
          usage = {
            input_tokens: chunk.usage.prompt_tokens,
            output_tokens: chunk.usage.completion_tokens,
            total_tokens: chunk.usage.total_tokens,
          };
        }
      }
      
      // Yield response done
      const output: ContentItem[] = fullContent ? [{ type: 'text', text: fullContent }] : [];
      
      yield {
        type: 'response.done',
        event_id: generateEventId(),
        response_id: responseId,
        response: {
          id: responseId,
          status: 'completed',
          output,
          usage,
        },
      };
    } catch (error) {
      yield {
        type: 'response.error',
        event_id: generateEventId(),
        response_id: responseId,
        error: {
          code: 'webllm_error',
          message: (error as Error).message,
        },
      };
    }
  }
}
