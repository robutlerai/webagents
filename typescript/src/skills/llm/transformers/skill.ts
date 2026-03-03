/**
 * Transformers.js Skill
 * 
 * In-browser LLM inference using @huggingface/transformers.
 * Supports WebGPU with WASM fallback.
 * 
 * @see https://github.com/huggingface/transformers.js
 * @see https://huggingface.co/docs/transformers.js
 */

import { Skill } from '../../../core/skill.js';
import { handoff } from '../../../core/decorators.js';
import type { SkillConfig, Context } from '../../../core/types.js';
import type { Capabilities, ContentItem, UsageStats } from '../../../uamp/types.js';
import type { ClientEvent, ServerEvent, InputTextEvent, SessionCreateEvent } from '../../../uamp/events.js';
import { generateEventId } from '../../../uamp/events.js';

// Types for @huggingface/transformers (peer dependency)
interface TextGenerationPipeline {
  (
    text: string | string[],
    options?: {
      max_new_tokens?: number;
      temperature?: number;
      top_p?: number;
      do_sample?: boolean;
      callback_function?: (output: string) => void;
    }
  ): Promise<Array<{ generated_text: string }>>;
  tokenizer?: {
    decode(tokens: number[]): string;
  };
}

interface ProgressCallback {
  (progress: { status: string; progress?: number; file?: string }): void;
}

type PipelineFn = (
  task: 'text-generation',
  model: string,
  options?: {
    device?: 'webgpu' | 'wasm' | 'cpu';
    progress_callback?: ProgressCallback;
    dtype?: string;
  }
) => Promise<TextGenerationPipeline>;

/**
 * Transformers.js skill configuration
 */
export interface TransformersSkillConfig extends SkillConfig {
  /** Model ID from Hugging Face Hub (e.g., 'Xenova/Phi-3-mini-4k-instruct') */
  model: string;
  /** Device to use: 'webgpu' (default), 'wasm', or 'cpu' */
  device?: 'webgpu' | 'wasm' | 'cpu';
  /** Temperature for generation */
  temperature?: number;
  /** Max tokens to generate */
  max_tokens?: number;
  /** Top-p sampling */
  top_p?: number;
  /** Data type (e.g., 'fp16', 'q4', 'q8') */
  dtype?: string;
}

/**
 * Transformers.js Skill for in-browser LLM inference
 * 
 * @example
 * ```typescript
 * import { TransformersSkill } from 'webagents/skills/llm/transformers';
 * 
 * const skill = new TransformersSkill({
 *   model: 'Xenova/Phi-3-mini-4k-instruct',
 *   device: 'webgpu'
 * });
 * 
 * const agent = new BaseAgent({ skills: [skill] });
 * ```
 */
export class TransformersSkill extends Skill {
  private pipeline: TextGenerationPipeline | null = null;
  private pipelineFn: PipelineFn | null = null;
  private modelConfig: TransformersSkillConfig;
  private isInitializing = false;
  
  constructor(config: TransformersSkillConfig) {
    super({ ...config, name: config.name || 'transformers' });
    this.modelConfig = config;
  }
  
  /**
   * Initialize the skill - loads the transformers.js library
   */
  async initialize(): Promise<void> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const transformers = await import('@huggingface/transformers' as any);
      this.pipelineFn = transformers.pipeline as unknown as PipelineFn;
    } catch {
      console.warn('Transformers.js not available - @huggingface/transformers not installed');
    }
  }
  
  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.pipeline = null;
  }
  
  /**
   * Get Transformers.js capabilities
   */
  getCapabilities(): Capabilities {
    return {
      id: this.modelConfig.model,
      provider: 'transformers.js',
      modalities: ['text'],
      supports_streaming: true,
      supports_thinking: false,
      supports_caching: true,
      extensions: {
        runtime: 'browser',
        engine: 'webgpu+wasm',
        device: this.modelConfig.device || 'webgpu',
      },
    };
  }
  
  /**
   * Ensure pipeline is initialized
   */
  private async ensurePipeline(
    onProgress?: (status: string, progress?: number) => void
  ): Promise<TextGenerationPipeline> {
    if (this.pipeline) {
      return this.pipeline;
    }
    
    if (!this.pipelineFn) {
      throw new Error('Transformers.js not available - call initialize() first or install @huggingface/transformers');
    }
    
    if (this.isInitializing) {
      while (this.isInitializing) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (this.pipeline) return this.pipeline;
    }
    
    this.isInitializing = true;
    
    try {
      this.pipeline = await this.pipelineFn('text-generation', this.modelConfig.model, {
        device: this.modelConfig.device || 'webgpu',
        dtype: this.modelConfig.dtype,
        progress_callback: (progress) => {
          if (onProgress) {
            onProgress(progress.status, progress.progress ? progress.progress * 100 : undefined);
          }
        },
      });
      return this.pipeline;
    } finally {
      this.isInitializing = false;
    }
  }
  
  /**
   * Format messages into a prompt string
   */
  private formatPrompt(events: ClientEvent[]): string {
    const parts: string[] = [];
    
    for (const event of events) {
      if (event.type === 'session.create') {
        const createEvent = event as SessionCreateEvent;
        if (createEvent.session.instructions) {
          parts.push(`System: ${createEvent.session.instructions}`);
        }
      } else if (event.type === 'input.text') {
        const inputEvent = event as InputTextEvent;
        const role = inputEvent.role === 'system' ? 'System' : 'User';
        parts.push(`${role}: ${inputEvent.text}`);
      }
    }
    
    parts.push('Assistant:');
    return parts.join('\n\n');
  }
  
  /**
   * Process UAMP events with Transformers.js
   */
  @handoff({ name: 'transformers', priority: 4 })
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
      // Ensure pipeline is loaded
      const pipeline = await this.ensurePipeline((status, progress) => {
        console.log(`Transformers.js: ${status}${progress ? ` (${progress.toFixed(0)}%)` : ''}`);
      });
      
      // Format prompt
      const prompt = this.formatPrompt(events);
      
      if (!prompt || prompt === 'Assistant:') {
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
      
      // Generate response
      // Note: Transformers.js streaming is callback-based, we collect and yield
      let generatedText = '';
      
      const result = await pipeline(prompt, {
        max_new_tokens: this.modelConfig.max_tokens ?? 512,
        temperature: this.modelConfig.temperature ?? 0.7,
        top_p: this.modelConfig.top_p ?? 0.95,
        do_sample: true,
        callback_function: (output: string) => {
          // This gets called with partial output
          const newText = output.slice(generatedText.length);
          if (newText) {
            generatedText = output;
          }
        },
      });
      
      // Extract generated text (remove the prompt prefix)
      const fullOutput = result[0]?.generated_text || '';
      const responseText = fullOutput.startsWith(prompt) 
        ? fullOutput.slice(prompt.length).trim()
        : fullOutput.trim();
      
      // Yield as single delta (streaming would require different approach)
      if (responseText) {
        yield {
          type: 'response.delta',
          event_id: generateEventId(),
          response_id: responseId,
          delta: {
            type: 'text',
            text: responseText,
          },
        };
      }
      
      // Yield response done
      const output: ContentItem[] = responseText ? [{ type: 'text', text: responseText }] : [];
      const usage: UsageStats = {
        input_tokens: 0, // Transformers.js doesn't easily provide token counts
        output_tokens: 0,
        total_tokens: 0,
      };
      
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
          code: 'transformers_error',
          message: (error as Error).message,
        },
      };
    }
  }
}
