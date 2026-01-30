/**
 * Speech-to-Text Skill
 * 
 * Uses Whisper or Moonshine models via Transformers.js
 * for automatic speech recognition in the browser.
 * 
 * Routing capabilities:
 * - subscribes: ['input.audio'] - processes audio input
 * - produces: ['input.text'] - outputs text for LLM processing
 */

import { Skill } from '../../core/skill.js';
import { tool, handoff } from '../../core/decorators.js';
import type { ClientEvent, ServerEvent } from '../../uamp/events.js';
import type { Context } from '../../core/types.js';

export interface STTConfig {
  /** Model ID (default: 'Xenova/whisper-tiny.en') */
  model?: string;
  /** Language code for transcription */
  language?: string;
  /** Chunk length in seconds for long audio */
  chunkLength?: number;
  /** Stride length in seconds */
  strideLength?: number;
}

interface TranscriptionResult {
  text: string;
  chunks?: Array<{
    text: string;
    timestamp: [number, number];
  }>;
  language?: string;
}

/**
 * Speech-to-Text skill using Whisper/Moonshine models
 * 
 * @example
 * ```typescript
 * const stt = new SpeechToTextSkill({ model: 'Xenova/whisper-small.en' });
 * await stt.initialize();
 * 
 * const result = await stt.transcribe(audioBlob);
 * console.log(result.text);
 * ```
 */
export class SpeechToTextSkill extends Skill {
  private config: STTConfig;
  private pipeline: any = null;
  private isInitialized = false;

  // Available models (from HuggingFace)
  static MODELS = {
    // Whisper models (OpenAI) - most accurate
    'whisper-tiny.en': 'Xenova/whisper-tiny.en',
    'whisper-small.en': 'Xenova/whisper-small.en',
    'whisper-medium.en': 'Xenova/whisper-medium.en',
    'whisper-large-v3': 'onnx-community/whisper-large-v3',
    // Moonshine models (Useful Sensors) - faster, smaller
    'moonshine-tiny': 'onnx-community/moonshine-tiny',
    'moonshine-base': 'onnx-community/moonshine-base',
  };

  constructor(config: STTConfig = {}) {
    super();
    this.config = {
      model: config.model || 'Xenova/whisper-tiny.en',
      language: config.language || 'en',
      chunkLength: config.chunkLength || 30,
      strideLength: config.strideLength || 5,
    };
  }

  get id(): string {
    return 'speech-to-text';
  }

  get name(): string {
    return 'Speech-to-Text';
  }

  get description(): string {
    return 'Transcribe audio to text using Whisper or Moonshine models';
  }

  /**
   * Initialize the STT pipeline
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    // Dynamic import for browser compatibility
    const { pipeline, env } = await import('@huggingface/transformers');
    env.allowLocalModels = false;

    this.pipeline = await pipeline(
      'automatic-speech-recognition',
      this.config.model,
      {
        dtype: 'q4', // Quantized for performance
      }
    );

    this.isInitialized = true;
  }

  /**
   * Transcribe audio to text
   */
  @tool({
    name: 'transcribe',
    description: 'Transcribe audio to text',
    parameters: {
      audio: { type: 'string', description: 'Audio data (base64 or URL)' },
      returnTimestamps: { type: 'boolean', description: 'Return word timestamps' },
    },
  })
  async transcribe(
    audio: Blob | ArrayBuffer | Float32Array | string,
    options: { returnTimestamps?: boolean } = {}
  ): Promise<TranscriptionResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    // Convert input to appropriate format
    let audioInput: any = audio;
    if (audio instanceof Blob) {
      audioInput = await audio.arrayBuffer();
    }

    const result = await this.pipeline(audioInput, {
      chunk_length_s: this.config.chunkLength,
      stride_length_s: this.config.strideLength,
      return_timestamps: options.returnTimestamps,
      language: this.config.language,
    });

    return {
      text: result.text,
      chunks: result.chunks,
      language: this.config.language,
    };
  }

  /**
   * Stream transcription (for real-time)
   */
  async *transcribeStream(
    audioStream: ReadableStream<Float32Array>
  ): AsyncGenerator<{ text: string; isFinal: boolean }> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const reader = audioStream.getReader();
    let buffer: Float32Array[] = [];
    const sampleRate = 16000;
    const chunkSize = sampleRate * this.config.chunkLength!;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer.push(value);
        const totalSamples = buffer.reduce((sum, chunk) => sum + chunk.length, 0);

        if (totalSamples >= chunkSize) {
          // Concatenate buffer
          const audio = new Float32Array(totalSamples);
          let offset = 0;
          for (const chunk of buffer) {
            audio.set(chunk, offset);
            offset += chunk.length;
          }

          const result = await this.pipeline(audio, {
            chunk_length_s: this.config.chunkLength,
            stride_length_s: this.config.strideLength,
          });

          yield { text: result.text, isFinal: false };
          buffer = [];
        }
      }

      // Process remaining audio
      if (buffer.length > 0) {
        const totalSamples = buffer.reduce((sum, chunk) => sum + chunk.length, 0);
        const audio = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of buffer) {
          audio.set(chunk, offset);
          offset += chunk.length;
        }

        const result = await this.pipeline(audio);
        yield { text: result.text, isFinal: true };
      }
    } finally {
      reader.releaseLock();
    }
  }

  async cleanup(): Promise<void> {
    this.pipeline = null;
    this.isInitialized = false;
  }

  // ============================================================================
  // UAMP Handoff for Router Integration
  // ============================================================================

  /**
   * UAMP handoff for processing audio input events
   * 
   * Subscribes to 'input.audio' events and produces 'input.text' events
   * for downstream processing by LLM skills.
   */
  @handoff({
    name: 'speech-to-text',
    subscribes: ['input.audio'],
    produces: ['input.text'],
    priority: 100, // High priority for audio input
    description: 'Transcribe audio to text',
  })
  async *processAudioUAMP(
    events: ClientEvent[],
    context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    for (const event of events) {
      if (event.type === 'input.audio') {
        const audioEvent = event as { audio?: string | ArrayBuffer; event_id?: string };
        
        if (audioEvent.audio) {
          try {
            // Convert base64 to ArrayBuffer if needed
            let audioData: ArrayBuffer;
            if (typeof audioEvent.audio === 'string') {
              // Assume base64 encoded
              const binaryString = atob(audioEvent.audio);
              const bytes = new Uint8Array(binaryString.length);
              for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }
              audioData = bytes.buffer;
            } else {
              audioData = audioEvent.audio;
            }

            // Transcribe audio
            const result = await this.transcribe(audioData);

            // Yield text event for router to route to LLM
            yield {
              type: 'input.text',
              event_id: `stt-${Date.now()}`,
              text: result.text,
              role: 'user',
            } as unknown as ServerEvent;
          } catch (error) {
            yield {
              type: 'response.error',
              event_id: `error-${Date.now()}`,
              error: {
                type: 'transcription_error',
                message: (error as Error).message,
              },
            } as unknown as ServerEvent;
          }
        }
      }
    }
  }
}
