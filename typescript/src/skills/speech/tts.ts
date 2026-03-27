/**
 * Text-to-Speech Skill
 * 
 * Uses SpeechT5 or Kokoro models via Transformers.js
 * for speech synthesis in the browser.
 * 
 * Routing capabilities:
 * - subscribes: ['response.delta', 'response.done'] - processes LLM responses
 * - produces: ['audio.delta'] - outputs audio for delivery to client
 */

import { Skill } from '../../core/skill';
import { tool, handoff } from '../../core/decorators';
import type { ClientEvent, ServerEvent } from '../../uamp/events';
import type { Context } from '../../core/types';

export interface TTSConfig {
  /** Model ID (default: 'Xenova/speecht5_tts') */
  model?: string;
  /** Speaker embeddings model */
  speakerModel?: string;
  /** Voice/speaker ID */
  voice?: string;
  [key: string]: unknown;
}

interface SynthesisResult {
  audio: Float32Array;
  sampleRate: number;
  duration: number;
}

/**
 * Text-to-Speech skill using SpeechT5 or Kokoro models
 * 
 * @example
 * ```typescript
 * const tts = new TextToSpeechSkill({ model: 'Xenova/speecht5_tts' });
 * await tts.initialize();
 * 
 * const result = await tts.synthesize('Hello, world!');
 * // Play audio...
 * ```
 */
export class TextToSpeechSkill extends Skill {
  protected override config: TTSConfig;
  private pipeline: any = null;
  private speakerEmbeddings: any = null;
  private isInitialized = false;

  // Available models (from HuggingFace)
  static MODELS = {
    // SpeechT5 (Microsoft) - good quality, medium size
    'speecht5': 'Xenova/speecht5_tts',
    // Kokoro (Papla/ETH) - high quality, multiple voices
    'kokoro': 'onnx-community/Kokoro-82M-v1.0-ONNX',
    // MMS TTS (Meta) - multilingual
    'mms-tts-eng': 'Xenova/mms-tts-eng',
  };

  // Speaker embeddings for SpeechT5
  static SPEAKERS = {
    'cmu_us_slt': 'Xenova/cmu-arctic-xvectors', // Female
  };

  constructor(config: TTSConfig = {}) {
    super({ name: 'Text-to-Speech', ...config });
    this.config = {
      model: config.model || 'Xenova/speecht5_tts',
      speakerModel: config.speakerModel || 'Xenova/cmu-arctic-xvectors',
      voice: config.voice || 'cmu_us_slt',
    };
  }

  get id(): string {
    return 'text-to-speech';
  }

  get description(): string {
    return 'Synthesize speech from text using SpeechT5 or Kokoro models';
  }

  /**
   * Initialize the TTS pipeline
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    // Dynamic import for browser compatibility
    const { pipeline, env } = await import('@huggingface/transformers');
    env.allowLocalModels = false;

    this.pipeline = await pipeline(
      'text-to-speech',
      this.config.model,
      {
        dtype: 'fp32', // TTS typically needs fp32
      }
    );

    // Load speaker embeddings for SpeechT5
    if (this.config.model?.includes('speecht5')) {
      const { AutoModel: _AutoModel } = await import('@huggingface/transformers');
      // Speaker embeddings are loaded from the model itself
      // For SpeechT5, we'll use default embeddings
    }

    this.isInitialized = true;
  }

  /**
   * Synthesize speech from text
   */
  @tool({
    name: 'synthesize',
    description: 'Convert text to speech audio',
    parameters: {
      text: { type: 'string', description: 'Text to synthesize' },
      voice: { type: 'string', description: 'Voice/speaker ID (optional)' },
    },
  })
  async synthesize(text: string, _voice?: string): Promise<SynthesisResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const result = await this.pipeline(text, {
      speaker_embeddings: this.speakerEmbeddings,
    });

    return {
      audio: result.audio,
      sampleRate: result.sampling_rate,
      duration: result.audio.length / result.sampling_rate,
    };
  }

  /**
   * Synthesize and play audio directly
   */
  async speak(text: string): Promise<void> {
    const result = await this.synthesize(text);
    await this.playAudio(result.audio, result.sampleRate);
  }

  /**
   * Play audio using Web Audio API
   */
  private async playAudio(audio: Float32Array, sampleRate: number): Promise<void> {
    if (typeof AudioContext === 'undefined') {
      throw new Error('AudioContext not available');
    }

    const audioContext = new AudioContext({ sampleRate });
    const buffer = audioContext.createBuffer(1, audio.length, sampleRate);
    buffer.copyToChannel(audio as Float32Array<ArrayBuffer>, 0);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);

    return new Promise((resolve) => {
      source.onended = () => {
        audioContext.close();
        resolve();
      };
      source.start();
    });
  }

  /**
   * Get audio as Blob for download/streaming
   */
  async synthesizeToBlob(text: string, _format: 'wav' | 'mp3' = 'wav'): Promise<Blob> {
    const result = await this.synthesize(text);
    
    // Convert Float32Array to WAV
    const wavData = this.encodeWav(result.audio, result.sampleRate);
    return new Blob([wavData], { type: 'audio/wav' });
  }

  /**
   * Encode audio as WAV
   */
  private encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Audio data
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }

    return buffer;
  }

  async cleanup(): Promise<void> {
    this.pipeline = null;
    this.speakerEmbeddings = null;
    this.isInitialized = false;
  }

  // ============================================================================
  // UAMP Handoff for Router Integration
  // ============================================================================

  /**
   * Buffer for accumulating text deltas before synthesis
   */
  private textBuffer = '';

  /**
   * UAMP handoff for processing response text events
   * 
   * Subscribes to 'response.delta' and 'response.done' events and produces
   * 'audio.delta' events for delivery to the client.
   */
  @handoff({
    name: 'text-to-speech',
    subscribes: ['response.delta', 'response.done'],
    produces: ['audio.delta'],
    priority: 100, // High priority for audio output
    description: 'Synthesize speech from LLM responses',
  })
  async *processTextUAMP(
    events: ClientEvent[],
    _context: Context
  ): AsyncGenerator<ServerEvent, void, unknown> {
    for (const event of events) {
      if ((event.type as string) === 'response.delta') {
        const deltaEvent = event as { delta?: { text?: string }; event_id?: string };
        
        if (deltaEvent.delta?.text) {
          // Accumulate text
          this.textBuffer += deltaEvent.delta.text;
          
          // Check if we have a sentence boundary
          const sentenceEnd = /[.!?]\s*$/;
          if (sentenceEnd.test(this.textBuffer)) {
            const text = this.textBuffer;
            this.textBuffer = '';
            
            try {
              const result = await this.synthesize(text);
              
              // Convert Float32Array to base64 for transmission
              const audioBase64 = this.float32ToBase64(result.audio);
              
              yield {
                type: 'audio.delta',
                event_id: `tts-${Date.now()}`,
                audio: audioBase64,
                sample_rate: result.sampleRate,
              } as unknown as ServerEvent;
            } catch (error) {
              yield {
                type: 'response.error',
                event_id: `error-${Date.now()}`,
                error: {
                  type: 'synthesis_error',
                  message: (error as Error).message,
                },
              } as unknown as ServerEvent;
            }
          }
        }
      } else if ((event.type as string) === 'response.done') {
        // Flush any remaining text
        if (this.textBuffer.trim()) {
          const text = this.textBuffer;
          this.textBuffer = '';
          
          try {
            const result = await this.synthesize(text);
            const audioBase64 = this.float32ToBase64(result.audio);
            
            yield {
              type: 'audio.delta',
              event_id: `tts-${Date.now()}`,
              audio: audioBase64,
              sample_rate: result.sampleRate,
            } as unknown as ServerEvent;
          } catch (error) {
            yield {
              type: 'response.error',
              event_id: `error-${Date.now()}`,
              error: {
                type: 'synthesis_error',
                message: (error as Error).message,
              },
            } as unknown as ServerEvent;
          }
        }
        
        // Signal audio stream complete
        yield {
          type: 'audio.done',
          event_id: `tts-done-${Date.now()}`,
        } as unknown as ServerEvent;
      }
    }
  }

  /**
   * Convert Float32Array to base64 string
   */
  private float32ToBase64(audio: Float32Array): string {
    const buffer = new ArrayBuffer(audio.length * 4);
    const view = new Float32Array(buffer);
    view.set(audio);
    
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
}
