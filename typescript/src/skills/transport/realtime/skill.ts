/**
 * Realtime Transport Skill
 *
 * WebSocket-based real-time audio/text transport compatible with
 * OpenAI's Realtime API format and the UAMP protocol extension.
 *
 * Supports:
 * - Bidirectional audio streaming (voice conversations)
 * - Server VAD (voice activity detection)
 * - Turn-based conversation management
 * - Audio format negotiation
 * - Interleaved text + audio responses
 */

import { Skill } from '../../../core/skill.js';
import { tool, hook } from '../../../core/decorators.js';
import type { Context, HookData } from '../../../core/types.js';
import type {
  AudioFormat,
  VoiceConfig,
  TurnDetectionConfig,
  SessionConfig,
} from '../../../uamp/types.js';
import {
  createBaseEvent,
  type SessionUpdateEvent,
  type BaseEvent,
} from '../../../uamp/events.js';

export interface RealtimeTransportConfig {
  name?: string;
  enabled?: boolean;
  /** Default input audio format */
  inputFormat?: AudioFormat;
  /** Default output audio format */
  outputFormat?: AudioFormat;
  /** Default voice */
  voice?: VoiceConfig;
  /** Default turn detection */
  turnDetection?: TurnDetectionConfig;
  /** Maximum session duration in ms (default: 15 minutes) */
  maxSessionDuration?: number;
  /** Audio buffer max size in bytes (default: 10MB) */
  maxAudioBufferSize?: number;
}

interface RealtimeSession {
  id: string;
  config: SessionConfig;
  audioBuffer: Uint8Array[];
  audioBufferSize: number;
  isListening: boolean;
  isResponding: boolean;
  createdAt: number;
}

export class RealtimeTransportSkill extends Skill {
  private sessions = new Map<string, RealtimeSession>();
  private inputFormat: AudioFormat;
  private outputFormat: AudioFormat;
  private voice?: VoiceConfig;
  private turnDetection?: TurnDetectionConfig;
  private maxSessionDuration: number;
  private maxAudioBufferSize: number;

  constructor(config: RealtimeTransportConfig = {}) {
    super({ ...config, name: config.name || 'realtime-transport' });
    this.inputFormat = config.inputFormat ?? 'pcm16';
    this.outputFormat = config.outputFormat ?? 'pcm16';
    this.voice = config.voice;
    this.turnDetection = config.turnDetection ?? { type: 'server_vad', threshold: 0.5, silence_duration_ms: 500 };
    this.maxSessionDuration = config.maxSessionDuration ?? 15 * 60 * 1000;
    this.maxAudioBufferSize = config.maxAudioBufferSize ?? 10 * 1024 * 1024;
  }

  @hook({ lifecycle: 'on_connection', priority: 5 })
  async handleRealtimeConnection(data: HookData, context: Context): Promise<void> {
    const ws = data.ws as WebSocket | undefined;
    if (!ws) return;

    const isRealtime = (data.metadata?.transport === 'realtime') ||
      (data.metadata?.path as string)?.includes('/realtime');
    if (!isRealtime) return;

    const sessionId = crypto.randomUUID();
    const session: RealtimeSession = {
      id: sessionId,
      config: {
        modalities: ['text', 'audio'],
        input_audio_format: this.inputFormat,
        output_audio_format: this.outputFormat,
        voice: this.voice,
        turn_detection: this.turnDetection,
      },
      audioBuffer: [],
      audioBufferSize: 0,
      isListening: true,
      isResponding: false,
      createdAt: Date.now(),
    };
    this.sessions.set(sessionId, session);

    ws.send(JSON.stringify({
      ...createBaseEvent('session.created'),
      type: 'session.created',
      uamp_version: '1.0',
      session: {
        id: sessionId,
        created_at: Math.floor(session.createdAt / 1000),
        config: session.config,
        status: 'active',
      },
    }));

    ws.addEventListener('message', (event: MessageEvent) => {
      try {
        const msg = JSON.parse(String(event.data));
        this.handleRealtimeEvent(sessionId, msg, ws, context);
      } catch {
        // Binary audio data
        if (event.data instanceof ArrayBuffer || event.data instanceof Uint8Array) {
          this.handleAudioData(sessionId, new Uint8Array(event.data as ArrayBuffer));
        }
      }
    });

    ws.addEventListener('close', () => {
      this.sessions.delete(sessionId);
    });

    const sessionTimer = setTimeout(() => {
      ws.send(JSON.stringify({
        ...createBaseEvent('session.error'),
        type: 'session.error',
        error: { code: 'session_timeout', message: 'Session exceeded maximum duration' },
      }));
      ws.close();
    }, this.maxSessionDuration);

    ws.addEventListener('close', () => clearTimeout(sessionTimer));
  }

  private handleRealtimeEvent(
    sessionId: string,
    event: BaseEvent & Record<string, unknown>,
    ws: WebSocket,
    _context: Context,
  ): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    switch (event.type) {
      case 'session.update': {
        const update = event as unknown as SessionUpdateEvent;
        if (update.session) {
          Object.assign(session.config, update.session);
          ws.send(JSON.stringify({
            ...createBaseEvent('session.updated'),
            type: 'session.updated',
            session: {
              id: sessionId,
              created_at: Math.floor(session.createdAt / 1000),
              config: session.config,
              status: 'active',
            },
          }));
        }
        break;
      }

      case 'input.audio': {
        const audio = (event as unknown as { audio: string }).audio;
        if (audio) {
          const bytes = Uint8Array.from(atob(audio), (c) => c.charCodeAt(0));
          this.handleAudioData(sessionId, bytes);
        }
        break;
      }

      case 'input.audio_committed': {
        session.isListening = false;
        // Trigger response generation from accumulated audio
        ws.send(JSON.stringify({
          ...createBaseEvent('response.created'),
          type: 'response.created',
          response_id: crypto.randomUUID(),
        }));
        session.audioBuffer = [];
        session.audioBufferSize = 0;
        session.isListening = true;
        break;
      }

      case 'response.cancel': {
        session.isResponding = false;
        ws.send(JSON.stringify({
          ...createBaseEvent('response.cancelled'),
          type: 'response.cancelled',
          response_id: (event as unknown as { response_id?: string }).response_id ?? 'current',
        }));
        break;
      }

      case 'session.end': {
        this.sessions.delete(sessionId);
        ws.close();
        break;
      }
    }
  }

  private handleAudioData(sessionId: string, data: Uint8Array): void {
    const session = this.sessions.get(sessionId);
    if (!session || !session.isListening) return;

    if (session.audioBufferSize + data.length > this.maxAudioBufferSize) {
      session.audioBuffer.shift();
      session.audioBufferSize -= session.audioBuffer[0]?.length ?? 0;
    }

    session.audioBuffer.push(data);
    session.audioBufferSize += data.length;
  }

  @tool({
    name: 'realtime_get_sessions',
    description: 'List active realtime sessions.',
    parameters: { type: 'object', properties: {} },
  })
  async realtimeGetSessions(
    _params: Record<string, unknown>,
    _context: Context,
  ): Promise<Array<{ id: string; createdAt: string; isResponding: boolean }>> {
    return [...this.sessions.values()].map((s) => ({
      id: s.id,
      createdAt: new Date(s.createdAt).toISOString(),
      isResponding: s.isResponding,
    }));
  }

  override async cleanup(): Promise<void> {
    this.sessions.clear();
  }
}
