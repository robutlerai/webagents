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

import { Skill } from '../../../core/skill';
import { tool, hook } from '../../../core/decorators';
import type { Context, HookData } from '../../../core/types';
import type {
  AudioFormat,
  VoiceConfig,
  TurnDetectionConfig,
  SessionConfig,
} from '../../../uamp/types';
import {
  createBaseEvent,
  type SessionUpdateEvent,
  type BaseEvent,
} from '../../../uamp/events';
import {
  openGeminiLiveSession,
  type GeminiLiveSession,
} from '../../../adapters/google-live';

/**
 * Provider-side voice config. When present, each realtime connection opens
 * an upstream provider session (currently Gemini Live) and bridges audio
 * between the widget WS and the provider WS. When absent, the skill runs in
 * its legacy pass-through mode (echo/buffer only — used by older tests).
 */
export interface RealtimeProviderConfig {
  provider: 'gemini' | 'openai';
  /** Provider model, e.g. `gemini-3.1-flash-live-preview`. */
  model: string;
  /** Provider voice id, e.g. `Aoede`. */
  voiceId: string;
  systemPrompt: string;
  /** Provider API key — stays server-side, never propagated to the widget. */
  apiKey: string;
  /** Test seam: inject a WebSocket implementation for the provider WS. */
  webSocketImpl?: typeof WebSocket;
  /** Test seam: override the provider WS endpoint. */
  endpoint?: string;
}

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
  /**
   * Maximum session duration in ms. Defaults to `LIVE_VOICE_MAX_SESSION_MS`
   * env (Plan v3-03), falling back to 30 minutes.
   */
  maxSessionDuration?: number;
  /** Audio buffer max size in bytes (default: 10MB) */
  maxAudioBufferSize?: number;
  /** Upstream provider session config (Mode 2). */
  provider?: RealtimeProviderConfig;
  /** Fired when a realtime connection opens (for billing lock). */
  onSessionStart?: (sessionId: string) => void;
  /** Fired when a realtime connection closes (for billing settle). */
  onSessionEnd?: (sessionId: string) => void;
}

function resolveMaxSessionDuration(explicit?: number): number {
  if (explicit && explicit > 0) return explicit;
  const env =
    typeof process !== 'undefined'
      ? Number(process.env?.LIVE_VOICE_MAX_SESSION_MS)
      : NaN;
  if (Number.isFinite(env) && env > 0) return env;
  return 30 * 60 * 1000;
}

interface RealtimeSession {
  id: string;
  config: SessionConfig;
  audioBuffer: Uint8Array[];
  audioBufferSize: number;
  isListening: boolean;
  isResponding: boolean;
  createdAt: number;
  /** Upstream provider session, when provider config is present. */
  upstream?: GeminiLiveSession;
  /** Whether a user activity (turn) is currently open toward the provider. */
  activityOpen: boolean;
}

export class RealtimeTransportSkill extends Skill {
  private sessions = new Map<string, RealtimeSession>();
  private inputFormat: AudioFormat;
  private outputFormat: AudioFormat;
  private voice?: VoiceConfig;
  private turnDetection?: TurnDetectionConfig;
  private maxSessionDuration: number;
  private maxAudioBufferSize: number;
  private providerConfig?: RealtimeProviderConfig;
  private onSessionStart?: (sessionId: string) => void;
  private onSessionEnd?: (sessionId: string) => void;

  constructor(config: RealtimeTransportConfig = {}) {
    super({ ...config, name: config.name || 'realtime-transport' });
    this.inputFormat = config.inputFormat ?? 'pcm16';
    this.outputFormat = config.outputFormat ?? 'pcm16';
    this.voice = config.voice;
    this.turnDetection = config.turnDetection ?? { type: 'server_vad', threshold: 0.5, silence_duration_ms: 500 };
    this.maxSessionDuration = resolveMaxSessionDuration(config.maxSessionDuration);
    this.maxAudioBufferSize = config.maxAudioBufferSize ?? 10 * 1024 * 1024;
    this.providerConfig = config.provider;
    this.onSessionStart = config.onSessionStart;
    this.onSessionEnd = config.onSessionEnd;
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
      activityOpen: false,
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

    // Mode 2: open the upstream provider session and bridge audio. The
    // provider WS is held server-side; its api key never reaches the widget.
    if (this.providerConfig) {
      this.openUpstream(session, ws);
    }
    this.onSessionStart?.(sessionId);

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
      const s = this.sessions.get(sessionId);
      s?.upstream?.close();
      this.sessions.delete(sessionId);
      this.onSessionEnd?.(sessionId);
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
          if (session.upstream) {
            // Open a provider activity (turn) on the first chunk of an
            // utterance; subsequent chunks just stream through.
            if (!session.activityOpen) {
              session.upstream.sendActivityStart();
              session.activityOpen = true;
            }
            session.upstream.sendAudio(bytes);
          }
        }
        break;
      }

      case 'input.audio_committed': {
        session.isListening = false;
        if (session.upstream && session.activityOpen) {
          // End the user's turn — the provider now generates a response.
          session.upstream.sendActivityEnd();
          session.activityOpen = false;
        }
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
        // Barge-in: interrupt the in-flight provider turn.
        session.upstream?.interrupt();
        session.activityOpen = true;
        ws.send(JSON.stringify({
          ...createBaseEvent('response.cancelled'),
          type: 'response.cancelled',
          response_id: (event as unknown as { response_id?: string }).response_id ?? 'current',
        }));
        break;
      }

      case 'session.end': {
        session.upstream?.close();
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

  /**
   * Open the upstream provider session for a realtime connection and bridge
   * provider audio back to the widget as `response.audio.delta` UAMP events.
   * The provider WS lives entirely server-side.
   */
  private openUpstream(session: RealtimeSession, ws: WebSocket): void {
    const cfg = this.providerConfig;
    if (!cfg) return;
    if (cfg.provider !== 'gemini') {
      // Only Gemini Live is wired for the v3-03 vertical slice. Surface a
      // loud error rather than silently buffering audio that goes nowhere.
      ws.send(JSON.stringify({
        ...createBaseEvent('session.error'),
        type: 'session.error',
        error: { code: 'provider_not_supported', message: `voice provider "${cfg.provider}" not wired` },
      }));
      return;
    }

    session.upstream = openGeminiLiveSession({
      apiKey: cfg.apiKey,
      model: cfg.model,
      voiceId: cfg.voiceId,
      systemPrompt: cfg.systemPrompt,
      webSocketImpl: cfg.webSocketImpl,
      endpoint: cfg.endpoint,
      onAudioChunk: (wavBase64) => {
        session.isResponding = true;
        ws.send(JSON.stringify({
          ...createBaseEvent('response.audio.delta'),
          type: 'response.audio.delta',
          audio: wavBase64,
        }));
      },
      onTurnComplete: () => {
        session.isResponding = false;
        ws.send(JSON.stringify({
          ...createBaseEvent('response.done'),
          type: 'response.done',
        }));
      },
      onInterrupted: () => {
        ws.send(JSON.stringify({
          ...createBaseEvent('response.cancelled'),
          type: 'response.cancelled',
          response_id: 'current',
        }));
      },
      onError: (err) => {
        ws.send(JSON.stringify({
          ...createBaseEvent('session.error'),
          type: 'session.error',
          error: { code: 'upstream_error', message: err.message },
        }));
      },
    });
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
