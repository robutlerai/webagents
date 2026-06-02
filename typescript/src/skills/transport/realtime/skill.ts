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
  GEMINI_LIVE_INPUT_RATE,
} from '../../../adapters/google-live';
import {
  openOpenAIRealtimeSession,
  OPENAI_REALTIME_INPUT_RATE,
  type RealtimeUpstreamSession,
  type RealtimeUsage,
  type RealtimeHistoryItem,
} from '../../../adapters/openai-realtime';

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
  /**
   * Per-session media direction. Omitted ⇒ full duplex `{audioIn:true,
   * audioOut:true}`. `audioOut:false` makes the provider answer in TEXT (the
   * widget speaks it with on-device TTS); `audioIn:false` takes the widget's
   * on-device-STT text via `input.text` and the provider speaks.
   */
  modalities?: { audioIn: boolean; audioOut: boolean };
  /**
   * The agent's DECLARED tools the model may call over the voice session (the
   * model can only call these). Each call is executed via `onToolCall` and the
   * result fed back to the provider so the spoken reply uses it.
   */
  tools?: Array<{ name: string; description?: string; parameters?: unknown }>;
  /** Execute a declared tool by name → its (JSON-serializable) result. */
  onToolCall?: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  /** Fired when a realtime connection opens (for billing lock). */
  onSessionStart?: (sessionId: string) => void;
  /** Fired when a realtime connection closes (for billing settle). */
  onSessionEnd?: (sessionId: string) => void;
  /**
   * Fired whenever the upstream provider reports CUMULATIVE token usage, for
   * token-exact billing. The relay keeps the latest snapshot and settles on it.
   */
  onUsage?: (sessionId: string, usage: RealtimeUsage) => void;
  /**
   * Prior conversation turns to seed the provider session at connect, so the
   * model continues an existing chat. Forwarded verbatim to the adapter.
   */
  initialHistory?: RealtimeHistoryItem[];
  /**
   * Fired at the end of each realtime turn with the accumulated user + agent
   * transcripts (from input/output transcription), so the relay can persist
   * the turn to the chat. Either field may be empty.
   */
  onTurn?: (sessionId: string, turn: { userText: string; agentText: string }) => void;
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
  upstream?: RealtimeUpstreamSession;
  /** Whether a user activity (turn) is currently open toward the provider. */
  activityOpen: boolean;
  /** Accumulated user transcript for the current turn (input transcription). */
  turnUserText: string;
  /** Accumulated agent transcript for the current turn (output transcription). */
  turnAgentText: string;
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
  private modalities: { audioIn: boolean; audioOut: boolean };
  private agentTools?: Array<{ name: string; description?: string; parameters?: unknown }>;
  private onToolCall?: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  private onSessionStart?: (sessionId: string) => void;
  private onSessionEnd?: (sessionId: string) => void;
  private onUsage?: (sessionId: string, usage: RealtimeUsage) => void;
  private initialHistory?: RealtimeHistoryItem[];
  private onTurn?: (sessionId: string, turn: { userText: string; agentText: string }) => void;

  constructor(config: RealtimeTransportConfig = {}) {
    super({ ...config, name: config.name || 'realtime-transport' });
    this.inputFormat = config.inputFormat ?? 'pcm16';
    this.outputFormat = config.outputFormat ?? 'pcm16';
    this.voice = config.voice;
    this.turnDetection = config.turnDetection ?? { type: 'server_vad', threshold: 0.5, silence_duration_ms: 500 };
    this.maxSessionDuration = resolveMaxSessionDuration(config.maxSessionDuration);
    this.maxAudioBufferSize = config.maxAudioBufferSize ?? 10 * 1024 * 1024;
    this.providerConfig = config.provider;
    this.modalities = config.modalities ?? { audioIn: true, audioOut: true };
    this.agentTools = config.tools;
    this.onToolCall = config.onToolCall;
    this.onSessionStart = config.onSessionStart;
    this.onSessionEnd = config.onSessionEnd;
    this.onUsage = config.onUsage;
    this.initialHistory = config.initialHistory;
    this.onTurn = config.onTurn;
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
        modalities: this.modalities.audioOut ? ['text', 'audio'] : ['text'],
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
      turnUserText: '',
      turnAgentText: '',
    };
    this.sessions.set(sessionId, session);

    // The widget resamples its mic to the provider's input rate (Gemini 16kHz,
    // OpenAI 24kHz). Surface it on session.created so the widget doesn't have to
    // hard-code per-provider rates.
    const providerInputRate =
      this.providerConfig?.provider === 'openai'
        ? OPENAI_REALTIME_INPUT_RATE
        : this.providerConfig?.provider === 'gemini'
          ? GEMINI_LIVE_INPUT_RATE
          : undefined;

    ws.send(JSON.stringify({
      ...createBaseEvent('session.created'),
      type: 'session.created',
      uamp_version: '1.0',
      session: {
        id: sessionId,
        created_at: Math.floor(session.createdAt / 1000),
        config: session.config,
        ...(providerInputRate ? { input_sample_rate: providerInputRate } : {}),
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

      case 'input.text': {
        // Text-in modality (audioIn:false): the widget did its own STT and
        // sends the finished transcript. Forward it as a complete user turn;
        // the provider responds (audio or text per the session modalities).
        const text = (event as unknown as { text?: string }).text;
        if (text && session.upstream) {
          session.upstream.sendText(text);
          ws.send(JSON.stringify({
            ...createBaseEvent('response.created'),
            type: 'response.created',
            response_id: crypto.randomUUID(),
          }));
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

    // Provider-agnostic bridge callbacks — both adapters return the same
    // RealtimeUpstreamSession shape and emit RAW PCM16 base64 chunks.
    const onAudioChunk = (pcmBase64: string) => {
      session.isResponding = true;
      ws.send(JSON.stringify({
        ...createBaseEvent('response.audio.delta'),
        type: 'response.audio.delta',
        audio: pcmBase64,
      }));
    };
    // Agent transcript (output transcription in native-audio mode, or the
    // streamed text in text-out mode): forward each delta so the widget shows
    // an agent bubble — and in text-out mode speaks it with on-device TTS.
    // Accumulate the turn's text for persistence (onTurn).
    const onText = (text: string) => {
      if (!text) return;
      session.isResponding = true;
      session.turnAgentText += text;
      ws.send(JSON.stringify({
        ...createBaseEvent('response.text.delta'),
        type: 'response.text.delta',
        delta: text,
      }));
    };
    // User transcript (input transcription of the mic audio): forward each
    // delta as a distinct event so the widget shows the user's own words, and
    // accumulate it for persistence.
    const onUserText = (text: string) => {
      if (!text) return;
      session.turnUserText += text;
      ws.send(JSON.stringify({
        ...createBaseEvent('input.transcript.delta'),
        type: 'input.transcript.delta',
        delta: text,
      }));
    };
    const onTurnComplete = () => {
      session.isResponding = false;
      const userText = session.turnUserText.trim();
      const agentText = session.turnAgentText.trim();
      session.turnUserText = '';
      session.turnAgentText = '';
      if ((userText || agentText) && this.onTurn) {
        try {
          this.onTurn(session.id, { userText, agentText });
        } catch {
          /* persistence is best-effort — never break the audio bridge */
        }
      }
      ws.send(JSON.stringify({
        ...createBaseEvent('response.done'),
        type: 'response.done',
      }));
    };
    const onInterrupted = () => {
      ws.send(JSON.stringify({
        ...createBaseEvent('response.cancelled'),
        type: 'response.cancelled',
        response_id: 'current',
      }));
    };
    const onError = (err: Error) => {
      ws.send(JSON.stringify({
        ...createBaseEvent('session.error'),
        type: 'session.error',
        error: { code: 'upstream_error', message: err.message },
      }));
    };
    const onUsage = (usage: RealtimeUsage) => this.onUsage?.(session.id, usage);

    // Tool calling: the model requests a DECLARED tool → we run it via
    // `onToolCall` and hand the result back so the spoken reply uses it. The
    // model can only call tools we declared, so nothing is hallucinated.
    const onFunctionCall = (callId: string, name: string, argsJson: string) => {
      const reply = (out: unknown) => {
        try { session.upstream?.submitToolResult(callId, JSON.stringify(out ?? null)); } catch { /* session closed */ }
      };
      if (!this.onToolCall) { reply({ error: 'tool execution unavailable' }); return; }
      let args: Record<string, unknown> = {};
      try { args = JSON.parse(argsJson || '{}') as Record<string, unknown>; } catch { /* bad args → {} */ }
      Promise.resolve(this.onToolCall(name, args))
        .then((result) => reply(result))
        .catch((err) => reply({ error: err instanceof Error ? err.message : String(err) }));
    };
    const tools = this.agentTools && this.agentTools.length ? this.agentTools : undefined;

    const responseAudio = this.modalities.audioOut;
    if (cfg.provider === 'gemini') {
      session.upstream = openGeminiLiveSession({
        apiKey: cfg.apiKey,
        model: cfg.model,
        voiceId: cfg.voiceId,
        systemPrompt: cfg.systemPrompt,
        webSocketImpl: cfg.webSocketImpl,
        endpoint: cfg.endpoint,
        responseAudio,
        tools,
        initialHistory: this.initialHistory,
        onFunctionCall,
        onAudioChunk,
        onText,
        onUserText,
        onTurnComplete,
        onInterrupted,
        onError,
        onUsage,
      });
    } else if (cfg.provider === 'openai') {
      session.upstream = openOpenAIRealtimeSession({
        apiKey: cfg.apiKey,
        model: cfg.model,
        voiceId: cfg.voiceId,
        systemPrompt: cfg.systemPrompt,
        // RealtimeProviderConfig.webSocketImpl is typed against the global WS
        // (Gemini's transport); the OpenAI adapter takes the `ws` lib's class.
        webSocketImpl: cfg.webSocketImpl as never,
        endpoint: cfg.endpoint,
        responseAudio,
        tools,
        initialHistory: this.initialHistory,
        onFunctionCall,
        onAudioChunk,
        onText,
        onUserText,
        onTurnComplete,
        onInterrupted,
        onError,
        onUsage,
      });
    } else {
      ws.send(JSON.stringify({
        ...createBaseEvent('session.error'),
        type: 'session.error',
        error: { code: 'provider_not_supported', message: `voice provider "${cfg.provider}" not wired` },
      }));
    }
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
