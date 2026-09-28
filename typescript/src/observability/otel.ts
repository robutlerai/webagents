/**
 * OpenTelemetry for the agent loop (2026-09-26, gap-closure plan item 2.4).
 *
 * Spans and metrics that follow the OpenTelemetry GenAI semantic conventions
 * (`gen_ai.*` attributes): the agent run (`invoke_agent <agent>`), every model
 * call (`chat <model>`, with the provider and the tokens), every tool call
 * (`execute_tool <tool>`) and every payment settle (`settle_payment`, the
 * amount in credits). The same names and attributes as the Python SDK
 * (`webagents/observability/otel.py`), pinned by the shared fixture
 * `python/tests/fixtures/w2ops/otel.json`.
 *
 * NO NEW DEPENDENCY. `@opentelemetry/api` is used only when it is already
 * importable, and imported dynamically (the `ensureMCP()` pattern of
 * `skills/mcp/skill.ts`): the portal typechecks this source against a
 * node_modules that does not carry the package, and a static import would
 * fail that gate. Without the package, or with the switch off, every call
 * here is a no-op that costs one property read.
 *
 * SWITCHED ON by `observability: {otel: true}` in the agent file, or by
 * WEBAGENTS_OTEL=1 in the environment (the file wins when it says something).
 * Where the spans go is the host's business: whoever registers a
 * TracerProvider with the API (an SDK, an exporter, a collector agent) gets
 * them; nothing is exported by default.
 *
 * NEVER MESSAGE TEXT, TOOL ARGUMENTS OR TOOL RESULTS in an attribute (S-227
 * in the portal's security log, the rule the trace lines already follow):
 * a span is a record that leaves the process. Names, models, token counts,
 * durations and outcomes only.
 */

/** The agent file's `observability:` block, parsed. */
export interface ObservabilityConfig {
  /** Emit OpenTelemetry spans and metrics. */
  otel?: boolean;
}

export const OBSERVABILITY_KEY = 'observability';
export const OTEL_ENV_VAR = 'WEBAGENTS_OTEL';
const ON_VALUES = new Set(['1', 'true', 'on', 'yes']);

/** A block that cannot be used as written, with the fixture's sentence. */
export class ObservabilityConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ObservabilityConfigError';
  }
}

/**
 * The `observability:` block as an agent file writes it: `{otel: true}`, or
 * a bare boolean as the short form. An unknown key or a value that is not a
 * boolean is refused with the fixture's sentence, as the Python schema
 * refuses it, so a file means one thing to both SDKs.
 */
export function parseObservability(value: unknown): ObservabilityConfig {
  if (value === undefined || value === null) return {};
  if (typeof value === 'boolean') return { otel: value };
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new ObservabilityConfigError('observability: otel must be true or false');
  }
  const block = value as Record<string, unknown>;
  for (const key of Object.keys(block)) {
    if (key !== 'otel') throw new ObservabilityConfigError(`observability: unknown key '${key}'. It takes otel.`);
  }
  if (block.otel === undefined || block.otel === null) return {};
  if (typeof block.otel !== 'boolean') throw new ObservabilityConfigError('observability: otel must be true or false');
  return { otel: block.otel };
}

/** Whether spans are wanted: the file's word, else the environment's. */
export function otelEnabled(
  config: ObservabilityConfig | undefined,
  env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {},
): boolean {
  if (config?.otel !== undefined) return config.otel;
  const raw = env[OTEL_ENV_VAR];
  return raw ? ON_VALUES.has(raw.trim().toLowerCase()) : false;
}

// ---------------------------------------------------------------------------
// The API, loaded when it is there
// ---------------------------------------------------------------------------

/** The part of `@opentelemetry/api` this module uses, as small local types. */
export interface OtelSpan {
  setAttribute(key: string, value: string | number | boolean): unknown;
  setStatus(status: { code: number; message?: string }): unknown;
  end(endTime?: number): void;
}
export interface OtelTracer {
  startSpan(
    name: string,
    options?: { kind?: number; startTime?: number; attributes?: Record<string, string | number | boolean> },
    context?: unknown,
  ): OtelSpan;
}
export interface OtelHistogram {
  record(value: number, attributes?: Record<string, string | number | boolean>): void;
}
export interface OtelCounter {
  add(value: number, attributes?: Record<string, string | number | boolean>): void;
}
export interface OtelMeter {
  createHistogram(name: string, options?: { unit?: string; description?: string }): OtelHistogram;
  createCounter(name: string, options?: { unit?: string; description?: string }): OtelCounter;
}
export interface OtelApi {
  trace: { getTracer(name: string, version?: string): OtelTracer; setSpan(context: unknown, span: OtelSpan): unknown };
  context: { active(): unknown };
  metrics: { getMeter(name: string, version?: string): OtelMeter };
  SpanKind: { INTERNAL: number; CLIENT: number };
  SpanStatusCode: { OK: number; ERROR: number };
}

/** How the API is imported; a test hands in one that answers an in-memory one, or fails. */
export interface OtelImports {
  api: () => Promise<unknown>;
}

// Dynamic, with a literal specifier (`@vite-ignore`, `as string`): see the
// file comment. A package that is not installed rejects here, and that is
// the no-op case, not an error.
const DEFAULT_IMPORTS: OtelImports = {
  api: () => import(/* @vite-ignore */ '@opentelemetry/api' as string),
};

export const INSTRUMENTATION_NAME = 'webagents';

interface Instruments {
  api: OtelApi;
  tracer: OtelTracer;
  tokenUsage: OtelHistogram;
  operationDuration: OtelHistogram;
  paymentCredits: OtelCounter;
}

let imports: OtelImports = DEFAULT_IMPORTS;
let loading: Promise<Instruments | null> | undefined;

function usable(api: unknown): api is OtelApi {
  const a = api as Partial<OtelApi> | null | undefined;
  return Boolean(
    a && a.trace && typeof a.trace.getTracer === 'function' && typeof a.trace.setSpan === 'function'
      && a.context && typeof a.context.active === 'function'
      && a.metrics && typeof a.metrics.getMeter === 'function'
      && a.SpanKind && a.SpanStatusCode,
  );
}

/** The API's instruments, once; `null` when the package cannot load. */
async function instruments(): Promise<Instruments | null> {
  if (!loading) {
    loading = (async () => {
      let api: unknown;
      try {
        api = await imports.api();
      } catch {
        return null;
      }
      // An ESM interop default wrapper, when the loader hands one over.
      const candidate = usable(api) ? api : usable((api as { default?: unknown })?.default) ? (api as { default: OtelApi }).default : null;
      if (!candidate) return null;
      const tracer = candidate.trace.getTracer(INSTRUMENTATION_NAME);
      const meter = candidate.metrics.getMeter(INSTRUMENTATION_NAME);
      return {
        api: candidate,
        tracer,
        tokenUsage: meter.createHistogram('gen_ai.client.token.usage', { unit: '{token}', description: 'Tokens used per model call' }),
        operationDuration: meter.createHistogram('gen_ai.client.operation.duration', { unit: 's', description: 'Duration of a model or tool call' }),
        paymentCredits: meter.createCounter('webagents.payment.credits', { unit: '{credit}', description: 'Credits settled by this agent' }),
      };
    })();
  }
  return loading;
}

/** The API, or `null` when `@opentelemetry/api` is not importable here. */
export async function loadOtelApi(): Promise<OtelApi | null> {
  return (await instruments())?.api ?? null;
}

/**
 * For tests: import the API some other way (an in-memory one), or restore the
 * real import. Forgets what was loaded, so the next call loads again.
 */
export function setOtelImportsForTests(override?: Partial<OtelImports>): void {
  imports = { ...DEFAULT_IMPORTS, ...override };
  loading = undefined;
}

// ---------------------------------------------------------------------------
// One agent run
// ---------------------------------------------------------------------------

export interface ModelCallInfo {
  provider: string;
  requestModel: string;
  responseModel?: string;
  inputTokens?: number;
  outputTokens?: number;
  /** Milliseconds since the epoch. */
  startedAt: number;
  endedAt?: number;
  /** The failure's kind (an error code or class name), when it failed. */
  error?: string;
}

export interface ToolCallInfo {
  name: string;
  callId?: string;
  startedAt: number;
  endedAt?: number;
  error?: string;
}

export interface PaymentSettleInfo {
  credits: number;
  lockId?: string;
  startedAt?: number;
  error?: string;
}

/** The run's span, and the child spans recorded under it. A no-op when off. */
export interface AgentRun {
  /** Whether anything is recorded (the switch is on and the API is here). */
  readonly active: boolean;
  modelCall(info: ModelCallInfo): void;
  toolCall(info: ToolCallInfo): void;
  paymentSettle(info: PaymentSettleInfo): void;
  end(error?: string): void;
}

/** The run handle when nothing is recorded. */
export const NOOP_AGENT_RUN: AgentRun = {
  active: false,
  modelCall() {},
  toolCall() {},
  paymentSettle() {},
  end() {},
};

type Attributes = Record<string, string | number | boolean>;

class RecordedRun implements AgentRun {
  readonly active = true;
  private readonly span: OtelSpan;
  private readonly parent: unknown;
  private ended = false;

  constructor(
    private readonly inst: Instruments,
    private readonly agentName: string,
    conversationId: string | undefined,
  ) {
    const attributes: Attributes = { 'gen_ai.operation.name': 'invoke_agent', 'gen_ai.agent.name': agentName };
    if (conversationId) attributes['gen_ai.conversation.id'] = conversationId;
    this.span = inst.tracer.startSpan(`invoke_agent ${agentName}`, { kind: inst.api.SpanKind.INTERNAL, startTime: Date.now(), attributes }, inst.api.context.active());
    this.parent = inst.api.trace.setSpan(inst.api.context.active(), this.span);
  }

  private child(name: string, kind: number, startedAt: number, attributes: Attributes, endedAt: number | undefined, error: string | undefined): void {
    const { api, tracer } = this.inst;
    const span = tracer.startSpan(name, { kind, startTime: startedAt, attributes }, this.parent);
    if (error) {
      span.setAttribute('error.type', error);
      span.setStatus({ code: api.SpanStatusCode.ERROR });
    }
    span.end(endedAt ?? Date.now());
  }

  modelCall(info: ModelCallInfo): void {
    const attributes: Attributes = {
      'gen_ai.operation.name': 'chat',
      'gen_ai.provider.name': info.provider,
      'gen_ai.request.model': info.requestModel,
    };
    if (info.responseModel) attributes['gen_ai.response.model'] = info.responseModel;
    if (info.inputTokens !== undefined) attributes['gen_ai.usage.input_tokens'] = info.inputTokens;
    if (info.outputTokens !== undefined) attributes['gen_ai.usage.output_tokens'] = info.outputTokens;
    const endedAt = info.endedAt ?? Date.now();
    this.child(`chat ${info.requestModel}`, this.inst.api.SpanKind.CLIENT, info.startedAt, attributes, endedAt, info.error);
    const metricAttributes: Attributes = {
      'gen_ai.operation.name': 'chat',
      'gen_ai.provider.name': info.provider,
      'gen_ai.request.model': info.requestModel,
    };
    if (info.inputTokens !== undefined) this.inst.tokenUsage.record(info.inputTokens, { ...metricAttributes, 'gen_ai.token.type': 'input' });
    if (info.outputTokens !== undefined) this.inst.tokenUsage.record(info.outputTokens, { ...metricAttributes, 'gen_ai.token.type': 'output' });
    this.inst.operationDuration.record(Math.max(0, endedAt - info.startedAt) / 1000, metricAttributes);
  }

  toolCall(info: ToolCallInfo): void {
    const attributes: Attributes = {
      'gen_ai.operation.name': 'execute_tool',
      'gen_ai.tool.name': info.name,
      'gen_ai.tool.type': 'function',
    };
    if (info.callId) attributes['gen_ai.tool.call.id'] = info.callId;
    const endedAt = info.endedAt ?? Date.now();
    this.child(`execute_tool ${info.name}`, this.inst.api.SpanKind.INTERNAL, info.startedAt, attributes, endedAt, info.error);
    this.inst.operationDuration.record(Math.max(0, endedAt - info.startedAt) / 1000, {
      'gen_ai.operation.name': 'execute_tool',
      'gen_ai.tool.name': info.name,
    });
  }

  paymentSettle(info: PaymentSettleInfo): void {
    const attributes: Attributes = { 'gen_ai.agent.name': this.agentName, 'webagents.payment.credits': info.credits };
    if (info.lockId) attributes['webagents.payment.lock_id'] = info.lockId;
    const startedAt = info.startedAt ?? Date.now();
    this.child('settle_payment', this.inst.api.SpanKind.CLIENT, startedAt, attributes, undefined, info.error);
    if (!info.error && info.credits > 0) this.inst.paymentCredits.add(info.credits, { 'gen_ai.agent.name': this.agentName });
  }

  end(error?: string): void {
    if (this.ended) return;
    this.ended = true;
    if (error) {
      this.span.setAttribute('error.type', error);
      this.span.setStatus({ code: this.inst.api.SpanStatusCode.ERROR });
    } else {
      this.span.setStatus({ code: this.inst.api.SpanStatusCode.OK });
    }
    this.span.end(Date.now());
  }
}

/**
 * Start the run's span, when the switch is on and the API is here; the no-op
 * handle otherwise. Never throws: a tracing failure must not fail a turn.
 */
export async function startAgentRun(
  config: ObservabilityConfig | undefined,
  info: { agentName: string; conversationId?: string },
): Promise<AgentRun> {
  if (!otelEnabled(config)) return NOOP_AGENT_RUN;
  try {
    const inst = await instruments();
    if (!inst) return NOOP_AGENT_RUN;
    return new RecordedRun(inst, info.agentName, info.conversationId);
  } catch {
    return NOOP_AGENT_RUN;
  }
}

/** The context key the run handle travels under, for skills that settle payments. */
export const OTEL_RUN_CONTEXT_KEY = '_otel_run';

/** A span's `error.type` for a thrown value: its code, else its class name. */
export function errorType(error: unknown): string {
  const err = error as { code?: unknown; name?: unknown } | null | undefined;
  if (err && typeof err.code === 'string' && err.code) return err.code;
  if (err && typeof err.name === 'string' && err.name) return err.name;
  return 'Error';
}
