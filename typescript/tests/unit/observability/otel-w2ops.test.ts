/**
 * OpenTelemetry for the agent loop (2026-09-26, gap-closure plan item 2.4,
 * lane w2-ops), against the shared fixture
 * `python/tests/fixtures/w2ops/otel.json`, which the Python suite reads too
 * (`tests/test_otel_w2ops.py`): the switch, the span names and attributes,
 * the metrics, and the rule that no message text or tool argument reaches
 * an attribute (S-227).
 *
 * The API is handed in through the module's import seam as an IN-MEMORY
 * recorder, so the scenario runs wherever these tests run. The last test
 * uses the real `@opentelemetry/api` when it is installed here, and is
 * skipped with that reason when it is not: the SDK declares no dependency
 * on it.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, tool } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';
import {
  ObservabilityConfigError,
  OTEL_ENV_VAR,
  loadOtelApi,
  otelEnabled,
  parseObservability,
  setOtelImportsForTests,
  startAgentRun,
  type OtelApi,
} from '../../../src/observability/otel';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/w2ops/otel.json'), 'utf8')) as {
  config: { file_key: string; keys: string[]; env_var: string; on_values: string[]; invalid: string; not_bool: string };
  instrumentation: { tracer: string; meter: string };
  spans: Record<string, { name: string; attributes: Record<string, string> }>;
  metrics: Record<string, { kind: string; unit: string }>;
  scenario: {
    agent: string;
    message: string;
    tool: string;
    tool_args: Record<string, string>;
    tool_result: string;
    model: string;
    provider: string;
    calls: Array<{ input_tokens: number; output_tokens: number }>;
    expected_spans: Array<{ name: string; attributes: Record<string, string | number> }>;
    expected_token_points: Array<{ 'gen_ai.token.type': string; value: number }>;
    payment: { credits: number; lock_id: string; expected_counter: number };
  };
};

// ---------------------------------------------------------------------------
// An in-memory OpenTelemetry API: what `observability/otel.ts` calls, recorded
// ---------------------------------------------------------------------------

interface RecordedSpan {
  name: string;
  kind: number;
  attributes: Record<string, string | number | boolean>;
  parent: RecordedSpan | undefined;
  status: number | undefined;
  start: number;
  end: number | undefined;
}
interface Point {
  instrument: string;
  value: number;
  attributes: Record<string, string | number | boolean>;
}

function recorder(): { api: OtelApi; spans: RecordedSpan[]; points: Point[]; instruments: Array<{ name: string; kind: string; unit?: string }> } {
  const spans: RecordedSpan[] = [];
  const points: Point[] = [];
  const instruments: Array<{ name: string; kind: string; unit?: string }> = [];
  const api: OtelApi = {
    SpanKind: { INTERNAL: 0, CLIENT: 2 },
    SpanStatusCode: { OK: 1, ERROR: 2 },
    context: { active: () => ({}) },
    trace: {
      getTracer: () => ({
        startSpan(name, options, ctx) {
          const span: RecordedSpan = {
            name,
            kind: options?.kind ?? 0,
            attributes: { ...(options?.attributes ?? {}) },
            parent: (ctx as { span?: RecordedSpan } | undefined)?.span,
            status: undefined,
            start: options?.startTime ?? Date.now(),
            end: undefined,
          };
          spans.push(span);
          return {
            setAttribute(key, value) {
              span.attributes[key] = value;
            },
            setStatus(status) {
              span.status = status.code;
            },
            end(endTime) {
              span.end = endTime ?? Date.now();
            },
          };
        },
      }),
      setSpan: (_ctx, span) => ({ span: spans.find((s) => (s as unknown) === (span as unknown)) ?? spans[spans.length - 1] }),
    },
    metrics: {
      getMeter: () => ({
        createHistogram(name, options) {
          instruments.push({ name, kind: 'histogram', unit: options?.unit });
          return { record: (value, attributes) => points.push({ instrument: name, value, attributes: { ...(attributes ?? {}) } }) };
        },
        createCounter(name, options) {
          instruments.push({ name, kind: 'counter', unit: options?.unit });
          return { add: (value, attributes) => points.push({ instrument: name, value, attributes: { ...(attributes ?? {}) } }) };
        },
      }),
    },
  };
  // `setSpan` above needs the recorded span, not the handle: hand the
  // handle a back-reference instead, so the parent link is exact.
  const tracer = api.trace.getTracer('x');
  api.trace.getTracer = () => ({
    startSpan(name, options, ctx) {
      const handle = tracer.startSpan(name, options, ctx) as { setAttribute: unknown; setStatus: unknown; end: unknown; _span?: RecordedSpan };
      handle._span = spans[spans.length - 1];
      return handle as unknown as ReturnType<OtelApi['trace']['getTracer']> extends { startSpan(...a: never[]): infer S } ? S : never;
    },
  });
  api.trace.setSpan = (_ctx, span) => ({ span: (span as { _span?: RecordedSpan })._span });
  return { api, spans, points, instruments };
}

// ---------------------------------------------------------------------------
// The scenario's stub model and tool
// ---------------------------------------------------------------------------

const S = FIXTURE.scenario;

class StubModel extends Skill {
  calls = 0;
  @handoff({ name: 'stub-model', priority: 10 })
  async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
    this.calls += 1;
    const usage = S.calls[Math.min(this.calls, S.calls.length) - 1];
    context.set('_llm_capabilities', { model: S.model, provider: S.provider });
    const responseId = generateEventId();
    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
    if (this.calls === 1) {
      const call = { id: 'call-1', name: S.tool, arguments: JSON.stringify(S.tool_args) };
      yield { type: 'response.delta', event_id: generateEventId(), response_id: responseId, delta: { type: 'tool_call', tool_call: call } } as ServerEvent;
      context.set('_llm_usage', { model: S.model, provider: S.provider, input_tokens: usage.input_tokens, output_tokens: usage.output_tokens, is_byok: true });
      yield createResponseDoneEvent(responseId, [{ type: 'tool_call', tool_call: call }], 'completed', {
        input_tokens: usage.input_tokens, output_tokens: usage.output_tokens, total_tokens: usage.input_tokens + usage.output_tokens,
      });
      return;
    }
    yield { type: 'response.delta', event_id: generateEventId(), response_id: responseId, delta: { type: 'text', text: 'done' } } as ServerEvent;
    context.set('_llm_usage', { model: S.model, provider: S.provider, input_tokens: usage.input_tokens, output_tokens: usage.output_tokens, is_byok: true });
    yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'done' }], 'completed', {
      input_tokens: usage.input_tokens, output_tokens: usage.output_tokens, total_tokens: usage.input_tokens + usage.output_tokens,
    });
  }
}

class StubTools extends Skill {
  @tool({ name: 'echo_tool', description: 'Echoes' })
  async echo(_params: Record<string, unknown>, _context: Context): Promise<string> {
    return S.tool_result;
  }
}

async function runScenario(observability: { otel?: boolean } | undefined): Promise<ReturnType<typeof recorder>> {
  const rec = recorder();
  setOtelImportsForTests({ api: async () => rec.api });
  const agent = new BaseAgent({ name: S.agent, skills: [new StubModel(), new StubTools()], ...(observability ? { observability } : {}) });
  await agent.initialize();
  await agent.run([{ role: 'user', content: S.message }]);
  return rec;
}

const savedEnv = process.env[OTEL_ENV_VAR];
beforeEach(() => {
  delete process.env[OTEL_ENV_VAR];
});
afterEach(() => {
  setOtelImportsForTests();
  if (savedEnv === undefined) delete process.env[OTEL_ENV_VAR];
  else process.env[OTEL_ENV_VAR] = savedEnv;
});

describe('the switch (fixture config)', () => {
  it('reads the agent file block, its short form, and refuses what it does not know', () => {
    expect(parseObservability({ otel: true })).toEqual({ otel: true });
    expect(parseObservability({ otel: false })).toEqual({ otel: false });
    expect(parseObservability(true)).toEqual({ otel: true });
    expect(parseObservability(undefined)).toEqual({});
    expect(parseObservability({})).toEqual({});
    expect(() => parseObservability({ otle: true })).toThrow(new ObservabilityConfigError(FIXTURE.config.invalid.replace('{key}', 'otle')));
    expect(() => parseObservability({ otel: 'yes' })).toThrow(new ObservabilityConfigError(FIXTURE.config.not_bool));
    expect(FIXTURE.config.keys).toEqual(['otel']);
  });

  it('the environment switches it on, and the file wins when it says something', () => {
    expect(FIXTURE.config.env_var).toBe(OTEL_ENV_VAR);
    for (const value of FIXTURE.config.on_values) expect(otelEnabled(undefined, { [OTEL_ENV_VAR]: value.toUpperCase() })).toBe(true);
    expect(otelEnabled(undefined, { [OTEL_ENV_VAR]: '0' })).toBe(false);
    expect(otelEnabled(undefined, {})).toBe(false);
    expect(otelEnabled({ otel: false }, { [OTEL_ENV_VAR]: '1' })).toBe(false);
    expect(otelEnabled({ otel: true }, {})).toBe(true);
  });
});

describe('the spans of one run (fixture scenario)', () => {
  it('records the run, its model calls and its tool call, with the fixture names and attributes, and nothing from the messages', async () => {
    const rec = await runScenario({ otel: true });
    const ordered = [...rec.spans].sort((a, b) => a.start - b.start);
    expect(ordered.map((s) => s.name)).toEqual(S.expected_spans.map((s) => s.name));
    for (const [i, expected] of S.expected_spans.entries()) {
      for (const [key, value] of Object.entries(expected.attributes)) expect(ordered[i].attributes[key], `${expected.name} ${key}`).toBe(value);
      expect(ordered[i].end, `${expected.name} ended`).toBeDefined();
    }
    // The run is the parent of every other span.
    const run = ordered[0];
    expect(run.parent).toBeUndefined();
    for (const child of ordered.slice(1)) expect(child.parent).toBe(run);
    // S-227: no message, no tool arguments, no tool result in any attribute.
    const sentinels = [S.message, S.tool_args.text, S.tool_result];
    for (const span of rec.spans) {
      for (const value of Object.values(span.attributes)) {
        for (const sentinel of sentinels) expect(String(value)).not.toContain(sentinel);
      }
    }
    // The fixture's attribute keys are the ones the spans carry.
    expect(Object.keys(FIXTURE.spans.model_call.attributes)).toEqual(['gen_ai.operation.name', 'gen_ai.provider.name', 'gen_ai.request.model']);
    expect(Object.keys(FIXTURE.spans.tool_call.attributes)).toEqual(['gen_ai.operation.name', 'gen_ai.tool.name', 'gen_ai.tool.type']);
  });

  it('records the token usage and duration metrics with the fixture instruments', async () => {
    const rec = await runScenario({ otel: true });
    const tokenPoints = rec.points.filter((p) => p.instrument === 'gen_ai.client.token.usage');
    expect(tokenPoints.map((p) => ({ 'gen_ai.token.type': p.attributes['gen_ai.token.type'], value: p.value }))).toEqual(S.expected_token_points);
    for (const p of tokenPoints) {
      expect(p.attributes['gen_ai.operation.name']).toBe('chat');
      expect(p.attributes['gen_ai.provider.name']).toBe(S.provider);
      expect(p.attributes['gen_ai.request.model']).toBe(S.model);
    }
    const durations = rec.points.filter((p) => p.instrument === 'gen_ai.client.operation.duration');
    expect(durations.map((p) => p.attributes['gen_ai.operation.name'])).toEqual(['chat', 'execute_tool', 'chat']);
    for (const [name, spec] of Object.entries(FIXTURE.metrics)) {
      const made = rec.instruments.find((i) => i.name === name);
      expect(made, name).toBeDefined();
      expect(made!.kind).toBe(spec.kind);
      expect(made!.unit).toBe(spec.unit);
    }
  });

  it('records a payment settle under the run, in credits, and counts it', async () => {
    const rec = recorder();
    setOtelImportsForTests({ api: async () => rec.api });
    const run = await startAgentRun({ otel: true }, { agentName: S.agent });
    expect(run.active).toBe(true);
    run.paymentSettle({ credits: S.payment.credits, lockId: S.payment.lock_id });
    run.end();
    const settle = rec.spans.find((s) => s.name === FIXTURE.spans.payment_settle.name);
    expect(settle).toBeDefined();
    expect(settle!.attributes['gen_ai.agent.name']).toBe(S.agent);
    expect(settle!.attributes['webagents.payment.credits']).toBe(S.payment.credits);
    expect(settle!.attributes['webagents.payment.lock_id']).toBe(S.payment.lock_id);
    expect(settle!.parent?.name).toBe(`invoke_agent ${S.agent}`);
    const counted = rec.points.filter((p) => p.instrument === 'webagents.payment.credits');
    expect(counted).toEqual([{ instrument: 'webagents.payment.credits', value: S.payment.expected_counter, attributes: { 'gen_ai.agent.name': S.agent } }]);
  });

  it('records nothing when the switch is off, and the file wins over the environment', async () => {
    process.env[OTEL_ENV_VAR] = '1';
    expect((await runScenario({ otel: false })).spans).toEqual([]);
    expect((await runScenario(undefined)).spans.length).toBeGreaterThan(0);
    delete process.env[OTEL_ENV_VAR];
    expect((await runScenario(undefined)).spans).toEqual([]);
  });

  it('is a no-op when the package cannot be imported', async () => {
    setOtelImportsForTests({ api: async () => { throw new Error("Cannot find package '@opentelemetry/api'"); } });
    expect(await loadOtelApi()).toBeNull();
    const run = await startAgentRun({ otel: true }, { agentName: S.agent });
    expect(run.active).toBe(false);
    run.modelCall({ provider: 'x', requestModel: 'y', startedAt: Date.now() });
    run.end();
  });
});

describe('the real @opentelemetry/api', () => {
  let real: unknown;
  let reason = '';
  beforeEach(async () => {
    try {
      real = await import(/* @vite-ignore */ '@opentelemetry/api' as string);
    } catch (error) {
      real = undefined;
      reason = (error as Error).message.split('\n')[0];
    }
  });

  it('records the run through it when it is installed here, and is skipped with the reason when it is not', async () => {
    if (!real) {
      console.log(`skipped: @opentelemetry/api is not installed in this environment (${reason})`);
      return;
    }
    // The API alone has no exporter; a provider made of the API's own
    // no-op tracer still answers spans, and that is enough to prove the
    // load path works against the real package.
    setOtelImportsForTests();
    const api = await loadOtelApi();
    expect(api).not.toBeNull();
    const run = await startAgentRun({ otel: true }, { agentName: S.agent });
    expect(run.active).toBe(true);
    run.toolCall({ name: S.tool, startedAt: Date.now() });
    run.end();
  });
});
