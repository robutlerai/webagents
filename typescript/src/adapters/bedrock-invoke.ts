/**
 * Claude on Amazon Bedrock's InvokeModel API, with the Anthropic body.
 *
 * Bedrock serves some Claude models only through InvokeModel and Converse,
 * not through its Anthropic Messages endpoint (Sonnet 4.6, Opus 4.6, Opus 4.5
 * and Sonnet 4.5, per the AWS API compatibility matrix). InvokeModel takes the
 * Anthropic Messages body with the differences below, so the SDK's Anthropic
 * adapter still builds it and the SDK's Anthropic parser still parses it, and
 * everything the Anthropic API supports (client tools, `cache_control`,
 * thinking, cache usage counts, stop reasons) passes through unchanged.
 * Converse would lose those for no gain, which is why Claude uses this API
 * rather than Converse.
 *
 *   | Messages wire              | Invoke wire                                                  |
 *   | `{base}/v1/messages`       | `{runtime}/model/{id}/invoke-with-response-stream` or `/invoke` |
 *   | `x-api-key`                | `Authorization: Bearer <Bedrock API key>`                    |
 *   | `anthropic-version` header | `"anthropic_version": "bedrock-2023-05-31"` in the body       |
 *   | `anthropic-beta` header    | `"anthropic_beta": [...]` in the body                         |
 *   | `model`, `stream` in body  | removed: the path decides both                                |
 *
 * The stream is Bedrock's binary event stream (`./bedrock-eventstream.ts`):
 * each `chunk` frame's payload is `{"bytes":"<base64>"}`, and the base64
 * decodes to exactly one Anthropic stream event. `invokeStreamToAnthropicSSE`
 * re-emits each as `event: <type>\ndata: <json>\n\n`, so any consumer of
 * Anthropic SSE (this parser, an HTTP client the stream is passed through
 * to) reads it unchanged. The final event's `amazon-bedrock-invocationMetrics`
 * is removed, so downstream consumers see the same events Anthropic's own API
 * sends. A non-stream Invoke answer is already Anthropic message JSON.
 *
 * A refusal (`stop_reason: "refusal"`, HTTP 200) is an ANSWER here as on the
 * direct API; the Anthropic parser reports it as a blocked `finish` chunk.
 */
import type { LLMAdapter, AdapterRequestParams, AdapterRequest, AdapterChunk } from './types';
import { anthropicAdapter } from './anthropic';
import {
  bedrockEventStreamToSse,
  isEventStream,
  base64ToBytes,
  utf8,
  bedrockBaseUrl,
  BedrockStreamError,
  EventStreamError,
  type BedrockFrame,
  type BedrockSseEncoder,
} from './bedrock-eventstream';

export const BEDROCK_ANTHROPIC_VERSION = 'bedrock-2023-05-31';

export interface BedrockAnthropicInvokeOptions {
  /** Adapter name. Default `bedrock-anthropic-invoke`. */
  name?: string;
  /** The bedrock-runtime base URL; wins over `region`. */
  baseUrl?: string;
  /** Region for the default base URL. Default `us-east-1`. */
  region?: string;
  /**
   * The id in the URL path: an inference profile id (`global.anthropic.claude-sonnet-4-6`),
   * a base model id or an ARN. Default: the last segment of `params.model`.
   * Never a string from an untrusted request: it becomes a URL path segment
   * (encoded, but it still selects the model the caller's credential invokes).
   */
  modelId?: string | ((params: AdapterRequestParams) => string);
  /** Extra `anthropic_beta` values (Bedrock wants betas for some client tools the direct API made GA). */
  betas?: readonly string[] | ((params: AdapterRequestParams) => readonly string[]);
}

function resolveModelId(params: AdapterRequestParams, opt: BedrockAnthropicInvokeOptions['modelId']): string {
  if (typeof opt === 'function') return opt(params);
  if (typeof opt === 'string') return opt;
  return params.model.includes('/') ? params.model.split('/').pop()! : params.model;
}

/** The Invoke request for `params`: the Anthropic adapter's body, moved onto InvokeModel. */
export function buildBedrockAnthropicInvokeRequest(params: AdapterRequestParams, opts: BedrockAnthropicInvokeOptions = {}): AdapterRequest {
  const stream = params.stream !== false;
  const direct = anthropicAdapter.buildRequest(params);
  const body = JSON.parse(direct.body) as Record<string, unknown>;
  delete body.model;
  delete body.stream;
  const betas = new Set<string>(
    (direct.headers['anthropic-beta'] ?? '').split(',').map((s) => s.trim()).filter(Boolean),
  );
  const extra = typeof opts.betas === 'function' ? opts.betas(params) : (opts.betas ?? []);
  for (const b of extra) if (b) betas.add(b);
  const out: Record<string, unknown> = { anthropic_version: BEDROCK_ANTHROPIC_VERSION, ...body };
  if (betas.size > 0) out.anthropic_beta = [...betas];
  const base = bedrockBaseUrl(opts);
  const modelId = resolveModelId(params, opts.modelId);
  return {
    url: `${base}/model/${encodeURIComponent(modelId)}/${stream ? 'invoke-with-response-stream' : 'invoke'}`,
    headers: {
      'content-type': 'application/json',
      accept: stream ? 'application/vnd.amazon.eventstream' : 'application/json',
      'x-amzn-bedrock-accept': 'application/json',
      authorization: `Bearer ${params.apiKey}`,
    },
    body: JSON.stringify(out),
  };
}

/** The Anthropic event a `chunk` frame carries, or null for a frame that is not a chunk. */
function invokeEvent(frame: Extract<BedrockFrame, { kind: 'event' }>): Record<string, unknown> | null {
  if (frame.type !== 'chunk') return null;
  const b64 = frame.json.bytes;
  if (typeof b64 !== 'string' || !b64) return null;
  let event: unknown;
  try {
    event = JSON.parse(utf8(base64ToBytes(b64)));
  } catch {
    throw new EventStreamError('an Invoke chunk did not decode to an Anthropic event');
  }
  if (!event || typeof event !== 'object' || Array.isArray(event)) throw new EventStreamError('an Invoke chunk did not decode to an Anthropic event');
  return event as Record<string, unknown>;
}

/** The encoder from Invoke `chunk` frames to Anthropic SSE (see the header). */
export function createInvokeAnthropicSseEncoder(): BedrockSseEncoder {
  return {
    frame(frame) {
      const event = invokeEvent(frame);
      if (!event) return '';
      delete event['amazon-bedrock-invocationMetrics'];
      const type = typeof event.type === 'string' && /^[a-z_]+$/.test(event.type) ? event.type : 'message';
      return `event: ${type}\ndata: ${JSON.stringify(event)}\n\n`;
    },
    // `message_start` opens the answer (id, model, input usage),
    // `content_block_start` opens a block before any of its content, and
    // `ping` says nothing: none carries any of the answer, so priming holds
    // them and an exception straight after them still falls back
    // (./bedrock-eventstream.ts). Asked only while priming, so the second
    // decode of those small chunks costs nothing that matters.
    opening(frame) {
      const type = invokeEvent(frame)?.type;
      return type === 'message_start' || type === 'content_block_start' || type === 'ping';
    },
    // After the first chunk an exception is what Anthropic itself would send
    // mid-stream: an `error` event, which the Anthropic parser throws on and
    // an HTTP client already handles.
    exception(exceptionType, message) {
      return `event: error\ndata: ${JSON.stringify({ type: 'error', error: { type: exceptionType, message } })}\n\n`;
    },
    end() {
      return '';
    },
  };
}

/**
 * A 2xx Invoke event stream as Anthropic SSE, primed: an exception (or a
 * malformed or empty stream) before the first chunk resolves to a synthetic
 * non-2xx `Response` instead (`bedrockEventStreamToSse`).
 */
export function invokeStreamToAnthropicSSE(res: Response): Promise<Response> {
  return bedrockEventStreamToSse(res, createInvokeAnthropicSseEncoder());
}

/** Parse an Invoke stream (raw event stream or already transcoded) with the Anthropic parser. */
export async function* parseBedrockAnthropicInvokeStream(response: Response): AsyncGenerator<AdapterChunk> {
  let sse = response;
  if (isEventStream(response)) {
    sse = await invokeStreamToAnthropicSSE(response);
    if (!sse.ok) {
      const said = await sse.json().catch(() => ({})) as { type?: string; message?: string };
      throw new BedrockStreamError(said.type ?? 'exception', said.message ?? `HTTP ${sse.status}`);
    }
  }
  yield* anthropicAdapter.parseStream(sse);
}

/**
 * An SDK adapter for Claude on Bedrock InvokeModel: give it a region (or a
 * base URL) and the model id, pass the Bedrock API key as `apiKey`.
 */
export function createBedrockAnthropicInvokeAdapter(opts: BedrockAnthropicInvokeOptions = {}): LLMAdapter {
  return {
    name: opts.name ?? 'bedrock-anthropic-invoke',
    mediaSupport: anthropicAdapter.mediaSupport,
    buildRequest(params: AdapterRequestParams): AdapterRequest {
      return buildBedrockAnthropicInvokeRequest(params, opts);
    },
    parseStream(response: Response) {
      return parseBedrockAnthropicInvokeStream(response);
    },
  };
}
