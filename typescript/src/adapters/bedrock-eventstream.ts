/**
 * Amazon Bedrock's binary event stream (`application/vnd.amazon.eventstream`),
 * decoded and encoded without the AWS SDK.
 *
 * Two Bedrock runtime APIs stream this framing rather than SSE: InvokeModel
 * (`./bedrock-invoke.ts`), the way to reach the Claude models Bedrock does
 * not serve on its Anthropic Messages endpoint, and Converse
 * (`./bedrock-converse.ts`), the only chat API of Llama, DeepSeek R1, Nova,
 * Pixtral, Palmyra and the older Mistral models. With this module both can be
 * called with a Bedrock API key (a bearer token) and a region.
 *
 * Framing, verified against `@smithy/eventstream-codec` 4.2.14 (the codec the
 * AWS SDK uses) and `@aws-crypto/crc32` 5.2.0:
 *
 *   offset 0   uint32 BE  total length (prelude through message CRC)
 *   offset 4   uint32 BE  headers length
 *   offset 8   uint32 BE  prelude CRC32 over bytes 0..7
 *   offset 12  headers, then the payload
 *   end - 4    uint32 BE  message CRC32 over every byte before it
 *
 * Each header is `uint8 name length`, the UTF-8 name, `uint8 type`, the value.
 * Types: 0 true, 1 false, 2 int8, 3 int16, 4 int32, 5 int64, 6 bytes (uint16
 * length), 7 string (uint16 length), 8 timestamp (int64 ms), 9 UUID (16
 * bytes). Messages cap at 16 MiB and headers at 128 KiB.
 *
 * Errors. A CRC mismatch, a truncated message, an unknown header type or a
 * header that runs past its block ENDS THE STREAM WITH AN ERROR, never a
 * silent skip: a skipped frame could be the one carrying the token usage or
 * the stop reason. Unknown `:event-type` values are ignored by the encoders
 * (AWS adds new event members over time).
 *
 * Platform-neutral on purpose: the adapters barrel loads in browsers too, so
 * no `node:` import. The CRC is the plain IEEE CRC32 (polynomial 0xEDB88320)
 * from a local table; the tests pin it to `zlib.crc32`.
 *
 * Priming (`bedrockEventStreamToSse`). Bedrock answers HTTP 200 and may then
 * send an exception as the FIRST frame (throttling, a model that is not
 * ready). A caller that falls back to another provider on a non-2xx can only
 * do so before it has passed a byte on, so the transcoder reads the stream up
 * to its first frame with output before it returns: an exception there
 * becomes a synthetic non-2xx `Response` (status table in
 * `bedrockExceptionStatus`), and the caller treats it exactly like an HTTP
 * refusal. A frame that only opens the answer or a block of it (Converse
 * `messageStart` and `contentBlockStart`, Invoke's `message_start`,
 * `content_block_start` and `ping`; the encoder's `opening`) is held rather
 * than counted as output, so an exception straight after it still falls back.
 * After the first frame with output an exception ends the stream as an
 * error.
 */

/** A decoded header value, by wire type (see the header comment). */
export type EventStreamHeaderValue = boolean | number | bigint | string | Uint8Array | Date;

/** One decoded event-stream message. */
export interface EventStreamMessage {
  headers: Record<string, EventStreamHeaderValue>;
  payload: Uint8Array;
}

/** The stream is malformed: the caller must treat it as ended with an error. */
export class EventStreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EventStreamError';
  }
}

/**
 * An exception Bedrock sent INSIDE a 2xx stream after its first output
 * frame (`:message-type: exception`, or an unmodelled `error` frame).
 */
export class BedrockStreamError extends Error {
  constructor(readonly exceptionType: string, message: string) {
    super(`Bedrock stream error: ${exceptionType}: ${message}`);
    this.name = 'BedrockStreamError';
  }
}

/**
 * A request a Bedrock adapter will not send as built (a tool the model cannot
 * take on that API, an unpaired tool call, a name outside AWS's pattern...).
 * Thrown by `buildRequest` so the caller can serve the call some other way
 * instead of spending a refused round trip; `reason` is short and stable.
 */
export class BedrockRequestUnsupported extends Error {
  constructor(readonly reason: string) {
    super(`Bedrock cannot serve this request: ${reason}`);
    this.name = 'BedrockRequestUnsupported';
  }
}

export const EVENT_STREAM_MAX_MESSAGE_BYTES = 16 * 1024 * 1024;
export const EVENT_STREAM_MAX_HEADER_BYTES = 128 * 1024;

const PRELUDE = 8;
const CRC = 4;
const MIN_MESSAGE = PRELUDE + 2 * CRC;

const utf8Decoder = new TextDecoder();
const utf8Encoder = new TextEncoder();

let crcTable: Uint32Array | null = null;
function table(): Uint32Array {
  if (crcTable) return crcTable;
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  crcTable = t;
  return t;
}

/** CRC32 (IEEE), the checksum both event-stream CRCs use. Same value as `zlib.crc32`. */
export function crc32(bytes: Uint8Array): number {
  const t = table();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = t[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  if (a.length === 0) return b;
  const out = new Uint8Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

/**
 * Incremental decoder: push network chunks in any split, get whole messages
 * out. Throws `EventStreamError` on anything malformed; after a throw the
 * decoder is spent.
 */
export class EventStreamDecoder {
  private buf: Uint8Array = new Uint8Array(0);

  /** Bytes held for a message not yet complete. */
  get buffered(): number {
    return this.buf.length;
  }

  push(chunk: Uint8Array): EventStreamMessage[] {
    this.buf = concat(this.buf, chunk);
    const out: EventStreamMessage[] = [];
    while (this.buf.length >= PRELUDE + CRC) {
      const view = new DataView(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength);
      const total = view.getUint32(0, false);
      const headersLength = view.getUint32(4, false);
      if (crc32(this.buf.subarray(0, PRELUDE)) !== view.getUint32(PRELUDE, false)) {
        throw new EventStreamError('prelude CRC mismatch');
      }
      if (total < MIN_MESSAGE || total > EVENT_STREAM_MAX_MESSAGE_BYTES) {
        throw new EventStreamError(`message length ${total} outside ${MIN_MESSAGE}..${EVENT_STREAM_MAX_MESSAGE_BYTES}`);
      }
      if (headersLength > EVENT_STREAM_MAX_HEADER_BYTES || headersLength > total - MIN_MESSAGE) {
        throw new EventStreamError(`headers length ${headersLength} does not fit a ${total}-byte message`);
      }
      if (this.buf.length < total) break;
      const message = this.buf.subarray(0, total);
      if (crc32(message.subarray(0, total - CRC)) !== view.getUint32(total - CRC, false)) {
        throw new EventStreamError('message CRC mismatch');
      }
      const headersStart = PRELUDE + CRC;
      out.push({
        headers: parseHeaders(message.subarray(headersStart, headersStart + headersLength)),
        payload: message.slice(headersStart + headersLength, total - CRC),
      });
      this.buf = this.buf.subarray(total);
    }
    return out;
  }

  /** The stream ended: anything still held is a truncated message. */
  end(): void {
    if (this.buf.length > 0) throw new EventStreamError(`stream ended inside a message (${this.buf.length} bytes held)`);
  }
}

function parseHeaders(b: Uint8Array): Record<string, EventStreamHeaderValue> {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const headers: Record<string, EventStreamHeaderValue> = {};
  let p = 0;
  const need = (n: number) => {
    if (p + n > b.length) throw new EventStreamError('header runs past the header block');
  };
  while (p < b.length) {
    need(1);
    const nameLength = view.getUint8(p++);
    need(nameLength);
    const name = utf8Decoder.decode(b.subarray(p, p + nameLength));
    p += nameLength;
    need(1);
    const type = view.getUint8(p++);
    switch (type) {
      case 0: headers[name] = true; break;
      case 1: headers[name] = false; break;
      case 2: need(1); headers[name] = view.getInt8(p); p += 1; break;
      case 3: need(2); headers[name] = view.getInt16(p, false); p += 2; break;
      case 4: need(4); headers[name] = view.getInt32(p, false); p += 4; break;
      case 5: need(8); headers[name] = view.getBigInt64(p, false); p += 8; break;
      case 6: {
        need(2);
        const len = view.getUint16(p, false);
        p += 2;
        need(len);
        headers[name] = b.slice(p, p + len);
        p += len;
        break;
      }
      case 7: {
        need(2);
        const len = view.getUint16(p, false);
        p += 2;
        need(len);
        headers[name] = utf8Decoder.decode(b.subarray(p, p + len));
        p += len;
        break;
      }
      case 8: need(8); headers[name] = new Date(Number(view.getBigInt64(p, false))); p += 8; break;
      case 9: need(16); headers[name] = b.slice(p, p + 16); p += 16; break;
      default: throw new EventStreamError(`unknown header type ${type}`);
    }
  }
  return headers;
}

/** A header to encode: the wire type (0-9, see the header comment) and its value. */
export interface EventStreamHeaderInput {
  name: string;
  type: number;
  value?: boolean | number | bigint | string | Uint8Array | Date;
}

/**
 * Encode one message. For tests, fixtures and anyone emulating Bedrock; the
 * adapters only decode.
 */
export function encodeEventStreamMessage(headers: EventStreamHeaderInput[], payload: Uint8Array): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const h of headers) {
    const name = utf8Encoder.encode(h.name);
    if (name.length > 255) throw new EventStreamError(`header name ${h.name} is too long`);
    let value: Uint8Array;
    switch (h.type) {
      case 0: case 1: value = new Uint8Array(0); break;
      case 2: value = new Uint8Array(1); new DataView(value.buffer).setInt8(0, Number(h.value)); break;
      case 3: value = new Uint8Array(2); new DataView(value.buffer).setInt16(0, Number(h.value), false); break;
      case 4: value = new Uint8Array(4); new DataView(value.buffer).setInt32(0, Number(h.value), false); break;
      case 5: value = new Uint8Array(8); new DataView(value.buffer).setBigInt64(0, BigInt(h.value as bigint | number), false); break;
      case 6: case 7: {
        const raw = h.type === 7 ? utf8Encoder.encode(String(h.value ?? '')) : (h.value as Uint8Array);
        value = new Uint8Array(2 + raw.length);
        new DataView(value.buffer).setUint16(0, raw.length, false);
        value.set(raw, 2);
        break;
      }
      case 8: {
        value = new Uint8Array(8);
        const ms = h.value instanceof Date ? h.value.getTime() : Number(h.value);
        new DataView(value.buffer).setBigInt64(0, BigInt(ms), false);
        break;
      }
      case 9: {
        const raw = h.value as Uint8Array;
        if (!(raw instanceof Uint8Array) || raw.length !== 16) throw new EventStreamError('a UUID header takes 16 bytes');
        value = raw;
        break;
      }
      default: throw new EventStreamError(`unknown header type ${h.type}`);
    }
    const head = new Uint8Array(1 + name.length + 1);
    head[0] = name.length;
    head.set(name, 1);
    head[1 + name.length] = h.type;
    parts.push(head, value);
  }
  const headersLength = parts.reduce((n, p) => n + p.length, 0);
  const total = PRELUDE + CRC + headersLength + payload.length + CRC;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, total, false);
  view.setUint32(4, headersLength, false);
  view.setUint32(PRELUDE, crc32(out.subarray(0, PRELUDE)), false);
  let p = PRELUDE + CRC;
  for (const part of parts) { out.set(part, p); p += part.length; }
  out.set(payload, p);
  view.setUint32(total - CRC, crc32(out.subarray(0, total - CRC)), false);
  return out;
}

/** An event frame the way Bedrock sends one: `:message-type: event`, the member name, a JSON payload. */
export function bedrockEventMessage(eventType: string, json: unknown): Uint8Array {
  return encodeEventStreamMessage([
    { name: ':event-type', type: 7, value: eventType },
    { name: ':content-type', type: 7, value: 'application/json' },
    { name: ':message-type', type: 7, value: 'event' },
  ], utf8Encoder.encode(JSON.stringify(json)));
}

/** An exception frame the way Bedrock sends one inside a 2xx stream. */
export function bedrockExceptionMessage(exceptionType: string, message: string): Uint8Array {
  return encodeEventStreamMessage([
    { name: ':exception-type', type: 7, value: exceptionType },
    { name: ':content-type', type: 7, value: 'application/json' },
    { name: ':message-type', type: 7, value: 'exception' },
  ], utf8Encoder.encode(JSON.stringify({ message })));
}

/** Decode a whole body into messages; the reader is cancelled if the consumer stops early or decoding fails. */
export async function* readEventStream(body: ReadableStream<Uint8Array>): AsyncGenerator<EventStreamMessage> {
  const reader = body.getReader();
  const decoder = new EventStreamDecoder();
  let finished = false;
  try {
    for (;;) {
      const r = await reader.read();
      if (r.done) { finished = true; break; }
      for (const m of decoder.push(r.value)) yield m;
    }
    decoder.end();
  } finally {
    if (finished) reader.releaseLock();
    else await reader.cancel().catch(() => {});
  }
}

/** One decoded frame: an event with its JSON payload, or the exception the stream ended with. */
export type BedrockFrame =
  | { kind: 'event'; type: string; json: Record<string, unknown> }
  | { kind: 'exception'; type: string; message: string };

/** Classify a message by `:message-type`. A payload that is not JSON is malformed. */
export function classifyFrame(m: EventStreamMessage): BedrockFrame {
  const messageType = m.headers[':message-type'];
  if (messageType === 'event') {
    const text = utf8Decoder.decode(m.payload);
    let json: unknown;
    try {
      json = text ? JSON.parse(text) : {};
    } catch {
      throw new EventStreamError(`event ${String(m.headers[':event-type'] ?? '')} carries a payload that is not JSON`);
    }
    return { kind: 'event', type: String(m.headers[':event-type'] ?? ''), json: (json && typeof json === 'object' ? json : {}) as Record<string, unknown> };
  }
  if (messageType === 'exception') {
    let message = utf8Decoder.decode(m.payload);
    try {
      const parsed = JSON.parse(message) as { message?: unknown; Message?: unknown };
      const said = parsed.message ?? parsed.Message;
      if (typeof said === 'string') message = said;
    } catch { /* not JSON: keep the raw text */ }
    return { kind: 'exception', type: String(m.headers[':exception-type'] ?? 'exception'), message };
  }
  return {
    kind: 'exception',
    type: String(m.headers[':error-code'] ?? 'error'),
    message: String(m.headers[':error-message'] ?? utf8Decoder.decode(m.payload)),
  };
}

/** True for a body in Bedrock's binary event-stream framing (by content type). */
export function isEventStream(res: Response): boolean {
  return (res.headers.get('content-type') ?? '').toLowerCase().startsWith('application/vnd.amazon.eventstream');
}

/**
 * Frame reader with prompt cancellation: `cancel()` cancels the network
 * reader at once, even while a read is pending (an async generator would
 * queue the cancel behind that read).
 */
export class BedrockFrameReader {
  private readonly reader: ReadableStreamDefaultReader<Uint8Array>;
  private readonly decoder = new EventStreamDecoder();
  private queue: EventStreamMessage[] = [];
  private finished = false;

  constructor(body: ReadableStream<Uint8Array>) {
    this.reader = body.getReader();
  }

  /** The next frame, or null at a clean end. Throws `EventStreamError` on a malformed stream (the reader is cancelled first). */
  async next(): Promise<BedrockFrame | null> {
    try {
      for (;;) {
        const m = this.queue.shift();
        if (m) return classifyFrame(m);
        if (this.finished) return null;
        const r = await this.reader.read();
        if (r.done) {
          this.finished = true;
          this.decoder.end();
          return null;
        }
        this.queue = this.decoder.push(r.value);
      }
    } catch (err) {
      await this.cancel(err);
      throw err;
    }
  }

  async cancel(reason?: unknown): Promise<void> {
    this.queue = [];
    if (this.finished) return;
    this.finished = true;
    await this.reader.cancel(reason).catch(() => {});
  }
}

/**
 * Status a synthetic refusal gets for an exception in the FIRST frame, chosen
 * for the fallback it triggers (a caller's retry cooldown typically keys on
 * the status: 429 short, other 4xx long, 5xx medium). AWS documents
 * `modelStreamErrorException` as 424, but 424 would read as a client error
 * and back off a transient model fault for as long as a bad request; 502
 * says what it is.
 */
export function bedrockExceptionStatus(exceptionType: string): number {
  const t = exceptionType.replace(/^.*#/, '').toLowerCase();
  switch (t) {
    case 'validationexception': return 400;
    case 'accessdeniedexception': return 403;
    case 'resourcenotfoundexception': return 404;
    case 'throttlingexception':
    case 'servicequotaexceededexception':
    case 'modelnotreadyexception': return 429;
    case 'modeltimeoutexception': return 504;
    case 'modelstreamerrorexception': return 502;
    case 'internalserverexception': return 500;
    case 'serviceunavailableexception': return 503;
    default: return 502;
  }
}

/** A synthetic refusal: what a caller sees when the stream failed before its first output. */
export function bedrockRefusalResponse(exceptionType: string, message: string, status = bedrockExceptionStatus(exceptionType)): Response {
  return new Response(JSON.stringify({ message, type: exceptionType }), {
    status,
    headers: { 'content-type': 'application/json', 'x-amzn-errortype': exceptionType },
  });
}

/** How a family's SSE is written from Bedrock frames (`./bedrock-converse.ts`, `./bedrock-invoke.ts`). */
export interface BedrockSseEncoder {
  /** SSE text for one event frame; '' for a frame with nothing to say (unknown members included). May throw `EventStreamError`. */
  frame(frame: Extract<BedrockFrame, { kind: 'event' }>): string;
  /**
   * True for a frame that OPENS the answer, or a block of it, without
   * carrying any of it (Converse `messageStart`, which says only `role:
   * assistant`, and `contentBlockStart`; Invoke's `message_start`,
   * `content_block_start` and `ping` chunks). Priming holds such a frame's text and keeps
   * reading, so an exception right after it is still a refusal the caller
   * can fall back on: nothing of the answer has been passed on yet. Optional;
   * omitted, every frame with text ends priming.
   */
  opening?(frame: Extract<BedrockFrame, { kind: 'event' }>): boolean;
  /** SSE text for an exception AFTER the first output, or null to end the stream with a `BedrockStreamError`. */
  exception(exceptionType: string, message: string): string | null;
  /** SSE text after the last frame of a clean end. */
  end(): string;
}

/**
 * Transcode a 2xx Bedrock event stream into the encoder's SSE, primed: the
 * stream is read up to the first frame with output before this resolves
 * (past any frame the encoder calls `opening`, whose text is held, not
 * sent), and an exception, a malformed frame or an empty stream before that
 * point gives a synthetic non-2xx refusal instead (see the header). The
 * returned body is pull-driven (backpressure reaches the network) and
 * cancelling it cancels the upstream read.
 */
export async function bedrockEventStreamToSse(res: Response, encoder: BedrockSseEncoder): Promise<Response> {
  if (!res.body) return bedrockRefusalResponse('emptyEventStream', 'the response carried no body');
  const frames = new BedrockFrameReader(res.body);
  let first = '';
  for (;;) {
    let frame: BedrockFrame | null;
    try {
      frame = await frames.next();
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') throw err;
      return bedrockRefusalResponse('eventStreamError', (err as Error)?.message ?? String(err));
    }
    if (!frame) return bedrockRefusalResponse('emptyEventStream', 'the event stream ended before any output');
    if (frame.kind === 'exception') {
      await frames.cancel();
      return bedrockRefusalResponse(frame.type, frame.message);
    }
    let text: string;
    let opening: boolean;
    try {
      text = encoder.frame(frame);
      opening = text !== '' && encoder.opening?.(frame) === true;
    } catch (err) {
      await frames.cancel(err);
      return bedrockRefusalResponse('eventStreamError', (err as Error)?.message ?? String(err));
    }
    first += text;
    if (text && !opening) break;
  }

  let pending: string | null = first;
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (pending !== null) {
        controller.enqueue(utf8Encoder.encode(pending));
        pending = null;
        return;
      }
      for (;;) {
        let frame: BedrockFrame | null;
        let text: string;
        try {
          frame = await frames.next();
          if (!frame) {
            const tail = encoder.end();
            if (tail) controller.enqueue(utf8Encoder.encode(tail));
            controller.close();
            return;
          }
          if (frame.kind === 'exception') {
            await frames.cancel();
            const said = encoder.exception(frame.type, frame.message);
            if (said === null) {
              controller.error(new BedrockStreamError(frame.type, frame.message));
            } else {
              controller.enqueue(utf8Encoder.encode(said));
              controller.close();
            }
            return;
          }
          text = encoder.frame(frame);
        } catch (err) {
          await frames.cancel(err);
          controller.error(err);
          return;
        }
        if (text) {
          controller.enqueue(utf8Encoder.encode(text));
          return;
        }
      }
    },
    async cancel(reason) {
      await frames.cancel(reason);
    },
  });
  const headers = new Headers({ 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
  const requestId = res.headers.get('x-amzn-requestid');
  if (requestId) headers.set('x-amzn-requestid', requestId);
  return new Response(body, { status: res.status, statusText: res.statusText, headers });
}

/** Base64 to bytes, platform-neutral (`atob` exists in browsers and in Node 16+). */
export function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Bytes to base64, the inverse of `base64ToBytes`. */
export function bytesToBase64(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

/** UTF-8 text of bytes. */
export function utf8(bytes: Uint8Array): string {
  return utf8Decoder.decode(bytes);
}

/** The runtime base URL for a region (`https://bedrock-runtime.<region>.amazonaws.com`). The region is not checked here. */
export function bedrockRuntimeUrl(region: string): string {
  return `https://bedrock-runtime.${region}.amazonaws.com`;
}

/** An AWS region name (`us-east-1`, `us-gov-west-1`, `ap-southeast-2`). */
const AWS_REGION_RE = /^[a-z]{2}(?:-[a-z]+)+-\d{1,2}$/;

/**
 * The base URL a Bedrock builder sends to: `baseUrl` as given (trailing
 * slashes dropped), else the runtime host of `region` (default `us-east-1`).
 * A region that is not a region name is refused (`BedrockRequestUnsupported`)
 * rather than spliced into the host, because the bearer key goes wherever
 * this URL points (a value such as `eu-west-1.evil.example/x?` would
 * otherwise send it to another host).
 */
export function bedrockBaseUrl(opts: { baseUrl?: string; region?: string }): string {
  if (opts.baseUrl) return opts.baseUrl.replace(/\/+$/, '');
  const region = opts.region ?? 'us-east-1';
  if (!AWS_REGION_RE.test(region)) throw new BedrockRequestUnsupported(`region ${region.slice(0, 40)} is not an AWS region name`);
  return bedrockRuntimeUrl(region);
}
