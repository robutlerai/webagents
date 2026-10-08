/**
 * Bedrock's binary event stream (src/adapters/bedrock-eventstream.ts). The
 * decoder sits between AWS and every byte of output and usage on the
 * InvokeModel and Converse APIs, so the properties pinned here are the ones
 * a silent bug would corrupt: any network split decodes the same, every
 * corruption ENDS the stream with an error (never a skipped frame), the caps
 * hold before a byte is buffered, and an exception in the first frame becomes
 * a refusal a caller can fall back on. Frames are built in the test with the
 * module's own encoder, whose CRC is pinned to node's `zlib.crc32` (the
 * smithy codec is not a dependency of this package).
 */
import { describe, it, expect, vi } from 'vitest';
import { crc32 as zlibCrc32 } from 'node:zlib';
import {
  crc32,
  EventStreamDecoder,
  EventStreamError,
  BedrockStreamError,
  encodeEventStreamMessage,
  bedrockEventMessage,
  bedrockExceptionMessage,
  readEventStream,
  classifyFrame,
  isEventStream,
  bedrockExceptionStatus,
  bedrockEventStreamToSse,
  base64ToBytes,
  bytesToBase64,
  EVENT_STREAM_MAX_MESSAGE_BYTES,
  EVENT_STREAM_MAX_HEADER_BYTES,
  bedrockBaseUrl,
  BedrockRequestUnsupported,
  type BedrockSseEncoder,
} from '../../../src/adapters/bedrock-eventstream.js';

const enc = new TextEncoder();

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function streamOf(chunks: Uint8Array[], onCancel?: (reason: unknown) => void): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(chunks[i++]);
      else controller.close();
    },
    cancel(reason) { onCancel?.(reason); },
  });
}

function eventStreamResponse(chunks: Uint8Array[], onCancel?: (reason: unknown) => void): Response {
  return new Response(streamOf(chunks, onCancel), { status: 200, headers: { 'content-type': 'application/vnd.amazon.eventstream' } });
}

/** A prelude with a correct CRC for the given lengths (no body: for the cap checks). */
function prelude(total: number, headersLength: number): Uint8Array {
  const b = new Uint8Array(12);
  const v = new DataView(b.buffer);
  v.setUint32(0, total, false);
  v.setUint32(4, headersLength, false);
  v.setUint32(8, crc32(b.subarray(0, 8)), false);
  return b;
}

const ALL_TYPES = [
  { name: 't', type: 0 },
  { name: 'f', type: 1 },
  { name: 'i8', type: 2, value: -7 },
  { name: 'i16', type: 3, value: -1234 },
  { name: 'i32', type: 4, value: 2_000_000_000 },
  { name: 'i64', type: 5, value: BigInt('-9007199254740993') },
  { name: 'bytes', type: 6, value: new Uint8Array([0, 1, 254, 255]) },
  { name: ':event-type', type: 7, value: 'contentBlockDelta' },
  { name: 'ts', type: 8, value: new Date(1_791_000_000_000) },
  { name: 'uuid', type: 9, value: new Uint8Array(Array.from({ length: 16 }, (_, i) => i * 13)) },
];

describe('crc32', () => {
  it('is the IEEE CRC32: the check value, and zlib on arbitrary bytes', () => {
    expect(crc32(enc.encode('123456789'))).toBe(0xcbf43926);
    for (const n of [0, 1, 7, 64, 1000, 65_537]) {
      const bytes = new Uint8Array(n).map((_, i) => (i * 31 + n) & 0xff);
      expect(crc32(bytes)).toBe(zlibCrc32(bytes));
    }
  });
});

describe('EventStreamDecoder', () => {
  it('decodes every header type, 0 to 9, and the payload', () => {
    const payload = enc.encode('{"a":1}');
    const [m] = new EventStreamDecoder().push(encodeEventStreamMessage(ALL_TYPES, payload));
    expect(m.headers).toEqual({
      t: true,
      f: false,
      i8: -7,
      i16: -1234,
      i32: 2_000_000_000,
      i64: BigInt('-9007199254740993'),
      bytes: new Uint8Array([0, 1, 254, 255]),
      ':event-type': 'contentBlockDelta',
      ts: new Date(1_791_000_000_000),
      uuid: new Uint8Array(Array.from({ length: 16 }, (_, i) => i * 13)),
    });
    expect(new TextDecoder().decode(m.payload)).toBe('{"a":1}');
  });

  it('decodes the same messages whatever byte the network splits a stream at', () => {
    const frames = [
      bedrockEventMessage('messageStart', { role: 'assistant' }),
      bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { text: 'Hi there' }, p: 'abcdefgh' }),
      bedrockEventMessage('metadata', { usage: { inputTokens: 9, outputTokens: 2 } }),
    ];
    const whole = concat(frames);
    const reference = new EventStreamDecoder().push(whole).map(classifyFrame);
    expect(reference).toHaveLength(3);
    for (let cut = 0; cut <= whole.length; cut++) {
      const d = new EventStreamDecoder();
      const got = [...d.push(whole.subarray(0, cut)), ...d.push(whole.subarray(cut))].map(classifyFrame);
      expect(got, `split at ${cut}`).toEqual(reference);
      expect(() => d.end()).not.toThrow();
    }
    // And one byte at a time.
    const d = new EventStreamDecoder();
    const got = [];
    for (let i = 0; i < whole.length; i++) got.push(...d.push(whole.subarray(i, i + 1)));
    expect(got.map(classifyFrame)).toEqual(reference);
  });

  it('ends the stream with an error when either CRC is corrupt, wherever the corruption is', () => {
    const frame = bedrockEventMessage('contentBlockDelta', { delta: { text: 'x' } });
    for (let i = 0; i < frame.length; i++) {
      const bad = frame.slice();
      bad[i] ^= 0x01;
      // Bytes 0..11 are covered by the prelude CRC (8..11 ARE it); the rest by the message CRC.
      const expected = i < 12 ? /prelude CRC mismatch/ : /message CRC mismatch/;
      expect(() => new EventStreamDecoder().push(bad), `byte ${i}`).toThrow(expected);
    }
  });

  it('refuses a message over 16 MiB from its prelude alone, before buffering it', () => {
    const d = new EventStreamDecoder();
    expect(() => d.push(prelude(EVENT_STREAM_MAX_MESSAGE_BYTES + 1, 0))).toThrow(EventStreamError);
    expect(() => new EventStreamDecoder().push(prelude(15, 0))).toThrow(/outside/);
    // Exactly at the cap is allowed through the prelude check (and waits for its bytes).
    const ok = new EventStreamDecoder();
    expect(ok.push(prelude(EVENT_STREAM_MAX_MESSAGE_BYTES, 0))).toEqual([]);
    expect(ok.buffered).toBe(12);
  });

  it('refuses headers over 128 KiB, or longer than the message holds', () => {
    expect(() => new EventStreamDecoder().push(prelude(1_000_000, EVENT_STREAM_MAX_HEADER_BYTES + 1))).toThrow(/headers length/);
    expect(() => new EventStreamDecoder().push(prelude(100, 85))).toThrow(/headers length/);
    expect(new EventStreamDecoder().push(prelude(100, 84))).toEqual([]);
  });

  it('ends the stream with an error when it stops inside a message', () => {
    const frame = bedrockEventMessage('messageStop', { stopReason: 'end_turn' });
    for (const cut of [1, 11, 12, frame.length - 1]) {
      const d = new EventStreamDecoder();
      expect(d.push(frame.subarray(0, cut))).toEqual([]);
      expect(() => d.end(), `cut ${cut}`).toThrow(/stream ended inside a message/);
    }
  });

  it('refuses an unknown header type and a header that runs past its block', () => {
    // Hand-built header block: name "x", type 10.
    const unknown = buildRaw(new Uint8Array([1, 0x78, 10]), new Uint8Array(0));
    expect(() => new EventStreamDecoder().push(unknown)).toThrow(/unknown header type 10/);
    // A string header claiming 50 bytes in a 6-byte block.
    const overrun = buildRaw(new Uint8Array([1, 0x78, 7, 0, 50, 0x61]), new Uint8Array(0));
    expect(() => new EventStreamDecoder().push(overrun)).toThrow(/runs past/);
  });
});

/** A message with a hand-made header block and valid CRCs. */
function buildRaw(headers: Uint8Array, payload: Uint8Array): Uint8Array {
  const total = 12 + headers.length + payload.length + 4;
  const out = new Uint8Array(total);
  const v = new DataView(out.buffer);
  v.setUint32(0, total, false);
  v.setUint32(4, headers.length, false);
  v.setUint32(8, crc32(out.subarray(0, 8)), false);
  out.set(headers, 12);
  out.set(payload, 12 + headers.length);
  v.setUint32(total - 4, crc32(out.subarray(0, total - 4)), false);
  return out;
}

describe('classifyFrame', () => {
  const one = (bytes: Uint8Array) => classifyFrame(new EventStreamDecoder().push(bytes)[0]);

  it('reads an event, an exception and an unmodelled error frame', () => {
    expect(one(bedrockEventMessage('messageStart', { role: 'assistant' }))).toEqual({ kind: 'event', type: 'messageStart', json: { role: 'assistant' } });
    expect(one(bedrockExceptionMessage('throttlingException', 'Too many requests'))).toEqual({ kind: 'exception', type: 'throttlingException', message: 'Too many requests' });
    const error = encodeEventStreamMessage([
      { name: ':message-type', type: 7, value: 'error' },
      { name: ':error-code', type: 7, value: 'InternalFailure' },
      { name: ':error-message', type: 7, value: 'boom' },
    ], new Uint8Array(0));
    expect(one(error)).toEqual({ kind: 'exception', type: 'InternalFailure', message: 'boom' });
  });

  it('calls an event whose payload is not JSON malformed', () => {
    const bad = encodeEventStreamMessage([
      { name: ':message-type', type: 7, value: 'event' },
      { name: ':event-type', type: 7, value: 'chunk' },
    ], enc.encode('{not json'));
    expect(() => one(bad)).toThrow(EventStreamError);
  });
});

describe('readEventStream and isEventStream', () => {
  it('yields each message of a body and errors on a truncated one', async () => {
    const a = bedrockEventMessage('messageStart', { role: 'assistant' });
    const b = bedrockEventMessage('messageStop', { stopReason: 'end_turn' });
    const got = [];
    for await (const m of readEventStream(streamOf([a.subarray(0, 5), concat([a.subarray(5), b])]))) got.push(classifyFrame(m).type);
    expect(got).toEqual(['messageStart', 'messageStop']);
    await expect((async () => { for await (const _ of readEventStream(streamOf([b.subarray(0, 20)]))) { /* drain */ } })())
      .rejects.toThrow(/stream ended inside a message/);
  });

  it('knows the framing by its content type', () => {
    expect(isEventStream(new Response('', { headers: { 'content-type': 'application/vnd.amazon.eventstream' } }))).toBe(true);
    expect(isEventStream(new Response('', { headers: { 'content-type': 'text/event-stream' } }))).toBe(false);
  });

  it('round-trips base64', () => {
    const bytes = new Uint8Array(70_000).map((_, i) => (i * 7) & 0xff);
    expect(base64ToBytes(bytesToBase64(bytes))).toEqual(bytes);
    expect(new TextDecoder().decode(base64ToBytes(Buffer.from('{"type":"ping","x":"é"}').toString('base64')))).toBe('{"type":"ping","x":"é"}');
  });
});

describe('bedrockExceptionStatus (the priming status table)', () => {
  it('maps each exception to the status whose cooldown fits it', () => {
    expect(bedrockExceptionStatus('validationException')).toBe(400);
    expect(bedrockExceptionStatus('accessDeniedException')).toBe(403);
    expect(bedrockExceptionStatus('resourceNotFoundException')).toBe(404);
    expect(bedrockExceptionStatus('throttlingException')).toBe(429);
    expect(bedrockExceptionStatus('serviceQuotaExceededException')).toBe(429);
    expect(bedrockExceptionStatus('modelTimeoutException')).toBe(504);
    // AWS says 424; a 4xx would cool a transient model fault like a bad request.
    expect(bedrockExceptionStatus('modelStreamErrorException')).toBe(502);
    expect(bedrockExceptionStatus('internalServerException')).toBe(500);
    expect(bedrockExceptionStatus('serviceUnavailableException')).toBe(503);
    expect(bedrockExceptionStatus('somethingNew')).toBe(502);
    expect(bedrockExceptionStatus('com.amazonaws.bedrock#ThrottlingException')).toBe(429);
  });
});

/** An encoder that writes `data: <type>` per event and says what it is told about the end. */
function echoEncoder(onException: string | null = null): BedrockSseEncoder {
  return {
    frame: (f) => (f.type === 'skip' ? '' : `data: ${f.type}\n\n`),
    opening: (f) => f.type === 'open',
    exception: (type) => (onException === null ? null : `${onException}:${type}\n\n`),
    end: () => 'data: [END]\n\n',
  };
}

describe('bedrockEventStreamToSse (primed)', () => {
  it('turns an exception in the FIRST frame into a synthetic refusal with the table\'s status', async () => {
    for (const [type, status] of [['throttlingException', 429], ['validationException', 400], ['modelStreamErrorException', 502]] as const) {
      const cancel = vi.fn();
      const res = await bedrockEventStreamToSse(eventStreamResponse([bedrockExceptionMessage(type, 'nope')], cancel), echoEncoder());
      expect(res.status).toBe(status);
      expect(await res.json()).toEqual({ message: 'nope', type });
      expect(res.headers.get('x-amzn-errortype')).toBe(type);
    }
  });

  it('reads past frames with no output before deciding (an unknown member first is not an answer yet)', async () => {
    const res = await bedrockEventStreamToSse(eventStreamResponse([
      bedrockEventMessage('skip', {}),
      bedrockExceptionMessage('serviceUnavailableException', 'later'),
    ]), echoEncoder());
    expect(res.status).toBe(503);
  });

  it('holds an opening frame: an exception straight after it is still a refusal, and its text leads the stream otherwise', async () => {
    const cancel = vi.fn();
    const refused = await bedrockEventStreamToSse(eventStreamResponse([
      bedrockEventMessage('open', {}),
      bedrockExceptionMessage('throttlingException', 'slow down'),
    ], cancel), echoEncoder());
    expect(refused.status).toBe(429);
    expect(await refused.json()).toEqual({ message: 'slow down', type: 'throttlingException' });
    // An opening frame and then the end is no answer either.
    expect((await bedrockEventStreamToSse(eventStreamResponse([bedrockEventMessage('open', {})]), echoEncoder())).status).toBe(502);
    const served = await bedrockEventStreamToSse(eventStreamResponse([
      bedrockEventMessage('open', {}),
      bedrockEventMessage('a', {}),
    ]), echoEncoder());
    expect(served.status).toBe(200);
    expect(await served.text()).toBe('data: open\n\ndata: a\n\ndata: [END]\n\n');
  });

  it('refuses an empty or malformed stream before any output', async () => {
    expect((await bedrockEventStreamToSse(eventStreamResponse([]), echoEncoder())).status).toBe(502);
    const corrupt = bedrockEventMessage('messageStart', {});
    corrupt[corrupt.length - 1] ^= 0xff;
    const res = await bedrockEventStreamToSse(eventStreamResponse([corrupt]), echoEncoder());
    expect(res.status).toBe(502);
    expect((await res.json()).type).toBe('eventStreamError');
  });

  it('streams SSE after the first output, ending with the encoder\'s tail', async () => {
    const res = await bedrockEventStreamToSse(eventStreamResponse([
      bedrockEventMessage('a', {}),
      concat([bedrockEventMessage('skip', {}), bedrockEventMessage('b', {})]),
    ]), echoEncoder());
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('text/event-stream');
    expect(await res.text()).toBe('data: a\n\ndata: b\n\ndata: [END]\n\n');
  });

  it('ends the stream with a BedrockStreamError for an exception after the first output, or with the encoder\'s text', async () => {
    const errored = await bedrockEventStreamToSse(eventStreamResponse([
      bedrockEventMessage('a', {}),
      bedrockExceptionMessage('modelStreamErrorException', 'broke'),
    ]), echoEncoder());
    await expect(errored.text()).rejects.toBeInstanceOf(BedrockStreamError);
    const said = await bedrockEventStreamToSse(eventStreamResponse([
      bedrockEventMessage('a', {}),
      bedrockExceptionMessage('modelStreamErrorException', 'broke'),
    ]), echoEncoder('ERR'));
    expect(await said.text()).toBe('data: a\n\nERR:modelStreamErrorException\n\n');
  });

  it('errors the stream on a CRC mismatch mid-stream, never skipping the frame', async () => {
    const bad = bedrockEventMessage('b', {});
    bad[20] ^= 0x10;
    const res = await bedrockEventStreamToSse(eventStreamResponse([bedrockEventMessage('a', {}), bad]), echoEncoder());
    await expect(res.text()).rejects.toThrow(/CRC mismatch/);
  });

  it('cancels the upstream read when the consumer cancels', async () => {
    // The upstream stays open after its frames, as a stream still generating
    // does. A finite upstream can be read ahead to its end and close before
    // the cancel arrives, and cancelling a closed stream never reaches its
    // source, so whether the spy fired would depend on scheduling.
    const cancel = vi.fn();
    const chunks = [bedrockEventMessage('a', {}), bedrockEventMessage('b', {})];
    let i = 0;
    const upstream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (i < chunks.length) {
          controller.enqueue(chunks[i++]);
          return;
        }
        return new Promise<void>(() => {});
      },
      cancel(reason) { cancel(reason); },
    });
    const res = await bedrockEventStreamToSse(
      new Response(upstream, { status: 200, headers: { 'content-type': 'application/vnd.amazon.eventstream' } }),
      echoEncoder(),
    );
    const reader = res.body!.getReader();
    expect(new TextDecoder().decode((await reader.read()).value)).toBe('data: a\n\n');
    await reader.cancel('client went away');
    expect(cancel).toHaveBeenCalledWith('client went away');
  });
});

// The bearer key goes wherever the built URL points, so a region is checked
// before it is spliced into the host.
describe('bedrockBaseUrl', () => {
  it('derives the runtime host from a region name, and takes a base URL as given', () => {
    expect(bedrockBaseUrl({})).toBe('https://bedrock-runtime.us-east-1.amazonaws.com');
    expect(bedrockBaseUrl({ region: 'us-gov-west-1' })).toBe('https://bedrock-runtime.us-gov-west-1.amazonaws.com');
    expect(bedrockBaseUrl({ region: 'ap-southeast-2' })).toBe('https://bedrock-runtime.ap-southeast-2.amazonaws.com');
    expect(bedrockBaseUrl({ baseUrl: 'https://vpce.example/runtime//', region: 'nonsense' })).toBe('https://vpce.example/runtime');
  });

  it('refuses a region that is not a region name rather than build a host from it', () => {
    for (const region of ['eu-west-1.evil.example/x?', 'us-east-1@evil.example', 'US-EAST-1', '', 'us-east']) {
      expect(() => bedrockBaseUrl({ region }), region).toThrow(BedrockRequestUnsupported);
    }
  });
});
