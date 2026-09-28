/**
 * Verifying the exportable TrustFlow record (plan item 2.7, 2026-09-26): the
 * vectors both SDKs verify (`python/tests/fixtures/trust/trustflow_record.json`,
 * signed by the portal's own signer byte for byte, pinned there by
 * tests/unit/reputation/trust-record-w2trust.test.ts; Python
 * tests/trustflow/test_trust_record_w2trust.py), the key set fetch, and the
 * A2A card extension.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  TRUSTFLOW_RECORD_EXTENSION_URI,
  _resetTrustKeySets,
  decodeTrustRecord,
  trustRecordExtension,
  trustRecordFromCard,
  verifyTrustRecord,
  withTrustRecordExtension,
} from '../../../src/trustflow/trust-record';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/trust/trustflow_record.json'), 'utf8')) as {
  jwks: { keys: JsonWebKey[] };
  issuer: string;
  header: Record<string, string>;
  payload: Record<string, unknown>;
  now: number;
  records: Record<string, string>;
  cases: Array<{ name: string; record: string; subject?: Record<string, string>; issuer?: string; expect: { ok: boolean; code?: string; kid?: string } }>;
  extension: { uri: string; description: string; card_with_record: Record<string, unknown> };
};

const held = { keys: FIXTURE.jwks.keys, issuer: FIXTURE.issuer, now: FIXTURE.now };

describe('the shared vectors', () => {
  for (const c of FIXTURE.cases) {
    it(c.name, async () => {
      const result = await verifyTrustRecord(FIXTURE.records[c.record], {
        ...held,
        ...(c.issuer ? { issuer: c.issuer } : {}),
        ...(c.subject ? { subject: c.subject } : {}),
      });
      expect(result.ok, JSON.stringify(result)).toBe(c.expect.ok);
      if (result.ok) {
        expect(result.record).toEqual(FIXTURE.payload);
        if (c.expect.kid) expect(result.kid).toBe(c.expect.kid);
      } else {
        expect(result.code).toBe(c.expect.code);
      }
    });
  }

  it('decodes the header and claims without deciding anything', () => {
    expect(decodeTrustRecord(FIXTURE.records.valid)).toEqual({ header: FIXTURE.header, payload: FIXTURE.payload });
    expect(decodeTrustRecord(FIXTURE.records.tampered)?.payload.score).toBe(0.99);
    expect(decodeTrustRecord('nope')).toBeNull();
  });
});

describe('the key set', () => {
  const JWKS_URL = `${FIXTURE.issuer}/.well-known/jwks.json`;
  beforeEach(() => _resetTrustKeySets());
  afterEach(() => {
    delete process.env.ROBUTLER_PLATFORM_ISSUER;
  });

  it('is fetched from the ISSUER the caller expects, held, and refetched once for an unknown kid', async () => {
    const fetch = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe(JWKS_URL);
      return new Response(JSON.stringify(FIXTURE.jwks), { headers: { 'content-type': 'application/json' } });
    }) as unknown as typeof globalThis.fetch;
    expect((await verifyTrustRecord(FIXTURE.records.valid, { issuer: FIXTURE.issuer, now: FIXTURE.now, fetch })).ok).toBe(true);
    expect((await verifyTrustRecord(FIXTURE.records.valid, { issuer: FIXTURE.issuer, now: FIXTURE.now, fetch })).ok).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(1);
    const unknown = await verifyTrustRecord(FIXTURE.records.unknown_kid, { issuer: FIXTURE.issuer, now: FIXTURE.now, fetch });
    expect(unknown).toMatchObject({ ok: false, code: 'no_key' });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('never fetches from an issuer the filter refuses, and reports the refusal', async () => {
    const fetch = vi.fn();
    const result = await verifyTrustRecord(FIXTURE.records.valid, { issuer: 'http://169.254.169.254', now: FIXTURE.now, fetch: fetch as never });
    expect(result).toMatchObject({ ok: false, code: 'key_set' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('a key set that cannot be read is a refusal, not a throw', async () => {
    const fetch = vi.fn(async () => new Response('down', { status: 503 })) as unknown as typeof globalThis.fetch;
    const result = await verifyTrustRecord(FIXTURE.records.valid, { issuer: FIXTURE.issuer, now: FIXTURE.now, fetch });
    expect(result).toMatchObject({ ok: false, code: 'key_set' });
  });

  it('the default issuer is configuration (ROBUTLER_PLATFORM_ISSUER), never the record', async () => {
    process.env.ROBUTLER_PLATFORM_ISSUER = FIXTURE.issuer;
    expect((await verifyTrustRecord(FIXTURE.records.valid, { keys: FIXTURE.jwks.keys, now: FIXTURE.now })).ok).toBe(true);
    process.env.ROBUTLER_PLATFORM_ISSUER = 'https://staging.robutler.net';
    expect(await verifyTrustRecord(FIXTURE.records.valid, { keys: FIXTURE.jwks.keys, now: FIXTURE.now })).toMatchObject({ ok: false, code: 'issuer' });
  });
});

describe('the A2A card extension', () => {
  const bare = (() => {
    const card = JSON.parse(JSON.stringify(FIXTURE.extension.card_with_record)) as Record<string, unknown>;
    delete (card.capabilities as Record<string, unknown>).extensions;
    return card;
  })();

  it('writes the fixture entry under the Robutler URI', () => {
    expect(TRUSTFLOW_RECORD_EXTENSION_URI).toBe(FIXTURE.extension.uri);
    expect(trustRecordExtension(FIXTURE.records.valid)).toEqual({ uri: FIXTURE.extension.uri, description: FIXTURE.extension.description, params: { record: FIXTURE.records.valid } });
    expect(withTrustRecordExtension(bare, FIXTURE.records.valid)).toEqual(FIXTURE.extension.card_with_record);
    expect((bare.capabilities as Record<string, unknown>).extensions).toBeUndefined();
  });

  it('replaces an earlier record and keeps other extensions', () => {
    const other = { uri: 'https://other.example/ext', params: { a: 1 } };
    const once = withTrustRecordExtension({ ...bare, capabilities: { streaming: true, extensions: [other] } }, 'old.record.x');
    const twice = withTrustRecordExtension(once, FIXTURE.records.valid);
    expect((twice.capabilities as { extensions: unknown[] }).extensions).toEqual([other, trustRecordExtension(FIXTURE.records.valid)]);
  });

  it('reads the record back, unverified, and verifies it for the card subject', async () => {
    const record = trustRecordFromCard(FIXTURE.extension.card_with_record);
    expect(record).toBe(FIXTURE.records.valid);
    expect(trustRecordFromCard(bare)).toBeNull();
    const verified = await verifyTrustRecord(record!, { ...held, subject: { url: 'https://agents.example.com/agents/scout' } });
    expect(verified.ok).toBe(true);
    const foreign = await verifyTrustRecord(record!, { ...held, subject: { url: 'https://elsewhere.example/agents/scout' } });
    expect(foreign).toMatchObject({ ok: false, code: 'subject' });
  });
});
