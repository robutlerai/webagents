/**
 * Signature verification — slack v0, twilio HMAC-SHA1, meta sha256,
 * discord Ed25519. Tests cover happy path, tampered body, replay
 * window, dual-secret rotation.
 */
import { describe, it, expect } from 'vitest';
import { createHmac } from 'node:crypto';
import {
  verifyMetaSignature,
  verifySlackSignature,
  verifyTwilioSignature,
  verifyXWebhookSignature,
  buildXCrcResponse,
  verifyGoogleChatJwt,
} from '../../../../src/skills/messaging/shared';

describe('verifySlackSignature', () => {
  const secret = 'sshh';
  const body = 'token=foo&team_id=T1';
  const ts = String(Math.floor(Date.now() / 1000));
  const sig = 'v0=' + createHmac('sha256', secret).update(`v0:${ts}:${body}`).digest('hex');

  it('accepts valid signature', () => {
    expect(verifySlackSignature({ signingSecret: secret, timestamp: ts, signature: sig, rawBody: body })).toBe(true);
  });

  it('rejects tampered body', () => {
    expect(verifySlackSignature({ signingSecret: secret, timestamp: ts, signature: sig, rawBody: body + 'x' })).toBe(false);
  });

  it('rejects timestamps older than 5 minutes (replay)', () => {
    const oldTs = String(Math.floor(Date.now() / 1000) - 60 * 10);
    const oldSig = 'v0=' + createHmac('sha256', secret).update(`v0:${oldTs}:${body}`).digest('hex');
    expect(verifySlackSignature({ signingSecret: secret, timestamp: oldTs, signature: oldSig, rawBody: body })).toBe(false);
  });

  it('rejects with empty inputs', () => {
    expect(verifySlackSignature({ signingSecret: '', timestamp: ts, signature: sig, rawBody: body })).toBe(false);
    expect(verifySlackSignature({ signingSecret: secret, timestamp: '', signature: sig, rawBody: body })).toBe(false);
    expect(verifySlackSignature({ signingSecret: secret, timestamp: ts, signature: '', rawBody: body })).toBe(false);
  });
});

describe('verifyTwilioSignature', () => {
  const authToken = 'twilio-token';
  const url = 'https://example.com/twilio/status';
  const params = { MessageSid: 'SM1', MessageStatus: 'delivered' };
  const concat = url + Object.keys(params).sort().map((k) => k + (params as Record<string, string>)[k]).join('');
  const sig = createHmac('sha1', authToken).update(concat).digest('base64');

  it('accepts valid signature', () => {
    expect(verifyTwilioSignature({ authToken, url, params, signature: sig })).toBe(true);
  });

  it('rejects tampered params', () => {
    expect(verifyTwilioSignature({ authToken, url, params: { ...params, MessageStatus: 'failed' }, signature: sig })).toBe(false);
  });

  it('rejects with empty token', () => {
    expect(verifyTwilioSignature({ authToken: '', url, params, signature: sig })).toBe(false);
  });
});

describe('verifyMetaSignature', () => {
  const secret = 'app-secret';
  const body = '{"object":"page"}';
  const sigHex = createHmac('sha256', secret).update(body).digest('hex');

  it('accepts sha256= prefixed signature', () => {
    expect(verifyMetaSignature({ appSecrets: [secret], signatureHeader: 'sha256=' + sigHex, rawBody: body })).toBe(true);
  });

  it('supports dual-secret rotation (older secret accepted)', () => {
    expect(verifyMetaSignature({ appSecrets: ['new-secret', secret], signatureHeader: 'sha256=' + sigHex, rawBody: body })).toBe(true);
  });

  it('rejects unprefixed signature', () => {
    expect(verifyMetaSignature({ appSecrets: [secret], signatureHeader: sigHex, rawBody: body })).toBe(false);
  });

  it('rejects tampered body', () => {
    expect(verifyMetaSignature({ appSecrets: [secret], signatureHeader: 'sha256=' + sigHex, rawBody: body + 'x' })).toBe(false);
  });
});

describe('verifyXWebhookSignature', () => {
  const secret = 'x-consumer-secret';
  const body = '{"for_user_id":"123"}';
  const sigB64 = createHmac('sha256', secret).update(body).digest('base64');

  it('accepts sha256= prefixed signature', () => {
    expect(
      verifyXWebhookSignature({ consumerSecret: secret, signatureHeader: 'sha256=' + sigB64, rawBody: body }),
    ).toBe(true);
  });

  it('rejects unprefixed signature', () => {
    expect(
      verifyXWebhookSignature({ consumerSecret: secret, signatureHeader: sigB64, rawBody: body }),
    ).toBe(false);
  });

  it('rejects tampered body', () => {
    expect(
      verifyXWebhookSignature({ consumerSecret: secret, signatureHeader: 'sha256=' + sigB64, rawBody: body + 'x' }),
    ).toBe(false);
  });

  it('rejects with empty inputs', () => {
    expect(
      verifyXWebhookSignature({ consumerSecret: '', signatureHeader: 'sha256=' + sigB64, rawBody: body }),
    ).toBe(false);
    expect(
      verifyXWebhookSignature({ consumerSecret: secret, signatureHeader: '', rawBody: body }),
    ).toBe(false);
  });
});

describe('buildXCrcResponse', () => {
  it('produces sha256= base64 HMAC over the crc_token', () => {
    const secret = 'x-consumer-secret';
    const token = 'crc-challenge-token-abc';
    const expected = 'sha256=' + createHmac('sha256', secret).update(token).digest('base64');
    expect(buildXCrcResponse(secret, token)).toBe(expected);
  });
});

describe('verifyGoogleChatJwt - shape rejections (signature path requires real RSA key)', () => {
  // We can't reproduce a valid Google-signed RSA-SHA256 JWT in unit tests
  // without faking the JWKS endpoint, so these tests cover the explicit
  // pre-signature guards (split, alg, iss, aud, exp).
  function fakeJwt(header: object, payload: object): string {
    const enc = (o: object) =>
      Buffer.from(JSON.stringify(o)).toString('base64url');
    return `${enc(header)}.${enc(payload)}.AAAA`;
  }

  it('rejects empty / malformed jwt', async () => {
    expect(await verifyGoogleChatJwt({ jwt: '', expectedAudience: 'p' })).toBe(false);
    expect(await verifyGoogleChatJwt({ jwt: 'not-a-jwt', expectedAudience: 'p' })).toBe(false);
  });

  it('rejects non-RS256 alg', async () => {
    const jwt = fakeJwt(
      { alg: 'HS256', kid: 'k1' },
      { iss: 'chat@system.gserviceaccount.com', aud: 'p', exp: Date.now() / 1000 + 600 },
    );
    expect(await verifyGoogleChatJwt({ jwt, expectedAudience: 'p' })).toBe(false);
  });

  it('rejects wrong issuer', async () => {
    const jwt = fakeJwt(
      { alg: 'RS256', kid: 'k1' },
      { iss: 'attacker@gserviceaccount.com', aud: 'p', exp: Date.now() / 1000 + 600 },
    );
    expect(await verifyGoogleChatJwt({ jwt, expectedAudience: 'p' })).toBe(false);
  });

  it('rejects wrong audience', async () => {
    const jwt = fakeJwt(
      { alg: 'RS256', kid: 'k1' },
      { iss: 'chat@system.gserviceaccount.com', aud: 'other', exp: Date.now() / 1000 + 600 },
    );
    expect(await verifyGoogleChatJwt({ jwt, expectedAudience: 'p' })).toBe(false);
  });

  it('rejects expired tokens', async () => {
    const jwt = fakeJwt(
      { alg: 'RS256', kid: 'k1' },
      { iss: 'chat@system.gserviceaccount.com', aud: 'p', exp: Date.now() / 1000 - 600 },
    );
    expect(await verifyGoogleChatJwt({ jwt, expectedAudience: 'p' })).toBe(false);
  });
});
