/**
 * defaultEnvTokenResolver — env var fallback used when the host has not
 * supplied a custom getToken.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { defaultEnvTokenResolver, PROVIDER_ENV } from '../../../../src/skills/messaging/shared';

const ENV_KEYS = new Set<string>();
function setEnv(k: string, v: string) {
  ENV_KEYS.add(k);
  process.env[k] = v;
}

beforeEach(() => {
  ENV_KEYS.clear();
});

afterEach(() => {
  for (const k of ENV_KEYS) delete process.env[k];
});

describe('defaultEnvTokenResolver', () => {
  it('returns null when token env is not set', async () => {
    delete process.env.TELEGRAM_BOT_TOKEN;
    const r = defaultEnvTokenResolver();
    expect(await r.getToken({ provider: 'telegram' })).toBeNull();
  });

  it('returns null for unknown provider', async () => {
    const r = defaultEnvTokenResolver();
    expect(await r.getToken({ provider: 'nonsense' })).toBeNull();
  });

  it('reads telegram token + bot username metadata', async () => {
    setEnv('TELEGRAM_BOT_TOKEN', 'BOT123');
    setEnv('TELEGRAM_BOT_USERNAME', 'mybot');
    const r = await defaultEnvTokenResolver().getToken({ provider: 'telegram' });
    expect(r?.token).toBe('BOT123');
    expect(r?.metadata).toEqual({ botUsername: 'mybot' });
  });

  it('reads twilio token + sid + from + service sid', async () => {
    setEnv('TWILIO_AUTH_TOKEN', 'auth');
    setEnv('TWILIO_ACCOUNT_SID', 'AC1');
    setEnv('TWILIO_FROM_NUMBER', '+15555550100');
    const r = await defaultEnvTokenResolver().getToken({ provider: 'twilio' });
    expect(r?.token).toBe('auth');
    expect(r?.metadata).toMatchObject({ accountSid: 'AC1', fromNumber: '+15555550100' });
  });

  it('exposes all 10 messaging providers via PROVIDER_ENV', () => {
    const expected = [
      'telegram', 'twilio', 'slack', 'discord',
      'whatsapp', 'messenger', 'instagram',
      'linkedin', 'bluesky', 'reddit',
    ];
    for (const p of expected) {
      expect(PROVIDER_ENV[p]).toBeDefined();
      expect(PROVIDER_ENV[p].tokenEnv).toMatch(/^[A-Z][A-Z0-9_]+$/);
    }
  });
});
