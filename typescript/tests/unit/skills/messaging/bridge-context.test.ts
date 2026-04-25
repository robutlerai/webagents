/**
 * BridgeContext helpers — readBridgeContext / bridgeMatches /
 * buildBridgeAwarenessPrompt.
 */
import { describe, it, expect } from 'vitest';
import {
  bridgeMatches,
  buildBridgeAwarenessPrompt,
  readBridgeContext,
} from '../../../../src/skills/messaging/shared';

describe('readBridgeContext', () => {
  it('returns undefined when ctx is missing', () => {
    expect(readBridgeContext(undefined)).toBeUndefined();
  });

  it('returns undefined when bridge metadata is malformed', () => {
    expect(readBridgeContext({ metadata: { bridge: 'no' } } as never)).toBeUndefined();
    expect(readBridgeContext({ metadata: { bridge: { source: 'x' } } } as never)).toBeUndefined();
    expect(readBridgeContext({ metadata: {} } as never)).toBeUndefined();
  });

  it('returns full bridge object when shape is valid', () => {
    const b = readBridgeContext({
      metadata: { bridge: { source: 'telegram', contactExternalId: '42', contactDisplayName: 'A' } },
    } as never);
    expect(b?.source).toBe('telegram');
    expect(b?.contactExternalId).toBe('42');
  });
});

describe('bridgeMatches', () => {
  it('matches by provider id', () => {
    const ctx = {
      metadata: { bridge: { source: 'whatsapp', contactExternalId: '1' } },
    } as never;
    expect(bridgeMatches(ctx, 'whatsapp')).toBeDefined();
    expect(bridgeMatches(ctx, 'telegram')).toBeUndefined();
  });
});

describe('buildBridgeAwarenessPrompt', () => {
  it('mentions provider, send tool, contact name', () => {
    const p = buildBridgeAwarenessPrompt({
      provider: 'telegram',
      contactDisplayName: 'Alice',
      sendToolName: 'telegram_send_text',
    });
    expect(p).toContain('telegram');
    expect(p).toContain('Alice');
    expect(p).toContain('telegram_send_text');
  });

  it('falls back to provider name when contact name is null', () => {
    const p = buildBridgeAwarenessPrompt({
      provider: 'slack',
      contactDisplayName: null,
      sendToolName: 'slack_send_dm',
    });
    expect(p).toContain('slack');
    expect(p).toContain('slack_send_dm');
  });
});
