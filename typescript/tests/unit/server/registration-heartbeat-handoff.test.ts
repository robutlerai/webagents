/**
 * Registration hands its bearer to the heartbeat itself (2026-09-24).
 *
 * The bearer a self-registering agent receives was the caller's to export as
 * WEBAGENTS_AGENT_TOKEN and restart with, so the heartbeat could read it. The
 * bridge read the same variable and refused it (no `agent_id` claim), so an
 * agent that followed the docs could not also use PortalConnect.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import * as path from 'node:path';

import { loadOrCreateAgentIdentity } from '../../../src/crypto/identity-store';
import {
  heartbeatRunning,
  registerWithPlatform,
  startHeartbeat,
  stopHeartbeat,
} from '../../../src/server/registration';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

const ISSUER = 'https://agent.example.com/agents/mini';

afterEach(() => {
  vi.unstubAllGlobals();
  stopHeartbeat(ISSUER);
  delete process.env.ROBUTLER_API_URL;
  delete process.env.WEBAGENTS_AGENT_TOKEN;
});

function stubPlatform(calls: { url: string; auth?: string }[]) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
    calls.push({ url, auth: headers.get('authorization') ?? undefined });
    if (url.endsWith('/api/auth/cli/token')) {
      return new Response(JSON.stringify({ access_token: 'bearer-from-registration', username: 'com.example.agent', user_id: 'u-1' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
    return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } });
  }));
}

describe('registerWithPlatform', () => {
  it('starts the heartbeat with the bearer it received, with nothing exported', async () => {
    process.env.ROBUTLER_API_URL = 'https://platform.example.com';
    const keysDir = tempDir('wa-handoff-keys-');
    const identity = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
    const calls: { url: string; auth?: string }[] = [];
    stubPlatform(calls);

    const result = await registerWithPlatform(identity);
    expect(result.ok).toBe(true);
    // THE FRICTION: this needed WEBAGENTS_AGENT_TOKEN exported and a restart.
    expect(heartbeatRunning(ISSUER)).toBe(true);
    await vi.waitFor(() =>
      expect(calls.some((c) => c.url.endsWith('/api/agents/heartbeat') && c.auth === 'Bearer bearer-from-registration')).toBe(true),
    );
  });

  it('does not start a second heartbeat when serve() already runs one', async () => {
    process.env.ROBUTLER_API_URL = 'https://platform.example.com';
    const keysDir = tempDir('wa-handoff-keys-');
    const identity = await loadOrCreateAgentIdentity('mini', { issuer: ISSUER, keysDir });
    const calls: { url: string; auth?: string }[] = [];
    stubPlatform(calls);
    startHeartbeat('mini', { token: 'agent-key', key: ISSUER });

    await registerWithPlatform(identity);
    await new Promise((r) => setTimeout(r, 20));
    const beats = calls.filter((c) => c.url.endsWith('/api/agents/heartbeat'));
    expect(beats.every((c) => c.auth === 'Bearer agent-key')).toBe(true);
  });
});
