/**
 * S-226: a served agent that cannot verify its callers answers only this
 * machine (2026-09-24).
 *
 * Measured before the fix against `webagents serve` built from this tree: it
 * listened on every interface, answered an unrelated origin's preflight with
 * `access-control-allow-origin: *` and `authorization` allowed, and let a
 * cross-origin `POST /chat/completions` carrying `Bearer <anything>` through to
 * the model, with `*` on the reply as well (from the fetch-handler fallback,
 * not just the middleware). WebSocket handshakes had no origin check at all.
 */

import { describe, expect, it } from 'vitest';

import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { createAgentApp } from '../../../src/server/node.js';
import {
  defaultHostname,
  isLoopbackOrigin,
  originPolicy,
  upgradeOriginAllowed,
} from '../../../src/server/origin-policy.js';
import { UAMPTransportSkill } from '../../../src/skills/transport/uamp/skill.js';

/** Stands in for the platform AuthSkill: the policy keys on the class name. */
class AuthSkill extends Skill {}

const EVIL = 'https://unrelated.example';
const LOCAL_UI = 'http://localhost:5173';

function appFor(skills: Skill[] = [], cors?: boolean | string[]) {
  const agent = new BaseAgent({ name: 'mini', instructions: 'x', skills });
  return createAgentApp(agent, { basePath: '/agents/mini', logging: false, cors });
}

async function preflight(app: ReturnType<typeof appFor>['app'], origin: string) {
  return app.request('/agents/mini/chat/completions', {
    method: 'OPTIONS',
    headers: {
      Origin: origin,
      'Access-Control-Request-Method': 'POST',
      'Access-Control-Request-Headers': 'authorization,content-type',
    },
  });
}

describe('the origin policy', () => {
  it('knows a page served from this machine', () => {
    for (const origin of ['http://localhost:3000', 'http://127.0.0.1:8080', 'http://[::1]:9000', 'https://localhost']) {
      expect(isLoopbackOrigin(origin)).toBe(true);
    }
    for (const origin of [EVIL, 'http://localhost.evil.example', 'http://127.0.0.1.nip.io']) {
      expect(isLoopbackOrigin(origin)).toBe(false);
    }
  });

  it('answers loopback origins only for an agent that cannot verify callers', () => {
    const policy = originPolicy(undefined, false);
    expect(policy(EVIL)).toBeNull();
    expect(policy(LOCAL_UI)).toBe(LOCAL_UI);
  });

  it('keeps the permissive policy for an agent that verifies its callers', () => {
    expect(originPolicy(undefined, true)(EVIL)).toBe('*');
  });

  it('lets an explicit setting win either way', () => {
    expect(originPolicy(true, false)(EVIL)).toBe('*');
    expect(originPolicy(false, true)(EVIL)).toBeNull();
    expect(originPolicy([EVIL], false)(EVIL)).toBe(EVIL);
    expect(originPolicy([EVIL], false)('https://other.example')).toBeNull();
  });

  it('binds loopback unless the agent is meant to be reached or verifies callers', () => {
    expect(defaultHostname({ verifiesCredentials: false })).toBe('127.0.0.1');
    expect(defaultHostname({ publicUrl: 'https://agent.example.com', verifiesCredentials: false })).toBe('0.0.0.0');
    expect(defaultHostname({ verifiesCredentials: true })).toBe('0.0.0.0');
  });

  it('holds WebSocket handshakes to the same rule, and leaves non-browser clients alone', () => {
    const policy = originPolicy(undefined, false);
    expect(upgradeOriginAllowed(policy, EVIL)).toBe(false);
    expect(upgradeOriginAllowed(policy, LOCAL_UI)).toBe(true);
    expect(upgradeOriginAllowed(policy, undefined)).toBe(true);
  });
});

describe('a served agent without an AuthSkill', () => {
  it('does not approve an unrelated origin in a preflight', async () => {
    const res = await preflight(appFor().app, EVIL);
    // THE BUG: `access-control-allow-origin: *` here.
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });

  it('puts no allow-origin on the model reply either, from any layer', async () => {
    const res = await appFor().app.request('/agents/mini/chat/completions', {
      method: 'POST',
      headers: { Origin: EVIL, Authorization: 'Bearer anything', 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    });
    // THE BUG: the fetch-handler fallback stamped `*` on this response itself.
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });

  it('still answers a local page on another port', async () => {
    const res = await preflight(appFor().app, LOCAL_UI);
    expect(res.headers.get('access-control-allow-origin')).toBe(LOCAL_UI);
  });

  it('refuses a cross-origin WebSocket handshake before anything else', async () => {
    const skill = new UAMPTransportSkill();
    const agent = new BaseAgent({ name: 'mini', instructions: 'x', skills: [skill] });
    await skill.initialize?.(agent as never);
    const { handleUpgrade } = createAgentApp(agent, { basePath: '/agents/mini', logging: false });

    const written: string[] = [];
    const socket = {
      write: (chunk: string) => { written.push(String(chunk)); return true; },
      destroy: () => {},
      on: () => socket,
      removeListener: () => socket,
    };
    handleUpgrade(
      {
        url: '/agents/mini/uamp?token=anything',
        method: 'GET',
        headers: { host: 'localhost:3000', origin: EVIL, upgrade: 'websocket', connection: 'Upgrade' },
      } as never,
      socket as never,
      Buffer.alloc(0),
    );
    expect(written.join('')).toContain('403 Forbidden');
  });
});

describe('a served agent with an AuthSkill', () => {
  it('keeps answering any origin, because it verifies what it is sent', async () => {
    const res = await preflight(appFor([new AuthSkill()]).app, EVIL);
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
  });
});
