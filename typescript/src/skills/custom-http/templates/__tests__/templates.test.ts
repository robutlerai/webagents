/**
 * Smoke-tests for the 14 webapp templates.
 *
 * These don't try to be exhaustive — they assert each template's
 * source compiles, parses cleanly when wrapped in a function body,
 * and produces a sane response shape against a stub ctx. Catches
 * regressions where someone edits a template and breaks the syntax
 * or starts referencing a field that's not on the runtime ctx.
 */

import { describe, it, expect } from 'vitest';
import {
  WEBAPP_TEMPLATES,
  WEBAPP_TEMPLATE_NAMES,
  getWebappTemplate,
  listWebappTemplates,
} from '../index';
import { makeStubCtx, type StubCtx } from './stub-ctx';

/**
 * Compile a template's `code` into a callable handler. We strip the
 * `// @robutler-function` frontmatter and rewrite `export async
 * function handler(...)` → `async function handler(...)` so the body
 * is plain JS we can wrap in `new Function`. (`new Function` runs in
 * non-module mode where `export` is a syntax error.)
 */
function compileHandler(code: string): (ctx: StubCtx) => Promise<unknown> {
  const stripped = code
    .replace(/^\/\/.*$/gm, '')
    .replace(/\bexport\s+(async\s+)?function\b/g, (_m, asyncKw) => `${asyncKw ?? ''}function`)
    .replace(/\bexport\s+const\b/g, 'const')
    .trim();
  const wrapped = `${stripped}\nreturn handler;`;
  const factory = new Function(wrapped);
  const handler = factory();
  if (typeof handler !== 'function') {
    throw new Error('compiled template did not return a callable `handler`');
  }
  return handler as (ctx: StubCtx) => Promise<unknown>;
}

interface ResponseLike {
  status: number;
  headers?: Record<string, string>;
  body?: string;
}

describe('webapp templates registry', () => {
  it('exports all 14 templates with the documented metadata shape', () => {
    expect(WEBAPP_TEMPLATE_NAMES).toHaveLength(14);
    for (const name of WEBAPP_TEMPLATE_NAMES) {
      const t = WEBAPP_TEMPLATES[name];
      expect(t.name).toBe(name);
      expect(typeof t.description).toBe('string');
      expect(t.description.length).toBeGreaterThan(0);
      expect(typeof t.when_to_use).toBe('string');
      expect(typeof t.security_notes).toBe('string');
      expect(typeof t.required_permissions).toBe('object');
      expect(typeof t.code).toBe('string');
      expect(t.code.length).toBeGreaterThan(0);
    }
  });

  it('listWebappTemplates returns name + description for every template', () => {
    const list = listWebappTemplates();
    expect(list).toHaveLength(WEBAPP_TEMPLATE_NAMES.length);
    for (const entry of list) {
      expect(entry.name).toBeTruthy();
      expect(entry.description).toBeTruthy();
    }
  });

  it('getWebappTemplate returns by name', () => {
    expect(getWebappTemplate('minimal_html_page').name).toBe('minimal_html_page');
  });

  it('no template body uses the legacy ctx.kv.set name', () => {
    // The unified KV API exposes `put` (write) and `get` (read). The
    // older `set` alias was removed in Phase 5; templates that resurrect
    // it would 500 at runtime. This regression assertion catches that.
    for (const name of WEBAPP_TEMPLATE_NAMES) {
      const body = WEBAPP_TEMPLATES[name].code;
      expect(
        body.includes('ctx.kv.set'),
        `template "${name}" uses ctx.kv.set; rename to ctx.kv.put`,
      ).toBe(false);
    }
  });
});

describe('every template compiles and returns a sane response shape', () => {
  for (const name of WEBAPP_TEMPLATE_NAMES) {
    it(`${name}`, async () => {
      const t = getWebappTemplate(name);
      const handler = compileHandler(t.code);
      const ctx = makeStubCtx({
        auth: { authenticated: true, userId: 'visitor-7', agentId: 'agent-uuid-aaaa', profile: { displayName: 'Visitor Seven', avatarUrl: 'https://x/y.png' } },
      });
      // Some templates only handle specific methods; if their default
      // (GET) returns 405, force a method that matches.
      let res = (await handler(ctx)) as ResponseLike;
      if (res.status === 405) {
        ctx.request.method = 'POST';
        res = (await handler(ctx)) as ResponseLike;
      }
      expect(typeof res.status).toBe('number');
      expect(res.status).toBeGreaterThanOrEqual(100);
      expect(res.status).toBeLessThan(600);
    });
  }
});

describe('identity-first templates handle anonymous visitors gracefully', () => {
  const anonymousFriendly = [
    'visitor_session_personalized',
    'session_check',
    'minimal_html_page',
    'multi_route_dispatch',
    'json_api_endpoint',
    'signin_with_robutler',
  ] as const;

  for (const name of anonymousFriendly) {
    it(`${name}: no auth → no thrown error`, async () => {
      const t = getWebappTemplate(name);
      const handler = compileHandler(t.code);
      const ctx = makeStubCtx({ auth: { authenticated: false } });
      const res = (await handler(ctx)) as ResponseLike;
      expect(res.status).toBeLessThan(500);
    });
  }
});

describe('kv_visitor_state stores per-visitor under user_id namespace', () => {
  it('GET → empty, POST → saved, GET again → returns saved body', async () => {
    const handler = compileHandler(getWebappTemplate('kv_visitor_state').code);
    const ctx = makeStubCtx({ auth: { authenticated: true, userId: 'visitor-7' } });
    const empty = JSON.parse(((await handler(ctx)) as ResponseLike).body ?? 'null');
    expect(empty).toEqual({});
    ctx.request = { method: 'POST', path: '/state', headers: {}, body: { foo: 'bar' } };
    await handler(ctx);
    ctx.request = { method: 'GET', path: '/state', headers: {} };
    const got = JSON.parse(((await handler(ctx)) as ResponseLike).body ?? 'null');
    expect(got).toEqual({ foo: 'bar' });
  });

  it('rejects unauthenticated callers with 401', async () => {
    const handler = compileHandler(getWebappTemplate('kv_visitor_state').code);
    const ctx = makeStubCtx({ auth: { authenticated: false } });
    const res = (await handler(ctx)) as ResponseLike;
    expect(res.status).toBe(401);
  });
});

describe('agent_to_agent_endpoint rejects unknown callers', () => {
  it('caller not in allowlist → 403', async () => {
    const handler = compileHandler(getWebappTemplate('agent_to_agent_endpoint').code);
    const ctx = makeStubCtx({
      auth: { authenticated: true, agentId: 'unknown-agent' },
      request: { method: 'POST', path: '/api', headers: {}, body: { hello: 'world' } },
    });
    const res = (await handler(ctx)) as ResponseLike;
    expect(res.status).toBe(403);
  });
});

describe('logout returns Set-Cookie with the agt_<agentId>_ prefix', () => {
  it('cookie name follows the agent-namespace convention', async () => {
    const handler = compileHandler(getWebappTemplate('logout').code);
    const ctx = makeStubCtx({ auth: { authenticated: true, userId: 'visitor-7', agentId: 'agent-uuid-aaaa' } });
    const res = (await handler(ctx)) as ResponseLike;
    expect(res.status).toBe(302);
    expect(res.headers?.['set-cookie']).toMatch(/^agt_agent-uuid-aaaa_session=/);
    expect(res.headers?.['set-cookie']).toMatch(/Max-Age=0/);
  });
});

describe('signin_with_robutler defends against open-redirect via return_to', () => {
  it('rejects scheme-relative redirects (//evil.com)', async () => {
    const handler = compileHandler(getWebappTemplate('signin_with_robutler').code);
    const ctx = makeStubCtx({ request: { method: 'GET', path: '/login', headers: {}, query: { return_to: '//evil.com/path' } } });
    const res = (await handler(ctx)) as ResponseLike;
    expect(res.headers?.location).toMatch(/redirect=%2F$/);
  });

  it('preserves valid same-host paths', async () => {
    const handler = compileHandler(getWebappTemplate('signin_with_robutler').code);
    const ctx = makeStubCtx({ request: { method: 'GET', path: '/login', headers: {}, query: { return_to: '/dashboard?tab=2' } } });
    const res = (await handler(ctx)) as ResponseLike;
    expect(res.headers?.location).toMatch(/redirect=%2Fdashboard%3Ftab%3D2$/);
  });
});

describe('csrf_protected_form rotates token on submit', () => {
  it('GET issues a token, mismatched POST is rejected, matched POST rotates it', async () => {
    const handler = compileHandler(getWebappTemplate('csrf_protected_form').code);
    const ctx = makeStubCtx({ auth: { authenticated: true, userId: 'visitor-7' } });

    const first = (await handler(ctx)) as ResponseLike;
    const t1 = first.body!.match(/value="([^"]+)"/)![1];

    ctx.request = { method: 'POST', path: '/form', headers: {}, body: { csrf: 'wrong', message: 'hi' } };
    const reject = (await handler(ctx)) as ResponseLike;
    expect(reject.status).toBe(403);

    ctx.request = { method: 'POST', path: '/form', headers: {}, body: { csrf: t1, message: 'hi' } };
    const accept = (await handler(ctx)) as ResponseLike;
    expect(accept.status).toBe(200);

    ctx.request = { method: 'POST', path: '/form', headers: {}, body: { csrf: t1, message: 'replay' } };
    const replay = (await handler(ctx)) as ResponseLike;
    expect(replay.status).toBe(403);
  });
});
