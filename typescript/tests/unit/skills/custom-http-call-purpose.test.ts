/**
 * S13 (E2E-PAY, dev push 13, 2026-11-17). The host stamps why it made a call on `ctx.metadata.callPurpose` (the
 * checkout's buyer gate), and the custom_http skill carries it to the function source as `source.purpose`, where the
 * portal's quota hook reads it. Only a known purpose passes; anything else is dropped.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { extractPurpose } from '../../../src/skills/custom-http/skill';
import { FUNCTION_CALL_PURPOSES } from '../../../src/skills/functions/context';

const ctx = (metadata: Record<string, unknown> | undefined) => ({ metadata }) as never;

describe('extractPurpose', () => {
  it('passes buyer-gate through', () => {
    expect(FUNCTION_CALL_PURPOSES).toContain('buyer-gate');
    expect(extractPurpose(ctx({ callPurpose: 'buyer-gate' }))).toBe('buyer-gate');
  });

  it('drops anything unknown, missing or not a string', () => {
    expect(extractPurpose(ctx({ callPurpose: 'admin' }))).toBeUndefined();
    expect(extractPurpose(ctx({ callPurpose: ['buyer-gate'] }))).toBeUndefined();
    expect(extractPurpose(ctx({}))).toBeUndefined();
    expect(extractPurpose(ctx(undefined))).toBeUndefined();
  });

  it('the endpoint source carries it beside billedTo, and nested calls do not inherit it', () => {
    const src = readFileSync(join(__dirname, '../../../src/skills/custom-http/skill.ts'), 'utf8');
    expect(src).toContain('...(extractPurpose(ctx) ? { purpose: extractPurpose(ctx) } : {}),');
    const fnSkill = readFileSync(join(__dirname, '../../../src/skills/functions/skill.ts'), 'utf8');
    const child = fnSkill.slice(fnSkill.indexOf('const childCtx: SerializableContext = {'), fnSkill.indexOf('auth: { userId: null, agentId: skill.agentId'));
    expect(child).not.toContain('purpose');
  });
});

describe('no request can claim a purpose (S13 condition: only the host dispatcher sets it)', () => {
  it('a request body or header naming callPurpose does not reach the source', async () => {
    const { buildRequestMetadata } = await import('../../../src/server/handler');
    const req = new Request('https://x.test/agents/a/run', { method: 'POST', headers: { 'x-call-purpose': 'buyer-gate', callPurpose: 'buyer-gate' } });
    const meta = buildRequestMetadata(req, { callPurpose: 'buyer-gate', chat_id: 'c1' });
    expect(meta.callPurpose).toBeUndefined();
    expect(meta.chat_id).toBe('c1');
    expect(extractPurpose(ctx(meta))).toBeUndefined();
  });

  it('the HTTP servers build a fresh metadata object for every request, with no body or header fields spread in', () => {
    for (const f of ['node.ts', 'multi.ts']) {
      const src = readFileSync(join(__dirname, '../../../src/server', f), 'utf8');
      for (const m of src.matchAll(/context\.metadata = \{([^}]*)\};/g)) {
        expect(m[1]).not.toMatch(/\.\.\./);
        expect(m[1]).not.toMatch(/callPurpose/);
      }
    }
  });

  it('extractPurpose reads only ctx.metadata, never the request', () => {
    const src = readFileSync(join(__dirname, '../../../src/skills/custom-http/skill.ts'), 'utf8');
    const fn = src.slice(src.indexOf('export function extractPurpose('), src.indexOf('\n}\n', src.indexOf('export function extractPurpose(')));
    expect(fn).toContain('ctx?.metadata');
    expect(fn).not.toMatch(/request|headers|body|query/);
  });
});
