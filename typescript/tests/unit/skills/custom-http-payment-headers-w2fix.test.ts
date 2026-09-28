/**
 * S-296 (2026-09-26): the custom HTTP skill's header filter, the defensive
 * mirror of the platform dispatcher's, withholds the buyer's payment
 * credential from the function. A priced endpoint's paywall reads
 * `PAYMENT-SIGNATURE` (x402 v2), `X-PAYMENT` (v1) or `Payment-Authorization`
 * (MPP) from the request and then runs the handler with the same request; the
 * filter used to forward all three, so a self-hosted agent's function
 * received a bearer it could settle or spend.
 *
 * The Python custom_http skill is declaration-only (it builds no function
 * request from headers), so this has no Python twin.
 */
import { describe, expect, it } from 'vitest';
import { CustomHttpSkill } from '../../../src/skills/custom-http/skill';
import type { FunctionRuntimeSkill } from '../../../src/skills/functions/skill';
import type { SerializableContext } from '../../../src/skills/functions/executor-client';

const AGENT_ID = '11111111-2222-4333-8444-555555555555';

function skillWithRecorder(): { skill: CustomHttpSkill; seen: SerializableContext[] } {
  const seen: SerializableContext[] = [];
  const runtime = {
    get: () => ({ manifest: { permissions: {} } }),
    invoke: async (_name: string, ctx: SerializableContext) => {
      seen.push(ctx);
      return { ok: true, result: { status: 200, headers: { 'content-type': 'application/json' }, body: '{"ok":true}' } };
    },
  } as unknown as FunctionRuntimeSkill;
  const skill = new CustomHttpSkill({
    runtime,
    endpoints: [{ id: 'quote', path: '/quote', method: 'POST', auth: 'public', use: 'quote' }],
  });
  return { skill, seen };
}

describe('S-296: the function never sees a payment credential', () => {
  it('strips PAYMENT-SIGNATURE, X-PAYMENT and Payment-Authorization, keeps ordinary headers and the body', async () => {
    const { skill, seen } = skillWithRecorder();
    const endpoint = skill.httpEndpoints.find((e) => e.path === '/quote');
    expect(endpoint).toBeDefined();
    const request = new Request('https://agent.example.com/agents/mini/quote?q=1', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'payment-signature': 'eyJ4NDAyVmVyc2lvbiI6Mn0=',
        'x-payment': 'eyJ4NDAyVmVyc2lvbiI6MX0=',
        'payment-authorization': 'Payment credential',
        'x-trace': 'kept',
        cookie: `agt_${AGENT_ID}_theme=dark; session=platform-cookie`,
      },
      body: JSON.stringify({ q: 'how much' }),
    });
    const response = await endpoint!.handler(request, { metadata: { agentId: AGENT_ID } } as never);
    expect(response.status).toBe(200);
    expect(seen).toHaveLength(1);
    const headers = seen[0].request.headers;
    for (const name of ['payment-signature', 'x-payment', 'payment-authorization', 'authorization']) {
      expect(headers).not.toHaveProperty(name);
    }
    expect(headers['x-trace']).toBe('kept');
    expect(headers['content-type']).toBe('application/json');
    // The existing filters still hold beside the new names.
    expect(headers.cookie).toBe(`agt_${AGENT_ID}_theme=dark`);
    expect(seen[0].request.body).toEqual({ q: 'how much' });
    expect(seen[0].request.query).toEqual({ q: '1' });
  });
});
