/**
 * FunctionRuntimeSkill quota-gate hook (2026-08-10 account-bucket model).
 *
 * The host injects `config.gate`; these tests pin the contract:
 * blocked → typed failure with retryAfterSec; ok → executor runs, release
 * fires on EVERY exit (including executor throw and mint failure), settle
 * gets the executor's actuals; gate throw → fail open; nested `ctx.fn`
 * children carry the parent's `billedTo`.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  FunctionRuntimeSkill,
  type FunctionGateHook,
} from '../../../../src/skills/functions/skill.js';
import type { SerializableContext } from '../../../../src/skills/functions/executor-client.js';

const MANIFEST = {
  name: 'fn1',
  runtime: 'js-v1',
  entrypoint: 'handler',
} as never;

const FN = {
  fn1: {
    name: 'fn1',
    manifest: MANIFEST,
    codeRef: { kind: 'inline', source: 'export function handler() {}' } as never,
    bundleSha256: 'sha',
  },
};

function ctx(over: Partial<SerializableContext['source']> = {}): SerializableContext {
  return {
    source: { skill: 'custom_http', consumerId: 'ep1', invocationId: 'inv1', ...over },
    auth: { userId: null, agentId: 'agent1', scopes: [] },
    limits: { wallMs: 1000, cpuMs: 500, memoryMb: 64, ingressBytes: 1e6, egressBytes: 1e6 },
  } as SerializableContext;
}

const okExecutor = {
  invoke: async () => ({
    ok: true,
    result: 'ran',
    durationMs: 3,
    cpuMs: 42,
    ingressBytes: 5,
    egressBytes: 7,
  }),
} as never;

describe('FunctionRuntimeSkill gate hook', () => {
  it('blocked gate returns the failure with retryAfterSec, executor never runs', async () => {
    const executorSpy = vi.fn();
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: { invoke: executorSpy } as never,
      gate: async () => ({
        ok: false,
        errorCode: 'FN_QUOTA_EXHAUSTED',
        errorMessage: 'daily compute used',
        retryAfterSec: 1234,
      }),
    });
    const r = await skill.invoke('fn1', ctx());
    expect(r.ok).toBe(false);
    expect(r.errorCode).toBe('FN_QUOTA_EXHAUSTED');
    expect(r.retryAfterSec).toBe(1234);
    expect(executorSpy).not.toHaveBeenCalled();
  });

  it('ok gate runs the executor, then release and settle with actuals', async () => {
    const release = vi.fn(async () => {});
    const settle = vi.fn(async () => {});
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: okExecutor,
      gate: async () => ({ ok: true, release, settle }),
    });
    const r = await skill.invoke('fn1', ctx());
    expect(r.ok).toBe(true);
    expect(release).toHaveBeenCalledTimes(1);
    // settle is fire-and-forget — flush microtasks before asserting.
    await new Promise((res) => setImmediate(res));
    expect(settle).toHaveBeenCalledWith({ cpuMs: 42, ingressBytes: 5, egressBytes: 7 });
  });

  it('release fires when the EXECUTOR throws', async () => {
    const release = vi.fn(async () => {});
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: { invoke: async () => { throw new Error('executor down'); } } as never,
      gate: async () => ({ ok: true, release }),
    });
    await expect(skill.invoke('fn1', ctx())).rejects.toThrow('executor down');
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('release fires when the host-bridge mint fails after reservation', async () => {
    const release = vi.fn(async () => {});
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: okExecutor,
      hostBridge: async () => { throw new Error('mint down'); },
      gate: async () => ({ ok: true, release }),
    });
    const r = await skill.invoke('fn1', ctx());
    expect(r.ok).toBe(false);
    expect(r.errorCode).toBe('HOST_BRIDGE_MINT_FAILED');
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('a THROWING gate fails open', async () => {
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: okExecutor,
      gate: (async () => { throw new Error('redis wedged'); }) as FunctionGateHook,
    });
    const r = await skill.invoke('fn1', ctx());
    expect(r.ok).toBe(true);
  });

  it('validateOnly runs skip the gate', async () => {
    const gate = vi.fn();
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: okExecutor,
      gate: gate as never,
    });
    await skill.invoke('fn1', ctx(), { validateOnly: true });
    expect(gate).not.toHaveBeenCalled();
  });

  it('gate receives the source (billedTo included) and manifest', async () => {
    const gate = vi.fn(async () => ({ ok: true as const }));
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: okExecutor,
      gate,
    });
    await skill.invoke('fn1', ctx({ billedTo: 'surface-owner-1' }));
    expect(gate).toHaveBeenCalledWith(
      expect.objectContaining({
        functionName: 'fn1',
        manifest: MANIFEST,
        source: expect.objectContaining({ billedTo: 'surface-owner-1' }),
      }),
    );
  });

  it('CHAINED (nested) invocations are gated too, with the propagated billedTo', async () => {
    // Production nesting: executor → fn-host `fn.invoke`, which re-enters
    // skill.invoke with a chain and a source carrying the token's billedTo.
    const seen: Array<string | undefined> = [];
    const skill = new FunctionRuntimeSkill({
      agentId: 'agent1',
      functions: FN,
      executor: okExecutor,
      gate: async ({ source }) => {
        seen.push(source.billedTo);
        return { ok: true };
      },
    });
    const r = await skill.invoke('fn1', ctx({ skill: 'function', billedTo: 'surface-owner-1' }), {
      chain: {
        rootInvocationId: 'root-inv',
        depth: 1,
        path: ['parent-fn'],
        budgetRemaining: { wallMs: 10_000, cpuMs: 5_000, memoryMb: 64, ingressBytes: 1e6, egressBytes: 1e6 },
      },
    });
    expect(r.ok).toBe(true);
    expect(seen).toEqual(['surface-owner-1']);
  });
});
