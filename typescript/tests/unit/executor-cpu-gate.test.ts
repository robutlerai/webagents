/**
 * SelfCpuGate — the executor's CPU-pressure admission signal.
 *
 * Regression guard for the loadavg bug: the previous gate compared the
 * NODE's `os.loadavg()` (not namespaced in containers) against the
 * threshold, so a noisy neighbor on the same Kubernetes node made the
 * executor reject every invocation while itself idle. The gate must
 * measure THIS process's cpu time against ITS OWN allotment.
 */
import { describe, it, expect } from 'vitest';
import { SelfCpuGate, detectCgroupCpuBudgetCores } from '../../src/executor/worker-pool';

/**
 * Scriptable clock + cpu counter. The test mutates `t` (ms) and `cpu`
 * (µs of cumulative process cpu time) between `pct()` calls.
 */
function fakeEnv() {
  const state = { t: 0, cpu: 0 };
  return {
    state,
    opts: {
      now: () => state.t,
      readCpuUsage: () => ({ user: state.cpu, system: 0 }) as NodeJS.CpuUsage,
    },
  };
}

describe('SelfCpuGate', () => {
  it('reports ~0% when the process is idle, regardless of node load', () => {
    const { state, opts } = fakeEnv();
    const gate = new SelfCpuGate({ budgetCores: 1, ...opts });
    state.t = 1_000;
    state.cpu = 2_000; // 2ms of cpu over a 1s window = 0.2%
    expect(gate.pct()).toBeLessThan(1);
  });

  it('reports high pressure when the process consumes its allotment', () => {
    const { state, opts } = fakeEnv();
    const gate = new SelfCpuGate({ budgetCores: 1, ...opts });
    state.t = 1_000;
    state.cpu = 950_000; // 950ms cpu over 1s = 95% of a 1-core budget
    expect(gate.pct()).toBeGreaterThan(85);
  });

  it('scales pressure by the core budget', () => {
    const { state, opts } = fakeEnv();
    const gate = new SelfCpuGate({ budgetCores: 4, ...opts });
    state.t = 1_000;
    state.cpu = 950_000; // same burn, 4-core budget → ~23.75%
    const pct = gate.pct();
    expect(pct).toBeGreaterThan(20);
    expect(pct).toBeLessThan(30);
  });

  it('smooths with EWMA across windows instead of spiking', () => {
    const { state, opts } = fakeEnv();
    const gate = new SelfCpuGate({ budgetCores: 1, ...opts });
    state.t = 1_000;
    state.cpu = 1_000_000; // window 1: 100%
    expect(gate.pct()).toBe(100); // first sample seeds the EWMA
    state.t = 2_000; // window 2: zero additional cpu
    expect(gate.pct()).toBe(50); // 100 * 0.5 + 0 * 0.5
  });

  it('does not resample inside the minimum interval', () => {
    const { state, opts } = fakeEnv();
    const gate = new SelfCpuGate({ budgetCores: 1, ...opts });
    state.t = 100; // below the 500ms sampling floor
    state.cpu = 100_000;
    expect(gate.pct()).toBe(0); // no sample yet → defaults to 0 (admit)
  });

  it('clamps runaway readings to 100', () => {
    const { state, opts } = fakeEnv();
    const gate = new SelfCpuGate({ budgetCores: 1, ...opts });
    state.t = 1_000;
    state.cpu = 9_000_000; // 9s of cpu in 1s of wall (worker threads)
    expect(gate.pct()).toBe(100);
  });
});

describe('detectCgroupCpuBudgetCores', () => {
  it('returns a positive number or null without throwing on any host', () => {
    const budget = detectCgroupCpuBudgetCores();
    expect(budget === null || budget > 0).toBe(true);
  });
});
