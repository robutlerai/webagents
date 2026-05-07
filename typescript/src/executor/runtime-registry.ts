/**
 * RuntimeRegistry — id → runtime dispatch table.
 *
 * Runtimes register themselves at module load (or are registered by the
 * executor entrypoint). Disabled runtimes are still listed (so manifest
 * validation can return `RUNTIME_DISABLED` instead of `RUNTIME_UNKNOWN`)
 * but `prepare` / `validate` reject up front.
 */

import type { ExecutorRuntime, ExecutorRuntimeId } from './types';

const REGISTRY = new Map<ExecutorRuntimeId, ExecutorRuntime>();

export function registerRuntime(rt: ExecutorRuntime): void {
  REGISTRY.set(rt.id, rt);
}

export const RuntimeRegistry = {
  get(id: ExecutorRuntimeId): ExecutorRuntime | undefined {
    return REGISTRY.get(id);
  },
  has(id: ExecutorRuntimeId): boolean {
    return REGISTRY.has(id);
  },
  list(): ExecutorRuntimeId[] {
    return [...REGISTRY.keys()];
  },
  enabled(): ExecutorRuntimeId[] {
    return [...REGISTRY.values()].filter((r) => r.enabled).map((r) => r.id);
  },
};
