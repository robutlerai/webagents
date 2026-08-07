/**
 * Per-run context store — the TypeScript counterpart of the Python SDK's
 * `ContextVar` (webagents/python/webagents/server/context/context_vars.py).
 *
 * The Python SDK creates a FRESH context per request/run and binds it to a
 * `ContextVar`, so concurrent invocations of one agent never see each
 * other's identity, chat binding or payment token. The TypeScript SDK kept
 * a single `context` field on the Agent instance instead — which is fine
 * for one-conversation-at-a-time, but the portal serves one cached agent
 * instance to N concurrent callers, so runs overwrote each other's values
 * (diagnosed 2026-08-06 from S5: one chat id rejected for several different
 * payers). This module closes that gap; `AsyncLocalStorage` is the direct
 * Node equivalent of `ContextVar`.
 *
 * WHY THE INDIRECTION: `AsyncLocalStorage` lives in `node:async_hooks`,
 * which does not exist in browsers — and `core/agent.ts` is imported by the
 * browser extension (`src/extension/background/agent-runtime.ts`). A STATIC
 * `node:` import would therefore land in that bundle. So the constructor is
 * resolved through a dynamic import, started at module load, and every
 * store degrades to a pass-through when it is unavailable. Pass-through
 * behaviour is exactly the pre-2026-08-06 semantics (single shared
 * context), which is correct for a single-user browser agent.
 */

export interface RunContextStore<T> {
  /** The value bound to the current async execution, if any. */
  getStore(): T | undefined;
  /** Run `fn` with `value` bound for its entire async subtree. */
  run<R>(value: T, fn: () => R): R;
}

type AlsCtor = new <T>() => RunContextStore<T>;

let alsCtor: AlsCtor | null = null;

/**
 * Begin resolving AsyncLocalStorage. Started at MODULE load, and the
 * promise is exported via `whenRunContextReady()` so the first run can
 * AWAIT it — without that, runs issued in the first few ticks of process
 * life would silently fall back to the shared-context behaviour this
 * module exists to remove.
 */
const readyPromise: Promise<void> = (() => {
  const g = globalThis as { process?: { versions?: { node?: string } } };
  if (!g.process?.versions?.node) return Promise.resolve();
  // Dynamic (not static) so bundlers targeting the browser are unaffected.
  return import('node:async_hooks')
    .then((m) => {
      alsCtor = m.AsyncLocalStorage as unknown as AlsCtor;
    })
    .catch(() => {
      alsCtor = null;
    });
})();

/**
 * Resolves once the async-context implementation has been resolved (or
 * definitively ruled out). Await this before the first `run()` so isolation
 * is active from the very first invocation.
 */
export function whenRunContextReady(): Promise<void> {
  return readyPromise;
}

/**
 * Create a store for per-run values. The underlying AsyncLocalStorage is
 * instantiated lazily on first use, so a store created before the dynamic
 * import settles still becomes isolating once it does.
 */
export function createRunContextStore<T>(): RunContextStore<T> {
  let real: RunContextStore<T> | null = null;

  const resolve = (): RunContextStore<T> | null => {
    if (!real && alsCtor) real = new alsCtor<T>();
    return real;
  };

  return {
    getStore: () => resolve()?.getStore(),
    run: (value, fn) => {
      const store = resolve();
      return store ? store.run(value, fn) : fn();
    },
  };
}

/** True when real async-context isolation is active (diagnostics/tests). */
export function runContextIsolationAvailable(): boolean {
  return alsCtor !== null;
}
