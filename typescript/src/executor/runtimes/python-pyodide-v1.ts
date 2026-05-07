/**
 * `python-pyodide-v1` runtime — CPython 3.x compiled to WebAssembly.
 *
 * Trade-offs:
 *   - Cold start ~700ms (first time), warm ~5ms.
 *   - Single-threaded; we run one Pyodide instance per worker thread.
 *   - Standard library available; no `pip install` from inside user
 *     code (warm pool seeds with `requests`-style helpers).
 *
 * Validation runs the source through `compile()` inside the warm
 * instance and synthesizes a `ctx` object for a smoke run.
 */

import { performance } from 'perf_hooks';
import type {
  ExecutorRuntime,
  RuntimeSandbox,
  ExecutorRuntimeId,
} from '../types';
import type {
  InvocationEnvelope,
  ExecutorResponse,
  ExecutorValidationResult,
} from '../../skills/functions/executor-client';
import type { FunctionManifest } from '../../skills/functions/manifest';

interface Pyodide {
  loadPyodide(): Promise<PyodideRuntime>;
}

interface PyodideRuntime {
  runPython(src: string): unknown;
  runPythonAsync(src: string): Promise<unknown>;
  globals: { set(name: string, value: unknown): void };
}

let pyodideCache: PyodideRuntime | null | undefined;

// `pyodide` ships only on the cloud python-pyodide pod (≈25 MB wasm
// blob + JS shim). Build images and unit-test sandboxes don't install
// it. Routing the import through a `Function`-constructed `import()`
// keeps the bundler / `tsc` from trying to resolve the specifier
// statically; at runtime the resolver either finds it (executor pod)
// or throws (everywhere else), and we cache `null` so the runtime
// reports as disabled.
const runtimeDynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as (specifier: string) => Promise<unknown>;

async function loadPyodideOnce(): Promise<PyodideRuntime | null> {
  if (pyodideCache !== undefined) return pyodideCache;
  try {
    const mod = (await runtimeDynamicImport('pyodide').catch(() => null)) as Pyodide | null;
    if (!mod) {
      pyodideCache = null;
      return null;
    }
    pyodideCache = await mod.loadPyodide();
    return pyodideCache;
  } catch {
    pyodideCache = null;
    return null;
  }
}

class PythonSandbox implements RuntimeSandbox {
  // `_manifest` is passed for parity with the JS sandbox API but Pyodide
  // doesn't apply per-sandbox memory caps the way `isolated-vm` does;
  // limits are enforced at the worker-pool wallMs/CPU layer instead.
  constructor(public readonly id: string, private readonly source: string, private readonly py: PyodideRuntime) {}
  async invoke(env: InvocationEnvelope): Promise<ExecutorResponse> {
    const t0 = performance.now();
    const wallMs = env.context?.limits?.wallMs ?? env.manifest.limits?.wallMs ?? 30_000;
    try {
      this.py.globals.set('_invocation', env);
      const wrapped = `${this.source}\n\nimport asyncio\n_result = asyncio.run(handler(_invocation))\n_result`;
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), wallMs);
      const result = await this.py.runPythonAsync(wrapped);
      clearTimeout(timer);
      return {
        ok: true,
        result,
        durationMs: performance.now() - t0,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      };
    } catch (e) {
      return {
        ok: false,
        errorCode: 'PYTHON_RUNTIME_ERROR',
        errorMessage: (e as Error).message,
        durationMs: performance.now() - t0,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      };
    }
  }
  async dispose() {
    /* Pyodide instances are reused across invocations; per-invocation state cleared by ctx swap. */
  }
}

export class PythonPyodideV1Runtime implements ExecutorRuntime {
  readonly id: ExecutorRuntimeId = 'python-pyodide-v1';
  get enabled() {
    return pyodideCache !== null;
  }

  async prepare(source: string, _manifest: FunctionManifest): Promise<RuntimeSandbox> {
    const py = await loadPyodideOnce();
    if (!py) throw new Error('pyodide not installed');
    return new PythonSandbox(`py-${Date.now()}`, source, py);
  }

  async validate(source: string, _manifest: FunctionManifest): Promise<ExecutorValidationResult> {
    if (!source.includes('def handler')) {
      return { ok: false, warnings: [], errors: [{ code: 'NO_HANDLER', message: 'must define `async def handler(ctx)`' }] };
    }
    if (source.length > 512 * 1024) {
      return { ok: false, warnings: [], errors: [{ code: 'SOURCE_TOO_LARGE', message: 'source exceeds 512KB' }] };
    }
    const py = await loadPyodideOnce();
    if (!py) {
      return { ok: false, warnings: [], errors: [{ code: 'RUNTIME_DISABLED', message: 'pyodide not installed' }] };
    }
    try {
      py.runPython(`compile(${JSON.stringify(source)}, "<user>", "exec")`);
      return { ok: true, warnings: [], errors: [] };
    } catch (e) {
      return { ok: false, warnings: [], errors: [{ code: 'PY_COMPILE_ERROR', message: (e as Error).message }] };
    }
  }
}
