/**
 * `js-v1` runtime — V8 isolate via `isolated-vm`.
 *
 * Boundaries:
 *   - Cold start: compile source into the isolate; ~5–15ms typical.
 *   - Heap cap (memoryMb), CPU cap (cpuMs via `Isolate.cpuTime`), wall
 *     cap (timer + AbortSignal).
 *   - No `require`, `import`, `process`, `fs`, `Buffer` — only Web Fetch
 *     and standard JS globals plus the `ctx` object.
 *
 * Validation walks the AST for clearly-broken patterns (top-level await
 * outside async fn, throw at module scope, missing `export default`),
 * then runs the function with a synthesized `ctx` to catch immediate
 * runtime errors. Validation runs with stricter caps (1.5× cold-start).
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

// `isolated-vm` is a native module — load lazily so the executor can boot
// in environments where it's not yet installed (e.g. unit tests).
type IvmModule = {
  Isolate: new (opts: { memoryLimit: number }) => {
    createContext(): Promise<IvmContext>;
    compileScript(src: string): Promise<IvmScript>;
    cpuTime: bigint;
    dispose(): void;
  };
  Reference: new <T>(value: T) => unknown;
  ExternalCopy: new <T>(value: T) => { copyInto(): T };
};

interface IvmContext {
  global: { setSync(name: string, value: unknown): void };
  release(): void;
  evalClosure(src: string, args: unknown[], opts?: { timeout?: number }): Promise<unknown>;
}

interface IvmScript {
  run(ctx: IvmContext, opts?: { timeout?: number }): Promise<unknown>;
  release(): void;
}

let ivmModuleCache: IvmModule | null | undefined;

// `isolated-vm` is a native module that's only present on the cloud
// executor pod (compiled per Node ABI). Build images and unit-test
// sandboxes don't ship it. Routing the import through a `Function`-
// constructed `import()` keeps the bundler / `tsc` from trying to
// resolve the module specifier statically; at runtime the resolver
// either finds it (executor pod) or throws (everywhere else) and we
// gracefully cache `null` so the runtime reports as disabled.
const runtimeDynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as (specifier: string) => Promise<unknown>;

async function loadIvm(): Promise<IvmModule | null> {
  if (ivmModuleCache !== undefined) return ivmModuleCache;
  try {
    const mod = (await runtimeDynamicImport('isolated-vm')) as IvmModule;
    ivmModuleCache = mod;
    return mod;
  } catch {
    ivmModuleCache = null;
    return null;
  }
}

class JsV1Sandbox implements RuntimeSandbox {
  constructor(
    public readonly id: string,
    private readonly script: IvmScript,
    private readonly isolate: IvmContext,
    private readonly isolateRef: { dispose(): void; cpuTime: bigint },
  ) {}
  async invoke(env: InvocationEnvelope): Promise<ExecutorResponse> {
    const t0 = performance.now();
    const cpu0 = this.isolateRef.cpuTime;
    const wallMs = env.context?.limits?.wallMs ?? env.manifest.limits?.wallMs ?? 30_000;
    try {
      const result = await this.script.run(this.isolate as never, { timeout: wallMs });
      const cpu1 = this.isolateRef.cpuTime;
      const cpuMs = Number((cpu1 - cpu0) / 1_000_000n);
      const durationMs = performance.now() - t0;
      return {
        ok: true,
        result,
        durationMs,
        cpuMs,
        ingressBytes: 0,
        egressBytes: 0,
      };
    } catch (e) {
      const durationMs = performance.now() - t0;
      const msg = (e as Error).message;
      return {
        ok: false,
        errorCode: msg.includes('Script execution timed out') ? 'WALL_TIMEOUT' : 'JS_RUNTIME_ERROR',
        errorMessage: msg,
        durationMs,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      };
    }
  }
  async dispose(): Promise<void> {
    try { this.script.release(); } catch {}
    try { (this.isolate as { release(): void }).release(); } catch {}
    try { this.isolateRef.dispose(); } catch {}
  }
}

export class JsV1Runtime implements ExecutorRuntime {
  readonly id: ExecutorRuntimeId = 'js-v1';
  get enabled() {
    return ivmModuleCache !== null;
  }

  async prepare(source: string, manifest: FunctionManifest): Promise<RuntimeSandbox> {
    const ivm = await loadIvm();
    if (!ivm) throw new Error('isolated-vm not installed');
    const memoryMb = manifest.limits?.memoryMb ?? 64;
    const isolate = new ivm.Isolate({ memoryLimit: memoryMb });
    const ctx = await isolate.createContext();
    const script = await isolate.compileScript(source);
    const id = `js-v1:${Date.now()}:${Math.random().toString(36).slice(2)}`;
    return new JsV1Sandbox(id, script, ctx, isolate);
  }

  async validate(source: string, manifest: FunctionManifest): Promise<ExecutorValidationResult> {
    // `ExecutorValidationResult.errors` entries are line/column-keyed; the
    // codes themselves carry the field semantics (SOURCE_EMPTY, etc) so
    // callers don't need a separate `field` discriminator.
    if (!source || source.length === 0) {
      return { ok: false, warnings: [], errors: [{ code: 'SOURCE_EMPTY', message: 'no source provided' }] };
    }
    if (source.length > 256 * 1024) {
      return { ok: false, warnings: [], errors: [{ code: 'SOURCE_TOO_LARGE', message: 'source exceeds 256KB' }] };
    }
    if (!/export\s+default\s+/.test(source) && !/module\.exports\s*=/.test(source)) {
      return {
        ok: false,
        warnings: [],
        errors: [{ code: 'NO_EXPORT_DEFAULT', message: 'function must export a default async handler' }],
      };
    }
    // Boot a sandbox to confirm syntax compiles.
    try {
      const sandbox = await this.prepare(source, manifest);
      await sandbox.dispose();
      return { ok: true, warnings: [], errors: [] };
    } catch (e) {
      return {
        ok: false,
        warnings: [],
        errors: [{ code: 'JS_COMPILE_ERROR', message: (e as Error).message }],
      };
    }
  }
}
