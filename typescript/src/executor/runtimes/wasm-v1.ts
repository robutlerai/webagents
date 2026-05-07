/**
 * `wasm-v1` runtime — reserved slot, ships disabled.
 *
 * The protocol/manifest/validator/executor dispatch all know about this
 * id, but `enabled` is hard-false. When a function declares
 * `runtime: 'wasm-v1'` the validator returns `RUNTIME_DISABLED` and the
 * Functions pane surfaces an upgrade hint.
 *
 * Compile-from-source toolchains (QuickJS-WASM, MicroPython-WASM,
 * Rust/AssemblyScript/TinyGo) ship in a separate webagents build
 * workstream — this slot exists so we don't need to ship a SDK
 * upgrade when that lands.
 */

import type {
  ExecutorRuntime,
  RuntimeSandbox,
  ExecutorRuntimeId,
} from '../types';
import type {
  ExecutorValidationResult,
} from '../../skills/functions/executor-client';
import type { FunctionManifest } from '../../skills/functions/manifest';

export class WasmV1Runtime implements ExecutorRuntime {
  readonly id: ExecutorRuntimeId = 'wasm-v1';
  readonly enabled = false;

  async prepare(_source: string, _manifest: FunctionManifest): Promise<RuntimeSandbox> {
    throw new Error('wasm-v1 runtime is reserved but not yet enabled');
  }

  async validate(_source: string, _manifest: FunctionManifest): Promise<ExecutorValidationResult> {
    return {
      ok: false,
      warnings: [],
      errors: [
        {
          code: 'RUNTIME_DISABLED',
          message: 'wasm-v1 is reserved but not enabled in v1; declare js-v1 or python-pyodide-v1',
        },
      ],
    };
  }
}
