/**
 * Function manifest validator
 *
 * Stage 1 of the function validation pipeline (parse + schema check).
 * Stages 2-5 (lint, bundle, smoke run, persist) live in the function-executor
 * service and the portal save route. This module is shared between cloud
 * (portal save handler) and localhost (`webagents fn deploy`).
 *
 * Errors are structured (`ManifestValidationError[]`) so the owner UX can
 * render inline diagnostics; nothing here ever throws on user input.
 */

import { parse as parseYaml } from 'yaml';
import {
  type CodeRef,
  type FunctionManifest,
  type FunctionPermissions,
  type FunctionLimits,
  type ManifestValidationError,
  MAX_INLINE_BYTES,
  MAX_INLINE_B64_BYTES,
  SUPPORTED_RUNTIMES,
} from './manifest';

/** Result of `validateManifest`. */
export interface ManifestValidationResult {
  ok: boolean;
  errors: ManifestValidationError[];
  /** Same manifest, normalised — defaults applied, permissions canonicalised. */
  manifest: FunctionManifest;
}

const FUNCTION_NAME_RE = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;

/**
 * Validate a parsed manifest object.
 *
 * `cloud` (default true) enables stricter checks (`file` codeRef rejected,
 * runtime allowlist enforced). Localhost `webagentsd` passes `false` to
 * accept the local-only `file` kind and the `local` folder kind.
 *
 * Note: this is a pure schema validator. The executor still re-validates
 * after parse/lint/bundle and rejects unsafe code (eval, dynamic imports).
 */
export function validateManifest(
  raw: unknown,
  opts: { cloud?: boolean; functionName?: string } = {},
): ManifestValidationResult {
  const errors: ManifestValidationError[] = [];
  const cloud = opts.cloud !== false;

  if (raw === null || typeof raw !== 'object') {
    return {
      ok: false,
      errors: [
        { code: 'INTERNAL', message: 'Manifest must be a JSON object', path: '' },
      ],
      manifest: { runtime: 'js-v1' },
    };
  }
  const m = raw as Partial<FunctionManifest> & Record<string, unknown>;

  // -- name --------------------------------------------------------------
  if (opts.functionName !== undefined && !FUNCTION_NAME_RE.test(opts.functionName)) {
    errors.push({
      code: 'INVALID_NAME',
      message: `Function name "${opts.functionName}" must match ${FUNCTION_NAME_RE.source}`,
      path: 'name',
    });
  }

  // -- runtime -----------------------------------------------------------
  if (typeof m.runtime !== 'string') {
    errors.push({ code: 'MISSING_RUNTIME', message: 'manifest.runtime is required', path: 'runtime' });
  } else if (m.runtime === 'wasm-v1') {
    errors.push({
      code: 'RUNTIME_DISABLED',
      message: 'wasm-v1 runtime slot is reserved but disabled in v1',
      path: 'runtime',
    });
  } else if (m.runtime === 'python-pyodide-v1') {
    // Pyodide deferred — see ADR-0008. The slot remains in the type
    // union so persisted manifests still load with a clear error.
    errors.push({
      code: 'RUNTIME_DISABLED',
      message:
        'python-pyodide-v1 is deferred (ADR-0008). Use js-v1 for now; ' +
        'Python support is tracked for a future v2 milestone.',
      path: 'runtime',
    });
  } else if (!SUPPORTED_RUNTIMES.includes(m.runtime as never)) {
    errors.push({
      code: 'RUNTIME_UNKNOWN',
      message: `Unknown runtime "${m.runtime}". Supported: ${SUPPORTED_RUNTIMES.join(', ')}`,
      path: 'runtime',
    });
  }

  // The manifest deliberately has no `type` field — see manifest.ts
  // (HTTP / cron / tool / WebSocket exposure is a binding concern, not
  // a function-level one). The runtime `WS_NOT_YET_SUPPORTED` error
  // code is still raised by `CustomWebsocketSkill` at the dispatcher
  // edge for any v1 attempt to bind a function as a WebSocket endpoint.

  // -- code ref ----------------------------------------------------------
  if (m.code !== undefined) {
    validateCodeRef(m.code, { cloud, errors });
  }

  // -- permissions -------------------------------------------------------
  if (m.permissions !== undefined) {
    validatePermissions(m.permissions, { cloud, errors });
  }

  // -- limits ------------------------------------------------------------
  if (m.limits !== undefined) {
    validateLimits(m.limits, errors);
  }

  // -- parameters (JSON Schema) -----------------------------------------
  if (m.parameters !== undefined) {
    if (m.parameters === null || typeof m.parameters !== 'object' || Array.isArray(m.parameters)) {
      errors.push({
        code: 'INVALID_PARAMETERS_SCHEMA',
        message: 'manifest.parameters must be a JSON Schema object',
        path: 'parameters',
      });
    } else if ((m.parameters as Record<string, unknown>).type !== 'object') {
      errors.push({
        code: 'INVALID_PARAMETERS_SCHEMA',
        message: 'manifest.parameters.type must be "object"',
        path: 'parameters.type',
      });
    }
  }

  // Normalise — fill in defaults so downstream code can rely on shape.
  const manifest: FunctionManifest = {
    runtime: (m.runtime as FunctionManifest['runtime']) ?? 'js-v1',
    entrypoint: typeof m.entrypoint === 'string' ? m.entrypoint : 'default',
    name: typeof m.name === 'string' ? m.name : opts.functionName,
    description: typeof m.description === 'string' ? m.description : undefined,
    code: m.code,
    permissions: m.permissions ?? {},
    limits: m.limits ?? {},
    bundleSha256: typeof m.bundleSha256 === 'string' ? m.bundleSha256 : undefined,
    parameters: m.parameters as Record<string, unknown> | undefined,
  };

  return { ok: errors.length === 0, errors, manifest };
}

function validateCodeRef(
  ref: CodeRef,
  ctx: { cloud: boolean; errors: ManifestValidationError[] },
): void {
  if (!ref || typeof ref !== 'object') {
    ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code is not an object', path: 'code' });
    return;
  }
  switch (ref.kind) {
    case 'content':
      if (typeof ref.contentId !== 'string' || ref.contentId.length === 0) {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.contentId is required', path: 'code.contentId' });
      }
      // sha256 is optional on content refs — the runtime computes it
      // from the loaded bytes. When the author chooses to pin a hash
      // manually we still validate the shape.
      if (ref.sha256 !== undefined && (typeof ref.sha256 !== 'string' || ref.sha256.length !== 64)) {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.sha256 must be a 64-char hex hash when provided', path: 'code.sha256' });
      }
      break;
    case 'https':
      if (!/^https:\/\//.test(ref.url)) {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.url must be https://', path: 'code.url' });
      }
      if (typeof ref.sha256 !== 'string' || ref.sha256.length !== 64) {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.sha256 must be a 64-char hex hash', path: 'code.sha256' });
      }
      break;
    case 'file':
      if (ctx.cloud) {
        ctx.errors.push({
          code: 'FILE_REF_REJECTED_IN_CLOUD',
          message: 'code.kind="file" is only allowed on localhost',
          path: 'code.kind',
        });
      } else if (typeof ref.path !== 'string' || ref.path.length === 0) {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.path is required for kind=file', path: 'code.path' });
      } else if (ref.path.includes('..')) {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.path may not contain ".."', path: 'code.path' });
      }
      break;
    case 'inline':
      if (typeof ref.source !== 'string') {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.source must be a string', path: 'code.source' });
      } else if (Buffer.byteLength(ref.source, 'utf8') > MAX_INLINE_BYTES) {
        ctx.errors.push({
          code: 'CODE_TOO_LARGE',
          message: `inline source exceeds ${MAX_INLINE_BYTES} bytes`,
          path: 'code.source',
        });
      }
      break;
    case 'inlineB64':
      if (typeof ref.source !== 'string') {
        ctx.errors.push({ code: 'INVALID_CODE_REF', message: 'code.source must be a string', path: 'code.source' });
      } else if (ref.source.length > MAX_INLINE_B64_BYTES) {
        ctx.errors.push({
          code: 'CODE_TOO_LARGE',
          message: `inlineB64 source exceeds ${MAX_INLINE_B64_BYTES} bytes`,
          path: 'code.source',
        });
      }
      break;
    default:
      ctx.errors.push({
        code: 'INVALID_CODE_REF',
        message: `Unknown code.kind "${(ref as { kind: string }).kind}"`,
        path: 'code.kind',
      });
  }
}

function validatePermissions(
  perms: FunctionPermissions,
  ctx: { cloud: boolean; errors: ManifestValidationError[] },
): void {
  if (perms.fetch !== undefined) {
    if (!Array.isArray(perms.fetch)) {
      ctx.errors.push({ code: 'INVALID_FETCH_PATTERN', message: 'permissions.fetch must be an array', path: 'permissions.fetch' });
    } else {
      for (const pat of perms.fetch) {
        if (pat === '*') continue;
        if (typeof pat !== 'string' || !/^https?:\/\//.test(pat)) {
          ctx.errors.push({
            code: 'INVALID_FETCH_PATTERN',
            message: `permissions.fetch entries must be "*" or "https://..." patterns; got ${JSON.stringify(pat)}`,
            path: 'permissions.fetch',
          });
        }
      }
    }
  }

  if (perms.secrets !== undefined && !Array.isArray(perms.secrets)) {
    ctx.errors.push({ code: 'INVALID_SECRETS', message: 'permissions.secrets must be an array of binding names', path: 'permissions.secrets' });
  }

  if (perms.kv !== undefined) {
    if (typeof perms.kv === 'string') {
      if (!['none', 'ro', 'rw'].includes(perms.kv)) {
        ctx.errors.push({
          code: 'INVALID_PERMISSIONS',
          message: 'permissions.kv must be "none" | "ro" | "rw" or an object { self?, visitor?, agent_scope? }',
          path: 'permissions.kv',
        });
      }
    } else if (typeof perms.kv === 'object' && perms.kv !== null) {
      const obj = perms.kv as { self?: unknown; visitor?: unknown; agent_scope?: unknown };
      const validMode = (v: unknown) => v === undefined || v === 'none' || v === 'ro' || v === 'rw';
      if (!validMode(obj.self)) {
        ctx.errors.push({
          code: 'INVALID_PERMISSIONS',
          message: 'permissions.kv.self must be "none" | "ro" | "rw"',
          path: 'permissions.kv.self',
        });
      }
      if (!validMode(obj.visitor)) {
        ctx.errors.push({
          code: 'INVALID_PERMISSIONS',
          message: 'permissions.kv.visitor must be "none" | "ro" | "rw"',
          path: 'permissions.kv.visitor',
        });
      }
      if (obj.agent_scope !== undefined && typeof obj.agent_scope !== 'boolean') {
        ctx.errors.push({
          code: 'INVALID_PERMISSIONS',
          message: 'permissions.kv.agent_scope must be a boolean',
          path: 'permissions.kv.agent_scope',
        });
      }
    } else {
      ctx.errors.push({
        code: 'INVALID_PERMISSIONS',
        message: 'permissions.kv must be a string or object',
        path: 'permissions.kv',
      });
    }
  }

  if (perms.visitor_profile !== undefined) {
    if (!Array.isArray(perms.visitor_profile)) {
      ctx.errors.push({
        code: 'INVALID_PERMISSIONS',
        message: 'permissions.visitor_profile must be an array of "name" | "avatar" | "email"',
        path: 'permissions.visitor_profile',
      });
    } else {
      const allowed = new Set(['name', 'avatar', 'email']);
      for (const f of perms.visitor_profile) {
        if (typeof f !== 'string' || !allowed.has(f)) {
          ctx.errors.push({
            code: 'INVALID_PERMISSIONS',
            message: `permissions.visitor_profile entries must be one of "name" | "avatar" | "email" (got ${JSON.stringify(f)})`,
            path: 'permissions.visitor_profile',
          });
        }
      }
    }
  }

  if (perms.folders !== undefined) {
    if (!Array.isArray(perms.folders)) {
      ctx.errors.push({ code: 'INVALID_PERMISSIONS', message: 'permissions.folders must be an array', path: 'permissions.folders' });
    } else {
      for (const f of perms.folders) {
        if (!f.binding || !f.kind) {
          ctx.errors.push({ code: 'INVALID_PERMISSIONS', message: 'folder binding requires both "binding" and "kind"', path: 'permissions.folders' });
        }
        if (f.kind === 'local' && ctx.cloud) {
          ctx.errors.push({
            code: 'INVALID_PERMISSIONS',
            message: 'folder.kind="local" is only allowed on localhost',
            path: 'permissions.folders',
          });
        }
      }
    }
  }
}

function validateLimits(limits: FunctionLimits, errors: ManifestValidationError[]): void {
  for (const key of ['wallMs', 'cpuMs', 'memoryMb', 'ingressBytes', 'egressBytes', 'bodyBytesMax'] as const) {
    const v = limits[key];
    if (v === undefined) continue;
    if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) {
      errors.push({
        code: 'INVALID_LIMITS',
        message: `limits.${key} must be a positive finite number`,
        path: `limits.${key}`,
      });
    }
  }
}

/**
 * Parse a single-file frontmatter block. Frontmatter is delimited by the
 * standard `/* @robutler-function ... *\/` block at the top of the file
 * (JS / TS) or `# @robutler-function: ...` lines (Python / shell-style).
 *
 * Three body formats are accepted (auto-detected, in order):
 *   1. JSON:    { "runtime": "js-v1", "entrypoint": "handler" }
 *   2. YAML:    runtime: js-v1
 *               entrypoint: handler
 *               permissions:
 *                 fetch: ["https://api.openai.com/*"]
 *   3. @key:    @runtime js-v1
 *               @entrypoint handler
 *               @permissions secrets=OPENAI_KEY,kv=ro
 *
 * YAML is the preferred ergonomic format for new functions — it nests
 * naturally and tracks the host-language-safe JSDoc wrapper. JSON and
 * @key formats stay supported so existing files keep working.
 *
 * Owners can also point to an external `manifest.json` via folder layout —
 * that path is parsed via plain `JSON.parse` and skips frontmatter logic.
 */
export function parseFrontmatter(source: string): Partial<FunctionManifest> | null {
  // Match both `/* @robutler-function ... */` and JSDoc `/** @robutler-function ... */`.
  const re = /\/\*[\s*]*@robutler-function\b([\s\S]*?)\*\//m;
  const match = re.exec(source);
  if (!match) return null;
  const block = match[1].trim();

  // Strip leading `*` per line and reassemble JSON
  const text = block
    .split('\n')
    .map((line) => line.replace(/^\s*\*\s?/, ''))
    .join('\n')
    .trim();

  if (text.length === 0) return {};

  // 1. JSON.
  if (text.startsWith('{')) {
    try {
      return JSON.parse(text) as Partial<FunctionManifest>;
    } catch {
      return null;
    }
  }

  // 2. @key value (legacy). The simple line-prefix check is enough — YAML
  // fields like `runtime: js-v1` don't start with `@`.
  if (/^@\w+/m.test(text.split('\n').find((l) => l.trim().length > 0) ?? '')) {
    const manifest: Record<string, unknown> = {};
    const lineRe = /^@(\w+)\s*(.*)$/;
    for (const line of text.split('\n')) {
      const lm = lineRe.exec(line.trim());
      if (!lm) continue;
      const [, key, value] = lm;
      const v = value.trim();
      if (!v) continue;
      if (key === 'permissions') {
        manifest.permissions = parseKvPairs(v);
      } else if (key === 'limits') {
        manifest.limits = parseKvPairs(v, true);
      } else {
        manifest[key] = v;
      }
    }
    return manifest as Partial<FunctionManifest>;
  }

  // 3. YAML.
  try {
    const parsed = parseYaml(text);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    return parsed as Partial<FunctionManifest>;
  } catch {
    return null;
  }
}

/**
 * Strip a `@robutler-function` frontmatter block from JS / TS source. Useful
 * for callers that need the manifest separately (`parseFrontmatter`) and the
 * remaining function body (`stripFrontmatter`).
 */
export function stripFrontmatter(source: string): string {
  return source.replace(/\/\*[\s*]*@robutler-function\b[\s\S]*?\*\//m, '').trim();
}

function parseKvPairs(s: string, numeric = false): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const pair of s.split(',')) {
    const [k, vRaw] = pair.split('=');
    if (!k || vRaw === undefined) continue;
    const v = vRaw.trim();
    if (numeric) {
      const n = Number(v);
      out[k.trim()] = Number.isFinite(n) ? n : v;
    } else if (v.includes(';')) {
      out[k.trim()] = v.split(';').map((x) => x.trim()).filter(Boolean);
    } else {
      out[k.trim()] = v;
    }
  }
  return out;
}
