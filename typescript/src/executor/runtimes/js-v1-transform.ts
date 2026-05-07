/**
 * Source transformer for the js-v1 runtime.
 *
 * User code may export a handler in any of these shapes:
 *
 *   1. ESM default:    `export default async function (ctx) { ... }`
 *   2. ESM named:      `export async function handler(ctx) { ... }`
 *   3. CJS:            `module.exports = async function (ctx) { ... }`
 *   4. CJS named:      `module.exports.handler = async function (ctx) { ... }`
 *
 * The isolate (V8 only, no Node) runs the source as classic script — no
 * ESM module loader, no CommonJS `require`. The transform rewrites
 * exports into a single global function `__handler` that the runtime
 * invokes via `evalClosure`.
 *
 * Resolution order when no `manifest.entrypoint` is given: default
 * export → `handler` → first named function export → throws
 * `JS_NO_HANDLER`.
 *
 * When the manifest names `entrypoint: "handler"` but the source only
 * has a default export (anonymous or mis-scaffolded), we still resolve
 * to `default`. Tooling often pins `handler` as the conventional name
 * while authors write `export default async function (ctx) { … }`.
 *
 * Pure: no isolated-vm dependency, fully unit-testable.
 */

export interface TransformResult {
  /** Source ready to be compiled inside an isolate; ends with `globalThis.__handler = …`. */
  source: string;
  /** Resolved entrypoint name; `'default'` for the ESM default export. */
  resolvedEntrypoint: string;
}

export class TransformError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

const ESM_DEFAULT_RX =
  /export\s+default\s+(async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)?\s*\(/;
const ESM_DEFAULT_ARROW_RX = /export\s+default\s+(async\s*)?\(/;
const ESM_NAMED_RX =
  /export\s+(async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(/g;
const ESM_NAMED_CONST_RX =
  /export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/g;
const CJS_DEFAULT_RX = /module\.exports\s*=\s*/;
const CJS_NAMED_RX =
  /module\.exports\.([A-Za-z_$][\w$]*)\s*=\s*|exports\.([A-Za-z_$][\w$]*)\s*=\s*/g;

/**
 * Detect handler exports in `source`. Returns the resolution table the
 * isolate-side bootstrap will consult. Used by `transformSource` and
 * the validator's pre-flight checks.
 */
export function detectExports(source: string): {
  hasEsmDefault: boolean;
  esmNamed: Set<string>;
  hasCjsDefault: boolean;
  cjsNamed: Set<string>;
} {
  const esmNamed = new Set<string>();
  for (const m of source.matchAll(ESM_NAMED_RX)) {
    if (m[2]) esmNamed.add(m[2]);
  }
  for (const m of source.matchAll(ESM_NAMED_CONST_RX)) {
    if (m[1]) esmNamed.add(m[1]);
  }
  const cjsNamed = new Set<string>();
  for (const m of source.matchAll(CJS_NAMED_RX)) {
    const n = m[1] ?? m[2];
    if (n) cjsNamed.add(n);
  }
  return {
    hasEsmDefault: ESM_DEFAULT_RX.test(source) || ESM_DEFAULT_ARROW_RX.test(source),
    esmNamed,
    hasCjsDefault: CJS_DEFAULT_RX.test(source),
    cjsNamed,
  };
}

/**
 * Rewrite `source` so it can run as a classic script in a bare V8
 * isolate. The output:
 *
 *   - Replaces ESM `export default` / `export const X` / `export async
 *     function X` with plain declarations.
 *   - Replaces CJS `module.exports = ...` with a stub object.
 *   - Appends `globalThis.__handler = <resolved entry>;` so the runtime
 *     can pick it up via `evalClosure('return __handler(...args)')`.
 *
 * `entrypointHint` (manifest) takes precedence; otherwise the resolution
 * order is default → `handler` → first named.
 */
export function transformSource(
  source: string,
  entrypointHint?: string,
): TransformResult {
  const exports = detectExports(source);

  // Resolve entrypoint up front so we can fail fast.
  const wantsName =
    entrypointHint && entrypointHint !== 'default' ? entrypointHint : null;
  let resolved: string;
  if (wantsName) {
    const named =
      exports.esmNamed.has(wantsName) || exports.cjsNamed.has(wantsName);
    if (!named) {
      // Conventional alias: manifest says `handler` but the author only
      // provided a default export — treat as default so scaffolds stay valid.
      if (
        wantsName === 'handler' &&
        (exports.hasEsmDefault || exports.hasCjsDefault)
      ) {
        resolved = 'default';
      } else {
        throw new TransformError(
          'JS_NO_HANDLER',
          `entrypoint "${wantsName}" not exported`,
        );
      }
    } else {
      resolved = wantsName;
    }
  } else if (exports.hasEsmDefault || exports.hasCjsDefault) {
    resolved = 'default';
  } else if (exports.esmNamed.has('handler') || exports.cjsNamed.has('handler')) {
    resolved = 'handler';
  } else if (exports.esmNamed.size > 0) {
    resolved = exports.esmNamed.values().next().value as string;
  } else if (exports.cjsNamed.size > 0) {
    resolved = exports.cjsNamed.values().next().value as string;
  } else {
    throw new TransformError(
      'JS_NO_HANDLER',
      'no handler export detected (need export default, export function, or module.exports)',
    );
  }

  // ---------------------------------------------------------------------
  // Rewrite. We do simple textual rewrites — the isolate has no module
  // loader, so we just need the source to evaluate as a script and stash
  // the handler on `globalThis.__handler`.
  // ---------------------------------------------------------------------
  let body = source;

  // CJS: provide a stub `module.exports` object visible to the user code.
  const cjsPrelude = `var module = { exports: {} }; var exports = module.exports;\n`;
  if (exports.hasCjsDefault || exports.cjsNamed.size > 0) {
    body = cjsPrelude + body;
  }

  // ESM `export default` → assign to global slot.
  body = body.replace(/export\s+default\s+/, 'globalThis.__esmDefault = ');

  // ESM `export const X = ...` → leave the binding, drop the keyword.
  body = body.replace(/export\s+(const|let|var)\s+/g, '$1 ');

  // ESM `export async function X` / `export function X` → drop keyword.
  body = body.replace(
    /export\s+(async\s+)?function(\s+)/g,
    '$1function$2',
  );

  // ESM `export { X, Y as Z };` → drop entirely (we use globals).
  body = body.replace(/export\s*\{[^}]*\}\s*;?/g, '');

  // Resolution suffix.
  let suffix: string;
  if (resolved === 'default') {
    suffix =
      '\nglobalThis.__handler = (typeof globalThis.__esmDefault === "function") ' +
      '? globalThis.__esmDefault ' +
      ': (typeof module !== "undefined" && typeof module.exports === "function") ' +
      '? module.exports ' +
      ': (typeof module !== "undefined" && module.exports && typeof module.exports.default === "function") ' +
      '? module.exports.default ' +
      ': null;\n';
  } else {
    suffix =
      `\nglobalThis.__handler = (typeof ${resolved} === "function") ` +
      `? ${resolved} ` +
      `: (typeof module !== "undefined" && module.exports && typeof module.exports.${resolved} === "function") ` +
      `? module.exports.${resolved} ` +
      `: null;\n`;
  }
  body += suffix;
  body +=
    'if (typeof globalThis.__handler !== "function") { ' +
    'throw new Error("JS_NO_HANDLER: handler is not a function"); }\n';

  return { source: body, resolvedEntrypoint: resolved };
}
