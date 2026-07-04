/**
 * `js-v1` isolate bootstrap.
 *
 * Two pieces, both consumed by `js-v1.ts`:
 *
 *  - `ISOLATE_BOOTSTRAP` — runs ONCE at sandbox prepare time. Installs
 *    Web Platform globals (console, crypto, structuredClone, Headers,
 *    Request, Response, AbortController/AbortSignal/EventTarget,
 *    URLPattern) and blocks the dangerous Node ambient bindings
 *    (`process`, `Buffer`, `require`, `eval`, `Function` constructor).
 *
 *    The pure-JS classes are deliberately small but spec-faithful for
 *    the surface user functions reach for: `new Headers([['x','y']])`,
 *    `new Request('https://...', { method, body })`, `Response.json(o)`,
 *    `new AbortController().signal.aborted`, etc. They do NOT talk to
 *    undici directly — `ctx.fetch` accepts plain `{url, init}` shapes
 *    and returns `{status, statusText, headers, bodyBytes}` envelopes
 *    that the ctx-builder converts back into `Response`-like objects.
 *
 *  - `CTX_BUILDER_SOURCE` — runs ONCE per invocation, after the user
 *    source has set `globalThis.__handler`. Wires `ctx.{fetch, secrets,
 *    kv, content, folders, fn, portal, log, emit}` into a single
 *    `globalThis.__ctx` object that the handler invocation script
 *    consumes.
 *
 * Both strings reference host-installed `__hostXxx` references that
 * `js-v1.ts:installCoreGlobals` plants on the isolate's global object
 * before this script runs.
 */

export const ISOLATE_BOOTSTRAP = `
"use strict";

// ---- Block dangerous Node globals that may leak via ambient bindings ----
// (None should be present in a bare V8 isolate, but defensive deletes
// keep that contract explicit.)
delete globalThis.process;
delete globalThis.Buffer;
delete globalThis.require;
delete globalThis.module;
delete globalThis.__dirname;
delete globalThis.__filename;
// Block eval / Function constructors at the global slot. User code can
// still re-bind locally; that's their problem.
globalThis.eval = function () { throw new Error('EVAL_DENIED: eval is disabled in js-v1'); };
globalThis.Function = function () { throw new Error('FUNCTION_DENIED: Function constructor is disabled in js-v1'); };

// ---- console ----
globalThis.console = {
  debug: (...a) => __hostConsoleDebug.applyIgnored(undefined, [a.length === 1 ? a[0] : a]),
  log:   (...a) => __hostConsoleInfo.applyIgnored(undefined, [a.length === 1 ? a[0] : a]),
  info:  (...a) => __hostConsoleInfo.applyIgnored(undefined, [a.length === 1 ? a[0] : a]),
  warn:  (...a) => __hostConsoleWarn.applyIgnored(undefined, [a.length === 1 ? a[0] : a]),
  error: (...a) => __hostConsoleError.applyIgnored(undefined, [a.length === 1 ? a[0] : a]),
};

// ---- TextEncoder / TextDecoder ----
// V8 ships these under the global slot in modern builds; defensive
// fallback uses URL-encoding which we don't want, so just check.
if (typeof TextEncoder === 'undefined') {
  globalThis.TextEncoder = class TextEncoder {
    encode(s) {
      const out = [];
      for (let i = 0; i < s.length; i++) {
        let c = s.charCodeAt(i);
        if (c < 0x80) out.push(c);
        else if (c < 0x800) { out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f)); }
        else if (c < 0xd800 || c >= 0xe000) { out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f)); }
        else {
          c = 0x10000 + (((c & 0x3ff) << 10) | (s.charCodeAt(++i) & 0x3ff));
          out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 0x3f), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
        }
      }
      return new Uint8Array(out);
    }
  };
}
if (typeof TextDecoder === 'undefined') {
  // Minimal UTF-8 TextDecoder. The pure-JS Response/Request body
  // mixin in this same bootstrap depends on it (\`text()\` → decode),
  // and V8 isolates do NOT carry the native one — without this
  // polyfill, every \`await response.text()\` from user code crashes
  // with "TextDecoder is not defined". Spec-faithful enough for
  // \`response.json()\` and string body handling; non-UTF-8 labels
  // and BOM stripping are intentionally out of scope (\`label\` is
  // ignored — the default 'utf-8' is the only encoding ever
  // produced by host fetches and the bootstrap's own \`__makeBody\`).
  globalThis.TextDecoder = class TextDecoder {
    constructor(label) { this.encoding = (label || 'utf-8').toString().toLowerCase(); }
    decode(input) {
      if (input == null) return '';
      const bytes = input instanceof Uint8Array
        ? input
        : input.buffer ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
        : new Uint8Array(input);
      let result = '';
      let i = 0;
      while (i < bytes.length) {
        const b1 = bytes[i++];
        if (b1 < 0x80) {
          result += String.fromCharCode(b1);
        } else if (b1 < 0xc0) {
          // Stray continuation byte — emit U+FFFD per spec.
          result += '\\uFFFD';
        } else if (b1 < 0xe0) {
          const b2 = bytes[i++] || 0;
          result += String.fromCharCode(((b1 & 0x1f) << 6) | (b2 & 0x3f));
        } else if (b1 < 0xf0) {
          const b2 = bytes[i++] || 0;
          const b3 = bytes[i++] || 0;
          result += String.fromCharCode(((b1 & 0x0f) << 12) | ((b2 & 0x3f) << 6) | (b3 & 0x3f));
        } else {
          const b2 = bytes[i++] || 0;
          const b3 = bytes[i++] || 0;
          const b4 = bytes[i++] || 0;
          let cp = ((b1 & 0x07) << 18) | ((b2 & 0x3f) << 12) | ((b3 & 0x3f) << 6) | (b4 & 0x3f);
          cp -= 0x10000;
          result += String.fromCharCode(0xd800 | (cp >> 10), 0xdc00 | (cp & 0x3ff));
        }
      }
      return result;
    }
  };
}

// ---- crypto ----
// ---- atob / btoa ----
// Pure V8 has no atob/btoa (they're Web APIs, not JS builtins) — yet the
// ctx.content / ctx.folders wrappers below depend on them for binary
// payloads, so every base64 body silently threw ReferenceError until these
// polyfills landed. Chunk-free, allocation-light, standard alphabet.
const __B64C = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
if (typeof globalThis.atob !== 'function') {
  globalThis.atob = function (input) {
    const str = String(input).replace(/=+$/, '');
    if (/[^A-Za-z0-9+/]/.test(str)) throw new Error('atob: invalid base64');
    let out = '';
    for (let i = 0; i < str.length; i += 4) {
      const len = Math.min(4, str.length - i);
      let n = 0;
      for (let j = 0; j < 4; j++) n = n * 64 + (j < len ? __B64C.indexOf(str[i + j]) : 0);
      const take = len >= 4 ? 3 : len - 1;
      for (let j = 0; j < take; j++) out += String.fromCharCode((n >> (8 * (2 - j))) & 0xff);
    }
    return out;
  };
}
if (typeof globalThis.btoa !== 'function') {
  globalThis.btoa = function (input) {
    const str = String(input);
    let out = '';
    for (let i = 0; i < str.length; i += 3) {
      const c0 = str.charCodeAt(i);
      const c1 = i + 1 < str.length ? str.charCodeAt(i + 1) : NaN;
      const c2 = i + 2 < str.length ? str.charCodeAt(i + 2) : NaN;
      if (c0 > 255 || c1 > 255 || c2 > 255) throw new Error('btoa: character out of range');
      const n = (c0 << 16) | ((c1 || 0) << 8) | (c2 || 0);
      out += __B64C[(n >> 18) & 63] + __B64C[(n >> 12) & 63]
        + (Number.isNaN(c1) ? '=' : __B64C[(n >> 6) & 63])
        + (Number.isNaN(c2) ? '=' : __B64C[n & 63]);
    }
    return out;
  };
}
/** Uint8Array/ArrayBuffer → base64 WITHOUT spreading into arguments — a
 *  multi-MB String.fromCharCode(...bytes) spread overflows the call stack. */
function __bytesToB64(data) {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  let bin = '';
  for (let i = 0; i < u8.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

globalThis.crypto = {
  randomUUID: () => __hostRandomUUID.applySync(undefined, []),
  getRandomValues: (target) => {
    const buf = __hostGetRandomValues.applySync(undefined, [target.byteLength], { result: { copy: true } });
    new Uint8Array(target.buffer, target.byteOffset, target.byteLength).set(new Uint8Array(buf));
    return target;
  },
  subtle: new Proxy({}, {
    get: (_t, name) => {
      if (typeof name !== 'string') return undefined;
      return async (...args) => {
        const op = name;
        let payload;
        switch (op) {
          case 'digest':     payload = { algorithm: args[0], data: args[1] }; break;
          case 'importKey':  payload = { format: args[0], keyData: args[1], algorithm: args[2], extractable: args[3], keyUsages: args[4] }; break;
          case 'sign':       payload = { algorithm: args[0], key: args[1], data: args[2] }; break;
          case 'verify':     payload = { algorithm: args[0], key: args[1], signature: args[2], data: args[3] }; break;
          case 'encrypt':
          case 'decrypt':    payload = { algorithm: args[0], key: args[1], data: args[2] }; break;
          default: throw new Error('UNSUPPORTED_SUBTLE_OP: ' + op);
        }
        return await __hostSubtle.apply(undefined, [op, payload], {
          arguments: { copy: true },
          result: { copy: true, promise: true },
        });
      };
    }
  }),
};

// ---- structuredClone ----
globalThis.structuredClone = function (value) {
  return __hostStructuredClone.applySync(undefined, [value], { arguments: { copy: true }, result: { copy: true } });
};

// ---- EventTarget / Event ----
// Minimal spec-faithful implementation. Listeners are invoked with the
// event object; \`removeEventListener\` matches by callback identity +
// (capture flag, when used). Stop / preventDefault are supported but
// AbortSignal is the only Event subclass we ship.
globalThis.Event = class Event {
  constructor(type, init) {
    this.type = String(type);
    this.bubbles = !!(init && init.bubbles);
    this.cancelable = !!(init && init.cancelable);
    this.defaultPrevented = false;
    this.target = null;
    this.currentTarget = null;
  }
  preventDefault() { if (this.cancelable) this.defaultPrevented = true; }
  stopPropagation() { /* single-target; no propagation graph */ }
  stopImmediatePropagation() { this.__stopped = true; }
};

globalThis.EventTarget = class EventTarget {
  constructor() {
    this.__listeners = Object.create(null);
  }
  addEventListener(type, listener, options) {
    if (typeof listener !== 'function' && (typeof listener !== 'object' || listener === null)) return;
    const list = this.__listeners[type] || (this.__listeners[type] = []);
    const opts = typeof options === 'boolean' ? { capture: options } : (options || {});
    if (list.some((e) => e.listener === listener && !!e.capture === !!opts.capture)) return;
    list.push({ listener, capture: !!opts.capture, once: !!opts.once, passive: !!opts.passive });
  }
  removeEventListener(type, listener, options) {
    const list = this.__listeners[type];
    if (!list) return;
    const opts = typeof options === 'boolean' ? { capture: options } : (options || {});
    const i = list.findIndex((e) => e.listener === listener && !!e.capture === !!opts.capture);
    if (i >= 0) list.splice(i, 1);
  }
  dispatchEvent(event) {
    if (!event || typeof event !== 'object') throw new TypeError('Event expected');
    event.target = this;
    event.currentTarget = this;
    const list = (this.__listeners[event.type] || []).slice();
    for (const entry of list) {
      if (event.__stopped) break;
      try {
        const fn = typeof entry.listener === 'function' ? entry.listener : entry.listener.handleEvent;
        fn.call(entry.listener, event);
      } catch (e) {
        // Spec says report-the-exception; we surface via console.
        try { console.error('[event listener error]', e && e.message); } catch (_) {}
      }
      if (entry.once) this.removeEventListener(event.type, entry.listener, { capture: entry.capture });
    }
    return !event.defaultPrevented;
  }
};

// ---- AbortSignal / AbortController ----
globalThis.AbortSignal = class AbortSignal extends globalThis.EventTarget {
  constructor() {
    super();
    this.aborted = false;
    this.reason = undefined;
    this.onabort = null;
  }
  static abort(reason) {
    const s = new globalThis.AbortSignal();
    s.aborted = true;
    s.reason = reason === undefined ? new Error('AbortError') : reason;
    return s;
  }
  static timeout(ms) {
    const s = new globalThis.AbortSignal();
    setTimeout(() => {
      if (s.aborted) return;
      s.aborted = true;
      s.reason = new Error('TimeoutError');
      const ev = new globalThis.Event('abort');
      try { if (typeof s.onabort === 'function') s.onabort.call(s, ev); } catch (_) {}
      s.dispatchEvent(ev);
    }, Number(ms) || 0);
    return s;
  }
  throwIfAborted() { if (this.aborted) throw this.reason; }
};

globalThis.AbortController = class AbortController {
  constructor() { this.signal = new globalThis.AbortSignal(); }
  abort(reason) {
    if (this.signal.aborted) return;
    this.signal.aborted = true;
    this.signal.reason = reason === undefined ? new Error('AbortError') : reason;
    const ev = new globalThis.Event('abort');
    try { if (typeof this.signal.onabort === 'function') this.signal.onabort.call(this.signal, ev); } catch (_) {}
    this.signal.dispatchEvent(ev);
  }
};

// ---- Headers ----
// HTTP headers are case-insensitive; the spec also separates Set-Cookie.
// We normalise to lowercase and store as a single map of arrays so
// \`getSetCookie\` can return all values.
globalThis.Headers = class Headers {
  constructor(init) {
    this.__map = Object.create(null);
    if (!init) return;
    if (init instanceof globalThis.Headers) {
      init.forEach((v, k) => this.append(k, v));
      return;
    }
    if (Array.isArray(init)) {
      for (const pair of init) {
        if (!Array.isArray(pair) || pair.length !== 2) throw new TypeError('Headers init: bad pair');
        this.append(pair[0], pair[1]);
      }
      return;
    }
    if (typeof init === 'object') {
      for (const k of Object.keys(init)) this.append(k, init[k]);
    }
  }
  __key(name) {
    if (typeof name !== 'string') throw new TypeError('header name must be a string');
    return name.toLowerCase();
  }
  append(name, value) {
    const k = this.__key(name);
    if (!this.__map[k]) this.__map[k] = [];
    this.__map[k].push(String(value));
  }
  set(name, value) { this.__map[this.__key(name)] = [String(value)]; }
  get(name) {
    const k = this.__key(name);
    return this.__map[k] ? this.__map[k].join(', ') : null;
  }
  has(name) { return !!this.__map[this.__key(name)]; }
  delete(name) { delete this.__map[this.__key(name)]; }
  getSetCookie() { return (this.__map['set-cookie'] || []).slice(); }
  forEach(cb, thisArg) {
    for (const k of Object.keys(this.__map).sort()) {
      cb.call(thisArg, this.__map[k].join(', '), k, this);
    }
  }
  *entries() {
    for (const k of Object.keys(this.__map).sort()) yield [k, this.__map[k].join(', ')];
  }
  *keys() { for (const k of Object.keys(this.__map).sort()) yield k; }
  *values() { for (const k of Object.keys(this.__map).sort()) yield this.__map[k].join(', '); }
  [Symbol.iterator]() { return this.entries(); }
};

// ---- Body shared mixin ----
function __makeBody(input, init) {
  // Accepts string, ArrayBuffer, Uint8Array, URLSearchParams, plain
  // object (treated as JSON for Response.json shorthand). Stores as
  // ArrayBuffer + a content-type hint.
  if (input == null) return { buffer: new ArrayBuffer(0), contentType: null };
  if (typeof input === 'string') {
    const bytes = new TextEncoder().encode(input);
    return {
      buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      contentType: 'text/plain;charset=UTF-8',
    };
  }
  if (input instanceof ArrayBuffer) return { buffer: input.slice(0), contentType: null };
  if (ArrayBuffer.isView(input)) {
    return {
      buffer: input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength),
      contentType: null,
    };
  }
  if (input instanceof URLSearchParams) {
    const bytes = new TextEncoder().encode(input.toString());
    return {
      buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      contentType: 'application/x-www-form-urlencoded;charset=UTF-8',
    };
  }
  // Generic object — treat as JSON.
  const json = JSON.stringify(input);
  const bytes = new TextEncoder().encode(json);
  return {
    buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: 'application/json;charset=UTF-8',
  };
}

function __bodyMethods(proto) {
  proto.arrayBuffer = async function () {
    if (this.bodyUsed) throw new TypeError('Body already used');
    this.bodyUsed = true;
    return this.__bodyBuffer.slice(0);
  };
  proto.text = async function () {
    if (this.bodyUsed) throw new TypeError('Body already used');
    this.bodyUsed = true;
    return new TextDecoder().decode(new Uint8Array(this.__bodyBuffer));
  };
  proto.json = async function () {
    const t = await this.text();
    return JSON.parse(t);
  };
  proto.bytes = async function () {
    const ab = await this.arrayBuffer.call(this);
    return new Uint8Array(ab);
  };
}

// ---- Request ----
globalThis.Request = class Request {
  constructor(input, init) {
    init = init || {};
    if (input instanceof globalThis.Request) {
      this.url = input.url;
      this.method = init.method || input.method;
      this.headers = new globalThis.Headers(init.headers || input.headers);
      const body = init.body !== undefined ? init.body : null;
      const made = __makeBody(body);
      this.__bodyBuffer = made.buffer;
      this.bodyUsed = false;
      this.signal = init.signal || input.signal || null;
    } else {
      this.url = String(input);
      this.method = (init.method || 'GET').toUpperCase();
      this.headers = new globalThis.Headers(init.headers);
      const made = __makeBody(init.body);
      if (made.contentType && !this.headers.has('content-type')) {
        this.headers.set('content-type', made.contentType);
      }
      this.__bodyBuffer = made.buffer;
      this.bodyUsed = false;
      this.signal = init.signal || null;
    }
  }
  clone() {
    const c = new globalThis.Request(this.url, {
      method: this.method,
      headers: this.headers,
      body: this.__bodyBuffer.byteLength ? this.__bodyBuffer.slice(0) : null,
      signal: this.signal,
    });
    return c;
  }
};
__bodyMethods(globalThis.Request.prototype);

// ---- Response ----
globalThis.Response = class Response {
  constructor(body, init) {
    init = init || {};
    this.status = init.status === undefined ? 200 : (init.status | 0);
    this.statusText = init.statusText || '';
    this.headers = new globalThis.Headers(init.headers);
    const made = __makeBody(body);
    if (made.contentType && !this.headers.has('content-type')) {
      this.headers.set('content-type', made.contentType);
    }
    this.__bodyBuffer = made.buffer;
    this.bodyUsed = false;
    this.ok = this.status >= 200 && this.status < 300;
    this.redirected = false;
    this.type = 'default';
    this.url = '';
  }
  static json(data, init) {
    const headers = new globalThis.Headers((init && init.headers) || undefined);
    if (!headers.has('content-type')) headers.set('content-type', 'application/json;charset=UTF-8');
    return new globalThis.Response(JSON.stringify(data), { status: (init && init.status) || 200, statusText: init && init.statusText, headers });
  }
  static error() {
    const r = new globalThis.Response(null, { status: 0 });
    r.type = 'error';
    return r;
  }
  static redirect(url, status) {
    const code = status || 302;
    if ([301, 302, 303, 307, 308].indexOf(code) === -1) throw new RangeError('invalid redirect status');
    return new globalThis.Response(null, { status: code, headers: { location: String(url) } });
  }
  clone() {
    const r = new globalThis.Response(this.__bodyBuffer.slice(0), {
      status: this.status,
      statusText: this.statusText,
      headers: this.headers,
    });
    return r;
  }
};
__bodyMethods(globalThis.Response.prototype);

// ---- URLPattern ----
// Bridge to the host's URLPattern (native on Node 22+, polyfilled via
// the urlpattern-polyfill npm package on Node 20 — see js-v1.ts
// module init). We do NOT reimplement the spec. The bridge call is
// sync (applySync); wrapping the result lets us expose .test/.exec
// without keeping a host reference per-call.
globalThis.URLPattern = class URLPattern {
  constructor(input, baseURL) {
    this.__pattern = __hostUrlPatternCreate.applySync(undefined, [input, baseURL], {
      arguments: { copy: true },
      result: { copy: true },
    });
    if (this.__pattern && this.__pattern.error) {
      throw new TypeError('URLPattern: ' + this.__pattern.error);
    }
  }
  test(input, baseURL) {
    return __hostUrlPatternTest.applySync(undefined, [this.__pattern.id, input, baseURL], {
      arguments: { copy: true },
      result: { copy: true },
    });
  }
  exec(input, baseURL) {
    return __hostUrlPatternExec.applySync(undefined, [this.__pattern.id, input, baseURL], {
      arguments: { copy: true },
      result: { copy: true },
    });
  }
};
`;

export const CTX_BUILDER_SOURCE = `
const sync = globalThis.__ctxSync;
const rawBody = globalThis.__ctxRawBody;

function asResponse(r) {
  // Build a real Response from {status, statusText, ok, headers, bodyBytes}.
  // Headers comes back as a plain object record; convert to Headers.
  const headers = new globalThis.Headers(r.headers || {});
  const resp = new globalThis.Response(r.bodyBytes ? new Uint8Array(r.bodyBytes) : null, {
    status: r.status,
    statusText: r.statusText,
    headers,
  });
  return resp;
}

const log = {
  debug: (...a) => __ctxLog.applyIgnored(undefined, ['debug', a.map(String).join(' '), undefined]),
  info:  (...a) => __ctxLog.applyIgnored(undefined, ['info',  a.map(String).join(' '), undefined]),
  warn:  (...a) => __ctxLog.applyIgnored(undefined, ['warn',  a.map(String).join(' '), undefined]),
  error: (...a) => __ctxLog.applyIgnored(undefined, ['error', a.map(String).join(' '), undefined]),
};

const emit = (event, payload) => __ctxEmit.applyIgnored(undefined, [event, payload]);

const callHost = async (method, args) => {
  return await __ctxHost.apply(undefined, [method, args], {
    arguments: { copy: true },
    result: { copy: true, promise: true },
  });
};

const fetchImpl = async (url, init) => {
  // Accept Request as first argument.
  if (url && typeof url === 'object' && typeof url.url === 'string') {
    const req = url;
    url = req.url;
    if (!init) init = { method: req.method, headers: req.headers };
  }
  // Convert Headers to a plain record for transport.
  let headers;
  if (init && init.headers) {
    if (init.headers instanceof globalThis.Headers) {
      headers = {};
      init.headers.forEach((v, k) => { headers[k] = v; });
    } else {
      headers = init.headers;
    }
  }
  const body = init && init.body;
  const bodyXfer = body == null ? undefined
    : typeof body === 'string' ? body
    : body instanceof ArrayBuffer ? body
    : ArrayBuffer.isView(body) ? body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength)
    : JSON.stringify(body);
  const r = await __ctxFetch.apply(undefined, [url, {
    method: init && init.method,
    headers,
    body: bodyXfer,
    timeoutMs: init && init.timeoutMs,
  }], {
    arguments: { copy: true },
    result: { copy: true, promise: true },
  });
  return asResponse(r);
};

const secrets = {
  get: (name) => callHost('secrets.get', { name }),
  put: (name, value) => callHost('secrets.put', { name, value }),
  list: () => callHost('secrets.list', {}),
};
// ctx.kv supports BOTH the legacy single-string-key form AND the new
// object form { user_id, key, scope, ttlSeconds, value? }. Object form
// is required for visitor-keyed writes and cross-function (scope:
// 'agent') sharing — see KvCallArgs in functions/context.ts.
const kvIsObjArg = (a) => a !== null && typeof a === 'object' && typeof a.key === 'string';
const kv = {
  get: (a) => kvIsObjArg(a)
    ? callHost('kv.get', { key: a.key, user_id: a.user_id, scope: a.scope })
    : callHost('kv.get', { key: a }),
  put: (a, value, opts) => {
    if (kvIsObjArg(a)) {
      // Single-arg object form: { user_id, key, value, ttlSeconds, scope }.
      // The "value" MUST live on the object — extra positional args are
      // ignored to match the type signature.
      return callHost('kv.put', {
        key: a.key,
        value: a.value,
        user_id: a.user_id,
        scope: a.scope,
        ttlSeconds: a.ttlSeconds,
      });
    }
    return callHost('kv.put', {
      key: a,
      value,
      ttlSeconds: opts && (opts.ttlSeconds ?? (typeof opts.ttlMs === 'number' ? Math.floor(opts.ttlMs / 1000) : undefined)),
    });
  },
  delete: (a) => kvIsObjArg(a)
    ? callHost('kv.delete', { key: a.key, user_id: a.user_id, scope: a.scope })
    : callHost('kv.delete', { key: a }),
  list: (a, opts) => {
    if (a !== null && typeof a === 'object') {
      return callHost('kv.list', {
        prefix: a.prefix,
        user_id: a.user_id,
        scope: a.scope,
        limit: a.limit,
        cursor: a.cursor,
      });
    }
    return callHost('kv.list', { prefix: a, limit: opts && opts.limit, cursor: opts && opts.cursor });
  },
};
const content = {
  get: async (id) => {
    const r = await callHost('content.read', { id });
    if (!r) return null;
    const body = r.body && r.body.kind === 'utf8'
      ? new TextEncoder().encode(r.body.data).buffer
      : r.body
        ? Uint8Array.from(atob(r.body.data), (c) => c.charCodeAt(0)).buffer
        : new ArrayBuffer(0);
    return { id: r.id, mimeType: r.mimeType, displayName: r.displayName, size: r.size, arrayBuffer: async () => body };
  },
  put: async (item) => {
    const data = typeof item.data === 'string'
      ? { kind: 'utf8', data: item.data }
      : { kind: 'base64', data: __bytesToB64(item.data) };
    return await callHost('content.write', { id: item.id, body: data });
  },
};
const folders = new Proxy({}, {
  get: (_t, alias) => {
    if (typeof alias !== 'string') return undefined;
    return {
      list: (opts) => callHost('folders.list', { binding: alias, prefix: opts && opts.prefix, limit: opts && opts.limit, cursor: opts && opts.cursor }),
      read: async (name) => {
        const r = await callHost('folders.read', { binding: alias, name });
        if (!r) return null;
        const body = r.body && r.body.kind === 'utf8'
          ? new TextEncoder().encode(r.body.data).buffer
          : r.body
            ? Uint8Array.from(atob(r.body.data), (c) => c.charCodeAt(0)).buffer
            : new ArrayBuffer(0);
        return { id: r.id, mimeType: r.mimeType, name: r.displayName, size: r.size, arrayBuffer: async () => body };
      },
      write: async (name, data) => {
        const body = typeof data === 'string'
          ? { kind: 'utf8', data }
          : { kind: 'base64', data: __bytesToB64(data) };
        return await callHost('folders.write', { binding: alias, name, body });
      },
    };
  }
});
const fn = {
  list: () => callHost('fn.list', {}),
  invoke: (name, args, opts) => callHost('fn.invoke', {
    name,
    args,
    idempotencyKey: opts && opts.idempotencyKey,
    // Forward the caller's chain so fan-out depth + cycle detection
    // applies host-side. Defaults to sync.chain when the caller did not
    // override.
    chain: (opts && opts.chain) || sync.chain,
  }),
};
const portal = new Proxy({}, {
  get: (_t, name) => {
    if (typeof name !== 'string') return undefined;
    if (name === 'payment') {
      return new Proxy({}, {
        get: (_t2, op) => {
          if (typeof op !== 'string') return undefined;
          return (...args) => callHost('portal.dispatch', { method: 'payment.' + op, args: paymentArgs(op, args) });
        }
      });
    }
    return (...args) => callHost('portal.dispatch', { method: name, args: portalArgs(name, args) });
  }
});

function portalArgs(method, args) {
  switch (method) {
    case 'verifyToken':    return { token: args[0], opts: args[1] };
    case 'verifyHmac':     return args[0];
    case 'lookupAgent':    return { idOrUsername: args[0] };
    case 'callTool':       return { agentRef: args[0], toolName: args[1], params: args[2], opts: args[3] };
    case 'getOwner':       return {};
    case 'notifyOwner':    return args[0];
    case 'signContentUrl': return { contentId: args[0], opts: args[1] };
    default:               return args;
  }
}
function paymentArgs(op, args) {
  switch (op) {
    case 'lock':    return { paymentToken: args[0], amountNanocents: String(args[1]), reason: args[2] };
    case 'settle':  return { lockId: args[0], amountNanocents: String(args[1]), recipientId: args[2] };
    case 'release': return { lockId: args[0] };
    default:        return args;
  }
}

globalThis.__ctx = {
  ...sync,
  request: sync.request ? { ...sync.request, rawBody: rawBody ? new Uint8Array(rawBody) : undefined } : undefined,
  fetch: fetchImpl,
  secrets, kv, content, folders, fn, portal,
  log, emit,
};

// Top-level fetch alias — user code that does \`fetch(url)\` works.
// Egress is gated by manifest.permissions.fetch (enforced in the worker).
globalThis.fetch = fetchImpl;
`;
