/**
 * The Web Bot Auth request signer: RFC 9421 HTTP Message Signatures under
 * draft-ietf-webbotauth-httpsig-protocol-00 ("P" below), as the platform's
 * verifier requires them. Written 2026-09-17 for ADR 0038 step 5 (W2 design
 * document, sections 2.1 to 2.7 are normative for every byte this file
 * emits). This REPLACED the bearer assertion `AgentIdentity.mintToken` used
 * to mint: a JWT proves possession of a key, a signed request proves the
 * key holder sent THIS method to THIS host and path with THIS body, so a
 * captured credential verifies against nothing else.
 *
 * THE WIRE (design 2.1). Three headers, plus `Content-Digest` (RFC 9530)
 * whenever the request carries a non-empty body:
 *
 *   Signature-Agent: sig1="https://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri
 *   Signature-Input: sig1=("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="sig1");created=...;expires=...;keyid="<thumbprint>";alg="ed25519";nonce="<b64>";tag="web-bot-auth"
 *   Signature: sig1=:<base64 of the 64 signature bytes>:
 *   Content-Digest: sha-256=:<base64 of SHA-256 over the body>:
 *
 * Every one of them is an RFC 9651 Structured Field, and the verifier
 * re-serialises what it parsed to rebuild the signature base, so the
 * SERIALISATION is part of the contract: a String is escaped and quoted, a
 * Token is bare, an Integer is decimal, a Byte Sequence is standard base64
 * (never base64url) between colons, and an Inner List's parameters follow
 * its closing parenthesis. The subset this signer emits is hand-written
 * below, about sixty lines, with no parser and therefore no dependency: the
 * platform parses with `structured-headers` and the two must agree on
 * these bytes, which the cross-language vectors of design section 10.3 pin.
 *
 * THE COVERED SET is fixed (design 2.2): `@method`, `@authority`, `@path`,
 * `@query`, `content-digest` when there is a body, and the `Signature-Agent`
 * member keyed by the signature's own label, in that order. Nothing less
 * verifies (a signature over `@authority` alone "verifies against any
 * method, path, or body sent to that authority until it expires", P 5.2)
 * and nothing more is needed by any route. `@query` rather than
 * `@query-param`, because this signer never reorders a query and the
 * verifier refuses the parser `@query-param` needs.
 *
 * COVERED HEADERS (2026-09-18, machine-purchase design section 6.4). A paid
 * retry carries `Payment-Authorization` and `Robutler-Terms-Accepted`, and
 * the platform admits the payment and the assent only when BOTH are among
 * the signature's covered components, so that a proxy or a log between the
 * agent and the platform cannot attach a credential, or an acceptance of
 * the Terms, that the agent's own key never signed. `SignRequestOptions
 * .coveredHeaders` names plain header fields to cover, and they are placed
 * after `content-digest` and before the `signature-agent` member, so the
 * fixed set keeps its shape and the member stays last. The component name
 * is the field name lowercased and bare (the verifier refuses `;sf`, `;bs`,
 * `;tr` and `;key` on anything but `signature-agent`), and the value is the
 * field value with leading and trailing whitespace stripped, exactly what
 * the verifier reads back with `Headers.get(name).trim()` (RFC 9421
 * section 2.1). A header named but absent from the message is refused
 * before signing, because the verifier would refuse the signature as
 * malformed and the buyer would learn nothing from that 401. The vectors in
 * `vectors-covered-headers.json` pin these bytes across both SDKs.
 *
 * THE PARAMETERS (design 2.3): `created` now, `expires` sixty seconds
 * later, `keyid` the RFC 7638 thumbprint of the signing key, `alg`
 * `ed25519`, a `nonce` of 64 random bytes in standard base64, and `tag`
 * `web-bot-auth`. The platform spends each nonce once, so the sixty
 * seconds bound only the window before a signature is presented.
 *
 * `Signature-Agent` (design 2.4) names the KEY SET, `{agentUrl}/.well-known/
 * jwks.json`, and the platform derives the principal, the agent URL, by
 * stripping that suffix. The form is a setting with three values so the next
 * revision of P is a one-line default change: `dictionary-typed` (P-00, the
 * default), `dictionary-untyped` (the editor's copy, same value, no `type`)
 * and `legacy-string` (a bare String naming the ORIGIN, what Cloudflare
 * interop reads; it cannot name an agent mounted under a path).
 *
 * ROTATION (design 2.5). An identity holding more than one key signs the
 * request once per key, labels `sig1`, `sig2`, each with its own nonce,
 * `keyid` and `Signature-Agent` member. That is what lets the platform's
 * key-continuity rule (a new key is admitted only when a key it already
 * holds co-signed) be switched on with no SDK change.
 *
 * WHAT IS REFUSED BEFORE SIGNING: an agent URL that is loopback (the
 * `http://localhost:<port>` fallback `serve()` uses when no public URL is
 * configured; the platform refuses `localhost` by name before it resolves
 * anything) or that is not https (operator decision 2: the platform admits a
 * plaintext key set only where `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1`, its
 * local overlay, so the same switch is honoured here by name). Refusing at
 * the signer, with a sentence that names the variable to set, is what turns
 * an otherwise bare 401 into a diagnosis.
 */

import { MAX_HELD_KEYS, type HeldKey } from './identity';

// ---------------------------------------------------------------------------
// Wire constants (design section 2)
// ---------------------------------------------------------------------------

/** P section 5.2: the `tag` every Web Bot Auth signature carries. */
export const SIGNATURE_TAG = 'web-bot-auth';
/** RFC 9421 section 3.3.7 registry name. The only algorithm this SDK signs with (ADR operator decision 3). */
export const SIGNATURE_ALG = 'ed25519';
/** Design section 2.3: `expires - created`. Cloudflare's guidance and mppx's default. */
export const SIGNATURE_LIFETIME_SECONDS = 60;
/** Design section 2.3: the verifier refuses a longer window, so the signer refuses to produce one. */
export const SIGNATURE_MAX_LIFETIME_SECONDS = 3600;
/** Design section 2.3 and P Appendix E: 64 random bytes, standard base64 (88 characters). */
export const NONCE_BYTES = 64;
/** Design section 2.5: the first label; a second held key signs as `sig2`. */
export const DEFAULT_LABEL = 'sig1';
/** Design section 2.4: the key set lives under the agent URL at this path. */
export const KEY_SET_WELL_KNOWN_SUFFIX = '/.well-known/jwks.json';

export type SignatureAgentForm = 'dictionary-typed' | 'dictionary-untyped' | 'legacy-string';

export const SIGNATURE_AGENT_FORMS: readonly SignatureAgentForm[] = [
  'dictionary-typed',
  'dictionary-untyped',
  'legacy-string',
];

/** What the signer needs from an identity: the agent URL and one signer per held key. `AgentIdentity` satisfies it. */
export interface SigningIdentity {
  /** The agent URL, the principal: `https://host/agents/mini`. No trailing slash. */
  readonly issuer: string;
  /** Every key the identity holds, current first (design section 2.5). */
  getHeldKeys(): ReadonlyArray<HeldKey>;
}

export interface SignRequestOptions {
  /** The `Signature-Agent` form. Default `dictionary-typed` (P-00). */
  form?: SignatureAgentForm;
  /** The first signature's label. Default `sig1`; further held keys sign as `sig2`, `sig3`. */
  label?: string;
  /** `expires - created`, seconds. Default 60; at most 3600 (the verifier's cap). */
  lifetimeSeconds?: number;
  /**
   * Sign for a plaintext agent URL. Default: `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1`
   * in this process's environment, the switch the platform's local overlay
   * sets (operator decision 2). Loopback is refused whatever this says.
   */
  allowHttp?: boolean;
  /**
   * Plain header fields to cover beside the fixed set (file comment,
   * "COVERED HEADERS"). Names are matched case-insensitively and covered
   * lowercased, in the order given, a repeated name covered once. The
   * message must carry every one of them, and none may be `content-digest`
   * (covered by the body rule), `signature-agent`, `signature-input` or
   * `signature` (the signature's own fields), or a `@` derived component.
   */
  coveredHeaders?: readonly string[];
  /**
   * TEST SEAMS. A fixed `created` and a fixed nonce make a signature
   * reproducible, which the cross-language vectors need. Production callers
   * never pass either: a reused nonce is `signature_replayed` on the second
   * request.
   *
   * ONE NONCE PER KEY. A plain string is one nonce, so it is accepted only
   * while the identity holds ONE key. Until 2026-09-19 it was used for every
   * label, and an identity holding two keys sent `sig1` and `sig2` with the
   * same nonce: the platform spends each nonce once and answered
   * `signature_replayed` to the second label, so the very request that
   * performs a rotation was the one refused. With more than one held key
   * pass the function form, which is asked once per label index and must
   * return a different value each time; both are checked before any key
   * signs. The Python signer has always required one nonce per key.
   */
  created?: number;
  nonce?: string | ((labelIndex: number) => string);
}

/** The headers a signed message carries. `content-digest` only when the body is non-empty. */
export interface SignedHeaders {
  'signature-agent': string;
  'signature-input': string;
  signature: string;
  'content-digest'?: string;
}

/** One label's worth of what was signed, for tests, vectors and logs. Never sent. */
export interface SignedLabel {
  label: string;
  kid: string;
  nonce: string;
  /** The exact ASCII string the signature covers (RFC 9421 section 2.5). */
  base: string;
}

export interface SignedMessage {
  headers: SignedHeaders;
  labels: SignedLabel[];
  created: number;
  expires: number;
}

/** The message a signature covers, independent of the `Request` class. */
export interface MessageToSign {
  method: string;
  /** Absolute URL. `@authority`, `@path` and `@query` are read off it as the WHATWG parser normalises them. */
  url: string;
  /** The body bytes. Absent or empty means no `Content-Digest` and no `content-digest` component. */
  body?: Uint8Array;
  /**
   * The message's header fields, read only for `coveredHeaders`: a `Headers`
   * (repeated fields already joined with `, `, RFC 9421 section 2.1 step 4)
   * or a plain record matched case-insensitively. A message that names a
   * covered header and carries no `headers` is refused.
   */
  headers?: Pick<Headers, 'get'> | Readonly<Record<string, string>>;
}

/** Fields a signature may never list as a covered header: the body rule covers the first, the rest are the signature itself. */
export const COVERED_HEADERS_RESERVED: ReadonlySet<string> = new Set(['content-digest', 'signature-agent', 'signature-input', 'signature']);

// RFC 9110 section 5.1 token, lowercased: the field-name alphabet.
const FIELD_NAME_RE = /^[a-z0-9!#$%&'*+\-.^_`|~]+$/;

/**
 * The covered header names as they are placed in the signature: trimmed,
 * lowercased, deduplicated in first-seen order, and refused when reserved,
 * derived (`@`) or not a field name. Exported so a caller composing a
 * covered list (the MPP buyer adds its own names to the operator's) can
 * see the list the signer will use.
 */
export function normalizeCoveredHeaders(names: readonly string[] | undefined): string[] {
  const out: string[] = [];
  for (const raw of names ?? []) {
    if (typeof raw !== 'string') throw new Error('coveredHeaders must be header field names');
    const name = raw.trim().toLowerCase();
    if (name.startsWith('@')) throw new Error(`coveredHeaders names header fields; ${JSON.stringify(raw)} is a derived component`);
    if (!FIELD_NAME_RE.test(name)) throw new Error(`coveredHeaders: ${JSON.stringify(raw)} is not a header field name`);
    if (COVERED_HEADERS_RESERVED.has(name)) {
      throw new Error(
        `coveredHeaders: ${name} is covered by the signer itself and cannot be listed` +
          (name === 'content-digest' ? ' (it is covered whenever the message has a body)' : ''),
      );
    }
    if (!out.includes(name)) out.push(name);
  }
  return out;
}

function readHeader(headers: MessageToSign['headers'], name: string): string | null {
  if (!headers) return null;
  if (typeof (headers as Pick<Headers, 'get'>).get === 'function') {
    const value = (headers as Pick<Headers, 'get'>).get(name);
    return value === undefined ? null : value;
  }
  for (const [key, value] of Object.entries(headers as Record<string, string>)) {
    if (key.toLowerCase() === name) return value;
  }
  return null;
}

// ---------------------------------------------------------------------------
// RFC 9651 Structured Fields, the serialising subset (section 4.1)
// ---------------------------------------------------------------------------

/** A Token bare item (RFC 9651 section 3.3.4). A wrapper so it is never confused with a String. */
export class SfToken {
  constructor(readonly value: string) {}
}

export type SfBareItem = string | number | boolean | SfToken | Uint8Array;
export type SfParameters = ReadonlyArray<readonly [string, SfBareItem]>;
export interface SfItem {
  value: SfBareItem;
  params?: SfParameters;
}
export interface SfInnerList {
  items: ReadonlyArray<SfItem>;
  params?: SfParameters;
}

const KEY_RE = /^[a-z*][a-z0-9_\-.*]*$/;
const TOKEN_RE = /^[A-Za-z*][A-Za-z0-9:/!#$%&'*+\-.^_`|~]*$/;
const PRINTABLE_ASCII_RE = /^[\x20-\x7e]*$/;

export class StructuredFieldError extends Error {
  constructor(message: string) {
    super(`structured field: ${message}`);
    this.name = 'StructuredFieldError';
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/** RFC 9651 section 4.1.3.1 to 4.1.8: one bare item. */
export function serializeBareItem(value: SfBareItem): string {
  if (typeof value === 'number') {
    // Section 4.1.4: Integers only. A Decimal is never emitted by this signer.
    if (!Number.isInteger(value) || Math.abs(value) > 999_999_999_999_999) {
      throw new StructuredFieldError('not a serialisable integer');
    }
    return String(value);
  }
  if (typeof value === 'string') {
    // Section 4.1.6: printable ASCII only, `\` and `"` escaped, quoted.
    if (!PRINTABLE_ASCII_RE.test(value)) throw new StructuredFieldError('string is not printable ASCII');
    return `"${value.replace(/(["\\])/g, '\\$1')}"`;
  }
  if (typeof value === 'boolean') return value ? '?1' : '?0';
  if (value instanceof SfToken) {
    if (!TOKEN_RE.test(value.value)) throw new StructuredFieldError('not a token');
    return value.value;
  }
  // Section 4.1.8: standard base64 with padding, between colons.
  return `:${bytesToBase64(value)}:`;
}

/** RFC 9651 section 4.1.1.2. A `true` value is the bare key. */
export function serializeParameters(params: SfParameters | undefined): string {
  if (!params) return '';
  let out = '';
  for (const [key, value] of params) {
    if (!KEY_RE.test(key)) throw new StructuredFieldError(`bad parameter key ${JSON.stringify(key)}`);
    out += `;${key}`;
    if (value !== true) out += `=${serializeBareItem(value)}`;
  }
  return out;
}

/** RFC 9651 section 4.1.3: bare item then parameters. */
export function serializeItem(item: SfItem): string {
  return serializeBareItem(item.value) + serializeParameters(item.params);
}

/** RFC 9651 section 4.1.1.1: `(` items `)` then the list's own parameters. */
export function serializeInnerList(list: SfInnerList): string {
  return `(${list.items.map(serializeItem).join(' ')})${serializeParameters(list.params)}`;
}

/** RFC 9651 section 4.1.2. Members in the order given; a `true` Item value is the bare key with its parameters. */
export function serializeDictionary(members: ReadonlyArray<readonly [string, SfItem | SfInnerList]>): string {
  return members
    .map(([key, member]) => {
      if (!KEY_RE.test(key)) throw new StructuredFieldError(`bad dictionary key ${JSON.stringify(key)}`);
      if ('items' in member) return `${key}=${serializeInnerList(member)}`;
      if (member.value === true) return `${key}${serializeParameters(member.params)}`;
      return `${key}=${serializeItem(member)}`;
    })
    .join(', ');
}

// ---------------------------------------------------------------------------
// The pieces (design sections 2.2 to 2.4)
// ---------------------------------------------------------------------------

/** RFC 9530: `sha-256=:<base64>:` over `bytes`. The only algorithm the SDKs send. */
export async function contentDigest(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer));
  return serializeDictionary([['sha-256', { value: digest }]]);
}

/** Design section 2.3: 64 random bytes as standard base64, 88 characters. */
export function randomNonce(): string {
  const bytes = new Uint8Array(NONCE_BYTES);
  crypto.getRandomValues(bytes);
  return bytesToBase64(bytes);
}

/**
 * The value a `Signature-Agent` member carries for an agent URL under a form
 * (design section 2.4): the key-set URL for both dictionary forms, the
 * ORIGIN for the legacy string (the platform reads a bare string as a
 * directory origin, design section 3.1, so a path-bearing legacy value would
 * derive no principal).
 */
export function signatureAgentValue(agentUrl: string, form: SignatureAgentForm): string {
  const url = new URL(agentUrl);
  if (form === 'legacy-string') return url.origin;
  return `${url.origin}${url.pathname.replace(/\/+$/, '')}${KEY_SET_WELL_KNOWN_SUFFIX}`;
}

/**
 * RFC 9421 section 2.5: the signature base. `lines` are
 * `[identifier, value]` pairs in covered order, `identifier` already
 * serialised (`"@method"`, `"signature-agent";key="sig1"`), and
 * `signatureParams` is the `Signature-Input` member re-serialised without
 * its label. Joined by LF, no trailing newline, printable ASCII throughout:
 * a value carrying a CR or LF could forge a line, so it is refused.
 */
export function buildSignatureBase(lines: ReadonlyArray<readonly [string, string]>, signatureParams: string): string {
  const out: string[] = [];
  for (const [identifier, value] of lines) {
    if (!PRINTABLE_ASCII_RE.test(value)) throw new StructuredFieldError(`value of ${identifier} is not printable ASCII`);
    out.push(`${identifier}: ${value}`);
  }
  out.push(`"@signature-params": ${signatureParams}`);
  return out.join('\n');
}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

function isLoopbackHost(hostname: string): boolean {
  return (
    hostname === 'localhost' ||
    hostname.endsWith('.localhost') ||
    hostname === '127.0.0.1' ||
    hostname === '[::1]' ||
    hostname === '::1'
  );
}

/**
 * The agent URL an identity may sign as, or a thrown sentence saying why
 * not (the file comment's last paragraph). `registerWithPlatform` turns the
 * throw into its `{ ok: false, error }`; every other caller gets the throw.
 */
export function assertSignableAgentUrl(agentUrl: string, options: { allowHttp?: boolean } = {}): URL {
  let url: URL;
  try {
    url = new URL(agentUrl);
  } catch {
    throw new Error(`agent URL is not a URL: ${agentUrl}`);
  }
  if (isLoopbackHost(url.hostname)) {
    throw new Error(
      `agent URL ${agentUrl} is a loopback address. The PLATFORM fetches the ` +
        'key set and the agent card from it, so set publicUrl / WEBAGENTS_PUBLIC_URL ' +
        'to an address reachable from the internet',
    );
  }
  const allowHttp = options.allowHttp ?? envVar('ROBUTLER_AGENT_URL_ALLOW_PRIVATE') === '1';
  if (url.protocol !== 'https:' && !(allowHttp && url.protocol === 'http:')) {
    throw new Error(
      `agent URL ${agentUrl} is not https. The platform accepts a plaintext key set ` +
        'only where ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1 (its local overlay); set ' +
        'publicUrl / WEBAGENTS_PUBLIC_URL to an https address, or set ' +
        'ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1 for this process to sign for a plaintext one',
    );
  }
  return url;
}

function labelsFor(first: string, count: number): string[] {
  if (!KEY_RE.test(first)) throw new StructuredFieldError(`bad signature label ${JSON.stringify(first)}`);
  const stem = first.replace(/\d+$/, '');
  return Array.from({ length: count }, (_, i) => (i === 0 ? first : `${stem}${i + 1}`));
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

/**
 * Sign a message description: the pure core `signRequest` wraps and the
 * tests and vectors call directly. One signature per held key, every one
 * over the same base shape and its own nonce and `keyid`.
 */
export async function signMessage(
  identity: SigningIdentity,
  message: MessageToSign,
  options: SignRequestOptions = {},
): Promise<SignedMessage> {
  const agentUrl = assertSignableAgentUrl(identity.issuer, { allowHttp: options.allowHttp });
  const form = options.form ?? 'dictionary-typed';
  if (!SIGNATURE_AGENT_FORMS.includes(form)) throw new Error(`unknown Signature-Agent form ${String(form)}`);
  const lifetime = options.lifetimeSeconds ?? SIGNATURE_LIFETIME_SECONDS;
  if (!Number.isInteger(lifetime) || lifetime <= 0 || lifetime > SIGNATURE_MAX_LIFETIME_SECONDS) {
    throw new Error(`signature lifetime must be an integer between 1 and ${SIGNATURE_MAX_LIFETIME_SECONDS} seconds`);
  }
  const keys = identity.getHeldKeys();
  if (keys.length === 0) throw new Error('the identity holds no key to sign with');
  // `AgentIdentity` refuses this at initialize(); a caller's own
  // SigningIdentity gets the same answer here rather than a request the
  // platform calls signature_malformed (2026-09-18).
  if (keys.length > MAX_HELD_KEYS) {
    throw new Error(
      `${keys.length} keys held, but the platform verifies at most ${MAX_HELD_KEYS} signatures ` +
        'per request; hold the current key and one previous key while rotating',
    );
  }

  const url = new URL(message.url);
  const body = message.body ?? new Uint8Array(0);
  const digest = body.byteLength > 0 ? await contentDigest(body) : undefined;
  // Covered headers (file comment): resolved once, before any key signs, so
  // a missing field refuses the whole message rather than one label of it.
  const coveredHeaders: Array<readonly [string, string]> = normalizeCoveredHeaders(options.coveredHeaders).map((name) => {
    const raw = readHeader(message.headers, name);
    if (raw === null) {
      throw new Error(
        `cannot cover header ${name}: the message does not carry it. A covered header must be set on the ` +
          'request before it is signed, since the verifier reads the value off the wire',
      );
    }
    return [name, raw.trim()] as const;
  });
  const created = options.created ?? Math.floor(Date.now() / 1000);
  if (!Number.isInteger(created)) throw new Error('created must be an integer');
  const expires = created + lifetime;
  const labels = labelsFor(options.label ?? DEFAULT_LABEL, keys.length);
  // One nonce per label, resolved before any key signs (SignRequestOptions
  // .nonce): a shared nonce is `signature_replayed` on the second label.
  if (typeof options.nonce === 'string' && keys.length > 1) {
    throw new Error(
      `a fixed nonce string cannot sign for ${keys.length} held keys: every label needs its own nonce ` +
        '(the platform spends each nonce once and answers signature_replayed to the second). ' +
        'Pass nonce as a function of the label index, or leave it unset for a random nonce per label',
    );
  }
  const nonces = keys.map((_, i) => (typeof options.nonce === 'function' ? options.nonce(i) : (options.nonce ?? randomNonce())));
  for (const nonce of nonces) {
    if (typeof nonce !== 'string' || nonce.length === 0) throw new Error('a nonce must be a non-empty string');
  }
  if (new Set(nonces).size !== nonces.length) {
    throw new Error('every label needs its own nonce: the nonce function returned the same value for two labels');
  }
  const agentValue = signatureAgentValue(`${agentUrl.origin}${agentUrl.pathname}`, form);

  // The `Signature-Agent` field: one member per label for the dictionary
  // forms (the member key equals the label, so both readings of P issue
  // #128 select it; every member carries the same value, since every held
  // key publishes in the same set), a single bare String for the legacy form.
  const member: SfItem =
    form === 'dictionary-typed'
      ? { value: agentValue, params: [['type', new SfToken('jwks_uri')]] }
      : { value: agentValue };
  const signatureAgent =
    form === 'legacy-string'
      ? serializeItem({ value: agentValue })
      : serializeDictionary(labels.map((label) => [label, member] as const));

  const inputs: Array<readonly [string, SfInnerList]> = [];
  const signatures: Array<readonly [string, SfItem]> = [];
  const signed: SignedLabel[] = [];

  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    const label = labels[i];
    const nonce = nonces[i];

    // Design section 2.2: the covered components, in this order and no
    // other; the covered headers (design 6.4) sit between the body digest
    // and the signature-agent member, which stays last.
    const components: SfItem[] = [
      { value: '@method' },
      { value: '@authority' },
      { value: '@path' },
      { value: '@query' },
      ...(digest ? [{ value: 'content-digest' } as SfItem] : []),
      ...coveredHeaders.map(([name]) => ({ value: name }) as SfItem),
      form === 'legacy-string'
        ? { value: 'signature-agent' }
        : { value: 'signature-agent', params: [['key', label]] },
    ];
    const values: string[] = [
      message.method.toUpperCase(),
      // WHATWG `host`: lowercased, default port omitted (RFC 9421 section 2.2.3).
      url.host,
      url.pathname,
      // Section 2.2.7: the query with its leading `?`; `?` alone when absent.
      url.search === '' ? '?' : url.search,
      ...(digest ? [digest] : []),
      ...coveredHeaders.map(([, value]) => value),
      // The component value of a Dictionary member is the member serialised
      // as an Item WITH its parameters (RFC 9421 section 2.1.2), so
      // `;type=jwks_uri` is inside the signature; the legacy string's value
      // is the whole field.
      form === 'legacy-string' ? signatureAgent : serializeItem(member),
    ];
    // Design section 2.3, in this order.
    const input: SfInnerList = {
      items: components,
      params: [
        ['created', created],
        ['expires', expires],
        ['keyid', key.kid],
        ['alg', SIGNATURE_ALG],
        ['nonce', nonce],
        ['tag', SIGNATURE_TAG],
      ],
    };
    const signatureParams = serializeInnerList(input);
    const base = buildSignatureBase(
      components.map((c, j) => [serializeItem(c), values[j]] as const),
      signatureParams,
    );
    const signature = await key.sign(new TextEncoder().encode(base));
    if (signature.byteLength !== 64) throw new Error(`key ${key.kid} produced a ${signature.byteLength}-byte signature, not 64`);

    inputs.push([label, input]);
    signatures.push([label, { value: signature }]);
    signed.push({ label, kid: key.kid, nonce, base });
  }

  const headers: SignedHeaders = {
    'signature-agent': signatureAgent,
    'signature-input': serializeDictionary(inputs),
    signature: serializeDictionary(signatures),
    ...(digest ? { 'content-digest': digest } : {}),
  };
  return { headers, labels: signed, created, expires };
}

/**
 * Sign a `Request`: a new `Request` with the same method, URL, headers,
 * body, redirect mode and signal, plus the three signature headers and
 * `Content-Digest` when the body is non-empty. The body is read here (it
 * has to be digested) and re-attached as bytes; a caller keeps nothing of
 * the input it passed. `options.coveredHeaders` are read off the request's
 * own headers, so set them before signing.
 */
export async function signRequest(
  identity: SigningIdentity,
  request: Request,
  options: SignRequestOptions = {},
): Promise<Request> {
  const hasBody = request.body !== null;
  const body = hasBody ? new Uint8Array(await request.arrayBuffer()) : undefined;
  const signed = await signMessage(
    identity,
    { method: request.method, url: request.url, body, headers: request.headers },
    options,
  );
  const headers = new Headers(request.headers);
  for (const [name, value] of Object.entries(signed.headers)) headers.set(name, value);
  return new Request(request.url, {
    method: request.method,
    headers,
    body: hasBody ? body : undefined,
    redirect: request.redirect,
    signal: request.signal,
  });
}

/**
 * `fetch`, signed. `input` and `init` are what `fetch` takes; `options` is
 * the signer's. The platform verifies the signature against the key set it
 * fetches from `identity.issuer`, so that URL must be public and https.
 */
export async function signedFetch(
  identity: SigningIdentity,
  input: string | URL | Request,
  init?: RequestInit,
  options: SignRequestOptions = {},
): Promise<Response> {
  return fetch(await signRequest(identity, new Request(input, init), options));
}
