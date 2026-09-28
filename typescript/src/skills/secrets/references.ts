/**
 * `${secret:NAME}` and `${env:NAME}` in an MCP server's `env`, `headers` and
 * `url` (S-292, 2026-09-26): the one grammar both SDKs resolve, the sentences
 * they say when they cannot, what a report shows instead of a value, and the
 * patterns that make a literal look like a secret.
 *
 * WHY. Until this, a self-hosted agent's MCP credentials could only be literal
 * text in `AGENT.md` or `mcp.json`: a plain file that is easy to commit, and
 * that the agent's own file and shell tools can read into a conversation. The
 * SDK already had an owner-only store for exactly this kind of value (the OS
 * keystore behind `webagents secrets set`, `./store.ts`), and the MCP config
 * could not point at it. Now `${secret:NAME}` names an entry in that store and
 * `${env:NAME}` names a process variable; both are replaced at connect time,
 * in memory, and the value they resolve to is never written back, never
 * logged, and masked wherever a server's configuration is shown.
 *
 * THE GRAMMAR, the same in `python/webagents/agents/skills/local/secrets/
 * references.py` and pinned by `secret_refs` in the shared fixture
 * `python/tests/fixtures/mcp_tool/config_shapes.json`:
 *
 *   - `${secret:NAME}` reads the keystore `webagents secrets set NAME` writes,
 *     for the active profile; `${env:NAME}` reads the process environment.
 *   - NAME is an environment-variable-shaped name: letters, digits and
 *     underscores, not starting with a digit, at most 128 characters, which is
 *     also what `webagents secrets set` accepts, so every reference that parses
 *     is one the CLI can satisfy.
 *   - `$${` is a literal `${`. Anything else that opens with `${` is refused,
 *     because a `${vault:X}` or a bare `${NAME}` passed through as text is a
 *     typo nobody sees until a server answers 401.
 *   - References are resolved in `env` values, `headers` values and the
 *     address (`url`, `httpUrl`, `mcpUrlTemplate`). A reference in `command`
 *     or `args` is REFUSED by the loader: a command line is readable by every
 *     local account through the process list, so a value there is not a
 *     secret whatever store it came from.
 *
 * MASKING. A configuration shown in a report (`serverReport()`, `doctor`, an
 * error) shows a value as written when it is a well-formed reference, since
 * `${secret:GH}` gives nothing away, and `****` otherwise: a literal in the
 * file is a secret too, and that is the case the loader warns about.
 *
 * No `node:` import here, static or dynamic: the MCP skill that uses this is
 * built for bundlers too. The one thing borrowed from the CLI, how a hint
 * names a command under `--profile`, is repeated in `secretsSetHint` and held
 * equal to `cli/config-store.ts` `cliCommand` by a test.
 */

/** What a report shows instead of a resolved or literal value. */
export const SECRET_MASK = '****';

/** NAME in `${secret:NAME}` and `${env:NAME}`: env-var shaped, at most 128 characters. */
export const REFERENCE_NAME = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;

export type ReferenceScheme = 'secret' | 'env';

/** One `${scheme:NAME}` as written. */
export interface SecretReference {
  scheme: ReferenceScheme;
  name: string;
  /** The reference as written, e.g. `${secret:GH}`; what a sentence names. */
  text: string;
}

/**
 * The sentences, as templates (`{text}` is the reference as written). The
 * fixture holds the same table; a sentence cannot change in one SDK only.
 */
export const REFERENCE_SENTENCES = {
  notClosed: '{text} is not closed: a reference is ${secret:NAME} or ${env:NAME}, and $${ is a literal ${',
  unknown: '{text} is not a reference this SDK knows: write ${secret:NAME} or ${env:NAME}, or $${ for a literal ${',
  badName: '{text} does not name a secret or a variable: use letters, digits and underscores, not starting with a digit',
  secretNotSet: '{text} is not set: store it with `{hint}`',
  envNotSet: '{text} is not set in the environment',
  commandLine: 'puts a reference in {field}, which other local accounts can read from the process list: pass it through env instead',
  literal: 'has what looks like a secret written into {field} {key}: keep it out of the file with ${secret:{suggested}} and `{hint}`',
  atConnect: '{where} of MCP server "{server}": {sentence}',
} as const;

/**
 * A literal that looks like a credential: the key prefixes people paste most
 * (OpenAI and Anthropic `sk-`, GitHub `ghp_` and `github_pat_`, Slack `xox`,
 * AWS `AKIA`) and a bearer of twenty or more characters. Strings, not
 * RegExp objects, so the fixture can hold the same list for Python.
 */
export const SECRET_LOOKING_PATTERNS: readonly string[] = ['^sk-', '^ghp_', '^github_pat_', '^xox', '^AKIA', '^Bearer\\s+\\S{20,}'];

function fillSentence(template: string, values: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) => (key in values ? values[key] : match));
}

/**
 * A reference that cannot be resolved, or is not one. `missingSecret` names a
 * `secret:` that is not stored; `missingEnv` an `env:` variable that is not
 * set (2026-09-26, so `doctor` can name it in its fix line).
 */
export class SecretReferenceError extends Error {
  readonly reference: string;
  readonly missingSecret?: string;
  readonly missingEnv?: string;

  constructor(message: string, reference: string, missingSecret?: string, missingEnv?: string) {
    super(message);
    this.name = 'SecretReferenceError';
    this.reference = reference;
    if (missingSecret !== undefined) this.missingSecret = missingSecret;
    if (missingEnv !== undefined) this.missingEnv = missingEnv;
  }
}

/**
 * `webagents secrets set NAME`, with `--profile <name>` while a profile is
 * active: the twin of `cli/config-store.ts` `cliCommand`, repeated here so
 * this module stays free of the CLI's `node:` imports. A hint without the
 * profile would store the secret in the DEFAULT profile, and the reference,
 * read from the active one, would stay unresolved (the S-219 shape).
 */
export function secretsSetHint(name: string): string {
  const active = typeof process !== 'undefined' ? process.env?.WEBAGENTS_PROFILE || undefined : undefined;
  let base = 'webagents';
  if (active) {
    const word = /^[A-Za-z0-9._-]+$/.test(active) ? active : `'${active.replace(/'/g, `'"'"'`)}'`;
    base = `webagents --profile ${word}`;
  }
  return `${base} secrets set ${name}`;
}

/**
 * The literal runs and references of `value`, in order. Throws
 * {@link SecretReferenceError} for a `${` that does not close, a scheme this
 * SDK does not know, or a name outside the grammar.
 */
export function tokenizeReferences(value: string): Array<string | SecretReference> {
  const out: Array<string | SecretReference> = [];
  let literal = '';
  let i = 0;
  while (i < value.length) {
    if (value.startsWith('$${', i)) {
      literal += '${';
      i += 3;
      continue;
    }
    if (value.startsWith('${', i)) {
      const close = value.indexOf('}', i + 2);
      if (close === -1) {
        // The sentence names the reference as far as it parses, never what
        // follows it (S-303, 2026-09-26): the tail of the value may be a
        // literal key, and an error is logged.
        const shown = unclosedReferenceText(value.slice(i));
        throw new SecretReferenceError(fillSentence(REFERENCE_SENTENCES.notClosed, { text: shown }), shown);
      }
      const text = value.slice(i, close + 1);
      const body = value.slice(i + 2, close);
      const colon = body.indexOf(':');
      const scheme = colon === -1 ? '' : body.slice(0, colon);
      const name = colon === -1 ? '' : body.slice(colon + 1);
      if (scheme !== 'secret' && scheme !== 'env') {
        throw new SecretReferenceError(fillSentence(REFERENCE_SENTENCES.unknown, { text }), text);
      }
      if (!REFERENCE_NAME.test(name)) {
        throw new SecretReferenceError(fillSentence(REFERENCE_SENTENCES.badName, { text }), text);
      }
      if (literal) {
        out.push(literal);
        literal = '';
      }
      out.push({ scheme, name, text });
      i = close + 1;
      continue;
    }
    literal += value[i];
    i += 1;
  }
  if (literal) out.push(literal);
  return out;
}

/** What an unclosed `${...` is named as: `${`, the scheme, the colon and the name as far as they parse, and nothing after. */
export function unclosedReferenceText(rest: string): string {
  return /^\$\{[A-Za-z]{0,16}:?[A-Za-z0-9_]{0,32}/.exec(rest)?.[0] ?? '${';
}

/** Whether `value` holds at least one well-formed reference (and nothing malformed). */
export function hasReference(value: string): boolean {
  try {
    return tokenizeReferences(value).some((token) => typeof token !== 'string');
  } catch {
    return false;
  }
}

/** Whether `value` contains `${` at all: what the command-line refusal looks for. */
export function mentionsReference(value: string): boolean {
  return value.includes('${');
}

/** Where a reference's value comes from. */
export interface ReferenceLookup {
  /** The keystore `webagents secrets set` writes; null or undefined when the name is not stored. */
  secret: (name: string) => Promise<string | null | undefined> | string | null | undefined;
  /** The process environment. */
  env: Record<string, string | undefined>;
}

export interface ExpandedValue {
  /** The value with every reference replaced. */
  text: string;
  /** Every resolved value, for masking a report or an error that might repeat one. */
  values: string[];
}

/**
 * `value` with its references replaced. Throws {@link SecretReferenceError},
 * naming the reference and never a value, when one cannot be resolved.
 */
export async function expandReferences(value: string, lookup: ReferenceLookup): Promise<ExpandedValue> {
  let text = '';
  const values: string[] = [];
  for (const token of tokenizeReferences(value)) {
    if (typeof token === 'string') {
      text += token;
      continue;
    }
    if (token.scheme === 'env') {
      const found = lookup.env[token.name];
      if (found === undefined || found === '') {
        throw new SecretReferenceError(fillSentence(REFERENCE_SENTENCES.envNotSet, { text: token.text }), token.text, undefined, token.name);
      }
      values.push(found);
      text += found;
      continue;
    }
    const found = await lookup.secret(token.name);
    if (found === null || found === undefined || found === '') {
      throw new SecretReferenceError(
        fillSentence(REFERENCE_SENTENCES.secretNotSet, { text: token.text, hint: secretsSetHint(token.name) }),
        token.text,
        token.name,
      );
    }
    values.push(found);
    text += found;
  }
  return { text, values };
}

/** The loader's refusal for a reference in `command` or `args`. */
export function commandLineRefusal(field: 'command' | 'args'): string {
  return fillSentence(REFERENCE_SENTENCES.commandLine, { field });
}

/** Whether a literal value (one with no reference) looks like a credential. */
export function looksLikeSecret(value: string): boolean {
  if (mentionsReference(value)) return false;
  return SECRET_LOOKING_PATTERNS.some((pattern) => new RegExp(pattern).test(value));
}

/**
 * The secret name a warning suggests for a literal: `<SERVER>_<KEY>`, upper
 * case, anything outside the grammar folded to `_`, so two servers with an
 * `Authorization` header do not get the same suggestion.
 */
export function suggestedSecretName(server: string, key: string): string {
  let name = `${server}_${key}`.toUpperCase().replace(/[^A-Z0-9_]/g, '_');
  if (/^[0-9]/.test(name)) name = `_${name}`;
  return name.slice(0, 128);
}

/** The loader's warning for a secret-looking literal in `env` or `headers`. */
export function literalWarning(server: string, field: 'env' | 'headers', key: string): string {
  const suggested = suggestedSecretName(server, key);
  return fillSentence(REFERENCE_SENTENCES.literal, { field, key, suggested, hint: secretsSetHint(suggested) });
}

/** A resolution failure as the connect-time error says it: where it sits, which server, and the sentence. */
export function atConnectSentence(server: string, field: string, key: string | undefined, sentence: string): string {
  return fillSentence(REFERENCE_SENTENCES.atConnect, { where: key ? `${field} ${key}` : field, server, sentence });
}

/**
 * A literal run beside a reference, as a report shows it (S-303, 2026-09-26):
 * a value that holds a reference used to be shown as written, so a literal
 * key sitting beside one was printed. Every whitespace-delimited token of the
 * run that looks like a secret, or is 16 or more characters long, is masked;
 * a scheme word (`Bearer`), a short path (`/x`) and the spaces around them
 * stay, so the report still says how the value is shaped.
 */
function maskLiteralRun(run: string): string {
  return run.replace(/\S+/g, (token) => (looksLikeSecret(token) || token.length >= 16 ? SECRET_MASK : token));
}

/** A value as a report shows it: each reference as written, every literal beside one masked as `maskLiteralRun` says, else the mask. */
export function maskValue(value: string): string {
  if (!hasReference(value)) return SECRET_MASK;
  return tokenizeReferences(value)
    .map((token) => (typeof token === 'string' ? maskLiteralRun(token) : token.text))
    .join('');
}

/**
 * An address as a report shows it: without any `user:password@` (kept when
 * it is references), and with every query value masked unless the value is a
 * reference, the host and path kept so the report still says which server.
 * Plain string work rather than `URL`, so both SDKs print the same bytes.
 * A reference anywhere in the address used to show the whole address as
 * written, literal query values and password included (S-303).
 */
export function maskUrl(url: string): string {
  const noUser = url.replace(/^([A-Za-z][A-Za-z0-9+.-]*:\/\/)([^/?#@]*)@/, (whole, scheme: string, userInfo: string) => (hasReference(userInfo) ? whole : scheme));
  const q = noUser.indexOf('?');
  if (q === -1) return noUser;
  const hashAt = noUser.indexOf('#', q);
  const query = hashAt === -1 ? noUser.slice(q + 1) : noUser.slice(q + 1, hashAt);
  const masked = query
    .split('&')
    .map((part) => {
      const eq = part.indexOf('=');
      if (eq === -1) return part;
      const value = part.slice(eq + 1);
      return hasReference(value) ? part : `${part.slice(0, eq)}=${SECRET_MASK}`;
    })
    .join('&');
  return `${noUser.slice(0, q + 1)}${masked}${hashAt === -1 ? '' : noUser.slice(hashAt)}`;
}

/** A map of values as a report shows it. */
export function maskMap(values: Record<string, string> | undefined): Record<string, string> | undefined {
  if (!values) return undefined;
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(values)) out[key] = maskValue(String(value));
  return out;
}

/** `text` with every resolved value replaced by the mask, longest first, so an error can be logged. */
export function maskText(text: string, values: readonly string[]): string {
  let out = text;
  for (const value of [...new Set(values)].filter((v) => v.length > 0).sort((a, b) => b.length - a.length)) {
    out = out.split(value).join(SECRET_MASK);
  }
  return out;
}
