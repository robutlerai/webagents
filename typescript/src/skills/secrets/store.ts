/**
 * Secret storage backends, with no agent machinery attached.
 *
 * WHY THIS IS SEPARATE FROM `skill.ts`: the first consumer is not an LLM, it
 * is `registerWithPlatform`, which runs at boot before any agent exists. A
 * store that could only be reached through a `Skill` would have forced the
 * server to construct an agent to persist a token. `skill.ts` is a thin
 * decorator wrapper over what lives here.
 *
 * WHY IT EXISTS AT ALL: `registerWithPlatform` now answers with a PLATFORM
 * BEARER valid for seven days (`EXPIRES_IN_SECONDS = 7 * 24 * 60 * 60` on the
 * portal's `/api/auth/cli/token`), and the examples had nowhere to put it but
 * a dotfile or `WEBAGENTS_AGENT_TOKEN`. That token is worse to leak than it
 * looks, for two reasons recorded in the portal's security log:
 *
 *   * S-037: it carries `agents:own` and the signer sets no `jti`, so there is
 *     no revocation lever at all. Rotating the key published on the agent card
 *     does NOT invalidate it, because nothing on its validation path reads
 *     that key. A leaked one is good for a week.
 *   * S-034: the AOAuth assertion it was exchanged for needs no `exp` and has
 *     no replay defence, so the exchange can be repeated.
 *
 * A credential that cannot be revoked is one you must not lose in the first
 * place, which is the whole argument for putting it in the operating system's
 * keystore rather than in a file next to the code.
 *
 * BACKENDS, in the order they are tried:
 *
 *   1. `keystore` - `@napi-rs/keyring`, which wraps the Rust `keyring` crate:
 *      macOS Keychain, Linux Secret Service (libsecret over DBus) and Windows
 *      Credential Manager behind one API. It is an OPTIONAL dependency loaded
 *      by dynamic import, deliberately. This SDK ships browser and extension
 *      agents and has zero native dependencies today; making a native addon
 *      mandatory would break bundling for consumers that never touch a
 *      keystore. Optional plus dynamic import means an install on an
 *      unsupported platform degrades instead of failing.
 *   2. `file` - `~/.webagents/secrets/<namespace>.json`, 0600 in a 0700
 *      directory. PLAINTEXT. It exists because a container or a headless CI
 *      box has no keystore and an agent still has to run there.
 *
 * The fallback is the part that has to be got right, because the failure mode
 * worth avoiding is not "no keystore", it is "the developer believed there was
 * one". So it announces itself three times over: once when the store opens
 * (naming the reason, the fix and the path), again on every write, and a third
 * time in the `backend`/`warning` fields of every tool result, which is the
 * copy an LLM reading the result actually sees. Set
 * `WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1` (or `requireKeystore: true`) and the
 * fallback becomes an error instead, which is the lever for a deployment that
 * would rather fail than write a bearer to disk.
 *
 * NOTHING HERE EVER LOGS A SECRET VALUE. Names only, at every level.
 */

/** Which backend a value actually landed in. */
export type SecretBackendKind = 'keystore' | 'file';

/** What the OS keystore is doing, and why, in a form worth printing. */
export interface SecretBackendStatus {
  /** The backend in use right now. */
  backend: SecretBackendKind;
  /** True only when secrets are in the operating system's own keystore. */
  keystore: boolean;
  /** Namespace the store is scoped to. Two namespaces never see each other. */
  namespace: string;
  /**
   * Why the keystore is unavailable. Present only when `backend` is `file`,
   * and it is the actual failure text rather than a generic sentence, because
   * "no DBus session" and "package not installed" want different fixes.
   */
  reason?: string;
  /** Where the plaintext file lives. Present only when `backend` is `file`. */
  path?: string;
  /**
   * One line, already worded for a human, when the store is NOT in a keystore.
   * Handed back through every tool result so the warning survives into a
   * transcript that nobody was watching the process logs for.
   */
  warning?: string;
}

export interface SecretStoreOptions {
  /**
   * Collision boundary. Two agents on one machine must not read each other's
   * secrets, and the OS keystore is machine-wide, so everything is filed under
   * `webagents:<namespace>`.
   *
   * Defaults to `WEBAGENTS_SECRETS_NAMESPACE`, then `'webagents'`. `serve()`
   * passes the agent name. Two agents that share a name DO share secrets;
   * that is a property of the name, not a bug to work around here.
   */
  namespace?: string;
  /**
   * Refuse the plaintext fallback rather than take it. Defaults to
   * `WEBAGENTS_SECRETS_REQUIRE_KEYSTORE` being `1` or `true`.
   */
  requireKeystore?: boolean;
  /**
   * Directory for the fallback file. Defaults to `WEBAGENTS_SECRETS_DIR`, then
   * `~/.webagents/secrets`, alongside the keys directory the identity store
   * already uses.
   */
  secretsDir?: string;
  /**
   * Suppress the console warnings. For tests and for a caller that has already
   * surfaced `status()` itself. It does NOT suppress the `warning` field on
   * results, which is the channel that must never be switchable off.
   */
  quiet?: boolean;
  /**
   * `'auto'` (default) probes for a keystore and falls back. `'file'` skips
   * the probe and uses the plaintext file deliberately.
   *
   * Defaults to `WEBAGENTS_SECRETS_BACKEND`. It exists for two reasons. One:
   * an operator who has decided the file is what they want should be able to
   * say so rather than arrange for the keystore to fail. Two, and this is the
   * one that made it non-negotiable: without it the fallback is UNTESTABLE on
   * any machine that has a working keystore, so the path that matters most
   * would only ever be exercised where it matters least. Choosing `'file'`
   * does not quieten a single warning.
   */
  backend?: 'auto' | 'file';
}

/**
 * The three operations `registerWithPlatform` needs. Declared here and
 * duplicated structurally in `server/registration.ts` on purpose: the server
 * layer imports nothing from `skills/`, so registration accepts anything of
 * this shape without dragging the keystore into its module graph.
 */
export interface SecretStoreLike {
  get(name: string): Promise<string | null>;
  set(name: string, value: string): Promise<SecretBackendKind>;
  delete(name: string): Promise<boolean>;
}

/** Thrown when `requireKeystore` is set and there is no keystore. */
export class KeystoreUnavailableError extends Error {
  constructor(reason: string) {
    super(
      `no OS keystore available (${reason}), and WEBAGENTS_SECRETS_REQUIRE_KEYSTORE ` +
        'refuses the plaintext fallback. Install the optional keystore support ' +
        '(`npm install @napi-rs/keyring`) on a machine with a keystore, or unset ' +
        'the variable to accept a 0600 file.',
    );
    this.name = 'KeystoreUnavailableError';
  }
}

/**
 * Secret names go into a keychain service key and a JSON object key, so keep
 * them boring. Rejecting rather than sanitising: silently mapping two distinct
 * names onto one storage key is how a `set` overwrites a secret its caller
 * never named.
 */
const NAME_PATTERN = /^[A-Za-z0-9._-]{1,128}$/;

function assertValidName(name: string): void {
  if (!NAME_PATTERN.test(name)) {
    throw new Error(
      `invalid secret name ${JSON.stringify(name)}: use 1-128 characters from ` +
        'A-Z a-z 0-9 . _ -',
    );
  }
}

function envVar(name: string): string | undefined {
  return typeof process !== 'undefined' ? process.env?.[name] : undefined;
}

function envFlag(name: string): boolean {
  const raw = (envVar(name) ?? '').trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes';
}

/** The keychain service key. One string, so the two backends agree on it. */
export function serviceKey(namespace: string): string {
  return `webagents:${namespace}`;
}

// ---------------------------------------------------------------------------
// Keystore backend
// ---------------------------------------------------------------------------

/** The slice of `@napi-rs/keyring` this uses. */
interface KeyringEntry {
  setPassword(password: string): void;
  getPassword(): string | null;
  deletePassword(): boolean;
}
interface KeyringModule {
  Entry: new (service: string, username: string) => KeyringEntry;
}

/**
 * Resolve `@napi-rs/keyring` once per process.
 *
 * Cached as a settled result rather than as a promise-per-call because the
 * miss is the common case (the package is optional) and a rejected dynamic
 * import re-runs the resolver every time otherwise.
 */
let keyringProbe: Promise<{ mod: KeyringModule | null; reason: string }> | null = null;

function loadKeyring(): Promise<{ mod: KeyringModule | null; reason: string }> {
  if (keyringProbe) return keyringProbe;
  keyringProbe = (async () => {
    if (typeof process === 'undefined' || !process.versions?.node) {
      return { mod: null, reason: 'not running under Node, so there is no OS keystore to reach' };
    }
    try {
      // Indirected through a variable so a bundler resolving static specifiers
      // does not try to pull a native addon into a browser build.
      const specifier = '@napi-rs/keyring';
      const mod = (await import(/* @vite-ignore */ specifier)) as unknown as KeyringModule;
      if (typeof mod?.Entry !== 'function') {
        return { mod: null, reason: '@napi-rs/keyring loaded but exports no Entry class' };
      }
      return { mod, reason: '' };
    } catch (err) {
      const message = (err as Error)?.message ?? String(err);
      // Two very different situations read the same to a developer unless the
      // message says which: the optional package was never installed, or it
      // installed fine and the platform has no usable keystore behind it (a
      // Linux container with no DBus session is the common one).
      const missing = /Cannot find module|ERR_MODULE_NOT_FOUND|Failed to load native binding/i.test(
        message,
      );
      return {
        mod: null,
        reason: missing
          ? 'optional package @napi-rs/keyring is not installed (npm install @napi-rs/keyring)'
          : `@napi-rs/keyring failed to load: ${message}`,
      };
    }
  })();
  return keyringProbe;
}

// ---------------------------------------------------------------------------
// File backend
// ---------------------------------------------------------------------------

/** Filesystem-safe file stem, matching `identity-store.ts`. */
function fileStem(namespace: string): string {
  return namespace.replace(/[^A-Za-z0-9._-]/g, '_');
}

async function resolveSecretsDir(configured?: string): Promise<string> {
  const os = await import('node:os');
  const path = await import('node:path');
  const dir = configured || envVar('WEBAGENTS_SECRETS_DIR');
  return dir || path.join(os.homedir(), '.webagents', 'secrets');
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/**
 * Named secrets in the OS keystore, or in a 0600 file that says so.
 *
 * Build one with {@link openSecretStore}, which probes the keystore up front
 * so `status()` is answerable before the first read.
 */
export class SecretStore implements SecretStoreLike {
  readonly namespace: string;
  private readonly keyring: KeyringModule | null;
  private readonly unavailableReason: string;
  private readonly filePath: string;
  private readonly quiet: boolean;
  /** One warning per process per store, not one per operation. */
  private warnedOnOpen = false;

  constructor(init: {
    namespace: string;
    keyring: KeyringModule | null;
    unavailableReason: string;
    filePath: string;
    quiet: boolean;
  }) {
    this.namespace = init.namespace;
    this.keyring = init.keyring;
    this.unavailableReason = init.unavailableReason;
    this.filePath = init.filePath;
    this.quiet = init.quiet;
  }

  /** Where a value would go right now, and why. */
  status(): SecretBackendStatus {
    if (this.keyring) {
      return { backend: 'keystore', keystore: true, namespace: this.namespace };
    }
    return {
      backend: 'file',
      keystore: false,
      namespace: this.namespace,
      reason: this.unavailableReason,
      path: this.filePath,
      warning: this.fallbackWarning(),
    };
  }

  /**
   * The sentence a developer has to read. Names the path, because "not in a
   * keystore" without "here is where it is instead" is not actionable.
   */
  private fallbackWarning(): string {
    return (
      `secrets for "${this.namespace}" are NOT in an OS keystore: ${this.unavailableReason}. ` +
      `They are stored as PLAINTEXT in ${this.filePath} (0600). Anything that can read ` +
      'that file, including a backup and any process running as this user, can read the ' +
      'secrets. Set WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1 to refuse this fallback.'
    );
  }

  /**
   * Print the fallback warning once for this store.
   *
   * Called by {@link openSecretStore} at OPEN rather than at first access,
   * deliberately. A warning that waits for the first read arrives at a moment
   * the developer has no reason to be watching, and for a store that is only
   * ever written to at shutdown it may not arrive at all.
   */
  warnIfFallback(): void {
    if (this.keyring || this.quiet || this.warnedOnOpen) return;
    this.warnedOnOpen = true;
    console.warn(`[webagents] ${this.fallbackWarning()}`);
  }

  private entry(name: string): KeyringEntry {
    if (!this.keyring) throw new KeystoreUnavailableError(this.unavailableReason);
    return new this.keyring.Entry(serviceKey(this.namespace), name);
  }

  private async readFileMap(): Promise<Record<string, string>> {
    try {
      const fs = await import('node:fs/promises');
      const raw = await fs.readFile(this.filePath, 'utf-8');
      const parsed = JSON.parse(raw) as unknown;
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
      const out: Record<string, string> = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof value === 'string') out[key] = value;
      }
      return out;
    } catch {
      // Missing or unreadable both mean "no secrets here". A corrupt file is
      // deliberately not fatal: refusing to boot because a cache of a
      // re-derivable token failed to parse is worse than re-deriving it.
      return {};
    }
  }

  private async writeFileMap(map: Record<string, string>): Promise<void> {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    await fs.mkdir(path.dirname(this.filePath), { recursive: true, mode: 0o700 });
    // `mode` on writeFile only applies when the file is CREATED, so an
    // existing file keeps whatever mode it had. chmod after the write repairs
    // one an earlier version left at 0644, matching what `crypto/jwks.py`
    // does for the agent key.
    await fs.writeFile(this.filePath, JSON.stringify(map, null, 2), { mode: 0o600 });
    await fs.chmod(this.filePath, 0o600).catch(() => {});
  }

  /** The value, or `null` when there is none. Never logged. */
  async get(name: string): Promise<string | null> {
    assertValidName(name);
    if (this.keyring) {
      try {
        return this.entry(name).getPassword() ?? null;
      } catch {
        // The Rust crate raises for "no entry" on some backends rather than
        // returning null. Absent is not an error to a caller.
        return null;
      }
    }
    this.warnIfFallback();
    const map = await this.readFileMap();
    return Object.prototype.hasOwnProperty.call(map, name) ? map[name] : null;
  }

  /** Store a value. Returns the backend it actually landed in. */
  async set(name: string, value: string): Promise<SecretBackendKind> {
    assertValidName(name);
    if (typeof value !== 'string' || value.length === 0) {
      throw new Error(`refusing to store an empty value for ${name}`);
    }
    if (this.keyring) {
      this.entry(name).setPassword(value);
      return 'keystore';
    }
    this.warnIfFallback();
    if (!this.quiet) {
      // Every write, not just the first: a developer who scrolled past the
      // open-time warning still sees this one next to the operation that
      // caused it. The NAME is printed, never the value.
      console.warn(
        `[webagents] wrote secret "${name}" as PLAINTEXT to ${this.filePath} ` +
          '(no OS keystore available)',
      );
    }
    const map = await this.readFileMap();
    map[name] = value;
    await this.writeFileMap(map);
    return 'file';
  }

  /**
   * Remove a secret. Sweeps BOTH backends and returns whether anything went.
   *
   * Both, because the interesting case is a secret written to the file on a
   * headless box that later grew a keystore: deleting only the active backend
   * leaves the plaintext copy behind, which is exactly the state a developer
   * calling `delete` believes they have escaped.
   */
  async delete(name: string): Promise<boolean> {
    assertValidName(name);
    let removed = false;
    if (this.keyring) {
      try {
        removed = this.entry(name).deletePassword();
      } catch {
        removed = false;
      }
    }
    const map = await this.readFileMap();
    if (Object.prototype.hasOwnProperty.call(map, name)) {
      delete map[name];
      await this.writeFileMap(map);
      removed = true;
    }
    return removed;
  }

  /**
   * Names of the secrets this store can enumerate. NEVER values.
   *
   * The keystore backend cannot enumerate: `@napi-rs/keyring` has
   * `findCredentials`, but it is not implemented on every platform the crate
   * supports and a list that is silently short is worse than no list. So the
   * store keeps its own index of names it has written, in a file that holds
   * no secret material, and says so through `complete`.
   */
  async list(): Promise<{ names: string[]; complete: boolean }> {
    if (!this.keyring) {
      const map = await this.readFileMap();
      return { names: Object.keys(map).sort(), complete: true };
    }
    const index = await this.readIndex();
    return { names: index, complete: false };
  }

  private get indexPath(): string {
    return `${this.filePath.replace(/\.json$/, '')}.index.json`;
  }

  private async readIndex(): Promise<string[]> {
    try {
      const fs = await import('node:fs/promises');
      const raw = await fs.readFile(this.indexPath, 'utf-8');
      const parsed = JSON.parse(raw) as unknown;
      return Array.isArray(parsed) ? parsed.filter((n): n is string => typeof n === 'string').sort() : [];
    } catch {
      return [];
    }
  }

  /** Record or forget a name in the keystore-mode index. Names only. */
  async noteIndex(name: string, present: boolean): Promise<void> {
    if (!this.keyring) return;
    const current = new Set(await this.readIndex());
    if (present) current.add(name);
    else current.delete(name);
    try {
      const fs = await import('node:fs/promises');
      const path = await import('node:path');
      await fs.mkdir(path.dirname(this.indexPath), { recursive: true, mode: 0o700 });
      await fs.writeFile(this.indexPath, JSON.stringify([...current].sort(), null, 2), {
        mode: 0o600,
      });
    } catch {
      // The index is a convenience. Losing it must never fail a store that
      // already succeeded.
    }
  }
}

/**
 * Open a secret store, probing the keystore first.
 *
 * Throws {@link KeystoreUnavailableError} when there is no keystore and the
 * caller (or `WEBAGENTS_SECRETS_REQUIRE_KEYSTORE`) refused the fallback.
 */
export async function openSecretStore(
  options: SecretStoreOptions = {},
): Promise<SecretStore> {
  const namespace = (
    options.namespace ||
    envVar('WEBAGENTS_SECRETS_NAMESPACE') ||
    'webagents'
  ).trim();
  const requireKeystore =
    options.requireKeystore ?? envFlag('WEBAGENTS_SECRETS_REQUIRE_KEYSTORE');
  const forced =
    options.backend ?? ((envVar('WEBAGENTS_SECRETS_BACKEND') ?? '').trim().toLowerCase() || 'auto');

  const { mod, reason } =
    forced === 'file'
      ? { mod: null, reason: 'the file backend was selected explicitly (WEBAGENTS_SECRETS_BACKEND)' }
      : await loadKeyring();

  if (!mod && requireKeystore) throw new KeystoreUnavailableError(reason);

  // Always resolved, even in keystore mode. `delete()` sweeps the file so a
  // plaintext copy written before the keystore existed does not survive, and
  // the name index lives beside it; both need a real path, and an empty one
  // silently resolved against the process working directory.
  const path = await import('node:path');
  const dir = await resolveSecretsDir(options.secretsDir);
  const filePath = path.join(dir, `${fileStem(namespace)}.json`);

  const store = new SecretStore({
    namespace,
    keyring: mod,
    unavailableReason: reason,
    filePath,
    quiet: options.quiet ?? false,
  });
  // At open, not at first use. See `warnIfFallback`.
  store.warnIfFallback();
  return store;
}
