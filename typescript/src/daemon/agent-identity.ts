/**
 * The signing identity a daemon-served agent holds (2026-09-27, the
 * a2a-delegate-webhooks lane): the same shape `serve()` gives an agent, so a
 * `cron:` webhook is signed exactly as a served agent signs one. The Python
 * twin is `cli/daemon/identity.py`; the shared fixture
 * `python/tests/fixtures/daemon/signedhooks.json` pins the paths and words.
 *
 * WHY. `serve()` persists an Ed25519 key per agent and publishes its key set
 * at `{agent URL}/.well-known/jwks.json` (`crypto/identity-store.ts`); the
 * daemon built its agents with no identity at all, so `deliver.ts` posted
 * every webhook unsigned and the run's record said so. A receiver that
 * verifies Web Bot Auth (`crypto/web-bot-auth-verify.ts`) could not accept a
 * scheduled report from a daemon.
 *
 * WHERE THE KEY LIVES (S-309, 2026-09-27, the agent-secrets lane): in the
 * store `serve()` keeps every agent's key in, `WEBAGENTS_KEYS_DIR` else
 * `~/.webagents/keys`, as `<name>.ed25519.jwk.json`, 0600 in a 0700
 * directory (`crypto/identity-store.ts`). For one day the daemon alone kept
 * it in the agent folder, at `.webagents/keys/` beside the agent file, so
 * that the file, its schedule state and its key travelled together. They
 * must not: the agent folder is where git commits go (nothing writes a
 * `.gitignore` there, so `git add -A` committed the private key), where
 * folders get copied and shared, and where the file tools look, while the
 * sandbox denies confined commands `~/.webagents` and only write-denies the
 * folder's own `.webagents/`. In a container, point `WEBAGENTS_KEYS_DIR` at
 * a mounted secret, as for `serve()`.
 *
 * A key found at the old place is moved into the store on load: the same
 * bytes, the same thumbprint, so the agent's identity does not change, and
 * one line says so. When the folder is a git work tree and the file was
 * ever tracked, a second line says to rotate the key, because a key that
 * was committed is a key someone else may hold. A key in both places
 * holding two different keys is a conflict (`AgentKeyConflictError`), and
 * the agent serves without an identity, as it does for any unusable key
 * file. A legacy file that is not a usable key is never moved.
 *
 * ONE NAME, ONE KEY, SAID OUT LOUD. The store names a file by agent name,
 * so two folders whose agent files share a name would share one key, and
 * silently. A non-secret sidecar, `<name>.ed25519.origin.json`, records the
 * folder a daemon key was first made for (or first used from), and a daemon
 * loading the same name from another folder says so once per process rather
 * than sharing in silence. The identity still loads: refusing would take an
 * agent down for a naming clash the operator can fix by renaming one file.
 * The fixture's `store` section pins the paths and the lines.
 *
 * THE ISSUER is `{public URL}/agents/{name}`: the daemon mounts every agent
 * under `/agents/:name`, and serves that agent's key set at
 * `/agents/:name/.well-known/jwks.json` (`server.ts`), so a verifier that
 * strips the well-known suffix off `Signature-Agent` fetches the set from
 * the daemon itself. The public URL is `WEBAGENTS_PUBLIC_URL` when set, the
 * daemon's own address otherwise. The signer refuses a loopback or plain
 * http issuer as it always has (`http-signature.ts`), so a daemon with no
 * public URL still posts unsigned and the record says why; set
 * `WEBAGENTS_PUBLIC_URL` to the https address the daemon is reachable at to
 * sign, as `serve()` asks.
 */

import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import type { AgentIdentity } from '../crypto/identity';
import { AgentKeyConflictError, AgentKeyFileError, agentKeyFileNames, loadOrCreateAgentIdentity } from '../crypto/identity-store';

/** Where the daemon kept an agent's key for one day, under the agent's folder (the fixture's `store.legacy_dir`). */
export const LEGACY_DAEMON_KEYS_DIR = path.join('.webagents', 'keys');
/** Where the daemon mounts its agents (the fixture's `issuer` and `jwks_route`). */
export const DAEMON_AGENTS_PREFIX = '/agents';

/** The lines the key move and the shared-name check say (the fixture's `store`). */
export const DAEMON_KEY_LINES = {
  moved: '[webagents] moved the signing key of agent {agent} from {legacy} to {store}: the same key with the same thumbprint, so its identity is unchanged.',
  tracked:
    '[webagents] {legacy} was committed to git in {folder}, so rotate the key: move {store} aside and restart to mint a new one, then remove the old key from the repository\'s history.',
  shared:
    '[webagents] the signing key of agent {agent} in {store} was made for {origin}; {folder} now signs with the same key. Give one of the two agents another name to give it a key of its own.',
} as const;

export function daemonKeyLine(kind: keyof typeof DAEMON_KEY_LINES, values: Record<string, string>): string {
  return DAEMON_KEY_LINES[kind].replace(/\{(agent|legacy|store|folder|origin)\}/g, (_match, key: string) => values[key] ?? `{${key}}`);
}

/** The store every served agent's key lives in: `WEBAGENTS_KEYS_DIR`, else `~/.webagents/keys` (`identity-store.ts`). */
export function daemonKeysDir(): string {
  const configured = (process.env.WEBAGENTS_KEYS_DIR ?? '').trim();
  return configured || path.join(os.homedir(), '.webagents', 'keys');
}

/** The old key directory of the agent whose file lives in `agentDir` (file comment). */
export function legacyDaemonKeysDir(agentDir: string): string {
  return path.join(agentDir, LEGACY_DAEMON_KEYS_DIR);
}

/** The non-secret sidecar beside an agent's key file, naming the folder the key was made for (the fixture's `store.origin_file`). */
export function agentKeyOriginFileName(agentName: string): string {
  return agentKeyFileNames(agentName).current.replace(/\.jwk\.json$/, '.origin.json');
}

/** The shared-name line is said once per process per agent and folder, not on every rescan. */
const saidShared = new Set<string>();

function realpathOr(target: string): string {
  try {
    return fs.realpathSync.native(target);
  } catch {
    return path.resolve(target);
  }
}

/** Whether `raw` is an Ed25519 private JWK, as the store writes one; `x` identifies the key. */
function jwkPublicHalf(raw: string): string | null {
  try {
    const jwk = JSON.parse(raw) as { kty?: unknown; crv?: unknown; d?: unknown; x?: unknown };
    if (jwk && jwk.kty === 'OKP' && jwk.crv === 'Ed25519' && typeof jwk.d === 'string' && typeof jwk.x === 'string') return jwk.x;
  } catch {
    // Not JSON.
  }
  return null;
}

/** Whether `file` (inside `agentDir`) was ever committed to, or is staged in, the git work tree `agentDir` is in. */
function wasTrackedByGit(agentDir: string, file: string): boolean {
  const git = (args: string[]) =>
    execFileSync('git', ['-C', agentDir, ...args], { stdio: ['ignore', 'pipe', 'ignore'], timeout: 5000 }).toString().trim();
  try {
    if (git(['rev-parse', '--is-inside-work-tree']) !== 'true') return false;
    const relative = path.relative(agentDir, file);
    if (git(['log', '--all', '--format=%H', '-n', '1', '--', relative])) return true;
    return git(['ls-files', '--', relative]).length > 0;
  } catch {
    return false;
  }
}

/**
 * Move the keys an earlier daemon left in `<agentDir>/.webagents/keys` into
 * `store` (file comment): the current key and a previous one held for
 * rotation. Throws `AgentKeyFileError` for a legacy file that is not a
 * usable key (it is left where it is) and `AgentKeyConflictError` when the
 * store already holds a different key under the same name.
 */
async function adoptLegacyKeys(agentName: string, agentDir: string, store: string): Promise<void> {
  const names = agentKeyFileNames(agentName);
  // `WEBAGENTS_KEYS_DIR` pointed at the folder itself: nothing to move, and
  // a move onto itself would delete the key.
  if (realpathOr(store) === realpathOr(legacyDaemonKeysDir(agentDir))) return;
  for (const file of [names.current, names.previous]) {
    const legacy = path.join(legacyDaemonKeysDir(agentDir), file);
    let raw: string;
    try {
      raw = fs.readFileSync(legacy, 'utf8');
    } catch (err) {
      if ((err as { code?: string }).code === 'ENOENT') continue;
      throw new AgentKeyFileError(legacy, `could not be read (${(err as Error).message})`, err);
    }
    const x = jwkPublicHalf(raw);
    if (x === null) throw new AgentKeyFileError(legacy, 'is not an Ed25519 private JWK (kty "OKP", crv "Ed25519", with d and x)');
    const target = path.join(store, file);
    let theirs: string | null = null;
    try {
      theirs = fs.readFileSync(target, 'utf8');
    } catch (err) {
      if ((err as { code?: string }).code !== 'ENOENT') throw new AgentKeyFileError(target, `could not be read (${(err as Error).message})`, err);
    }
    if (theirs !== null) {
      if (jwkPublicHalf(theirs) !== x) throw new AgentKeyConflictError(legacy, target);
      // The store already holds this very key: the folder copy is a duplicate.
    } else {
      fs.mkdirSync(store, { recursive: true, mode: 0o700 });
      try {
        fs.chmodSync(store, 0o700);
      } catch {
        // A directory this process does not own: the file mode still holds.
      }
      // Created exclusively, 0600, never replacing: the same rule as a new key.
      const fd = fs.openSync(target, 'wx', 0o600);
      try {
        fs.writeSync(fd, raw);
        fs.fsyncSync(fd);
      } finally {
        fs.closeSync(fd);
      }
    }
    fs.unlinkSync(legacy);
    try {
      fs.rmdirSync(path.dirname(legacy));
    } catch {
      // Not empty, or already gone: either is fine.
    }
    console.log(daemonKeyLine('moved', { agent: agentName, legacy, store: target }));
    if (file === names.current && wasTrackedByGit(agentDir, legacy)) {
      console.log(daemonKeyLine('tracked', { legacy, folder: agentDir, store: target }));
    }
  }
}

/** Record the folder this agent's key was made for, or say that another folder made it (file comment). */
function recordKeyOrigin(agentName: string, agentDir: string, store: string): void {
  const file = path.join(store, agentKeyOriginFileName(agentName));
  const folder = realpathOr(agentDir);
  let existing: { folder?: unknown } | null = null;
  try {
    existing = JSON.parse(fs.readFileSync(file, 'utf8')) as { folder?: unknown };
  } catch (err) {
    if ((err as { code?: string }).code !== 'ENOENT') return; // Unreadable or not JSON: not ours to rewrite.
  }
  if (existing === null) {
    try {
      fs.mkdirSync(store, { recursive: true, mode: 0o700 });
      fs.writeFileSync(file, `${JSON.stringify({ agent: agentName, folder, recorded_at: new Date().toISOString() })}\n`, { flag: 'wx' });
    } catch {
      // A store this process cannot write (an ephemeral key was said already): nothing to record.
    }
    return;
  }
  if (typeof existing.folder === 'string' && existing.folder !== folder) {
    const once = `${agentName}\n${folder}`;
    if (saidShared.has(once)) return;
    saidShared.add(once);
    const keyFile = path.join(store, agentKeyFileNames(agentName).current);
    console.log(daemonKeyLine('shared', { agent: agentName, store: keyFile, origin: existing.folder, folder }));
  }
}

/** The address the daemon publishes for its agents: `WEBAGENTS_PUBLIC_URL`, else its own bind address (the fixture's `public_url_cases`). */
export function daemonPublicUrl(options: { hostname: string; port: number; publicUrl?: string }): string {
  const configured = (options.publicUrl ?? (typeof process !== 'undefined' ? process.env?.WEBAGENTS_PUBLIC_URL : undefined) ?? '').trim();
  if (configured) return configured.replace(/\/+$/, '');
  const host = options.hostname.includes(':') && !options.hostname.startsWith('[') ? `[${options.hostname}]` : options.hostname;
  return `http://${host}:${options.port}`;
}

/** The agent URL, the principal every signature names: `{publicUrl}/agents/{name}`. */
export function daemonAgentUrl(publicUrl: string, agentName: string): string {
  return `${publicUrl.replace(/\/+$/, '')}${DAEMON_AGENTS_PREFIX}/${agentName}`;
}

/**
 * Load, or create and persist, the identity of the agent `definition`
 * declares, in the store `serve()` uses (file comment), after moving any key
 * an earlier daemon left in the agent folder and noting the folder the key
 * belongs to. Throws `AgentKeyFileError` when a key file exists and cannot
 * be used, as `serve()` does: a key the platform may have pinned is never
 * replaced.
 */
export async function daemonAgentIdentity(definition: { name: string; filePath: string }, publicUrl: string): Promise<AgentIdentity> {
  const agentDir = path.dirname(path.resolve(definition.filePath));
  const store = daemonKeysDir();
  await adoptLegacyKeys(definition.name, agentDir, store);
  const identity = await loadOrCreateAgentIdentity(definition.name, {
    issuer: daemonAgentUrl(publicUrl, definition.name),
  });
  recordKeyOrigin(definition.name, agentDir, store);
  return identity;
}
