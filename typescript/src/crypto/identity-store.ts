/**
 * Persistent agent identity.
 *
 * WHY THIS EXISTS AT ALL: the platform pins the agent's key at registration
 * and verifies every later request against what it pinned. Since 2026-09-17
 * (ADR 0038 step 5) what it pins is the key set it fetched from
 * `{agentUrl}/.well-known/jwks.json`, keyed by RFC 7638 thumbprint, and a
 * request signs with `keyid` = that thumbprint; before that day it was the
 * SPKI PEM read off the card. Either way a key generated per boot works
 * exactly until the first restart and then fails every verification, with a
 * card and a key set that still look perfectly correct. The Python SDK has
 * always persisted (`crypto/jwks.py` `ensure_ed25519_key`); this is the TS
 * counterpart, and it belongs to the server rather than to any one entry
 * point: it moved here out of a wrapper entry point that no longer exists.
 *
 * Stored as a private JWK (which carries both `d` and `x`, so the public half
 * is derived on load) with 0600 permissions in a 0700 directory, under
 * `WEBAGENTS_KEYS_DIR` or `~/.webagents/keys`, the same location the Python
 * SDK uses. Two files per agent, `<stem>` being the agent name made
 * filesystem-safe:
 *
 *   <stem>.ed25519.jwk.json            the current key
 *   <stem>.ed25519.previous.jwk.json   the ONE key still held while rotating
 *
 * A MISSING FILE IS THE ONLY REASON TO GENERATE (S-185, fixed 2026-09-19).
 * Until that day one `try` wrapped the read, the JSON parse and the import,
 * and its `catch` stayed silent for any error without a `code`. A JSON
 * `SyntaxError` has none, so a truncated key file (the write was not atomic)
 * fell through to key generation and `writeFile` OVERWROTE the only copy of
 * the agent's pinned private key: under URL control the platform then sees an
 * unannounced rotation, and under the continuity rule the agent is locked
 * out with no co-signer. Now `ENOENT` alone means "no key yet". Every other
 * failure (permissions, a directory in the way, bad JSON, a JWK that is not
 * an Ed25519 private key, an import error) throws `AgentKeyFileError` naming
 * the file, and nothing is written. The operator restores the file, or moves
 * it aside on purpose; the SDK never makes that decision for them. The
 * Python store has always raised here.
 *
 * A NEW KEY NEVER REPLACES A FILE. It is written to a temporary file in the
 * same directory (created exclusively, 0600, flushed), then hard-linked to
 * its final name, which fails with `EEXIST` rather than replacing: two
 * processes booting the same agent for the first time end up holding the
 * SAME key (the loser loads the winner's file) instead of one of them
 * holding a key that is no longer on disk. Where hard links are unsupported
 * the fallback is a rename. A crash mid-write leaves a `.tmp` beside an
 * absent key file, never half a key under the real name.
 *
 * ROTATION (W2 design section 2.5, parity with Python 2026-09-19). The store
 * read exactly one JWK and never filled `previousKeys`, so the dual-signing
 * rotation `AgentIdentity` and `signMessage` implement was unreachable for
 * anyone using `serve()`. The previous-key file is now loaded as the held
 * key: the key set lists it after the current one and every request is
 * co-signed with it. To rotate: rename the current file to
 * `<stem>.ed25519.previous.jwk.json`, restart (a new current key is
 * generated), and delete the previous file once the platform has admitted
 * the new key. One previous file, so never more than `MAX_HELD_KEYS` keys.
 *
 * PERMISSIONS ARE REPAIRED ON LOAD. `mode` on `mkdir` and `writeFile`
 * applies only at creation, so a key an earlier version (or a careless
 * `cp`) left group- or world-readable stayed that way. An existing key file
 * is tightened to 0600 and its directory to 0700, best effort: a directory
 * this process does not own is left alone, and the key still loads.
 */

import { AgentIdentity, type HeldKeyPair, type KeyLike } from './identity';
import { generateKeyPair, exportJWK, importJWK, type JWK } from 'jose';

export interface IdentityStoreOptions {
  /**
   * The agent URL recorded on the identity: the principal the platform
   * registers, where the key set and the card are served. `serve()` composes
   * it as `publicUrl + basePath`.
   */
  issuer: string;
  /**
   * Where the key lives. Defaults to `WEBAGENTS_KEYS_DIR`, then
   * `~/.webagents/keys`. Pass `null` for an explicitly ephemeral key (tests).
   */
  keysDir?: string | null;
}

/**
 * An agent key file exists and cannot be used (file comment, "A MISSING FILE
 * IS THE ONLY REASON TO GENERATE"). `file` is the path; `cause` is the
 * underlying error when there was one. Thrown instead of generating, because
 * generating would replace the identity the platform pinned.
 */
export class AgentKeyFileError extends Error {
  readonly file: string;

  constructor(file: string, problem: string, cause?: unknown) {
    super(
      `[webagents] the agent key file ${file} ${problem}. It holds the identity the platform pinned, ` +
        'so it is never replaced automatically: restore it from a backup, or move it aside yourself to ' +
        'have a NEW key generated (the platform will see that as a key rotation).',
    );
    this.name = 'AgentKeyFileError';
    this.file = file;
    if (cause !== undefined) (this as { cause?: unknown }).cause = cause;
  }
}

/** Filesystem-safe file stem for an agent name. */
function keyFileStem(agentName: string): string {
  return agentName.replace(/[^A-Za-z0-9._-]/g, '_');
}

/** The two files an agent's identity lives in (file comment). Exported for operators' tooling and the tests. */
export function agentKeyFileNames(agentName: string): { current: string; previous: string } {
  const stem = keyFileStem(agentName);
  return { current: `${stem}.ed25519.jwk.json`, previous: `${stem}.ed25519.previous.jwk.json` };
}

type Fs = typeof import('node:fs/promises');
type Path = typeof import('node:path');

function errorCode(err: unknown): string | undefined {
  const code = (err as { code?: unknown } | null)?.code;
  return typeof code === 'string' ? code : undefined;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Tighten an existing key file to 0600 and its directory to 0700 (file
 * comment, "PERMISSIONS ARE REPAIRED ON LOAD"). Best effort and silent on
 * failure; POSIX only, since Windows reports synthetic mode bits.
 */
async function repairPermissions(fs: Fs, path: Path, file: string): Promise<void> {
  if (typeof process !== 'undefined' && process.platform === 'win32') return;
  for (const [target, mode] of [
    [file, 0o600],
    [path.dirname(file), 0o700],
  ] as const) {
    try {
      const current = (await fs.stat(target)).mode & 0o777;
      if ((current & 0o077) !== 0) {
        await fs.chmod(target, mode);
        console.warn(
          `[webagents] ${target} was accessible to other users (mode ${current.toString(8)}); tightened to ${mode.toString(8)}`,
        );
      }
    } catch {
      // A directory this process does not own, or a filesystem without
      // modes: the key still loads, and the file mode is what matters most.
    }
  }
}

/**
 * Read one key file. `null` when, and only when, the file does not exist;
 * any other failure throws `AgentKeyFileError` and nothing is generated.
 */
async function readKeyFile(fs: Fs, path: Path, file: string): Promise<HeldKeyPair | null> {
  let raw: string;
  try {
    raw = await fs.readFile(file, 'utf8');
  } catch (err) {
    if (errorCode(err) === 'ENOENT') return null;
    throw new AgentKeyFileError(file, `could not be read (${errorMessage(err)})`, err);
  }
  let jwk: JWK;
  try {
    jwk = JSON.parse(raw) as JWK;
  } catch (err) {
    throw new AgentKeyFileError(file, `is not valid JSON (${errorMessage(err)}; a truncated write looks like this)`, err);
  }
  if (
    typeof jwk !== 'object' ||
    jwk === null ||
    jwk.kty !== 'OKP' ||
    jwk.crv !== 'Ed25519' ||
    typeof jwk.d !== 'string' ||
    typeof jwk.x !== 'string'
  ) {
    throw new AgentKeyFileError(file, 'is not an Ed25519 private JWK (kty "OKP", crv "Ed25519", with d and x)');
  }
  let pair: HeldKeyPair;
  try {
    const privateKey = (await importJWK(jwk, 'EdDSA')) as KeyLike;
    const { d: _d, ...publicJwk } = jwk;
    const publicKey = (await importJWK(publicJwk, 'EdDSA')) as KeyLike;
    pair = { privateKey, publicKey };
  } catch (err) {
    throw new AgentKeyFileError(file, `could not be imported as an Ed25519 key (${errorMessage(err)})`, err);
  }
  await repairPermissions(fs, path, file);
  return pair;
}

function randomSuffix(): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Persist a new private JWK under `file` without ever replacing a file
 * (file comment, "A NEW KEY NEVER REPLACES A FILE"). Returns `'created'`, or
 * `'exists'` when another process created `file` first, in which case the
 * caller loads THAT key and drops its own.
 */
async function writeKeyFileOnce(fs: Fs, path: Path, file: string, jwk: JWK): Promise<'created' | 'exists'> {
  const dir = path.dirname(file);
  await fs.mkdir(dir, { recursive: true, mode: 0o700 });
  // `mode` on mkdir applies only when the directory is created.
  await fs.chmod(dir, 0o700).catch(() => undefined);
  const tmp = path.join(dir, `.${path.basename(file)}.${randomSuffix()}.tmp`);
  try {
    // `wx`: created exclusively, so the temporary name is never someone else's file.
    const handle = await fs.open(tmp, 'wx', 0o600);
    try {
      await handle.writeFile(JSON.stringify(jwk));
      await handle.sync();
    } finally {
      await handle.close();
    }
    try {
      await fs.link(tmp, file);
    } catch (err) {
      if (errorCode(err) === 'EEXIST') return 'exists';
      // No hard links here (some network and FUSE mounts): rename instead.
      // The file was absent a moment ago, which is the whole reason a key
      // was generated.
      await fs.rename(tmp, file);
      return 'created';
    }
    return 'created';
  } finally {
    await fs.unlink(tmp).catch(() => undefined);
  }
}

/**
 * Load this agent's Ed25519 key from disk, or create and persist one, and
 * hold the previous key beside it when its file is present (file comment).
 *
 * Throws `AgentKeyFileError` when a key file exists and cannot be used.
 */
export async function loadOrCreateAgentIdentity(
  agentName: string,
  options: IdentityStoreOptions,
): Promise<AgentIdentity> {
  const { issuer, keysDir: keysDirOption } = options;
  const makeIdentity = (current?: HeldKeyPair, previous?: HeldKeyPair | null) =>
    new AgentIdentity({
      agentId: agentName,
      issuer,
      privateKey: current?.privateKey,
      publicKey: current?.publicKey,
      previousKeys: previous ? [previous] : undefined,
    });

  if (keysDirOption === null) {
    const identity = makeIdentity();
    await identity.initialize();
    return identity;
  }

  let fs: Fs;
  let path: Path;
  let os: typeof import('node:os');
  try {
    fs = await import('node:fs/promises');
    path = await import('node:path');
    os = await import('node:os');
  } catch {
    // Non-Node runtime: no filesystem, ephemeral key (and say so).
    console.warn(
      '[webagents] no filesystem available; using an EPHEMERAL agent key. ' +
        'Platform registration will break on restart.',
    );
    const identity = makeIdentity();
    await identity.initialize();
    return identity;
  }

  const keysDir =
    keysDirOption ??
    (typeof process !== 'undefined' ? process.env?.WEBAGENTS_KEYS_DIR : undefined) ??
    path.join(os.homedir(), '.webagents', 'keys');
  const names = agentKeyFileNames(agentName);
  const keyFile = path.join(keysDir, names.current);
  const previousFile = path.join(keysDir, names.previous);

  // Both reads throw on anything but a missing file, BEFORE anything is
  // generated or written: an unreadable previous key is as much a reason to
  // stop as an unreadable current one (the operator put it there to co-sign).
  let current = await readKeyFile(fs, path, keyFile);
  const previous = await readKeyFile(fs, path, previousFile);

  if (!current) {
    const generated = await generateKeyPair('EdDSA', {
      crv: 'Ed25519',
      extractable: true, // required to serialise the private half
    });
    current = { privateKey: generated.privateKey as KeyLike, publicKey: generated.publicKey as KeyLike };
    try {
      const outcome = await writeKeyFileOnce(fs, path, keyFile, await exportJWK(current.privateKey));
      if (outcome === 'exists') {
        // Another process created the key between the read above and the
        // link: hold ITS key, so both processes are the same identity.
        const theirs = await readKeyFile(fs, path, keyFile);
        if (!theirs) throw new AgentKeyFileError(keyFile, 'appeared and vanished while this process was creating it');
        current = theirs;
      } else {
        console.log(`[webagents] created agent key ${keyFile}`);
      }
    } catch (err) {
      if (err instanceof AgentKeyFileError) throw err;
      console.warn(
        `[webagents] could not persist the agent key to ${keyFile} ` +
          `(${errorMessage(err)}); this boot's key is EPHEMERAL and platform ` +
          'registration will break on restart.',
      );
    }
  }

  const identity = makeIdentity(current, previous);
  await identity.initialize();
  if (previous && identity.getHeldKeys().length > 1) {
    console.log(
      `[webagents] holding the previous agent key ${identity.getHeldKeys()[1].kid} from ${previousFile} ` +
        'beside the current one; every request is co-signed with it. Delete the file once the platform has admitted the new key.',
    );
  }
  return identity;
}
