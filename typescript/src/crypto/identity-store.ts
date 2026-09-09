/**
 * Persistent agent identity.
 *
 * WHY THIS EXISTS AT ALL: platform registration stores the `metadata.publicKey`
 * it read off the agent card in `agent_registrations.publicKey`, and verifies
 * every subsequent AOAuth token against THAT stored copy. A key generated per
 * boot therefore works exactly until the first restart and then fails every
 * verification, with a card that still looks perfectly correct. The Python SDK
 * has always persisted (`crypto/jwks.py` `ensure_keys`); this is the TS
 * counterpart, and it belongs to the server rather than to any one entry
 * point — it moved here out of the deleted `host()` wrapper.
 *
 * Stored as a private JWK (which carries both `d` and `x`, so the public half
 * is derived on load) with 0600 permissions, under `WEBAGENTS_KEYS_DIR` or
 * `~/.webagents/keys` — the same location the Python SDK uses.
 */

import { AgentIdentity } from './identity';
import { generateKeyPair, exportJWK, importJWK, type JWK, type KeyLike } from 'jose';

export interface IdentityStoreOptions {
  /** Issuer recorded on the identity (the agent's public URL). */
  issuer: string;
  /**
   * Where the key lives. Defaults to `WEBAGENTS_KEYS_DIR`, then
   * `~/.webagents/keys`. Pass `null` for an explicitly ephemeral key (tests).
   */
  keysDir?: string | null;
  /**
   * Hosting prefix that goes on minted tokens as `agent_path`.
   *
   * It is what lets a host serve more than one agent. The platform keys a
   * registration on `iss + agent_path + '/' + sub`, and falls back to the bare
   * `iss` when the claim is absent — and `agent_registrations.agent_url` is
   * unique, so a host whose agents all omit it caps at exactly one registered
   * agent, with the second one silently answering as the first.
   */
  agentPath?: string;
}

/** Filesystem-safe file stem for an agent name. */
function keyFileStem(agentName: string): string {
  return agentName.replace(/[^A-Za-z0-9._-]/g, '_');
}

/**
 * Load this agent's Ed25519 key from disk, or create and persist one.
 */
export async function loadOrCreateAgentIdentity(
  agentName: string,
  options: IdentityStoreOptions,
): Promise<AgentIdentity> {
  const { issuer, keysDir: keysDirOption, agentPath } = options;
  const makeIdentity = (privateKey?: KeyLike, publicKey?: KeyLike) =>
    new AgentIdentity({ agentId: agentName, issuer, privateKey, publicKey, agentPath });

  if (keysDirOption === null) {
    const identity = makeIdentity();
    await identity.initialize();
    return identity;
  }

  let fs: typeof import('node:fs/promises');
  let path: typeof import('node:path');
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
  const keyFile = path.join(keysDir, `${keyFileStem(agentName)}.ed25519.jwk.json`);

  try {
    const raw = await fs.readFile(keyFile, 'utf8');
    const jwk = JSON.parse(raw) as JWK;
    const privateKey = (await importJWK(jwk, 'EdDSA')) as KeyLike;
    const { d: _d, ...publicJwk } = jwk;
    const publicKey = (await importJWK(publicJwk, 'EdDSA')) as KeyLike;
    const identity = makeIdentity(privateKey, publicKey);
    await identity.initialize();
    return identity;
  } catch (err) {
    const code = (err as { code?: string }).code;
    if (code && code !== 'ENOENT') {
      console.warn(`[webagents] could not read ${keyFile}: ${(err as Error).message}`);
    }
  }

  const { privateKey, publicKey } = await generateKeyPair('EdDSA', {
    crv: 'Ed25519',
    extractable: true, // required to serialise the private half
  });
  try {
    await fs.mkdir(keysDir, { recursive: true, mode: 0o700 });
    await fs.writeFile(keyFile, JSON.stringify(await exportJWK(privateKey)), { mode: 0o600 });
    console.log(`[webagents] created agent key ${keyFile}`);
  } catch (err) {
    console.warn(
      `[webagents] could not persist the agent key to ${keyFile} ` +
        `(${(err as Error).message}); this boot's key is EPHEMERAL and platform ` +
        'registration will break on restart.',
    );
  }
  const identity = makeIdentity(privateKey as KeyLike, publicKey as KeyLike);
  await identity.initialize();
  return identity;
}
