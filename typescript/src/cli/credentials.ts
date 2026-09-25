/**
 * The CLI's platform credential, in the OS keystore.
 *
 * WHY (2026-09-23, logged as S-211). `saveAuth` was `fs.writeFileSync` with no
 * `mode`, into a directory created 0755, so the token landed at 0644 and any
 * other local user could read it. The Python CLI had the identical bug in a
 * differently-named file.
 *
 * This does NOT implement a credential store. This SDK already ships a good one
 * for `SecretsSkill` (`src/skills/secrets/store.ts`): keystore first via
 * `@napi-rs/keyring`, an owner-only 0600 file in a 0700 directory as the
 * documented fallback, and a re-`chmod` after writing because `mode` on
 * `writeFile` only applies when the file is created. Writing a third store
 * would mean getting those details right a third time.
 *
 * A KEYSTORE CAN BLOCK. On macOS the first access by a given binary may raise
 * an authorization dialog, and a process with nobody to click it waits.
 * Observed 2026-09-23: this CLI hung on exactly that in a headless shell while
 * the Python one did not, because the two use different keyring bindings and
 * macOS authorises per binary. It is inherent to OS keystores, and one more
 * reason the file fallback and `WEBAGENTS_TOKEN` exist. Anything that must not
 * block should pass `--token` or set the env var, both of which are checked
 * BEFORE the keystore is opened.
 *
 * PRECEDENCE for reading a token, matching Python:
 *
 *   1. an explicit `--token` flag
 *   2. `WEBAGENTS_TOKEN` in the environment
 *   3. the keystore (or its 0600 file fallback)
 */

import * as path from 'node:path';
import { globalDir, profileName, scopedNamespace } from './config-store';

/** Keystore namespace. Distinct from any agent's own secrets. */
export const CLI_NAMESPACE = 'cli';

/** The single entry inside it. */
export const TOKEN_KEY = 'platform_token';

/** Read before the keystore. Named for the CLI, because it is the CLI's credential. */
export const TOKEN_ENV_VAR = 'WEBAGENTS_TOKEN';

async function store(profile?: string) {
  const { openSecretStore } = await import('../skills/secrets/store.js');
  // The FACTORY, not the constructor: it is what probes for a keystore and
  // decides the backend.
  //
  // Under a profile BOTH the namespace and the fallback directory move. Only
  // the directory used to, which isolated two profiles' tokens on a machine
  // with no keystore and shared one keychain entry on a machine with one
  // (S-219, fixed on this side 2026-09-23, after the Python side).
  const resolved = profileName(profile);
  return openSecretStore({
    namespace: scopedNamespace(CLI_NAMESPACE, resolved),
    secretsDir: path.join(globalDir(resolved), 'secrets'),
    quiet: true,
  });
}

/**
 * `--token`, for this run only. Kept here rather than in `process.env`: the
 * environment is handed to every command the `shell` skill runs, and a
 * platform token has no business there (the Python CLI's `set_flag_token`).
 */
let flagToken: string | undefined;

export function setFlagToken(token: string | undefined): void {
  flagToken = token || undefined;
}

/** The platform token, by the precedence in the module docstring. */
export async function getToken(explicit?: string, profile?: string): Promise<string | null> {
  if (explicit) return explicit;
  if (flagToken) return flagToken;
  const fromEnv = process.env[TOKEN_ENV_VAR];
  if (fromEnv) return fromEnv;
  try {
    return await (await store(profile)).get(TOKEN_KEY);
  } catch {
    // No keystore and no fallback: the caller treats null as "not logged in".
    return null;
  }
}

/** Store the token. Returns the backend it landed in. */
export async function setToken(token: string, profile?: string): Promise<'keystore' | 'file'> {
  const s = await store(profile);
  // `set` returns the backend the value actually landed in, which is the
  // honest answer rather than what was probed for at open.
  return s.set(TOKEN_KEY, token);
}

/** Remove the stored token. */
export async function clearToken(profile?: string): Promise<boolean> {
  try {
    return await (await store(profile)).delete(TOKEN_KEY);
  } catch {
    return false;
  }
}

/**
 * Which backend is in use and why. What `doctor` prints.
 *
 * "I thought my token was in the Keychain" is the exact failure the underlying
 * store was written to prevent, and it can only be prevented by SAYING which
 * backend answered.
 */
export async function backendStatus(profile?: string): Promise<Record<string, unknown>> {
  let status: Record<string, unknown>;
  try {
    const s = await store(profile);
    status = { ...s.status() };
  } catch (e) {
    status = { backend: 'unavailable', reason: (e as Error).message };
  }
  status.envVarSet = Boolean(process.env[TOKEN_ENV_VAR]);
  if (status.envVarSet) {
    // Worth saying plainly: an env var silently outranks whatever is stored, so
    // a stale export explains "I logged in and it still uses the old account".
    status.note = `${TOKEN_ENV_VAR} is set and takes precedence over the stored token.`;
  }
  return status;
}
