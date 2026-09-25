/**
 * Provider keys the CLI keeps for the person (2026-09-24).
 *
 * The chat offers to take a key when it has no way to run a model
 * (`app.ts`). A key taken there has to be there next time too, or the offer
 * comes back on every start. It goes in the SAME store as the Python CLI's
 * `webagents secrets set` (namespace `providers`, profile-scoped, the OS
 * keystore or an owner-only file; the two stores agree on the keychain
 * service and the file name), so a key entered in either CLI works in both.
 *
 * Read only where the environment has no value: an exported key always
 * wins, so a stored one can never change a session that already had one.
 * Only the variables the chat's providers read are looked up, by name.
 *
 * NEVER PUT INTO `process.env`. The chat hands a stored key to the model
 * client alone (`apiKeys` in `skills/resolve.ts`). The `shell` skill runs
 * every command with the chat's whole environment (S-220 in the portal's
 * security log: the unsandboxed path inherits it), so a key exported into
 * it would be one `echo $OPENAI_API_KEY`, or one prompt injection, away from
 * the transcript. A key the person exported themselves is already there;
 * one kept in the keychain does not have to be.
 */

import * as path from 'node:path';
import { globalDir, profileName, scopedNamespace } from './config-store';
import { providerEnvVars, type LLMProvider } from '../skills/llm/providers';
import { keyProviders } from './model-access';

/** The Python CLI's `secrets.NAMESPACE`. */
export const PROVIDERS_NAMESPACE = 'providers';

/** The profile's provider-key store. */
export async function providerKeyStore() {
  const { openSecretStore } = await import('../skills/secrets/store.js');
  const resolved = profileName();
  return openSecretStore({
    namespace: scopedNamespace(PROVIDERS_NAMESPACE, resolved),
    secretsDir: path.join(globalDir(resolved), 'secrets'),
    quiet: true,
  });
}

/**
 * The stored provider keys `env` does not already have, by variable name.
 * A store that cannot be opened or read counts as empty: the chat then says
 * there is no key, which is true.
 */
export async function readStoredProviderKeys(
  env: Record<string, string | undefined> = process.env,
): Promise<Record<string, string>> {
  const wanted = keyProviders()
    .flatMap((provider) => [...providerEnvVars(provider)])
    .filter((name) => !env[name]);
  const found: Record<string, string> = {};
  if (!wanted.length) return found;
  let store: Awaited<ReturnType<typeof providerKeyStore>>;
  try {
    store = await providerKeyStore();
  } catch {
    return found;
  }
  for (const name of wanted) {
    try {
      const value = await store.get(name);
      if (value) found[name] = value;
    } catch {
      // Unreadable is treated as absent.
    }
  }
  return found;
}

/** Store a provider key. Returns the backend it landed in. */
export async function storeProviderKey(name: string, value: string): Promise<'keystore' | 'file'> {
  return (await providerKeyStore()).set(name, value);
}

/**
 * The stored key for `provider` when the environment sets none of the
 * variables its skill reads: the first of those names that is stored.
 */
export function storedKeyFor(
  provider: LLMProvider,
  stored: Record<string, string>,
  env: Record<string, string | undefined> = process.env,
): string | undefined {
  const names = providerEnvVars(provider);
  if (names.some((name) => env[name])) return undefined;
  for (const name of names) if (stored[name]) return stored[name];
  return undefined;
}
