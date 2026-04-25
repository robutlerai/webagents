/**
 * Default token writer for standalone hosts. Logs a warning and no-ops —
 * persisting credentials to env at runtime is not safe, so OAuth callback
 * `@http` endpoints in non-portal hosts are expected to print the token
 * to the operator and instruct them to set the env var manually.
 *
 * Hosts that want true persistence supply their own `setToken`. The portal
 * factory wires `setToken` to `upsertConnectedAccountFromToken`.
 */
import type { TokenWriter } from './options';

export function noopTokenWriter(): TokenWriter {
  return {
    async setToken(input) {
      console.warn(
        `[messaging-skill] no token writer configured. Provider=${input.provider}; ` +
          `token would be persisted but the host did not provide setToken. ` +
          `Set the relevant env var (see PROVIDER_ENV) and restart.`,
      );
      return { accountId: 'env' };
    },
  };
}
