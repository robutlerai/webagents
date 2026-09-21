/**
 * The agent card: the registration-time metadata document the platform reads
 * once per principal at `{agentUrl}/.well-known/agent.json` (ADR 0038 step 5,
 * W2 design sections 3.3 and 9.1, 2026-09-17). Shared by `createFetchHandler`
 * and `WebAgentsServer`, which until this file existed served two different
 * cards (and the multi-agent server none at all, so a multi-agent TypeScript
 * host could not register whatever else was right).
 *
 * THE CARD IS NO LONGER WHERE THE KEY COMES FROM. Before this day the card
 * carried the SPKI PEM at `publicKey` and `metadata.publicKey` and the
 * platform imported it to verify a bearer. The key now comes from the key set
 * the signed request names (`Signature-Agent`), and the card is metadata:
 * name, description, capabilities, skills, plus the three fields that make it
 * SELF-NAMING (the CIMD rule, C section 3; R section 3.1.1), each checked by
 * the platform with a plain string comparison:
 *
 *   * `client_id` equals the URL the card was fetched from;
 *   * `url` equals the principal, the agent URL;
 *   * `jwks_uri` equals the key-set URL the signature was resolved through.
 *
 * A card failing any of these is `card_not_self_naming` or
 * `card_key_set_mismatch`, so every one of them is DERIVED from one value,
 * the principal, and can only disagree with the signer if the identity's
 * issuer and the served path disagree, which `serve()` composes from the same
 * `publicUrl + basePath`. This closes S-035: a card served at one URL can no
 * longer vouch for an agent at another, because it names the URL it lives at.
 *
 * `authentication.schemes` is `HTTPSig` when the agent signs (it has an
 * identity and publishes a key set) and `Bearer` otherwise: a server without
 * an identity still takes api keys and platform service tokens on
 * `/chat/completions`, and a card claiming a scheme the agent cannot perform
 * is worse than one claiming less.
 */

import type { IAgent } from '../core/types';
import { canonicalAgentUrl } from '../crypto/identity';

export const CARD_WELL_KNOWN_SUFFIX = '/.well-known/agent.json';
export const KEY_SET_WELL_KNOWN_SUFFIX = '/.well-known/jwks.json';

export interface AgentCardOptions {
  /**
   * The agent URL, the principal. Absolute for an agent that registers
   * (`https://host/agents/mini`); a relative reference (`/agents/mini`) when
   * no public URL is configured, which is honest about what is known and is
   * resolved by any consumer against the document it just fetched.
   */
  principal: string;
  /** True when the agent has a signing identity, i.e. publishes a key set at `${principal}/.well-known/jwks.json`. */
  signs: boolean;
}

export interface AgentCard {
  name: string;
  description?: string;
  /** The card's own URL (C section 3). */
  client_id: string;
  /** The principal. */
  url: string;
  /** The key set, present when the agent signs. */
  jwks_uri?: string;
  capabilities: { streaming: boolean; pushNotifications: boolean };
  authentication: { schemes: string[] };
  skills: Array<{ id: string; name: string; description?: string }>;
}

/**
 * The card for `agent` served at `principal`. Pure: no environment, no request.
 * The principal is read through `canonicalAgentUrl`, the spelling the
 * platform compares against and the identity signs with (2026-09-18).
 */
export function buildAgentCard(agent: IAgent, options: AgentCardOptions): AgentCard {
  const base = canonicalAgentUrl(options.principal);
  return {
    name: agent.name,
    description: agent.description,
    client_id: `${base}${CARD_WELL_KNOWN_SUFFIX}`,
    url: base || '/',
    ...(options.signs ? { jwks_uri: `${base}${KEY_SET_WELL_KNOWN_SUFFIX}` } : {}),
    capabilities: { streaming: true, pushNotifications: false },
    authentication: { schemes: [options.signs ? 'HTTPSig' : 'Bearer'] },
    skills: (agent.getToolDefinitions?.() ?? [])
      .filter((t) => t.type === 'function' && 'function' in t)
      .map((t) => {
        const ft = t as { type: 'function'; function: { name: string; description?: string } };
        return { id: ft.function.name, name: ft.function.name, description: ft.function.description };
      }),
  };
}
