/**
 * The LLM provider registry.
 *
 * One table describing every provider this SDK can talk to: the skill names
 * that select it, the environment variable that carries its credential, and
 * whether it needs one at all.
 *
 * WHY THIS EXISTS (2026-09-23). `webagents models` used to print a hardcoded
 * array of MODEL IDS: `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`,
 * `grok-2`. Every one of them was superseded while the list sat there, because
 * model ids turn over every few months and nothing in the build could notice.
 * Provider names do not turn over, so the CLI lists PROVIDERS and shows the
 * model-string format instead of guessing at ids it cannot keep current.
 *
 * The same table is what an API-key preflight needs, which is the other reason
 * it is a module rather than a literal inside one command: "which providers are
 * configured on this machine" and "what do I tell the user to export" are the
 * same question asked twice.
 *
 * KEEPING IT HONEST: `envVar` values below were read out of each skill, not
 * assumed. If a skill changes how it resolves its key, this table is wrong and
 * the preflight will lie, so change both together.
 *
 * `defaultModel` values are taken from the Python SDK's per-provider
 * `uamp_adapter.py` catalogs, which are the better-maintained model lists in
 * this repo. Keep the two in step: a default here that its catalog does not
 * know falls through to generic capabilities and loses the thinking, caching
 * and tool flags the real model has.
 *
 * THE GOOGLE KEY (2026-09-25). This SDK's Google skill read `GOOGLE_API_KEY`
 * and the Python one `GOOGLE_GEMINI_API_KEY` or `GEMINI_API_KEY`, so a
 * developer who exported the variable one SDK documented and ran the other got
 * an authentication failure. Both now read `GOOGLE_API_KEY`, then
 * `GOOGLE_GEMINI_API_KEY`, then `GEMINI_API_KEY`: the portal's own order
 * (`lib/llm/provider-keys.ts`). `envVars` lists every variable a skill reads;
 * `envVar`, its first, is the one advice names.
 */

/** How a provider is credentialed. */
export type ProviderCredential =
  /** Needs an API key in `envVar`. */
  | 'api-key'
  /** Runs locally in the browser or on-device; no credential. */
  | 'local'
  /** Billed through the platform; needs a portal session, not a provider key. */
  | 'platform';

export interface LLMProvider {
  /** Canonical id, also the primary skill name. */
  id: string;
  /** Every skill name that selects this provider, including the canonical id. */
  aliases: string[];
  /** One line, for `webagents models`. */
  description: string;
  /** How it is credentialed. */
  credential: ProviderCredential;
  /** The environment variable carrying the credential, when `credential` is 'api-key' or 'platform'. */
  envVar?: string;
  /** Every variable the skill reads, in its order, when there is more than `envVar`. */
  envVars?: readonly string[];
  /** The shape of a model string for this provider, e.g. `openai/<model>`. */
  modelFormat: string;
  /**
   * A model id known to exist, used only to make `init` produce something that
   * runs. NOT a recommendation and NOT kept current automatically.
   *
   * This is the one place a concrete model id appears in the SDK, on purpose:
   * ids rot, so confining them to a single field with this comment means the
   * fix is one line rather than a hunt. Before a release, check each against
   * the provider's own model list and update. Absent for `local` providers,
   * where the model is chosen at load time by the runtime.
   */
  defaultModel?: string;
}

export const LLM_PROVIDERS: readonly LLMProvider[] = [
  {
    id: 'openai',
    aliases: ['openai'],
    description: 'OpenAI hosted models',
    credential: 'api-key',
    envVar: 'OPENAI_API_KEY',
    modelFormat: 'openai/<model>',
    defaultModel: 'gpt-4o-mini',
  },
  {
    id: 'anthropic',
    aliases: ['anthropic', 'claude'],
    description: 'Anthropic hosted models',
    credential: 'api-key',
    envVar: 'ANTHROPIC_API_KEY',
    modelFormat: 'anthropic/<model>',
    defaultModel: 'claude-haiku-4-5-20251001',
  },
  {
    id: 'google',
    // `llm`: the Python SDK's old name for its Google skill, accepted here too.
    aliases: ['google', 'gemini', 'llm'],
    description: 'Google hosted models',
    credential: 'api-key',
    envVar: 'GOOGLE_API_KEY',
    envVars: ['GOOGLE_API_KEY', 'GOOGLE_GEMINI_API_KEY', 'GEMINI_API_KEY'],
    modelFormat: 'google/<model>',
    defaultModel: 'gemini-2.5-flash',
  },
  {
    id: 'xai',
    aliases: ['xai', 'grok'],
    description: 'xAI hosted models',
    credential: 'api-key',
    envVar: 'XAI_API_KEY',
    modelFormat: 'xai/<model>',
    defaultModel: 'grok-3',
  },
  {
    id: 'fireworks',
    aliases: ['fireworks'],
    description: 'Fireworks hosted open models',
    credential: 'api-key',
    envVar: 'FIREWORKS_API_KEY',
    modelFormat: 'fireworks/<model>',
    defaultModel: 'deepseek-v3p2',
  },
  {
    id: 'proxy',
    aliases: ['proxy'],
    description: 'Platform-hosted inference, billed to your Robutler account',
    credential: 'platform',
    envVar: 'ROBUTLER_LLM_PROXY_URL',
    modelFormat: 'proxy/<model>',
    defaultModel: 'gpt-4o-mini',
  },
  {
    id: 'webllm',
    aliases: ['webllm'],
    description: 'In-browser inference via WebGPU',
    credential: 'local',
    modelFormat: 'webllm/<model>',
  },
  {
    id: 'transformers',
    aliases: ['transformers'],
    description: 'In-browser inference via Transformers.js',
    credential: 'local',
    modelFormat: 'transformers/<model>',
  },
] as const;

/** Every variable a provider's skill reads its credential from, in order. */
export function providerEnvVars(provider: LLMProvider): readonly string[] {
  return provider.envVars ?? (provider.envVar ? [provider.envVar] : []);
}

/** The provider a skill name selects, or `undefined` if none does. */
export function findProvider(skillName: string): LLMProvider | undefined {
  const lower = skillName.toLowerCase();
  return LLM_PROVIDERS.find((p) => p.aliases.includes(lower));
}

/**
 * The model to give the LLM skill for `providerId`: the agent's own when it
 * names that provider or none, otherwise `undefined` (the skill's default).
 * A model naming ANOTHER provider is not forced onto this one: `openai` with
 * `anthropic/...` would only send an Anthropic id to OpenAI.
 */
export function modelForProvider(providerId: string, model: string | undefined): string | undefined {
  if (!model) return undefined;
  const slash = model.indexOf('/');
  if (slash === -1) return model;
  return findProvider(model.slice(0, slash))?.id === providerId ? model : undefined;
}

/**
 * The key an agent needs and this environment does not have (2026-09-24).
 *
 * The provider is the one the agent's first declared LLM skill selects; with
 * none declared it is OpenAI. (The chat no longer assumes OpenAI for an agent
 * that names no LLM skill; `cli/model-access.ts` decides that case, and
 * callers pass a declared provider.) `undefined` when the key is present, or
 * the provider needs no key.
 *
 * Without this check a missing key surfaced as "OpenAI API key not
 * configured" and a Node stack trace from inside the first model call, after
 * the agent had already started. The Python CLI's `missing_key_for_model`
 * answers the same question.
 */
export function missingProviderKey(
  declaredSkills: readonly string[],
  env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {},
): { provider: LLMProvider; envVar: string } | undefined {
  const declared = declaredSkills
    .map((name) => findProvider(String(name)))
    .find((p): p is LLMProvider => p !== undefined);
  const provider = declared ?? findProvider('openai');
  if (!provider || provider.credential !== 'api-key' || !provider.envVar) return undefined;
  return providerEnvVars(provider).some((name) => env[name]) ? undefined : { provider, envVar: provider.envVar };
}

/**
 * Which providers have their credential present in this environment.
 *
 * `local` providers are always considered available because they need nothing.
 * This reads `process.env` directly and is therefore Node-only; callers in
 * browser builds should not use it.
 */
export function configuredProviders(
  env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {},
): { available: LLMProvider[]; missing: LLMProvider[] } {
  const available: LLMProvider[] = [];
  const missing: LLMProvider[] = [];
  for (const p of LLM_PROVIDERS) {
    if (p.credential === 'local' || providerEnvVars(p).some((name) => env[name])) available.push(p);
    else missing.push(p);
  }
  return { available, missing };
}
