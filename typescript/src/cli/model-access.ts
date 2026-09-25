/**
 * Which model the chat's agent runs on, and how it gets there (2026-09-24).
 *
 * The chat attached `OpenAISkill` to every agent that named no LLM skill, so
 * with no `OPENAI_API_KEY` the first reply was "OpenAI API key not
 * configured", even for a person signed in to Robutler, which serves models,
 * and even with another provider's key set. The same decision as the Python
 * CLI (`python/webagents/cli/model_access.py`) now:
 *
 *  1. The agent names a model (`model:` or `-m`): its provider's key when it is
 *     here; otherwise, when signed in, the SAME model through Robutler's LLM
 *     proxy, paid from the person's credits. A provider this SDK has no client
 *     for (one the registry does not know) goes the same way.
 *  2. It names none: the first provider this SDK can load that has a key, at
 *     its default model; otherwise, when signed in, Robutler's default
 *     (`auto/balanced`, which the platform maps to its current model).
 *  3. Neither: `none`, with the reason, and the chat offers to sign in or to
 *     take a key there and then (`cli/app.ts`).
 *
 * `auto/…`, `proxy/…` and `robutler/…` name models only Robutler serves.
 */

import { LLM_PROVIDERS, findProvider, providerEnvVars, type LLMProvider } from '../skills/llm/providers';
import { cliCommand } from './config-store';

/** The platform's own choice of model, resolved server-side. */
export const PROXY_DEFAULT_MODEL = 'auto/balanced';

/** Providers the chat can load by name (`skills/resolve.ts` LLM_LOADERS). */
const LOADABLE = new Set(['openai', 'anthropic', 'google', 'xai', 'fireworks']);

export interface ModelAccess {
  /** `direct` (the provider, with this machine's key), `proxy` (Robutler) or `none`. */
  kind: 'direct' | 'proxy' | 'none';
  /** The model, e.g. `openai/gpt-4o` or `auto/balanced`; undefined when `none`. */
  model?: string;
  provider?: LLMProvider;
  /** Why it is not `direct`, when it is not. */
  reason?: string;
}

type Env = Record<string, string | undefined>;

function hasKey(provider: LLMProvider, env: Env): boolean {
  return providerEnvVars(provider).some((name) => Boolean(env[name]));
}

/** `openai/gpt-4o` for a bare `gpt-4o`: the chat's historical default provider. */
function withProvider(model: string): { provider?: LLMProvider; model: string } {
  const slash = model.indexOf('/');
  if (slash === -1) return { provider: findProvider('openai'), model: `openai/${model}` };
  return { provider: findProvider(model.slice(0, slash)), model };
}

/** The decision above. `signedIn` is asked only when a key does not already answer. */
export async function resolveModelAccess(
  declaredModel: string | undefined,
  options: { signedIn: () => Promise<boolean> | boolean; env?: Env },
): Promise<ModelAccess> {
  const env = options.env ?? (typeof process !== 'undefined' ? process.env : {});
  if (declaredModel) {
    const [prefix, ...rest] = declaredModel.split('/');
    if (prefix === 'auto' || prefix === 'proxy' || prefix === 'robutler') {
      const model = prefix === 'auto' ? declaredModel : rest.join('/') || PROXY_DEFAULT_MODEL;
      if (await options.signedIn()) return { kind: 'proxy', model };
      return { kind: 'none', reason: `${declaredModel} is served by Robutler` };
    }
    const { provider, model } = withProvider(declaredModel);
    const loadable = Boolean(provider && LOADABLE.has(provider.id));
    if (loadable && hasKey(provider!, env)) return { kind: 'direct', model, provider };
    // A provider this SDK has no client for (a name the registry does not
    // know) is not "direct": there is nothing here to call it with.
    // Robutler may still serve it.
    const reason = loadable
      ? `${provider!.envVar} is not set`
      : `this SDK has no built-in client for ${provider?.id ?? model.split('/')[0]}`;
    if (await options.signedIn()) return { kind: 'proxy', model, provider, reason };
    return { kind: 'none', provider, reason };
  }
  for (const provider of LLM_PROVIDERS) {
    if (LOADABLE.has(provider.id) && provider.defaultModel && hasKey(provider, env)) {
      return { kind: 'direct', model: `${provider.id}/${provider.defaultModel}`, provider };
    }
  }
  if (await options.signedIn()) return { kind: 'proxy', model: PROXY_DEFAULT_MODEL, reason: 'no provider key is set' };
  return { kind: 'none', reason: 'no provider key is set' };
}

/** For the welcome card and the footer: the model, and how it is reached when not obvious. */
export function describeAccess(access: ModelAccess, fallback = ''): string {
  if (access.kind === 'proxy') return `${access.model} via Robutler`;
  return access.model ?? fallback;
}

/** One sentence naming both ways out, for `none`. */
export function unavailableMessage(access: ModelAccess): string {
  const signIn = `Sign in with \`${cliCommand('login')}\` to use Robutler's models`;
  // `auto/...`: only Robutler serves it, so a key would not help.
  if (access.reason?.endsWith('is served by Robutler')) return `No model for this agent: ${access.reason}. ${signIn}.`;
  const provider = access.provider;
  const key =
    provider && LOADABLE.has(provider.id) && provider.envVar
      ? `add a key with \`${cliCommand(`secrets set ${provider.envVar}`)}\``
      : provider || access.reason?.startsWith('this SDK')
        ? `name a model this SDK can call with your own key (${[...LOADABLE].map((id) => `${id}/...`).join(', ')})`
        : `add a key with \`${cliCommand('secrets set <NAME>')}\` (${keyProviders().map((p) => p.envVar).join(', ')})`;
  return `No model for this agent: ${access.reason}. ${signIn}, or ${key}.`;
}

/** The `/llm` socket of the platform this profile signs in to. */
export function platformLlmUrl(platformUrl: string): string {
  const explicit = typeof process !== 'undefined' ? process.env?.ROBUTLER_LLM_PROXY_URL : undefined;
  if (explicit) return explicit;
  const base = platformUrl.replace(/\/+$/, '');
  if (base.startsWith('https://')) return `wss://${base.slice('https://'.length)}/llm`;
  if (base.startsWith('http://')) return `ws://${base.slice('http://'.length)}/llm`;
  return `${base}/llm`;
}

/** Every provider the chat can take a key for, in the order it asks. */
export function keyProviders(): LLMProvider[] {
  return LLM_PROVIDERS.filter((p) => LOADABLE.has(p.id) && p.envVar);
}
