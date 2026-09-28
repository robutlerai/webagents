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
 *
 * OTHER CALLERS' TURNS NEVER RUN ON THE SIGN-IN (S-327, 2026-09-28). The
 * decision above is the owner's own chat and `-p`. `serve`, `mcp serve`, the
 * daemon and `cron run` answer other callers, and `attachModelForCallers`
 * decides for them: the agent's provider key, or Robutler's models when the
 * AGENT has its own platform credential (`server/agent-credential.ts`), built
 * with no sign-in and paid by each caller's payment token (`callersPay` on
 * the proxy skill), or nothing, with a sentence naming both ways out, which
 * `serve` refuses to start with. They had no decision at all: an agent file
 * naming no LLM skill served with no model, even with the key exported (B6).
 * The Python twin is `model_access.choose_model_access(for_callers=True)`;
 * the words are the shared fixture `cli/final_sdk_serve_model.json`.
 */

import type { ISkill } from '../core/types';
import { LLM_PROVIDERS, findProvider, modelForProvider, providerBaseUrl, providerEnvVars, type LLMProvider } from '../skills/llm/providers';
import { cliCommand } from './config-store';

/** The platform's own choice of model, resolved server-side. */
export const PROXY_DEFAULT_MODEL = 'auto/balanced';

/** Providers the chat can load by name (`skills/resolve.ts` LLM_LOADERS). */
const LOADABLE = new Set(['openai', 'anthropic', 'google', 'xai', 'fireworks', 'ollama']);

export interface ModelAccess {
  /** `direct` (the provider, with this machine's key), `proxy` (Robutler) or `none`. */
  kind: 'direct' | 'proxy' | 'none';
  /** The model, e.g. `openai/gpt-4o` or `auto/balanced`; undefined when `none`. */
  model?: string;
  provider?: LLMProvider;
  /** Why it is not `direct`, when it is not. */
  reason?: string;
  /**
   * The turns are other callers' (`serve`, the daemon; S-327): Robutler's
   * models run on the agent's own platform credential, never the sign-in,
   * and `unavailableMessage` names those ways out.
   */
  forCallers?: boolean;
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
    // A provider with no credential (Ollama, plan item 2.8) is reached
    // directly whenever it is named; whether it answers is doctor's question.
    if (loadable && (provider!.credential === 'none' || hasKey(provider!, env))) return { kind: 'direct', model, provider };
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
    // Only a provider with a key here: a local server (Ollama) is never
    // assumed to be running, so it is used only when the file names it.
    if (LOADABLE.has(provider.id) && provider.credential === 'api-key' && provider.defaultModel && hasKey(provider, env)) {
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

/**
 * For /status and doctor: where a local model is reached (`ollama/llama3.2,
 * at http://localhost:11434/v1`, plan item 2.8), or `undefined` for every
 * other route. The Python chat says the same (fixture `status_route`).
 */
export function describeLocalRoute(
  access: ModelAccess | undefined,
  env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {},
): string | undefined {
  if (!access || access.kind !== 'direct' || !access.provider || access.provider.credential !== 'none') return undefined;
  const base = providerBaseUrl(access.provider, env);
  return base ? `${access.model}, at ${base}` : access.model;
}

/** One sentence naming both ways out, for `none`. */
export function unavailableMessage(access: ModelAccess): string {
  if (access.forCallers) return callersUnavailableMessage(access);
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

/**
 * The sentence for `serve` and the daemon when no model can run for other
 * callers (S-327): the reason, that the sign-in is never used for them, and
 * both ways out. The Python `callers_unavailable_message` says the same
 * (fixture `cli/final_sdk_serve_model.json`, `refusals`).
 */
export function callersUnavailableMessage(access: ModelAccess): string {
  const credential =
    `give the agent its own platform credential (\`${cliCommand('publish')}\`, or WEBAGENTS_AGENT_TOKEN) ` +
    "so each caller's payment token pays for Robutler's models";
  const provider = access.provider;
  let ways: string;
  if (access.reason?.endsWith('is served by Robutler')) {
    ways = credential[0].toUpperCase() + credential.slice(1);
  } else if (provider && LOADABLE.has(provider.id) && provider.envVar) {
    ways = `Add a key with \`${cliCommand(`secrets set ${provider.envVar}`)}\`, or ${credential}`;
  } else if (provider || access.reason?.startsWith('this SDK')) {
    ways = `Name a model this SDK can call with your own key (${[...LOADABLE].map((id) => `${id}/...`).join(', ')}), or ${credential}`;
  } else {
    ways = `Add a key with \`${cliCommand('secrets set <NAME>')}\` (${keyProviders().map((p) => p.envVar).join(', ')}), or ${credential}`;
  }
  return `No model for this agent's callers: ${access.reason}. A served agent never runs on your sign-in. ${ways}.`;
}

/** What {@link attachModelForCallers} needs: the agent as `resolveSkillsByName` built it. */
export interface CallersModelInput {
  agentName: string;
  /** The skill names as the file lists them. */
  declaredSkills: readonly string[];
  /** The resolved skills; the model's skill is added, or a keyless one removed, in place. */
  skills: ISkill[];
  /** Each resolved skill by the name the file gave it. */
  byName: ReadonlyMap<string, ISkill>;
  /** Names that matched nothing or failed to build. */
  notBuilt: ReadonlySet<string>;
  /** The agent's `model:`. */
  model?: string;
  /** Stored provider keys, handed to the model client only. */
  apiKeys: Partial<Record<string, string>>;
  /** What the decision reads: the environment plus the stored keys. */
  env: Env;
  /** Robutler's `/llm` socket, for the proxy skill; its own default otherwise. */
  proxyUrl?: string;
  /** Whether the agent has its own platform credential; `resolveAgentCredential` by default. */
  ownCredential?: () => Promise<boolean>;
  /**
   * The OWNER's own turns instead (ACP, and `mcp serve` over stdio: the
   * editor or MCP client the owner started on this machine): the chat's
   * decision, the sign-in included, as the Python `for_callers=False` does.
   * Unset for everything other callers reach (S-327).
   */
  owner?: {
    signedIn: () => Promise<boolean>;
    platformToken: () => Promise<string | null | undefined>;
  };
}

export interface CallersModel {
  access: ModelAccess;
  /** The model the agent runs, for `BaseAgent`'s `model` (and the startup line). */
  model?: string;
  /** Why no model can run for callers, naming both ways out; unset when one can. */
  problem?: string;
}

/**
 * The model for OTHER CALLERS' turns (`serve`, `mcp serve`, the daemon, `cron
 * run`; S-327): the chat's decision (`cli/app.ts`) with the sign-in replaced
 * by the agent's own platform credential, and a proxy skill that carries no
 * sign-in (`callersPay`). A listed LLM skill whose key is here keeps its
 * choice; one whose key is missing gives way to the decision for its own
 * model, as in the chat. Never throws for a missing model: the caller says
 * `problem`, and `serve` refuses to start with it.
 */
export async function attachModelForCallers(input: CallersModelInput): Promise<CallersModel> {
  const forCallers = !input.owner;
  // What "Robutler's models may run" means: the agent's own credential for
  // other callers' turns, the sign-in for the owner's own.
  const ownCredential =
    input.owner?.signedIn ??
    input.ownCredential ??
    (async () => {
      const { resolveAgentCredential } = await import('../server/agent-credential');
      return Boolean(await resolveAgentCredential(input.agentName));
    });
  const declared = input.declaredSkills
    .filter((name) => !input.notBuilt.has(name))
    .map((name) => findProvider(name))
    .find((provider): provider is LLMProvider => provider !== undefined);
  let wanted = input.model;
  if (declared?.id === 'proxy') {
    // The file names Robutler's models: its skill was built with no sign-in
    // (`callersPay`), and runs only on the agent's own credential.
    const built = [...input.byName.entries()].find(([name]) => findProvider(name)?.id === 'proxy')?.[1] as
      | { modelConfig?: { model?: string } }
      | undefined;
    const model = built?.modelConfig?.model ?? input.model ?? PROXY_DEFAULT_MODEL;
    if (!(await ownCredential())) {
      const access: ModelAccess = { kind: 'none', reason: `${model} is served by Robutler`, forCallers };
      return { access, problem: unavailableMessage(access) };
    }
    return { access: { kind: 'proxy', model, provider: declared, forCallers }, model };
  }
  if (declared && (declared.credential !== 'api-key' || hasKey(declared, input.env))) {
    // The agent's own provider, with its key here: its choice stands.
    const own = modelForProvider(declared.id, input.model);
    const model = own
      ? own.includes('/') ? own : `${declared.id}/${own}`
      : declared.defaultModel ? `${declared.id}/${declared.defaultModel}` : input.model;
    return { access: { kind: 'direct', model, provider: declared, forCallers }, model };
  }
  if (declared) {
    // Listed, and its key is not here: whatever runs instead takes its place.
    const own = modelForProvider(declared.id, input.model);
    wanted = own
      ? own.includes('/') ? own : `${declared.id}/${own}`
      : declared.defaultModel ? `${declared.id}/${declared.defaultModel}` : undefined;
    const index = input.skills.findIndex((skill) => (skill as { name?: string }).name === declared.id);
    if (index !== -1) input.skills.splice(index, 1);
  }
  const access: ModelAccess = { ...(await resolveModelAccess(wanted, { signedIn: ownCredential, env: input.env })), forCallers };
  if (access.kind === 'direct') {
    const { resolveSkillsByName } = await import('../skills/resolve');
    const built = await resolveSkillsByName([access.provider!.id], { model: access.model, apiKeys: input.apiKeys });
    input.skills.push(...built.skills);
    if (built.failed.length) {
      return { access, model: access.model, problem: `Could not load the ${access.provider!.id} client: ${built.failed[0].reason}` };
    }
    return { access, model: access.model };
  }
  if (access.kind === 'proxy') {
    const { LLMProxySkill } = await import('../skills/llm/proxy/skill');
    const payer = input.owner ? { platformToken: input.owner.platformToken } : { callersPay: true };
    input.skills.push(
      new LLMProxySkill({ model: access.model, ...(input.proxyUrl ? { proxyUrl: input.proxyUrl } : {}), ...payer }) as unknown as ISkill,
    );
    return { access, model: access.model };
  }
  return { access, problem: unavailableMessage(access) };
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
