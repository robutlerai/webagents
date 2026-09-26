/**
 * Resolve skill NAMES, as written in an agent file, to skill INSTANCES.
 *
 * WHY THIS EXISTS (2026-09-23). Two places needed this and only one had it.
 * `src/daemon/server.ts` carried a private `resolveSkills()` that understood
 * five names; `webagents serve` had nothing, so `serve-action.ts` built a
 * `BaseAgent` from `name`/`description`/`instructions` only and silently
 * DISCARDED the `skills` and `model` that `webagents init` had just written
 * into `agent.json`. A scaffolded agent therefore served with no LLM skill at
 * all, which reads as "the agent does not answer" rather than as a missing
 * feature.
 *
 * LLM providers come from the provider registry (`skills/llm/providers.ts`), so
 * every alias that registry knows about (`claude` for anthropic, `gemini` for
 * google, `grok` for xai) works here too, and adding a provider is one edit
 * rather than three.
 *
 * Unknown names are REPORTED, not swallowed. The daemon's version logged a
 * warning and continued, which is right for a long-running process, but the
 * caller should be the one deciding: `serve` wants to fail loudly on a typo'd
 * skill name, because the agent it would otherwise start is not the agent the
 * file describes.
 *
 * THE FILE'S `model:` REACHES THE LLM SKILL (2026-09-24). The loaders built
 * each LLM skill with no config, and every one of them then falls back to its
 * own hardcoded model (`gpt-4o` for OpenAI). So the scaffold `init` writes,
 * `model: openai/gpt-4o-mini` with `skills: [openai]`, sent `gpt-4o` on every
 * request, in `serve` and in the REPL alike: a first-time developer billed at
 * gpt-4o rates by a file that says gpt-4o-mini. Measured against a stand-in
 * endpoint that logged the model it received.
 */

import type { ISkill } from '../core/types';
import { findProvider, modelForProvider } from './llm/providers';

/** Options for {@link resolveSkillsByName}. */
export interface ResolveSkillsOptions {
  /**
   * The agent's `model:`, as `provider/model` or a bare id. Given to the LLM
   * skill whose provider it names, or to any LLM skill when it names none.
   */
  model?: string;
  /**
   * For the `proxy` skill (Robutler's LLM proxy): where it is and who pays.
   * The CLI passes the platform's `/llm` socket and the stored login token.
   */
  proxy?: { proxyUrl?: string; platformToken?: string | (() => string | null | undefined | Promise<string | null | undefined>) };
  /**
   * Provider keys to hand to the LLM skill itself, by provider id, for keys
   * that are not in the environment (the CLI's stored keys). Given to the
   * skill's own config so they never enter `process.env`, which the `shell`
   * skill passes to every command it runs.
   */
  apiKeys?: Partial<Record<string, string>>;
  /**
   * The agent's folder: where `filesystem`, `shell` and `todo` work, as the
   * Python loader roots them (2026-09-25). Unset, they work in `process.cwd()`.
   */
  agentDir?: string;
  /**
   * The signed-in person's platform token, for `discovery` to search with
   * when the agent has no credential of its own (2026-09-25). ONLY the chat
   * and `-p` pass it, where the person at the terminal is the only caller;
   * `serve` and the daemon must not, or every caller would search as the
   * owner (`discovery/skill.ts`, rule 3).
   */
  personToken?: () => Promise<string | null | undefined>;
}

export interface ResolvedSkills {
  skills: ISkill[];
  /** Each loaded skill by the name the agent file gave it, for `access.tools` (ADR-0045). */
  byName: Map<string, ISkill>;
  /** Names that matched nothing. Never silently dropped. */
  unknown: string[];
  /** Names that matched but whose module failed to load, with the reason. */
  failed: { name: string; reason: string }[];
}

/**
 * Non-LLM skills selectable by name from an agent file. Each gets the entry's
 * config, `{}` for a bare name: `- rest: {sign: always}` configures `rest`
 * (2026-09-25). Until then only names reached here, and `serve` turned a
 * config entry into the name "[object Object]".
 */
const NON_LLM_LOADERS: Record<string, (config: Record<string, unknown>, options?: ResolveSkillsOptions) => Promise<ISkill>> = {
  discovery: async (_config, options) => {
    const { PortalDiscoverySkill } = await import('./discovery/skill.js');
    return new PortalDiscoverySkill(options?.personToken ? { personToken: options.personToken } : {}) as unknown as ISkill;
  },
  filesystem: async (config, options) => {
    const { FilesystemSkill } = await import('./filesystem/skill.js');
    return new FilesystemSkill({ ...(options?.agentDir ? { baseDir: options.agentDir } : {}), ...config }) as unknown as ISkill;
  },
  shell: async (config, options) => {
    const { ShellSkill } = await import('./shell/skill.js');
    return new ShellSkill({ ...(options?.agentDir ? { baseDir: options.agentDir } : {}), ...config }) as unknown as ISkill;
  },
  // Task tracking, the design the Python skill adopted (2026-09-25), kept in
  // `.webagents/todos.json` in the agent's folder.
  todo: async (config, options) => {
    const { TodoSkill } = await import('./todo/skill.js');
    const { join } = await import('node:path');
    return new TodoSkill({
      ...(options?.agentDir ? { filePath: join(options.agentDir, '.webagents', 'todos.json') } : {}),
      ...config,
    }) as unknown as ISkill;
  },
  rest: async (config) => {
    const { RestSkill } = await import('./rest/skill.js');
    return new RestSkill(config) as unknown as ISkill;
  },
  // Conversations kept per verified caller when served, the Python skill's
  // twin (2026-09-25, `session/skill.ts`). The chat reads the entry's
  // `backend` itself and does not load it.
  session: async (config, options) => {
    const { SessionSkill } = await import('./session/skill.js');
    return new SessionSkill({ ...(options?.agentDir ? { agentDir: options.agentDir } : {}), ...config }) as unknown as ISkill;
  },
  // The Python SDK's `web` skill, the same tool (2026-09-25).
  web: async (config) => {
    const { WebSkill } = await import('./web/skill.js');
    return new WebSkill(config) as unknown as ISkill;
  },
};

/** A `skills:` entry: a bare name, or `{name: config}`. */
export type SkillEntryInput = string | Record<string, unknown>;

/** The name and config of one entry; `undefined` name for an entry that names nothing. */
export function skillEntryParts(entry: SkillEntryInput): { name?: string; config: Record<string, unknown> } {
  if (typeof entry === 'string') return { name: entry, config: {} };
  if (entry && typeof entry === 'object') {
    const keys = Object.keys(entry);
    if (keys.length === 1) {
      const value = entry[keys[0]];
      const config = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
      return { name: keys[0], config };
    }
  }
  return { config: {} };
}

/**
 * LLM providers this SDK can construct from a bare name, by registry id. The
 * model, when given, is passed through as written: every adapter strips the
 * `provider/` prefix itself.
 */
const LLM_LOADERS: Record<
  string,
  (model?: string, options?: ResolveSkillsOptions, entry?: Record<string, unknown>) => Promise<ISkill>
> = {
  // Robutler's LLM proxy (2026-09-24). Listed in the provider registry and
  // selectable in `skills:`, but no loader existed, so `skills: [proxy]`
  // matched `hasLLM`, skipped the fallback and left the agent with no model.
  // `proxy/<model>` names the route, not the model: the platform does not
  // strip the prefix, so it is removed here.
  proxy: async (model, options) => {
    const { LLMProxySkill } = await import('./llm/proxy/skill.js');
    const bare = model?.startsWith('proxy/') ? model.slice('proxy/'.length) : model;
    return new LLMProxySkill({
      ...(bare ? { model: bare } : {}),
      ...(options?.proxy?.proxyUrl ? { proxyUrl: options.proxy.proxyUrl } : {}),
      ...(options?.proxy?.platformToken ? { platformToken: options.proxy.platformToken } : {}),
    }) as unknown as ISkill;
  },
  openai: async (model, options, entry) => {
    const { OpenAISkill } = await import('./llm/openai/skill.js');
    return new OpenAISkill({ ...llmConfig('openai', model, options), ...entry }) as unknown as ISkill;
  },
  anthropic: async (model, options, entry) => {
    const { AnthropicSkill } = await import('./llm/anthropic/skill.js');
    return new AnthropicSkill({ ...llmConfig('anthropic', model, options), ...entry }) as unknown as ISkill;
  },
  google: async (model, options, entry) => {
    const { GoogleSkill } = await import('./llm/google/skill.js');
    return new GoogleSkill({ ...llmConfig('google', model, options), ...entry }) as unknown as ISkill;
  },
  xai: async (model, options, entry) => {
    const { XAISkill } = await import('./llm/xai/skill.js');
    return new XAISkill({ ...llmConfig('xai', model, options), ...entry }) as unknown as ISkill;
  },
  // Fireworks (2026-09-25): in the provider registry, and served by the
  // Python SDK, but no loader, so `skills: [fireworks]` and a `fireworks/...`
  // model did nothing here. The skill prepends `accounts/fireworks/models/`
  // itself, so the model goes in without its `fireworks/` prefix.
  fireworks: async (model, options, entry) => {
    const { FireworksSkill } = await import('./llm/fireworks/skill.js');
    const bare = model?.startsWith('fireworks/') ? model.slice('fireworks/'.length) : model;
    return new FireworksSkill({ ...llmConfig('fireworks', bare, options), ...entry }) as unknown as ISkill;
  },
};

/**
 * An entry's config for a model skill, in this SDK's key names: the agent
 * file's `api_key` and `base_url` (the Python skills' names) become `apiKey`
 * and `baseUrl`; `temperature` and `max_tokens` are the same in both. Its
 * `model` is applied by the caller.
 */
function llmEntryConfig(config: Record<string, unknown>): Record<string, unknown> {
  const { model: _model, api_key, base_url, ...rest } = config;
  return {
    ...rest,
    ...(typeof api_key === 'string' && api_key ? { apiKey: api_key } : {}),
    ...(typeof base_url === 'string' && base_url ? { baseUrl: base_url } : {}),
  };
}

/** A provider skill's config: the model when given, and a key the environment lacks. */
function llmConfig(providerId: string, model: string | undefined, options?: ResolveSkillsOptions): { model?: string; apiKey?: string } {
  const apiKey = options?.apiKeys?.[providerId];
  return { ...(model ? { model } : {}), ...(apiKey ? { apiKey } : {}) };
}

/** Every name `skills:` can use here: the model providers and the local skills, sorted. */
export function resolvableSkillNames(): string[] {
  return [...Object.keys(LLM_LOADERS), ...Object.keys(NON_LLM_LOADERS)].sort();
}

export async function resolveSkillsByName(
  entries: readonly SkillEntryInput[],
  options: ResolveSkillsOptions = {},
): Promise<ResolvedSkills> {
  const skills: ISkill[] = [];
  const byName = new Map<string, ISkill>();
  const unknown: string[] = [];
  const failed: { name: string; reason: string }[] = [];

  for (const entry of entries) {
    const { name: entryName, config } = skillEntryParts(entry);
    if (!entryName) continue;
    const name = entryName;
    const lower = String(name).toLowerCase();
    // A provider alias resolves to its canonical id first, so `claude` and
    // `anthropic` land on the same loader.
    const provider = findProvider(lower);
    const llmLoader = provider ? LLM_LOADERS[provider.id] : undefined;
    // The proxy serves every provider's models, so it takes the agent's model
    // whatever its prefix; `modelForProvider` would drop `openai/gpt-4o` as
    // "another provider's".
    const model = provider?.id === 'proxy' ? options.model : provider ? modelForProvider(provider.id, options.model) : undefined;
    const nonLlm = NON_LLM_LOADERS[lower];
    // An entry's own config reaches the skill, and its `model` wins, as in
    // the Python loader: `- openai: {model: gpt-4o, temperature: 0.2}`.
    const ownModel = typeof config.model === 'string' && config.model ? config.model : undefined;
    const loader = llmLoader
      ? () => llmLoader(ownModel ?? model, options, llmEntryConfig(config))
      : nonLlm
        ? () => nonLlm(config, options)
        : undefined;

    if (!loader) {
      unknown.push(name);
      continue;
    }
    try {
      const skill = await loader();
      skills.push(skill);
      byName.set(name, skill);
    } catch (err) {
      failed.push({ name, reason: (err as Error).message });
    }
  }

  return { skills, byName, unknown, failed };
}
