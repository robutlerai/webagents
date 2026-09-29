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
import type { SandboxDeclaration } from '../sandbox/policy';
import { findProvider, modelForProvider } from './llm/providers';
import type { MCPSkillConfig } from './mcp/skill';

/** Options for {@link resolveSkillsByName}. */
export interface ResolveSkillsOptions {
  /**
   * The agent's `model:`, as `provider/model` or a bare id. Given to the LLM
   * skill whose provider it names, or to any LLM skill when it names none.
   */
  model?: string;
  /**
   * For the `proxy` skill (Robutler's LLM proxy): where it is and who pays.
   * The chat passes the platform's `/llm` socket and the stored login token;
   * `serve` and the daemon pass `callersPay` and never a token, so each call
   * is paid by the caller's own payment token (S-327).
   */
  proxy?: {
    proxyUrl?: string;
    platformToken?: string | (() => string | null | undefined | Promise<string | null | undefined>);
    callersPay?: boolean;
  };
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
  /**
   * The agent file's `sandbox:` block, checked by the loader, for `shell`
   * to enforce at the OS level (2026-09-26, gap-closure plan item 1.2). A
   * skill-level `sandbox` key in the entry's own config still wins, because
   * a more specific declaration should, as in the Python loader.
   */
  sandbox?: SandboxDeclaration;
  /**
   * The agent file's `agent_skills:` (plan item 1.4, 2026-09-26): folders of
   * SKILL.md skills kept outside `.agents/skills`. With `agentDir` set, the
   * SKILL.md skills there and under `<agentDir>/.agents/skills` are loaded as
   * one skill named `agent_skills` (`skillmd/skillmd-skill.ts`), as the
   * Python `load_skills` does.
   */
  agentSkills?: readonly string[];
  /**
   * The agent file being run, for `filesystem` (S-314, 2026-09-27): a write
   * to it asks the owner even under a name the agent-file patterns do not
   * cover. The chat passes the file it loaded; the daemon's files are named
   * by the patterns already.
   */
  agentFile?: string;
  /**
   * The chat's yes/no for a file-tool write to one of the agent's control
   * files (S-314): the interactive chat alone passes it, shows the diff and
   * asks the person at the terminal. Without it (`serve`, the daemon, `-p`)
   * `filesystem` refuses such a write with a sentence.
   */
  confirmControlWrite?: (file: string, diff: string) => Promise<boolean>;
}

export interface ResolvedSkills {
  skills: ISkill[];
  /** Each loaded skill by the name the agent file gave it, for `access.tools` (ADR-0045). */
  byName: Map<string, ISkill>;
  /** Names that matched nothing. Never silently dropped. */
  unknown: string[];
  /** Names that matched but whose module failed to load, with the reason. */
  failed: { name: string; reason: string }[];
  /**
   * The SKILL.md skills (plan item 1.4): the names loaded, the folders that
   * could not load (reported, never fatal) and the loader's warnings, for
   * the caller to say (`skillmdReportLines`).
   */
  skillmd: { skills: string[]; skipped: { name: string; location: string; reason: string }[]; warnings: string[] };
}

/**
 * What did not load among the SKILL.md skills, one line each, in the words
 * the Python loaders print (`say_skillmd_report`; fixture `skillmd.json`,
 * `messages.load_skipped` and `messages.load_warning`).
 */
export function skillmdReportLines(skillmd: ResolvedSkills['skillmd']): string[] {
  return [
    ...skillmd.skipped.map((s) => `SKILL.md skill ${s.name} at ${s.location} skipped: ${s.reason}`),
    ...skillmd.warnings.map((w) => `SKILL.md skills: ${w}`),
  ];
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
    return new FilesystemSkill({
      ...(options?.agentDir ? { baseDir: options.agentDir } : {}),
      ...(options?.agentFile ? { agentFile: options.agentFile } : {}),
      ...(options?.confirmControlWrite ? { confirmControlWrite: options.confirmControlWrite } : {}),
      ...config,
    }) as unknown as ISkill;
  },
  shell: async (config, options) => {
    const { ShellSkill } = await import('./shell/skill.js');
    return new ShellSkill({
      ...(options?.agentDir ? { baseDir: options.agentDir } : {}),
      ...(options?.sandbox ? { sandbox: options.sandbox } : {}),
      ...config,
    }) as unknown as ISkill;
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
  // Memory scoped by verified caller (plan item 2.1, 2026-09-26): `- memory`
  // or `- memory: {local, portal, notes_budget, compaction}`, checked here as
  // the Python loader checks it (`memory/skill.ts` `parseMemoryConfig`, pinned
  // by `python/tests/fixtures/memory_tool/definition.json`), so a mistyped
  // entry is reported as a skill that failed to load, with the sentence.
  memory: async (config, options) => {
    const { MemorySkill, parseMemoryConfig } = await import('./memory/skill.js');
    const parsed = parseMemoryConfig(config);
    return new MemorySkill({
      local: parsed.local,
      portal: parsed.portal,
      notes_budget: parsed.notesBudget,
      compaction: parsed.compaction,
      ...(options?.agentDir ? { agentDir: options.agentDir } : {}),
    }) as unknown as ISkill;
  },
  // The Python SDK's `web` skill, the same tool (2026-09-25).
  web: async (config) => {
    const { WebSkill } = await import('./web/skill.js');
    return new WebSkill(config) as unknown as ISkill;
  },
  // MCP servers named in the file (plan item 0.3, 2026-09-26): the two shapes
  // the Python loader accepts (`mcp/config.ts`, pinned by the shared fixture
  // `python/tests/fixtures/mcp_tool/config_shapes.json`). The SDK is loaded
  // HERE, so a file naming servers the SDK cannot serve is reported as a skill
  // that failed to load, with the reason, rather than as an agent that quietly
  // has no MCP tools. A bare `- mcp` reads mcp.json next to the agent file.
  mcp: async (config, options) => {
    const { MCPSkill, loadMcpSdk, ownerReferenceSources } = await import('./mcp/skill.js');
    await loadMcpSdk();
    const servers = Object.keys(config).length ? { mcp: config as unknown as NonNullable<MCPSkillConfig['mcp']> } : {};
    return new MCPSkill({
      ...(options?.agentDir ? { baseDir: options.agentDir } : {}),
      ...servers,
      // An agent FILE, run by its local owner, is the one place `${env:NAME}`
      // and `${secret:NAME}` resolve (S-295, 2026-09-26): the sources are the
      // owner's own environment and keystore. A host that builds the skill
      // from saved data passes none, and expands nothing.
      references: ownerReferenceSources(),
    }) as unknown as ISkill;
  },
  // The A2A v1.0 transport (plan items 1.1 and 1.3, 2026-09-26), the same
  // name and keys as the Python loader (`cli/agent_builder.py`), pinned by
  // `python/tests/fixtures/a2a/config_shapes.json`: `peers`,
  // `task_ttl_seconds`, `blocking_timeout_seconds` and the card fields.
  a2a: async (config) => {
    const { A2ATransportSkill } = await import('./transport/a2a/skill.js');
    return new A2ATransportSkill(config) as unknown as ISkill;
  },
  // The other three transport names the Python loader knows (plan item 1.1,
  // 2026-09-26), with its config keys, pinned by the `config_shapes` of
  // `python/tests/fixtures/acp/acp_protocol.json`: `completions` and
  // `realtime` read no key, and `acp` reads `sessions_dir`. A key the Python
  // skill does not read is not passed on here either, so the two SDKs build
  // the same skill from the same entry.
  completions: async () => {
    const { CompletionsTransportSkill } = await import('./transport/completions/skill.js');
    return new CompletionsTransportSkill({}) as unknown as ISkill;
  },
  realtime: async () => {
    const { RealtimeTransportSkill } = await import('./transport/realtime/skill.js');
    return new RealtimeTransportSkill({}) as unknown as ISkill;
  },
  acp: async (config) => {
    const { ACPTransportSkill } = await import('./transport/acp/skill.js');
    const sessionsDir = typeof config.sessions_dir === 'string' && config.sessions_dir ? config.sessions_dir : undefined;
    return new ACPTransportSkill(sessionsDir ? { sessionsDir } : {}) as unknown as ISkill;
  },
  // TrustFlow (plan items 2.5 and 2.7, 2026-09-26): the `trust` tool and this
  // agent's own signed record, the Python `trust` skill's twin
  // (`skills/trust/trust-skill.ts`, pinned by
  // `python/tests/fixtures/trust/trust_tool_definition.json`). TrustFlow is a
  // platform service; the skill asks the platform as the agent.
  trust: async (config) => {
    const { TrustSkill } = await import('./trust/trust-skill.js');
    return new TrustSkill(config) as unknown as ISkill;
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
      ...(options?.proxy?.callersPay
        ? { callersPay: true }
        : options?.proxy?.platformToken ? { platformToken: options.proxy.platformToken } : {}),
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
  // Local models through Ollama (plan item 2.8, 2026-09-26): the OpenAI
  // wire shape at OLLAMA_BASE_URL, no key. The skill strips nothing, so the
  // `ollama/` prefix comes off here.
  ollama: async (model, _options, entry) => {
    const { OllamaSkill } = await import('./llm/ollama/skill.js');
    const bare = model?.startsWith('ollama/') ? model.slice('ollama/'.length) : model;
    return new OllamaSkill({ ...(bare ? { model: bare } : {}), ...entry }) as unknown as ISkill;
  },
};

/**
 * An entry's config for a model skill, in this SDK's key names: the agent
 * file's `api_key` and `base_url` (the Python skills' names) become `apiKey`
 * and the base URL; `temperature` and `max_tokens` are the same in both. Its
 * `model` is applied by the caller.
 *
 * THE BASE URL IS SET UNDER BOTH SPELLINGS (2026-09-26, found by lane
 * w1-protocols). This mapped `base_url` to `baseUrl` alone, and `OpenAISkill`
 * reads `baseURL` (`llm/openai/skill.ts`), so the file's key never arrived
 * and only the OPENAI_BASE_URL variable pointed that skill anywhere; the
 * Fireworks skill reads `baseUrl`. Each skill finds the spelling it reads.
 * Pinned by `python/tests/fixtures/w2ops/models.json` (`base_url`), which
 * Ollama's resolution depends on.
 */
function llmEntryConfig(config: Record<string, unknown>): Record<string, unknown> {
  const { model: _model, api_key, base_url, ...rest } = config;
  return {
    ...rest,
    ...(typeof api_key === 'string' && api_key ? { apiKey: api_key } : {}),
    ...(typeof base_url === 'string' && base_url ? { baseURL: base_url, baseUrl: base_url } : {}),
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

/** Options for {@link withFallbackModels}. */
export interface FallbackModelsOptions extends ResolveSkillsOptions {
  /** The agent's own model as `provider/model`, for the chain's first label and the notes. */
  primaryModel?: string;
  /** What the decision sees: the environment plus stored keys, for "its key is not set". */
  env?: Record<string, string | undefined>;
  /**
   * Other callers' turns (`serve`, the daemon; S-327): a Robutler fallback
   * needs the agent's own platform credential (pass `proxy` with
   * `callersPay` only when it has one), and the refusal says so.
   */
  forCallers?: boolean;
}

/**
 * The agent's LLM skill wrapped with its `fallback_models:` (plan item 2.8,
 * 2026-09-26): the first skill in `skills` that carries a model handoff is
 * the primary, each fallback `provider/model` is built through the loaders
 * above, and the primary's place in the list is taken by a
 * `FailoverLLMSkill` over the chain. A fallback that cannot be built (an
 * unknown provider, its key not set here, Robutler's models without a
 * sign-in) is reported in `failed` and left out, never fatal: the agent
 * still runs on what it has. Nothing changes when there is no LLM skill to
 * wrap or no fallback to add.
 */
export async function withFallbackModels(
  skills: ISkill[],
  fallbackModels: readonly string[],
  options: FallbackModelsOptions = {},
): Promise<{ skills: ISkill[]; failed: { name: string; reason: string }[]; failover?: import('./llm/failover/skill').FailoverLLMSkill }> {
  const failed: { name: string; reason: string }[] = [];
  if (!fallbackModels.length) return { skills, failed };
  const index = skills.findIndex((skill) => (skill.handoffs?.length ?? 0) > 0);
  if (index === -1) return { skills, failed };
  const primary = skills[index];
  const env = options.env ?? (typeof process !== 'undefined' ? process.env : {});
  const { FailoverLLMSkill } = await import('./llm/failover/skill.js');
  const { missingProviderKey } = await import('./llm/providers.js');
  // `viaRobutler` is unset for the primary: the chat reads its own access for that one (2026-09-29).
  const members: Array<{ skill: ISkill; model: string; viaRobutler?: boolean }> = [
    { skill: primary, model: options.primaryModel ?? modelLabelOf(primary) },
  ];
  for (const fallback of fallbackModels) {
    const [prefix] = fallback.split('/');
    const robutler = prefix === 'auto' || prefix === 'proxy' || prefix === 'robutler';
    const provider = robutler ? findProvider('proxy') : findProvider(prefix);
    if (!provider || !LLM_LOADERS[provider.id]) {
      failed.push({ name: `fallback ${fallback}`, reason: `this SDK has no client for ${prefix}` });
      continue;
    }
    if (robutler && !options.proxy?.proxyUrl) {
      // For other callers (`callersPay`, S-327) the way in is the agent's own
      // platform credential, never a sign-in: the Python loader's words.
      const reason = options.forCallers
        ? "Robutler's models need the agent's own platform credential here"
        : "Robutler's models need a sign-in";
      failed.push({ name: `fallback ${fallback}`, reason });
      continue;
    }
    const missing = provider.credential === 'api-key' ? missingProviderKey([provider.id], { ...env, ...keysAsEnv(options.apiKeys, provider.id, provider.envVar) }) : undefined;
    if (missing) {
      failed.push({ name: `fallback ${fallback}`, reason: `${missing.envVar} is not set` });
      continue;
    }
    // `auto/...` goes to the proxy as written; `proxy/x` and `robutler/x` name the route.
    const model = robutler && prefix !== 'auto' ? fallback.slice(prefix.length + 1) : fallback;
    try {
      const skill = await LLM_LOADERS[provider.id](model, options, {});
      // `viaRobutler`: only a Robutler turn costs credits (the chat's cost line, 2026-09-29).
      members.push({ skill, model: robutler ? (prefix === 'auto' ? fallback : model) : fallback, viaRobutler: robutler });
    } catch (err) {
      failed.push({ name: `fallback ${fallback}`, reason: (err as Error).message });
    }
  }
  if (members.length === 1) return { skills, failed };
  const failover = new FailoverLLMSkill({ chain: members });
  const out = [...skills];
  out[index] = failover as unknown as ISkill;
  return { skills: out, failed, failover };
}

/** A stored key as the environment variable its provider reads, for the "is not set" check. */
function keysAsEnv(apiKeys: Partial<Record<string, string>> | undefined, providerId: string, envVar: string | undefined): Record<string, string> {
  const key = apiKeys?.[providerId];
  return key && envVar ? { [envVar]: key } : {};
}

/** A model skill's `provider/model`, from its config and the provider its name selects. */
function modelLabelOf(skill: ISkill): string {
  const own = (skill as { modelConfig?: { model?: string }; model?: string }).modelConfig?.model
    ?? (skill as { model?: string }).model
    ?? 'unknown';
  if (own.includes('/')) return own;
  const provider = findProvider(skill.name)?.id ?? (skill.name === 'llm-proxy' ? 'proxy' : skill.name);
  return `${provider}/${own}`;
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

  // SKILL.md skills (plan item 1.4): `.agents/skills/*` in the agent's
  // folder plus the folders `agent_skills:` names, as one skill under
  // `agent_skills`, owner-only until `access: tools:` opens it. Only with an
  // agent folder: the model-fallback call passes none and gets none.
  const skillmd: ResolvedSkills['skillmd'] = { skills: [], skipped: [], warnings: [] };
  if (options.agentDir) {
    const { discoverSkills } = await import('./skillmd/skillmd-loader.js');
    const found = discoverSkills(options.agentDir, options.agentSkills ?? []);
    skillmd.skills = found.skills.map((s) => s.name);
    skillmd.skipped = found.skipped.map((s) => ({ name: s.name, location: s.location, reason: s.reason }));
    skillmd.warnings = [...found.warnings];
    if (found.skills.length) {
      const { SKILL_KEY, SkillMdSkill } = await import('./skillmd/skillmd-skill.js');
      const skill = new SkillMdSkill({
        skills: found.skills,
        skipped: found.skipped,
        warnings: found.warnings,
        agentDir: options.agentDir,
        ...(options.sandbox ? { sandbox: options.sandbox } : {}),
      }) as unknown as ISkill;
      skills.push(skill);
      byName.set(SKILL_KEY, skill);
    }
  }

  return { skills, byName, unknown, failed, skillmd };
}
