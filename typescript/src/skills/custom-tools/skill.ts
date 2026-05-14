/**
 * `CustomToolsSkill`
 *
 * Reads `agent_configs.skills.custom_tools.tools[]` and registers each
 * entry as an LLM tool on the agent. The tool handler resolves the
 * referenced function via `FunctionRuntimeSkill.invoke(use, ctx)` and
 * returns the executor result to the LLM (sanitised by the existing
 * tool-result sanitiser).
 *
 * Validation at create-time:
 *   - `parameters` is a syntactically valid JSON Schema (type: object).
 *   - Referenced function exists in `agent_configs.functions`.
 *   - Smoke run with synthesized payload (handled by the executor).
 */

import { Skill } from '../../core/skill';
import { prompt } from '../../core/decorators';
import type { Context, SkillConfig, Tool } from '../../core/types';
import type { JSONSchema } from '../../uamp/types';
import type { FunctionRuntimeSkill } from '../functions/skill';
import type { SerializableContext } from '../functions/executor-client';
import type { FunctionLimitsResolved } from '../functions/context';

export interface CustomToolEntry {
  id: string;
  name: string;
  description?: string;
  parameters?: JSONSchema;
  /** Function name to invoke (required). */
  use: string;
  enabled?: boolean;
}

export interface CustomToolsSkillConfig extends SkillConfig {
  tools?: CustomToolEntry[];
  runtime?: FunctionRuntimeSkill;
  resolveRuntime?: () => FunctionRuntimeSkill | undefined;
}

const DEFAULT_LIMITS: FunctionLimitsResolved = {
  wallMs: 30_000,
  cpuMs: 5_000,
  memoryMb: 128,
  ingressBytes: 1_000_000,
  egressBytes: 1_000_000,
};

function defaultRuntimeResolver(skill: CustomToolsSkill): () => FunctionRuntimeSkill | undefined {
  return () => {
    const agent = (skill as unknown as { _agent?: { skills?: Array<{ name?: string }> } })._agent;
    if (!agent?.skills) return undefined;
    return agent.skills.find((s) => s?.name === 'function-runtime') as FunctionRuntimeSkill | undefined;
  };
}

export class CustomToolsSkill extends Skill {
  readonly name = 'custom_tools';
  readonly dependencies = ['function-runtime'] as const;

  private readonly toolEntries: CustomToolEntry[];
  private explicitRuntime?: FunctionRuntimeSkill;
  private readonly resolveRuntime: () => FunctionRuntimeSkill | undefined;
  // @ts-expect-error read indirectly through `defaultRuntimeResolver`
  private _agent?: unknown;

  constructor(config: CustomToolsSkillConfig = {}) {
    super(config);
    this.toolEntries = (config.tools ?? []).filter((t) => t.enabled !== false);
    this.explicitRuntime = config.runtime;
    this.resolveRuntime =
      config.resolveRuntime ?? defaultRuntimeResolver(this);
    this.registerAll();
  }

  setAgent(agent: unknown): void {
    this._agent = agent;
  }

  list(): readonly CustomToolEntry[] {
    return this.toolEntries;
  }

  validate(declaredFunctions: ReadonlySet<string>): Array<{ id: string; error: string }> {
    const errors: Array<{ id: string; error: string }> = [];
    const seenNames = new Set<string>();
    for (const t of this.toolEntries) {
      if (!t.id) errors.push({ id: '', error: 'tool.id is required' });
      if (!t.name) errors.push({ id: t.id, error: 'tool.name is required' });
      if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(t.name ?? '')) {
        errors.push({ id: t.id, error: `tool.name "${t.name}" is invalid` });
      }
      if (seenNames.has(t.name)) {
        errors.push({ id: t.id, error: `duplicate tool.name "${t.name}"` });
      } else {
        seenNames.add(t.name);
      }
      if (!t.use) {
        errors.push({ id: t.id, error: 'custom_tools entries require "use"' });
      } else if (!declaredFunctions.has(t.use)) {
        errors.push({ id: t.id, error: `function "${t.use}" not declared in agent_configs.functions` });
      }
      const schemaErrors = validateJsonSchema(t.parameters);
      for (const err of schemaErrors) {
        errors.push({ id: t.id, error: `parameters: ${err}` });
      }
    }
    return errors;
  }

  private resolveRuntimeOrThrow(): FunctionRuntimeSkill {
    const r = this.explicitRuntime ?? this.resolveRuntime();
    if (!r) {
      throw new Error(
        'CustomToolsSkill requires FunctionRuntimeSkill to be mounted (auto-mounted via Skill.dependencies).',
      );
    }
    return r;
  }

  private registerAll(): void {
    for (const entry of this.toolEntries) {
      const tool: Tool = {
        name: entry.name,
        description: entry.description,
        parameters: entry.parameters,
        enabled: true,
        handler: async (params, ctx) => {
          const runtime = this.resolveRuntimeOrThrow();
          const serializable: SerializableContext = {
            source: {
              skill: 'custom_tools',
              consumerId: entry.id,
              invocationId: cryptoRandomId(),
            },
            toolCall: { name: entry.name, params, callId: cryptoRandomId() },
            auth: (() => {
              // The transport-layer AuthInfo uses snake_case; the function
              // runtime envelope expects camelCase. We accept BOTH casings
              // here defensively — different transports (HTTP route vs.
              // delegate adapter) have historically differed.
              const a = (ctx.auth ?? {}) as unknown as Record<string, unknown>;
              const userId = (a.user_id as string | undefined) ?? (a.userId as string | undefined) ?? null;
              const agentId = (a.agent_id as string | undefined) ?? (a.agentId as string | undefined) ?? null;
              return {
                userId,
                agentId,
                scopes: (a.scopes as string[] | undefined) ?? [],
                authenticated:
                  (a.authenticated as boolean | undefined) ?? Boolean(userId),
                claims: a.claims as Record<string, unknown> | undefined,
              };
            })(),
            limits: DEFAULT_LIMITS,
          };
          const r = await runtime.invoke<unknown>(entry.use, serializable);
          if (!r.ok) {
            // Sanitised error for the LLM — never raw internals.
            return {
              ok: false,
              error: {
                code: r.errorCode ?? 'FUNCTION_ERROR',
                message: r.errorMessage ?? 'function failed',
              },
            };
          }
          // Surface metering so the agent_fee charge type can settle.
          if (r._metering) {
            ctx.set('_metering', r._metering);
          }
          return r.result;
        },
      };
      this.registerTool(tool);
    }
  }

  @prompt({ priority: 72, name: 'customToolsRuntime', scope: 'all' })
  customToolsRuntime(_ctx: Context): string {
    const lines: string[] = ['## Custom LLM tools (function-backed)'];

    if (this.toolEntries.length === 0) {
      lines.push(
        'No custom tools are configured. To expose one, declare a function (`declare_function`) and attach it via `add_to_skill skill="custom_tools"` with a JSON-schema for parameters.',
      );
    } else {
      lines.push('You can call these tools just like any platform tool. The implementation is a user-authored function executed in the sandbox:');
      for (const t of this.toolEntries) {
        lines.push(`- \`${t.name}\` — ${t.description ?? '(no description)'} (impl: ${t.use})`);
      }
    }

    lines.push(
      '',
      '### When to reach for these vs platform tools',
      '- Custom tools wrap **user / owner-authored business logic** (proprietary APIs, paid third-party services, agent-specific computation). Prefer them whenever the agent\'s described capability matches one — they exist precisely because the platform tool surface is intentionally generic.',
      '- Reach for platform tools (`search`, `delegate`, `web_search`, `text_editor`, `bash`, etc.) for the generic equivalents — file editing, agent discovery, public web fetch, etc. Do NOT call a custom tool whose description sounds vaguely related when a platform tool already covers the case.',
      '- Custom tool errors come back as `{ ok: false, error: { code, message } }` (sanitised) — surface the message to the user when actionable; do NOT retry the same call with identical params after an error.',
      '',
      '### Schema discipline (when declaring NEW custom tools)',
      '- `parameters` MUST be a JSON Schema with `type: "object"`. Every parameter the function actually reads should appear under `properties` with a clear description — the LLM uses these to decide what to pass. Untyped or undocumented params are guess-work.',
      '- Mark required parameters in `required: [...]`. Optional ones with sensible defaults documented in the description.',
      '- Tool names must match `^[A-Za-z][A-Za-z0-9_]{0,63}$`. Pick verb-style names (`fetch_inventory`, `convert_currency`) so the LLM can route to them naturally.',
    );

    return lines.join('\n');
  }
}

function validateJsonSchema(s: JSONSchema | undefined): string[] {
  if (s === undefined) return [];
  if (s === null || typeof s !== 'object') return ['must be an object'];
  if ((s as Record<string, unknown>).type !== 'object') return ['type must be "object"'];
  const props = (s as Record<string, unknown>).properties;
  if (props !== undefined && (props === null || typeof props !== 'object')) {
    return ['properties must be an object'];
  }
  return [];
}

function cryptoRandomId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `inv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}
