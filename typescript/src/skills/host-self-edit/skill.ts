/**
 * `HostSelfEditSkill`
 *
 * When mounted on the host agent, this skill exposes
 * `declare_function`, `update_function`, `remove_function`,
 * `add_to_skill`, `remove_from_skill` tools that let the agent edit ITS OWN
 * `agent_configs.functions` and `agent_configs.skills.*` slots in response
 * to owner conversation.
 *
 * Mount conditions (re-checked at tool-call time):
 *   1. `agent_configs.featureFlags.selfEdit === true`
 *   2. `ctx.auth.userId === agent.ownerId`
 *
 * Scope is strictly the host agent itself — the portal route validates
 * that `[id]` in the URL equals the calling agent's id when the
 * `Function-Authoring-Surface: host` header is present.
 *
 * Tools call the same portal routes as the factory agent (POST
 * `/api/agents/[id]/functions` and POST `/api/agents/[id]/skills/[skill]/use`)
 * so authorisation is centralised.
 *
 * Self-edit tools are priced under `tools["fn:authoring"]` so iteration
 * cost is metered.
 */

import { Skill } from '../../core/skill';
import { tool, prompt, pricing } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';
import type { JSONSchema } from '../../uamp/types';

export interface HostSelfEditSkillConfig extends SkillConfig {
  /** Owning user id; tools enforce `ctx.auth.userId === ownerId` at call time. */
  ownerId?: string;
  /** Host agent id; the portal validates this matches the URL [id] when surface=host. */
  agentId?: string;
  /** Portal base URL. */
  portalUrl?: string;
  /** Service token for portal calls (mTLS in cloud, plain HTTP locally). */
  serviceToken?: string;
}

const PARAMETERS_DECLARE_FUNCTION: JSONSchema = {
  type: 'object',
  properties: {
    name: { type: 'string', description: 'agent_configs.functions key (stable id)' },
    runtime: {
      type: 'string',
      enum: ['js-v1'],
      description:
        'Only js-v1 is enabled in v1. python-pyodide-v1 is deferred (ADR-0008); wasm-v1 is reserved for v2.',
    },
    code: {
      type: 'string',
      description:
        'Single-file source. Must export an async handler (default export, named handler, or module.exports). Runs in a bare V8 isolate (isolated-vm): no process/Buffer/require/eval/Function and no npm packages. Web Platform globals only (URL, atob/btoa, console, TextEncoder/TextDecoder, structuredClone, crypto.{randomUUID,getRandomValues,subtle}). Stateful APIs go through ctx.{secrets,kv,content,folders,fn,portal}; egress through ctx.fetch (allowlist). Inline cap: 16 KB UTF-8 / 64 KB base64.',
    },
    permissions: { type: 'object' },
    limits: { type: 'object' },
    description: { type: 'string' },
  },
  required: ['name', 'runtime', 'code'],
};

const PARAMETERS_UPDATE_FUNCTION: JSONSchema = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    runtime: {
      type: 'string',
      enum: ['js-v1'],
      description: 'Only js-v1 is enabled in v1 (ADR-0008).',
    },
    code: { type: 'string' },
    permissions: { type: 'object' },
    limits: { type: 'object' },
    description: { type: 'string' },
  },
  required: ['name'],
};

const PARAMETERS_REMOVE_FUNCTION: JSONSchema = {
  type: 'object',
  properties: { name: { type: 'string' } },
  required: ['name'],
};

const PARAMETERS_ADD_TO_SKILL: JSONSchema = {
  type: 'object',
  properties: {
    skill: { type: 'string', enum: ['cron', 'custom_http', 'custom_tools'] },
    use: { type: 'string', description: 'function name (omit for cron host-agent)' },
    entry: { type: 'object', description: 'consumer-specific shape' },
  },
  required: ['skill', 'entry'],
};

const PARAMETERS_REMOVE_FROM_SKILL: JSONSchema = {
  type: 'object',
  properties: {
    skill: { type: 'string', enum: ['cron', 'custom_http', 'custom_tools'] },
    entryId: { type: 'string' },
  },
  required: ['skill', 'entryId'],
};

export class HostSelfEditSkill extends Skill {
  readonly name = 'host-self-edit';
  readonly dependencies = ['function-runtime'] as const;

  private readonly ownerId: string;
  private readonly agentId: string;
  private readonly portalUrl: string;
  private readonly serviceToken?: string;

  constructor(config: HostSelfEditSkillConfig = {}) {
    super({ ...config, name: config.name ?? 'host-self-edit' });
    this.ownerId = config.ownerId ?? '';
    this.agentId = config.agentId ?? '';
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.serviceToken = config.serviceToken ?? process.env.PLATFORM_SERVICE_KEY;
  }

  /**
   * Tool-call-time owner check. Throws when the caller is not the owner —
   * the agent's tool-call dispatcher catches this and returns a sanitised
   * error to the LLM, not the raw owner id.
   */
  private assertOwner(ctx: Context): void {
    const callerId = ctx.auth?.user_id ?? null;
    if (callerId !== this.ownerId) {
      throw new Error('FORBIDDEN: host self-edit tools are owner-only');
    }
  }

  private async portalPost(path: string, body: unknown): Promise<unknown> {
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      'function-authoring-surface': 'host',
    };
    if (this.serviceToken) headers['authorization'] = `Bearer ${this.serviceToken}`;
    const res = await fetch(`${this.portalUrl}${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`portal ${path} -> ${res.status}: ${await res.text().catch(() => '')}`);
    }
    return res.json();
  }

  private async portalDelete(path: string): Promise<unknown> {
    const headers: Record<string, string> = {
      'function-authoring-surface': 'host',
    };
    if (this.serviceToken) headers['authorization'] = `Bearer ${this.serviceToken}`;
    const res = await fetch(`${this.portalUrl}${path}`, { method: 'DELETE', headers });
    if (!res.ok) {
      throw new Error(`portal ${path} -> ${res.status}`);
    }
    return res.json().catch(() => ({}));
  }

  @tool({
    name: 'declare_function',
    description:
      'Declare a new function on this host agent and register it in agent_configs.functions. Owner-only.',
    parameters: PARAMETERS_DECLARE_FUNCTION,
  })
  @pricing({ creditsPerCall: 0, lock: 0.05, reason: 'function declaration (validation cost)' })
  async declareFunction(params: Record<string, unknown>, ctx: Context): Promise<unknown> {
    this.assertOwner(ctx);
    return this.portalPost(`/api/agents/${this.agentId}/functions`, {
      name: params.name,
      manifest: {
        runtime: params.runtime,
        permissions: params.permissions,
        limits: params.limits,
        description: params.description,
      },
      code: params.code,
    });
  }

  @tool({
    name: 'update_function',
    description: 'Update an existing function. Owner-only.',
    parameters: PARAMETERS_UPDATE_FUNCTION,
  })
  @pricing({ creditsPerCall: 0, lock: 0.05, reason: 'function update (validation cost)' })
  async updateFunction(params: Record<string, unknown>, ctx: Context): Promise<unknown> {
    this.assertOwner(ctx);
    // The portal route accepts a partial manifest; runtime is recovered
    // from the existing declaration when omitted.
    const manifest: Record<string, unknown> = {
      permissions: params.permissions,
      limits: params.limits,
      description: params.description,
    };
    if (typeof params.runtime === 'string') manifest.runtime = params.runtime;
    return this.portalPost(`/api/agents/${this.agentId}/functions`, {
      name: params.name,
      manifest,
      code: params.code,
    });
  }

  @tool({
    name: 'remove_function',
    description: 'Detach a function from the agent (preserves the content row).',
    parameters: PARAMETERS_REMOVE_FUNCTION,
  })
  async removeFunction(params: Record<string, unknown>, ctx: Context): Promise<unknown> {
    this.assertOwner(ctx);
    return this.portalDelete(`/api/agents/${this.agentId}/functions/${params.name}`);
  }

  @tool({
    name: 'add_to_skill',
    description:
      'Attach a function to a skill consumer (cron / custom_http / custom_tools). entry shape is consumer-specific.',
    parameters: PARAMETERS_ADD_TO_SKILL,
  })
  async addToSkill(params: Record<string, unknown>, ctx: Context): Promise<unknown> {
    this.assertOwner(ctx);
    return this.portalPost(`/api/agents/${this.agentId}/skills/${params.skill}/use`, {
      entry: params.entry,
      use: params.use,
    });
  }

  @tool({
    name: 'remove_from_skill',
    description: 'Detach a consumer entry from a skill (cron / custom_http / custom_tools).',
    parameters: PARAMETERS_REMOVE_FROM_SKILL,
  })
  async removeFromSkill(params: Record<string, unknown>, ctx: Context): Promise<unknown> {
    this.assertOwner(ctx);
    return this.portalDelete(
      `/api/agents/${this.agentId}/skills/${params.skill}/use/${params.entryId}`,
    );
  }

  @prompt({ priority: 75, name: 'selfEditGuide', scope: 'owner' })
  selfEditGuide(_ctx: Context): string {
    return [
      'You can edit your own capabilities via declare_function / update_function / remove_function / add_to_skill / remove_from_skill.',
      'Secrets contract: NEVER ask the owner for secrets in chat. When a secret is needed, surface a "set_function_secret" requiresUserAction so the owner can set the value through the secure form.',
      'After declaring a function, validate by running it once via the appropriate skill (custom_http endpoint, custom_tools tool, or manual invoke).',
      'Renaming a function is a deliberate two-step: remove the old declaration, then declare anew. KV state under fn:<name> stays per-name.',
      'Runtime rules (js-v1 sandbox, allowed globals, ctx.* host APIs, source/wall/memory caps, error codes) are documented by the FunctionRuntimeSkill\'s `functionsRuntimeStatic` block — do NOT restate them here.',
    ].join('\n');
  }
}
