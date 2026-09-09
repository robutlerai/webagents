/**
 * `SecretsSkill`
 *
 * Named credentials in the operating system's own keystore (macOS Keychain,
 * Linux Secret Service, Windows Credential Manager), with a fallback that
 * refuses to be quiet about being a fallback. The backends live in
 * `./store`; this file is the agent-facing wrapper.
 *
 * THE ONE DESIGN DECISION WORTH ARGUING ABOUT: `secrets_get` does NOT return
 * the secret. It returns whether one exists.
 *
 * The reason is that the primary consumer of a stored credential is CODE, not
 * a model. `store.get(name)` in a fetch call is the normal path; the model
 * needs to know a credential is present so it can stop asking for one, and
 * almost never needs the bytes. Handing a bearer to the model instead puts it
 * in the transcript, in whatever the transcript is persisted to, and in the
 * inference provider's records, which are three copies of a credential in
 * places nobody will think to rotate. `reveal: true` exists for the genuine
 * exception and is refused unless the developer set `allowReveal` at
 * construction, so revealing is a decision made in code rather than one the
 * model can make for itself.
 *
 * That matters more here than it would for an API key, because the credential
 * this skill was built for is the platform bearer from `registerWithPlatform`:
 * seven days, `agents:own`, and no `jti`, so nothing can revoke it (portal
 * security log S-037). Rotating the key on the agent card does not touch it.
 *
 * Every tool is `owner` scope. A secret store readable by a counterparty
 * agent is not a secret store.
 */

import { Skill } from '../../core/skill';
import { tool, prompt } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';
import {
  openSecretStore,
  type SecretBackendStatus,
  type SecretStore,
  type SecretStoreOptions,
} from './store';

export interface SecretsSkillConfig extends SkillConfig, SecretStoreOptions {
  /**
   * Permit `secrets_get({ reveal: true })` to return the value. Defaults to
   * `false`. See the file header: this is off because a revealed secret is a
   * secret copied into a transcript.
   */
  allowReveal?: boolean;
}

/** What every tool here answers with, so the backend is never implicit. */
interface BackendEnvelope {
  /** `keystore` or `file`. */
  backend: SecretBackendStatus['backend'];
  /** True only when the OS keystore holds this. */
  keystore: boolean;
  /**
   * Present whenever `keystore` is false. This is the copy the MODEL reads,
   * which is the point: process logs are easy to miss and a tool result is
   * not.
   */
  warning?: string;
}

export class SecretsSkill extends Skill {
  private readonly options: SecretsSkillConfig;
  private readonly allowReveal: boolean;
  private store: SecretStore | null = null;

  constructor(config: SecretsSkillConfig = {}) {
    super({ ...config, name: config.name || 'secrets' });
    this.options = config;
    this.allowReveal = config.allowReveal ?? false;
  }

  /**
   * Open the store eagerly so the fallback warning is printed at boot rather
   * than at the first tool call, which may be hours later or never.
   */
  async initialize(): Promise<void> {
    await this.getStore();
  }

  /**
   * The store this skill wraps, for code that wants the credential rather
   * than a tool call. This is the handle to pass to `registerWithPlatform`.
   */
  async getStore(): Promise<SecretStore> {
    if (!this.store) {
      this.store = await openSecretStore(this.options);
    }
    return this.store;
  }

  private async envelope(): Promise<BackendEnvelope> {
    const status = (await this.getStore()).status();
    return status.keystore
      ? { backend: status.backend, keystore: true }
      : { backend: status.backend, keystore: false, warning: status.warning };
  }

  @prompt({ priority: 45, name: 'secretsGuide', scope: 'owner' })
  secretsGuide(_ctx: Context): string {
    return [
      '## Secrets skill',
      '',
      'Named credentials in the operating system keystore, or in an owner-only file when the machine has no keystore.',
      '',
      '### Rules',
      '- `secrets_get` tells you whether a secret EXISTS. It does not give you the value, and that is deliberate: the code that uses a credential reads it directly, and a value returned here would be copied into this conversation.',
      '- Never paste a credential into a normal reply, a file, or a commit. `secrets_set` is the only place one belongs.',
      '- Read `backend` on every result. When it is `file` the secret is stored as plaintext on disk and the `warning` field says where. Tell the owner rather than treating it as normal.',
      '- Deleting is cheap and reversible only by the owner re-entering the credential. Confirm before `secrets_delete`.',
      '',
      '### Names',
      'Use 1 to 128 characters from A-Z a-z 0-9 . _ and -. Names are scoped to this agent, so `platform_token` here does not collide with another agent\'s `platform_token` on the same machine.',
    ].join('\n');
  }

  @tool({
    name: 'secrets_set',
    audience: 'owner',
    description:
      'Store a named secret in the OS keystore (or an owner-only file if the machine has no keystore). Returns which backend it landed in. Prefer having the owner enter the value here over keeping it in a file or an environment variable.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        name: {
          type: 'string',
          description: 'Secret name, 1-128 characters from A-Z a-z 0-9 . _ and -',
        },
        value: { type: 'string', description: 'The secret value. Never echo this back.' },
      },
      required: ['name', 'value'],
    },
  })
  async secretsSet(
    params: { name: string; value: string },
    _context: Context,
  ): Promise<BackendEnvelope & { name: string; stored: true }> {
    const store = await this.getStore();
    await store.set(params.name, params.value);
    await store.noteIndex(params.name, true);
    return { name: params.name, stored: true, ...(await this.envelope()) };
  }

  @tool({
    name: 'secrets_get',
    audience: 'owner',
    description:
      'Report whether a named secret exists. Does NOT return the value: code reads the credential directly, and returning it here would copy it into this conversation.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        name: { type: 'string', description: 'Secret name' },
        reveal: {
          type: 'boolean',
          description:
            'Return the value as well. Refused unless the developer enabled it when constructing the skill.',
        },
      },
      required: ['name'],
    },
  })
  async secretsGet(
    params: { name: string; reveal?: boolean },
    _context: Context,
  ): Promise<BackendEnvelope & { name: string; exists: boolean; value?: string; note?: string }> {
    const store = await this.getStore();
    const value = await store.get(params.name);
    const envelope = await this.envelope();
    const base = { name: params.name, exists: value !== null, ...envelope };
    if (!params.reveal) return base;
    if (!this.allowReveal) {
      return {
        ...base,
        note:
          'reveal refused: this skill was constructed without allowReveal. The value is ' +
          'readable from code via the skill store, which keeps it out of this conversation.',
      };
    }
    return value === null ? base : { ...base, value };
  }

  @tool({
    name: 'secrets_list',
    audience: 'owner',
    description:
      'List the names of stored secrets. Never returns values. On a keystore backend the list is what this agent has written, which may be short of what the keystore actually holds.',
    parameters: { type: 'object', additionalProperties: false, properties: {} },
  })
  async secretsList(
    _params: Record<string, never>,
    _context: Context,
  ): Promise<BackendEnvelope & { names: string[]; complete: boolean; note?: string }> {
    const store = await this.getStore();
    const { names, complete } = await store.list();
    return {
      names,
      complete,
      ...(complete
        ? {}
        : {
            note:
              'the OS keystore cannot be enumerated portably, so this lists only the names ' +
              'this agent recorded. A secret set by something else is still readable by name.',
          }),
      ...(await this.envelope()),
    };
  }

  @tool({
    name: 'secrets_delete',
    audience: 'owner',
    description:
      'Remove a named secret from every backend, including a plaintext copy left behind from before a keystore was available. Returns whether anything was removed.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: { name: { type: 'string', description: 'Secret name' } },
      required: ['name'],
    },
  })
  async secretsDelete(
    params: { name: string },
    _context: Context,
  ): Promise<BackendEnvelope & { name: string; removed: boolean }> {
    const store = await this.getStore();
    const removed = await store.delete(params.name);
    await store.noteIndex(params.name, false);
    return { name: params.name, removed, ...(await this.envelope()) };
  }

  @tool({
    name: 'secrets_status',
    audience: 'owner',
    description:
      'Report where secrets are being stored and why. Use this before telling the owner their credential is safe.',
    parameters: { type: 'object', additionalProperties: false, properties: {} },
  })
  async secretsStatus(
    _params: Record<string, never>,
    _context: Context,
  ): Promise<SecretBackendStatus> {
    return (await this.getStore()).status();
  }
}
