/**
 * The session skill (2026-09-25): an agent keeps its conversations.
 *
 * ONE MEANING IN BOTH SDKS. In an agent file:
 *
 *     skills:
 *       - session                        # conversations kept on this machine
 *       - session: {backend: robutler}   # ... and yours on Robutler too, as chats
 *
 *   * IN THE CHAT (and `-p`) the chat keeps the conversation itself, on this
 *     machine (`cli/sessions.ts`), and with `backend: robutler` also as your
 *     chat with the agent on Robutler (`cli/robutler-sessions.ts`). This
 *     skill is never loaded there: it would be a second writer.
 *   * SERVED (`serve`, the daemon), this skill keeps each VERIFIED caller's
 *     conversation, when the request names it with `metadata.session_id`:
 *     the owner's where the chat keeps theirs, so `/resume` finds them;
 *     anyone else's under `callers/<hash>/`, one namespace per caller.
 *     Anonymous callers are not kept, and a session id is only ever looked
 *     up inside its caller's namespace, so naming someone else's id finds
 *     nothing of theirs. A request carries the whole conversation, as an
 *     OpenAI-style client sends it, and what is kept is that conversation
 *     (the person's and the agent's words) and the reply
 *     (`conversationToKeep`). `backend: robutler` changes nothing here:
 *     conversations that come through Robutler are chats there already, and
 *     a served agent does not write chats in anyone's name.
 *
 * WHAT THIS REPLACED. A per-chat key-value scratchpad for the model
 * (`session_get`, `session_set`, ...), keyed on a chat id the CALLER
 * supplied, so any caller could read and change another's entries (S-262),
 * and no agent file could name it. The Python skill of the same name was a
 * transcript recorder whose HTTP routes answered anyone (S-249). Both are
 * this now: `python/webagents/agents/skills/local/session/skill.py`, pinned
 * with this one by `python/tests/fixtures/sessions/sessions.json`.
 */

import { Skill } from '../../core/skill';
import { hook } from '../../core/decorators';
import type { AuthInfo, Context, HookData } from '../../core/types';
import { tierOf, userPrincipals } from '../access/skill';

export type SessionBackend = 'local' | 'robutler';

export interface SessionConfig {
  name?: string;
  enabled?: boolean;
  /** `local` (the default) or `robutler` (file comment). */
  backend?: string;
  /** The agent's folder, where its conversations belong (the resolver passes it). Default: the working directory. */
  agentDir?: string;
  /** The name its conversations are kept under. Default: the agent's own name. */
  agentName?: string;
}

/** The backend an entry names; throws the sentence a person can act on. */
export function sessionBackendOf(value: unknown): SessionBackend {
  if (value === undefined || value === null || value === 'local') return 'local';
  if (value === 'robutler') return 'robutler';
  throw new Error(`session: backend must be "local" or "robutler", not ${JSON.stringify(value)}.`);
}

const SESSION_ID_RE = /^[A-Za-z0-9._-]{1,128}$/;

/** The session a request names (`metadata.session_id`), when it is one that can name a file. */
export function requestSessionId(metadata: unknown): string | undefined {
  const id = (metadata as { session_id?: unknown } | undefined)?.session_id;
  return typeof id === 'string' && SESSION_ID_RE.test(id) && id !== '.' && id !== '..' ? id : undefined;
}

/**
 * Whose conversation a caller's is: `owner` for the agent's owner, else the
 * first identity something verified (`user:`, `agent:`, `key:`), else null:
 * an anonymous caller's is not kept.
 */
export function conversationOwner(auth: Partial<AuthInfo> | undefined): 'owner' | string | null {
  if (!auth || auth.authenticated === false) return null;
  if (tierOf(auth) === 'owner') return 'owner';
  const listed = (auth as { principals?: unknown }).principals;
  const verified = Array.isArray(listed)
    ? listed.filter((p): p is string => typeof p === 'string' && /^(user|agent|key):./.test(p))
    : userPrincipals(auth);
  return verified[0] ?? null;
}

/** What is kept of a turn: the request's own words (the person's and the agent's, text only) and the reply. */
export function conversationToKeep(
  request: readonly { role?: unknown; content?: unknown }[],
  reply: unknown,
): { role: 'user' | 'assistant'; content: string }[] {
  const words = request
    .filter(
      (m): m is { role: 'user' | 'assistant'; content: string } =>
        (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim() !== '',
    )
    .map((m) => ({ role: m.role, content: m.content }));
  if (typeof reply === 'string' && reply.trim() !== '') words.push({ role: 'assistant', content: reply });
  return words;
}

export class SessionSkill extends Skill {
  readonly backend: SessionBackend;
  private agentDir: string | undefined;
  private agentName: string | undefined;
  /** Set by `BaseAgent.addSkill` (it checks for `setAgent`): whose name the conversations are kept under. */
  private _agent?: unknown;

  constructor(config: SessionConfig = {}) {
    super({ ...config, name: config.name || 'session' });
    this.backend = sessionBackendOf(config.backend);
    this.agentDir = config.agentDir;
    this.agentName = config.agentName;
  }

  setAgent(agent: unknown): void {
    this._agent = agent;
  }

  /** After a served turn: keep the caller's conversation, when the request names it (file comment). */
  @hook({ lifecycle: 'after_run', priority: 90 })
  async keepConversation(data: HookData, context: Context): Promise<void> {
    const id = requestSessionId(context.metadata);
    if (!id) return;
    const whose = conversationOwner(context.auth as Partial<AuthInfo> | undefined);
    if (!whose) return;
    const messages = conversationToKeep((data.messages ?? []) as { role?: unknown; content?: unknown }[], data.response);
    if (!messages.length) return;
    const { callerSessionsDir, loadSession, saveSession, sessionsDir } = await import('../../cli/sessions.js');
    const folder = this.agentDir ?? process.cwd();
    const name = this.agentName ?? (this._agent as { name?: string } | undefined)?.name ?? 'agent';
    const dir = whose === 'owner' ? sessionsDir(folder, name) : callerSessionsDir(folder, name, whose);
    const kept = loadSession(dir, id);
    saveSession(dir, {
      session_id: id,
      agent_name: name,
      created_at: kept?.created_at ?? '',
      updated_at: '',
      messages,
      metadata: { ...(kept?.metadata ?? {}), sdk: 'typescript' },
      input_tokens: kept?.input_tokens ?? 0,
      output_tokens: kept?.output_tokens ?? 0,
    });
  }
}
