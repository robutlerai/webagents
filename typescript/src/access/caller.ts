/**
 * Who a turn is for, as the scope checks read it (ADR-0045).
 *
 * `LOCAL_OWNER` is the person at this terminal, or the daemon running a
 * schedule the owner wrote: the OWNER of the agent (2026-09-25). Every local
 * turn once ran anonymous, so an owner-scoped tool or prompt (the REST tool is
 * one) never reached the one person the local chat serves, while an HTTP
 * caller of the same agent file is whoever it proves to be. The chat
 * (`cli/app.ts`) and the schedule runner (`daemon/schedule-runner.ts`) pass
 * it as `RunOptions.auth`; the Python SDK's twin is `access/caller.py`.
 */

/** The agent's owner, for a turn that starts on this machine. */
export const LOCAL_OWNER: Readonly<Record<string, unknown>> = Object.freeze({
  authenticated: true,
  scope: 'owner',
  provider: 'local',
});

/**
 * The channel a relayed sender wrote from, as the platform asserts it
 * (`caller.channel = {type, sender_id}` on the relayed frame; the channel
 * relay, plan item 2.2). The type is a slug, lower-cased here; the sender id
 * is one token with no whitespace, never the wildcard. Anything else is not
 * a channel and reads as no channel, so the caller stays a plain user. The
 * vocabulary is pinned by `python/tests/fixtures/access/channel_caller.json`.
 */
export interface ChannelIdentity {
  type: string;
  sender_id: string;
}

const CHANNEL_TYPE = /^[a-z][a-z0-9-]{0,31}$/;

export function channelIdentityOf(raw: unknown): ChannelIdentity | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const { type, sender_id } = raw as { type?: unknown; sender_id?: unknown };
  if (typeof type !== 'string' || typeof sender_id !== 'string') return null;
  const lowered = type.toLowerCase();
  if (!CHANNEL_TYPE.test(lowered)) return null;
  if (!sender_id || sender_id.length > 200 || sender_id === '*' || /\s/.test(sender_id)) return null;
  return { type: lowered, sender_id };
}
