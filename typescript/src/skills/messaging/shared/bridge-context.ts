/**
 * Bridge-context helpers shared by every messaging skill.
 *
 * Reads `ctx.metadata.bridge` (a {@link BridgeContext}) supplied by the
 * host runtime. When the inbound came from this skill's provider AND we
 * have a contactExternalId, tools can default `recipient` to that value
 * (so an LLM that calls `telegram_send_text` without an explicit chat_id
 * still routes to the bridged contact).
 */
import type { BridgeContext, Context } from '../../../core/types';

export function readBridgeContext(ctx: Context | undefined): BridgeContext | undefined {
  const bridge = (ctx?.metadata as Record<string, unknown> | undefined)?.bridge;
  if (!bridge || typeof bridge !== 'object') return undefined;
  const b = bridge as BridgeContext;
  if (!b.source || !b.contactExternalId) return undefined;
  return b;
}

export function bridgeMatches(ctx: Context | undefined, provider: string): BridgeContext | undefined {
  const b = readBridgeContext(ctx);
  return b && b.source === provider ? b : undefined;
}

/**
 * Build a short instruction that contextualises a system prompt for a
 * bridged conversation. Skills inject this through @prompt so the agent
 * knows it's talking through a third-party platform and that it MUST use
 * the platform send tool to actually deliver replies.
 */
export function buildBridgeAwarenessPrompt(args: {
  provider: string;
  contactDisplayName?: string | null;
  sendToolName: string;
}): string {
  const who = args.contactDisplayName ? `${args.contactDisplayName} (${args.provider})` : args.provider;
  return [
    `You are talking to ${who} through the ${args.provider} bridge.`,
    `The user does NOT see your plain assistant messages — you must call`,
    `\`${args.sendToolName}\` to deliver a reply. The recipient defaults to the`,
    `bridged contact, so you can usually omit the recipient argument.`,
  ].join(' ');
}
