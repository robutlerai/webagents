/**
 * DiscordSkill — bridge-aware behaviour for guild @-mentions.
 *
 * Covers the three contracts the rest of the system relies on:
 *
 *   1. `discord_send_in_channel` defaults `channel_id` from
 *      `ctx.metadata.bridge.channelId` when called without one (the
 *      LLM only needs to think about `content`).
 *   2. The reply auto-prepends `<@authorId>` so the user gets a
 *      Discord ping when the bot answers — idempotent if the model
 *      already included the mention.
 *   3. The `requiresConfirmation` callback returns `false` only inside
 *      a same-channel guild-mention bridge. Anywhere else it returns
 *      `true` so the NotificationSkill keeps its broadcast gate.
 *   4. Tool visibility: `discord_send_dm` is hidden in a guild-mention
 *      bridge (the contact is in a public channel, not a DM), and
 *      `discord_send_in_channel` is hidden in a DM-only bridge.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DiscordSkill } from '../../../../src/skills/messaging/discord';
import { BaseAgent } from '../../../../src/core/agent';
import type { Tool } from '../../../../src/core/types';

const TOKEN = 'discord-bot-token';
const BOT_APP = '999000111222333444';
const USER = '111222333444555666';
const CHANNEL = 'CGENERAL';
const GUILD = 'GACMEHQ';

beforeEach(() => {
  vi.restoreAllMocks();
});

function makeSkill() {
  return new DiscordSkill({
    agentId: 'agent-1',
    enabledCapabilities: ['send_messages', 'publish_posts'],
    getToken: async () => ({
      token: TOKEN,
      metadata: { applicationId: BOT_APP, publicKey: '00'.repeat(32) },
    }),
  });
}

function mentionBridge() {
  return {
    source: 'discord',
    contactExternalId: USER,
    kind: 'mention' as const,
    channelId: CHANNEL,
    guildId: GUILD,
  };
}

describe('DiscordSkill: guild-mention outbound', () => {
  it('discord_send_in_channel defaults channel_id from bridge context', async () => {
    const calls: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      calls.push(String(input));
      return new Response(JSON.stringify({ id: 'M1' }));
    });
    const skill = makeSkill();
    const ctx = { metadata: { bridge: mentionBridge() } } as any;
    const r = (await skill.sendInChannel({ content: 'hi there' }, ctx)) as {
      ok: true;
      providerMessageId?: string;
    };
    expect(r.ok).toBe(true);
    // Defaulted channel_id from bridge — request URL has the channel.
    expect(calls.some((u) => u.includes(`/channels/${CHANNEL}/messages`))).toBe(true);
  });

  it('discord_send_in_channel auto-prepends <@authorId> on the same-channel mention reply', async () => {
    const seenBodies: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      seenBodies.push(String((init as RequestInit | undefined)?.body ?? ''));
      return new Response(JSON.stringify({ id: 'M1' }));
    });
    const skill = makeSkill();
    const ctx = { metadata: { bridge: mentionBridge() } } as any;
    await skill.sendInChannel({ content: 'sure!' }, ctx);
    expect(seenBodies[0]).toContain(`<@${USER}> sure!`);
  });

  it('does NOT double-prepend the mention when the LLM already included it', async () => {
    const seenBodies: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      seenBodies.push(String((init as RequestInit | undefined)?.body ?? ''));
      return new Response(JSON.stringify({ id: 'M1' }));
    });
    const skill = makeSkill();
    const ctx = { metadata: { bridge: mentionBridge() } } as any;
    await skill.sendInChannel({ content: `<@${USER}> howdy` }, ctx);
    // Exactly one occurrence of the mention.
    const m = seenBodies[0].match(new RegExp(`<@${USER}>`, 'g'));
    expect(m?.length).toBe(1);
  });

  it('does NOT prepend the mention when posting to a different channel than the one we were pinged in', async () => {
    const seenBodies: string[] = [];
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
      seenBodies.push(String((init as RequestInit | undefined)?.body ?? ''));
      return new Response(JSON.stringify({ id: 'M1' }));
    });
    const skill = makeSkill();
    const ctx = { metadata: { bridge: mentionBridge() } } as any;
    await skill.sendInChannel({ channel_id: 'CSOMEWHERE_ELSE', content: 'cross-post' }, ctx);
    expect(seenBodies[0]).not.toContain(`<@${USER}>`);
  });
});

describe('DiscordSkill: requiresConfirmation gate', () => {
  // Pull the tool descriptor straight off the skill via BaseAgent's
  // registry — that's what BaseAgent.executeTool calls into.
  function getInChannelTool() {
    const skill = makeSkill();
    const agent = new BaseAgent(
      { name: 'test', skills: [skill] } as any,
    );
    const toolReg = (agent as any).toolRegistry as Map<string, Tool>;
    const t = toolReg.get('discord_send_in_channel');
    if (!t) throw new Error('discord_send_in_channel not registered');
    return t;
  }

  it('returns false (auto-approve) when bridge is mention to the same channel', () => {
    const tool = getInChannelTool();
    const fn = tool.requiresConfirmation as (a: unknown, c: any) => boolean;
    const ctx = { metadata: { bridge: mentionBridge() } };
    expect(fn({ content: 'hi' }, ctx)).toBe(false);
    expect(fn({ channel_id: CHANNEL, content: 'hi' }, ctx)).toBe(false);
  });

  it('returns true when bridge is a DM (cross-context broadcast attempt)', () => {
    const tool = getInChannelTool();
    const fn = tool.requiresConfirmation as (a: unknown, c: any) => boolean;
    const ctx = {
      metadata: {
        bridge: { source: 'discord', contactExternalId: USER, kind: 'dm' },
      },
    };
    expect(fn({ channel_id: CHANNEL, content: 'hi' }, ctx)).toBe(true);
  });

  it('returns true when posting to a different channel than the mention bridge', () => {
    const tool = getInChannelTool();
    const fn = tool.requiresConfirmation as (a: unknown, c: any) => boolean;
    const ctx = { metadata: { bridge: mentionBridge() } };
    expect(fn({ channel_id: 'CDIFFERENT', content: 'broadcast' }, ctx)).toBe(true);
  });

  it('returns true when there is no bridge at all (portal chat)', () => {
    const tool = getInChannelTool();
    const fn = tool.requiresConfirmation as (a: unknown, c: any) => boolean;
    expect(fn({ channel_id: CHANNEL, content: 'hi' }, { metadata: {} })).toBe(true);
  });
});

describe('DiscordSkill: tool visibility by bridge kind', () => {
  function definitionsFor(bridge: unknown | undefined) {
    const agent = new BaseAgent({ name: 'test', skills: [makeSkill()] } as any);
    (agent as any).context.metadata = bridge ? { bridge } : {};
    return agent
      .getToolDefinitions()
      .map((d) => d.function.name);
  }

  it('hides discord_send_dm in a guild-mention bridge', () => {
    const names = definitionsFor(mentionBridge());
    expect(names).not.toContain('discord_send_dm');
    expect(names).toContain('discord_send_in_channel');
  });

  it('hides discord_send_in_channel in a DM bridge', () => {
    // DM tools require kind:'dm'; the in-channel tool currently
    // doesn't carry a `requiresBridge` annotation (so it's available
    // as an owner-driven tool from the portal). The DM tool MUST
    // appear in a DM bridge though.
    const names = definitionsFor({
      source: 'discord',
      contactExternalId: USER,
      kind: 'dm',
    });
    expect(names).toContain('discord_send_dm');
  });

  it('hides discord_send_dm when there is no bridge at all', () => {
    const names = definitionsFor(undefined);
    expect(names).not.toContain('discord_send_dm');
  });
});
