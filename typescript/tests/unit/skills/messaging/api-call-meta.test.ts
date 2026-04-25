/**
 * Contract test: every messaging skill flowing through `wrapApiCall` MUST
 * supply a structured `ApiCallMeta` object (provider + type at minimum)
 * and return the discriminated `ApiCallResult` envelope unchanged.
 *
 * This guards against accidental drift back to the throw-on-failure shim
 * the SDK used to ship before phase 1b.
 */
import { describe, it, expect, vi } from 'vitest';
import {
  TelegramSkill,
  TwilioSkill,
  SlackSkill,
  DiscordSkill,
  WhatsAppSkill,
  MessengerSkill,
  InstagramSkill,
  LinkedInSkill,
  RedditSkill,
  TikTokSkill,
  SendGridSkill,
  XSkill,
  GoogleChatSkill,
  type ApiCallMeta,
  type MessagingSkillOptions,
} from '../../../../src/skills/messaging';

interface SkillCase {
  name: string;
  ctor: new (opts: MessagingSkillOptions) => unknown;
  enabledCapabilities: string[];
  metadata?: Record<string, unknown>;
  /** Function that triggers a wrapApiCall on the skill. */
  invoke: (skill: any) => Promise<unknown>;
  expectedProvider: string;
}

const cases: SkillCase[] = [
  {
    name: 'TelegramSkill.sendText',
    ctor: TelegramSkill,
    enabledCapabilities: ['send_messages'],
    invoke: (s) => s.sendText({ chat_id: 1, text: 'hi' }),
    expectedProvider: 'telegram',
  },
  {
    name: 'TwilioSkill.sendSms',
    ctor: TwilioSkill,
    enabledCapabilities: ['send_messages'],
    metadata: { accountSid: 'AC1', fromNumber: '+15551112222' },
    invoke: (s) => s.sendSms({ to: '+15553334444', body: 'hi' }),
    expectedProvider: 'twilio',
  },
  {
    name: 'SlackSkill.sendDm',
    ctor: SlackSkill,
    enabledCapabilities: ['send_messages'],
    invoke: (s) => s.sendDm({ user_id: 'U1', text: 'hi' }),
    expectedProvider: 'slack',
  },
  {
    name: 'DiscordSkill.sendDm',
    ctor: DiscordSkill,
    enabledCapabilities: ['send_messages'],
    invoke: (s) => s.sendDm({ user_id: '1', text: 'hi' }),
    expectedProvider: 'discord',
  },
  {
    name: 'WhatsAppSkill.sendText',
    ctor: WhatsAppSkill,
    enabledCapabilities: ['send_messages'],
    metadata: { phoneNumberId: '123' },
    invoke: (s) => s.sendText({ to: '+15553334444', text: 'hi' }),
    expectedProvider: 'whatsapp',
  },
  {
    name: 'MessengerSkill.sendText',
    ctor: MessengerSkill,
    enabledCapabilities: ['send_messages'],
    metadata: { pageId: 'p1', pageAccessToken: 'tok' },
    invoke: (s) => s.sendText({ recipient_psid: 'r1', text: 'hi' }),
    expectedProvider: 'messenger',
  },
  {
    name: 'InstagramSkill.sendText',
    ctor: InstagramSkill,
    enabledCapabilities: ['send_messages'],
    metadata: { igUserId: 'ig1' },
    invoke: (s) => s.sendDm({ recipient_id: 'r1', text: 'hi' }),
    expectedProvider: 'instagram',
  },
  {
    name: 'LinkedInSkill.post',
    ctor: LinkedInSkill,
    enabledCapabilities: ['publish_posts'],
    metadata: { providerHandle: 'urn:li:person:abc' },
    invoke: (s) => s.post({ text: 'hi' }),
    expectedProvider: 'linkedin',
  },
  {
    name: 'RedditSkill.post',
    ctor: RedditSkill,
    enabledCapabilities: ['publish_posts'],
    metadata: { userAgent: 'test/1.0' },
    invoke: (s) => s.post({ subreddit: 'x', title: 'hi' }),
    expectedProvider: 'reddit',
  },
  {
    name: 'TikTokSkill.publishVideo',
    ctor: TikTokSkill,
    enabledCapabilities: ['publish_posts'],
    metadata: { openId: 'open-1' },
    invoke: (s) => s.publishVideo({ videoUrl: 'https://x/y.mp4' }),
    expectedProvider: 'tiktok',
  },
  {
    name: 'SendGridSkill.sendEmail',
    ctor: SendGridSkill,
    enabledCapabilities: ['send_messages'],
    metadata: { fromEmail: 'a@b.com' },
    invoke: (s) => s.sendEmail({ to: 'x@y.com', subject: 'hi', text: 'hi' }),
    expectedProvider: 'sendgrid',
  },
  {
    name: 'XSkill.sendDm',
    ctor: XSkill,
    enabledCapabilities: ['send_messages'],
    invoke: (s) => s.sendDm({ participant_id: '1', text: 'hi' }),
    expectedProvider: 'x',
  },
  {
    name: 'GoogleChatSkill.sendMessage',
    ctor: GoogleChatSkill,
    enabledCapabilities: ['send_messages'],
    invoke: (s) => s.sendMessage({ space: 'spaces/abc', text: 'hi' }),
    expectedProvider: 'google-chat',
  },
];

describe('messaging skills wrapApiCall contract', () => {
  for (const c of cases) {
    it(`${c.name} forwards a structured ApiCallMeta and returns ApiCallResult`, async () => {
      // Stub fetch so we don't actually hit any provider API.
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response('{}', { status: 200, headers: { 'x-message-id': 'm-1' } }),
      );

      const observed: ApiCallMeta[] = [];
      const skill = new c.ctor({
        agentId: 'a-1',
        integrationId: 'i-1',
        enabledCapabilities: c.enabledCapabilities,
        getToken: async () => ({ token: 't', metadata: c.metadata }),
        wrapApiCall: async (meta, fn) => {
          observed.push(meta);
          try {
            return { ok: true, data: await fn() };
          } catch (err) {
            return {
              ok: false,
              retriable: false,
              reason: 'provider_api_error',
              message: err instanceof Error ? err.message : String(err),
            };
          }
        },
      });

      const r = await c.invoke(skill);
      // Either ok:true (mock fetch worked) or ok:false (provider_api_error).
      expect(r).toBeTypeOf('object');
      expect(typeof (r as { ok?: boolean }).ok).toBe('boolean');
      // wrapApiCall MUST have been invoked with the right provider + a non-empty `type`.
      expect(observed.length).toBeGreaterThan(0);
      const first = observed[0]!;
      expect(first.provider).toBe(c.expectedProvider);
      expect(typeof first.type).toBe('string');
      expect(first.type.length).toBeGreaterThan(0);
      expect(first.agentId).toBe('a-1');
      expect(first.integrationId).toBe('i-1');
    });
  }
});
