# Messaging skills

Reusable, **database-agnostic** integration skills for messaging and social
platforms. Each skill is a class that extends `MessagingSkill`, declares its
tools with `@tool`, contributes context-sensitive prompts with `@prompt`,
and exposes per-agent webhook callbacks with `@http`.

## Why this exists

Two host shapes need to drive the same skills:

1. **Robutler portal** — multi-tenant SaaS that resolves credentials from
   `connected_accounts`, applies kill-switches and spend caps, persists
   refreshed tokens back to the DB, and runs an approval gate for
   publish-style tools.
2. **Standalone webagents host** — `npx @webagents/cli`, OSS daemon, custom
   runtimes — that wants to plug a single bot in via env vars and does
   not care about a DB.

The skills accept a `MessagingSkillOptions` bag so hosts inject what they
need (token resolver, token writer, API-call wrapper, approval gate,
public base URL). When nothing is supplied the skill falls back to
`process.env` and a no-op writer so `tsx examples/telegram-echo.ts` just
works.

## Provider matrix

| Provider    | Skill class       | Capabilities                                                           | Webhook `@http` endpoints                |
| ----------- | ----------------- | ---------------------------------------------------------------------- | ---------------------------------------- |
| Telegram    | `TelegramSkill`   | `send_messages`, `inbound_messages`                                    | `POST /messaging/telegram/set-webhook`   |
| Twilio      | `TwilioSkill`     | `send_messages`                                                        | `POST /messaging/twilio/status`          |
| Slack       | `SlackSkill`      | `send_messages`, `inbound_messages`                                    | `POST /messaging/slack/interactivity`    |
| Discord     | `DiscordSkill`    | `send_messages`, `inbound_messages`                                    | `POST /messaging/discord/interactions`   |
| WhatsApp    | `WhatsAppSkill`   | `send_messages`, `manage_templates`                                    | _(host-level webhook)_                   |
| Messenger   | `MessengerSkill`  | `send_messages`, `read_page_posts`, `publish_page_posts`               | _(host-level webhook)_                   |
| Instagram   | `InstagramSkill`  | `send_messages`, `publish_posts`, `manage_comments`                    | _(host-level webhook)_                   |
| LinkedIn    | `LinkedInSkill`   | `publish_posts`                                                        | _(none)_                                 |
| Bluesky     | `BlueskySkill`    | `publish_posts`                                                        | _(none)_                                 |
| Reddit      | `RedditSkill`     | `publish_posts`                                                        | _(none)_                                 |

The Meta family (WhatsApp / Messenger / Instagram) shares one app-level
webhook handler; per-agent endpoints aren't required because Meta does
not deliver per-bot URLs. Telegram, Twilio, Slack, and Discord deliver
to per-skill `@http` endpoints mounted by the host's catch-all
dispatcher.

## Constructor options

All skills accept the same `MessagingSkillOptions`:

```ts
interface MessagingSkillOptions {
  agentId?: string;
  integrationId?: string;
  enabledCapabilities?: string[];     // omit to enable all advertised caps
  getToken?: TokenResolver['getToken'];     // host credential lookup
  setToken?: TokenWriter['setToken'];       // host token persistence
  wrapApiCall?: ApiCallWrapper['wrapApiCall']; // kill-switch + metrics
  requirePostApproval?: boolean;            // gate publish-style tools
  requestApproval?: PostApprovalGate['requestApproval']; // gate callback
  httpEndpointBaseUrl?: string;             // overrides BASE_URL/PORTAL_URL
}
```

When `getToken` is omitted the skill falls back to env vars per
`shared/env-resolver.ts`. When `setToken` is omitted the skill logs a
warning if a callback tries to persist a refreshed token (no-op writer).

## Bridge awareness

Every skill registers a `*-bridge-awareness` `@prompt` that fires only
when `ctx.metadata.bridge.source === skill.provider`. The prompt
instructs the model to reply via the platform send tool (the skill's
`SEND_TOOL` constant) instead of letting an assistant message stay
inside the portal. Hosts that mirror agent replies into chat history
(legacy bridge-injection path) can disable the prompt by clearing
`ctx.metadata.bridge`.

## See also

- `shared/options.ts` — full type definitions.
- `shared/messaging-skill-base.ts` — capability gating, token resolution,
  apiCall wrapping, bridge-context lookup.
- `docs/integrations/<provider>.md` (in the portal repo) for the
  end-to-end OAuth / webhook setup walkthrough.
- `docs/architecture/messaging-bridge.md` — full bridge architecture.
