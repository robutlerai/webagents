# TelegramSkill

Send messages and surface bridge-aware prompts for a Telegram bot.

## Tools

| Tool                     | Capability         | Notes                                              |
| ------------------------ | ------------------ | -------------------------------------------------- |
| `telegram_send_text`     | `send_messages`    | Required reply path for bridged Telegram contacts. |
| `telegram_send_photo`    | `send_messages`    | Image reply.                                       |
| `telegram_send_document` | `send_messages`    | File reply.                                        |
| `telegram_send_typing`   | `send_messages`    | Shows typing indicator (~5s).                      |

`telegram_send_text` accepts `chat_id`, `text`, optional `parse_mode`
(`HTML` or `MarkdownV2`), `reply_to_message_id`, and
`disable_web_page_preview`. When the call comes from a bridged Telegram
chat the skill auto-fills `chat_id` from `ctx.metadata.bridge.contactExternalId`.

## Webhook endpoints

| Method | Path                                | Auth      | Notes                                |
| ------ | ----------------------------------- | --------- | ------------------------------------ |
| POST   | `/messaging/telegram/set-webhook`   | `session` | Convenience to call `setWebhook`.    |

Hosts handle inbound updates at their own webhook URL (e.g. the portal
serves `/api/webhooks/telegram`).

## Standalone

```ts
import { TelegramSkill } from '@webagents/core/skills/messaging/telegram';
const skill = new TelegramSkill();
// Reads TELEGRAM_BOT_TOKEN from env via defaultEnvTokenResolver.
```

## Hosted (portal-style)

```ts
const skill = new TelegramSkill({
  agentId,
  integrationId,
  enabledCapabilities: ['send_messages', 'inbound_messages'],
  getToken: portalGetToken,
  wrapApiCall: portalWrapApiCall,
});
```
