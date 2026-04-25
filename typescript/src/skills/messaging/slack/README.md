# SlackSkill

Slack DM / channel posting + interactivity callback handler.

## Tools

| Tool                     | Capability      | Notes                                          |
| ------------------------ | --------------- | ---------------------------------------------- |
| `slack_send_dm`          | `send_messages` | Opens an IM channel and posts; required reply  |
|                          |                 | path for bridged Slack contacts.               |
| `slack_post_in_channel`  | `send_messages` | Subject to host approval gate.                 |
| `slack_reply_in_thread`  | `send_messages` | `chat.postMessage` with `thread_ts`.           |

`requirePostApproval=true` plus a configured `requestApproval` callback
routes `slack_post_in_channel` through the host's pending-draft store.

## Webhook endpoints

| Method | Path                               | Auth        | Notes                                                              |
| ------ | ---------------------------------- | ----------- | ------------------------------------------------------------------ |
| POST   | `/messaging/slack/interactivity`   | `signature` | Validates `v0:`-prefixed `X-Slack-Signature` per Slack docs.       |

The `signing secret` MUST be available via env (`SLACK_SIGNING_SECRET`)
or via metadata to verify signatures.

## Token shape

```jsonc
ResolvedToken.token = "xoxb-..."
ResolvedToken.metadata = { "signingSecret": "<v0 signing secret>" }
```

## Env fallback

`SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`.
