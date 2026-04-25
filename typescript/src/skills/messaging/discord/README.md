# DiscordSkill

Discord bot DM/channel/interaction tooling + interactions callback
handler with Ed25519 signature verification.

## Tools

| Tool                                | Capability      | Notes                                       |
| ----------------------------------- | --------------- | ------------------------------------------- |
| `discord_send_dm`                   | `send_messages` | Opens DM channel via `POST /users/@me/channels`. |
| `discord_send_in_channel`           | `send_messages` | Subject to approval gate.                   |
| `discord_register_slash_command`    | `send_messages` | Guild-scoped registration.                  |

## Webhook endpoints

| Method | Path                                  | Auth        | Notes                                                         |
| ------ | ------------------------------------- | ----------- | ------------------------------------------------------------- |
| POST   | `/messaging/discord/interactions`     | `signature` | Verifies `X-Signature-Ed25519` + `X-Signature-Timestamp`.     |

Discord requires verifying interactions endpoints before activating the
app — the host MUST respond to PING (type=1) with `{ type: 1 }`. The
skill does this automatically when the signature passes.

## Token shape

```jsonc
ResolvedToken.token = "<DISCORD_BOT_TOKEN>"
ResolvedToken.metadata = { "publicKey": "<application public key>" }
```

`publicKey` falls back to `DISCORD_PUBLIC_KEY`.

## Env fallback

`DISCORD_BOT_TOKEN`, `DISCORD_PUBLIC_KEY`, `DISCORD_APPLICATION_ID`.
