# TwilioSkill

SMS / MMS sending via the Twilio Messages API. Bridge-aware so an agent
in a Twilio-bridged chat replies via `twilio_send_sms` instead of an
in-portal assistant message.

## Tools

| Tool             | Capability      | Notes                                                  |
| ---------------- | --------------- | ------------------------------------------------------ |
| `twilio_send_sms`| `send_messages` | Required reply path for bridged SMS contacts.          |
| `twilio_send_mms`| `send_messages` | Adds `MediaUrl` to the SMS body.                       |

A2P-10DLC compliance is the host's responsibility; Twilio rejects
`+1`-bound traffic without a registered campaign. The portal sets
`metadata.a2pCampaignRegistered=true` to gate sends; standalone hosts
should preflight outside the skill.

## Token shape

```jsonc
ResolvedToken.token = "<TWILIO_AUTH_TOKEN>"
ResolvedToken.metadata = {
  "subaccountSid": "ACxxxx",   // required
  "phoneNumber":  "+1...",     // optional default From
  "appSecret":     "<webhook validation token>" // optional
}
```

## Webhook endpoints

| Method | Path                            | Auth        | Notes                                 |
| ------ | ------------------------------- | ----------- | ------------------------------------- |
| POST   | `/messaging/twilio/status`      | `signature` | Validates `X-Twilio-Signature`.       |

Standalone hosts that want signature verification must wire
`TWILIO_AUTH_TOKEN` into env so the helper can reproduce the signature.

## Env fallback

`TWILIO_AUTH_TOKEN`, `TWILIO_ACCOUNT_SID`, `TWILIO_PHONE_NUMBER`.
