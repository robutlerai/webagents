# WhatsAppSkill

Send messages via WhatsApp Cloud API (Meta Graph). Enforces the 24-hour
customer-service window via `whatsapp_send_text`; agents step outside
the window with `whatsapp_send_template` (approved templates only).

## Tools

| Tool                       | Capability          | Notes                                          |
| -------------------------- | ------------------- | ---------------------------------------------- |
| `whatsapp_send_text`       | `send_messages`     | Required reply path inside the 24h CS window.  |
| `whatsapp_send_template`   | `send_messages`     | Approved templates; allowed outside the window.|
| `whatsapp_list_templates`  | `manage_templates`  | Lists WABA templates.                          |

## Token shape

```jsonc
ResolvedToken.token = "<system user token | long-lived user token>"
ResolvedToken.metadata = {
  "phoneNumberId": "...",  // required
  "wabaId":        "...",  // required
  "appSecret":     "..."   // optional, enables appsecret_proof
}
```

`appSecret` falls back to `META_APP_SECRET`.

## Env fallback

`WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_BUSINESS_ACCOUNT_ID`,
`META_APP_SECRET`.

## See also

Meta Cloud API reference:
<https://developers.facebook.com/docs/whatsapp/cloud-api>
