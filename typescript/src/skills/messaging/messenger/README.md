# MessengerSkill

Reply to Messenger conversations via Page tokens, list and publish Page
posts.

## Tools

| Tool                                     | Capability             | Notes                                                              |
| ---------------------------------------- | ---------------------- | ------------------------------------------------------------------ |
| `messenger_send_text`                    | `send_messages`        | RESPONSE messaging type — enforced 24h CS window.                  |
| `messenger_send_with_human_agent_tag`    | `send_messages`        | MESSAGE_TAG=HUMAN_AGENT — up to 7d, requires human agent permission.|
| `messenger_get_page_posts`               | `read_page_posts`      | Lists recent Page posts.                                           |
| `messenger_create_post`                  | `publish_page_posts`   | Subject to host approval gate.                                     |

## Token shape

```jsonc
ResolvedToken.token = "<Page access token>"
ResolvedToken.metadata = {
  "pageId":    "...", // required
  "appSecret": "..."  // optional, enables appsecret_proof
}
```

## Env fallback

`MESSENGER_PAGE_ACCESS_TOKEN`, `MESSENGER_PAGE_ID`, `META_APP_SECRET`.

## See also

<https://developers.facebook.com/docs/messenger-platform>
