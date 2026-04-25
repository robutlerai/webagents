# InstagramSkill

Direct-message replies via the Instagram Messaging API + Graph
publishing for IG Business / Creator accounts.

## Tools

| Tool                            | Capability         | Notes                                                                 |
| ------------------------------- | ------------------ | --------------------------------------------------------------------- |
| `instagram_send_dm`             | `send_messages`    | Routes via the connected Page (RESPONSE messaging type, 24h window).  |
| `instagram_publish_image`       | `publish_posts`    | Two-step container flow on `graph.instagram.com`.                     |
| `instagram_reply_to_comment`    | `manage_comments`  | Reply to comments on the connected account's posts.                   |

## Token shape

```jsonc
ResolvedToken.token = "<Page access token | IG-User token>"
ResolvedToken.metadata = {
  "igUserId":  "...", // required
  "pageId":    "...", // required for DM replies
  "appSecret": "..."  // optional, enables appsecret_proof
}
```

## Env fallback

`INSTAGRAM_TOKEN`, `INSTAGRAM_USER_ID`, `INSTAGRAM_PAGE_ID`,
`META_APP_SECRET`.

## See also

<https://developers.facebook.com/docs/instagram-platform>
