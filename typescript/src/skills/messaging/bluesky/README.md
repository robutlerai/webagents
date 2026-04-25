# BlueskySkill

Publish posts on Bluesky via the AT Protocol `com.atproto.repo.createRecord`
RPC.

## Tools

| Tool            | Capability       | Notes                                                    |
| --------------- | ---------------- | -------------------------------------------------------- |
| `bluesky_post`  | `publish_posts`  | Subject to host approval gate. Max 300 graphemes.        |

## Token shape

```jsonc
ResolvedToken.token = "<accessJwt from createSession>"
ResolvedToken.metadata = {
  "did":     "did:plc:xxxxx",     // required
  "pdsHost": "https://bsky.social" // optional
}
```

Hosts are responsible for refreshing the AT Protocol session JWT
(`com.atproto.server.refreshSession`) — the skill does not refresh.

## Env fallback

`BLUESKY_ACCESS_JWT`, `BLUESKY_DID`, `BLUESKY_PDS_HOST`.

## See also

<https://docs.bsky.app/docs/api/com-atproto-repo-create-record>
