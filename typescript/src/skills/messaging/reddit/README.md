# RedditSkill

Submit posts to a subreddit using OAuth2.

## Tools

| Tool          | Capability       | Notes                                                                                  |
| ------------- | ---------------- | -------------------------------------------------------------------------------------- |
| `reddit_post` | `publish_posts`  | Subject to host approval gate. Supports `kind=self|link`, `nsfw`, `spoiler` flags.     |

## Token shape

```jsonc
ResolvedToken.token = "<OAuth2 bearer token (submit scope)>"
ResolvedToken.metadata = { "userAgent": "myapp/1.0 by u/foo" }
```

`metadata.userAgent` falls back to `REDDIT_USER_AGENT` then to a generic
default. Reddit is strict — set a unique UA in production.

## Env fallback

`REDDIT_ACCESS_TOKEN`, `REDDIT_USER_AGENT`.

## See also

<https://www.reddit.com/dev/api/#POST_api_submit>
