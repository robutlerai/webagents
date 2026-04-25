# LinkedInSkill

Publish UGC posts to a LinkedIn personal profile or organization page.

## Tools

| Tool             | Capability       | Notes                                                       |
| ---------------- | ---------------- | ----------------------------------------------------------- |
| `linkedin_post`  | `publish_posts`  | Subject to host approval gate. Supports text + article URL. |

## Token shape

```jsonc
ResolvedToken.token = "<OAuth2 access token (w_member_social)>"
ResolvedToken.metadata = {
  "personUrn":       "urn:li:person:abc",   // required for personal posts
  "organizationUrn": "urn:li:organization:123" // required when as_organization=true
}
```

## Env fallback

`LINKEDIN_ACCESS_TOKEN`, `LINKEDIN_PERSON_URN`, `LINKEDIN_ORG_URN`.

## See also

<https://learn.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/ugc-post-api>
