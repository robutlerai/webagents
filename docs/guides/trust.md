---
title: Who can call your agent
description: The access block in an agent file, groups as tool and prompt scopes, and the identities an agent can verify.
---

# Who can call your agent

An agent decides who may call it, and what each caller gets, from identities it can verify. In an agent file that is the `access:` block; in code it is the scopes on tools and prompts. Both SDKs apply the same rules and answer with the same refusals.

## Identities

An identity comes only from a credential the agent verified, never from a request field, the `Host` header or where the request came from.

| Identity | Proven by |
| --- | --- |
| `agent:<URL>` | A Web Bot Auth signature (HTTP Message Signatures, RFC 9421) that verifies against the key set at that agent URL |
| `key:<thumbprint>` | The same signature: the RFC 7638 thumbprint of the key that verified it |
| `domain:<host>` | An `agent:` identity whose host is that name or a subdomain of it |
| `user:<id>`, `user:@<handle>` | A Robutler credential an auth skill verified: an API key, an owner assertion, or a platform service token addressed to this agent, which names the person it relays for |

The person at the terminal, in `webagents` chat and `webagents -p`, is the agent's owner.

## The access block

```yaml
---
name: concierge
skills:
  - openai
  - filesystem
  - rest
access:
  deny:
    - agent:https://spam.example/**
  groups:
    partners:
      - agent:https://*.acme.com/**
      - domain:partner.example
    family:
      - key:NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs
  default: public
  instructions:
    partners: PARTNERS.md
  tools:
    partners: [rest]
    family: [rest, filesystem]
---
You are the concierge for Acme.
```

| Key | Meaning |
| --- | --- |
| `deny` | Identities refused outright, whatever group they are in |
| `groups` | Group names (lower-case letters, digits, `-` and `_`) and the identities in each. `owner`, `admin`, `user`, `all` and `none` are reserved |
| `default` | The group of a caller in no group. `none` refuses them. Unset, it is `everyone` |
| `instructions` | A Markdown file per group, next to the agent file, added to the agent's instructions for callers in that group |
| `tools` | Skills (by their `skills:` name) or single tools that only the groups naming them may use, besides the owner |

In an `agent:` URL, `*` stands for one host label or one path segment and `**` for any number of path segments: `agent:https://*.acme.com/**` is every agent served under a direct subdomain of `acme.com`. `domain:partner.example` covers `partner.example` and its subdomains, not `evilpartner.example`.

### How a caller is placed

1. A caller matching `deny` is refused with `403`.
2. The owner, and an admin, get everything.
3. The caller joins every group one of its identities matches.
4. A caller in no group gets the `default` group, or `403` when `default: none`.

A signature that is present and does not verify is refused with `401`; it is never treated as an anonymous call. A bearer token the agent has no way to verify names no one, so its caller gets the `default` group.

### What the caller gets

- The tools of the skills the block does not name, as usual.
- The tools the block names, only when the caller is in a group naming them.
- The instructions file of each of its groups.
- A line telling the model who the turn is from: the verified agent URL or Robutler user, and its groups.

A tool the block does not name keeps its own scope: the `rest` skill's tool is the owner's alone until `tools:` hands it to a group.

## Serving an agent that verifies signatures

To check a signature, the agent needs the address callers sign for: set `WEBAGENTS_PUBLIC_URL` to it when you run `webagents serve`. A signature must be made for that host. Without it, a signed request is refused with `401` and says so.

The caller's key set is fetched from its agent URL over https, from a public address, without following redirects, and is cached. For local development, `ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1` allows plain http and private addresses.

## Scopes in code

Groups are scopes: a caller in `partners` holds `group:partners`. Declare one on a tool or a prompt:

```typescript tab="TypeScript"
class PartnerSkill extends Skill {
  @tool({ description: 'Partner pricing', scopes: ['group:partners'] })
  async partnerPrices(): Promise<string> {
    return '...';
  }

  @prompt({ scope: 'group:partners' })
  partnerGuide(): string {
    return 'Partners get net-30 terms.';
  }
}
```

```python tab="Python"
class PartnerSkill(Skill):
    @tool(scope="group:partners")
    async def partner_prices(self) -> str:
        """Partner pricing."""
        ...

    @prompt(scope="group:partners")
    def partner_guide(self) -> str:
        return "Partners get net-30 terms."
```

A declared scope is checked the same way everywhere, in both SDKs:

| Declared | Who may use it |
| --- | --- |
| `all`, or nothing | Everyone |
| `user` | A caller with a verified identity |
| `owner` | The owner, or an admin |
| `admin` | An admin |
| `group:<name>` | Members of that group, the owner, or an admin |
| anything else | Only a caller holding exactly that scope |
| a list | Anyone allowed by one of its entries |

The same rule decides who may call a scoped HTTP or WebSocket endpoint: see [Endpoints](../agent/endpoints.md#access-control-scopes).

## Refusals

A refused request gets JSON: `{"error": {"code": "...", "message": "..."}}`. The message says what to change.

| Status | Code | Why |
| --- | --- | --- |
| 403 | `forbidden` | Denied, in no group with `default: none`, or a verified caller a scoped endpoint is not open to |
| 401 | `unauthorized` | A scoped endpoint, and no identity the agent can verify |
| 401 | `signature_malformed`, `signature_params_invalid`, `signature_expired` | The signature headers, their parameters, or their time window |
| 401 | `signature_authority_mismatch` | Signed for another host, or the agent has no `WEBAGENTS_PUBLIC_URL` |
| 401 | `signature_coverage_insufficient` | The signature does not cover what it must: method, authority, path, query, the Signature-Agent member, and the body digest when there is a body |
| 401 | `signature_agent_invalid` | The Signature-Agent header does not name a usable key set |
| 401 | `key_set_unreachable`, `key_set_invalid`, `signature_key_unknown` | The key set could not be fetched or used, or does not hold the signing key |
| 401 | `signature_invalid`, `content_digest_mismatch`, `signature_replayed` | The signature does not verify, the body is not the one signed, or the signature was already used |

## Limits

- A Robutler user is identified only through an auth skill, which an agent file cannot name yet.
- A platform service token makes the person it relays for the owner, or gives them a `user:` identity, only when the agent can check that the token was addressed to it: set `WEBAGENTS_PUBLIC_URL`. Without it they are an ordinary verified caller.
- A Signature-Agent that names an agent card (`type=cimd`) instead of a key set is refused.
- An agent hosted on Robutler is configured on Robutler, not with this block.

## See also

- [REST calls](../skills/local/rest.md): an agent calling out, signed
- [Web Bot Auth client](./web-bot-auth-client.md): signing requests from your own code
