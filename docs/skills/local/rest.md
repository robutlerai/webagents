---
title: REST calls
description: The rest skill lets an agent call web APIs and other agents over HTTP(S), signing each request with Web Bot Auth when it can.
---

# REST calls

The `rest` skill gives an agent one tool, `rest_request`, that calls a web API or another agent over HTTP(S) and returns the response. When the agent is served at a public https address, every request it sends is signed with Web Bot Auth (HTTP Message Signatures, RFC 9421), so the service it calls can check which agent is calling. It behaves the same in the Python and TypeScript SDKs.

The built-in assistant (`webagents` with no agent file in the folder) has it, next to `filesystem`.

## Add it to an agent

```yaml
---
name: weather
model: openai/gpt-4o-mini
skills:
  - openai
  - rest
---
You answer questions about the weather. Use https://api.open-meteo.com for forecasts.
```

With settings:

```yaml
skills:
  - rest:
      sign: auto              # auto (default), always or never
      allow_private:          # private addresses this agent may call, IP or CIDR, optional port
        - 127.0.0.1:8080
        - 10.0.0.0/8
```

## What the model can send

| Argument | Meaning |
| --- | --- |
| `method` | `GET`, `HEAD`, `POST`, `PUT`, `PATCH`, `DELETE` or `OPTIONS` |
| `url` | An absolute http or https URL, with any query string |
| `headers` | Optional. A list of `"Name: value"` strings |
| `body` | Optional. Text; JSON is sent as `application/json` unless a `Content-Type` header says otherwise |
| `timeout_seconds` | Optional. 1 to 60, default 30, for the whole exchange |

## What comes back

A JSON object: `ok` (a 2xx status), `status`, the final `url`, how many `redirects` were followed, whether the request was `signed` (with `signed_as`, or `unsigned_reason`), the `content_type`, a few useful response `headers` (`location`, `retry-after`, `www-authenticate`, `link`, `etag`, `last-modified` and rate-limit headers), and the body as `text`, at most 100,000 characters. A binary body comes back as its `sha256` and size instead. `truncated` says when the body was cut.

A request the tool refuses comes back as `{"ok": false, "error": {"code", "message"}}`, with one of these codes: `invalid_request`, `invalid_url`, `forbidden_header`, `credential_in_request`, `blocked_address`, `network_error`, `timeout`, `too_large`, `redirect_limit` or `not_signed`.

## Signing

A request is signed when the agent has a public https address and a key: `webagents serve` with `WEBAGENTS_PUBLIC_URL` set to the address it is reachable at. The signature names the agent's key set at `<agent URL>/.well-known/jwks.json`, which the server publishes. In the local chat there is no public address, so requests go out unsigned, and each result says so.

`sign: always` refuses to send a request that could not be signed. `sign: never` never signs.

## What it refuses

The model choosing the URL may be reading a page someone else wrote, so the tool:

- calls public internet addresses only. Every address a name resolves to is checked, and the connection is made to the address that was checked. Private addresses are allowed only when `allow_private` lists them. Link-local and cloud metadata addresses (such as `169.254.169.254`) are refused whatever the list says.
- sets `Host`, the signature headers, payment headers and connection framing itself, and refuses a request that tries to set them.
- never sends a credential the program holds. A request whose URL, headers or body contains the value of an environment variable named like a credential (`*KEY*`, `*TOKEN*`, `*SECRET*` and similar) is refused.
- sends at most 1 MiB of body, and reads at most 1 MiB of answer.
- follows at most 3 redirects, for `GET` and `HEAD` only, checking and signing each hop again, and drops `Authorization` and `Cookie` when a redirect leaves the origin. Other methods get the redirect back.
- does not retry, keeps no cookies, and does not pay: a `402` comes back as it is.

## Who may use it

Only the agent's owner, by default: a signed request speaks as the agent. In the local chat you are the owner. To let others use it, name it for a group in the agent file's [access block](../../guides/trust.md):

```yaml
access:
  groups:
    partners: [agent:https://*.partner.example/**]
  tools:
    partners: [rest]
```
