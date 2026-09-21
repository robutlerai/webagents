---
title: Python ↔ TypeScript Parity Matrix
description: Source-of-truth mapping between the Python and TypeScript SDKs, used to drive the "Coming soon" tabs in the public docs.
---

# Python ↔ TypeScript Parity Matrix

This page is the single source of truth for which features ship in which SDK. Every public doc page that uses the synced `tab="TypeScript"` / `tab="Python"` pattern resolves "is this feature available?" against this matrix. When a feature ships in only one SDK, the missing tab renders a "Coming soon" stub linking to the relevant tracker.

> Last verified against [`webagents/python/webagents/`](../../python/webagents/) and [`webagents/typescript/src/`](../../typescript/src/) on the date of the most recent commit to this file. The "Identity and signing" and "Machine payments buyer" sections below were verified against the source on 2026-09-19; the rest were not re-read that day and are older.

## Decorators

| Decorator    | Python                                                                | TypeScript                                                  | Notes |
| ------------ | --------------------------------------------------------------------- | ----------------------------------------------------------- | ----- |
| `@tool`      | [`tools/decorators.py`](../../python/webagents/agents/tools/decorators.py) | [`core/decorators.ts`](../../typescript/src/core/decorators.ts) | Python: `scope=` (str or list). TS: `scopes: string[]`. |
| `@hook`      | yes                                                                   | yes                                                         | TS uses `lifecycle:` instead of positional `event`. |
| `@prompt`    | yes                                                                   | yes                                                         | Both support `priority` + `scope`. |
| `@handoff`   | yes                                                                   | yes                                                         | TS adds `subscribes` / `produces` for event routing. |
| `@http`      | yes                                                                   | yes                                                         | Python: `http("/path", method="get")`. TS: `http({ path, method: 'GET' })`. |
| `@websocket` | yes                                                                   | yes                                                         | |
| `@pricing`   | yes (`credits_per_call=`)                                             | yes (`creditsPerCall:`)                                     | |
| `@observe`   | n/a                                                                   | yes                                                         | TS-only; non-consuming event observer. |
| `@command`   | yes                                                                   | **Coming soon**                                             | CLI/REPL slash commands. |
| `@widget`    | yes                                                                   | **Coming soon**                                             | HTML widgets returned to capable clients. |

## Core (built-in skills)

| Capability  | Python (`agents/skills/core/`)                          | TypeScript (`skills/`)                              | Notes |
| ----------- | ------------------------------------------------------- | --------------------------------------------------- | ----- |
| LLM         | `core/llm` (openai, anthropic, google, …)               | `skills/llm` (openai, anthropic, google, fireworks, xai, transformers, webllm, proxy) | TS adds `transformers` and `webllm` for in-browser inference. |
| MCP         | `core/mcp`                                              | `skills/mcp` (`MCPSkill`)                           | |
| Transport   | `core/transport`                                        | `skills/transport` (a2a, acp, completions, portal, realtime, uamp) | |
| Memory      | `core/memory`                                           | exposed via `skills/storage` only                   | TS does not ship a discrete `core/memory` skill. |
| Guardrails  | `core/guardrails`                                       | **Coming soon**                                     | |
| Planning    | `core/planning`                                         | **Coming soon**                                     | |

## Local (workstation-side skills)

| Capability   | Python (`agents/skills/local/`) | TypeScript (`skills/`)                              | Notes |
| ------------ | ------------------------------- | --------------------------------------------------- | ----- |
| Browser      | `local/browser`                 | `skills/browser` (automation, camera, geolocation, microphone, notifications, search, storage, wakelock) | TS targets in-browser execution; Python targets Playwright. |
| Checkpoint   | `local/checkpoint`              | `skills/checkpoint` (`CheckpointSkill`)             | |
| Filesystem   | `local/filesystem`              | `skills/filesystem` (`FilesystemSkill`)             | |
| MCP          | `local/mcp`                     | `skills/mcp`                                        | TS uses one MCP skill for both local and remote servers. |
| Plugin       | `local/plugin`                  | `skills/plugin` (`PluginSkill`)                     | |
| RAG          | `local/rag`                     | `skills/rag` (`RAGSkill`)                           | |
| Sandbox      | `local/sandbox`                 | `skills/sandbox` (`SandboxSkill`)                   | |
| Session      | `local/session`                 | `skills/session` (`SessionSkill`)                   | |
| Shell        | `local/shell`                   | `skills/shell` (`ShellSkill`)                       | |
| Test runner  | `local/testrunner`              | `skills/testrunner` (`TestRunnerSkill`)             | |
| Todo         | `local/todo`                    | `skills/todo` (`TodoSkill`)                         | |
| Auth (local) | `local/auth`                    | **Coming soon**                                     | TS exposes only the platform `AuthSkill`. |
| CLI          | `local/cli`                     | **Coming soon**                                     | |
| LSP          | `local/lsp`                     | **Coming soon**                                     | |
| Web          | `local/web`                     | **Coming soon**                                     | |
| WebUI        | `local/webui`                   | **Coming soon**                                     | |

## Robutler / platform

| Capability         | Python (`agents/skills/robutler/`)        | TypeScript (`skills/`)                                            | Notes |
| ------------------ | ----------------------------------------- | ----------------------------------------------------------------- | ----- |
| Auth               | `robutler/auth`                           | `skills/auth` (`AuthSkill`)                                       | JWT verification via JWKS. |
| Chats              | `robutler/chats`                          | `skills/social` (`ChatsSkill`)                                    | |
| Discovery          | `robutler/discovery`                      | `skills/discovery` (`PortalDiscoverySkill`)                       | |
| NLI                | `robutler/nli`                            | `skills/nli` (`NLISkill`)                                         | |
| Notifications      | `robutler/notifications`                  | `skills/social` (`NotificationsSkill`)                            | |
| OpenAPI            | n/a (under `agents/skills/...`)           | `skills/openapi` (`OpenAPISkill`)                                 | |
| Payments           | `robutler/payments`                       | `skills/payments` (`PaymentSkill`)                                | |
| Payments (x402)    | `robutler/payments_x402`                  | `skills/payments` (`x402.ts`)                                     | |
| Machine payments buyer | `robutler/payments_x402/mpp_buyer.py`  | `skills/payments/mpp-buyer.ts`                                    | Full parity of behaviour, not of idiom. See "Machine payments buyer" below. |
| Settle result      | `robutler/payments/settle_result.py`      | `skills/payments/settle-result.ts`                                | Partial-settle outcome shared by both. |
| Portal Connect     | `robutler/portal_connect`                 | `skills/social` (PortalConnect)                                   | |
| Portal WS          | `robutler/portal_ws`                      | `skills/social` (PortalWS)                                        | |
| Storage / KV / JSON | `robutler/storage`, `robutler/kv`, `robutler/memory` | `skills/storage` (`RobutlerMemorySkill`, `RobutlerKVSkill`, `RobutlerJSONSkill`) | |
| Social             | `robutler/social`                         | `skills/social` (`SocialSkill`)                                   | |
| Messages           | `robutler/messages`                       | `skills/messaging/*` (slack, discord, telegram, whatsapp, twilio, sendgrid, x, bluesky, instagram, linkedin, messenger, reddit, tiktok, google-chat) | |
| CRM                | `robutler/crm`                            | **Coming soon**                                                   | |
| Handoff            | `robutler/handoff` (skill)                | uses `@handoff` decorator only                                    | TS exposes the decorator but not the skill module yet. |
| Integrations       | `robutler/integrations`                   | **Coming soon**                                                   | |
| Message history    | `robutler/message_history`                | folded into `skills/social` registry                              | |
| Namespace          | `robutler/namespace`                      | **Coming soon**                                                   | |
| Publish            | `robutler/publish`                        | **Coming soon**                                                   | |
| Files              | `robutler/files`                          | **Coming soon**                                                   | |

## Identity and signing

Both SDKs sign every request to Robutler (AOAuth, the platform's profile of Web Bot Auth) and both persist an Ed25519 key. The behaviour is the same on both; the asymmetries below are real and worth knowing before you copy a snippet across.

| Capability | Python | TypeScript | Notes |
| --- | --- | --- | --- |
| RFC 9421 signing | `crypto/http_signature.py` | `crypto/http-signature.ts` | Same covered components, same parameters, same `web-bot-auth` tag. |
| `@path` / `@query` spelling | WHATWG normalisation, then the signed URL is what is sent (`apply_signed_target`) | free from `new URL()` | Python reaches the TypeScript behaviour explicitly; a mismatch here used to read as `signature_invalid` over a base that looked correct. |
| Shared request-target vectors | `python/tests/fixtures/web_bot_auth/vectors-request-target.json` | same file, read across the tree from `tests/unit/crypto/http-signature-request-target.test.ts` | Seven cases. Both regenerate the document and compare byte for byte; neither rewrites it. The file lives in the Python tree, so a TypeScript-only checkout cannot run that test. |
| Signature lifetime | `check_lifetime()`, `SigningError` | inline in `signMessage`, plain `Error` | Both require an integer from 1 to 3600 seconds. Python also validates at `WebBotAuth` construction, and rejects `bool`, which is an `int` in Python. |
| One nonce per label | `nonces: Sequence[str]` | `nonce: string \| ((labelIndex) => string)` | Same invariant, different seam: two labels sharing a nonce is refused in both, because the platform spends a nonce per principal. |
| Key persistence | `JWKSManager` in `crypto/jwks.py`, `{name}.ed25519.pem` | `crypto/identity-store.ts`, `{name}.ed25519.jwk.json` | Same rules: a missing file is the only reason to generate, the write is a `0600` temp file hard-linked into place, and permissions are repaired on load. |
| Rotation file | `{name}.ed25519.previous.pem` | `{name}.ed25519.previous.jwk.json` | Python refuses a previous key whose thumbprint equals the current one; TypeScript does not check, so a copied rather than moved file yields two identical labels. |
| Unreadable key file | `RuntimeError` naming the path | `AgentKeyFileError` | A genuine gap: a TypeScript caller can discriminate on the class, a Python caller can only match the message. |
| Owner key on registration | `owner_api_key=`, `ROBUTLER_API_KEY` fallback, `owned` in the result | `ownerApiKey`, same fallback, same `owned` | Both cover `X-Robutler-Owner-Key` with the signature; uncovered, the platform ignores it and the agent registers ownerless. |
| Registration redirects | `follow_redirects=False` on the request, plus an explicit 3xx refusal whose error sentence says why | `redirect: 'error'`, asserted | Now an invariant on both sides rather than a property of the default client. The registering request is the one that may carry the operator's platform key, so a `Location` is never dialled. Python's refusal returns `{"ok": False, "status": <3xx>, "error": ...}`; TypeScript throws out of `fetch`. |
| Claim link | `claim_url(identity, agent_name, agent_user_id, *, platform_url, ttl_seconds, agent_url)` in `server/core/registration.py` | `claimUrl(identity, agentUserId, { platformUrl, ttlSeconds })` in `server/registration.ts` | Same shape, `{platform}/claim/{agent_user_id}#{token}`, same ten-minute default, `None`/`null` when no platform URL resolves. Signature asymmetry: Python takes the agent NAME as well, because its `mint_claim_token` is keyed on it, where the TypeScript identity already knows its own `agentId`. `mint_claim_token` / `mintClaimToken` remain underneath and are not what an operator should call: the helper owns the URL fragment. |
| Key directory | `server/core/key_directory.py`, mounted by `add_api_route` in `server/core/app.py` | `server/key-directory.ts`, dispatched from `server/handler.ts` (single agent) and `server/multi.ts` (every hosted identity) | `GET /.well-known/http-message-signatures-directory` at the ORIGIN, `application/http-message-signatures-directory+json`, `Cache-Control: public, max-age=3600`, 404 with no signing identity. Union of the hosted agents' Ed25519 keys, one entry per thumbprint. Asymmetry: the TypeScript single-agent `serve()` path lists that one identity, where `WebAgentsServer` and the Python server list every hosted agent. |

## Machine payments buyer

`mpp_buyer.py` and `mpp-buyer.ts` are close to a line-for-line match. The differences that change what an operator must do:

| Behaviour | Python | TypeScript | Notes |
| --- | --- | --- | --- |
| Redirects | `MppRedirectError`, `redirect_refused` | `MppRedirectError`, `redirect_refused` | Same class name and same reason string. Both refuse before paying and both also catch a redirect a misconfigured transport already followed. |
| Send timeout | 60 s default, guaranteed on a client the buyer opens itself | 60 s default, enforced by the buyer's own abort | The asymmetry that matters: pass your own `httpx` client to the Python buyer and its timeout, not the policy's, is what bounds the send. |
| Seller pin | `DISCOVERY_PIN_TTL_MS = 3_600_000` | same name, same value | One hour for a positive pin, five minutes for a document naming none, and a stale pin is never a fallback. The pin is the seller's Stripe profile id and stablecoin deposit address, read from the seller's `/openapi.json`; two values for one method pin nothing. |
| Reserved credential header | challenge is unreadable | challenge is unreadable | A challenge naming a header the signer, the buyer or the HTTP client owns is dropped before any payment source is called. Both surface it as the generic refusal `no_challenge`, so the specific reason never reaches the operator. |
| Paid streamed 2xx | `stream=True` on the send; `_settle_body` leaves an undeclared-length 2xx unread and `_HandedStream` puts back whatever it peeked at | `buyerReadsBody` / `readSmallJson`; an undeclared-length 2xx is left unread, and what is read is read from a clone | Same rule, both bounded at 64 KiB: a paid answer is handed to the caller unconsumed. The Python seam is the one to know, because a client you pass in is the object whose stream is being protected. |
| Purchase pointer | `purchase_at(...)`, `_pointer_verdict` | `purchaseAt({ url, from, ... })`, `pointerVerdict` | Same preconditions and the same two refusal reasons, `realm_not_allowed` and `pointer_needs_daily_cap`, both decided before anything is sent. Naming asymmetry only: Python takes `source_url=` where TypeScript takes `from`. |
| Who acts on a pointer | NLI skill (inline, socket side) and `_follow_purchase_pointer` in the LLM proxy skill | UAMP client, NLI skill, LLM proxy skill | A real gap, not an idiom: **Python has no UAMP client class** (`python/webagents/uamp/` is events and types only), so there is no shared seam to put this in and the socket-side handling lives in the NLI skill. Anything written about "the UAMP client follows the pointer" is TypeScript-only. |
| Replay without the dry token | `_without_payment_token` | `withoutPaymentToken` | Both strip `x-payment-token`, `x-payment` and `?payment_token=` before re-sending the original call, and keep the body and every other header. |

## Ecosystem integrations

| Capability   | Python (`agents/skills/ecosystem/`) | TypeScript (`skills/`)                  | Notes |
| ------------ | ----------------------------------- | --------------------------------------- | ----- |
| OpenAI       | `ecosystem/openai`                  | `skills/llm/openai`                     | TS treats OpenAI as an LLM provider, not a separate ecosystem integration. |
| X / Twitter  | `ecosystem/x_com`                   | `skills/messaging/x`                    | |
| crewai       | `ecosystem/crewai`                  | **Coming soon**                         | |
| Database     | `ecosystem/database`                | **Coming soon**                         | |
| fal          | `ecosystem/fal`                     | **Coming soon**                         | |
| Google       | `ecosystem/google`                  | **Coming soon** (chat covered by `skills/messaging/google-chat`) | |
| MongoDB      | `ecosystem/mongodb`                 | **Coming soon**                         | |
| n8n          | `ecosystem/n8n`                     | **Coming soon**                         | |
| Replicate    | `ecosystem/replicate`               | **Coming soon**                         | |
| UCP          | `ecosystem/ucp`                     | **Coming soon**                         | |
| Web          | `ecosystem/web`                     | **Coming soon**                         | |
| Zapier       | `ecosystem/zapier`                  | **Coming soon**                         | |

## TypeScript-only (no Python equivalent yet)

| TS module           | Notes |
| ------------------- | ----- |
| `skills/speech`     | STT / TTS for in-browser voice agents. |
| `skills/routing`    | `DynamicRoutingSkill` — runtime agent-to-agent discovery and delegation. |
| `skills/media`      | `StoreMediaSkill` — distinct from Python `core/media`. |
| `skills/messaging/{bluesky,instagram,linkedin,messenger,reddit,sendgrid,telegram,tiktok,twilio,whatsapp,google-chat}` | Provider modules that do not yet have Python counterparts. |

## CLI / Server

- Python ships a full CLI ([`webagents/python/webagents/cli/main.py`](../../python/webagents/cli/main.py)) with `serve | repl | daemon | sandbox | session` plus subcommands.
- TypeScript ships `webagents` and `robutler` bins ([`webagents/typescript/package.json`](../../typescript/package.json)) with a smaller surface — equivalent commands map onto Python where supported, with the rest marked **Coming soon**.

## Conventions for doc snippets

1. Every code example that demonstrates SDK usage must render both tabs (`tab="TypeScript"` then `tab="Python"`). The remark plugin in [`lib/remark-code-tabs.ts`](../../../lib/remark-code-tabs.ts) merges consecutive tagged blocks into `<Tabs groupId="lang" persist>`.
2. When a feature is "Coming soon" in a tab, the body of that tab is a single comment explaining the gap and (where useful) the closest current alternative. Inside MDX-only pages, a `Callout` may also be used; in plain Markdown, a `> Note:` blockquote is sufficient and renders correctly under both Fumadocs and mkdocs-material.
3. Snippets must always match the actual exported API. Verify against:
   - Python: [`webagents/python/webagents/agents/`](../../python/webagents/) (tools, skills, decorators).
   - TypeScript: [`webagents/typescript/src/`](../../typescript/src/) (`core/`, `skills/`, `server/`).
