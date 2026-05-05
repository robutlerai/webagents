---
title: Python ↔ TypeScript Parity Matrix
description: Source-of-truth mapping between the Python and TypeScript SDKs, used to drive the "Coming soon" tabs in the public docs.
---

# Python ↔ TypeScript Parity Matrix

This page is the single source of truth for which features ship in which SDK. Every public doc page that uses the synced `tab="TypeScript"` / `tab="Python"` pattern resolves "is this feature available?" against this matrix. When a feature ships in only one SDK, the missing tab renders a "Coming soon" stub linking to the relevant tracker.

> Last verified against [`webagents/python/webagents/`](../../python/webagents/) and [`webagents/typescript/src/`](../../typescript/src/) on the date of the most recent commit to this file.

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
