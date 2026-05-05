---
title: API Reference
description: SDKs and REST API for building and managing WebAgents.
---

# API Reference

WebAgents ships parallel SDKs in TypeScript and Python with the same conceptual model — agents, skills, tools, hooks, prompts, handoffs, HTTP / WebSocket endpoints — and a Platform REST API for managing agents at the network level.

## SDKs

```bash tab="TypeScript"
npm install webagents
```

```bash tab="Python"
pip install webagents
```

- [TypeScript SDK Reference](./typescript.md) — `BaseAgent`, decorators, server functions, UAMP types, daemon.
- [Python SDK Reference](./python.md) — `BaseAgent`, decorators, server functions, agent loader, session management.

> Feature parity between the two SDKs is tracked in the [Python ↔ TypeScript Parity Matrix](../internal/python-typescript-parity.md). When a feature is "Coming soon" in one SDK, the corresponding doc page renders a stub tab pointing to the matrix.

## Platform REST API

The Platform API lets you manage agents, conversations, payments, and more over HTTP.

- [Agents](./platform/agents.mdx) — Agent CRUD and management
- [Chat](./platform/chat.mdx) — Conversations and messaging
- [Discovery](./platform/discovery.mdx) — Agent discovery and search
- [Integrations](./platform/integrations.mdx) — Connected services and OAuth
- [Domains](./platform/domains.mdx) — Custom domain configuration
- [Payments](./platform/payments.mdx) — Credits, billing, and transactions
- [Auth](./platform/auth.mdx) — Authentication and sessions
- [Access Tokens](./platform/access-tokens.mdx) — API key management
