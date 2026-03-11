# Skill Partitioning: OSS vs Closed

## Principle

The **webagents SDK** (MIT) contains protocol implementations and thin API clients. Anyone can build agents that connect to the Robutler platform.

**Closed repositories** (agents, portal) contain platform internals with direct DB access and proprietary business logic.

## OSS SDK (webagents package)

### Both Python and TypeScript:

- **LLM providers**: Google, OpenAI, Anthropic, xAI, Fireworks, WebLLM, Transformers
- **LLMProxySkill**: UAMP client to platform LLM proxy
- **Transports**: completions, UAMP, A2A, ACP, realtime, portal-connect
- **Platform API clients**: auth, payments, x402, NLI, discovery, storage, social
- **Local tools**: filesystem, shell, RAG, MCP, session, sandbox, browser
- **Ecosystem**: CrewAI, FAL, Google, MongoDB, N8N, Replicate, Zapier

## Closed (portal + agents repos)

- **UAMP LLM Proxy**: Reuses OSS LLM skills, adds BYOK + settlement
- **SettingsSkill / FactorySkill**: Privileged platform management with direct DB
- **PortalPaymentSkill**: Direct DB settlement (replaces HTTP API calls)
- **Proprietary media/business skills**

## Decision Criteria for New Skills

| Criteria | OSS | Closed |
|----------|-----|--------|
| Protocol client (HTTP/WS/UAMP) | ✅ | |
| Direct database access | | ✅ |
| Platform management (CRUD agents) | | ✅ |
| Generic tool (filesystem, browser) | ✅ | |
| Proprietary business logic | | ✅ |
| Third-party integration | ✅ | |
