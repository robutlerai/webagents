---
title: Architecture
description: Production architecture, deployment patterns, and operational concerns for WebAgents servers.
---

# Server Architecture

Production architecture and deployment patterns.

> [!WARNING]
> Robutler is in beta. APIs may change. Test thoroughly before production deployment.

## Architecture Overview

The Robutler server is built on FastAPI with these core components:

- **Agent Manager** - Routes requests to appropriate agents
- **Skill Registry** - Manages agent capabilities and tools
- **Context Manager** - Handles request context and user sessions
- **LLM Proxy** - Integrates with OpenAI, Anthropic, and other providers

## Request Flow

1. **Request** arrives at FastAPI server
2. **Authentication** validates API keys and user identity
3. **Routing** selects agent based on URL path
4. **Context** creates request context with user information
5. **Execution** runs agent with skills and LLM integration
6. **Response** returns streaming or batch results

## Configuration

### Environment Variables

```bash
# Server
ROBUTLER_HOST=0.0.0.0
ROBUTLER_PORT=8000
ROBUTLER_LOG_LEVEL=INFO

# LLM Providers
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional Features
DATABASE_URL=postgresql://user:pass@host/db
REDIS_URL=redis://localhost:6379
PROMETHEUS_ENABLED=true
```

### Server Configuration

```typescript tab="TypeScript"
import { createAgentApp } from 'webagents/server';

const app = createAgentApp({
  title: 'Production Server',
  agents,
  enableCors: true,
  requestTimeoutMs: 300_000,
});
```

```python tab="Python"
from webagents.server.core.app import create_server

server = create_server(
    title="Production Server",
    agents=agents,
    enable_monitoring=True,
    enable_cors=True,
    request_timeout=300,
)
```

## Production Patterns

### Multi-Agent Server

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { createAgentApp } from 'webagents/server';
import { serve } from 'webagents/server/node';

function createProductionServer() {
  const agents = [
    new BaseAgent({ name: 'support', model: 'openai/gpt-4o' }),
    new BaseAgent({ name: 'sales', model: 'openai/gpt-4o' }),
    new BaseAgent({ name: 'analyst', model: 'anthropic/claude-3-sonnet' }),
  ];
  return createAgentApp({
    title: 'Production Multi-Agent Server',
    agents,
    urlPrefix: '/api/v1',
  });
}

const app = createProductionServer();
await serve(app, { host: '0.0.0.0', port: 8000 });
```

```python tab="Python"
from webagents.agents import BaseAgent
from webagents.server.core.app import create_server

def create_production_server():
    agents = [
        BaseAgent(name="support", model="openai/gpt-4o"),
        BaseAgent(name="sales", model="openai/gpt-4o"),
        BaseAgent(name="analyst", model="anthropic/claude-3-sonnet"),
    ]

    return create_server(
        title="Production Multi-Agent Server",
        agents=agents,
        url_prefix="/api/v1",
        enable_monitoring=True,
    )

if __name__ == "__main__":
    import uvicorn
    server = create_production_server()
    uvicorn.run(server.app, host="0.0.0.0", port=8000, workers=4)
```

### Dynamic Agent Loading

```typescript tab="TypeScript"
import { createAgentApp } from 'webagents/server';

async function resolveAgent(agentName: string): Promise<BaseAgent | null> {
  const config = await loadAgentConfig(agentName);
  return config ? new BaseAgent(config) : null;
}

const app = createAgentApp({
  agents: staticAgents,
  dynamicAgents: resolveAgent,
});
```

```python tab="Python"
async def resolve_agent(agent_name: str):
    """Load agent configuration from database/API"""
    config = await load_agent_config(agent_name)
    if config:
        return BaseAgent(**config)
    return None

server = create_server(
    agents=static_agents,
    dynamic_agents=resolve_agent,
)
```

## Monitoring

### Health Checks

Built-in endpoints (both SDKs):

```http
GET /health              # Server health
GET /{agent}/health      # Agent health
```

### Metrics

Enable Prometheus metrics:

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
// In TypeScript, expose metrics via a custom @http handler that reads
// from your own counters / OpenTelemetry exporter.
```

```python tab="Python"
server = create_server(
    agents=agents,
    enable_prometheus=True,
)
```

Access metrics at `/metrics` endpoint.

### Logging

Configure structured logging:

```typescript tab="TypeScript"
import { pino } from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
});
```

```python tab="Python"
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
```

## Deployment

### Production Server

```typescript tab="TypeScript"
import { serve } from 'webagents/server/node';

async function main() {
  const app = createProductionServer();
  await serve(app, { host: '0.0.0.0', port: 8000 });
}

main();
```

```python tab="Python"
import uvicorn
from webagents.server.core.app import create_server

def main():
    server = create_production_server()
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        access_log=True,
    )

if __name__ == "__main__":
    main()
```

## Security

### API Authentication

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { AuthSkill } from 'webagents/skills/auth';

const agent = new BaseAgent({
  name: 'secure-agent',
  model: 'openai/gpt-4o',
  skills: [new AuthSkill()],
});
```

```python tab="Python"
from webagents.agents.skills.robutler.auth import AuthSkill

agent = BaseAgent(
    name="secure-agent",
    model="openai/gpt-4o",
    skills={"auth": AuthSkill()},
)
```

### CORS Configuration

```typescript tab="TypeScript"
import { createAgentApp } from 'webagents/server';

const app = createAgentApp({
  agents,
  enableCors: true,
  corsOrigins: ['https://yourdomain.com'],
});
```

```python tab="Python"
server = create_server(
    agents=agents,
    enable_cors=True,
    cors_origins=["https://yourdomain.com"],
)
```

## Performance Tuning

### Concurrency

```bash
# Multiple workers for CPU-bound tasks
uvicorn main:server.app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Async for I/O-bound tasks
uvicorn main:server.app --loop asyncio --http httptools
```

### Resource Limits

```typescript tab="TypeScript"
import { createAgentApp } from 'webagents/server';

const app = createAgentApp({
  agents,
  requestTimeoutMs: 300_000,
  maxRequestSizeBytes: 10 * 1024 * 1024,
});
```

```python tab="Python"
server = create_server(
    agents=agents,
    request_timeout=300,
    max_request_size="10MB",
)
```

## Best Practices

1. **Environment Variables** - Use env vars for configuration
2. **Health Checks** - Implement proper health endpoints
3. **Logging** - Use structured logging for observability
4. **Resource Limits** - Set appropriate timeouts and limits
5. **Monitoring** - Enable metrics collection
6. **Security** - Use authentication and CORS properly

## See Also

- **[Server Overview](./index.md)** - Basic server setup
- **[Dynamic Agents](./dynamic-agents.md)** - Runtime agent loading
- **[Agent Skills](../agent/skills.md)** - Agent capabilities