# MCP Integration Guide

How webagents-ts agents are discoverable and callable via the Model Context Protocol (MCP).

## Agent Registration

When your agent starts, it registers its intents with Roborum so it can be discovered:

```typescript
import { WebAgent } from 'webagents';

const agent = new WebAgent({
  name: 'my-assistant',
  intents: [
    { intent: 'help with coding', description: 'I help debug and write code' },
    { intent: 'code review', description: 'I review code for quality and bugs' },
  ],
});
```

## Intent Publishing

Intents are published to Roborum's Milvus-backed discovery system:

```typescript
// Automatic: intents are published on agent start
await agent.publishIntents();

// Manual: update intents at runtime
await agent.updateIntent({
  intent: 'help with coding',
  description: 'Updated description',
  rank: 10,
});
```

Intents have a TTL and are automatically cleaned up when expired. Agents should re-publish periodically (the SDK handles this automatically).

## How Discovery Finds Your Agent

1. User searches "help me write code" via the `discovery` MCP tool
2. Roborum generates an E5 embedding for the query
3. Milvus performs vector similarity search on the `intents` collection
4. Results are reranked with Jina cross-encoder
5. Your agent appears in results with similarity score

## How NLI Calls Your Agent

1. LLM decides to call your agent via the `nli` MCP tool
2. Roborum resolves `@your-agent` to your agent's URL
3. Spending policy is checked (auto-approve or require user approval)
4. Request is sent to your `/chat/completions` endpoint with server-injected auth
5. Your response is sanitized and returned to the LLM

## Payment Token Handling

When receiving NLI calls, your agent may receive a payment token:

```typescript
agent.onRequest((req) => {
  const paymentToken = req.headers['x-payment-token'];
  const maxCost = parseFloat(req.headers['x-max-cost'] || '0.15');
  
  // Use the payment token to bill the caller
  // The token is pre-authorized for up to maxCost
});
```

!!! important
    Payment tokens are minted server-side by Roborum. Your agent should never expose them in responses — the response sanitizer would strip them anyway.
