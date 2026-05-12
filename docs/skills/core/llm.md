---
title: LLM Skills
description: Harmonized interface for OpenAI, Anthropic, Google, xAI, Fireworks, and proxy LLMs — provider-agnostic configuration, thinking, media support.
---

# LLM Skills

WebAgents provides a harmonized interface for interacting with various Large Language Model (LLM) providers. Whether you are using Google Gemini, OpenAI, Anthropic Claude, or xAI Grok, the configuration patterns for tools and reasoning capabilities remain consistent.

Every LLM skill is a thin wrapper over a **shared provider adapter** (`webagents/typescript/src/adapters/`). Adapters own all provider-specific logic — request building, stream parsing, media support declarations — while skills focus on lifecycle, context, and billing integration. This architecture means adding a new provider requires only a new adapter; billing, media handling, and tool pricing work automatically.

## Supported Providers

- **Google**: Gemini models via shared `googleAdapter`.
- **OpenAI**: GPT and o-series models via shared `openaiAdapter`.
- **Anthropic**: Claude models via shared `anthropicAdapter`.
- **xAI (Grok)**: Grok models via shared `xaiAdapter` (OpenAI-compatible).
- **Fireworks**: Open-weight models via shared `fireworksAdapter` (OpenAI-compatible).

## Configuration

LLM skills are configured in your `AGENT.md` file under the `skills` section or passed dynamically.

### Basic Configuration

All LLM skills accept these common parameters:

```yaml
skills:
  - google:
      model: "gemini-2.5-flash"
      temperature: 0.7
      max_tokens: 8192
      api_key: "${GOOGLE_API_KEY}" # Optional, uses env var by default
```

### Harmonized "Thinking" Configuration

Thinking effort is exposed across providers as a single canonical level. Adapters map the level onto each provider's native parameter (`thinkingConfig` for Google, `reasoning_effort` for OpenAI / xAI, `thinking` block for Anthropic).

```ts
type ThinkingLevel = 'off' | 'low' | 'medium' | 'high';
```

```yaml
skills:
  - google:
      model: "gemini-3-flash"
      thinking: "medium"   # 'off' | 'low' | 'medium' | 'high', or omit to use the model's catalog default
```

| Value | Behaviour |
| :--- | :--- |
| `'off'` | Disable thinking. For thinking-only models (Gemini 3 Pro, Grok 4) the adapter clamps to the lowest supported level. |
| `'low'` / `'medium'` / `'high'` | Map to the provider's native budget/effort. |
| omitted / `undefined` | Use the model's catalog default (`capabilities.thinking.default`). |

**Legacy compat shim:** `thinking: true` is treated as omitted (use default), `thinking: false` as `'off'`. Use string levels for new code.

#### Per-model capability declaration

Every model in `lib/models/catalog.ts` declares what it actually supports:

```ts
capabilities.thinking = {
  supported: true,
  levels: ['low', 'medium', 'high'],   // subset of off|low|medium|high
  default: 'medium',
};
```

The portal UI hides the control entirely when `supported: false`, and lists only the declared `levels`. The proxy clamps any incoming level against `levels` before calling the adapter (`clampThinkingLevel`), so the request never hits the provider with a value it would reject (e.g. Gemini 3 Pro rejecting `MINIMAL`).

#### Auto-Enabled Thinking

Some providers enable thinking automatically without explicit configuration:

- **Google Gemini 2.5+ / 3.x**: The adapter sets `thinkingConfig: { includeThoughts: true }` in `generationConfig` for all gemini-2.5 and gemini-3 series models. Thinking parts (with `part.thought === true`) are streamed as `thinking` chunks.
- **Fireworks / DeepSeek**: Reasoning models (DeepSeek R1, Kimi K2 Thinking, Qwen3, GLM, MiniMax, etc.) automatically emit `delta.reasoning_content` in the Chat Completions streaming response. The OpenAI-compatible adapter parses these as `thinking` chunks.
- **Anthropic**: Extended thinking is enabled for Claude models that support it. Thinking content blocks are parsed and streamed.

#### Thinking Persistence

Thinking content is **persisted** to messages in the database as `ThinkingContent` items within `contentItems`. This means thinking survives page reloads and is visible in chat history. Consecutive thinking deltas are merged into a single content item, and inline ordering relative to text and tool calls is preserved.

### Built-in Tools Configuration

Enable provider-specific built-in tools using harmonized names where possible.

```yaml
skills:
  - google:
      tools:
        - web_search        # Maps to Google Search
        - code_execution    # Maps to Google Code Execution
  
  - openai:
      tools:
        - web_search        # Maps to OpenAI Web Search
        - code_interpreter  # Maps to OpenAI Code Interpreter
        
  - anthropic:
      tools:
        - computer_use      # Maps to Claude Computer Use
        - bash              # Maps to Bash tool
        - text_editor       # Maps to Text Editor tool
```

**Common Tool Names:**

- `web_search`: General web search capability.
- `code_execution` / `code_interpreter`: Python code execution sandbox.

## Shared Adapter Architecture

All LLM skills delegate provider-specific work to shared adapters defined in `webagents/typescript/src/adapters/`. Each adapter implements the `LLMAdapter` interface:

```typescript
interface LLMAdapter {
  provider: string;
  mediaSupport: Record<string, 'base64' | 'url'>;
  buildRequest(params: AdapterRequestParams): AdapterRequest;
  parseStream(response: Response): AsyncGenerator<AdapterChunk>;
}
```

### Media Support

Each adapter declares which content modalities it supports and how (base64 inline data vs URL reference). The [MediaSkill](./media.md) reads these declarations to automatically convert content to the right format before an LLM call.

| Provider | Images | Audio | Documents | Video |
|----------|--------|-------|-----------|-------|
| Google | base64 | base64 | base64 | base64 |
| OpenAI | url | base64 | — | — |
| Anthropic | base64 | — | base64 | — |
| xAI | url | — | — | — |
| Fireworks | url | — | — | — |

### Context Integration

Every skill sets two context fields that other skills (PaymentSkill, MediaSkill) depend on:

- **`_llm_capabilities`**: Set *before* the LLM call with model name, pricing rates, and max output tokens. Used by PaymentSkill to lock funds.
- **`_llm_usage`**: Set *after* the LLM call with actual token counts, model used, and `is_byok` flag. Used by PaymentSkill to settle charges.

```typescript
// Before call
context.set('_llm_capabilities', {
  model: 'gemini-2.5-flash',
  pricing: { inputPer1kTokens: 0.00015, outputPer1kTokens: 0.0006 },
  maxOutputTokens: 8192,
});

// After call (from streaming done event)
context.set('_llm_usage', {
  model: 'gemini-2.5-flash',
  input_tokens: 1200,
  output_tokens: 450,
  is_byok: false,
});
```

## Developer Usage

```typescript tab="TypeScript"
import { BaseAgent } from 'webagents';
import { GoogleLLMSkill } from 'webagents/skills/llm';

const agent = new BaseAgent({
  name: 'gemini-agent',
  skills: [
    new GoogleLLMSkill({
      defaultModel: 'gemini-2.5-flash',
      thinking: 'high',
    }),
  ],
});

const response = await agent.run([
  { role: 'user', content: 'Explain quantum tunnelling.' },
]);
console.log(response.content);
```

```python tab="Python"
from webagents.agents.skills.core.llm.google.skill import GoogleAISkill

config = {
    "model": "gemini-2.5-flash",
    "thinking": "high",
}

skill = GoogleAISkill(config)
response = await skill.chat_completion(messages=[...])
```

## Adding a New Provider

1. Create an adapter in `webagents/typescript/src/adapters/your-provider.ts` implementing `LLMAdapter`.
2. Register it in `webagents/typescript/src/adapters/index.ts` via `getAdapter()`.
3. Create a thin skill wrapper in `webagents/typescript/src/skills/llm/your-provider/skill.ts` that calls `adapter.buildRequest()` and `adapter.parseStream()`, and sets `_llm_capabilities` / `_llm_usage` on the context.

Billing, media resolution, and tool pricing will work automatically through the PaymentSkill and MediaSkill hooks.
