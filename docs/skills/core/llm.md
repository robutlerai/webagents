---
title: LLM Skills
---
# LLM Skills

WebAgents provides a harmonized interface for interacting with various Large Language Model (LLM) providers. Whether you are using Google Gemini, OpenAI, Anthropic Claude, or xAI Grok, the configuration patterns for tools and reasoning capabilities remain consistent.

## Supported Providers

- **Google**: Native integration via `google-genai` SDK.
- **OpenAI**: Native integration via `openai` SDK.
- **Anthropic**: Native integration via `anthropic` SDK.
- **xAI (Grok)**: Integration via OpenAI-compatible API.

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

Enable internal reasoning (Chain of Thought) across supported models using a unified syntax. The framework automatically translates this to the provider's specific API parameters (e.g., `thinking_config` for Google, `reasoning_effort` for OpenAI o1, `thinking` block for Anthropic).

```yaml
skills:
  - google:
      model: "gemini-2.5-flash"
      thinking:
        enabled: true
        budget_tokens: 4096  # Token budget for thinking
        effort: "medium"     # Alternative to budget: low (1k), medium (4k), high (8k)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | `bool` | Activates reasoning/thinking mode. |
| `budget_tokens` | `int` | Maximum number of tokens to allocate for thoughts. |
| `effort` | `string` | Abstract effort level: `low`, `medium`, `high`. Used if `budget_tokens` is not set. |

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

## Developer Usage

When using the Python API directly:

```python
from webagents.agents.skills.core.llm.google.skill import GoogleAISkill

config = {
    "model": "gemini-2.5-flash",
    "thinking": {
        "enabled": True,
        "effort": "high"
    }
}

skill = GoogleAISkill(config)
response = await skill.chat_completion(messages=[...])
```
