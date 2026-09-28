---
title: Models
description: Which model an agent runs, with which key; local models through Ollama; any OpenAI-compatible endpoint; failover to other models; and what a conversation costs.
---

# Models

An agent names its model as `provider/model` and the provider's skill in its
file:

```yaml
---
name: writer
model: anthropic/claude-sonnet-4-6
skills:
  - anthropic
---
```

`webagents models` lists every provider, the model format it takes and what it
needs on this machine, and marks the ones that are ready here:

```bash
webagents models
```

## Keys

A provider is ready when its key is exported in the shell or stored with
`webagents secrets set`. The exported variable always wins. See
[Configuration](./configuration.md#environment-variables) for the variable of
each provider.

Without the key, and signed in with `webagents login`, the chat and
`webagents -p` run the same model through Robutler, paid from your credits.
An agent with no `model:` uses a provider you have a key for, or Robutler's
default model.

## Local Models with Ollama

`ollama/<model>` runs a model on an [Ollama](https://ollama.com) server, with
no key:

```yaml
---
name: local-helper
model: ollama/llama3.2
skills:
  - ollama
---
```

The server is `http://localhost:11434/v1` unless `OLLAMA_BASE_URL` names
another. Ollama is used only when the file names it; it is never picked for an
agent that names no model. `webagents models` marks it ready when the server
answers, and `webagents doctor` checks the agent's model: it says when nothing
answers at the address (start `ollama serve`) and when the model is not
pulled yet (`ollama pull llama3.2`). In the chat, `/status` shows the model
and the address it is reached at.

## Any OpenAI-Compatible Endpoint

`base_url` on a model skill's entry points it at another server that speaks
the OpenAI API, such as vLLM, LM Studio or a gateway:

```yaml
skills:
  - openai:
      base_url: http://localhost:8000/v1
```

`OPENAI_BASE_URL` does the same for every agent when the file names no
`base_url`.

## Failover

`fallback_models` lists models to try, in order, when the agent's `model` does
not answer:

```yaml
---
name: support
model: openai/gpt-4.1
fallback_models:
  - anthropic/claude-sonnet-4-6
  - ollama/llama3.2
skills:
  - openai
  - anthropic
  - ollama
---
```

The next model is tried when a request times out or gets no answer, or the
provider answers 408, 429, 500, 502, 503 or 504, and only before the model has
started to answer, so a reply is never repeated. Any other failure, such as a
bad request or a refused key, is reported as it is. Each switch leaves a note
in the transcript: `openai/gpt-4.1 did not answer (HTTP 503); trying
anthropic/claude-sonnet-4-6`. A caller of a served agent is never told the
provider's address, only that it could not be reached.

A fallback that cannot be built on this machine, for example one whose key is
not set, is left out and reported when the agent loads. The failover works the
same in the chat, `serve`, the daemon and schedules.

## What a Conversation Costs

The chat shows the conversation's tokens and, when it is known, their cost in
credits: in the footer after each reply, in `/status`, and in the line printed
on leaving.

- **Robutler's models**: the chat shows the cost the platform reports with a
  reply, as it is (`1.5k tokens, 0.0042 credits`), and tokens alone when a
  reply reports none.
- **Your own provider key** has no bill to read, so the cost is estimated from
  the provider's list price (one credit is one US dollar) and written with a
  tilde: `2.5k tokens, ~0.0006 credits`. Cache reads and long-context tiers are
  not counted.
- **A model the price table does not know**, and a local Ollama model, show
  tokens alone.

The cost of a conversation is kept with it, so `/resume` shows it again.
