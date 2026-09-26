---
title: Session
description: An agent keeps its conversations, on this machine and, for yours, on Robutler.
---

# Session

The session skill keeps an agent's conversations. It is the same skill in both
SDKs.

```yaml
skills:
  - session                        # conversations kept on this machine
  - session: {backend: robutler}   # and yours on Robutler too, as chats with the agent
```

## In the Chat

The chat always keeps its conversations on this machine, and `/resume`
continues them (see [Conversations](../../cli/session.md)). The session skill
decides whether they are also kept on Robutler.

With `backend: robutler`, each conversation you have with the agent in the
terminal is also your chat with it on Robutler: your messages as you, its
replies as the agent. It appears in your chat list on the web, and `/resume` on
another machine, or after you continued it on the web, lists it and picks it
up. Recording a turn does not make the agent answer again on Robutler and does
not notify you.

This needs two things: you signed in (`webagents login`) and the agent on
Robutler (`webagents publish`, which links the folder to it). Without either,
conversations stay on this machine and the chat says which one is missing.

## Served

Under `webagents serve` or `webagents daemon`, the skill keeps each verified
caller's conversation when the request names it:

```json
{
  "model": "openai/gpt-4o-mini",
  "messages": [{"role": "user", "content": "Plan the launch."}],
  "metadata": {"session_id": "launch-plan"}
}
```

- Your own requests, verified as the agent's owner, are kept with the chat's
  conversations, so `/resume` finds them.
- Anyone else's are kept under `callers/<hash>/` beside them, one namespace
  per caller: a session id only ever names a conversation of the caller who
  sent it.
- An anonymous caller's conversation is not kept, and neither is a request
  that names no session.

A request carries the whole conversation, as an OpenAI-style client sends it,
and what is kept is that conversation and the reply. Conversations that reach
the agent through Robutler are chats there already, so `backend: robutler`
changes nothing for a served agent.

## Where Conversations Live

```
~/.webagents/sessions/<folder>/<agent>/<id>.json                   # yours
~/.webagents/sessions/<folder>/<agent>/callers/<hash>/<id>.json    # other callers'
```

Files are readable only by you. Under `--profile <name>` they move to
`~/.webagents-<name>/`.

## In Code

```typescript tab="TypeScript"
import { SessionSkill } from 'webagents/skills/session';

const agent = new BaseAgent({ name: 'helper', skills: [new SessionSkill({ agentDir: process.cwd() })] });
```

```python tab="Python"
from webagents.agents.skills.local.session import SessionSkill

agent = BaseAgent(name="helper", skills={"session": SessionSkill({"agent_path": "."})})
```
