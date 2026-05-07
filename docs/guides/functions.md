# Functions walkthrough

This guide takes you from zero to a function-as-tool in five minutes.

## 1. Declare a function

```bash
webagents fn new calculator --runtime js-v1
```

Edit `./functions/calculator.js`:

```js
/**
 * @robutler-function
 * @runtime js-v1
 * @entrypoint handler
 * @description Evaluate a math expression
 */
export default async function handler(ctx) {
  const { expr } = ctx.toolCall.params;
  // Lazy-import a tiny safe-eval; obviously gate this for production.
  const result = Function('"use strict"; return (' + String(expr).replace(/[^\d+\-*/().\s]/g, '') + ')')();
  return { ok: true, result };
}
```

## 2. Declare it in `AGENT.md`

```yaml
---
functions:
  calculator:
    description: Evaluate a math expression
    permissions: { kv: ro }
    limits: { wallMs: 1000, cpuMs: 50, memoryMb: 32 }
skills:
  custom_tools:
    tools:
      - id: calc_tool
        name: calculate
        description: Run a math expression and return the result.
        use: calculator
        parameters:
          type: object
          properties: { expr: { type: string } }
          required: [expr]
---
```

## 3. Deploy

```bash
webagents fn deploy --agent my-agent
```

(or hit `Save` from the Functions pane in the portal UI).

## 4. Use it

In a chat with `@my-agent`, ask "what is 7 * 8?". The model picks up the new `calculate` tool from the system prompt (injected by `CustomToolsSkill.@prompt`), calls it, and the response surfaces back in the conversation.

## See also

- [Functions](../skills/platform/functions.md)
- [Custom tools](../skills/platform/custom-tools.md)
- [Host self-edit](../skills/platform/host-self-edit.md)
- [REST API — Functions](../api/functions.md)
