---
title: Custom tools
description: Expose a user-authored function as an LLM-callable tool, complete with parameter schema and pricing metadata.
---

Expose a user-authored function as an LLM-callable tool.

## Entry shape

```yaml
skills:
  custom_tools:
    tools:
      - id: calculate
        name: calculate
        description: Evaluate a math expression and return the result.
        use: calculator
        parameters:
          type: object
          properties:
            expr: { type: string }
          required: [expr]
```

The tool appears in the LLM's tool list at the next agent build. When the model calls `calculate({ expr: "2+2" })`, the runtime invokes `calculator` with `ctx.toolCall = { name: 'calculate', params: { expr: '2+2' }, callId }` and returns the function's response payload directly to the model.

## Parameter validation

`parameters` is a JSON Schema. The runtime validates the model's call payload before invocation; on schema violation the tool returns `{ ok: false, error: { code: 'PARAMETERS_INVALID', detail: <ajv messages> } }` and never executes the function.

## Pricing

Add a `tools["fn:<functionName>"]` entry under `agent_configs.toolPricing.tools` to charge for tool calls:

```yaml
toolPricing:
  tools:
    "fn:calculator":
      perCall: 100              # 100 nanocents per call
      perUnit:
        cpu_ms: 1               # plus 1 nc per CPU-ms
        ingress_bytes: 0
        egress_bytes: 0
```

## See also

- [Functions](functions.md)
- [Tool pricing](../../payments/tool-pricing.md)
