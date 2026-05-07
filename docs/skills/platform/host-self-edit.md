# Host self-edit

Let an agent declare, update, and remove its **own** user-authored functions when chatting with its owner.

## When mounted

`HostSelfEditSkill` is auto-mounted by `_buildAgent` only when **both** are true:

1. `agent_configs.featureFlags.selfEdit === true`.
2. The current invocation has `ctx.auth.userId === agent.ownerId` (re-checked at every tool call, not just at build time).

The owner toggles the flag in the **Functions pane → Self-edit** checkbox; default is off.

## Tools exposed

| Tool                  | Behaviour                                                              |
| ---                   | ---                                                                    |
| `declare_function`    | Adds an entry to `agent_configs.functions`.                            |
| `update_function`     | Updates an existing entry; rejects host-other-agent edits.             |
| `remove_function`     | Removes the entry and detaches all consumer references.                |
| `add_to_skill`        | Adds a usage to `skills.cron` / `skills.custom_http` / `skills.custom_tools`. |
| `remove_from_skill`   | Removes a specific usage.                                              |

All tools call back into the same portal routes as the factory agent, with the `Function-Authoring-Surface: host` header that the portal validates against the agent id (host-edit cannot edit other agents).

## Secret-handling contract

The skill's `@prompt` reminder is explicit: **never accept secrets in chat**. When a function declares secret bindings, the LLM surfaces a `set_function_secret` action that opens the owner-only Secrets form; the value flows through the encrypted memory store, never through the chat transcript.

## Pricing

Self-edit tools are priced under `tools["fn:authoring"]` so iteration cost is metered the same way as user invocations.

## See also

- [Functions](functions.md)
- [Factory agent awareness](../../guides/factory-agent.md)
