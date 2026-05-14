---
title: Cron
description: Two first-class scheduling systems for AI agents — function-cron (sandboxed function fan-out) and agent-cron (full LLM runs in a background chat).
---

Robutler ships **two** scheduling systems. Neither is legacy — they target
different runtimes and bill differently.

| | **Function cron** | **Agent cron** ("Automatic Runs") |
|---|---|---|
| Config | `agent_configs.skills.cron.schedules[]` | `agent_configs.enabledTools.schedule` |
| Dispatcher route | `/api/internal/functions/cron-tick` | `/api/internal/agents/scheduled` → `/run` |
| CronJob | `portal-functions-cron-tick` | `portal-agent-scheduler` |
| Fan-out | Many bindings per agent | One schedule per agent |
| What fires | Sandboxed function (no LLM) | Full LLM run in background chat |
| Billing | Per-fire CPU / bytes / invocations against `PLAN_FUNCTION_LIMITS.daily.*` | Per-fire LLM cost against the **owner's** credit balance |
| UI | Functions pane → binding badge | Settings → Automatic Runs section |

## Function cron

Many per-agent bindings. Each schedule fires a sandboxed function through the
function-executor — headless, no LLM, no chat user. Use for ETL, reports,
notifications, polling.

### Entry shape

```yaml
skills:
  cron:
    schedules:
      - id: nightly_report           # stable id (metrics, audit, dedupe)
        cron: 0 9 * * *              # 5-field POSIX cron, UTC
        use: dailyReport             # REQUIRED — name from agent_configs.functions
        enabled: true                # default true
        description: Daily revenue report
```

`use` is required: function-cron always invokes a declared function. There is
no host-agent-main-loop shorthand — that capability is provided by **agent
cron** below.

### Owner UX

Owners NEVER see raw cron syntax. The factory chat asks for the frequency in
plain English ("every 15 minutes", "weekdays at 9 AM Pacific") and the agent
converts to a UTC 5-field expression internally. The Functions pane renders
each binding with a humanized label ("daily 9 AM UTC", "every 15 minutes").

### Cloud vs local

- **Cloud**: the `portal-functions-cron-tick` Kubernetes CronJob (1-minute
  tick) hits `/api/internal/functions/cron-tick`, which scans every active
  agent's schedules and dispatches the due entries to
  `FunctionRuntimeSkill.invoke()`.
- **Local**: the `webagentsd` daemon runs the same 1-minute loop locally so
  `webagents dev` honours your schedules without any cloud setup.

### Limits

- Per-schedule min interval: 60s (one tick).
- Per-agent active-schedule cap — read from Stripe product metadata
  (`function_cron_schedules`) via `lib/plans/cron-limits.ts`. Safe defaults:
  free=1, starter=5, pro=50.
- Runtime cost (CPU / bytes / invocations) is metered through the standard
  function quota buckets (`PLAN_FUNCTION_LIMITS.daily.*` in Redis).
- Cron miss SLO ≤ 0.1% — see `infrastructure/monitoring/prometheus/rules/functions.yaml`.

## Agent cron ("Automatic Runs")

One schedule per agent. Fires a full LLM run in a designated background
chat, with the prompt posted as if the owner typed it themselves. Use for
conversational summaries the owner reads in their DMs.

### Config shape

```yaml
enabledTools:
  schedule:
    enabled: true
    frequency: daily               # hourly | daily | weekly
    preferredTime: '09:00'         # local time, evaluated in timeZone
    preferredDay: 1                # weekly: 0=Sunday … 6=Saturday
    timeZone: America/Los_Angeles  # IANA tz database name
    backgroundPrompt: |
      Summarise yesterday's revenue …
```

### Billing & limits

- Each fire is billed to the **owner**: the dispatcher posts the prompt with
  `senderId = agent.ownerId`, so the agent's response is charged through the
  standard chat-billing path — identical to the owner typing the prompt
  themselves.
- Under-balance fires (`owner.totalBalance + owner.demoBalance < minimumBalance ?? $0.10`)
  are skipped with a top-up notice in the background chat. Balance is the
  primary self-stop.
- Plan-tier soft rate limit: `agent_cron_runs_per_day` (Stripe product
  metadata), defaults free=1, starter=24, pro=-1 (unlimited). `0` disables
  agent-cron entirely.

## See also

- [Functions](functions.md)
