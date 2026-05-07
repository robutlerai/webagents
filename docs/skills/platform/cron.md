# Cron

Scheduled invocations of either a user-authored function or the host agent main loop.

`skills.cron.schedules[]` replaces the legacy `enabledTools.schedule` slot (one-shot pre-launch migration; no runtime back-compatibility).

## Entry shape

```yaml
skills:
  cron:
    schedules:
      - id: nightly_report           # stable id (used in metrics/audit)
        cron: 0 9 * * *              # 5-field POSIX cron (UTC by default)
        use: dailyReport             # function name from agent_configs.functions
        enabled: true                # default true
        timezone: America/Los_Angeles  # optional
      - id: poll_inbox
        cron: '*/5 * * * *'
        # `use` omitted ⇒ runs the host agent main loop
```

When `use` is set, the schedule invokes that function with a synthesised `FunctionContext` (`source.skill = 'cron'`, `schedule.plannedAt`, `schedule.firedAt`, `auth` populated from internal context). When `use` is omitted, the schedule runs the host agent's main loop just like the legacy `enabledTools.schedule`.

## Cloud vs local

- **Cloud**: the existing `agents-scheduled` Kubernetes `CronJob` (1-minute tick) hits `/api/internal/agents/scheduled`, which dispatches all due `skills.cron.schedules[]` for every active agent.
- **Local**: the `webagentsd` daemon runs the same 1-minute loop locally, reading `AGENT.md` and `./functions/` so `webagents dev` honours your schedules without any cloud setup.

## Limits

- Per-use min interval (default 1 minute).
- Per-agent concurrent-cron cap (plan-tier; see `PLAN_FUNCTION_LIMITS.schedules`).
- Cron miss SLO ≤ 0.1% — alert fires when the rate exceeds 0.1% over 30 minutes.

## See also

- [Functions](functions.md)
