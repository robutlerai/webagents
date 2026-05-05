---
title: Daemon
description: Background process manager for agents — start, stop, status, logs, expose, and cron scheduling.
---

# Daemon

The WebAgents daemon manages agent processes in the background — starting, stopping, and monitoring them.

> [!NOTE]
> `webagents daemon …` is **Python-only** today. In TypeScript, run agents directly with `npx webagents serve` or any process manager (PM2, systemd, Docker). Track parity in [internal/python-typescript-parity.md](../internal/python-typescript-parity.md).

## Commands

### Start

Start agents as background processes:

```bash
webagents daemon start
webagents daemon start --agent my-agent
webagents daemon start --port 8080
```

### Stop

Stop running agent processes:

```bash
webagents daemon stop
webagents daemon stop --agent my-agent
```

### Status

Check daemon status:

```bash
webagents daemon status
```

### Logs

View agent logs:

```bash
webagents daemon logs
webagents daemon logs --agent my-agent --follow
```

### Expose

Expose a local agent to the internet via tunnel:

```bash
webagents daemon expose --agent my-agent
```

This creates a public URL for the agent, useful for development and testing with the Robutler platform.

## Configuration

The daemon reads from `webagents.toml` or environment variables:

```toml
[daemon]
port = 8080
host = "0.0.0.0"
auto_restart = true
log_level = "info"
```

## Cron Scheduling

Agents can be scheduled to run periodically:

```typescript tab="TypeScript"
// Coming soon — track at https://github.com/robutlerai/webagents/issues
// The TS BaseAgent does not expose a `schedule` field today. Use a host
// scheduler (cron, systemd timer, Vercel Cron, GitHub Actions) to invoke
// `npx webagents run <agent> --prompt "..."` on your desired cadence.
```

```python tab="Python"
agent = BaseAgent(
    name="reporter",
    instructions="Generate daily report",
    schedule="0 9 * * *",  # 9 AM daily
)
```

The daemon's cron scheduler picks up scheduled agents and runs them at the specified intervals.
