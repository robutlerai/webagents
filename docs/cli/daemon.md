---
title: Daemon
description: webagents daemon serves every agent in a folder, reloads them as their files change, and runs their cron schedules.
---

# Daemon

`webagents daemon` watches a folder for `AGENT*.md` files, registers what it
finds, reloads an agent when its file changes, runs any `cron:` schedules the
files declare, and serves every agent over HTTP. It runs in the foreground,
logging to the terminal, until Ctrl+C. The command and its flags are the same
in both SDKs.

```bash
webagents daemon                          # this folder, on 127.0.0.1:8765
webagents daemon --watch ./agents         # another folder
webagents daemon --port 8766 --no-cron    # another port, schedules off
```

To keep one running after the terminal closes, run `webagents daemon` under
whatever runs services on the machine: launchd, systemd, a container.

The chat does not need a daemon: it builds the agent in its own process. For
one agent over HTTP, `webagents serve` is simpler.

## The Address

Highest first:

1. `--port` and `--host` on the command
2. `daemon.port` and `daemon.host` from `webagents config`:
   `./.webagents/config.json`, then `~/.webagents/config.json`
3. `127.0.0.1:8765`

```bash
webagents config set daemon.port 8766
webagents daemon                          # listens on 127.0.0.1:8766
```

**The daemon listens on this machine only unless `--host` says otherwise.**
Its management routes are not authenticated, because it is built for the
machine it runs on. `daemon.host` must be an address on this machine
(`127.0.0.1`, `::1` or `localhost`): a project's `.webagents/config.json` is
often committed, so a configured address never opens the daemon to the
network. Any other value is refused, naming the file it came from. To listen
elsewhere, say so on the command line: `webagents daemon --host 0.0.0.0`.
Inference routes require a credential either way.

## Cron Scheduling

An agent declares its schedule in its own front matter:

```markdown
---
name: reporter
description: Generates the daily report
cron: "0 9 * * *"
---

Generate a summary of yesterday's activity.
```

The daemon picks that up when it registers the file. `--no-cron` turns the
schedules off. For a single run on a schedule without a daemon, have the
machine's scheduler call `webagents -a reporter -p "..."`.
