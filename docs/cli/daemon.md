---
title: Daemon
description: webagents daemon serves every agent in a folder, reloads them as their files change, and runs their cron schedules; webagents cron lists and runs them.
---

# Daemon

`webagents daemon` serves every agent under a folder over HTTP: this folder,
or the one `--watch` names. It reloads an agent when its file changes, lets
it go when the file is deleted, and runs any `cron:` schedules the files
declare. It runs in the foreground until Ctrl+C. The command, its flags and
which files it serves are the same in both SDKs.

An agent is a file named exactly `AGENT.md` or `AGENT-<name>.md`, anywhere
under the folder, except inside a tool's own directory: `.git`, `.hg`, `.svn`,
`.webagents`, `node_modules`, `.venv`, `venv`, `__pycache__`, `.tox`,
`.mypy_cache` and `.pytest_cache` are never searched. `AGENTS.md` and
`agent.md` are not agents. When two files declare the same `name:`, the one
read last is served and the daemon says so. A file that does not load (bad
YAML, an unknown key, a symbolic link) is not served, and the daemon says why.

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
`daemon.host` must be an address on this machine (`127.0.0.1`, `::1` or
`localhost`): a project's `.webagents/config.json` is often committed, so a
configured address never opens the daemon to the network. Any other value is
refused, naming the file it came from. To listen elsewhere, say so on the
command line: `webagents daemon --host 0.0.0.0`.

The model routes need a credential either way, and so do the routes that
register or remove an agent. Schedules come only from agent files: there is
no route that adds one. `GET /agents/cron` lists them as `{"schedules": [...]}`.

## Schedules

An agent declares its schedules in its own front matter, as a list. Each one
names what to run, when, and where the result goes:

```markdown
---
name: reporter
description: Reports on a schedule
cron:
  - name: daily-report
    schedule: "0 9 * * 1-5"
    timezone: Europe/Berlin
    prompt: Summarize yesterday's activity.
    deliver:
      file: reports/daily.md
  - name: queue
    every: 30m
    prompt: Check the queue and report anything stuck.
    deliver:
      webhook:
        url: https://hooks.example.com/queue
        timeout: 20
        retries: 2
  - name: watch
    every: 1h
    heartbeat: true
    deliver:
      chat: owner
---

You report on things.
```

| Key | What it takes |
| --- | --- |
| `name` | Letters, digits, `-` and `_`; unique in the file |
| `schedule` or `every` | Exactly one. `schedule` is a five-field cron expression (minute, hour, day of month, month, day of week) using `*`, numbers, ranges, steps and comma lists. `every` is a whole number of `s`, `m`, `h` or `d`, at least one minute |
| `prompt` or `heartbeat` | Exactly one. `prompt` is the message the agent is given. `heartbeat: true` runs the agent's standing instructions and delivers nothing when it answers `HEARTBEAT_OK` |
| `timezone` | An IANA name such as `Europe/Berlin`; UTC when absent |
| `enabled` | `false` keeps the schedule listed but never runs it |
| `deliver` | Exactly one target: `file`, `webhook` or `chat` |

The string form, `cron: "0 9 * * *"`, is refused with a sentence that shows
the list form, because it named no prompt and no target. A mistyped key in a
schedule is refused with the keys it takes.

### Where results go

- **`file: <path>`** appends `## <schedule>, <time>` and the reply to a file
  inside the agent's folder. A path outside the folder, or a symbolic link
  that leaves it, is refused.
- **`webhook: <url>`** (or `{url, timeout, retries}`) POSTs the run as JSON:
  `agent`, `schedule`, `kind`, `prompt`, `content` and `ran_at`. The default
  timeout is 15 seconds. A webhook that cannot be reached, or that answers
  408, 429 or a 5xx, is tried again up to `retries` times (default 3), waiting
  1, 2, 4 seconds and so on, at most 30. When the agent holds a signing
  identity the request is signed with its Web Bot Auth key; otherwise it goes
  unsigned and the run record says so.

  The key is the same one `serve` would give the agent, kept in the same
  place: `WEBAGENTS_KEYS_DIR`, else `~/.webagents/keys`, as
  `<name>.ed25519.jwk.json`, readable by you alone. It never sits in the agent
  folder, which is where git commits go, where copies of the folder go, and
  where the agent's file tools look. In a container, point `WEBAGENTS_KEYS_DIR`
  at a mounted secret. A key an earlier build left at
  `<folder>/.webagents/keys/` is moved into the store the next time the daemon
  loads the agent, with its thumbprint unchanged, and the daemon says so; if
  git ever tracked that file, it also says to rotate the key. The store names
  a key by agent name, so two folders whose agent files share a name would
  share one key: the daemon records the folder a key was made for beside it
  (`<name>.ed25519.origin.json`, no secret in it) and tells you when another
  folder asks for the same name. Give one of the two agents another name to
  give it a key of its own.
- **`chat: owner`** records the run in your chat with the agent on Robutler:
  the prompt as your message and the reply as the agent's, one chat per
  schedule. It needs `webagents login` and a published agent.

A reply that is empty, or a heartbeat that answers `HEARTBEAT_OK`, delivers
nothing.

### How runs behave

Runs execute as the agent's owner. A schedule never runs twice at once; a
restart does not run a slot twice; a slot missed while the daemon was down
runs once when it comes back. The runner keeps its state in
`.webagents/cron/<agent>.json` beside the agent file.

`--no-cron` turns the schedules off.

## webagents cron

```bash
webagents cron list                     # every schedule under this folder
webagents cron list --watch ./agents    # under another folder
webagents cron run reporter daily-report   # run one now and deliver it
```

`list` reads the folder's agent files as the daemon would, and the runner's
state, and writes nothing. Its columns are `AGENT`, `SCHEDULE`, `WHEN`, `NEXT`
and `LAST`. A file the daemon would refuse is named on standard error, and
`list` then exits 1.

`run` builds the one agent the way the daemon does, runs the schedule now as
the owner, delivers as configured, and records the run, so `LAST` shows it.
Both take `--json` before the command. In the chat, `/cron` shows the same
table for the current agent and `/cron run <name>` runs one after asking.

## Hosted Agents

An agent hosted on Robutler does not use this daemon. Its schedules are set
on the platform; see [Cron](../skills/platform/cron.md).
