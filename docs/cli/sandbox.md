---
title: Sandbox
description: Confine what an agent's shell commands can read, write and reach, enforced by the operating system.
---

# Sandbox

Every shell command an agent runs, and every SKILL.md script, is confined by
the operating system. The sandbox is on by default: an agent with `shell`
and no `sandbox:` block runs its commands under the defaults below. An agent
file can widen or narrow what its commands may touch:

```markdown
---
name: researcher
skills:
  - shell
sandbox:
  preset: development     # strict | development | off
  files:
    write: [.]            # the agent's folder; a private scratch folder is always added
    read: all             # development: all; strict: the write folders plus system folders
    deny: []              # more paths commands may never read, on top of the built-in list
  network:
    hosts: []             # host names, or the groups npm, pypi, github
    local: false          # connecting to local servers, and listening on a port
    sockets: []           # unix sockets by path, for example the ssh-agent socket
  env: []                 # variables a command may see; none that look secret by default
---
```

The block above is exactly what an agent with no `sandbox:` block gets. Every
key maps to something the engine enforces. The older flat spellings keep
working: `allowed_folders` is `files.write`, a bare `network:` list is
`network.hosts`, and `env_passthrough` is `env`. `allowed_commands` and
`allowed_imports` are unchanged (see "What is not enforced").

This is enforced by the operating system, not by reading the command. The
engine is srt (`@anthropic-ai/sandbox-runtime`), the sandbox runtime behind
Claude Code: Seatbelt (`sandbox-exec`) on macOS, bubblewrap (`bwrap`) on
Linux, plus a proxy outside the sandbox that admits only the hosts you list.
Both SDKs run it, so the same file is confined the same way under either CLI.
The restriction is inherited by anything the command spawns, so it covers
pipelines, `python -c` and subshells, and a command cannot widen it from
inside.

## What a confined command can reach

Measured with real srt on macOS, using the CLI's own settings:

| Declared | https to any host | a local server on 127.0.0.1 | listening on a port | a unix socket (ssh-agent, Docker) | direct DNS |
| --- | --- | --- | --- | --- | --- |
| nothing (the defaults) | blocked | blocked | blocked | blocked | blocked |
| `network: hosts: [example.com]` | example.com only | blocked | blocked | blocked | blocked (the proxy resolves listed hosts) |
| `network: local: true` | as above | allowed | allowed | blocked | blocked |
| `network: sockets: [/path/to.sock]` | as above | as above | as above | that socket (macOS) | blocked |

Only shell commands and SKILL.md scripts are confined. The agent's own tools
(the model, `rest_request`, `delegate`) and MCP servers run in the agent
process, unconfined.

## Presets

| Preset | Writes | Reads | Network |
| --- | --- | --- | --- |
| `development` (default) | `files.write` plus the agent's folder (and a scratch folder) | anywhere except the built-in denies | `network.hosts`, else none |
| `strict` | only `files.write` (and a scratch folder) | only `files.write`, the agent's folder and system paths | `network.hosts`, else none |
| `off` | anywhere | anywhere | anywhere |

Reads are broad except under `strict`. Scoping reads is correct but expensive:
module resolution and toolchains traverse far more of the filesystem than you
would expect, so it is opt-in rather than the default.

`files.write` entries are relative to the agent's directory unless absolute.
Every path is resolved through symlinks first, which matters on macOS where
`/tmp`, `/etc` and `/var` all point into `/private`. `files.read` is `all`,
or a list of folders: a list scopes reads to those folders, the write
folders, the agent's folder and the system folders, as `strict` does.
`files.deny` adds paths commands may never read.

## Network

```markdown
sandbox:
  network:
    hosts:
      - github
      - "*.pypi.org"
      - 127.0.0.1:8080
    local: true
```

Each host entry is a host name, a wildcard under a domain, a host with a
port, or a group. An empty or missing list means no network at all. A URL, a
path, a bare `*` or an address range is refused when the file loads. Traffic
goes through a proxy srt runs outside the sandbox (`HTTP_PROXY` and friends
are set inside, and Node is told to honour them), which is where the list is
checked; a host that is not listed gets a 403, and a program that ignores the
proxy is stopped at the socket. curl, git over https, npm, pip, Python's
`urllib` and Node's `fetch` all reach a listed host.

The groups expand to exactly these hosts:

| Group | Hosts |
| --- | --- |
| `npm` | registry.npmjs.org |
| `pypi` | pypi.org, files.pythonhosted.org |
| `github` | github.com, api.github.com, codeload.github.com, objects.githubusercontent.com, raw.githubusercontent.com |

`local: true` lets commands connect to servers on your own machine and listen
on a port. `sockets` lists unix sockets by path; srt honours the list on
macOS only, and on Linux the policy reports it as unenforceable and grants
nothing.

## Environment

Anything whose name contains `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `PASSWD`,
`CREDENTIAL`, `PRIVATE`, `AUTH`, `SESSION` or `COOKIE` is unset inside unless
the file lists it under `env`. A listed name that your shell does not export
is read from the `.env` files the CLI loads (`./.env`, then the profile
folder's `.env`); nothing else from those files reaches the command, which
cannot read them itself.

## The built-in denies

These apply under every confined preset, and no key removes them:

- reads of the credential folders in your home directory (`.ssh`, `.aws`,
  `.config/gcloud`, `.gnupg`, `.kube`, `.docker`, `.netrc`, `.npmrc`,
  `.pypirc`, `.webagents`) and of `Library/Keychains`: on macOS the keychain
  trusts the program that created an item, and every item this CLI stores was
  created by an interpreter a command can start, so the keychain file itself
  stays unreadable. Commands therefore cannot use keychain-backed credential
  helpers (`git credential-osxkeychain`, `gh`), which is the intent;
- reads of the other files where credentials are commonly kept: git's
  credential store (`.git-credentials`) and the GitHub CLI's folder
  (`.config/gh`), shell and REPL histories (`.zsh_history`, `.bash_history`
  and others), database and cloud logins (`.pgpass`, `.my.cnf`, `.azure`,
  `.oci`, `.config/doctl`, `.config/stripe` and others), registry tokens
  (`.cargo/credentials.toml`, `.gem/credentials`), other agents' logins and
  conversations (`.claude.json`, `.claude/projects`, `.codex/auth.json`), and
  browser profiles, whose cookies are signed-in sessions. The full list is
  `CREDENTIAL_DIRS` in the SDK;
- reads of every profile folder, `~/.webagents-<profile>`, which holds
  history, sessions, checkpoints and, with the file backend, the stored
  secrets;
- reads of every `.env` and `.env.*` file under your home directory, not only
  the agent's own: a command in one project cannot read another project's
  keys. Pass a variable to a command with `env:` instead;
- reads of `.env`, `.env.*` and `.webagents/` in the agent's folder and every
  write folder, wherever that folder is;
- writes to the escalation set below.

A command cannot send what it reads anywhere unless its sandbox lists hosts,
but what it prints goes back to the model. That is why these are denied for
reading, not only for sending.

On macOS these are patterns, so a file created during the command is covered
too. On Linux the entries that exist when the command starts are denied by
name, and the `.env` files are found by a short walk of your home directory:
it looks four folder levels down, skips tool caches such as `node_modules`
and `.cache`, and does not see a file created after it ran.

## Opting out

The opt-out is explicit, and the CLI says so when the agent loads, in the
status row, in `/sandbox` and in `webagents doctor`:

- `sandbox: off` in the agent file (`preset: unrestricted` means the same);
- `webagents --no-sandbox`, for one run, which applies to your own shell
  commands only. It works before or after the command name, as `--profile`
  does.

Under either, commands run with your permissions. Callers other than the
owner are refused shell commands unless they are confined, whatever the
opt-out, and SKILL.md scripts keep running confined.

## When a command is refused

When a confined command's output shows a refusal, the tool appends one
sentence naming the switch that opens it: `network.hosts` (or a group) for a
host the proxy refused or a name that could not be resolved, `network.local`
for a local connection or a listening port, `network.sockets` for a unix
socket, and `env` for a variable or a `.env` read.

In the interactive chat, a host the proxy refused is asked about by name,
read from srt's own log: allow it once (the command re-runs with the host for
that run), always (the host is written into the agent file's `network.hosts`,
where you can see and review it, and the command re-runs), or not at all.
`serve`, the daemon and `-p` never ask; they refuse with the sentence.

## What it will not let you do

**Some paths stay read-only even inside an allowed folder**: `.git/hooks`,
`.git/config`, `.claude`, `.webagents`, `.vscode`, `.idea`, shell rc files,
`.gitconfig`, `.mcp.json` and `.env`, and the agent's own control files:
`AGENT.md`, every `AGENT-<name>.md`, `WEBAGENTS.md`, `mcp.json` and the whole
`.agents/skills` folder. A command that could write those could grant itself
permissions for the next command (a wider `network:`, an open `access:`
block, a new schedule or a new agent for the daemon to serve), which would
make the whole declaration advisory.

On Linux, srt denies a file only when it exists as the command starts, so an
`AGENT-<name>.md` that a command creates is not covered there; on macOS the
pattern itself is denied. Keep that in mind for a daemon on Linux, which
serves every agent file that appears in its folder. The chat never reloads an
agent file unasked, and says when one changed during a reply.

**webagents' own install stays read-only too, when it sits in a folder
commands may write.** With a project-local virtual environment (`.venv`) or
`node_modules` in the agent's folder, that environment, the webagents
package, the sandbox engine and the node that runs it are write-denied, since
a command that rewrote them would run its own code, unconfined, on the next
command or the next `webagents` start. A different virtual environment or
`node_modules` in the same folder stays writable. `webagents doctor` and
`webagents sandbox setup` say when this applies (the `install` line); a
command cannot `pip install` or `npm install` into that environment, so
install webagents outside the folder (pipx, a global install, or a virtual
environment elsewhere) if commands need to.

**SKILL.md scripts always run confined.** A script from a
[SKILL.md skill](../skills/agent-skills.md) runs through srt with the agent's
`sandbox:` settings, or `strict` when the agent declares none; `off` is
refused for scripts, and `--no-sandbox` does not apply to them. Every skill
folder is read-only to it, its own included.

**A temp directory is provided, and it is not your `$TMPDIR`.** Commands get
`$TMPDIR/webagents-sandbox`, and `TMPDIR` points there inside the sandbox.
npm's cache is pointed there too, because `~/.npm` is not writable inside.

**`git init` and `git clone` fail inside an allowed folder**, because they
write `.git/config`, which stays read-only. Commits, fetches and pushes work.

**A misspelled key is an error.** `presets: strict` stops the file from
loading with a "did you mean" rather than running with the looser default.

## Where the engine comes from

It comes with webagents, in both SDKs:

- **TypeScript** depends on srt, `@anthropic-ai/sandbox-runtime` 0.0.77, and
  runs it with the node that runs the CLI.
- **Python** carries the same srt inside the package, the files npm
  publishes for 0.0.77 and its four dependencies, unmodified. It runs them
  with `node` from your PATH when that is 20.11 or later (srt's own
  minimum), and otherwise with the node that `pip install webagents` brought
  on macOS and Linux through the `nodejs-wheel-binaries` package. Nothing is
  downloaded when a command runs.

`WEBAGENTS_SRT_CLI` (the absolute path of an srt install's `dist/cli.js`)
and `WEBAGENTS_SRT_NODE` (a node binary) override either choice. Whichever
wins, the version must be exactly 0.0.77: srt drops settings it does not know
silently, so a version this SDK has not been checked against is refused
rather than trusted.

What the packages cannot bring is the machine's part:

- **Linux** needs `bubblewrap`, `socat` and `ripgrep`, and unprivileged user
  namespaces (Ubuntu 24.04 restricts them: `sudo sysctl -w
  kernel.apparmor_restrict_unprivileged_userns=0`, or an AppArmor profile
  that grants bwrap `userns`).
- **Inside a container**, bubblewrap needs to create namespaces, which the
  default Docker and Podman profiles refuse: run the container with
  `--security-opt seccomp=unconfined --security-opt apparmor=unconfined` (or
  `--privileged`), as a user other than root.
- **Native Windows** has no sandbox; run webagents inside WSL 2 (Windows
  Subsystem for Linux), where the Linux sandbox works.

```bash
webagents sandbox setup
```

checks all of this and installs nothing. It runs one confined `true`
through the engine, prints each check with what to do about the ones that
fail (on Linux, the install line for your distribution), and exits 1 when
shell commands would be refused. `--json` gives the same checks as one
document.

## If the sandbox cannot run

It refuses, and the command does not run, whether the file declared a
sandbox or the defaults apply. The refusal says what this machine lacks,
then the opt-out:

```
Access denied: bwrap, socat not found in /usr/bin, /bin, /usr/sbin, /sbin,
/usr/local/bin; srt needs bubblewrap, socat and ripgrep on Linux: `sudo
apt-get install bubblewrap socat`. `webagents sandbox setup` checks this
machine. To run commands with your permissions instead, pass --no-sandbox for
this run or put `sandbox: off` in the agent file. The command was not run.
```

That is deliberate. A sandbox that silently does nothing is the failure this
feature exists to prevent, so an unavailable backend is an error rather than
a downgrade.

On macOS, srt cannot start when `$TMPDIR` is very long (a Unix socket path
is limited to 104 bytes), for example inside a deeply nested scratch
folder. `webagents sandbox setup` says so; point `TMPDIR` at a short path,
such as `/tmp/wa`.

**Callers other than the owner only ever run confined.** The shell tool is
owner-only unless the file's `access:` block hands it to a group; a caller in
that group is refused when the agent has opted out (`sandbox: off`,
`--no-sandbox`) or the sandbox cannot run here. The owner may still run
unconfined.

## What is not enforced

- **A confined command is never refused for its name.** The kernel decides
  what it may touch, so `sleep`, `mkdir` or a test runner simply run. The one
  exception is a name the shell skill's own `blocked_commands` lists
  (`- shell: {blocked_commands: [nc]}` under `skills:`), which is refused
  either way.
- **With `sandbox: off` or `--no-sandbox`, a fixed list is the only gate.** A
  command may start with `ls`, `cat`, `grep`, `find`, `head`, `tail`, `wc`,
  `echo`, `date`, `pwd`, `which`, `whereis`, `git`, `npm`, `pip`, `python`,
  `python3`, `node`, `uvx`, `curl`, `wget`, `rg` or `fd`, plus the shell
  skill's `allowed_commands`, and never with `rm`, `rmdir`, `dd`, `mkfs`,
  `fdisk`, `kill`, `killall`, `pkill`, `shutdown`, `reboot`, `halt`, `su`,
  `sudo`, `chmod` or `chown`. Every command in a chain is checked. The list
  is not a boundary: it reads the text a command was written as, while the
  shell expands, substitutes and chains before anything executes
  (`echo $(id -un)` passes it and runs `id`). `webagents doctor` will not
  claim otherwise.
- **`allowed_commands` inside `sandbox:` gates nothing.** It is accepted and
  kept for reports; to widen the list above, name the commands under the
  shell skill instead.
- **`allowed_imports` does nothing.** An OS sandbox confines file and socket
  access, not Python `import` statements. `doctor` reports it as inert.

## Checking it

```bash
webagents doctor
```

reports the state and the engine in use (`development (default), enforced by
srt 0.0.77`, `strict (agent file), enforced by ...`, `off (--no-sandbox):
not confined; ...`), or why there is none, with the same fix
`webagents sandbox setup` gives. In the chat, `/status` shows the state and
`/sandbox` adds the effective folders, hosts and switches.

> [!NOTE]
> Every combination above is verified on macOS with srt 0.0.77. The Linux
> path runs in CI (bubblewrap, socat, ripgrep on Ubuntu) and has had less
> exercise by hand; report anything that behaves differently.
