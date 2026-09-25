---
title: Sandbox
description: Confine what an agent's shell commands can read, write and reach, enforced by the operating system.
---

# Sandbox

An agent file can declare what its commands are allowed to touch:

```markdown
---
name: researcher
skills:
  - shell
sandbox:
  preset: strict
  allowed_folders:
    - ./workspace
---
```

This is enforced by the operating system, not by reading the command. Seatbelt
(`sandbox-exec`) on macOS, bubblewrap (`bwrap`) on Linux. The restriction is
inherited by anything the command spawns, so it covers pipelines, `python -c`
and subshells, and a command cannot widen it from inside.

## Presets

| Preset | Writes | Reads | Network |
| --- | --- | --- | --- |
| `strict` | only `allowed_folders` | only `allowed_folders` and system paths | denied |
| `development` (default) | `allowed_folders` plus the working directory | anywhere | denied |
| `unrestricted` | as `development` | anywhere | allowed |

Reads are broad except under `strict`. Scoping reads is correct but expensive:
module resolution and toolchains traverse far more of the filesystem than you
would expect, so it is opt-in rather than the default.

`allowed_folders` entries are relative to the agent's directory unless
absolute. Every path is resolved through symlinks first, which matters on macOS
where `/tmp`, `/etc` and `/var` all point into `/private`.

## What it will not let you do

**Some paths stay read-only even inside an allowed folder**: `.git/hooks`,
`.git/config`, `.claude`, `.webagents`, `.vscode`, `.idea`, shell rc files,
`.gitconfig`, `.mcp.json` and `.env`. A command that could write those could
grant itself permissions for the next command, which would make the whole
declaration advisory.

**A temp directory is provided, and it is not your `$TMPDIR`.** Commands get
`$TMPDIR/webagents-sandbox`, and `TMPDIR` points there inside the sandbox.

## If the sandbox cannot run

It refuses, and the command does not run.

```
Access denied: bwrap not found in /usr/bin, /usr/local/bin, /bin
(install bubblewrap). The agent declared a sandbox, so the command was not run.
```

That is deliberate. A declaration that silently does nothing is the failure
this feature exists to prevent, so an unavailable backend is an error rather
than a downgrade. On Linux, install `bubblewrap`. Windows has no backend.

## What is not enforced

- **Network is on or off**, not per host. No operating system primitive
  expresses "allow github.com": Landlock handles ports only, and Seatbelt has
  no concept of a hostname. Filtering by domain needs a proxy outside the
  sandbox, which is not built yet.
- **`allowed_commands` is not a boundary.** It decides what runs without
  prompting. It cannot be enforcement, because it inspects the text a command
  was written as, while the shell expands, substitutes and chains before
  anything executes. `webagents doctor` will not claim otherwise.
- **`allowed_imports` does nothing.** An OS sandbox confines file and socket
  access, not Python `import` statements. `doctor` reports it as inert.

## Checking it

```bash
webagents doctor
```

reports the backend in use, or why there is none.

> [!NOTE]
> Every combination above is verified on macOS. The Linux path is built to the
> bubblewrap documentation and has had less exercise; report anything that
> behaves differently.
