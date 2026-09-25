---
title: Publish
description: Send an agent from a folder to Robutler, keep the folder linked to the agent it publishes, and manage the keys on this machine.
---

# Publish

`webagents publish` sends the agent in the current folder to Robutler, which
hosts it from then on. The command, its flags and its messages are the same in
both SDKs.

```bash
webagents publish --dry-run     # see exactly what would be sent
webagents publish               # send it
```

`--dry-run` needs no sign-in. It prints the method, the route and the body,
which is the fastest way to check a file before you commit to anything. The
Python loader sends the agent with any inherited `WEBAGENTS.md` context merged
in; the TypeScript loader sends the file alone.

## Which Platform

`login`, `whoami`, `link` and `publish` talk to the same platform, resolved in
this order:

1. the `ROBUTLER_API_URL` environment variable
2. the `platform.url` config key
3. `https://robutler.ai`

`webagents login --url <portal>` signs in to that portal and sets `platform.url`
to it, so the commands after it go there too.

To work against a second platform, such as a local or staging deployment, give
it its own profile. A profile keeps its own sign-in and its own `platform.url`,
so it cannot publish with the other profile's credentials:

```bash
webagents --profile local login --url https://staging.example.com
webagents --profile local publish
```

`WEBAGENTS_PROFILE=local` is equivalent to `--profile local`. `webagents doctor`
names the portal the active profile signs in to.

## Linking

A platform username is minted once and can never be renamed, so `publish` never
creates an agent without asking. Linking a folder to an agent that already
exists removes the question entirely:

```bash
webagents link              # link to your agent with this file's name
webagents link other-name   # link to a different one
webagents link --show       # what this folder is linked to
webagents unlink            # forget the link; the agent itself is untouched
```

`link` matches the name against your agents' usernames (`alice.helper`, or its
`helper` part), then their display names. The link is two keys in
`./.webagents/config.json`, so it is per checkout rather than per machine: two
clones of the same repository can publish to different agents.

Once linked, `publish` updates that agent, and never sends `name`, so a rename
made in the web interface is not undone. Unlinked, it asks before creating one
and links the folder afterwards, so the next publish updates instead of minting
a second handle. `--yes` creates without asking, where nothing can answer, such
as CI.

### The Agent's API Key

Creating an agent returns its API key, and the platform returns it **once**.
`publish` stores it rather than printing it, so it does not end up in your
scrollback or a terminal recording, and prints the name it used.

You rarely need the value itself. An agent run from the linked folder finds its
key there by itself: the platform connection, the heartbeat, payments and the
other platform skills all look for it when nothing else is configured. Read it
back for a container or CI, where there is no keychain, and pass it as
`WEBAGENTS_AGENT_TOKEN`:

```bash
webagents secrets get AGENT_KEY_<NAME>          # confirms it is stored
webagents secrets get AGENT_KEY_<NAME> --show   # prints it, bare, for a script
```

`<NAME>` is the agent's full platform name in capitals, with dots and dashes as
underscores: `alice.my-agent` is stored as `AGENT_KEY_ALICE_MY_AGENT`.

If the store refuses the write, `publish` says so rather than continuing
quietly: at that point the key cannot be recovered and has to be regenerated
from the web interface.

## What Gets Sent

| `AGENT.md` | Platform field |
| --- | --- |
| `name:` | `name` (on creation only) |
| `description:` | `description` |
| the markdown body | `instructions` |
| `model:` | `model` |
| `intents:` | `intents` |
| `skills:` | `skills`, converted from a list to a mapping |

`skills:` is the one shape that changes. The file writes a list, the platform
takes a mapping, so `- memory` becomes `"memory": {}` and
`- mcp: {servers: [...]}` becomes `"mcp": {servers: [...]}`.

Fields the file does not set are omitted rather than sent empty.

## Keys on This Machine

`webagents secrets` manages model provider keys (and the agents' own API keys)
on **this machine**, in the system keychain (macOS Keychain, Linux Secret
Service, Windows Credential Manager), falling back to an owner-only file where
there is none. Both CLIs read the same store.

```bash
webagents secrets set OPENAI_API_KEY          # asks, with echo off
webagents secrets list
webagents secrets get OPENAI_API_KEY --show   # print one value
webagents secrets unset OPENAI_API_KEY
```

The value is never taken as an argument: an argument lands in your shell
history and in the process list. `secrets get` redacts unless you pass
`--show`.

The chat, `webagents -p` and `webagents serve` use the stored provider keys
wherever the shell exports none. An environment variable always wins over a
stored key, so storing one can never change the behavior of a shell that was
already exporting it; `secrets list` says which of the two each key comes from.

A keychain cannot be listed, so with one in use `secrets list` shows the keys
this CLI stored, and says so. A key written into the keychain by another tool
is readable with `secrets get` but does not appear in the list.

Under `--profile`, keys are stored separately, in both the keychain and the
fallback file.

> [!NOTE]
> These are keys for **your machine**, not for a published agent. The platform
> has no route yet that lets a CLI token set an agent's secrets: the only
> secret surface is per function and authenticates with a browser session.
> Set those in the web interface for now.
