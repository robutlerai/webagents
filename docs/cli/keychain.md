---
title: Keychain dialogs on macOS
description: What the macOS keychain dialog is when webagents asks for a sign-in or a key, what to click, and how to run with no dialog at all.
---

# Keychain dialogs on macOS

On a Mac, the CLI keeps your Robutler sign-in and the keys you store with
`webagents secrets set` in the login keychain. macOS sometimes shows a dialog
before webagents can read one of them. This page says what that dialog is,
what to click, and how to run where nobody can click anything.

## What the dialog is

macOS asks before a program reads a keychain item that a different program
created. The dialog names the program and the item, for example:

```text
"node" wants to use your confidential information stored in
"webagents (TypeScript) cli" in your keychain.
```

The program is the interpreter webagents runs on, not webagents itself:

| CLI | Program in the dialog |
|-----|-----------------------|
| Python | `Python` for Homebrew and python.org installs, otherwise the interpreter's file name, such as `python3.12` |
| TypeScript | `node` |

Each item's name starts with `webagents`:

| Item | What it holds |
|------|---------------|
| `webagents (Python) cli` | The Python CLI's sign-in |
| `webagents (TypeScript) cli` | The TypeScript CLI's sign-in |
| `webagents (Python) providers`, `webagents (TypeScript) providers` | Keys stored with `webagents secrets set`, one item per key |
| `webagents (Python) <agent>`, `webagents (TypeScript) <agent>` | An agent's own secrets (the Secrets skill) |
| `webagents:cli`, `webagents:providers`, `webagents:<agent>` | Items from an earlier version of webagents |

Under `--profile <name>` the names carry the profile, as in
`webagents (Python) cli-<name>`.

### What to click

- **Always Allow** for an item whose name starts with `webagents`. macOS then
  lets that program read it without asking again.
- **Allow** reads it this once; macOS asks again next time.
- **Deny** for anything whose name does not start with `webagents`. webagents
  never asks for another app's items.

If the dialog asks for a password, it is your Mac login password (the
keychain's password). It goes to macOS, never to webagents.

Before a read that may raise the dialog, the CLI says so in the terminal, in
four lines:

```text
macOS is about to ask whether node may use "webagents (TypeScript) cli" in your keychain: node is what webagents runs on.
Choose Always Allow so it does not ask again.
If it asks for your Mac password, the password goes to macOS, never to webagents.
webagents only ever asks for items whose name starts with "webagents", so choose Deny for anything else.
```

It says them once per run, and only in a terminal.

## When it asks

Each CLI reads only the items it created, so the Python CLI never asks about
the TypeScript CLI's items, and the reverse. The cost is one sign-in, and one
`webagents secrets set` per key, in each CLI you use. Where one CLI finds
nothing and the other has a sign-in or keys, `whoami`, `secrets list` and
`doctor` say so.

macOS asks in three cases:

- **After an interpreter upgrade.** Homebrew's Python and node are signed
  ad hoc, so the keychain knows a program by a hash of its binary. After
  `brew upgrade`, the upgraded program is a different program to macOS, and
  it asks once for each item. Choose Always Allow.
- **Items from an earlier version.** An item named `webagents:...` is copied
  to the CLI's own name the first time the CLI finds none of its own, in a
  terminal. The old item stays in place. macOS asks when the old item was
  created by another program.
- **Your answer was Allow rather than Always Allow.** macOS asks again.

## Where nobody can answer

A process with nobody at the screen never waits on the dialog. `serve`, the
daemon, a script, a pipe and CI do not make a keychain read that may ask.
They use `WEBAGENTS_TOKEN` or the owner-only file when there is one.
Otherwise they stop with one sentence:

```text
macOS may ask before node can use "webagents (TypeScript) cli" in your keychain, and nothing here can answer it: run `webagents whoami` once in a terminal, then try again.
```

`webagents whoami` in a terminal reads each item such a run could not read,
so macOS asks there, once. Choose Always Allow and the run works next time.

On a machine with no keychain for the current user (a HOME that is not your
own, as on some CI runners) the CLI uses the owner-only file on its own.

### Headless and CI

- **`WEBAGENTS_TOKEN`**: a platform token in the environment is read before
  the keychain, so a signed-in command never touches the keychain.
- **`WEBAGENTS_SECRETS_BACKEND=file`**: no keychain at all. The sign-in and
  keys go to owner-only files (0600, in a 0700 folder) under
  `~/.webagents/secrets`, or `~/.webagents-<profile>/secrets` under a profile.
  Both CLIs read the same files.
- **`WEBAGENTS_SECRETS_REQUIRE_KEYSTORE=1`**: the opposite choice. Where there
  is no keystore, the CLI refuses to fall back to a file and says why.

## What macOS guarantees, and what it does not

- webagents cannot read another app's keychain item without a dialog that
  names the program. Nothing is read unless you allow it.
- There are two exceptions: items another tool created with the same
  interpreter (a script of yours that runs on the same Python, for example),
  and items whose owner made them readable by every app.
- macOS guards reading an item, not writing one: any program you run can add
  an item, or replace an item's value, without a dialog.
- Files are different. Any program you run can read your files. That is why
  an agent's shell commands run in a [sandbox](./sandbox.md) that cannot read
  the credential folders or the keychain file, and why the file tools stay in
  the agent's folder and refuse `.env` and `.webagents/`.

On Linux and Windows the system keystore (Secret Service, Credential Manager)
asks no per-program question, so none of the dialogs above appear there.

## Seeing and removing webagents' items

**Keychain Access.** Open Keychain Access, choose the login keychain, and
search for `webagents`. The Name column is the item, and the Account column
is the secret's name (`platform_token` for a sign-in, `OPENAI_API_KEY` for a
key). Values are shown only if you ask, and macOS asks for your password first.

**The terminal.** This shows an item's attributes, never its value, and asks
nothing:

```bash
security find-generic-password -s "webagents (Python) cli" -a platform_token
```

**Removing.** Let the CLI that created an item remove it:

```bash
webagents logout                        # this CLI's sign-in
webagents secrets remove OPENAI_API_KEY # one stored key
```

Each also retires the matching item from an earlier version, so it is never
copied back. The Python CLI removes that item when macOS lets it without a
dialog. The TypeScript CLI cannot tell whether removing it would make macOS
ask, so it never tries. Either way, an old item that stays is named, with how
to remove it: in Keychain Access, search for the item's name and delete it.

## Checking it

`webagents doctor` has a `keychain` line, and the chat's `/status` has the
same `Keychain` row. It never prints a value. It says where the sign-in and
keys live, which CLI's items are there and which program last used them, and
whether the next use will ask:

```text
  ✓ keychain   macOS keychain: 2 items named "webagents (Python) ...", last used by Python 3.14.2; the next use will not ask
```

After an upgrade:

```text
  ▲ keychain   macOS keychain: 2 items named "webagents (Python) ...", last used by Python 3.14.1; Python was upgraded since the last use: macOS will ask once
```

Which program last used each item is kept in `keychain.json` in the profile's
folder (`~/.webagents` or `~/.webagents-<profile>`), an owner-only file that
holds names, paths and versions, and no secret.
