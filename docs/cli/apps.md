---
title: Apps
description: Create, push, link and release your apps on Robutler from a terminal with robutler app, and release them from CI with a release token.
---

# Apps

`robutler app` manages your apps on Robutler from a terminal: create one, copy your files into its draft, connect its
source repository, release it, and look after its hooks, secrets and function runs.

The app commands are in the Node CLI. Install either package; both provide the same `robutler` command:

```bash
npm install -g robutler        # the product tool
npm install -g webagents       # the SDK, which ships the same command
```

They use the same sign-in and the same platform as the rest of the CLI (see [Publish](./deploy.md) for
`ROBUTLER_API_URL`, `platform.url` and profiles):

```bash
robutler login
```

## The life of an app

```bash
robutler app create my-app                       # a new private app, bound to this folder
robutler app push                                # copy the served files into its draft
robutler app link my-org/my-app --path app       # connect the source repository, finished in your browser
robutler app release --ref v1 --expect-commit <sha>
robutler app status
```

An app has a **draft** (what you are working on, visible to you) and a **release** (what everyone else runs). `push`
and a linked repository change the draft. Only a release changes what others run.

### A folder is one app

`create` writes `.robutler/app.json` in the folder: the app's id on the platform you are signed in to, and its
name. It holds no credential and can be committed. Every other command reads it, so you run them from the folder
with no id. To bind a folder to an app that already exists, use `robutler app use <app id>`. To act on another app
for one command, pass `--app <id>` or set `ROBUTLER_APP_ID`.

A repository that keeps its served files in a subfolder says so in `robutler.json` at its root:

```json
{ "sourcePath": "app" }
```

`push` and `link` read `sourcePath`, so neither needs the folder named again.

## Commands

| Command | What it does |
| --- | --- |
| `app create <name> [--template <t>]` | Makes a new private app from a starter and binds this folder to it |
| `app use <app id>` | Binds this folder to an app that already exists |
| `app link <owner/repo> [--branch <b>] [--path <dir>] [--no-follow]` | Opens the portal page that connects a GitHub repository as the app's source |
| `app push [dir] [--dry-run] [--keep-extra]` | Makes the draft hold the files of `dir`: writes what changed, removes what is gone |
| `app status` | Shows the draft, the release and the repository link |
| `app release [--ref <tag>] [--expect-commit <sha>] [--accept-security <hash>] [--dry-run]` | Releases the draft, or a tag of the linked repository |
| `app release --ci --ref <tag> --expect-commit <sha>` | Releases a tag with a release token, as a workflow does |
| `app hooks run <up\|down\|post>` | Runs one of the app's release hooks now |
| `app hooks status` | Shows what the release hooks last did |
| `app secrets set <NAME> --function <fn>` | Sets a function's secret from standard input |
| `app secrets unset <NAME> --function <fn>` | Removes it |
| `app secrets list --function <fn>` | Lists the secret names the function declares and whether each is set |
| `app logs --function <fn> [--limit <n>]` | Shows recent runs of a function: time, status, duration, error code |
| `app token create [--name <text>] [--all-apps] [--days <n>]` | Makes a release token for CI |
| `app token list`, `app token revoke <id>` | Lists and revokes your release tokens |

Every command takes `--json`: one JSON document on standard output, `{"ok": true, "data": ...}` or
`{"ok": false, "error": {"code", "message", "fix"}}`, with everything else on standard error.

Exit codes: `0` done, `1` failed, `2` the command line was wrong, `3` a release needs your acceptance, `4` the step
continues in your browser.

## Pushing files

`push` copies a folder into the draft. The folder is the one named, else `sourcePath` from `robutler.json`, else the
current one, and it must hold `widget.json`. Files that are already in the draft with the same content are not sent.
Files in the draft that are not in the folder are removed, unless you pass `--keep-extra`. Libraries the app pins are
left alone. `--dry-run` lists what would change and sends nothing.

`.git`, `.robutler` and `node_modules` are never pushed. A symbolic link stops the push.

## Linking a repository

`link` does not link anything itself. It opens the portal's page for it in your browser, where you, signed in as the
app's owner, install the read-only GitHub App on the repository. The terminal's sign-in is not part of that address.
When the page is done, `robutler app status` shows the link.

With a linked repository, a release names a tag: `robutler app release --ref v1`.

## Releasing

A release is a person's act. `robutler app release` refuses to run with an API key or an agent's key, and for an
account that is not a person's: sign in with `robutler login` in your own browser, with no `--token` and no
`WEBAGENTS_TOKEN`.

The command first shows what the release is: the tag and the commit it is on, and what it changes in what the app
may reach (a new host it may call, a browser permission, a new function). That change has a hash.

```text
Release of my-app on robutler.ai, by @ada
  tag:     v2 of my-org/my-app
  commit:  3f6c0d1e...
  This version lets the app reach more than the current release (1):
    - May call api.example.com
  security hash: 9b1f...
```

- If the release changes nothing in what the app may reach, it goes ahead directly.
- If it widens it, or it is the first release (which makes the app public), nothing is released and the command exits
  with code 3. Read the list, then run the command again with `--accept-security <hash>`. The hash must be the one
  that was printed. The command then asks for the release and prints an approval page: the release happens only
  when you approve it there, in your browser, signed in as yourself. The approval is bound to that hash, so what you
  approve is exactly what you read. The terminal alone never widens an app or makes it public, so a copied sign-in
  cannot do either.
- `--expect-commit <sha>` stops the release unless the tag is on that commit. Use the full 40 characters.
- `--dry-run` shows all of this and releases nothing.

## Releasing from CI

A release token lets a workflow release an app from a tag, and does nothing else. It names one app (or all of
yours), lasts at most 90 days, and can be revoked at any time.

```bash
robutler app token create --days 90 | gh secret set ROBUTLER_RELEASE_TOKEN
```

The token is written to standard output once, with the explanation on standard error, so it can be piped straight
into a secret store. The command keeps no copy and writes it to no file. Like a release, making a token is a
person's act. Where the platform keeps release tokens to your browser session, the command opens Settings,
Developer, Release tokens instead, prints nothing on standard output, and exits with code 4.

In the repository, the reusable workflow does the release when a tag is pushed:

```yaml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    permissions:
      contents: read
    uses: robutlerai/app-ci/.github/workflows/release.yml@v1
    with:
      app-id: 00000000-0000-4000-8000-000000000000   # from `robutler app status`
      environment: production
      credential: token
    secrets:
      ROBUTLER_RELEASE_TOKEN: ${{ secrets.ROBUTLER_RELEASE_TOKEN }}
```

The same release from a shell, for a runner that is not GitHub's:

```bash
ROBUTLER_RELEASE_TOKEN=... robutler app release --ci --app <app id> --ref v2 --expect-commit <sha>
```

A release from CI never widens what the app may reach and never makes an app public. In those cases, and for a tag
that is not newer than the current release, nothing is released: the owner is asked to approve it, and the command
prints where.

## Secrets

A function reads a secret it declares in the app's manifest. Set its value from standard input:

```bash
printf %s "$STRIPE_KEY" | robutler app secrets set STRIPE_KEY --function checkout
robutler app secrets set STRIPE_KEY --function checkout     # at a terminal: typed, not shown
```

A value on the command line is refused, because it would stay in the shell history and the process list. The value
is never printed and is not part of any error message. `secrets list` shows names only.

The commands need the app's agent, the one that runs its functions. `robutler app status` records it in
`.robutler/app.json` when the platform names it; otherwise pass `--agent <id>`.

## Output

Names, messages and every other string the platform sends are printed as data: escape sequences, control characters
and direction overrides are removed first.
