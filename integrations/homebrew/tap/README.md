# Robutler Homebrew tap

[Homebrew](https://brew.sh) formulae for
[WebAgents](https://robutler.ai/develop/webagents), the SDK and command line
for building, serving and publishing AI agents.

## Install

```bash
brew install robutlerai/tap/webagents
```

This installs two commands:

- `webagents` chats with an agent, serves it over HTTP, runs a folder of
  agents with their schedules, and publishes it to Robutler.
- `robutler` opens the chat with the assistant that comes with WebAgents. It
  is `webagents -a robutler`.

The same package is also here under the name `robutler`, with the same two
commands:

```bash
brew install robutlerai/tap/robutler
```

Install one of the two, not both: they put the same commands in the same
place.

Use the full name, `robutlerai/tap/...`, as written. Homebrew trusts a formula
from a tap like this one when you install it by its full name, and refuses the
short name until then.

Then start with the
[Quickstart](https://robutler.ai/develop/webagents/quickstart):

```bash
mkdir hello && cd hello
webagents
```

## Upgrade and uninstall

Once installed, the short name works:

```bash
brew upgrade webagents
brew uninstall webagents
```

## What gets installed

Each formula installs the [`webagents`](https://www.npmjs.com/package/webagents)
package from npm and runs it on Homebrew's own Node 24, whichever `node` comes
first on your `PATH`. You do not need Node installed yourself.

On Linux, shell commands run in a sandbox that uses bubblewrap, socat and
ripgrep from the system's own package manager. `webagents sandbox setup` checks
the machine and prints the install line for what is missing.

## If `brew link` reports a conflict

Homebrew will not replace a `webagents` or `robutler` command it did not
install. If an earlier `npm install -g webagents` or `pip install webagents`
put one in Homebrew's `bin` folder, remove that install and link again:

```bash
npm uninstall -g webagents     # or: pip uninstall webagents
brew link webagents
```

## Issues and changes

Every file here is written from
[`integrations/homebrew`](https://github.com/robutlerai/webagents/tree/main/integrations/homebrew)
in the WebAgents repository when a version is published, so a change made in
this repository alone is overwritten by the next release. Please open issues
and pull requests in
[robutlerai/webagents](https://github.com/robutlerai/webagents).
