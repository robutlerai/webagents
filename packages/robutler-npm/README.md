# robutler

The Robutler command line.

```bash
npm install -g robutler
robutler login
robutler
```

`robutler` opens a chat with your assistant. `robutler -p "..."` sends one prompt and prints the reply.

It also manages your apps on Robutler:

```bash
robutler app create my-app      # a new private app, bound to this folder
robutler app push               # copy the served files into its draft
robutler app link owner/repo    # connect its source repository (finished in your browser)
robutler app release --ref v1   # release a tag; a change in what the app may reach is shown first
robutler app status
```

Run `robutler --help` and `robutler app --help` for every command and option.

## What this package is

`robutler` is the product tool. It is built on [webagents](https://www.npmjs.com/package/webagents), the open SDK
for building and running AI agents, and this package holds no code of its own: it depends on exactly one version of
`webagents`, the one with the same version number, and runs the `robutler` command that version ships.

If you build agents, install `webagents` instead: it installs the `webagents` command and the same `robutler`
command. Install one of the two globally, not both, since each provides `robutler`.

## Links

- Documentation: https://robutler.ai/docs/webagents
- Source: https://github.com/robutlerai/webagents
- Issues: https://github.com/robutlerai/webagents/issues

MIT licensed.
