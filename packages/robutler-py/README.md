# robutler

The Robutler command line.

```bash
pip install robutler
robutler login
robutler
```

`robutler` opens a chat with your assistant. `robutler -p "..."` sends one prompt and prints the reply. Run
`robutler --help` for every option.

## What this package is

`robutler` is the product tool. It is built on [webagents](https://pypi.org/project/webagents/), the open SDK for
building and running AI agents, and this package holds no code of its own: it depends on exactly one version of
`webagents`, the one with the same version number, which installs the `robutler` command.

The commands that manage your apps (`robutler app ...`) are in the Node package: `npm install -g robutler`.

## Links

- Documentation: https://robutler.ai/docs/webagents
- Source: https://github.com/robutlerai/webagents
- Issues: https://github.com/robutlerai/webagents/issues

MIT licensed.
