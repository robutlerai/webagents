# robutler

[![crates.io](https://img.shields.io/crates/v/robutler.svg)](https://crates.io/crates/robutler)
[![docs.rs](https://img.shields.io/docsrs/robutler)](https://docs.rs/robutler)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/robutlerai/webagents/blob/main/rust/LICENSE)

Robutler platform defaults for [`webagents`](https://crates.io/crates/webagents) device agents, and the `robutler-device` command line tool.

```sh
cargo install robutler
robutler-device regspec check pwm1.regspec.json   # validate a register spec
robutler-device regspec tools pwm1.regspec.json   # the tool list a model sees
robutler-device regspec gen   pwm1.regspec.json > src/pwm1.rs
robutler-device agent login @bench --portal https://robutler.ai   # sign this device in as the agent (no browser on the device)
robutler-device gateway check gateway.toml       # load a gateway configuration and open every device
robutler-device gateway run   gateway.toml       # put the devices on the platform as one agent
robutler-device twin pwm1.regspec.json uart --pty   # a simulated part on a PTY, for trying it without hardware
```

The binary is named `robutler-device` so that it never shadows the `robutler`
command of the npm and PyPI packages.

## Commands

| Command | What it does | Hosts |
| --- | --- | --- |
| `regspec check <spec.json>` | Validates a register spec and lists every problem | all |
| `regspec tools <spec.json>` | Prints the tool list a model sees, as JSON | all |
| `regspec gen <spec.json>` | Prints the typed Rust module for the spec, for firmware | all |
| `regspec try <spec.json> <tool> <args-json>...` | Runs tool calls in order against an in-memory copy of the part at its reset values and prints each result or refusal | all |
| `gateway check <config.toml>` | Loads a gateway configuration, opens every device, then exits | Linux |
| `gateway run <config.toml>` | Puts the configured devices on the platform as one agent | Linux |
| `agent login <@agent> --portal <url> [--name <device name>]` | Signs this device in as the agent; the owner allows it on the platform's `/device` page and the key stays on this device | Linux |
| `agent logout <@agent> --portal <url>` | Signs this device out | Linux |
| `twin <spec.json> uart --pty` | Serves a simulated part on a new pseudo-terminal and prints its path | Linux |
| `twin <spec.json> uart <path>` | Serves a simulated part on a serial port | Linux |
| `twin <spec.json> can <iface> <node>` | Serves a simulated part on SocketCAN | Linux |
| `twin esp-at --listen <addr>` | An ESP-AT network co-processor on a TCP serial line, for emulated firmware | Linux |

The `gateway`, `agent` and `twin` commands use serial ports, SocketCAN, i2c-dev
and spidev, so they are built on Linux only.

## The library

The library re-exports `webagents` and adds the platform's defaults (the agent socket URL and the environment variables the other Robutler SDKs read), so one dependency is enough for a device agent on Robutler.

| Item | What it is |
| --- | --- |
| `robutler::webagents` | The whole [`webagents`](https://crates.io/crates/webagents) crate, re-exported |
| `DEFAULT_PORTAL_WS` | The platform's agent socket, `wss://robutler.ai/ws` |
| `PORTAL_URL_ENV` | The environment variables read for the portal URL, in order: `WEBAGENTS_PORTAL_URL`, `PORTAL_WS_URL` |
| `AGENT_TOKEN_ENV` | The environment variable that holds a per-agent key: `WEBAGENTS_AGENT_TOKEN` |
| `resolve_portal_ws_url` | Resolves the agent socket URL from an explicit value, the environment or the default |

On Linux the re-exported `webagents` is built with its `linux` feature (the
gateway and its transports); on other hosts with its default features. For a
`no_std` target, or to choose features yourself, depend on `webagents`
directly.

## Documentation

- API reference: [docs.rs/robutler](https://docs.rs/robutler) and [docs.rs/webagents](https://docs.rs/webagents).
- Guides: [robutler.ai/develop/webagents/rust](https://robutler.ai/develop/webagents/rust).
- Examples: [rust/examples](https://github.com/robutlerai/webagents/tree/main/rust/examples).

## License

MIT
