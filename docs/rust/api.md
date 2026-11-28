---
title: Rust API reference
description: Where each part of the webagents and robutler crates is documented on docs.rs, one line per module.
---

# Rust API reference

The API reference of the Rust crates is generated from the source and
published on docs.rs. Every public item is documented there, with examples that
are compiled and run as tests.

| Crate | Reference | Install |
| --- | --- | --- |
| `webagents` | [docs.rs/webagents](https://docs.rs/webagents) | `cargo add webagents` |
| `robutler` | [docs.rs/robutler](https://docs.rs/robutler) | `cargo add robutler`, or `cargo install robutler` for the `robutler-device` command |

On docs.rs each module and item that needs a feature carries a label with the
feature's name. The feature table is on the [Rust overview](./index.md#features).

## `webagents`

### Always available (`no_std`, no heap)

| Module | What is in it |
| --- | --- |
| [`regspec`](https://docs.rs/webagents/latest/webagents/regspec/index.html) | The device table types: `Device`, `Register`, `Field`, `Rule`, `Bus`, `Access`, `Encoding`. The schema id `SCHEMA` |
| [`regspec::engine`](https://docs.rs/webagents/latest/webagents/regspec/engine/index.html) | `Engine`, which runs tool calls and enforces the envelope; `CallCtx`; `ToolError` and `Refusal`; `write_tools`, the tool list writer |
| [`regspec::dec`](https://docs.rs/webagents/latest/webagents/regspec/dec/index.html) | `Dec`, exact decimal numbers for units, and the count conversions with stated rounding |
| [`regspec::json`](https://docs.rs/webagents/latest/webagents/regspec/json/index.html) | `FlatObject`, a reader for flat tool arguments, and `BufWriter`, a writer into a fixed buffer |
| [`bus`](https://docs.rs/webagents/latest/webagents/bus/index.html) | `RegisterBus`, the two-method trait the engine drives; `Mmio` and `MemBus`; with `hal`, `I2cRegs` and `SpiRegs` |

### Host side of a spec (feature `std`, on by default)

| Module | What is in it |
| --- | --- |
| [`regspec::load`](https://docs.rs/webagents/latest/webagents/regspec/load/index.html) | `load_json`, which parses and validates a spec and reports every problem; the document types (`SpecDoc` and the rest), which are the reference of the JSON format |
| [`regspec::codegen`](https://docs.rs/webagents/latest/webagents/regspec/codegen/index.html) | `generate`, which turns a device table into a Rust module for firmware |
| [`uamp`](https://docs.rs/webagents/latest/webagents/uamp/index.html) | UAMP (Unified Agent Messaging Protocol) frames as `serde_json` values: builders for what an agent sends, `parse_inbound` for what it receives |

### On the device (`no_std`, no heap)

| Module | Feature | What is in it |
| --- | --- | --- |
| [`auth`](https://docs.rs/webagents/latest/webagents/auth/index.html) | `auth` | `DeviceKey`; request signing (`Request`, `SignedHeaders`); thumbprint, fingerprint and user codes; signed tool descriptions (`sign_tool_description`, `verify_tool_description`, `ToolDigest`) |
| [`net::ws`](https://docs.rs/webagents/latest/webagents/net/ws/index.html) | `net` | A WebSocket client over any `embedded-io` stream: `connect`, `send`, `recv` |
| [`net::uamp`](https://docs.rs/webagents/latest/webagents/net/uamp/index.html) | `net` | Writers for the UAMP frames a device sends, into any `fmt::Write` |
| [`net::jsonv`](https://docs.rs/webagents/latest/webagents/net/jsonv/index.html) | `net` | Reading values out of a JSON frame without a parse tree: `get`, `path`, `items`, `unescape` |
| [`net::esp_at`](https://docs.rs/webagents/latest/webagents/net/esp_at/index.html) | `net` | `EspAt`, a driver for an ESP-AT network co-processor, and `Link`, one of its TCP links as a byte stream |
| [`net::w5500`](https://docs.rs/webagents/latest/webagents/net/w5500/index.html) | `net` + `hal` | `W5500`, a driver for the WIZnet W5500 Ethernet controller, and `Socket`, one of its sockets as a byte stream |

### The gateway (host, `std`)

| Module | Feature | What is in it |
| --- | --- | --- |
| [`gateway`](https://docs.rs/webagents/latest/webagents/gateway/index.html) | `gateway` | `run`, the whole gateway; `GatewayConfig` and its sections, which are the reference of the configuration file; `build_devices`, `build_model`, `GatewayError` |
| [`gateway::device`](https://docs.rs/webagents/latest/webagents/gateway/device/index.html) | `gateway` | `ToolDevice`, one device behind the gateway; `Devices`, all of them; `ToolOutcome` |
| [`gateway::llm`](https://docs.rs/webagents/latest/webagents/gateway/llm/index.html) | `gateway` | `ModelBackend`, the model trait; `ModelRequest`, `ModelReply`, `ToolCall`; `PortalLlm`, the platform's model |
| [`gateway::accel`](https://docs.rs/webagents/latest/webagents/gateway/accel/index.html) | `gateway` | `AccelBackend`, the model on the board's AI accelerator |
| [`gateway::turn`](https://docs.rs/webagents/latest/webagents/gateway/turn/index.html) | `gateway` | `run_turn`, one turn of the tool loop; `TurnLimits`; the instructions the model is given |
| [`gateway::portal`](https://docs.rs/webagents/latest/webagents/gateway/portal/index.html) | `gateway` | `run_portal`, the agent socket with reconnects; `PortalConfig`, `PortalEvent` |
| [`gateway::signin`](https://docs.rs/webagents/latest/webagents/gateway/signin/index.html) | `gateway` | `SignIn`, the device sign-in client, and where the device key is stored |
| [`transport`](https://docs.rs/webagents/latest/webagents/transport/index.html) | `gateway` | `open` and `open_verified`, which turn a configured transport into a device; `verify_described` |
| [`transport::uart`](https://docs.rs/webagents/latest/webagents/transport/uart/index.html) | `gateway` | The register line protocol: `UartRegs` on the gateway side, `serve` for a simulated part |
| [`transport::can`](https://docs.rs/webagents/latest/webagents/transport/can/index.html) | `linux` | Registers over SocketCAN: `CanRegs` on the gateway side, `serve` for a simulated part |
| [`transport::sim`](https://docs.rs/webagents/latest/webagents/transport/sim/index.html) | `gateway` | Simulated parts: `RegModel`, `SimBus`, and I2C and SPI devices that hold a register model |
| [`transport::esp_at_twin`](https://docs.rs/webagents/latest/webagents/transport/esp_at_twin/index.html) | `gateway` | A software ESP-AT co-processor, for running firmware without the module |
| [`transport::w5500_sim`](https://docs.rs/webagents/latest/webagents/transport/w5500_sim/index.html) | `gateway` | A register-level simulator of the W5500 |

### A model on the board's accelerator (host, `std`)

| Module | Feature | What is in it |
| --- | --- | --- |
| [`model::bpe`](https://docs.rs/webagents/latest/webagents/model/bpe/index.html) | `model` | `Tokenizer`, a byte-level BPE tokenizer read from a `tokenizer.json` |
| [`model::chat`](https://docs.rs/webagents/latest/webagents/model/chat/index.html) | `model` | `render_qwen3`, the chat template with tools, and `parse_tool_calls` |
| [`model::accel`](https://docs.rs/webagents/latest/webagents/model/accel/index.html) | `model` | The accelerator's token link: `generate` on the host side, `serve_one` for a stand-in |

## `robutler`

| Item | What it is |
| --- | --- |
| [`robutler::webagents`](https://docs.rs/robutler/latest/robutler/index.html) | The whole `webagents` crate, re-exported |
| `DEFAULT_PORTAL_WS` | The platform's agent socket URL |
| `PORTAL_URL_ENV`, `AGENT_TOKEN_ENV` | The environment variables the Robutler SDKs read for the portal URL and the per-agent key |
| `resolve_portal_ws_url` | Resolves the agent socket URL from an explicit value, the environment or the default |

The `robutler-device` command's verbs are listed in the crate's
[documentation](https://docs.rs/robutler) and used throughout these pages:
`regspec check`, `regspec tools`, `regspec gen`, `regspec try`, `gateway
check`, `gateway run`, `agent login`, `agent logout` and `twin`.

## Reading the reference locally

```bash
cd rust
cargo doc --no-deps -p webagents --all-features --open
```

## Format references

Two formats are defined by Rust types, so their reference is the type's
documentation:

| Format | Types |
| --- | --- |
| Register spec, `webagents.regspec/1` (JSON) | [`regspec::load::SpecDoc`](https://docs.rs/webagents/latest/webagents/regspec/load/struct.SpecDoc.html), `RegisterDoc`, `FieldDoc`, `RuleDoc` |
| Gateway configuration (TOML) | [`gateway::GatewayConfig`](https://docs.rs/webagents/latest/webagents/gateway/struct.GatewayConfig.html), `PortalSection`, `LlmSection`, `DeviceSection`, `Transport` |
