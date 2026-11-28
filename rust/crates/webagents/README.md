# webagents

[![crates.io](https://img.shields.io/crates/v/webagents.svg)](https://crates.io/crates/webagents)
[![docs.rs](https://img.shields.io/docsrs/webagents)](https://docs.rs/webagents)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/robutlerai/webagents/blob/main/rust/LICENSE)

Device agents for the webagents protocol: typed, limit-checked hardware tools generated from register specs, with a `no_std` core.

A register spec (`webagents.regspec/1`) describes a part's registers the way its datasheet does, plus what a datasheet usually leaves to prose: units, scales, ranges, named mode values and the rules that keep the part safe ("duty never above the period", "sinusoidal commutation only while the period is between 20 and 200 us"). From one spec this crate gives you:

- **Tools for a model**: `<device>_read_<register>` and `<device>_set_<register>`, described as JSON Schema with limits and units, ready for any function-calling model.
- **A limit check on every write**: every write is checked against ranges, rules (including rules declared on other registers), host-granted confirmation and rate limits before the bus sees it. A refused write is reported with the reason and the limits in the part's units. Nothing is clamped silently; rounding to a field's resolution is reported in the result.
- **A typed Rust API** generated from the same spec for firmware written by hand, going through the same checks.

The engine needs no heap and no allocator, so the same tables run on a microcontroller, on a host gateway that bridges I2C/SPI/CAN/UART parts, and in tests.

```rust
use webagents::bus::MemBus;
use webagents::regspec::{load::load_json, CallCtx, Engine};

let dev = load_json(include_str!("../../../fixtures/pwm1.regspec.json")).unwrap();
let mut engine = Engine::new(dev, MemBus::<32>::new(), vec![0u64; dev.field_count]);
let mut out = [0u8; 512];
let ctx = CallCtx { now_ms: 0, confirmed: false };
let n = engine.call("pwm1_set_deadtime", r#"{"deadtime": 1.5}"#, &ctx, &mut out).unwrap();
println!("{}", core::str::from_utf8(&out[..n]).unwrap());
```

Generate the typed module with `robutler-device regspec gen spec.json > src/pwm1.rs` and include it with `include!("pwm1.rs")` (the file defines `mod pwm1`), or generate it from `build.rs` with `webagents::regspec::codegen::generate`.

## Gateway (feature `gateway`, Linux transports with `linux`)

A gateway puts parts that only have a bus on the agent network as one agent. It holds the platform's agent socket (UAMP over WebSocket, `/ws`), calls the model through the platform's `/llm` socket with each turn's own payment token, and runs the tool loop, with every tool call going through the part's envelope. Tool calls and their results are streamed to the chat as they happen, including refusals.

Transports: a serial line (or a PTY, a pseudo-terminal) with a small register line protocol, SocketCAN with one request and one answer frame per access, Linux i2c-dev and spidev through `embedded-hal`, and in-memory simulated parts. `robutler-device twin` serves a simulated part on a PTY or a CAN interface, so the whole path runs without hardware.

```toml
[portal]
url = "wss://robutler.ai/ws"
agent = "bench"
token_env = "WEBAGENTS_AGENT_TOKEN"   # the per-agent key is read from here, never written in the file

[[device]]
spec = "pwm1.regspec.json"
transport = { kind = "uart", path = "/dev/ttyUSB0", baud = 115200 }

[[device]]
spec = "sensor.regspec.json"
transport = { kind = "i2c", bus = "/dev/i2c-1" }
```

```sh
robutler-device gateway check gateway.toml
robutler-device gateway run gateway.toml
```

Instead of a token, a gateway can sign in as the agent with a key that never leaves the device: `robutler-device agent login @bench --portal https://robutler.ai` shows a code, the key's fingerprint and a URL; the owner allows it once on a phone; then `auth = "device"` in `[portal]` opens the agent socket with a one-use ticket fetched by a signed request.

A part that holds its own key proves which tools it offers: it signs its tool description (the exact bytes, with the same Ed25519 key and thumbprint as the sign-in), and with `device_key = "<its public key>"` in its `[[device]]` section the gateway verifies the signature, and that the list equals its own, before it offers any of the part's tools. The part also enforces the same limits itself, so a write that bypasses the gateway is refused at the part.

### A model on the board's accelerator

With `kind = "accelerator"` in `[llm]`, the gateway's model runs on the board's own accelerator instead of the platform's `/llm`: the gateway renders the conversation with the model's chat template, tokenizes it with the model's `tokenizer.json`, sends the ids over the accelerator link and parses the tool calls from the ids it gets back. The tool loop, the limits and the refusals are the same as with the platform's model, and no platform model call is made.

```toml
[llm]
kind = "accelerator"
link = "127.0.0.1:7100"              # the accelerator link
tokenizer = "qwen3/tokenizer.json"   # the model's own tokenizer file
```

The tokenizer and the Qwen3 chat template are checked byte for byte against vectors produced by Hugging Face `tokenizers` and `transformers`.

## Features

Only `std` is on by default.

| Feature | Default | What it adds | Without `std` |
| --- | --- | --- | --- |
| (none) | | Register specs as tables, the engine, exact decimal units, memory-mapped and in-memory buses | yes, no heap |
| `std` | yes | Spec loading from JSON with full validation, the Rust code generator, UAMP frames for hosts | no |
| `alloc` | | Heap-backed limit state | needs an allocator |
| `hal` | | `embedded-hal` 1.0 I2C and SPI register adapters; with `net`, a driver for the WIZnet W5500 Ethernet controller | yes |
| `auth` | | Agent sign-in by device authorization, device side: the Ed25519 device key, its RFC 7638 thumbprint and fingerprint, user codes, the RFC 9421 request signature the platform verifies, and signed tool descriptions. Checked byte for byte against the test vectors shared with the other SDKs | yes, no heap |
| `net` | | Networking for a device with its own connection: a WebSocket client over any `embedded-io` stream (plain TCP, `embedded-tls`, a co-processor link), a heap-free JSON frame reader, UAMP frame writers, an ESP-AT co-processor driver | yes, no heap |
| `model` | | A model on the device's own accelerator, host side: a byte-level BPE tokenizer read from a model's `tokenizer.json`, the Qwen3 chat template with tools, the tool-call parse, and the accelerator link | no |
| `gateway` | | The host gateway (tokio, WebSocket with rustls). Includes `model`, `hal`, `net` and `auth` | no |
| `linux` | | Serial, SocketCAN, i2c-dev and spidev transports for the gateway. Includes `gateway` | no |

Build for a microcontroller with `default-features = false`. The core builds
for `thumbv8m.main-none-eabihf` (Cortex-M33) and `riscv32i-unknown-none-elf`
(RV32I) with no allocator.

## Documentation

- API reference: [docs.rs/webagents](https://docs.rs/webagents).
- Guides: [robutler.ai/develop/webagents/rust](https://robutler.ai/develop/webagents/rust), with a quickstart, the gateway on a Raspberry Pi, microcontroller firmware, trust and local limits, the accelerator backend, and a reference of every refusal a tool call can return.
- Examples, one folder per use case: [rust/examples](https://github.com/robutlerai/webagents/tree/main/rust/examples).
- The command line tool is in the [`robutler`](https://crates.io/crates/robutler) crate: `cargo install robutler` installs `robutler-device`.

## License

MIT
