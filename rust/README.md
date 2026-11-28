# webagents for Rust

[![webagents on crates.io](https://img.shields.io/crates/v/webagents.svg?label=webagents)](https://crates.io/crates/webagents)
[![robutler on crates.io](https://img.shields.io/crates/v/robutler.svg?label=robutler)](https://crates.io/crates/robutler)
[![webagents on docs.rs](https://img.shields.io/docsrs/webagents?label=docs.rs%2Fwebagents)](https://docs.rs/webagents)
[![robutler on docs.rs](https://img.shields.io/docsrs/robutler?label=docs.rs%2Frobutler)](https://docs.rs/robutler)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Device agents for the webagents protocol: agents that run on a microcontroller, or on a gateway next to parts that only have a bus. These crates sit beside the [Python](../python) and [TypeScript](../typescript) SDKs in this repository and speak the same protocol, [UAMP](../docs/protocols/uamp.md) (Unified Agent Messaging Protocol).

| Crate | What it is |
| --- | --- |
| [`webagents`](crates/webagents) | The SDK. `no_std` core: register specs with units, ranges and safety rules; a heap-free engine that turns them into model-callable tools and refuses any write outside the envelope; a JSON loader and a Rust code generator. Host gateway (feature `gateway`): devices on the agent network over UAMP, the model through the platform, serial/CAN/I2C/SPI transports on Linux. Agent sign-in by device authorization (feature `auth`, `no_std`). |
| [`robutler`](crates/robutler) | Robutler platform defaults and the `robutler-device` command line tool (`cargo install robutler`). |

```sh
cd rust
cargo run -p robutler -- regspec check fixtures/pwm1.regspec.json
cargo run -p robutler -- regspec tools fixtures/pwm1.regspec.json
cargo test --workspace --all-features
cargo run -p robutler -- gateway check examples/gateway/sim.toml
```

## Features of `webagents`

Only `std` is on by default. Build for a microcontroller with `default-features = false`.

| Feature | Default | What it adds | Without `std` |
| --- | --- | --- | --- |
| (none) | | Register specs as tables, the engine, exact decimal units, memory-mapped and in-memory buses | yes, no heap |
| `std` | yes | Spec loading from JSON with full validation, the Rust code generator, UAMP frames for hosts | no |
| `alloc` | | Heap-backed limit state | needs an allocator |
| `hal` | | `embedded-hal` 1.0 I2C and SPI register buses; with `net`, the W5500 Ethernet driver | yes |
| `auth` | | Device sign-in: the Ed25519 device key, signed requests (RFC 9421), signed tool descriptions | yes, no heap |
| `net` | | WebSocket client over any `embedded-io` stream, UAMP frame writers, a JSON frame reader, an ESP-AT co-processor driver | yes, no heap |
| `model` | | A model on the board's own AI accelerator, host side: tokenizer, chat template, the token link | no |
| `gateway` | | The host gateway: agent socket, model, tool loop, device sign-in (includes `model`, `hal`, `net`, `auth`) | no |
| `linux` | | Serial, SocketCAN, i2c-dev and spidev transports for the gateway (includes `gateway`) | no |

## Documentation and examples

- The guides are in [`docs/rust`](../docs/rust/index.md): a quickstart, the gateway on a Raspberry Pi, microcontroller firmware, trust and local limits, the accelerator backend, and the refusals reference.
- The API reference is on docs.rs: [`webagents`](https://docs.rs/webagents), [`robutler`](https://docs.rs/robutler).
- [`examples/`](examples) has one folder per use case, each with a README.

## Bare-metal builds

The `no_std` core builds for bare-metal targets with no allocator:

```sh
cargo build -p webagents --no-default-features --features hal,net,auth --target thumbv8m.main-none-eabihf
cargo build -p webagents --no-default-features --target riscv32i-unknown-none-elf
```

`probes/size-probe` links the engine and a generated device table into a bare-metal image per target, so flash and RAM figures are measured, not estimated. `firmware/m33-agent` is a complete Cortex-M33 device agent that runs a full turn on QEMU (`mps2-an505`) against the mock platform in `crates/webagents/examples/`.

## Shared with the other SDKs

- The agent sign-in and signed tool description vectors live in [`test-fixtures/agent-device/`](../test-fixtures/agent-device) at the repository root, one set for every SDK; `crates/webagents/tests/auth_vectors.rs` checks the Rust signer against them byte for byte.
- Releases follow the repository's [RELEASE.md](../RELEASE.md): a `rust-v<version>` tag publishes `webagents` and then `robutler` to crates.io, after the whole of CI passes on the tagged commit.
- Changes are listed in the repository's [CHANGELOG](../CHANGELOG.md), tagged **[rs]**.

## Licence

MIT, as the rest of the repository. Third-party material under `rust/`, not part of either published crate:

- `firmware/m33-agent/vendor/embedded-tls`: embedded-tls 0.19.0, Apache-2.0 (its own `LICENSE`), with two marked changes (`PATCHED.md`).
- `crates/webagents/tests/fixtures/qwen3-vectors.json`: token ids and chat-template renders produced from Qwen/Qwen3-0.6B's tokenizer and template (Apache-2.0) by Hugging Face `tokenizers` and `transformers`, used as test vectors.
