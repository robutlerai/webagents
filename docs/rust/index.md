---
title: Rust
description: The webagents and robutler crates - agents that run on a microcontroller or next to hardware, with typed tools generated from a register map and limits the part enforces itself.
---

# Rust

The Rust crates are for agents that run on hardware or next to it: a
microcontroller that holds its own connection, or a small Linux host (a
gateway) beside parts that only have a bus. They sit next to the
[TypeScript](../api/typescript.md) and [Python](../api/python.md) SDKs and
speak the same protocol, [UAMP](../protocols/uamp.md) (Unified Agent Messaging
Protocol), so a device agent is called like any other agent.

What is different about hardware is that a wrong tool call can break
something. So the starting point is not a function you write but the part's
register map, with its limits:

1. You describe the part in a **register spec** (`webagents.regspec/1`, a JSON
   file): the registers a datasheet lists, plus units, ranges, named mode
   values and the rules that keep the part safe.
2. The spec becomes **typed tools** a model can call: `<device>_read_<register>`
   and `<device>_set_<register>`, with the limits and units in their schema.
3. An **engine** checks every write against the spec before the bus sees it. A
   write outside a range or against a rule is refused with the reason, in the
   part's units. Nothing is clamped silently.
4. A **gateway** puts the parts on the agent network as one agent, or the
   firmware does it itself.

| Crate | What it is |
| --- | --- |
| [`webagents`](https://crates.io/crates/webagents) | The SDK. A `no_std` core (register specs, the engine, bus adapters, device sign-in, networking for a device with its own connection) and, behind features, the host gateway with its transports. |
| [`robutler`](https://crates.io/crates/robutler) | Robutler platform defaults, and the `robutler-device` command line tool. It re-exports `webagents`. |

## Install

The command line tool, on the machine where you write specs and on the host
that runs a gateway:

```bash
cargo install robutler
robutler-device --version
```

The binary is `robutler-device`. It has its own name so that it never shadows
the `robutler` command of the npm and PyPI packages; all of them can be
installed side by side.

The library, in a project of your own:

```bash tab="Host (std)"
cargo add webagents
```

```bash tab="Gateway on Linux"
cargo add webagents --features linux
```

```bash tab="Firmware (no_std)"
cargo add webagents --no-default-features --features hal,net,auth
```

Both crates build with a current stable Rust toolchain. `robutler-device`
builds on any host for the `regspec` commands; the `gateway`, `agent` and `twin` commands are built on
Linux only, because they use serial ports, SocketCAN, i2c-dev and spidev.

## Features

`webagents` is one crate from an 8-bit register on a microcontroller to a
gateway with TLS, so most of it is behind features. Only `std` is on by
default.

| Feature | Default | What it adds | Runs without `std` |
| --- | --- | --- | --- |
| (none) | | Register specs as tables, the engine, exact decimal units, the memory-mapped and in-memory buses | yes, and without a heap |
| `std` | yes | Loading a spec from JSON with full validation, the Rust code generator, UAMP frames for hosts | no |
| `alloc` | | Heap-backed limit state (`Vec<u64>`) | needs an allocator |
| `hal` | | `embedded-hal` 1.0 I2C and SPI register buses, and with `net` the W5500 Ethernet driver | yes |
| `auth` | | Device sign-in: the Ed25519 device key, signed requests (RFC 9421), signed tool descriptions | yes, no heap |
| `net` | | A WebSocket client over any `embedded-io` stream, UAMP frame writers, a JSON frame reader, an ESP-AT network co-processor driver | yes, no heap |
| `model` | | A model on the board's own AI accelerator, host side: tokenizer, chat template, the token link | no |
| `gateway` | | The host gateway: the agent socket, the model, the tool loop, device sign-in. Includes `model`, `hal`, `net` and `auth` | no |
| `linux` | | Serial port, SocketCAN, i2c-dev and spidev transports for the gateway. Includes `gateway` | no |

The `no_std` core builds for `thumbv8m.main-none-eabihf` (Cortex-M33) and
`riscv32i-unknown-none-elf` (RV32I) with no allocator.

## Where to go next

| You want to | Read |
| --- | --- |
| See a register map become tools and an agent, without hardware | [Quickstart](./quickstart.md) |
| Put parts on a serial line, I2C, SPI or CAN behind a Raspberry Pi | [Gateway on a Raspberry Pi](./gateway-raspberry-pi.md) |
| Run the agent in firmware on a microcontroller | [Microcontroller firmware](./mcu-firmware.md) |
| Understand what the model can and cannot do to the part | [Trust and local limits](./trust-and-limits.md) |
| Run the model on the board's own AI accelerator | [Accelerator backend](./accelerator-backend.md) |
| Look up an error a tool call returned | [Refusals reference](./refusals-reference.md) |
| Find a type or a function | The API reference page of this section, or [docs.rs/webagents](https://docs.rs/webagents) |

Complete examples, one folder per use case, are in
[`rust/examples`](https://github.com/robutlerai/webagents/tree/main/rust/examples).
