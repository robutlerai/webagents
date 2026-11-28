---
title: Microcontroller firmware
description: A device agent in firmware on a Cortex-M33 with a network co-processor - what the firmware owns, TLS on the device, the ESP-AT and W5500 drivers, buffers and measured sizes.
---

# Microcontroller firmware

A part with a microcontroller and a network connection does not need a
gateway. Its firmware can be the agent: it holds one outbound WebSocket to the
platform, runs its own turns, calls the model for them and keeps the part
inside its limits, all without an operating system or a heap.

[`rust/firmware/m33-agent`](https://github.com/robutlerai/webagents/tree/main/rust/firmware/m33-agent)
is a complete example: a device agent for a microcontroller (Cortex-M33) whose
network is an ESP-AT co-processor on a UART. It runs a full turn in emulation
(QEMU's `mps2-an505` board) and is the reference for this page.

## What the firmware owns

```text
                         microcontroller                             co-processor        platform
 part  <-- registers --  engine (limits, rules)
                         tool loop, UAMP frames
                         WebSocket client
                         TLS 1.3 session  ------- AT commands ------>  TCP bytes  <----->  /ws, /llm
```

The firmware owns everything that matters for trust:

- **The TLS session.** TLS runs on the microcontroller, over a raw TCP link.
  The co-processor only carries bytes; it never sees a key or plaintext.
- **The WebSocket.** One outbound connection to the platform's agent socket.
  The device runs no server and listens on no port.
- **The turns.** For each relayed turn the firmware calls the model itself,
  over a second short-lived connection, paid by that turn's own token.
- **The envelope.** Every tool call goes through the same engine as on a
  gateway, over tables generated from the part's register spec.

The crate is used with `default-features = false`:

```toml
[dependencies]
webagents = { version = "0.2", default-features = false, features = ["net", "auth"] }
```

| Module | Feature | What the firmware uses it for |
| --- | --- | --- |
| `webagents::regspec` | (always) | The engine, and the tables generated from the spec |
| `webagents::bus` | (always) | `Mmio`, the memory-mapped register bus |
| `webagents::net::esp_at` | `net` | The ESP-AT co-processor driver and its TCP links as byte streams |
| `webagents::net::ws` | `net` | The WebSocket handshake and framing |
| `webagents::net::uamp` | `net` | Writers for the UAMP (Unified Agent Messaging Protocol) frames it sends |
| `webagents::net::jsonv` | `net` | Reading values out of received frames without a parse tree |
| `webagents::net::w5500` | `net` + `hal` | A W5500 Ethernet controller instead of a co-processor |
| `webagents::auth` | `auth` | The device key and the signed tool description |

None of these allocate. Every buffer is supplied by the caller, which is why
the firmware's memory use is a list of arrays you can read in its source.

## Generate the part's tables

Firmware does not parse a spec. The tables are generated on the host and
compiled in:

```bash
robutler-device regspec gen pwm1.regspec.json > src/pwm1.rs
```

The file defines a module named after the device (`pub mod pwm1 { ... }`), so
it is included as it is rather than declared with `mod`:

```rust
include!("pwm1.rs");
```

The output depends only on the spec, so it can be committed and reviewed like
any source. It contains the device table as `static` data, and a typed API for
code written by hand:

```rust
// Generated: the table, a fixed-size engine type and a typed wrapper.
use pwm1::{Ctrl, Pwm1, DEVICE, FIELD_COUNT};

let bus = unsafe { webagents::bus::Mmio::new(0x4001_0000) };   // the part's register block
let mut pwm = Pwm1::new(bus);
let ctx = webagents::regspec::CallCtx { now_ms: uptime_ms(), confirmed: false };

// A typed write goes through the same limits as a model's tool call.
let ctrl = pwm.read_ctrl()?;
pwm.write_ctrl(ctrl.with_en(true), &ctx)?;

// The model's tool calls go through the engine directly.
let mut out = [0u8; 1024];
let n = pwm.engine.call("pwm1_set_deadtime", r#"{"deadtime": 1.5}"#, &ctx, &mut out)?;
```

The same code can run from a `build.rs` with
`webagents::regspec::codegen::generate` (a host-side function, feature `std`).

## The network co-processor (ESP-AT)

`webagents::net::esp_at` drives Espressif's AT firmware on an ESP32 or ESP8266
over a serial link: Wi-Fi join and up to five TCP links.

You supply the serial port as three functions, and the driver does the rest:

```rust
use webagents::net::esp_at::{AtPort, EspAt, Link};

struct Port(Uart);

impl AtPort for Port {
    fn write(&mut self, bytes: &[u8]) { /* send to the co-processor's UART */ }
    fn read_byte(&mut self) -> Option<u8> { /* a received byte, or None; never block */ }
    fn now_ms(&mut self) -> u32 { /* a millisecond clock */ }
}

let esp = core::cell::RefCell::new(EspAt::new(Port(uart)));
esp.borrow_mut().init("network name", "password")?;     // station mode, join, passive receive
esp.borrow_mut().open(0, "robutler.ai", 443)?;          // link 0: TCP to the platform
let link = Link { at: &esp, id: 0, timeout_ms: 60_000 };   // an embedded-io byte stream
```

Two properties are deliberate:

- **TLS is not delegated.** The co-processor has its own TLS commands; the
  driver does not use them. The link is plain TCP, and the microcontroller's
  TLS session runs over it, so the private state of the session never leaves
  the chip you control.
- **Receive is passive.** Data waits in the co-processor until the driver asks
  for it, so nothing arrives in the middle of a command's answer, and a small
  device never has to buffer what it did not ask for.

A `Link` is an `embedded-io` `Read + Write` stream, so a TLS session and the
WebSocket client run over it unchanged. Two links take turns on one
co-processor: the agent socket stays open on link 0 while each model call
opens and closes link 1.

## Ethernet instead (W5500)

`webagents::net::w5500` (features `net` and `hal`) drives the WIZnet W5500
Ethernet controller over SPI. The TCP/IP stack is in the chip, so the
microcontroller still holds no network stack, and TLS still stays on the
microcontroller.

```rust
use webagents::net::w5500::{Socket, W5500};

let chip = core::cell::RefCell::new(W5500::new(spi, now_ms));   // an embedded-hal SpiDevice, a millisecond clock
chip.borrow_mut().init(mac, ip, gateway, mask)?;                 // reset, check the chip, set the addresses
chip.borrow_mut().connect(0, [203, 0, 113, 10], 443, 50_000)?;   // socket 0: TCP to an IPv4 address
let socket = Socket { chip: &chip, sock: 0, timeout_ms: 60_000 };   // an embedded-io byte stream
```

The W5500 has no DNS resolver, so hosts are given as IPv4 addresses, and the
addresses are static: the driver does not run DHCP. `init` reads the chip's
version register and fails with `NotW5500` if it is not `0x04`, which catches
wrong wiring before anything else does.

## TLS on the device

The example uses [`embedded-tls`](https://crates.io/crates/embedded-tls): TLS
1.3 with the AES-128-GCM-SHA256 cipher suite, the server's certificate chain
verified to **one pinned root**, and the host name checked. The root is
compiled into the image (ISRG Root X2 by default; `AGENT_ROOT_CA` selects
another DER file at build time).

The firmware carries a copy of `embedded-tls` 0.19.0 with two marked changes: one fix
to how it walks a certificate chain that has extra certificates after the trust
anchor, and the `PartialEq` that fix needs. Both are described in
`vendor/embedded-tls/PATCHED.md`.

Two things are true of the emulated board only, and the firmware says so in
its source:

- It has no hardware random source. The generator is seeded from a fixed value
  and the timer, which is not secure. Real hardware seeds from its true random
  number generator.
- It has no clock, so certificate validity dates are not checked. Real
  hardware supplies one.

## A turn in firmware

1. The firmware joins the network, opens link 0, completes TLS and the
   WebSocket upgrade to `/ws`, and sends `session.create` with the agent's key.
2. An `input.text` frame arrives: a turn, with its session id, the
   conversation so far and a payment token.
3. The firmware builds one model request in a fixed buffer: its instructions,
   the relayed history (plain text turns only; tool rows from a history it did
   not write never reach the model), the turn so far, and the tool list written
   straight from the device table.
4. It opens link 1, completes TLS and the upgrade to `/llm`, sends the request
   with the turn's token and waits for the answer.
5. Each tool call in the answer is streamed to the chat, run through the
   engine and its result streamed back, refusals included. If there were
   calls, it goes back to step 3 with the results, for at most 6 rounds.
6. It sends the model's text and `response.done`.

Because every buffer is fixed, a turn that does not fit ends with a
`response.error` whose code is `device_limit` (the conversation is too long,
too many tool calls in one answer, or tool results too large) instead of
failing somewhere less clear.

## Memory

There is no allocator. The example's buffers are `static` arrays, so their
size is visible in the image and the stack stays small:

| Buffer | Bytes | Holds |
| --- | --- | --- |
| TLS receive, agent socket | 16,640 | One full TLS record |
| TLS transmit, agent socket | 4,096 | |
| TLS receive, model call | 16,640 | One full TLS record |
| TLS transmit, model call | 4,096 | |
| WebSocket receive | 16,384 | The `input.text` frame, kept for the whole turn |
| Model request | 24,576 | Instructions, history, the turn so far and the tool list |
| Model answer | 12,288 | The `response.done` frame |
| Turn so far | 8,192 | Assistant tool calls and tool results of this turn |
| Outgoing frame | 4,096 | One UAMP frame |
| **Total** | **107,008** | |

The model request is the largest because it carries the tool list: about 8.5
KB for the example device's 20 tools. A part with fewer registers needs less,
and the buffers are constants at the top of `src/agent.rs`.

The engine's own state is one `u64` per field of the device (24 fields, 192
bytes for the example) and nothing else.

## Sizes

Measured, not estimated: the numbers are the section sizes of the linked
images (`readelf -S`), built with `opt-level = "z"`, link-time optimization
and `panic = "abort"`.

The example firmware, for `thumbv8m.main-none-eabihf` (Cortex-M33):

| Image | Flash: `.text` | Flash: `.rodata` | RAM: `.bss` |
| --- | --- | --- | --- |
| Agent with TLS on the device | 246,012 | 28,796 | 107,064 |
| Agent over plain TCP (for a local mock) | 54,252 | 18,344 | 65,592 |
| Part behind a gateway (sub-device role) | 51,684 | 13,504 | 56 |

All numbers are bytes. TLS with certificate verification is most of the
agent's flash, and the two TLS sessions' record buffers are most of its RAM.
The sub-device role needs almost no RAM of its own because it streams its tool
description instead of holding it.

The engine alone, from `rust/probes/size-probe`: one tool call on the example
device's tables (12 registers, 24 fields) over memory-mapped registers, with
the refusal renderer, linked into an otherwise empty bare-metal image.

| What is linked | Target | `.text` | `.rodata` | `.data` + `.bss` |
| --- | --- | --- | --- | --- |
| Engine, device tables, refusal renderer | Cortex-M33 | 24,158 | 7,836 | 580 |
| The same, plus the tool list writer | Cortex-M33 | 27,410 | 8,748 | 580 |
| W5500 driver and WebSocket client, no engine | Cortex-M33 | 9,872 | 841 | 581 |
| Engine, device tables, refusal renderer | RISC-V (RV32I) | 46,044 | 7,932 | 580 |
| The same, plus the tool list writer | RISC-V (RV32I) | 51,620 | 8,844 | 580 |
| W5500 driver and WebSocket client, no engine | RISC-V (RV32I) | 15,408 | 841 | 584 |

The RAM column of the probe is its own argument and result buffers (576
bytes); the engine adds none beyond its limit state. RV32I has no multiply or
divide instructions, so the exact decimal arithmetic is larger there.

Flash is `.text` plus `.rodata`; `.bss` is RAM that is zero at start. The
firmware also prints its deepest stack use at the end of a run (the stack is
painted at boot and the high-water mark read back), so stack size is measured
too.

To measure your own part, build `rust/probes/size-probe` with your generated
tables for your target.

## The part behind a gateway

The same firmware has a second role, for a part that sits behind a gateway
instead of holding its own connection (`AGENT_ROLE=subdevice`). Its UART is
then the bus:

- it serves the register line protocol (`R`, `W`) that the gateway's `uart`
  and `tcp` transports speak;
- **every raw write goes through its own engine** (`Engine::write_word`), so a
  write outside the limits is refused by the part itself, whatever sent it;
- it answers `D` with its tool description, signed with its own device key, so
  the gateway can verify that the limits it enforces are the limits the part
  signed.

The description is never held in RAM. At boot the tool writer runs once into a
hash to produce the signature, and on each `D` it runs again straight into the
UART. The private key seed is built into the image from a file
(`AGENT_DEVICE_SEED_FILE`) and is never printed; only the public key and its
thumbprint go to the console, which is where you copy the `device_key` for the
gateway's configuration from.

See [Trust and local limits](./trust-and-limits.md#signed-tool-descriptions)
for what the gateway checks.

## Build and run in emulation

The firmware is its own cargo workspace. Its settings are fixed at build time,
through environment variables:

| Variable | Meaning |
| --- | --- |
| `AGENT_HOST`, `AGENT_PORT` | Where the platform is, as the co-processor dials it |
| `AGENT_TLS=1` | Run TLS on the device. Without it the link is plain TCP, for a local mock |
| `AGENT_TOKEN` | The per-agent key |
| `AGENT_ROOT_CA` | DER file of the TLS trust anchor (default: ISRG Root X2) |
| `AGENT_EXPECT` | `turn` (complete a turn) or `refused` (the platform must refuse the key); decides what the emulated run reports as a pass |
| `AGENT_ROLE=subdevice` | Build the part-behind-a-gateway role |
| `AGENT_DEVICE_SEED_FILE` | Sub-device: a file with the device key seed as 64 hexadecimal characters |
| `AGENT_CREATED` | Sub-device: the time stated in the signed tool description, in Unix seconds (default 0) |

One emulated turn against a mock of the platform, with no hardware and no
account:

```bash
cd rust
cargo build -p robutler
cargo build -p webagents --features gateway --example mock_platform

cd firmware/m33-agent
AGENT_HOST=127.0.0.1 AGENT_PORT=18080 AGENT_TOKEN=placeholder-not-a-real-key cargo build --release
ROBUTLER=../../target/debug/robutler-device MOCK=../../target/debug/examples/mock_platform \
  ./run-turn.sh target/thumbv8m.main-none-eabihf/release/m33-agent
```

`run-turn.sh` starts three processes: the mock platform, a software ESP-AT
co-processor (`robutler-device twin esp-at`, which answers the AT commands and
opens real TCP connections), and QEMU with the firmware's second UART
connected to that co-processor. The mock sends one turn in which the model
asks for a dead time; the firmware runs the call through its engine and
answers; the mock checks every frame.

It needs `qemu-system-arm` with the `mps2-an505` machine, and the
`thumbv8m.main-none-eabihf` target (`rustup target add
thumbv8m.main-none-eabihf`).

With TLS, against a real host:

```bash
AGENT_HOST=robutler.ai AGENT_PORT=443 AGENT_TLS=1 AGENT_TOKEN=... cargo build --release
robutler-device twin esp-at --listen 127.0.0.1:18091 &
LINK=127.0.0.1:18091 ./run-qemu.sh target/thumbv8m.main-none-eabihf/release/m33-agent
```

`run-subdevice.sh` runs the sub-device role the same way and checks three
things: the gateway verifies the part's signed description with the right key,
refuses it with a wrong key, and a raw write below a limit is refused by the
part while the register keeps its value.

## Moving to your hardware

The example targets an emulated board. On a real one you replace:

- `src/uart.rs` with your UARTs (the console and the co-processor link), and
  the timer behind `now_ms`;
- `memory.x` with your chip's memory map;
- the register block: `Mmio::new` at your part's real base address, instead of
  the RAM array the emulation uses;
- the random seed, from your hardware random source, and a clock for
  certificate dates;
- the generated tables, from your own spec;
- the network name and password passed to `EspAt::init`.

The agent logic in `src/agent.rs` works on `embedded-io` streams and does not
change.
