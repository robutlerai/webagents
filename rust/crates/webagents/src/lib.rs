//! # webagents
//!
//! Device agents for the webagents protocol. A device agent puts a piece of
//! hardware on the agent network with tools a model can call, and keeps the
//! hardware inside its limits whatever the model asks.
//!
//! The limits are written once, next to the register map, in a register spec
//! (`webagents.regspec/1`): units, ranges, named mode values, and the rules
//! that keep the part safe. Everything else is derived from that file, and
//! every layer that touches the bus enforces the same table.
//!
//! ```
//! # #[cfg(feature = "std")] {
//! use webagents::bus::MemBus;
//! use webagents::regspec::{load::load_json, CallCtx, Engine};
//!
//! // A spec becomes a device table, and the table becomes tools.
//! let dev = load_json(include_str!("../../../fixtures/pwm1.regspec.json")).unwrap();
//! let mut engine = Engine::new(dev, MemBus::<32>::new(), vec![0u64; dev.field_count]);
//! let ctx = CallCtx { now_ms: 0, confirmed: false };
//! let mut out = [0u8; 512];
//!
//! // Inside the limits: written, and the result says what was applied.
//! let n = engine.call("pwm1_set_deadtime", r#"{"deadtime": 1.5}"#, &ctx, &mut out).unwrap();
//! assert!(core::str::from_utf8(&out[..n]).unwrap().starts_with(r#"{"ok":true"#));
//!
//! // Outside them: refused with the reason, and the bus is never touched.
//! let refused = engine.call("pwm1_set_deadtime", r#"{"deadtime": 0.2}"#, &ctx, &mut out).unwrap_err();
//! let mut why = String::new();
//! refused.write_json(dev, &mut why).unwrap();
//! assert!(why.contains("outside its limits 0.5 to 25.5 us"));
//! assert_eq!(engine.bus.writes, 1);
//! # }
//! ```
//!
//! # Modules
//!
//! | Module | What it holds | Feature | `no_std` |
//! | --- | --- | --- | --- |
//! | [`regspec`] | Register specs, the engine that turns one into tools and refuses any write outside the envelope, exact decimal units | none | yes, no heap |
//! | `regspec::load`, `regspec::codegen` | The JSON loader with full validation, and the Rust code generator for firmware | `std` | no |
//! | [`bus`] | The two-method register bus the engine drives; memory-mapped and in-memory buses | none | yes, no heap |
//! | `bus::I2cRegs`, `bus::SpiRegs` | `embedded-hal` 1.0 I2C and SPI register adapters | `hal` | yes, no heap |
//! | `auth` | The device key, its thumbprint and fingerprint, signed requests (RFC 9421), signed tool descriptions | `auth` | yes, no heap |
//! | `net` | A WebSocket client over any `embedded-io` stream, UAMP frame writers, a JSON frame reader, an ESP-AT co-processor driver, a W5500 driver (with `hal`) | `net` | yes, no heap |
//! | `uamp` | UAMP (Unified Agent Messaging Protocol) frames as `serde_json` values, for hosts | `std` | no |
//! | `model` | A model on the device's own accelerator: tokenizer, chat template, the token link | `model` | no |
//! | `gateway` | The host gateway: devices on the agent network as one agent, the model, the tool loop, device sign-in | `gateway` | no |
//! | `transport` | How a gateway reaches a part: simulated parts and serial over TCP; with `linux`, serial ports, SocketCAN, i2c-dev and spidev | `gateway` | no |
//!
//! # Features
//!
//! | Feature | Default | Adds | Implies |
//! | --- | --- | --- | --- |
//! | `std` | yes | Spec loading from JSON, validation reports, code generation, host UAMP frames | `alloc` |
//! | `alloc` | no | Heap-backed limit state (`Vec<u64>`) | |
//! | `hal` | no | `embedded-hal` 1.0 I2C and SPI register buses; the W5500 driver when `net` is on too | |
//! | `auth` | no | Device sign-in and signed tool descriptions, device side | |
//! | `net` | no | Networking for a device with its own connection | |
//! | `model` | no | Tokenizer, chat template and link for a model on the board's accelerator | `std` |
//! | `gateway` | no | The host gateway (tokio, WebSocket with rustls) | `std`, `model`, `hal`, `net`, `auth` |
//! | `linux` | no | Serial, SocketCAN, i2c-dev and spidev transports for the gateway | `gateway` |
//!
//! # On a microcontroller
//!
//! With `default-features = false` the crate is `#![no_std]` and needs no
//! allocator: the engine, the buses, `auth` and `net` use only buffers the
//! caller supplies. Firmware compiles the tables that
//! `robutler-device regspec gen` emits from the spec (or `regspec::codegen`
//! from a `build.rs` on the host) instead of parsing JSON, and runs the same
//! [`regspec::Engine`] over them, so a write that bypasses a gateway is
//! refused by the part itself. The core builds for `thumbv8m.main-none-eabihf`
//! (Cortex-M33) and `riscv32i-unknown-none-elf` (RV32I).
//!
//! # On a host
//!
//! The `gateway` feature puts parts that only have a bus on the agent network
//! as one agent; `gateway::run` is the whole program and the `robutler` crate's
//! `robutler-device` command wraps it.
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod bus;
pub mod regspec;

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod uamp;

#[cfg(feature = "net")]
#[cfg_attr(docsrs, doc(cfg(feature = "net")))]
pub mod net;

#[cfg(feature = "auth")]
#[cfg_attr(docsrs, doc(cfg(feature = "auth")))]
pub mod auth;

#[cfg(feature = "model")]
#[cfg_attr(docsrs, doc(cfg(feature = "model")))]
pub mod model;

#[cfg(feature = "gateway")]
#[cfg_attr(docsrs, doc(cfg(feature = "gateway")))]
pub mod gateway;
#[cfg(feature = "gateway")]
#[cfg_attr(docsrs, doc(cfg(feature = "gateway")))]
pub mod transport;
