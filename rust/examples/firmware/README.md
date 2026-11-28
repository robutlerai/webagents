# Microcontroller firmware

The firmware example is a cargo workspace of its own, because it builds for a
bare-metal target with its own profile:
[`rust/firmware/m33-agent`](../../firmware/m33-agent).

It is a complete device agent for a microcontroller (Cortex-M33) with an ESP-AT
network co-processor: its own TLS 1.3 session, one WebSocket to the platform,
the model calls for its turns, and the envelope of the part it drives, with no
heap. It runs a full turn on QEMU (`mps2-an505`).

```sh
# from rust/
cargo build -p robutler
cargo build -p webagents --features gateway --example mock_platform

cd firmware/m33-agent
AGENT_HOST=127.0.0.1 AGENT_PORT=18080 AGENT_TOKEN=placeholder-not-a-real-key cargo build --release
ROBUTLER=../../target/debug/robutler-device MOCK=../../target/debug/examples/mock_platform \
  ./run-turn.sh target/thumbv8m.main-none-eabihf/release/m33-agent
```

It needs the `thumbv8m.main-none-eabihf` target and a `qemu-system-arm` that
has the `mps2-an505` machine.

Related:

- [`rust/probes/size-probe`](../../probes/size-probe) links the engine and a
  generated device table into a bare-metal image, so flash and RAM are measured
  per target.
- The [Microcontroller firmware](../../../docs/rust/mcu-firmware.md) page
  explains what the firmware owns, its buffers and its sizes.
