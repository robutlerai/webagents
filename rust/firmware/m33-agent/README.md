# m33-agent

A device agent on a Cortex-M33 with an ESP-AT network co-processor. The firmware owns its TLS 1.3 session (`embedded-tls`, server chain verified to one pinned root, hostname checked), its one WebSocket to the platform's agent socket, the model calls for its turns (a second short-lived link per call, paid by the turn's own token), and the envelope of the part it drives. The co-processor only carries TCP bytes. The device runs no server.

Runs on QEMU `mps2-an505` (UART0 console, UART1 to the co-processor).

```sh
# one emulated turn against the mock platform (plain TCP)
AGENT_HOST=127.0.0.1 AGENT_PORT=18080 AGENT_TOKEN=placeholder-not-a-real-key cargo build --release
ROBUTLER=../../target/debug/robutler-device MOCK=../../target/debug/examples/mock_platform \
  ./run-turn.sh target/thumbv8m.main-none-eabihf/release/m33-agent

# TLS on the device against a real platform host
AGENT_HOST=robutler.ai AGENT_PORT=443 AGENT_TLS=1 AGENT_TOKEN=... cargo build --release
robutler-device twin esp-at --listen 127.0.0.1:18091 &
LINK=127.0.0.1:18091 ./run-qemu.sh target/thumbv8m.main-none-eabihf/release/m33-agent
```

Build-time settings: `AGENT_HOST`, `AGENT_PORT`, `AGENT_TLS=1`, `AGENT_TOKEN`, `AGENT_EXPECT` (`turn` or `refused`), `AGENT_ROOT_CA` (DER file of the trust anchor; default ISRG Root X2).

The emulated board has no hardware random source: the firmware seeds its generator from a fixed value and the timer, which is not secure and exists only for emulation. It has no clock either, so certificate validity dates are not checked. Real hardware seeds from its TRNG and supplies a clock.

`vendor/embedded-tls` is `embedded-tls` 0.19.0 with two marked changes, one fix to path building and the `PartialEq` it needs; see `vendor/embedded-tls/PATCHED.md`.
