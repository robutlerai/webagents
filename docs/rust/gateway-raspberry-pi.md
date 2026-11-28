---
title: Gateway on a Raspberry Pi
description: Put parts on a serial line, I2C, SPI or CAN behind a Raspberry Pi as one agent - install, the configuration file, checking it, device sign-in and running it as a service.
---

# Gateway on a Raspberry Pi

Most parts have no network. They have a bus: a serial line, I2C, SPI or CAN. A
gateway is a small Linux host next to them that holds the connection to the
platform, runs the model loop and enforces each part's limits. To the rest of
the agent network it is one agent whose tools are the parts' registers.

```text
platform  <-- agent socket (UAMP, the agent messaging protocol, over WebSocket) -->  gateway on the Pi  -- serial / I2C / SPI / CAN -->  parts
                                                            |
                                           every tool call is checked against
                                           the part's register spec first
```

This page uses a Raspberry Pi running a 64-bit Linux (aarch64). Nothing in the
gateway is specific to the Pi: any Linux host with the same device nodes works
the same way.

## Install

Install on the Pi itself, so the binary is built for it. It needs the Rust
toolchain and a C compiler (one of the dependencies builds its cryptography
from C and assembly):

```bash
sudo apt install build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install robutler
robutler-device --version
```

`cargo install robutler` builds the `robutler-device` command with every Linux
transport. There is nothing else to install: the serial, I2C, SPI and CAN
transports use the kernel's own interfaces.

## Wire up and enable the buses

Each transport opens a device node. On Raspberry Pi OS the buses are switched
on with `sudo raspi-config` (Interface Options), and a user needs to be in the
group that owns the node.

| Bus | Device node | Group | Notes |
| --- | --- | --- | --- |
| Serial port on GPIO 14 and 15 | `/dev/serial0` | `dialout` | Enable the serial port and disable the login shell on it, or the console will talk to your part |
| USB serial adapter | `/dev/ttyUSB0`, `/dev/ttyACM0` | `dialout` | Nothing to enable |
| I2C on GPIO 2 and 3 | `/dev/i2c-1` | `i2c` | Enable I2C |
| SPI, chip select 0 | `/dev/spidev0.0` | `spi` | Enable SPI |
| CAN | `can0` (a network interface) | | Needs a CAN interface board and its overlay; bring the interface up with its bit rate before starting the gateway |

```bash
sudo usermod -aG dialout,i2c,spi $USER     # then log out and in again
sudo ip link set can0 up type can bitrate 500000
```

## Describe each part

Every part needs a register spec. The [Quickstart](./quickstart.md) writes one
and tries it without hardware; do that first, because a spec that
`robutler-device regspec check` accepts and `regspec try` behaves well on is
most of the work.

The spec's `bus` section must match the transport you will use for I2C and
SPI, because the byte layout comes from it:

```json
"bus": { "kind": "i2c", "address": "0x48", "addr_bytes": 1, "data_bytes": 2, "big_endian": true }
```

| Key | Meaning | Default |
| --- | --- | --- |
| `address` | I2C only: the 7-bit address | required |
| `addr_bytes` | Bytes of register address on the wire, 1 to 4 | 1 |
| `data_bytes` | Bytes of register value on the wire, 1 to 4. Also the default register width | 1 |
| `big_endian` | Whether the value is sent most significant byte first | true |

For serial and CAN the `bus` section is not used for framing, and any kind
loads.

## The configuration file

One TOML file describes the agent and its parts. This one has a motor driver
on the serial port and a heater controller on I2C:

```toml
[portal]
url = "wss://robutler.ai/ws"
agent = "bench"
auth = "device"

[[device]]
spec = "pwm1.regspec.json"
transport = { kind = "uart", path = "/dev/serial0", baud = 115200 }

[[device]]
spec = "heater.regspec.json"
transport = { kind = "i2c", bus = "/dev/i2c-1" }
```

An unknown key anywhere in the file is an error, so a misspelt setting is never
silently ignored. Paths are relative to the configuration file.

### `[portal]`

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `url` | yes | | The platform's agent socket, `wss://robutler.ai/ws`. `https://robutler.ai` is accepted and resolved to it |
| `agent` | yes | | The agent's name on the platform |
| `auth` | no | | `"device"` to use this device's sign-in (see below). Without it the gateway uses a per-agent key from the environment |
| `token_env` | no | `"WEBAGENTS_AGENT_TOKEN"` | The name of the environment variable that holds the per-agent key. The key itself is never written in the file |
| `max_rounds` | no | 8 | Model calls per turn before the turn is stopped |
| `max_reconnect_attempts` | no | 10 | Consecutive failed connection attempts before the gateway exits with an error. A connection that opens resets the count |
| `reconnect_delay_s` | no | 5 | Seconds between connection attempts |

### `[llm]`

Optional. Without it the model is the platform's, chosen by the platform.

| Key | Default | Meaning |
| --- | --- | --- |
| `kind` | `"portal"` | `"portal"` for the platform's model, or `"accelerator"` for a model on the board's own AI accelerator |
| `model` | the platform chooses | Platform model to ask for, such as `"auto/balanced"` |
| `url` | the `/llm` socket next to `[portal] url` | The platform's model socket |
| `link`, `tokenizer`, `max_new`, `token_timeout_s` | | Accelerator only; see [Accelerator backend](./accelerator-backend.md) |

### `[[device]]`

One table per part. At least one is required. Every part's name comes from its
spec, and two parts may not have the same name, because the name prefixes the
tool names.

| Key | Required | Meaning |
| --- | --- | --- |
| `spec` | yes | Path to the part's `webagents.regspec/1` file |
| `transport` | yes | How its registers are reached: an inline table with a `kind` |
| `device_key` | no | The part's own public key, when the part signs its tool description; see [Trust and local limits](./trust-and-limits.md#signed-tool-descriptions) |

## Transports

### Serial: `uart`

```toml
transport = { kind = "uart", path = "/dev/serial0", baud = 115200, timeout_ms = 200 }
```

| Key | Default | Meaning |
| --- | --- | --- |
| `path` | required | The serial port, or a pseudo-terminal |
| `baud` | 115200 | Baud rate; 8 data bits, no parity, 1 stop bit |
| `timeout_ms` | 200 | How long to wait for the answer to one request |

A serial line has no standard register protocol, so the gateway defines a
small one that a part's firmware can serve from its UART in a few hundred
bytes, and that a person can type at a terminal:

```text
gateway -> part   R 0000000c              read the register at 0x0c
part -> gateway   V 000001f4              its value
gateway -> part   W 0000000c 000001f4     write it
part -> gateway   K                       done
part -> gateway   E <reason>              refused or failed
```

Each line ends with a newline. Numbers are hexadecimal, eight digits and lower
case from the gateway; either case is accepted. One request is in flight at a
time. Lines that are not answers are skipped as the part's own output, since
logs often share the UART, so a part should start each answer on a fresh line.
The gateway gives up on a request after 64 such lines or when the timeout
passes, and the tool call returns `bus_error`.

`D` asks a part for its signed tool description; a part that has none answers
`E`.

### Serial over TCP: `tcp`

```toml
transport = { kind = "tcp", addr = "192.168.1.40:4001", timeout_ms = 2000 }
```

The same line protocol over a TCP connection, for a serial line behind a
terminal server or `ser2net`. `timeout_ms` defaults to 2000.

### I2C: `i2c`

```toml
transport = { kind = "i2c", bus = "/dev/i2c-1" }
transport = { kind = "i2c", bus = "/dev/i2c-1", address = 0x49 }   # a second part of the same type
```

| Key | Default | Meaning |
| --- | --- | --- |
| `bus` | required | The I2C adapter's device node |
| `address` | from the spec | The part's 7-bit address, when it differs from the spec's |

The spec's `bus` must be of kind `i2c`. A read sends the register address
(most significant byte first), then reads the value after a repeated start. A
write sends the address followed by the value in one transfer. This is the
common layout of sensors and power management parts.

A plain I2C part cannot refuse anything, so its limits are enforced by the
gateway alone.

### SPI: `spi`

```toml
transport = { kind = "spi", path = "/dev/spidev0.0" }
```

The spec's `bus` must be of kind `spi`. One register access is one chip-select
window: the register address with bit 7 of its first byte set for a read and
clear for a write, then the value. The SPI mode and clock speed are the device
node's own settings; the gateway does not change them.

### CAN: `can`

```toml
transport = { kind = "can", iface = "can0", node = 5 }
```

| Key | Meaning |
| --- | --- |
| `iface` | The SocketCAN interface |
| `node` | The part's node id, 0 to 127 |

One request frame and one answer frame per register access, in classic CAN
frames with 11-bit identifiers so that any controller can serve them:

```text
request  id 0x600 + node   [op, off_lo, off_hi, v0, v1, v2, v3]   op 0x40 read, 0x23 write
answer   id 0x580 + node   [st, off_lo, off_hi, v0, v1, v2, v3]   st 0x43 read ok, 0x60 write ok, 0x80 refused
```

Values are little-endian. Register offsets must fit 16 bits. The gateway waits
200 ms for the answer and ignores other traffic on the bus. The frame layout
follows the shape of a CANopen expedited SDO (Service Data Object), so it reads familiarly on a bus
analyser, but it is not CANopen: there is no object dictionary, and the offsets
are register offsets.

### No hardware: `sim`

```toml
transport = { kind = "sim" }
```

An in-memory copy of the part at its reset values. The whole path except the
bus runs, which makes it the way to try a spec and a configuration before the
part is on the bench.

`robutler-device twin` goes one step further and plays the part on a real
transport, so the framing is exercised too:

```bash
robutler-device twin pwm1.regspec.json uart --pty       # prints the pseudo-terminal to put in `path`
robutler-device twin pwm1.regspec.json uart /dev/ttyUSB1
robutler-device twin pwm1.regspec.json can vcan0 5
```

## Check the configuration

```bash
robutler-device gateway check gateway.toml
```

`check` loads the file, validates every spec, opens every transport and exits.
It connects to nothing on the network. For each part it prints its name and
title, then one summary line. A part without a `device_key` also gets a note
that its tools are offered under the gateway's own sign-in:

```text
[webagents-gateway] pwm1: no device key; its tools are offered under the gateway's own sign-in
[webagents-gateway] heater: no device key; its tools are offered under the gateway's own sign-in
device pwm1: example three-phase PWM driver
device heater: bench heater controller
29 tools; agent bench at wss://robutler.ai/ws; key variable WEBAGENTS_AGENT_TOKEN is NOT set
```

The last part of the line says whether the per-agent key variable is set,
never its value. With `auth = "device"` the variable is not used and "NOT set"
is expected.

If something is wrong, `check` names it and exits with a failure: a spec
problem with the register and field it is in, a port that cannot be opened, a
spec whose `bus` does not match an `i2c` or `spi` transport, two parts with one
name.

## Sign the device in

A gateway proves that it is the agent in one of two ways.

**Device sign-in** keeps a key on the Pi and nothing in a file you have to
protect yourself. Run this as the user the gateway will run as:

```bash
robutler-device agent login @bench --portal https://robutler.ai
```

It prints a link, a short code and the fingerprint of the key it just made,
and draws the link as a QR code in the terminal:

```text
To sign this device in as @bench, open

  https://robutler.ai/device

and enter the code  WDJB-MJHT

Check that the page shows this device's fingerprint  BCDF-GHJK-LMNP-QRST
(or open https://robutler.ai/device?code=WDJB-MJHT directly, or scan this code with a phone):

  (the QR code is drawn here)

Waiting...
```

Open the link on a phone or a computer where you are signed in as the agent's
owner, check that the page shows the same fingerprint, and allow the device.
The command then prints `Signed in as @bench on robutler.ai.` and exits. The
Pi needs no browser and never sees your own credentials.

`--name <device name>` sets the name the device is listed under; the default
is the Pi's hostname. The key is stored under `~/.robutler` (or
`$WEBAGENTS_HOME`), readable only by that user. With `auth = "device"` in
`[portal]`, the gateway opens the agent socket with that key. `agent` must be
the name the command printed.

```bash
robutler-device agent logout @bench --portal https://robutler.ai
```

signs the device out and deletes its key, so `agent login` can be run again.
A key is never overwritten: if a sign-in never completed (the code expired or
was denied), `agent login` names the key it left behind; delete that file and
run the command again.

**A per-agent key** is the alternative. Leave `auth` out, and put a key that
was made for this one agent in the environment:

```bash
export WEBAGENTS_AGENT_TOKEN=...
```

An account-wide key is refused at start-up. The platform would accept the
connection and then never send the agent a turn, which is worse than an error.

## Run it

```bash
robutler-device gateway run gateway.toml
```

The gateway connects, announces the agent and waits for turns. It logs each
connection event to standard error:

```text
[robutler-device gateway] Connecting { attempt: 0 }
[robutler-device gateway] Open
[robutler-device gateway] SessionCreated { agent: Some("bench") }
[robutler-device gateway] TurnStarted { session_id: "..." }
```

Now talk to the agent on the platform. Each turn, the model is given the
parts' tools; its tool calls run on the Pi, through each part's envelope, and
every call and result is shown in the chat as it happens, including the
refusals.

- If the connection drops, the gateway reconnects after `reconnect_delay_s`.
  After `max_reconnect_attempts` failures in a row it exits with an error.
- `HandshakeRefused { status: 401 }` means the platform did not accept the key.
- Ctrl-C stops it: running turns are cancelled and the socket is closed.
- Calls to one part run one at a time, so two conversations can never
  interleave a read-modify-write of one register. Different parts are
  independent.
- A write to a field marked `confirm` in the spec is refused: the gateway does
  not take confirmations in chat. See [Trust and local
  limits](./trust-and-limits.md#confirmation).

## Run it as a service

A systemd unit keeps the gateway running across reboots and restarts it when
it gives up reconnecting:

```ini
[Unit]
Description=robutler-device gateway
Wants=network-online.target
After=network-online.target

[Service]
User=pi
SupplementaryGroups=dialout i2c spi
ExecStart=/home/pi/.cargo/bin/robutler-device gateway run /home/pi/gateway/gateway.toml
KillSignal=SIGINT
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp robutler-device.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robutler-device
journalctl -u robutler-device -f
```

- `User` is the user that ran `agent login`: the device key is in that user's
  home directory and a key file that anyone else can read is refused.
- `KillSignal=SIGINT` gives the gateway the signal it shuts down cleanly on.
- With a per-agent key instead of a device sign-in, put
  `WEBAGENTS_AGENT_TOKEN=<key>` in a file only root can read and add
  `EnvironmentFile=/etc/robutler-device.env` to `[Service]`.

The unit file and the configuration above are in
[`rust/examples/gateway-raspberry-pi`](https://github.com/robutlerai/webagents/tree/main/rust/examples/gateway-raspberry-pi);
a CAN configuration is in
[`rust/examples/gateway-can`](https://github.com/robutlerai/webagents/tree/main/rust/examples/gateway-can).

## From your own program

The command is a thin wrapper. The same gateway is one call from Rust, with
the `linux` feature of `webagents` and `tokio` (features `rt-multi-thread` and
`macros`) as dependencies:

```rust
use std::path::Path;
use webagents::gateway;

#[tokio::main]
async fn main() -> Result<(), gateway::GatewayError> {
    let (_stop, stopped) = tokio::sync::watch::channel(false);   // send `true` to stop
    gateway::run(Path::new("gateway.toml"), stopped, None).await
}
```
