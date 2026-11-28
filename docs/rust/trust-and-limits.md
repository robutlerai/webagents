---
title: Trust and local limits
description: What keeps a part safe when a model drives it - the envelope from the register spec, rules, confirmation, rate limits, signed tool descriptions and device keys.
---

# Trust and local limits

A model that drives hardware must never be the thing that keeps the hardware
inside its limits. A prompt can be wrong, a model can be talked into things,
and a gateway can be replaced. So the limits belong to the part: they are
written once, next to the register map, and every layer that can reach the bus
enforces the same table.

This page is about what is enforced, where, and what each party has to be
trusted for.

## The envelope

The envelope is everything the register spec says about what may be written.
The engine (`webagents::regspec::Engine`) checks a write in this order, and
every check happens before anything is written to the bus:

1. **Shape.** The arguments are a flat JSON object whose keys are writable
   fields of that register, each named once. A nested value, an unknown key or
   a read-only field is an error, never something skipped: an argument the tool
   did not use must not look accepted.
2. **Conversion.** Each value becomes a whole count. A physical value (`1.5`
   in microseconds) is converted with exact decimal arithmetic, never through a
   float, and rounded to the field's resolution. The rounding is reported in
   the result.
3. **Range.** The count is inside the field's range: the spec's `range`, or
   what the field's bits can hold.
4. **Confirmation and rate limits.** See below.
5. **Rules.** Every rule the write can affect holds for the values as they
   would be after the write.

A write that fails a check is **refused**: the tool returns an error result
with a code, the reason and the limits in the part's units, and ends with
"Nothing was written." It is never clamped. A request for 0.2 us of dead time
that quietly writes 0.5 us is a different command from the one the model thinks
it issued, and the model would go on reasoning from a value the part does not
hold.

```json
{"ok":false,"error":"out_of_range","message":"DEADTIME = 0.2 is outside its limits 0.5 to 25.5 us. Nothing was written."}
```

Rounding is the one adjustment the engine makes, and it says so:

```json
{"ok":true,"register":"DEADTIME","written":"0x0000000c","applied":{"DEADTIME":1.2},"rounded":{"DEADTIME":{"requested":1.23,"applied":1.2}},"units":{"DEADTIME":"us"}}
```

Every code is listed in the [Refusals reference](./refusals-reference.md).

### Ranges

```json
{ "bits": "7:0", "name": "DEADTIME", "unit": "us", "scale": 0.1, "range": { "min": 0.5 } }
```

A range is written in the field's physical units and stored as counts when the
spec is loaded. A bound between two steps is rounded inwards, so a limit is
never widened by rounding. A range the field's bits cannot hold, or an empty
one, is an error in the spec, and a spec with any error loads no table at all:
a half-understood limit is not a limit.

### Rules

A rule ties fields together. It reads like the datasheet sentence it comes
from:

```json
{ "bits": "15:0", "name": "DUTY", "unit": "us", "scale": 0.1,
  "rules": [ { "require": "self <= PERIOD.PERIOD", "why": "the on-time must not exceed the PWM period" } ] }
```

```json
{ "when": "CTRL.MODE == sinusoidal", "require": "self >= 20",
  "why": "sinusoidal commutation needs a PWM period of at least 20 us" }
```

- `require` must hold, or the write is refused with `why`. `when` makes the
  rule conditional.
- Each side of a comparison is `self` (the value being written), `REG.FIELD`,
  an enum name, `true` or `false` against a flag, or a number in the field's
  physical units. The operators are
  `==`, `!=`, `<`, `<=`, `>`, `>=`.
- `why` is required. It is what the model is told on a refusal, and it is
  listed in the tool's schema as a limit, so the model knows it before it asks.

A rule is written on one field but guards every field it names. `DUTY <=
PERIOD` written on `DUTY` also refuses a `PERIOD` write that would put the
period under the present duty, and `CTRL.MODE == sinusoidal` in a `when` also
refuses switching the mode while the period is outside 20 to 200 us. A limit
written once cannot be walked around through the other operand.

A field that a rule reads from another register is read from the part at write
time, so the loader refuses a rule that refers to a register whose read has a
side effect, or to a field that cannot be read.

### Confirmation

```json
{ "bits": "19:16", "name": "TRIP", "unit": "mV", "scale": 2.5, "offset": 100, "confirm": true }
```

A field marked `confirm` (a trip level, a safety interlock) is written only
when the **host** says a person confirmed this call. That is
`CallCtx::confirmed`, a value the code that runs the engine sets. It is not a
tool argument, so the model cannot grant it to itself, whatever it writes.

The gateway always runs tool calls with `confirmed: false`: it does not take
confirmations in chat. A write to a `confirm` field through the gateway is
therefore refused with `needs_confirmation`, and the model is instructed to say
that the change needs to be confirmed at the device. A program of your own that
has a real confirmation channel (a button on the device, a signed approval)
sets the flag for the one call it covers.

### Rate limits

```json
{ "bits": "7:0", "name": "POS", "min_interval_ms": 1000 }
```

`min_interval_ms` is the minimum time between two writes of a field (a relay,
a valve, flash memory with a limited number of write cycles). The engine keeps
the time of the last write per field and refuses an earlier one with
`rate_limited` and the number of milliseconds to wait. The clock is
`CallCtx::now_ms`, also set by the host: any clock that never goes backwards,
such as time since boot.

### Registers that are not safe to read

Some registers change state when read (a status register that clears on read,
a receive FIFO). Marked `"read_side_effect": true`, such a register is never
read behind the caller's back: a write must name every writable field, because
the engine will not read the register to fill in the rest, and its `read` tool
warns the model that reading changes the device.

## The limits are in the tool description

The tool list a model receives is generated from the same table, so it states
the envelope up front: each field's type, `minimum` and `maximum` in physical
units, the unit, the resolution, every rule that bounds it (including rules
declared on other fields), and whether it needs confirmation or is rate
limited.

```json
"deadtime": { "type": "number", "minimum": 0.5, "maximum": 25.5, "x-unit": "us",
              "description": "Dead time. In us, resolution 0.1." }
```

This is a courtesy to the model, not the protection. The description helps it
ask for something valid the first time; the engine is what refuses when it
does not.

The list is deterministic: the same spec always produces the same bytes. That
is what makes it possible to sign.

## The part enforces its limits itself

A gateway is a Linux host next to the part. If the envelope lived only there,
anything else that can reach the bus (another program on the host, a debug
adapter, a gateway built from a different spec) could write what it liked.

So a part with a microcontroller runs the same engine in its own firmware. The
tables are generated from the same spec (`robutler-device regspec gen`) and
compiled in; the core is `no_std` and uses no heap. Every raw register write
the part receives goes through `Engine::write_word`, which applies the same
range, confirmation, rate and rule checks as a tool call. A write below a limit
is answered with a refusal on the bus and the register keeps its value:

```text
W 00000010 00000003
E refused by the part's own limits
R 00000010
V 0000000a
```

The gateway reports that as `bus_error` to the model. With both sides built
from one spec this never happens in normal use, because the gateway refuses
first and explains why; it is the backstop for when the gateway is wrong,
bypassed or hostile.

The firmware in
[`rust/firmware/m33-agent`](https://github.com/robutlerai/webagents/tree/main/rust/firmware/m33-agent)
does this in its sub-device role; see [Microcontroller
firmware](./mcu-firmware.md).

## Signed tool descriptions

When the part enforces limits itself, one question is left for the gateway:
are the limits it is about to advertise and enforce the limits of **this**
part? A gateway configured with last year's spec would describe tools the part
then refuses.

A part that holds a key answers it by signing its tool description:

1. At boot the part writes its tool list (the exact bytes its engine
   generates) through a hash and signs the result with its Ed25519 device key.
   The message that is signed names the format, the key, the device and the
   time, and carries the SHA-256 of the description.
2. The gateway asks for the description on the serial line (`D`). The part
   sends the bytes and the signature.
3. The gateway verifies the signature with the part's **public** key from its
   configuration, and then compares the bytes with the tool list it built from
   its own spec. Both must match.

```toml
[[device]]
spec = "pwm1.regspec.json"
transport = { kind = "uart", path = "/dev/ttyUSB0" }
device_key = "<the part's public key>"   # Ed25519 public key, 43 characters of base64url
```

If the signature does not verify, or the part signed a different list, the
gateway does not start and offers none of that part's tools. There is no
partial mode. A device key needs a serial transport (`uart` or `tcp`), because
that is where the description is fetched.

Without `device_key` the tools are the gateway's own word: it logs that the
part's tools are offered under the gateway's sign-in, and builds them from its
spec alone.

## Device keys

A device proves itself with a key it made, not with a secret someone typed
into it.

- The key is **Ed25519**, generated on the device from the system's random
  source. The private key never leaves the device. On a gateway it is stored
  as a file readable only by its owner (mode 0600 in a 0700 directory,
  `~/.robutler` or `$WEBAGENTS_HOME`), is never overwritten and never printed;
  a key file that others can read is refused.
- The key's identity is its **thumbprint** (RFC 7638, SHA-256), and for people
  a **fingerprint** of 16 letters in four groups, `XXXX-XXXX-XXXX-XXXX`.
- **Sign-in** is the device authorization flow: `robutler-device agent login
  @agent --portal <url>` shows a short code, the fingerprint and a link (also
  as a QR code); the agent's owner opens the link on a phone or a computer,
  checks that the page shows the same fingerprint, and allows the device. The
  device has no browser and never sees the owner's credentials.
- From then on **every request is signed** (HTTP Message Signatures, RFC 9421)
  over the method, host, path, query, body digest and any bearer token, with a
  single-use nonce and a 60 second validity window. The agent socket is opened
  with a one-use ticket fetched by such a signed request. A short-lived access
  token exists too, but the platform accepts it only beside a signature that
  covers it, so a copied token is useless without the key.
- `robutler-device agent logout @agent --portal <url>` signs the device out and
  deletes its key, so the device can be signed in again later.

The same key, with the same thumbprint, is what signs a tool description.

The signature format is shared by every SDK of this repository and checked
byte for byte against one set of test vectors
([`test-fixtures/agent-device`](https://github.com/robutlerai/webagents/tree/main/test-fixtures/agent-device)).
The key and signing code are `no_std` and heap-free (feature `auth`), so
firmware uses the same implementation.

## A per-agent key instead

A gateway can also connect with a key from the environment (`token_env` in
`[portal]`, `WEBAGENTS_AGENT_TOKEN` by default). It must be a key minted for
that one agent. An account-wide key is refused at start-up, before any
connection: the platform would accept the connection and then never route a
turn to it, which looks like an agent that is online and silent.

The key is read from the environment variable the configuration names. It is
never written in the configuration file, and `robutler-device gateway check`
reports only whether the variable is set.

## What the gateway does not trust

- **The model's arguments.** They are text until the engine has checked them.
- **The conversation history.** A relayed history is reduced to plain text
  turns before the model sees it. Tool messages and non-text parts are dropped,
  so a history the gateway did not write cannot inject a tool result.
  `system` and `developer` rows in the relayed history are kept as text, the
  same as in the TypeScript and Python bridges, and reach the model after the
  gateway's own instructions.
- **Who is asking, beyond the platform's routing.** The gateway does not act on
  who is asking: every turn the platform routes to the agent gets the same tools
  and the same limits. Who may talk to the agent is decided by the agent's
  access settings on the platform.
- **Other traffic on the bus.** Lines on a serial port that are not protocol
  answers are skipped as the device's own output, and CAN frames from other
  nodes are ignored, never taken for an answer.

## What you still have to get right

The engine enforces the spec. It cannot know what the spec leaves out.

- **The spec is the safety case.** A limit that is not in the file does not
  exist. Write the ranges and rules from the datasheet's absolute maximum
  ratings and the application's own limits, and review the file like code.
  `robutler-device regspec check` validates its consistency, not its physics.
- **`regspec try` before hardware.** Run the calls you worry about against an
  in-memory copy of the part and read what is refused.
- **The envelope is per write.** It judges one register write against the
  present state of the part. A sequence of individually valid writes that is
  harmful as a whole (a slow ramp to a hot operating point) needs a rule that
  expresses it, a rate limit, or protection in the part.
- **A part without a microcontroller relies on its gateway.** A plain I2C or
  SPI sensor cannot refuse anything. Its envelope is the gateway's, so protect
  the host and its bus accordingly.
- **Hardware protection stays.** Fuses, current limits and watchdogs are still
  the last line. The envelope keeps a model from asking for what the part
  should never do; it does not replace what protects the part when something
  fails.
