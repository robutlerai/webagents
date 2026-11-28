---
title: Rust quickstart
description: Describe a part's registers and limits in a spec, see it become typed tools, try calls and refusals without hardware, then put it on the agent network with a gateway.
---

# Rust quickstart

From a register map to an agent, in commands you can paste. No hardware is
needed: the part is simulated until the last step, and replacing the simulated
part with a real one is a one-line change.

The part in this guide is a small heater controller. Its register map has one
limit of each kind, so every kind of refusal shows up.

## 1. Install

```bash
cargo install robutler
```

This installs the `robutler-device` command. The `regspec` commands used in
steps 2 to 4 work on any host; the gateway in step 6 needs Linux.

## 2. Describe the part

A register spec is the part's register map as a datasheet lists it, plus what a
datasheet usually leaves to prose: units, ranges, named values and the rules
that keep the part safe. Save this as `heater.regspec.json`:

```json
{
  "schema": "webagents.regspec/1",
  "device": { "name": "heater", "title": "bench heater controller" },
  "bus": { "kind": "i2c", "address": "0x48", "addr_bytes": 1, "data_bytes": 2 },
  "provenance": {
    "what": "A synthetic register map for the examples. It describes no real part."
  },
  "registers": [
    {
      "offset": "0x00", "name": "CTRL", "access": "R/W", "desc": "Control", "reset": 0,
      "fields": [
        { "bits": "0", "name": "EN", "desc": "Heater on" },
        { "bits": "2:1", "name": "MODE", "desc": "Operating mode",
          "enum": [ { "name": "normal", "value": 0, "desc": "Full setpoint range" },
                    { "name": "eco", "value": 1, "desc": "Setpoint limited to 60 degC" } ] }
      ]
    },
    {
      "offset": "0x01", "name": "SETPOINT", "access": "R/W", "desc": "Target temperature", "reset": 80,
      "fields": [
        { "bits": "7:0", "name": "SETPOINT", "desc": "Target temperature", "unit": "degC", "scale": 0.5, "reset": 80,
          "range": { "min": 5, "max": 90 },
          "rules": [
            { "when": "CTRL.MODE == eco", "require": "self <= 60", "why": "eco mode keeps the setpoint at or below 60 degC" }
          ] }
      ]
    },
    {
      "offset": "0x02", "name": "TEMP", "access": "R", "desc": "Measured temperature",
      "fields": [
        { "bits": "15:0", "name": "TEMP", "desc": "Measured temperature", "encoding": "signed", "unit": "degC", "scale": 0.1 }
      ]
    },
    {
      "offset": "0x03", "name": "TRIP", "access": "R/W", "desc": "Over-temperature cut-out", "reset": 190,
      "fields": [
        { "bits": "7:0", "name": "TRIP", "desc": "Temperature at which the heater is cut off", "unit": "degC", "scale": 0.5, "reset": 190,
          "range": { "min": 60, "max": 110 }, "confirm": true }
      ]
    },
    {
      "offset": "0x04", "name": "FAN", "access": "R/W", "desc": "Cooling fan", "reset": 0,
      "fields": [
        { "bits": "3:0", "name": "LEVEL", "desc": "Fan level, 0 is off", "range": { "max": 10 }, "min_interval_ms": 2000 }
      ]
    }
  ]
}
```

What the limits say:

| Field | Limit | In the spec |
| --- | --- | --- |
| `SETPOINT` | 5 to 90 degC, in steps of 0.5 | `"scale": 0.5`, `"range": { "min": 5, "max": 90 }` |
| `SETPOINT` | At most 60 degC while the mode is `eco` | a rule: `"when": "CTRL.MODE == eco"`, `"require": "self <= 60"` |
| `TRIP` | 60 to 110 degC, and only with a person's confirmation | `"range"`, `"confirm": true` |
| `FAN.LEVEL` | 0 to 10, changed at most once every 2 seconds | `"range": { "max": 10 }`, `"min_interval_ms": 2000` |
| `TEMP` | Read only | `"access": "R"` |

Check the file. Every problem in a spec is reported at once, with the register
and field it is in:

```bash
robutler-device regspec check heater.regspec.json
```

```text
heater.regspec.json: heater (bench heater controller), 5 registers, 6 fields, 4 writable registers, 1 rules: OK
```

## 3. See the tools

Each register becomes up to two tools: `heater_read_<register>` when it can be
read, and `heater_set_<register>` when it has a field that can be written.

```bash
robutler-device regspec tools heater.regspec.json
```

The output is the JSON tool list a model is given, in the function-calling
shape. This part has 9 tools:

```text
heater_read_ctrl        heater_set_ctrl(en, mode)
heater_read_setpoint    heater_set_setpoint(setpoint)
heater_read_temp
heater_read_trip        heater_set_trip(trip)
heater_read_fan         heater_set_fan(level)
```

One of them, formatted for reading:

```json
{
  "type": "function",
  "function": {
    "name": "heater_set_setpoint",
    "description": "Write SETPOINT (Target temperature) of the bench heater controller. Name only the fields to change; the others keep their value. A value outside a field's limits is refused with the reason, not adjusted.",
    "parameters": {
      "type": "object",
      "properties": {
        "setpoint": {
          "type": "number",
          "minimum": 5,
          "maximum": 90,
          "x-unit": "degC",
          "description": "Target temperature. In degC, resolution 0.5. Limit: eco mode keeps the setpoint at or below 60 degC."
        }
      },
      "additionalProperties": false,
      "minProperties": 1
    }
  }
}
```

The limits are in the schema, in the part's units: the range as `minimum` and
`maximum`, the unit, the resolution and the rule. A model can read them before
it asks. They are generated from the spec, so they cannot drift from what is
enforced.

## 4. Try calls without hardware

`regspec try` runs tool calls, in order, against a copy of the part at its
reset values, and prints what each one returns:

```bash
robutler-device regspec try heater.regspec.json \
    heater_set_setpoint '{"setpoint": 45.2}' \
    heater_set_setpoint '{"setpoint": 95}' \
    heater_set_setpoint '{"setpoint": 70}' \
    heater_set_ctrl     '{"mode": "eco"}' \
    heater_set_trip     '{"trip": 100}' \
    heater_read_setpoint '{}'
```

```text
heater_set_setpoint {"setpoint": 45.2}
  -> {"ok":true,"register":"SETPOINT","written":"0x005a","applied":{"SETPOINT":45},"rounded":{"SETPOINT":{"requested":45.2,"applied":45}},"units":{"SETPOINT":"degC"}}
heater_set_setpoint {"setpoint": 95}
  -> {"ok":false,"error":"out_of_range","message":"SETPOINT = 95 is outside its limits 5 to 90 degC. Nothing was written."}
heater_set_setpoint {"setpoint": 70}
  -> {"ok":true,"register":"SETPOINT","written":"0x008c","applied":{"SETPOINT":70},"units":{"SETPOINT":"degC"}}
heater_set_ctrl {"mode": "eco"}
  -> {"ok":false,"error":"rule","message":"eco mode keeps the setpoint at or below 60 degC (70 degC is not <= 60 degC). Nothing was written."}
heater_set_trip {"trip": 100}
  -> {"ok":false,"error":"needs_confirmation","message":"TRIP needs a person's confirmation for this write. Ask them, then the host retries with their confirmation. Nothing was written."}
heater_read_setpoint {}
  -> {"register":"SETPOINT","raw":"0x008c","fields":{"SETPOINT":70},"units":{"SETPOINT":"degC"}}
```

Reading the six results in order:

1. **Applied, and rounded.** 45.2 is between two 0.5 degC steps. The nearest
   step was written, and the result says so under `rounded`. This is the only
   adjustment the engine ever makes, and it is always reported.
2. **Refused: out of range.** 95 is above the 90 degC limit. The refusal names
   the field, the value and the limits in the part's units, and nothing was
   written. A value is never clamped to the limit.
3. **Applied.** 70 degC is fine in normal mode.
4. **Refused: a rule.** Switching to `eco` would leave the setpoint above 60
   degC. The rule is written on `SETPOINT`, and it also guards the mode switch,
   so the limit cannot be walked around by changing the other register first.
5. **Refused: needs confirmation.** The trip level is marked `confirm`. A model
   cannot confirm its own request; only the program that runs the engine can
   state that a person agreed.
6. **A read.** Fields come back in their units, and the setpoint is still the
   70 degC of step 3: the refused calls changed nothing.

Every refusal has a code (`out_of_range`, `rule`, `needs_confirmation`). They
are all listed in the [Refusals reference](./refusals-reference.md).

## 5. Call the tools from Rust

The command above is a thin wrapper around a few lines of Rust. In a project of
your own:

```bash
cargo new heater-demo && cd heater-demo
cargo add webagents
cp ../heater.regspec.json .
```

`src/main.rs`:

```rust
use webagents::bus::RegisterBus;
use webagents::regspec::{load::load_json, CallCtx, Engine};

/// The part, in this program: eight registers in memory. On real hardware this is a
/// bus adapter (I2C, SPI, memory-mapped), and nothing else changes.
struct Registers([u32; 8]);

impl RegisterBus for Registers {
    type Error = ();
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, ()> {
        self.0.get(offset as usize).copied().ok_or(())
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), ()> {
        *self.0.get_mut(offset as usize).ok_or(())? = value;
        Ok(())
    }
}

fn main() {
    // Validate the spec and build the device table.
    let dev = load_json(include_str!("../heater.regspec.json")).expect("the spec is valid");

    // The engine: the table, the bus, and one slot of state per field.
    let mut engine = Engine::new(dev, Registers([0; 8]), vec![0u64; dev.field_count]);

    // What the host knows about a call: the time, and whether a person confirmed it.
    let ctx = CallCtx { now_ms: 0, confirmed: false };

    let mut out = [0u8; 512];
    for (tool, args) in [
        ("heater_set_setpoint", r#"{"setpoint": 45}"#),
        ("heater_set_setpoint", r#"{"setpoint": 95}"#),
    ] {
        match engine.call(tool, args, &ctx, &mut out) {
            Ok(n) => println!("{}", std::str::from_utf8(&out[..n]).unwrap()),
            Err(refusal) => {
                let mut why = String::new();
                refusal.write_json(dev, &mut why).unwrap();
                println!("{why}");
            }
        }
    }
}
```

```bash
cargo run
```

```text
{"ok":true,"register":"SETPOINT","written":"0x005a","applied":{"SETPOINT":45},"units":{"SETPOINT":"degC"}}
{"ok":false,"error":"out_of_range","message":"SETPOINT = 95 is outside its limits 5 to 90 degC. Nothing was written."}
```

Three things carry over to every other use of the crate:

- **`RegisterBus`** is the only thing the engine needs from hardware: read a
  register, write a register. The crate has adapters for memory-mapped
  registers and for `embedded-hal` I2C and SPI buses; the gateway adds a serial
  line and CAN.
- **`Engine::call`** takes a tool name and its JSON arguments, exactly as a
  model produces them, and returns the JSON result or a refusal.
- **`CallCtx`** is what the model cannot assert: the clock for rate limits, and
  whether a person confirmed the call.

The engine itself needs no heap and no operating system. Firmware uses the same
engine over tables generated from the same spec with `robutler-device regspec
gen`; see [Microcontroller firmware](./mcu-firmware.md).

A longer version of this program, with every kind of refusal, is the
`regspec_tools` example in
[`rust/examples/regspec-tools`](https://github.com/robutlerai/webagents/tree/main/rust/examples/regspec-tools).

## 6. Make it an agent

A gateway puts the part on the agent network as an agent: it connects to the
platform, receives turns, asks the model, and runs the model's tool calls
through the engine. Save this as `gateway.toml`, next to the spec:

```toml
[portal]
url = "wss://robutler.ai/ws"
agent = "heater-bench"

[[device]]
spec = "heater.regspec.json"
transport = { kind = "sim" }
```

`kind = "sim"` is an in-memory copy of the part, as in step 4. Check the
configuration; this opens every device and connects to nothing. The first
line is a note that this part has no key of its own, and the last says whether
the per-agent key variable is set, which does not matter for the sign-in used
below:

```bash
robutler-device gateway check gateway.toml
```

```text
[webagents-gateway] heater: no device key; its tools are offered under the gateway's own sign-in
device heater: bench heater controller
9 tools; agent heater-bench at wss://robutler.ai/ws; key variable WEBAGENTS_AGENT_TOKEN is NOT set
```

To go online, the gateway has to prove that it is the agent `heater-bench`.
The agent must exist on the platform and be yours (use the name of an agent
you own on [robutler.ai](https://robutler.ai)). Sign this machine in as it:

```bash
robutler-device agent login @heater-bench --portal https://robutler.ai
```

The command prints a link, a short code and a fingerprint. Open the link where
you are signed in as the agent's owner, check that the page shows the same
fingerprint, and allow the device. The key that proves the gateway's identity
was made on this machine and stays on it. Then add one line to `[portal]`:

```toml
auth = "device"
```

and run the gateway:

```bash
robutler-device gateway run gateway.toml
```

```text
[robutler-device gateway] Connecting { attempt: 0 }
[robutler-device gateway] Open
[robutler-device gateway] SessionCreated { agent: Some("heater-bench") }
```

The agent is online. Talk to it on the platform:

> Set the heater to 45 degrees and turn it on.

The model reads the tool list, calls `heater_set_setpoint` and
`heater_set_ctrl`, and the chat shows each call and its result as it happens.
Ask for 95 degrees and the chat shows the refusal from step 4, and the model
explains the limit instead of pretending it worked. Whatever the model asks
for, the part stays inside the spec.

Ctrl-C stops the gateway.

## 7. Replace the simulated part

The spec says the heater is an I2C part at address `0x48`. On a Linux host
with that part on an I2C bus, change one line of `gateway.toml`:

```toml
transport = { kind = "i2c", bus = "/dev/i2c-1" }
```

Nothing else changes: the same spec, the same tools, the same refusals. The
transports are a serial line, I2C, SPI, CAN and serial over TCP.

## Next steps

- [Gateway on a Raspberry Pi](./gateway-raspberry-pi.md): every transport and
  configuration key, device sign-in, and running the gateway as a service.
- [Trust and local limits](./trust-and-limits.md): what is enforced where,
  signed tool descriptions and device keys.
- [Microcontroller firmware](./mcu-firmware.md): the agent in firmware, with no
  gateway.
- [Accelerator backend](./accelerator-backend.md): the model on the board's own
  AI accelerator.
