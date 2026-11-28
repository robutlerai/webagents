# Register tools from a spec

A register spec becomes tools a model can call, and an envelope that refuses
any write outside the part's limits. No hardware is involved: the registers are
held in memory.

- [`heater.regspec.json`](heater.regspec.json): the spec of a small example
  part (a heater controller on I2C) with one limit of each kind: a range, a
  rule between two registers, a field that needs a person's confirmation, and
  a rate limit.
- [`regspec_tools.rs`](../../crates/webagents/examples/regspec_tools.rs): loads
  the spec, prints the tool list, and makes calls that are applied, rounded
  and refused.

## From the command line

```sh
robutler-device regspec check examples/regspec-tools/heater.regspec.json    # validate, list every problem
robutler-device regspec tools examples/regspec-tools/heater.regspec.json    # the tool list a model sees (JSON)
robutler-device regspec try   examples/regspec-tools/heater.regspec.json \
    heater_set_setpoint '{"setpoint": 45.2}' \
    heater_set_setpoint '{"setpoint": 95}' \
    heater_set_trip     '{"trip": 100}' \
    heater_read_setpoint '{}'
```

`regspec try` runs the calls in order against a copy of the part at its reset
values and prints each result or refusal. Set `ROBUTLER_TRY_CONFIRMED=1` to run
the calls as confirmed by a person.

## From Rust

```sh
cargo run -p webagents --example regspec_tools
```

The program is about 100 lines. The part that matters:

```rust
let dev = load_json(include_str!("heater.regspec.json"))?;           // validate and build the table
let mut engine = Engine::new(dev, bus, vec![0u64; dev.field_count]);  // any RegisterBus
let ctx = CallCtx { now_ms, confirmed: false };                       // what the host knows
match engine.call("heater_set_setpoint", r#"{"setpoint": 95}"#, &ctx, &mut out) {
    Ok(n) => { /* out[..n] is the JSON result */ }
    Err(refusal) => { /* refusal.write_json(dev, ..) says why, in the part's units */ }
}
```

## For firmware

```sh
robutler-device regspec gen examples/regspec-tools/heater.regspec.json > heater.rs
```

prints the same table as `static` Rust data with a typed API, for a
microcontroller that has no heap and no JSON parser. See
[`firmware`](../firmware).
