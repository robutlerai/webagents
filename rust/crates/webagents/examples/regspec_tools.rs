//! A register spec turned into typed tools, with no hardware:
//!
//! ```sh
//! cargo run -p webagents --example regspec_tools
//! ```
//!
//! It loads `rust/examples/regspec-tools/heater.regspec.json`, prints the tool
//! list a model would be given, then makes a few calls against registers held
//! in memory: one that is applied, one that is rounded, and one refusal for
//! each kind of limit the spec declares (a range, a rule, a confirmation, a
//! rate limit). Every refusal leaves the register as it was, which the last
//! lines print.
//!
//! The same calls can be tried from the command line:
//!
//! ```sh
//! robutler-device regspec try rust/examples/regspec-tools/heater.regspec.json \
//!     heater_set_setpoint '{"setpoint": 95}'
//! ```

use webagents::bus::RegisterBus;
use webagents::regspec::engine::write_tools;
use webagents::regspec::load::load_json;
use webagents::regspec::{CallCtx, Device, Engine};

/// The part, for this example: eight registers in memory, addressed the way
/// the spec's I2C bus addresses them (register address 0, 1, 2, ...). On real
/// hardware this is `webagents::bus::I2cRegs` over an `embedded-hal` I2C bus,
/// and nothing else in this file changes.
struct Registers {
    words: [u32; 8],
    writes: usize,
}

#[derive(Debug)]
struct NoSuchRegister;

impl RegisterBus for Registers {
    type Error = NoSuchRegister;
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
        self.words.get(offset as usize).copied().ok_or(NoSuchRegister)
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
        *self.words.get_mut(offset as usize).ok_or(NoSuchRegister)? = value;
        self.writes += 1;
        Ok(())
    }
}

/// Runs one call and prints what a model would get back: the result, or the
/// refusal with its reason.
fn call(engine: &mut Engine<'static, Registers, Vec<u64>>, dev: &Device, tool: &str, args: &str, ctx: &CallCtx) {
    let mut out = [0u8; 1024];
    match engine.call(tool, args, ctx, &mut out) {
        Ok(n) => println!("{tool} {args}\n  -> {}", String::from_utf8_lossy(&out[..n])),
        Err(e) => {
            let mut refusal = String::new();
            e.write_json(dev, &mut refusal).expect("writing to a String cannot fail");
            println!("{tool} {args}\n  -> {refusal}");
        }
    }
}

fn main() {
    let dev = load_json(include_str!("../../../examples/regspec-tools/heater.regspec.json")).unwrap_or_else(|problems| {
        // Every problem in the spec, each with the register and field it is in.
        eprint!("{problems}");
        std::process::exit(1);
    });

    // The tool list, as a model is given it (OpenAI-style function tools).
    let mut tools = String::new();
    write_tools(dev, &mut tools).expect("writing to a String cannot fail");
    let tools: serde_json::Value = serde_json::from_str(&tools).expect("the tool list is JSON");
    println!("{} ({}) offers {} tools:", dev.name, dev.title, tools.as_array().map_or(0, Vec::len));
    for tool in tools.as_array().into_iter().flatten() {
        let params: Vec<&str> = tool["function"]["parameters"]["properties"].as_object().map(|p| p.keys().map(String::as_str).collect()).unwrap_or_default();
        println!("  {}({})", tool["function"]["name"].as_str().unwrap_or("?"), params.join(", "));
    }
    println!();

    // The part at its reset values.
    let mut regs = Registers { words: [0; 8], writes: 0 };
    for r in dev.registers {
        regs.words[r.offset as usize] = r.reset;
    }
    let mut engine = Engine::new(dev, regs, vec![0u64; dev.field_count]);

    // What the HOST knows about a call: the time, and whether a person confirmed it.
    let at = |now_ms| CallCtx { now_ms, confirmed: false };

    // Applied as asked.
    call(&mut engine, dev, "heater_set_setpoint", r#"{"setpoint": 45}"#, &at(0));
    // Between two 0.5 degC steps: the nearest is applied, and the result says so.
    call(&mut engine, dev, "heater_set_setpoint", r#"{"setpoint": 45.2}"#, &at(0));
    // Outside the range.
    call(&mut engine, dev, "heater_set_setpoint", r#"{"setpoint": 95}"#, &at(0));
    // A rule: eco mode caps the setpoint at 60 degC. In normal mode 70 degC
    // is fine. The rule is written on SETPOINT, and it also guards the mode
    // switch: going to eco with the setpoint at 70 degC is refused.
    call(&mut engine, dev, "heater_set_setpoint", r#"{"setpoint": 70}"#, &at(0));
    call(&mut engine, dev, "heater_set_ctrl", r#"{"mode": "eco"}"#, &at(0));
    // Needs a person's confirmation, which a model cannot give itself.
    call(&mut engine, dev, "heater_set_trip", r#"{"trip": 100}"#, &at(0));
    call(&mut engine, dev, "heater_set_trip", r#"{"trip": 100}"#, &CallCtx { now_ms: 0, confirmed: true });
    // Rate limited: the fan level may change once every two seconds.
    call(&mut engine, dev, "heater_set_fan", r#"{"level": 3}"#, &at(1_000));
    call(&mut engine, dev, "heater_set_fan", r#"{"level": 5}"#, &at(1_500));
    call(&mut engine, dev, "heater_set_fan", r#"{"level": 5}"#, &at(3_000));
    // Reads report fields in their units.
    call(&mut engine, dev, "heater_read_setpoint", "{}", &at(3_000));

    println!("\n10 set calls, {} register writes: every refused call left the part as it was.", engine.bus.writes);
}
