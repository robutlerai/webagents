//! The register-spec engine against the synthetic pwm1 driver spec: every check
//! in the write path, the descriptors the model sees, and validation.

use serde_json::Value as J;
use webagents::bus::MemBus;
use webagents::regspec::load::{load_json, SpecError};
use webagents::regspec::{engine::write_tools, CallCtx, Device, Engine, ToolError};

const SPEC: &str = include_str!("../../../fixtures/pwm1.regspec.json");

fn dev() -> &'static Device<'static> {
    load_json(SPEC).unwrap_or_else(|e| panic!("{e}"))
}

/// A bus holding every register at its reset value.
fn reset_bus(d: &Device) -> MemBus<32> {
    let mut b = MemBus::<32>::new();
    for r in d.registers {
        b.set(r.offset, r.reset);
    }
    b
}

fn engine() -> Engine<'static, MemBus<32>, Vec<u64>> {
    let d = dev();
    Engine::new(d, reset_bus(d), vec![0; d.field_count])
}

const CTX: CallCtx = CallCtx { now_ms: 0, confirmed: false };

fn ok(e: &mut Engine<'static, MemBus<32>, Vec<u64>>, tool: &str, args: &str) -> J {
    let mut out = [0u8; 2048];
    match e.call(tool, args, &CTX, &mut out) {
        Ok(n) => serde_json::from_slice(&out[..n]).unwrap(),
        Err(err) => panic!("{tool} {args}: {}", render(e.dev, &err)),
    }
}

fn refused(e: &mut Engine<'static, MemBus<32>, Vec<u64>>, tool: &str, args: &str) -> J {
    refused_ctx(e, tool, args, &CTX)
}

fn refused_ctx(e: &mut Engine<'static, MemBus<32>, Vec<u64>>, tool: &str, args: &str, ctx: &CallCtx) -> J {
    let mut out = [0u8; 2048];
    let writes = e.bus.writes;
    match e.call(tool, args, ctx, &mut out) {
        Ok(n) => panic!("{tool} {args} was accepted: {}", std::str::from_utf8(&out[..n]).unwrap()),
        Err(err) => {
            assert_eq!(e.bus.writes, writes, "a refused call must not write");
            serde_json::from_str(&render(e.dev, &err)).unwrap()
        }
    }
}

fn render(d: &Device, e: &ToolError) -> String {
    let mut s = String::new();
    e.write_json(d, &mut s).unwrap();
    s
}

#[test]
fn loads_and_counts() {
    let d = dev();
    assert_eq!(d.name, "pwm1");
    assert_eq!(d.registers.len(), 12);
    assert_eq!(d.field_count, d.registers.iter().map(|r| r.fields.len()).sum::<usize>());
}

#[test]
fn tool_list_is_valid_json_with_limits_and_units() {
    let mut s = String::new();
    write_tools(dev(), &mut s).unwrap();
    let tools: J = serde_json::from_str(&s).unwrap();
    let tools = tools.as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    // 12 readable registers, 8 with writable fields.
    assert_eq!(names.iter().filter(|n| n.contains("_read_")).count(), 12);
    assert_eq!(names.iter().filter(|n| n.contains("_set_")).count(), 8);
    assert!(names.contains(&"pwm1_set_deadtime"));
    assert!(!names.contains(&"pwm1_set_status"));
    let dt = tools.iter().find(|t| t["function"]["name"] == "pwm1_set_deadtime").unwrap();
    let p = &dt["function"]["parameters"]["properties"]["deadtime"];
    assert_eq!(p["type"], "number");
    assert_eq!(p["minimum"], 0.5);
    assert_eq!(p["maximum"], 25.5);
    assert_eq!(p["x-unit"], "us");
    let ctrl = tools.iter().find(|t| t["function"]["name"] == "pwm1_set_ctrl").unwrap();
    let props = &ctrl["function"]["parameters"]["properties"];
    assert_eq!(props["mode"]["enum"], serde_json::json!(["six_step", "sinusoidal", "foc"]));
    assert_eq!(props["en"]["type"], "boolean");
    assert_eq!(props["pole_pairs"]["minimum"], 1);
    // The PERIOD rule names CTRL.MODE, so MODE's description carries it.
    assert!(props["mode"]["description"].as_str().unwrap().contains("at least 20 us"));
    // The DUTY <= PERIOD rule is declared on DUTY; PERIOD's description shows it too.
    let per = tools.iter().find(|t| t["function"]["name"] == "pwm1_set_period").unwrap();
    assert!(per["function"]["parameters"]["properties"]["period"]["description"].as_str().unwrap().contains("must not exceed the PWM period"));
    let trim = tools.iter().find(|t| t["function"]["name"] == "pwm1_set_sense").unwrap();
    assert!(trim["function"]["parameters"]["properties"]["trip"]["description"].as_str().unwrap().contains("confirmation"));
    let angle = tools.iter().find(|t| t["function"]["name"] == "pwm1_read_angle").unwrap();
    assert!(angle["function"]["description"].as_str().unwrap().contains("changes the device's state"));
}

#[test]
fn descriptor_bytes_are_deterministic() {
    let (mut a, mut b) = (String::new(), String::new());
    write_tools(dev(), &mut a).unwrap();
    write_tools(load_json(SPEC).unwrap(), &mut b).unwrap();
    assert_eq!(a, b);
}

#[test]
fn in_range_write_lands_with_units() {
    let mut e = engine();
    let r = ok(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 1.0}"#);
    assert_eq!(r["applied"]["DEADTIME"], 1);
    assert_eq!(r["units"]["DEADTIME"], "us");
    assert_eq!(e.bus.get(0x10), 10);
    let r = ok(&mut e, "pwm1_read_deadtime", "{}");
    assert_eq!(r["fields"]["DEADTIME"], 1);
}

#[test]
fn out_of_range_is_refused_not_clamped() {
    let mut e = engine();
    let r = refused(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 0.4}"#);
    assert_eq!(r["error"], "out_of_range");
    let m = r["message"].as_str().unwrap();
    assert!(m.contains("0.4") && m.contains("0.5 to 25.5 us"), "{m}");
    assert_eq!(e.bus.get(0x10), 10, "untouched");
    assert_eq!(refused(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 25.6}"#)["error"], "out_of_range");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"pole_pairs": 0}"#)["error"], "out_of_range");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"pole_pairs": 16}"#)["error"], "out_of_range");
    assert_eq!(refused(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 1e300}"#)["error"], "out_of_range");
}

#[test]
fn rounding_to_resolution_is_reported() {
    let mut e = engine();
    let r = ok(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 1.04}"#);
    assert_eq!(r["applied"]["DEADTIME"], 1);
    assert_eq!(r["rounded"]["DEADTIME"]["requested"], 1.04);
    assert_eq!(r["rounded"]["DEADTIME"]["applied"], 1);
    let r = ok(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 1.1}"#);
    assert!(r.get("rounded").is_none());
}

#[test]
fn unnamed_fields_keep_their_value() {
    let mut e = engine();
    ok(&mut e, "pwm1_set_ctrl", r#"{"pole_pairs": 7, "dir": true}"#);
    ok(&mut e, "pwm1_set_ctrl", r#"{"en": true}"#);
    let r = ok(&mut e, "pwm1_read_ctrl", "{}");
    assert_eq!(r["fields"]["POLE_PAIRS"], 7);
    assert_eq!(r["fields"]["DIR"], true);
    assert_eq!(r["fields"]["EN"], true);
}

#[test]
fn trigger_bits_are_never_rewritten() {
    let mut e = engine();
    ok(&mut e, "pwm1_set_ctrl", r#"{"fault_clr": true}"#);
    assert_eq!(e.bus.get(0) & 0x8, 0x8);
    // The model of the part: FAULT_CLR reads 0. Even if a twin echoes it back,
    // a later read-modify-write must not write it again.
    ok(&mut e, "pwm1_set_ctrl", r#"{"en": true}"#);
    assert_eq!(e.bus.get(0) & 0x8, 0, "FAULT_CLR re-triggered by a read-modify-write");
}

#[test]
fn rule_on_another_register_guards_both_sides() {
    let mut e = engine();
    ok(&mut e, "pwm1_set_period", r#"{"period": 50}"#);
    let r = refused(&mut e, "pwm1_set_duty", r#"{"duty": 60}"#);
    assert_eq!(r["error"], "rule");
    assert!(r["message"].as_str().unwrap().contains("must not exceed the PWM period"));
    ok(&mut e, "pwm1_set_duty", r#"{"duty": 40}"#);
    // Declared once, on DUTY; lowering PERIOD under the present DUTY is refused too.
    let r = refused(&mut e, "pwm1_set_period", r#"{"period": 10}"#);
    assert_eq!(r["error"], "rule");
    ok(&mut e, "pwm1_set_period", r#"{"period": 40}"#);
}

#[test]
fn mode_change_checks_the_period_rule() {
    let mut e = engine();
    ok(&mut e, "pwm1_set_period", r#"{"period": 300}"#);
    let r = refused(&mut e, "pwm1_set_ctrl", r#"{"mode": "sinusoidal"}"#);
    assert_eq!(r["error"], "rule");
    assert!(r["message"].as_str().unwrap().contains("at most 200 us"), "{r}");
    ok(&mut e, "pwm1_set_period", r#"{"period": 100}"#);
    ok(&mut e, "pwm1_set_ctrl", r#"{"mode": "sinusoidal", "en": true}"#);
    assert_eq!(refused(&mut e, "pwm1_set_period", r#"{"period": 10}"#)["error"], "rule");
    ok(&mut e, "pwm1_set_ctrl", r#"{"mode": "six_step"}"#);
    ok(&mut e, "pwm1_set_period", r#"{"period": 10}"#);
}

#[test]
fn confirmation_comes_from_the_host() {
    let mut e = engine();
    let r = refused(&mut e, "pwm1_set_sense", r#"{"trip": 120}"#);
    assert_eq!(r["error"], "needs_confirmation");
    // An argument named confirm is not a field: the model cannot grant it.
    assert_eq!(refused(&mut e, "pwm1_set_sense", r#"{"trip": 120, "confirm": true}"#)["error"], "unknown_argument");
    let mut out = [0u8; 512];
    let n = e.call("pwm1_set_sense", r#"{"trip": 125}"#, &CallCtx { now_ms: 0, confirmed: true }, &mut out).unwrap();
    let r: J = serde_json::from_slice(&out[..n]).unwrap();
    assert_eq!(r["applied"]["TRIP"], 125);
    assert_eq!((e.bus.get(0x18) >> 16) & 0xf, 10);
    // Trims untouched by the RMW.
    assert_eq!(e.bus.get(0x18) & 0x3f3f, 0);
}

#[test]
fn sign_magnitude_and_signed_fields() {
    let mut e = engine();
    ok(&mut e, "pwm1_set_sense", r#"{"offset_a": -5, "offset_b": 31}"#);
    assert_eq!(e.bus.get(0x18) & 0x3f, 0x20 | 5);
    assert_eq!((e.bus.get(0x18) >> 8) & 0x3f, 31);
    assert_eq!(refused(&mut e, "pwm1_set_sense", r#"{"offset_a": -32}"#)["error"], "out_of_range");
    let r = ok(&mut e, "pwm1_read_sense", "{}");
    assert_eq!(r["fields"]["OFFSET_A"], -5);
    ok(&mut e, "pwm1_set_phase", r#"{"adv": -45}"#);
    let r = ok(&mut e, "pwm1_read_phase", "{}");
    assert_eq!(r["fields"]["ADV"], -45);
    assert_eq!(r["fields"]["LEAD"], 10, "LEAD keeps its reset value");
    e.bus.set(0x20, 0xffff_8000);
    let r = ok(&mut e, "pwm1_read_raw", "{}");
    assert_eq!(r["fields"]["X"], -32768);
    assert_eq!(r["fields"]["Y"], -1);
}

#[test]
fn argument_errors() {
    let mut e = engine();
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"bogus": 1}"#)["error"], "unknown_argument");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"en": 1, "EN": 0}"#)["error"], "duplicate_argument");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"en": 2}"#)["error"], "wrong_type");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"pole_pairs": true}"#)["error"], "wrong_type");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"pole_pairs": 2.5}"#)["error"], "not_a_whole_count");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"mode": "turbo"}"#)["error"], "bad_value");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"mode": 3}"#)["error"], "bad_value");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"en": {"x": 1}}"#)["error"], "bad_arguments");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", "{}")["error"], "no_arguments");
    assert_eq!(refused(&mut e, "pwm1_set_ctrl", r#"{"en": null}"#)["error"], "no_arguments");
    assert_eq!(refused(&mut e, "pwm1_set_status", r#"{"fault": true}"#)["error"], "unknown_tool");
    assert_eq!(refused(&mut e, "pwm1_read_ctrl", r#"{"en": true}"#)["error"], "unknown_argument");
    assert_eq!(refused(&mut e, "pwm2_read_ctrl", "{}")["error"], "unknown_tool");
    // Mode by number is accepted when the number is a named value; flag by 0/1 too.
    ok(&mut e, "pwm1_set_ctrl", r#"{"mode": 0, "en": 1}"#);
    assert_eq!(e.bus.get(0) & 1, 1);
}

#[test]
fn reads_report_physical_values() {
    let mut e = engine();
    e.bus.set(0x14, (1234 << 16) | 16384);
    let r = ok(&mut e, "pwm1_read_angle", "{}");
    assert_eq!(r["fields"]["ANGLE"], 90);
    assert_eq!(r["fields"]["MAG"], 1234);
    assert_eq!(r["units"]["ANGLE"], "deg");
    assert_eq!(ok(&mut e, "pwm1_read_period", "")["fields"]["PERIOD"], 50);
    let r = ok(&mut e, "pwm1_read_ctrl", "{}");
    assert_eq!(r["fields"]["SRC"], "hall");
    assert_eq!(r["fields"]["MODE"], "six_step");
}

// ---- rate limits and side-effect registers, on a small inline spec --------------------

const SMALL: &str = r#"{
  "schema": "webagents.regspec/1",
  "device": { "name": "pmic", "title": "test PMIC" },
  "bus": { "kind": "i2c", "address": "0x60", "addr_bytes": 1, "data_bytes": 1 },
  "registers": [
    { "offset": 4, "name": "VOUT", "fields": [
      { "bits": "7:0", "name": "VOUT", "unit": "V", "scale": 0.0125, "offset": 0.6, "range": { "min": 0.8, "max": 1.3 }, "min_interval_ms": 100 } ] },
    { "offset": 8, "name": "FIFO", "read_side_effect": true, "fields": [
      { "bits": "3:0", "name": "A" }, { "bits": "7:4", "name": "B" } ] }
  ]
}"#;

#[test]
fn rate_limit_and_side_effect_registers() {
    let d = load_json(SMALL).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(d.registers[0].width, 8, "width follows the I2C data width");
    let mut e = Engine::new(d, MemBus::<4>::new(), vec![0u64; d.field_count]);
    let mut out = [0u8; 512];
    let at = |t| CallCtx { now_ms: t, confirmed: false };
    let n = e.call("pmic_set_vout", r#"{"vout": 1.0}"#, &at(1000), &mut out).unwrap();
    let r: J = serde_json::from_slice(&out[..n]).unwrap();
    assert_eq!(r["applied"]["VOUT"], 1);
    assert_eq!(e.bus.get(4), 32);
    let err = e.call("pmic_set_vout", r#"{"vout": 1.1}"#, &at(1050), &mut out).unwrap_err();
    assert!(render(d, &err).contains("wait 50 ms"));
    e.call("pmic_set_vout", r#"{"vout": 1.1}"#, &at(1100), &mut out).unwrap();
    assert!(matches!(e.call("pmic_set_vout", r#"{"vout": 1.4}"#, &at(9000), &mut out), Err(ToolError::Refused(_))));

    let reads = e.bus.reads;
    let err = e.call("pmic_set_fifo", r#"{"a": 1}"#, &at(0), &mut out).unwrap_err();
    assert!(render(d, &err).contains("must_name_all_fields"));
    e.call("pmic_set_fifo", r#"{"a": 1, "b": 2}"#, &at(0), &mut out).unwrap();
    assert_eq!(e.bus.reads, reads, "a side-effect register is never read behind the caller's back");
    assert_eq!(e.bus.get(8), 0x21);
}

// ---- validation ----------------------------------------------------------------------

fn issues(src: &str) -> Vec<String> {
    match load_json(src) {
        Ok(_) => panic!("accepted"),
        Err(SpecError(v)) => v.into_iter().map(|i| i.to_string()).collect(),
    }
}

#[test]
fn validation_reports_every_problem() {
    let v = issues(
        r#"{
      "device": { "name": "bad dev" },
      "registers": [
        { "offset": 0, "name": "A", "fields": [
          { "bits": "3:0", "name": "X" },
          { "bits": "4:2", "name": "Y" },
          { "bits": "5", "name": "Z", "scale": 2 },
          { "bits": "15:8", "name": "W", "range": { "min": 0, "max": 300 } },
          { "bits": "40:33", "name": "V" },
          { "bits": "16", "name": "U", "rules": [ { "require": "self <= B.NOPE", "why": "x" } ] },
          { "bits": "17", "name": "T", "rules": [ { "require": "self == 1", "why": "" } ] } ] },
        { "offset": 0, "name": "A", "access": "RX" },
        { "offset": 8, "name": "R", "access": "R", "fields": [ { "bits": "0", "name": "F", "confirm": true } ] }
      ]
    }"#,
    );
    let all = v.join("\n");
    for needle in [
        "device.name",
        "overlap",
        "needs a unit",
        "outside what 8 bits can hold",
        "exceed the 32-bit register",
        "no field B.NOPE",
        "needs a 'why'",
        "duplicate register name",
        "also used by A",
        "access 'RX'",
        "limits on a field that cannot be written",
    ] {
        assert!(all.contains(needle), "missing '{needle}' in:\n{all}");
    }
}

#[test]
fn rules_must_not_read_side_effect_registers() {
    let v = issues(
        r#"{ "device": { "name": "d" }, "registers": [
        { "offset": 0, "name": "A", "fields": [ { "bits": "7:0", "name": "X", "rules": [ { "require": "self <= B.Y", "why": "w" } ] } ] },
        { "offset": 4, "name": "B", "access": "R", "read_side_effect": true, "fields": [ { "bits": "7:0", "name": "Y" } ] } ] }"#,
    );
    assert!(v.join("\n").contains("side effect"));
}

#[test]
fn a_datasheet_register_list_loads_as_is() {
    // A plain datasheet register list, with no units or limits.
    let d = load_json(
        r#"{ "device": { "name": "blk" }, "bus": { "kind": "mmio", "base": "0x1000" }, "registers": [
        { "offset": "0x00", "name": "CTRL", "access": "R/W", "desc": "Control.", "fields": [
          { "bits": "0", "name": "EN", "desc": "Enable" }, { "bits": "9:6", "name": "POLE_PAIRS", "desc": "Pole pairs" } ] },
        { "offset": "0x08", "name": "DUTY", "access": "R/W", "desc": "PWM duty in counts, 16 bits." } ] }"#,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(d.registers[1].fields.len(), 1);
    assert_eq!(d.registers[1].fields[0].width, 32);
    let mut e = Engine::new(d, MemBus::<4>::new(), vec![0u64; d.field_count]);
    let mut out = [0u8; 256];
    e.call("blk_set_duty", r#"{"duty": 70000}"#, &CTX, &mut out).unwrap();
    assert_eq!(e.bus.get(8), 70000);
}

/// The previous schema id still loads for one release, and is reported as
/// deprecated; any other id is refused.
#[test]
fn deprecated_schema_id_still_loads() {
    use webagents::regspec::{load::deprecated_schema, SCHEMA};
    assert!(SPEC.contains(SCHEMA));
    let old = SPEC.replacen(SCHEMA, "reda.regspec/1", 1);
    assert_eq!(load_json(&old).unwrap_or_else(|e| panic!("{e}")).name, dev().name);
    assert_eq!(deprecated_schema(&old), Some("reda.regspec/1"));
    assert_eq!(deprecated_schema(SPEC), None);
    let other = SPEC.replacen(SCHEMA, "other.regspec/1", 1);
    assert!(load_json(&other).is_err());
}

/// A name given to a numeric field says the field takes a number, rather than
/// listing no names.
#[test]
fn a_name_for_a_numeric_field_says_it_takes_a_number() {
    let mut e = engine();
    let r = refused(&mut e, "pwm1_set_ctrl", r#"{"pole_pairs": "four"}"#);
    assert_eq!(r["error"], "bad_value");
    let m = r.to_string();
    assert!(m.contains("takes a number, not a name"), "{m}");
    assert!(!m.contains("one of:."), "{m}");
}

/// The 64-character limit counts the longer `_read_` infix, so no read tool
/// name can pass 64 characters.
#[test]
fn tool_name_limit_counts_the_read_infix() {
    let spec = |reg: &str| {
        format!(
            r#"{{ "device": {{ "name": "d" }}, "registers": [
            {{ "offset": 0, "name": "{reg}", "fields": [ {{ "bits": "7:0", "name": "X" }} ] }} ] }}"#
        )
    };
    // "d" + "_read_" + 57 = 64: fits. 58 makes the read tool 65.
    assert!(load_json(&spec(&"R".repeat(57))).is_ok());
    let v = issues(&spec(&"R".repeat(58)));
    assert!(v.join("\n").contains("_read_"), "{v:?}");
}
