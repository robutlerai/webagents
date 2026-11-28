//! The generated pwm1 module: it must match the committed golden file (so a
//! change in the generator is seen in review), compile, and go through the same
//! envelope as the tools.
//!
//! Regenerate the golden file with `UPDATE_GOLDEN=1 cargo test --test codegen_pwm1`.

include!("gen/pwm1.rs");

use webagents::bus::MemBus;
use webagents::regspec::{CallCtx, ToolError};

const CTX: CallCtx = CallCtx { now_ms: 0, confirmed: false };

#[test]
fn golden_matches_generator() {
    let dev = webagents::regspec::load::load_json(include_str!("../../../fixtures/pwm1.regspec.json")).unwrap();
    let src = webagents::regspec::codegen::generate(dev);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/gen/pwm1.rs");
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::write(path, &src).unwrap();
    }
    assert_eq!(std::fs::read_to_string(path).unwrap(), src, "generated code changed; rerun with UPDATE_GOLDEN=1 and review the diff");
}

#[test]
fn typed_values_and_checked_writes() {
    let mut bus = MemBus::<32>::new();
    for r in pwm1::DEVICE.registers {
        bus.set(r.offset, r.reset);
    }
    let mut t = pwm1::Pwm1::new(bus);
    let ctrl = pwm1::Ctrl::RESET.with_en(true).with_mode_value(pwm1::CtrlMode::SixStep).with_pole_pairs(7);
    assert_eq!(ctrl.pole_pairs(), 7);
    assert_eq!(ctrl.mode_value(), Some(pwm1::CtrlMode::SixStep));
    t.write_period(pwm1::Period::RESET.with_period_count(500), &CTX).unwrap();
    t.write_ctrl(ctrl, &CTX).unwrap();
    assert_eq!(t.read_ctrl().unwrap(), ctrl);
    assert_eq!(t.read_period().unwrap().period(), 50.0);

    // Below the 0.5 us floor: refused, as for the model.
    let r = t.write_deadtime(pwm1::Deadtime::RESET.with_deadtime_count(4), &CTX);
    assert!(matches!(r, Err(ToolError::Refused(_))));
    // Sinusoidal needs PERIOD <= 200 us (2000 counts).
    t.write_period(pwm1::Period::RESET.with_period_count(3000), &CTX).unwrap();
    assert!(t.modify_ctrl(&CTX, |c| c.with_mode_value(pwm1::CtrlMode::Sinusoidal)).is_err());
    // Sign-magnitude helpers agree with the engine.
    let sense = pwm1::Sense::RESET.with_offset_a(-5);
    assert_eq!(sense.offset_a(), -5);
    assert_eq!(sense.0 & 0x3f, 0x25);
    assert_eq!(pwm1::Sense::RESET.trip(), 120.0);
    // A whole-word write names only the fields it changes: the trims move without
    // touching the trip level, so no confirmation is needed...
    t.write_sense(sense, &CTX).unwrap();
    // ...but changing the trip level needs a person's confirmation even from typed code.
    assert!(t.write_sense(pwm1::Sense::RESET.with_trip_count(9), &CTX).is_err());
    t.write_sense(pwm1::Sense::RESET.with_trip_count(9), &CallCtx { now_ms: 0, confirmed: true }).unwrap();
    assert_eq!(pwm1::FIELD_COUNT, pwm1::DEVICE.field_count);
}
