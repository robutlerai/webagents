//! Each transport carries the engine's register traffic, end to end through
//! the real adapter code: embedded-hal I2C and SPI framing against simulated
//! parts, the serial register protocol over a real PTY pair, and CAN over a
//! real SocketCAN interface (when `WEBAGENTS_TEST_CAN_IFACE` names one, e.g.
//! a `vcan0` made inside `unshare -rn`).
#![cfg(feature = "linux")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use webagents::bus::{I2cRegs, RegisterBus, SpiRegs};
use webagents::regspec::{load::load_json, CallCtx, Device, Engine};
use webagents::transport::sim::{RegModel, SimI2c, SimSpi};

const SENSOR: &str = r#"{
  "schema": "webagents.regspec/1",
  "device": { "name": "tmp1", "title": "example temperature sensor" },
  "bus": { "kind": "i2c", "address": "0x48", "addr_bytes": 1, "data_bytes": 2, "big_endian": true },
  "registers": [
    { "offset": 0, "name": "TEMP", "access": "R", "fields": [ { "bits": "15:0", "name": "TEMP", "encoding": "signed", "unit": "degC", "scale": 0.0078125 } ] },
    { "offset": 1, "name": "CONFIG", "fields": [
      { "bits": "0", "name": "SHUTDOWN" },
      { "bits": "7:6", "name": "RATE", "enum": { "0": "slow", "1": "normal", "2": "fast" } } ] },
    { "offset": 2, "name": "LIMIT", "fields": [ { "bits": "15:0", "name": "LIMIT", "encoding": "signed", "unit": "degC", "scale": 0.0078125, "range": { "min": -40, "max": 125 } } ] }
  ]
}"#;

const CTX: CallCtx = CallCtx { now_ms: 0, confirmed: false };

fn call<B: RegisterBus>(e: &mut Engine<'static, B, Vec<u64>>, tool: &str, args: &str) -> Value {
    let mut out = [0u8; 1024];
    match e.call(tool, args, &CTX, &mut out) {
        Ok(n) => serde_json::from_slice(&out[..n]).unwrap(),
        Err(err) => {
            let mut s = String::new();
            err.write_json(e.dev, &mut s).unwrap();
            serde_json::from_str(&s).unwrap()
        }
    }
}

fn sensor() -> &'static Device<'static> {
    load_json(SENSOR).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn i2c_framing_through_embedded_hal() {
    let dev = sensor();
    let model = RegModel::at_reset(dev);
    model.lock().unwrap().regs.insert(0, (-1280i16) as u16 as u32); // -10 degC
    let i2c = SimI2c::new(model.clone(), 0x48, 1, 2, true);
    let mut e = Engine::new(dev, I2cRegs { i2c, address: 0x48, addr_bytes: 1, data_bytes: 2, big_endian: true }, vec![0; dev.field_count]);
    assert_eq!(call(&mut e, "tmp1_read_temp", "{}")["fields"]["TEMP"], -10);
    assert_eq!(call(&mut e, "tmp1_set_limit", r#"{"limit": 85.5}"#)["ok"], true);
    assert_eq!(model.lock().unwrap().get(2), (85.5f64 / 0.0078125) as u32);
    assert_eq!(call(&mut e, "tmp1_set_limit", r#"{"limit": 130}"#)["error"], "out_of_range");
    assert_eq!(call(&mut e, "tmp1_set_config", r#"{"rate": "fast"}"#)["ok"], true);
    assert_eq!(model.lock().unwrap().get(1), 2 << 6);
    // A part that does not answer at the address: a bus error, reported as such.
    let mut wrong = Engine::new(
        dev,
        I2cRegs { i2c: SimI2c::new(model, 0x48, 1, 2, true), address: 0x49, addr_bytes: 1, data_bytes: 2, big_endian: true },
        vec![0; dev.field_count],
    );
    assert_eq!(call(&mut wrong, "tmp1_read_temp", "{}")["error"], "bus_error");
}

#[test]
fn spi_framing_through_embedded_hal() {
    let dev = sensor();
    let model = RegModel::at_reset(dev);
    let spi = SimSpi { model: model.clone(), addr_bytes: 1, data_bytes: 2, big_endian: true, read_bit: 0x80 };
    let mut e = Engine::new(dev, SpiRegs { spi, addr_bytes: 1, data_bytes: 2, big_endian: true, read_bit: 0x80 }, vec![0; dev.field_count]);
    assert_eq!(call(&mut e, "tmp1_set_limit", r#"{"limit": -20}"#)["ok"], true);
    assert_eq!(call(&mut e, "tmp1_read_limit", "{}")["fields"]["LIMIT"], -20);
    assert_eq!(model.lock().unwrap().get(2), (-2560i16) as u16 as u32);
}

#[test]
fn uart_register_protocol_over_a_real_pty() {
    use serialport::{SerialPort, TTYPort};
    let dev = load_json(include_str!("../../../fixtures/pwm1.regspec.json")).unwrap();
    let model = RegModel::at_reset(dev);
    let (mut master, mut slave) = TTYPort::pair().expect("a PTY pair");
    master.set_timeout(Duration::from_millis(500)).unwrap();
    slave.set_timeout(Duration::from_millis(50)).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let (m2, s2) = (model.clone(), stop.clone());
    let part = std::thread::spawn(move || webagents::transport::uart::serve(slave, m2, s2));
    let mut e = Engine::new(dev, webagents::transport::uart::UartRegs { port: master }, vec![0; dev.field_count]);
    assert_eq!(call(&mut e, "pwm1_set_period", r#"{"period": 40}"#)["ok"], true);
    assert_eq!(call(&mut e, "pwm1_set_duty", r#"{"duty": 50}"#)["error"], "rule");
    assert_eq!(call(&mut e, "pwm1_set_ctrl", r#"{"en": true, "mode": "sinusoidal"}"#)["ok"], true);
    assert_eq!(call(&mut e, "pwm1_read_ctrl", "{}")["fields"]["MODE"], "sinusoidal");
    assert_eq!(model.lock().unwrap().get(0x08), 400);
    stop.store(true, Ordering::Relaxed);
    part.join().unwrap().unwrap();
}

#[test]
fn can_register_frames_over_socketcan() {
    let Ok(iface) = std::env::var("WEBAGENTS_TEST_CAN_IFACE") else {
        eprintln!("skipped: set WEBAGENTS_TEST_CAN_IFACE (e.g. vcan0 inside `unshare -rn`)");
        return;
    };
    let dev = load_json(include_str!("../../../fixtures/pwm1.regspec.json")).unwrap();
    let model = RegModel::at_reset(dev);
    let stop = Arc::new(AtomicBool::new(false));
    let (m2, s2, i2) = (model.clone(), stop.clone(), iface.clone());
    let part = std::thread::spawn(move || webagents::transport::can::serve(&i2, 5, m2, s2));
    std::thread::sleep(Duration::from_millis(50));
    let bus = webagents::transport::can::CanRegs::open(&iface, 5).unwrap();
    let mut e = Engine::new(dev, bus, vec![0; dev.field_count]);
    assert_eq!(call(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 2.5}"#)["ok"], true);
    assert_eq!(call(&mut e, "pwm1_read_deadtime", "{}")["fields"]["DEADTIME"], 2.5);
    assert_eq!(call(&mut e, "pwm1_set_deadtime", r#"{"deadtime": 0.1}"#)["error"], "out_of_range");
    assert_eq!(model.lock().unwrap().get(0x10), 25);
    // No part at node 9: the request times out and is reported as a bus error.
    let mut none = Engine::new(dev, webagents::transport::can::CanRegs::open(&iface, 9).unwrap(), vec![0; dev.field_count]);
    assert_eq!(call(&mut none, "pwm1_read_ctrl", "{}")["error"], "bus_error");
    stop.store(true, Ordering::Relaxed);
    part.join().unwrap().unwrap();
}
