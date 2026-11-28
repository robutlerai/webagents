//! The gateway verifies a device's signed tool description
//! before it offers any of the device's tools: right key and the gateway's own
//! tool list accepted; a wrong key, bytes changed after signing, and a validly
//! signed but different tool list (other limits) all refused.
#![cfg(feature = "gateway")]

use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use webagents::auth::DeviceKey;
use webagents::gateway::Transport;
use webagents::regspec::engine::write_tools;
use webagents::regspec::load::load_json;
use webagents::transport::{open_verified, sim::RegModel, uart};

const SPEC: &str = include_str!("../../../fixtures/pwm1.regspec.json");

/// A device on a TCP "serial line" that answers D with `described`.
fn device(described: uart::Described) -> String {
    let dev = load_json(SPEC).unwrap();
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let model = RegModel::at_reset(dev);
    std::thread::spawn(move || {
        if let Ok((s, _)) = l.accept() {
            s.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
            let _ = uart::serve_described(s, model, Arc::new(AtomicBool::new(false)), described);
        }
    });
    addr
}

fn ours() -> Vec<u8> {
    let mut s = String::new();
    write_tools(load_json(SPEC).unwrap(), &mut s).unwrap();
    s.into_bytes()
}

fn tcp(addr: String) -> Transport {
    Transport::Tcp { addr, timeout_ms: Some(3000) }
}

#[test]
fn verified_before_offered() {
    let key = DeviceKey::from_seed(&[9; 32]);
    let other = DeviceKey::from_seed(&[10; 32]);
    let dev = load_json(SPEC).unwrap();
    let tools = ours();
    let sig = key.sign_tool_description("pwm1", 1795651200, &tools);

    // Right key, the gateway's own tool list: accepted, and the registers work.
    let addr = device(Some((tools.clone(), 1795651200, sig.as_str().to_string())));
    let mut d = open_verified(dev, &tcp(addr), Some(key.public_x())).expect("verified");
    let r = d.call("pwm1_read_deadtime", "{}", &Default::default());
    assert!(!r.is_error && r.json.contains("\"DEADTIME\":1"), "{}", r.json);

    // The same description checked against another key: refused.
    let addr = device(Some((tools.clone(), 1795651200, sig.as_str().to_string())));
    let e = open_verified(dev, &tcp(addr), Some(other.public_x())).err().unwrap().to_string();
    assert!(e.contains("does not verify"), "{e}");

    // Bytes changed after signing (a limit raised): refused.
    let mut raised = tools.clone();
    let i = raised.windows(7).position(|w| w == b"\"maximum\"".get(..7).unwrap()).unwrap();
    raised[i + 11] = b'9';
    let addr = device(Some((raised, 1795651200, sig.as_str().to_string())));
    let e = open_verified(dev, &tcp(addr), Some(key.public_x())).err().unwrap().to_string();
    assert!(e.contains("does not verify"), "{e}");

    // Validly signed, but not the gateway's tool list (other limits): refused.
    let different = b"[]".to_vec();
    let sig2 = key.sign_tool_description("pwm1", 1795651200, &different);
    let addr = device(Some((different, 1795651200, sig2.as_str().to_string())));
    let e = open_verified(dev, &tcp(addr), Some(key.public_x())).err().unwrap().to_string();
    assert!(e.contains("different tool list"), "{e}");

    // A device with no signed description, opened with a key: refused (no silent downgrade).
    let addr = device(None);
    assert!(open_verified(dev, &tcp(addr), Some(key.public_x())).is_err());
    // Without a key it is the gateway's own device (a part with no key).
    let addr = device(None);
    assert!(open_verified(dev, &tcp(addr), None).is_ok());
}
