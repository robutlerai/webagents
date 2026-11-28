//! The sub-device role: the part behind a gateway. Its UART is the bus; it speaks
//! the webagents register line protocol and answers `D` with its tool
//! description signed by its own device key, so the gateway
//! can check, before it offers a single tool, that the limits it enforces are
//! the limits this part signed. The part enforces the same table itself on every
//! write (`Engine::write_word`): a gateway that skipped its envelope would still
//! be refused here.
//!
//! The private key seed comes in at build time (AGENT_DEVICE_SEED_FILE) and is
//! never printed: only the public key and its thumbprint go to the console.
//! In emulation a `Q` line ends the run.

use core::fmt::Write as _;

use webagents::auth::{DeviceKey, ToolDigest};
use webagents::bus::{Mmio, RegisterBus};
use webagents::regspec::engine::write_tools;
use webagents::regspec::json::BufWriter;
use webagents::regspec::CallCtx;

use crate::pwm1;
use crate::uart::Uart;

const SEED: &[u8; 32] = include_bytes!(concat!(env!("OUT_DIR"), "/device-seed.bin"));
const DEVICE: &str = "pwm1";

static mut REGS: [u32; 12] = [0; 12];

macro_rules! log {
    ($($t:tt)*) => {{ let mut c = Uart::console(); let _ = writeln!(c, $($t)*); }};
}

fn created() -> u64 {
    option_env!("AGENT_CREATED").and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn send(link: &mut Uart, b: &[u8]) {
    for x in b {
        link.write_byte(*x);
    }
}

/// One instance of the tool writer for both of its sinks (the digest and the
/// link): two generic instances cost 7.4 KB of flash.
fn tools_to(mut w: &mut dyn core::fmt::Write) -> core::fmt::Result {
    write_tools(&pwm1::DEVICE, &mut w)
}

pub fn run() -> bool {
    if SEED == &[0u8; 32] {
        log!("sub-device: no device key in this image (build with AGENT_DEVICE_SEED_FILE)");
        return false;
    }
    // SAFETY: single core; each static has one user, taken once here.
    let regs = unsafe { &mut *core::ptr::addr_of_mut!(REGS) };
    for r in pwm1::DEVICE.registers {
        regs[(r.offset / 4) as usize] = r.reset;
    }
    // SAFETY: REGS is this part's register block in the emulation and has no other user.
    let bus = unsafe { Mmio::new(regs.as_mut_ptr() as usize) };
    let mut engine = pwm1::Engine::new(&pwm1::DEVICE, bus, [0; pwm1::FIELD_COUNT]);

    // The description is streamed, never held: hashed once here to sign, and
    // written again straight to the link on every `D`.
    let mut digest = ToolDigest::new();
    if tools_to(&mut digest).is_err() {
        return false;
    }
    let (n, sha) = digest.finish();
    let key = DeviceKey::from_seed(SEED);
    let created = created();
    let t0 = crate::now_ms();
    let sig = key.sign_tool_description_sha256(DEVICE, created, &sha);
    log!("sub-device {DEVICE}: tool description {n} bytes, signed in {} ms by keyid {}", crate::now_ms() - t0, key.keyid());
    log!("sub-device {DEVICE}: public key x {}", key.public_x());

    let mut link = Uart::link();
    let mut line = [0u8; 64];
    let mut len = 0usize;
    loop {
        let Some(b) = link.try_read() else { continue };
        if b != b'\n' {
            if b != b'\r' && len < line.len() {
                line[len] = b;
                len += 1;
            }
            continue;
        }
        let text = core::str::from_utf8(&line[..len]).unwrap_or("");
        len = 0;
        let mut it = text.split(' ');
        let hex = |s: Option<&str>| s.and_then(|x| u32::from_str_radix(x, 16).ok());
        match it.next() {
            Some("D") => {
                let mut h = [0u8; 24];
                let mut hw = BufWriter::new(&mut h);
                let _ = writeln!(hw, "T {n}");
                let hn = hw.len();
                send(&mut link, &h[..hn]);
                let _ = tools_to(&mut link);
                let mut t = [0u8; 120];
                let mut tw = BufWriter::new(&mut t);
                let _ = write!(tw, "\nS {created} {}\n", sig);
                let tn = tw.len();
                send(&mut link, &t[..tn]);
                log!("sub-device: sent the signed tool description");
            }
            Some("R") => match hex(it.next()).and_then(|o| pwm1::DEVICE.registers.iter().find(|r| r.offset == o)) {
                Some(r) => {
                    let v = engine.bus.read(r.offset, r.width).unwrap_or(0);
                    let mut o = [0u8; 16];
                    let mut ow = BufWriter::new(&mut o);
                    let _ = writeln!(ow, "V {v:08x}");
                    let on = ow.len();
                    send(&mut link, &o[..on]);
                }
                None => send(&mut link, b"E unknown register\n"),
            },
            Some("W") => {
                let (off, val) = (hex(it.next()), hex(it.next()));
                match (off.and_then(|o| pwm1::DEVICE.registers.iter().position(|r| r.offset == o)), val) {
                    (Some(ri), Some(v)) => {
                        let ctx = CallCtx { now_ms: crate::now_ms() as u64, confirmed: false };
                        match engine.write_word(ri, v, &ctx) {
                            Ok(()) => send(&mut link, b"K\n"),
                            Err(_) => {
                                log!("sub-device: refused a write to {} (outside the signed limits)", pwm1::DEVICE.registers[ri].name);
                                send(&mut link, b"E refused by the part's own limits\n");
                            }
                        }
                    }
                    _ => send(&mut link, b"E bad request\n"),
                }
            }
            Some("Q") => return true,
            _ => send(&mut link, b"E unknown request\n"),
        }
    }
}
