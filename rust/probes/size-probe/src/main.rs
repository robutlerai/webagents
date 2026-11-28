//! Bare-metal size probe. `_start` runs one tool call on the pwm1 tables over
//! memory-mapped I/O with arguments read from RAM, so nothing is folded away.
#![no_std]
#![no_main]

#[cfg(not(any(feature = "baseline", feature = "w5500")))]
include!("../../../crates/webagents/tests/gen/pwm1.rs");

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Arguments arrive in RAM (as from a socket); the result leaves through RAM.
#[no_mangle]
static mut ARGS: [u8; 64] = *b"{\"deadtime\": 1.0}                                               ";
#[no_mangle]
static mut OUT: [u8; 512] = [0; 512];
#[no_mangle]
static mut RESULT: usize = 0;

#[no_mangle]
#[link_section = ".text._start"]
pub extern "C" fn _start() -> ! {
    #[cfg(feature = "w5500")]
    w5500_path();
    #[cfg(not(any(feature = "baseline", feature = "w5500")))]
    {
        use webagents::bus::Mmio;
        use webagents::regspec::CallCtx;
        // SAFETY: the probe is never run; it only has to link.
        let bus = unsafe { Mmio::new(0x4000_0000) };
        let mut engine = pwm1::Engine::new(&pwm1::DEVICE, bus, [0; pwm1::FIELD_COUNT]);
        let args = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(ARGS)) };
        let args = core::str::from_utf8(&args).unwrap_or("{}");
        let out = unsafe { &mut *core::ptr::addr_of_mut!(OUT) };
        let n = match engine.call("pwm1_set_deadtime", args, &CallCtx::default(), out) {
            Ok(n) => n,
            // A device sends refusals back too, so the error renderer is linked.
            Err(e) => {
                let mut w = webagents::regspec::json::BufWriter::new(out);
                let _ = e.write_json(&pwm1::DEVICE, &mut w);
                w.len()
            }
        };
        #[cfg(feature = "tools")]
        let n = {
            let mut w = webagents::regspec::json::BufWriter::new(out);
            let _ = webagents::regspec::engine::write_tools(&pwm1::DEVICE, &mut w);
            n + w.len()
        };
        unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!(RESULT), n) };
    }
    loop {}
}

/// A SPI bus that reads and writes a volatile location, so the driver code is
/// linked whole and nothing is folded away.
#[cfg(feature = "w5500")]
struct VolatileSpi;

#[cfg(feature = "w5500")]
impl embedded_hal::spi::ErrorType for VolatileSpi {
    type Error = embedded_hal::spi::ErrorKind;
}

#[cfg(feature = "w5500")]
impl embedded_hal::spi::SpiDevice for VolatileSpi {
    fn transaction(&mut self, ops: &mut [embedded_hal::spi::Operation<'_, u8>]) -> Result<(), Self::Error> {
        static mut WIRE: u8 = 0;
        for op in ops {
            match op {
                embedded_hal::spi::Operation::Write(b) => {
                    for x in b.iter() {
                        unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!(WIRE), *x) };
                    }
                }
                embedded_hal::spi::Operation::Read(b) => {
                    for x in b.iter_mut() {
                        *x = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(WIRE)) };
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(feature = "w5500")]
fn w5500_path() {
    use core::cell::RefCell;
    use webagents::net::w5500::{Socket, W5500};
    use webagents::net::ws;
    fn now() -> u32 {
        unsafe { core::ptr::read_volatile(core::ptr::addr_of!(RESULT)) as u32 }
    }
    let chip = RefCell::new(W5500::new(VolatileSpi, now));
    if chip.borrow_mut().init([2, 0, 0, 0, 0, 1], [10, 0, 0, 2], [10, 0, 0, 1], [255, 255, 255, 0]).is_err() {
        return;
    }
    if chip.borrow_mut().connect(0, [10, 0, 0, 1], 443, 40000).is_err() {
        return;
    }
    let mut s = Socket { chip: &chip, sock: 0, timeout_ms: 1000 };
    let mut scratch = [0u8; 256];
    let args = unsafe { &*core::ptr::addr_of!(ARGS) };
    if ws::connect(&mut s, "h", "/ws", [0; 16], &mut scratch).is_ok() {
        let _ = ws::send(&mut s, ws::OP_TEXT, args, [1, 2, 3, 4]);
        let out = unsafe { &mut *core::ptr::addr_of_mut!(OUT) };
        if let Ok(ws::Msg::Text(n)) = ws::recv(&mut s, out, || [0; 4]) {
            unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!(RESULT), n) };
        }
    }
}
