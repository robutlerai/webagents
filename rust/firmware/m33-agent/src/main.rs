//! A device agent on a Cortex-M33 with an ESP-AT network co-processor.
//!
//! The firmware owns everything that matters for trust: its TLS session
//! (embedded-tls, server certificate checked against one pinned root), its one
//! WebSocket to the platform's agent socket, the model calls for its turns, and
//! the envelope of the part it drives. The co-processor only carries TCP
//! bytes; it never sees a key or plaintext. The device runs no server.
//!
//! Built for QEMU's mps2-an505 (the emulated proof). Configuration is fixed at
//! build time:
//!   AGENT_HOST, AGENT_PORT   where the platform is (as the co-processor dials it)
//!   AGENT_TLS=1              run TLS on the device (otherwise plain TCP, for a local mock)
//!   AGENT_TOKEN              per-agent key; the emulated runs only ever use a placeholder
//!   AGENT_EXPECT             "turn" (complete a turn, then the peer closes) or
//!                            "refused" (the platform must refuse the key with 4001)
#![no_std]
#![no_main]

mod agent;
mod subdevice;
mod uart;

// The example device's tables and typed API, generated from fixtures/pwm1.regspec.json (shared by both roles).
include!("../../../crates/webagents/tests/gen/pwm1.rs");

use core::sync::atomic::{AtomicU32, Ordering};

use cortex_m::peripheral::syst::SystClkSource;
use cortex_m_rt::{entry, exception};
use cortex_m_semihosting::debug;

pub static MS: AtomicU32 = AtomicU32::new(0);

pub fn now_ms() -> u32 {
    MS.load(Ordering::Relaxed)
}

#[exception]
fn SysTick() {
    MS.fetch_add(1, Ordering::Relaxed);
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    use core::fmt::Write;
    let mut c = uart::Uart::console();
    let _ = writeln!(c, "PANIC: {info}");
    debug::exit(debug::EXIT_FAILURE);
    loop {
        cortex_m::asm::wfi();
    }
}

/// Paints the free stack below the current frame so its high-water mark can be
/// read at the end (the deepest the stack ever went, measured, not estimated).
const PAINT: u32 = 0xA5A5_A5A5;
extern "C" {
    static mut __sheap: u32;
    static _stack_start: u32;
}

fn paint_stack() {
    let sp = cortex_m::register::msp::read() as usize;
    // SAFETY: the region between the end of .bss/.data (__sheap) and 256 bytes
    // below the live stack pointer is unused at this point (no heap).
    unsafe {
        let mut p = core::ptr::addr_of_mut!(__sheap) as usize;
        while p < sp - 256 {
            core::ptr::write_volatile(p as *mut u32, PAINT);
            p += 4;
        }
    }
}

pub fn stack_high_water() -> usize {
    // SAFETY: reads the painted region; nothing writes it but the stack.
    unsafe {
        let top = core::ptr::addr_of!(_stack_start) as usize;
        let mut p = core::ptr::addr_of!(__sheap) as usize;
        while p < top && core::ptr::read_volatile(p as *const u32) == PAINT {
            p += 4;
        }
        top - p
    }
}

#[entry]
fn main() -> ! {
    paint_stack();
    let mut cp = cortex_m::Peripherals::take().unwrap();
    // 25 MHz system clock on this board: one tick per millisecond.
    cp.SYST.set_clock_source(SystClkSource::Core);
    cp.SYST.set_reload(25_000 - 1);
    cp.SYST.clear_current();
    cp.SYST.enable_counter();
    cp.SYST.enable_interrupt();
    // AGENT_ROLE=subdevice: a part behind a gateway (its UART is the bus); otherwise the networked agent.
    let ok = if option_env!("AGENT_ROLE") == Some("subdevice") { subdevice::run() } else { agent::run() };
    {
        use core::fmt::Write;
        let mut c = uart::Uart::console();
        let _ = writeln!(c, "stack high-water: {} bytes; uptime {} ms; verdict {}", stack_high_water(), now_ms(), if ok { "PASS" } else { "FAIL" });
    }
    debug::exit(if ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {
        cortex_m::asm::wfi();
    }
}
