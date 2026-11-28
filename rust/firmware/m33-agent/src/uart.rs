//! CMSDK APB UART (as on the MPS2 AN505 FPGA image), polled.
//!
//! Secure aliases: UART0 0x5020_0000 is the console, UART1 0x5020_1000 is the
//! link to the network co-processor.

use core::ptr::{read_volatile, write_volatile};

const DATA: usize = 0x00;
const STATE: usize = 0x04;
const CTRL: usize = 0x08;
const BAUDDIV: usize = 0x10;
const TX_FULL: u32 = 1 << 0;
const RX_FULL: u32 = 1 << 1;

pub struct Uart {
    base: usize,
}

impl Uart {
    /// # Safety-free by construction: only the two fixed UART blocks are offered.
    pub fn console() -> Self {
        Self::init(0x5020_0000)
    }
    pub fn link() -> Self {
        Self::init(0x5020_1000)
    }
    fn init(base: usize) -> Self {
        // SAFETY: fixed peripheral addresses of this board; single core, no other owner.
        unsafe {
            write_volatile((base + BAUDDIV) as *mut u32, 16);
            write_volatile((base + CTRL) as *mut u32, 0b11);
        }
        Uart { base }
    }
    fn reg(&self, off: usize) -> *mut u32 {
        (self.base + off) as *mut u32
    }
    pub fn write_byte(&mut self, b: u8) {
        // SAFETY: see `init`.
        unsafe {
            while read_volatile(self.reg(STATE)) & TX_FULL != 0 {}
            write_volatile(self.reg(DATA), b as u32);
        }
    }
    pub fn try_read(&mut self) -> Option<u8> {
        // SAFETY: see `init`.
        unsafe {
            if read_volatile(self.reg(STATE)) & RX_FULL != 0 {
                Some(read_volatile(self.reg(DATA)) as u8)
            } else {
                None
            }
        }
    }
}

impl core::fmt::Write for Uart {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            self.write_byte(b);
        }
        Ok(())
    }
}
