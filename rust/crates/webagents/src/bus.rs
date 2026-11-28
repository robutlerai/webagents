//! The one thing the engine needs from hardware: read and write a register.
//!
//! Kept to two methods so every transport fits under it: memory-mapped I/O on
//! the chip that runs the agent, an I2C or SPI part behind a gateway, a twin in a
//! test. Adapters for `embedded-hal` 1.0 buses are behind the `hal` feature.
//!
//! This module is `no_std` and needs no allocator.
//!
//! | Type | Reaches the registers through | Needs |
//! | --- | --- | --- |
//! | [`Mmio`] | volatile loads and stores at a base address | nothing (`unsafe` to construct) |
//! | [`MemBus`] | an array in memory, with read and write counters | nothing |
//! | `I2cRegs` | an `embedded-hal` 1.0 `I2c` bus | feature `hal` |
//! | `SpiRegs` | an `embedded-hal` 1.0 `SpiDevice` | feature `hal` |
//!
//! The gateway's transports (a serial line, CAN, simulated parts) implement
//! the same trait in the `transport` module (feature `gateway`).
//!
//! # A bus of your own
//!
//! A transport this crate does not have is a few lines. The engine calls
//! [`RegisterBus::read`] and [`RegisterBus::write`] with the offsets and
//! widths the register spec states and never anything else:
//!
//! ```
//! use webagents::bus::RegisterBus;
//!
//! /// Two registers behind some link of your own.
//! struct TwoRegs {
//!     ctrl: u32,
//!     status: u32,
//! }
//!
//! #[derive(Debug)]
//! struct NoSuchRegister(u32);
//!
//! impl RegisterBus for TwoRegs {
//!     type Error = NoSuchRegister;
//!     fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
//!         match offset {
//!             0x00 => Ok(self.ctrl),
//!             0x04 => Ok(self.status),
//!             other => Err(NoSuchRegister(other)),
//!         }
//!     }
//!     fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
//!         match offset {
//!             0x00 => self.ctrl = value,
//!             other => return Err(NoSuchRegister(other)),
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let mut bus = TwoRegs { ctrl: 0, status: 0x10 };
//! bus.write(0x00, 32, 1).unwrap();
//! assert_eq!(bus.read(0x00, 32).unwrap(), 1);
//! assert!(bus.write(0x04, 32, 0).is_err());
//! ```

/// Register-level access to one device.
///
/// `offset` is the register's offset as the spec states it: a byte offset
/// from the device base for memory-mapped parts, the register address for
/// I2C and SPI parts. `width` is the register's width in bits (8, 16 or 32);
/// a value narrower than 32 bits travels in the low bits of the `u32`.
///
/// An implementation does exactly one register access per call and reports a
/// failure instead of retrying: the engine never repeats a write on its own,
/// because a register with a side effect must not be written twice.
///
/// `&mut B` is a `RegisterBus` whenever `B` is, so an engine can borrow a bus
/// that something else owns.
///
/// ```
/// use webagents::bus::{MemBus, RegisterBus};
///
/// let mut bus = MemBus::<4>::new();
/// bus.write(0x08, 32, 500).unwrap();
/// assert_eq!(bus.read(0x08, 32).unwrap(), 500);
/// assert_eq!((bus.reads, bus.writes), (1, 1));
/// // Offset 0x10 is the fifth word of a four-word bus.
/// assert!(bus.read(0x10, 32).is_err());
/// ```
pub trait RegisterBus {
    /// What the transport reports when a transfer fails (a NACK, a timeout, a
    /// refused frame). The engine does not interpret it: any error becomes
    /// [`ToolError::Bus`](crate::regspec::ToolError::Bus), because after a failed
    /// transfer the register's state is unknown.
    type Error: core::fmt::Debug;
    /// Read `width` bits (8, 16 or 32) at `offset`.
    fn read(&mut self, offset: u32, width: u8) -> Result<u32, Self::Error>;
    /// Write the low `width` bits (8, 16 or 32) of `value` at `offset`.
    fn write(&mut self, offset: u32, width: u8, value: u32) -> Result<(), Self::Error>;
}

impl<T: RegisterBus + ?Sized> RegisterBus for &mut T {
    type Error = T::Error;
    fn read(&mut self, offset: u32, width: u8) -> Result<u32, Self::Error> {
        (**self).read(offset, width)
    }
    fn write(&mut self, offset: u32, width: u8, value: u32) -> Result<(), Self::Error> {
        (**self).write(offset, width, value)
    }
}

/// Memory-mapped registers at a fixed base address.
///
/// A read or write is one volatile load or store of the register's width at
/// `base + offset`; any width other than 8 or 16 is accessed as 32 bits. It
/// cannot fail, so its error type is [`Infallible`](core::convert::Infallible).
///
/// # Safety
/// Constructing one asserts that `base` is the device's register block on this
/// target and that nothing else owns it.
pub struct Mmio {
    base: usize,
}

impl Mmio {
    /// Registers at `base`, the address the spec's `bus.base` names.
    ///
    /// # Safety
    /// See the type: `base` must be the device's register block, exclusively owned,
    /// and every offset the spec lists must be valid to load and store there.
    pub const unsafe fn new(base: usize) -> Self {
        Mmio { base }
    }
}

impl RegisterBus for Mmio {
    type Error = core::convert::Infallible;
    fn read(&mut self, offset: u32, width: u8) -> Result<u32, Self::Error> {
        let a = self.base + offset as usize;
        // SAFETY: the constructor's contract; the access width matches the register.
        unsafe {
            Ok(match width {
                8 => core::ptr::read_volatile(a as *const u8) as u32,
                16 => core::ptr::read_volatile(a as *const u16) as u32,
                _ => core::ptr::read_volatile(a as *const u32),
            })
        }
    }
    fn write(&mut self, offset: u32, width: u8, value: u32) -> Result<(), Self::Error> {
        let a = self.base + offset as usize;
        // SAFETY: as for `read`.
        unsafe {
            match width {
                8 => core::ptr::write_volatile(a as *mut u8, value as u8),
                16 => core::ptr::write_volatile(a as *mut u16, value as u16),
                _ => core::ptr::write_volatile(a as *mut u32, value),
            }
        }
        Ok(())
    }
}

/// A register file in memory, for tests and twins: `N` 32-bit words, with a
/// count of every read and write so a test can prove what reached the bus.
///
/// Offsets are byte offsets, 4-byte aligned, so `N` must be at least
/// `highest offset / 4 + 1`. The access width is ignored: every register is
/// stored as a whole word.
#[derive(Debug, Clone)]
pub struct MemBus<const N: usize> {
    /// The register words, indexed by `offset / 4`.
    pub regs: [u32; N],
    /// Number of [`RegisterBus::write`] calls so far, including failed ones. A
    /// test asserts on it to prove a refused call never reached the bus.
    pub writes: usize,
    /// Number of [`RegisterBus::read`] calls so far, including failed ones.
    pub reads: usize,
}

impl<const N: usize> MemBus<N> {
    /// `N` words, all zero, with both counters at zero. A part's reset values are
    /// not applied: [`MemBus::set`] them when a test depends on them.
    pub const fn new() -> Self {
        MemBus { regs: [0; N], writes: 0, reads: 0 }
    }
    /// The word at byte `offset` (offsets are 4-byte aligned in this model),
    /// without counting it as a bus read.
    ///
    /// # Panics
    /// If `offset / 4` is not below `N`.
    pub fn get(&self, offset: u32) -> u32 {
        self.regs[(offset / 4) as usize]
    }
    /// Stores `v` at byte `offset` without counting it as a bus write, to
    /// prepare a state the device "already has".
    ///
    /// # Panics
    /// If `offset / 4` is not below `N`, as for [`MemBus::get`].
    pub fn set(&mut self, offset: u32, v: u32) {
        self.regs[(offset / 4) as usize] = v;
    }
}

impl<const N: usize> Default for MemBus<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// The error of a [`MemBus`]: the byte offset that lies beyond its `N` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfRange(pub u32);

impl<const N: usize> RegisterBus for MemBus<N> {
    type Error = OutOfRange;
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
        self.reads += 1;
        self.regs.get((offset / 4) as usize).copied().ok_or(OutOfRange(offset))
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
        self.writes += 1;
        let slot = self.regs.get_mut((offset / 4) as usize).ok_or(OutOfRange(offset))?;
        *slot = value;
        Ok(())
    }
}

#[cfg(feature = "hal")]
#[cfg_attr(docsrs, doc(cfg(feature = "hal")))]
pub use hal::{I2cRegs, SpiRegs};

#[cfg(feature = "hal")]
mod hal {
    use super::RegisterBus;
    use embedded_hal::i2c::I2c;
    use embedded_hal::spi::{Operation, SpiDevice};

    fn put_be(buf: &mut [u8], v: u32) {
        let n = buf.len();
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (v >> (8 * (n - 1 - i))) as u8;
        }
    }
    fn get(buf: &[u8], big_endian: bool) -> u32 {
        let mut v = 0u32;
        if big_endian {
            for b in buf {
                v = (v << 8) | *b as u32;
            }
        } else {
            for b in buf.iter().rev() {
                v = (v << 8) | *b as u32;
            }
        }
        v
    }
    fn put(buf: &mut [u8], v: u32, big_endian: bool) {
        if big_endian {
            put_be(buf, v)
        } else {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (v >> (8 * i)) as u8;
            }
        }
    }

    /// Registers of an I2C target: address bytes then data bytes (the common
    /// sensor/PMIC layout). Register address is always sent big-endian.
    ///
    /// A read is one `write_read` (the address, a repeated start, then the
    /// value); a write is one `write` of the address followed by the value.
    /// The layout normally comes from the spec's
    /// [`Bus::I2c`](crate::regspec::Bus::I2c). `no_std`, no heap.
    #[cfg_attr(docsrs, doc(cfg(feature = "hal")))]
    pub struct I2cRegs<I> {
        /// The `embedded-hal` 1.0 I2C bus (or a device on a shared bus).
        pub i2c: I,
        /// The target's 7-bit address.
        pub address: u8,
        /// Width of a register address on the wire, 1 to 4 bytes.
        pub addr_bytes: u8,
        /// Width of a register's value on the wire, 1 to 4 bytes.
        pub data_bytes: u8,
        /// Byte order of the VALUE: most significant byte first when true.
        pub big_endian: bool,
    }

    impl<I: I2c> RegisterBus for I2cRegs<I> {
        type Error = I::Error;
        fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
            let mut a = [0u8; 4];
            let mut d = [0u8; 4];
            let (an, dn) = (self.addr_bytes as usize, self.data_bytes as usize);
            put_be(&mut a[..an], offset);
            self.i2c.write_read(self.address, &a[..an], &mut d[..dn])?;
            Ok(get(&d[..dn], self.big_endian))
        }
        fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
            let mut b = [0u8; 8];
            let (an, dn) = (self.addr_bytes as usize, self.data_bytes as usize);
            put_be(&mut b[..an], offset);
            put(&mut b[an..an + dn], value, self.big_endian);
            self.i2c.write(self.address, &b[..an + dn])
        }
    }

    /// Registers of a SPI device: address bytes (bit 7 of the first set for a
    /// read, the most common convention; set `read_bit` to 0 to disable) then data.
    ///
    /// Each access is one `SpiDevice` transaction, so the chip select frames
    /// it. The layout normally comes from the spec's
    /// [`Bus::Spi`](crate::regspec::Bus::Spi). `no_std`, no heap.
    #[cfg_attr(docsrs, doc(cfg(feature = "hal")))]
    pub struct SpiRegs<S> {
        /// The `embedded-hal` 1.0 SPI device (it owns the chip select, so one
        /// register access is one chip-select window).
        pub spi: S,
        /// Width of a register address on the wire, 1 to 4 bytes, sent most
        /// significant byte first.
        pub addr_bytes: u8,
        /// Width of a register's value on the wire, 1 to 4 bytes.
        pub data_bytes: u8,
        /// Byte order of the VALUE: most significant byte first when true.
        pub big_endian: bool,
        /// Mask ORed into the first address byte for a read and cleared for a
        /// write. `0x80` is the most common convention; `0` leaves the address as is.
        pub read_bit: u8,
    }

    impl<S: SpiDevice> RegisterBus for SpiRegs<S> {
        type Error = S::Error;
        fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
            let mut a = [0u8; 4];
            let mut d = [0u8; 4];
            let (an, dn) = (self.addr_bytes as usize, self.data_bytes as usize);
            put_be(&mut a[..an], offset);
            a[0] |= self.read_bit;
            self.spi.transaction(&mut [Operation::Write(&a[..an]), Operation::Read(&mut d[..dn])])?;
            Ok(get(&d[..dn], self.big_endian))
        }
        fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
            let mut b = [0u8; 8];
            let (an, dn) = (self.addr_bytes as usize, self.data_bytes as usize);
            put_be(&mut b[..an], offset);
            b[0] &= !self.read_bit;
            put(&mut b[an..an + dn], value, self.big_endian);
            self.spi.write(&b[..an + dn])
        }
    }
}
