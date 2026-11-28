//! Simulated parts: a register file shared between the gateway side (a
//! [`RegisterBus`], or an `embedded-hal` I2C or SPI device) and whatever plays
//! the part. With these the whole path, from tool call through the bus
//! adapter's byte framing to the register, runs without hardware.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use embedded_hal::i2c::{self, ErrorKind as I2cErrorKind, NoAcknowledgeSource};
use embedded_hal::spi::{self, ErrorKind as SpiErrorKind, Operation as SpiOp};

use crate::bus::RegisterBus;
use crate::regspec::Device;

/// A part's registers, by offset.
#[derive(Debug, Clone, Default)]
pub struct RegModel {
    /// Register values by offset. An offset that is not present reads as 0.
    pub regs: HashMap<u32, u32>,
    /// Number of register reads served, on any of the simulated buses.
    pub reads: u64,
    /// Number of register writes taken.
    pub writes: u64,
}

/// A register model shared between the side that plays the part and the side
/// that drives it (and a test that inspects it afterwards).
pub type SharedModel = Arc<Mutex<RegModel>>;

impl RegModel {
    /// Every register of `dev` at its reset value.
    pub fn at_reset(dev: &Device) -> SharedModel {
        Arc::new(Mutex::new(RegModel { regs: dev.registers.iter().map(|r| (r.offset, r.reset)).collect(), reads: 0, writes: 0 }))
    }
    /// The register at `off`, or 0 when nothing was ever stored there.
    pub fn get(&self, off: u32) -> u32 {
        self.regs.get(&off).copied().unwrap_or(0)
    }
}

fn lock(m: &SharedModel) -> std::sync::MutexGuard<'_, RegModel> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Direct register access to a model.
#[derive(Clone)]
pub struct SimBus(pub SharedModel);

impl RegisterBus for SimBus {
    type Error = core::convert::Infallible;
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
        let mut m = lock(&self.0);
        m.reads += 1;
        Ok(m.get(offset))
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
        let mut m = lock(&self.0);
        m.writes += 1;
        m.regs.insert(offset, value);
        Ok(())
    }
}

fn be(b: &[u8]) -> u32 {
    b.iter().fold(0u32, |a, x| (a << 8) | *x as u32)
}
fn decode(b: &[u8], big_endian: bool) -> u32 {
    if big_endian {
        be(b)
    } else {
        b.iter().rev().fold(0u32, |a, x| (a << 8) | *x as u32)
    }
}
fn encode(v: u32, out: &mut [u8], big_endian: bool) {
    let n = out.len();
    for (i, o) in out.iter_mut().enumerate() {
        let shift = if big_endian { 8 * (n - 1 - i) } else { 8 * i };
        *o = (v >> shift) as u8;
    }
}

/// An I2C target at `address` holding a register model: a write of
/// `addr_bytes` sets the register pointer, further bytes write the register,
/// a read returns it.
pub struct SimI2c {
    /// The registers behind this target.
    pub model: SharedModel,
    /// The 7-bit address the target answers to; any other address is not acknowledged.
    pub address: u8,
    /// Bytes of register address the target expects, most significant first.
    pub addr_bytes: usize,
    /// Bytes of register value; a write of any other length is an error.
    pub data_bytes: usize,
    /// Byte order of the value: most significant byte first when true.
    pub big_endian: bool,
    pointer: u32,
}

impl SimI2c {
    /// A target at `address` over `model`, with its register pointer at 0.
    pub fn new(model: SharedModel, address: u8, addr_bytes: usize, data_bytes: usize, big_endian: bool) -> Self {
        SimI2c { model, address, addr_bytes, data_bytes, big_endian, pointer: 0 }
    }
}

impl i2c::ErrorType for SimI2c {
    type Error = I2cErrorKind;
}

impl i2c::I2c for SimI2c {
    fn transaction(&mut self, address: u8, operations: &mut [i2c::Operation<'_>]) -> Result<(), Self::Error> {
        if address != self.address {
            return Err(I2cErrorKind::NoAcknowledge(NoAcknowledgeSource::Address));
        }
        for op in operations {
            match op {
                i2c::Operation::Write(bytes) => {
                    if bytes.len() < self.addr_bytes {
                        return Err(I2cErrorKind::Other);
                    }
                    self.pointer = be(&bytes[..self.addr_bytes]);
                    let data = &bytes[self.addr_bytes..];
                    if !data.is_empty() {
                        if data.len() != self.data_bytes {
                            return Err(I2cErrorKind::Other);
                        }
                        let mut m = lock(&self.model);
                        m.writes += 1;
                        m.regs.insert(self.pointer, decode(data, self.big_endian));
                    }
                }
                i2c::Operation::Read(buf) => {
                    let mut m = lock(&self.model);
                    m.reads += 1;
                    let v = m.get(self.pointer);
                    encode(v, buf, self.big_endian);
                }
            }
        }
        Ok(())
    }
}

/// A SPI device holding a register model: address bytes (top bit of the first
/// set for a read), then the data.
pub struct SimSpi {
    /// The registers behind this device.
    pub model: SharedModel,
    /// Bytes of register address the device expects, most significant first.
    pub addr_bytes: usize,
    /// Bytes of register value; a write of any other length is an error.
    pub data_bytes: usize,
    /// Byte order of the value: most significant byte first when true.
    pub big_endian: bool,
    /// The bit of the first address byte that marks a read (`0x80` usually).
    pub read_bit: u8,
}

impl spi::ErrorType for SimSpi {
    type Error = SpiErrorKind;
}

impl spi::SpiDevice for SimSpi {
    fn transaction(&mut self, operations: &mut [SpiOp<'_, u8>]) -> Result<(), Self::Error> {
        let mut pointer: Option<(u32, bool)> = None;
        for op in operations {
            match op {
                SpiOp::Write(bytes) if pointer.is_none() => {
                    if bytes.len() < self.addr_bytes {
                        return Err(SpiErrorKind::Other);
                    }
                    let mut a = bytes[..self.addr_bytes].to_vec();
                    let read = a[0] & self.read_bit != 0;
                    a[0] &= !self.read_bit;
                    let reg = be(&a);
                    pointer = Some((reg, read));
                    let data = &bytes[self.addr_bytes..];
                    if !read && !data.is_empty() {
                        if data.len() != self.data_bytes {
                            return Err(SpiErrorKind::Other);
                        }
                        let mut m = lock(&self.model);
                        m.writes += 1;
                        m.regs.insert(reg, decode(data, self.big_endian));
                    }
                }
                SpiOp::Read(buf) => {
                    let Some((reg, true)) = pointer else { return Err(SpiErrorKind::Other) };
                    let mut m = lock(&self.model);
                    m.reads += 1;
                    let v = m.get(reg);
                    encode(v, buf, self.big_endian);
                }
                SpiOp::DelayNs(_) => {}
                _ => return Err(SpiErrorKind::Other),
            }
        }
        Ok(())
    }
}
