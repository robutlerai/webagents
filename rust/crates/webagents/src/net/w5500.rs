//! Driver for the WIZnet W5500 Ethernet controller over SPI: TCP client
//! sockets with the TCP/IP stack in the chip, so the microcontroller holds no
//! network stack of its own. As with the ESP-AT co-processor, TLS stays on
//! the microcontroller: each socket is an `embedded-io` stream that a TLS
//! session and the WebSocket run over.
//!
//! Frames follow the W5500 datasheet (v1.1): a 16-bit address, a control byte
//! (block select, read/write, variable-length mode) and the data, inside one
//! chip-select window (an `embedded-hal` `SpiDevice` transaction). Socket
//! buffers are the reset size, 2 KB each way; the chip wraps buffer addresses
//! itself, so pointers are used as addresses directly.
//!
//! The W5500 has no DNS resolver: hosts are given as IPv4 addresses.

use core::cell::RefCell;

use embedded_hal::spi::{Operation, SpiDevice};

// Common register block.
const MR: u16 = 0x0000;
const GAR: u16 = 0x0001;
const SUBR: u16 = 0x0005;
const SHAR: u16 = 0x0009;
const SIPR: u16 = 0x000F;
const VERSIONR: u16 = 0x0039;
/// VERSIONR of every W5500.
pub const VERSION: u8 = 0x04;

// Socket register offsets.
const SN_MR: u16 = 0x00;
const SN_CR: u16 = 0x01;
const SN_SR: u16 = 0x03;
const SN_PORT: u16 = 0x04;
const SN_DIPR: u16 = 0x0C;
const SN_DPORT: u16 = 0x10;
const SN_TX_FSR: u16 = 0x20;
const SN_TX_WR: u16 = 0x24;
const SN_RX_RSR: u16 = 0x26;
const SN_RX_RD: u16 = 0x28;

const MR_TCP: u8 = 0x01;
const CR_OPEN: u8 = 0x01;
const CR_CONNECT: u8 = 0x04;
const CR_DISCON: u8 = 0x08;
const CR_CLOSE: u8 = 0x10;
const CR_SEND: u8 = 0x20;
const CR_RECV: u8 = 0x40;

/// Socket status (`Sn_SR`): closed.
pub const SR_CLOSED: u8 = 0x00;
/// Socket status: opened in TCP mode, not yet connected.
pub const SR_INIT: u8 = 0x13;
/// Socket status: connected; data can flow both ways.
pub const SR_ESTABLISHED: u8 = 0x17;
/// Socket status: the peer closed its side. Data already received can still be read.
pub const SR_CLOSE_WAIT: u8 = 0x1C;

/// Block select values (control byte bits 7..3).
pub const fn bsb_common() -> u8 {
    0
}
/// Block select of socket `n`'s register block (`n` is 0 to 7).
pub const fn bsb_socket(n: u8) -> u8 {
    n * 4 + 1
}
/// Block select of socket `n`'s transmit buffer.
pub const fn bsb_tx(n: u8) -> u8 {
    n * 4 + 2
}
/// Block select of socket `n`'s receive buffer.
pub const fn bsb_rx(n: u8) -> u8 {
    n * 4 + 3
}

/// Control byte: block, read (0) or write (1), variable-length mode (00).
pub const fn control(bsb: u8, write: bool) -> u8 {
    (bsb << 3) | ((write as u8) << 2)
}

/// Why a W5500 operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum W5500Error {
    /// The SPI transaction itself failed.
    Spi,
    /// VERSIONR did not read 0x04: no W5500 on this bus, or the wiring is wrong.
    NotW5500(u8),
    /// The chip did not take a command, connect, or free buffer space within
    /// [`W5500::timeout_ms`].
    Timeout,
    /// The connection was refused or dropped.
    Closed,
}

/// Milliseconds since an arbitrary start, for timeouts.
pub type Clock = fn() -> u32;

/// The driver: one W5500 on one SPI device. `no_std`, no heap, no state
/// beyond the SPI handle. Sockets are numbered 0 to 7.
pub struct W5500<S> {
    spi: S,
    now: Clock,
    /// Deadline for a command, a connect or waiting for transmit space, in
    /// milliseconds (default 10000).
    pub timeout_ms: u32,
}

impl<S: SpiDevice> W5500<S> {
    /// A driver over `spi`, with `now` as its millisecond clock. Nothing is sent
    /// until [`W5500::init`].
    pub fn new(spi: S, now: Clock) -> Self {
        W5500 { spi, now, timeout_ms: 10_000 }
    }

    /// Reads `buf.len()` bytes at `addr` of block `bsb`, in one chip-select window.
    pub fn read(&mut self, bsb: u8, addr: u16, buf: &mut [u8]) -> Result<(), W5500Error> {
        let h = [(addr >> 8) as u8, addr as u8, control(bsb, false)];
        self.spi.transaction(&mut [Operation::Write(&h), Operation::Read(buf)]).map_err(|_| W5500Error::Spi)
    }

    /// Writes `data` at `addr` of block `bsb`, in one chip-select window.
    pub fn write(&mut self, bsb: u8, addr: u16, data: &[u8]) -> Result<(), W5500Error> {
        let h = [(addr >> 8) as u8, addr as u8, control(bsb, true)];
        self.spi.transaction(&mut [Operation::Write(&h), Operation::Write(data)]).map_err(|_| W5500Error::Spi)
    }

    fn r8(&mut self, bsb: u8, addr: u16) -> Result<u8, W5500Error> {
        let mut b = [0u8; 1];
        self.read(bsb, addr, &mut b)?;
        Ok(b[0])
    }

    /// A 16-bit counter register, read until two reads agree (the datasheet's
    /// advice: the chip may update it between the two byte reads).
    fn r16_stable(&mut self, bsb: u8, addr: u16) -> Result<u16, W5500Error> {
        let mut last = None;
        loop {
            let mut b = [0u8; 2];
            self.read(bsb, addr, &mut b)?;
            let v = u16::from_be_bytes(b);
            if last == Some(v) {
                return Ok(v);
            }
            last = Some(v);
        }
    }

    fn w16(&mut self, bsb: u8, addr: u16, v: u16) -> Result<(), W5500Error> {
        self.write(bsb, addr, &v.to_be_bytes())
    }

    fn command(&mut self, sock: u8, cr: u8) -> Result<(), W5500Error> {
        self.write(bsb_socket(sock), SN_CR, &[cr])?;
        let start = (self.now)();
        // The chip clears Sn_CR once it has taken the command.
        while self.r8(bsb_socket(sock), SN_CR)? != 0 {
            if (self.now)().wrapping_sub(start) > self.timeout_ms {
                return Err(W5500Error::Timeout);
            }
        }
        Ok(())
    }

    /// The socket's status register (`Sn_SR`): one of the `SR_*` values, or a
    /// transient state while the chip connects or closes.
    pub fn status(&mut self, sock: u8) -> Result<u8, W5500Error> {
        self.r8(bsb_socket(sock), SN_SR)
    }

    /// Soft reset, check the chip, set the addresses.
    pub fn init(&mut self, mac: [u8; 6], ip: [u8; 4], gateway: [u8; 4], mask: [u8; 4]) -> Result<(), W5500Error> {
        self.write(bsb_common(), MR, &[0x80])?;
        let start = (self.now)();
        while self.r8(bsb_common(), MR)? & 0x80 != 0 {
            if (self.now)().wrapping_sub(start) > self.timeout_ms {
                return Err(W5500Error::Timeout);
            }
        }
        let v = self.r8(bsb_common(), VERSIONR)?;
        if v != VERSION {
            return Err(W5500Error::NotW5500(v));
        }
        self.write(bsb_common(), GAR, &gateway)?;
        self.write(bsb_common(), SUBR, &mask)?;
        self.write(bsb_common(), SHAR, &mac)?;
        self.write(bsb_common(), SIPR, &ip)
    }

    /// Opens socket `sock` as a TCP client to `ip:port` from `local_port`.
    pub fn connect(&mut self, sock: u8, ip: [u8; 4], port: u16, local_port: u16) -> Result<(), W5500Error> {
        self.command(sock, CR_CLOSE)?;
        self.write(bsb_socket(sock), SN_MR, &[MR_TCP])?;
        self.w16(bsb_socket(sock), SN_PORT, local_port)?;
        self.command(sock, CR_OPEN)?;
        if self.status(sock)? != SR_INIT {
            return Err(W5500Error::Closed);
        }
        self.write(bsb_socket(sock), SN_DIPR, &ip)?;
        self.w16(bsb_socket(sock), SN_DPORT, port)?;
        self.command(sock, CR_CONNECT)?;
        let start = (self.now)();
        loop {
            match self.status(sock)? {
                SR_ESTABLISHED => return Ok(()),
                SR_CLOSED => return Err(W5500Error::Closed),
                _ => {}
            }
            if (self.now)().wrapping_sub(start) > self.timeout_ms {
                return Err(W5500Error::Timeout);
            }
        }
    }

    /// Queues and sends `data`, waiting for buffer space as needed.
    pub fn send(&mut self, sock: u8, mut data: &[u8]) -> Result<(), W5500Error> {
        while !data.is_empty() {
            let start = (self.now)();
            let free = loop {
                let st = self.status(sock)?;
                if st != SR_ESTABLISHED && st != SR_CLOSE_WAIT {
                    return Err(W5500Error::Closed);
                }
                let f = self.r16_stable(bsb_socket(sock), SN_TX_FSR)? as usize;
                if f > 0 {
                    break f;
                }
                if (self.now)().wrapping_sub(start) > self.timeout_ms {
                    return Err(W5500Error::Timeout);
                }
            };
            let n = free.min(data.len());
            let wr = self.r16_stable(bsb_socket(sock), SN_TX_WR)?;
            self.write(bsb_tx(sock), wr, &data[..n])?;
            self.w16(bsb_socket(sock), SN_TX_WR, wr.wrapping_add(n as u16))?;
            self.command(sock, CR_SEND)?;
            data = &data[n..];
        }
        Ok(())
    }

    /// Bytes waiting in socket `sock`.
    pub fn available(&mut self, sock: u8) -> Result<usize, W5500Error> {
        Ok(self.r16_stable(bsb_socket(sock), SN_RX_RSR)? as usize)
    }

    /// Reads up to `buf.len()` waiting bytes.
    pub fn recv(&mut self, sock: u8, buf: &mut [u8]) -> Result<usize, W5500Error> {
        let n = self.available(sock)?.min(buf.len());
        if n == 0 {
            return Ok(0);
        }
        let rd = self.r16_stable(bsb_socket(sock), SN_RX_RD)?;
        self.read(bsb_rx(sock), rd, &mut buf[..n])?;
        self.w16(bsb_socket(sock), SN_RX_RD, rd.wrapping_add(n as u16))?;
        self.command(sock, CR_RECV)?;
        Ok(n)
    }

    /// Disconnects (a TCP FIN) and closes the socket. The disconnect's result is
    /// ignored, so closing a socket that is already closed succeeds.
    pub fn close(&mut self, sock: u8) -> Result<(), W5500Error> {
        let _ = self.command(sock, CR_DISCON);
        self.command(sock, CR_CLOSE)
    }
}

/// One socket of a shared W5500, as an `embedded-io` stream.
pub struct Socket<'a, S> {
    /// The shared driver. A `RefCell` because several sockets take turns on one
    /// chip, on one core.
    pub chip: &'a RefCell<W5500<S>>,
    /// The socket number, 0 to 7, already connected with [`W5500::connect`].
    pub sock: u8,
    /// Longest wait for data in one `read`, in milliseconds.
    pub timeout_ms: u32,
}

/// The error of a [`Socket`], wrapping the driver's so it implements
/// `embedded_io::Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SocketError(pub W5500Error);

impl core::fmt::Display for SocketError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "W5500 socket: {:?}", self.0)
    }
}
impl core::error::Error for SocketError {}
impl embedded_io::Error for SocketError {
    fn kind(&self) -> embedded_io::ErrorKind {
        match self.0 {
            W5500Error::Timeout => embedded_io::ErrorKind::TimedOut,
            W5500Error::Closed => embedded_io::ErrorKind::NotConnected,
            _ => embedded_io::ErrorKind::Other,
        }
    }
}

impl<S: SpiDevice> embedded_io::ErrorType for Socket<'_, S> {
    type Error = SocketError;
}

impl<S: SpiDevice> embedded_io::Read for Socket<'_, S> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut c = self.chip.borrow_mut();
        let start = (c.now)();
        loop {
            let n = c.recv(self.sock, buf).map_err(SocketError)?;
            if n > 0 {
                return Ok(n);
            }
            // Peer closed and nothing left: end of stream.
            let st = c.status(self.sock).map_err(SocketError)?;
            if matches!(st, SR_CLOSE_WAIT | SR_CLOSED) && c.available(self.sock).map_err(SocketError)? == 0 {
                return Ok(0);
            }
            if (c.now)().wrapping_sub(start) > self.timeout_ms {
                return Err(SocketError(W5500Error::Timeout));
            }
        }
    }
}

impl<S: SpiDevice> embedded_io::Write for Socket<'_, S> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        self.chip.borrow_mut().send(self.sock, buf).map_err(SocketError)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}
