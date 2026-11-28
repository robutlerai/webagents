//! A register-level simulator of the WIZnet W5500 behind an `embedded-hal`
//! `SpiDevice`, with real TCP behind its sockets: device code (the
//! [`crate::net::w5500`] driver and everything over it) runs unchanged against
//! it on a host.
//!
//! Modelled: SPI framing (address, control byte, variable-length data), the
//! common registers the driver uses, socket registers and commands (OPEN,
//! CONNECT, SEND, RECV, DISCON, CLOSE), 2 KB TX/RX buffers with address wrap,
//! TX free size and RX received size, CLOSE_WAIT when the peer closes. Not
//! modelled: interrupts, UDP/MACRAW, PHY, timing.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddrV4, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use embedded_hal::spi::{self, ErrorKind, Operation, SpiDevice};

use crate::net::w5500::{SR_CLOSED, SR_CLOSE_WAIT, SR_ESTABLISHED, SR_INIT, VERSION};

const BUF: usize = 2048;

struct Sock {
    regs: [u8; 0x30],
    tx: [u8; BUF],
    rx: [u8; BUF],
    tcp: Option<TcpStream>,
    pending: Arc<Mutex<VecDeque<u8>>>,
    eof: Arc<AtomicBool>,
}

impl Default for Sock {
    fn default() -> Self {
        Sock { regs: [0; 0x30], tx: [0; BUF], rx: [0; BUF], tcp: None, pending: Arc::default(), eof: Arc::default() }
    }
}

impl Sock {
    fn g16(&self, a: usize) -> u16 {
        u16::from_be_bytes([self.regs[a], self.regs[a + 1]])
    }
    fn s16(&mut self, a: usize, v: u16) {
        self.regs[a..a + 2].copy_from_slice(&v.to_be_bytes());
    }
    /// Derived registers, kept current: Sn_TX_FSR (0x20), Sn_RX_RSR (0x26).
    fn derive(&mut self) {
        let used_tx = self.g16(0x24).wrapping_sub(self.g16(0x22)) as usize;
        self.s16(0x20, (BUF - used_tx.min(BUF)) as u16);
        let rsr = self.g16(0x2A).wrapping_sub(self.g16(0x28));
        self.s16(0x26, rsr);
    }
    /// Moves bytes received from the network into the RX buffer, as space allows.
    fn pump(&mut self) {
        if self.tcp.is_some() {
            let mut q = self.pending.lock().unwrap();
            let used = self.g16(0x2A).wrapping_sub(self.g16(0x28)) as usize;
            let mut wr = self.g16(0x2A);
            let n = q.len().min(BUF - used);
            for b in q.drain(..n) {
                self.rx[wr as usize % BUF] = b;
                wr = wr.wrapping_add(1);
            }
            drop(q);
            self.s16(0x2A, wr);
            if self.eof.load(Ordering::Relaxed) && self.pending.lock().unwrap().is_empty() && self.regs[0x03] == SR_ESTABLISHED {
                self.regs[0x03] = SR_CLOSE_WAIT;
            }
        }
        self.derive();
    }
    fn command(&mut self, cr: u8) {
        match cr {
            0x01 if self.regs[0x00] & 0x0f == 0x01 => {
                for a in [0x22, 0x24, 0x28, 0x2A] {
                    self.s16(a, 0);
                }
                self.regs[0x03] = SR_INIT;
            }
            0x04 if self.regs[0x03] == SR_INIT => {
                let ip = [self.regs[0x0C], self.regs[0x0D], self.regs[0x0E], self.regs[0x0F]];
                let addr = SocketAddrV4::new(ip.into(), self.g16(0x10));
                match TcpStream::connect_timeout(&addr.into(), Duration::from_secs(5)) {
                    Ok(tcp) => {
                        let mut reader = tcp.try_clone().expect("clone");
                        let (pending, eof) = (Arc::new(Mutex::new(VecDeque::new())), Arc::new(AtomicBool::new(false)));
                        let (p2, e2) = (pending.clone(), eof.clone());
                        std::thread::spawn(move || {
                            let mut b = [0u8; 4096];
                            loop {
                                match reader.read(&mut b) {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => p2.lock().unwrap().extend(&b[..n]),
                                }
                            }
                            e2.store(true, Ordering::Relaxed);
                        });
                        self.tcp = Some(tcp);
                        self.pending = pending;
                        self.eof = eof;
                        self.regs[0x03] = SR_ESTABLISHED;
                    }
                    Err(_) => self.regs[0x03] = SR_CLOSED,
                }
            }
            0x20 => {
                let (rd, wr) = (self.g16(0x22), self.g16(0x24));
                let n = wr.wrapping_sub(rd) as usize;
                let data: Vec<u8> = (0..n).map(|i| self.tx[(rd as usize + i) % BUF]).collect();
                if let Some(t) = self.tcp.as_mut() {
                    if t.write_all(&data).is_err() {
                        self.regs[0x03] = SR_CLOSED;
                    }
                }
                self.s16(0x22, wr);
            }
            0x40 => {} // RECV: Sn_RX_RD was already written; RSR is derived.
            0x08 => {
                if let Some(t) = self.tcp.as_ref() {
                    let _ = t.shutdown(Shutdown::Write);
                }
                self.regs[0x03] = SR_CLOSED;
            }
            0x10 => {
                if let Some(t) = self.tcp.take() {
                    let _ = t.shutdown(Shutdown::Both);
                }
                self.regs[0x03] = SR_CLOSED;
            }
            _ => {}
        }
        self.regs[0x01] = 0;
    }
}

struct Chip {
    common: [u8; 0x40],
    socks: [Sock; 8],
    pub transactions: u64,
}

/// The simulated chip. Clone it to share one chip between handles.
#[derive(Clone)]
pub struct W5500Sim {
    chip: Arc<Mutex<Chip>>,
}

impl Default for W5500Sim {
    fn default() -> Self {
        let mut common = [0u8; 0x40];
        common[0x39] = VERSION;
        W5500Sim { chip: Arc::new(Mutex::new(Chip { common, socks: Default::default(), transactions: 0 })) }
    }
}

impl W5500Sim {
    /// A chip after reset: every socket closed, the version register reading 0x04.
    pub fn new() -> Self {
        Self::default()
    }
    /// SPI transactions served so far (one per chip-select window), to compare
    /// the cost of driver changes in bus traffic.
    pub fn transactions(&self) -> u64 {
        self.chip.lock().unwrap().transactions
    }
}

impl spi::ErrorType for W5500Sim {
    type Error = ErrorKind;
}

impl SpiDevice for W5500Sim {
    fn transaction(&mut self, operations: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        let mut chip = self.chip.lock().unwrap();
        chip.transactions += 1;
        let mut header: Vec<u8> = Vec::with_capacity(3);
        let mut pos = 0usize; // data bytes done
        for op in operations.iter_mut() {
            match op {
                Operation::Write(bytes) => {
                    for &b in bytes.iter() {
                        if header.len() < 3 {
                            header.push(b);
                            continue;
                        }
                        let (addr, bsb, write) = decode(&header);
                        if !write {
                            return Err(ErrorKind::Other);
                        }
                        access(&mut chip, bsb, addr.wrapping_add(pos as u16), Some(b));
                        pos += 1;
                    }
                }
                Operation::Read(buf) => {
                    if header.len() < 3 {
                        return Err(ErrorKind::Other);
                    }
                    let (addr, bsb, write) = decode(&header);
                    if write {
                        return Err(ErrorKind::Other);
                    }
                    for slot in buf.iter_mut() {
                        *slot = access(&mut chip, bsb, addr.wrapping_add(pos as u16), None);
                        pos += 1;
                    }
                }
                Operation::DelayNs(_) => {}
                _ => return Err(ErrorKind::Other),
            }
        }
        Ok(())
    }
}

fn decode(h: &[u8]) -> (u16, u8, bool) {
    (u16::from_be_bytes([h[0], h[1]]), h[2] >> 3, h[2] & 0x04 != 0)
}

/// One byte of access; `Some(v)` writes, `None` reads.
fn access(chip: &mut Chip, bsb: u8, addr: u16, val: Option<u8>) -> u8 {
    if bsb == 0 {
        let a = addr as usize;
        if a >= chip.common.len() {
            return 0;
        }
        match val {
            Some(v) if a == 0 => chip.common[0] = v & 0x7f, // soft reset clears itself
            Some(v) if a != 0x39 => chip.common[a] = v,
            Some(_) => {}
            None => return chip.common[a],
        }
        return 0;
    }
    let (n, kind) = (((bsb - 1) / 4) as usize, (bsb - 1) % 4);
    if n >= 8 {
        return 0;
    }
    let s = &mut chip.socks[n];
    match kind {
        0 => {
            let a = addr as usize;
            if a >= s.regs.len() {
                return 0;
            }
            match val {
                Some(v) if a == 0x01 => s.command(v),
                Some(v) if !matches!(a, 0x03 | 0x20 | 0x21 | 0x26 | 0x27 | 0x2A | 0x2B | 0x22 | 0x23) => {
                    s.regs[a] = v;
                    s.derive();
                }
                Some(_) => {}
                None => {
                    s.pump();
                    return s.regs[a];
                }
            }
            0
        }
        1 => {
            if let Some(v) = val {
                s.tx[addr as usize % BUF] = v;
            }
            0
        }
        2 => s.rx[addr as usize % BUF],
        _ => 0,
    }
}
