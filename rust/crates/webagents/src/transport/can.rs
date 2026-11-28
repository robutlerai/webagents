//! Registers over CAN (SocketCAN on Linux): one request frame, one answer frame,
//! classic CAN so it fits any controller.
//!
//! ```text
//! request  id 0x600 + node   [op, off_lo, off_hi, v0, v1, v2, v3]   op 0x40 read, 0x23 write
//! answer   id 0x580 + node   [st, off_lo, off_hi, v0, v1, v2, v3]   st 0x43 read ok, 0x60 write ok, 0x80 refused
//! ```
//!
//! Values are little-endian. The layout follows the shape of a CANopen
//! expedited SDO so it reads familiarly on a bus analyser, without claiming to
//! be CANopen (no object dictionary, offsets are register offsets).

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use socketcan::{CanFrame, CanSocket, EmbeddedFrame, Frame, Socket, StandardId};

use super::sim::SharedModel;
use crate::bus::RegisterBus;

/// CAN id of a request to node `n` is `REQ_BASE + n`.
pub const REQ_BASE: u16 = 0x600;
/// CAN id of node `n`'s answer is `ANS_BASE + n`.
pub const ANS_BASE: u16 = 0x580;
const OP_READ: u8 = 0x40;
const OP_WRITE: u8 = 0x23;
const ST_READ: u8 = 0x43;
const ST_WRITE: u8 = 0x60;
const ST_REFUSED: u8 = 0x80;

/// Why a register access over CAN failed.
#[derive(Debug)]
pub enum CanError {
    /// The CAN socket failed.
    Io(io::Error),
    /// No answer frame from the node within [`CanRegs::timeout`].
    Timeout,
    /// The node answered with status 0x80: it does not accept the request.
    Refused,
    /// Register offsets above 0xffff do not fit the frame. Also returned,
    /// with the offset asked for, when the node id is above 127.
    Offset(u32),
}

impl From<io::Error> for CanError {
    fn from(e: io::Error) -> Self {
        CanError::Io(e)
    }
}

/// The gateway side.
pub struct CanRegs {
    /// The SocketCAN socket, bound to one interface.
    pub sock: CanSocket,
    /// The part's node id, 0 to 127.
    pub node: u8,
    /// Longest wait for the answer to one request (default 200 ms).
    pub timeout: Duration,
}

impl CanRegs {
    /// Opens the SocketCAN interface `iface` (`can0`, `vcan0`) for register
    /// access to `node`. The interface must already be up, with its bit rate set.
    pub fn open(iface: &str, node: u8) -> io::Result<Self> {
        let sock = CanSocket::open(iface)?;
        Ok(CanRegs { sock, node, timeout: Duration::from_millis(200) })
    }

    fn exchange(&mut self, op: u8, offset: u32, value: u32) -> Result<u32, CanError> {
        if offset > 0xffff || self.node > 0x7f {
            return Err(CanError::Offset(offset));
        }
        let v = value.to_le_bytes();
        let data = [op, offset as u8, (offset >> 8) as u8, v[0], v[1], v[2], v[3]];
        let id = StandardId::new(REQ_BASE + self.node as u16).expect("node is 7 bits");
        let frame = CanFrame::new(id, &data).expect("7 bytes fit a classic frame");
        self.sock.write_frame(&frame)?;
        let deadline = Instant::now() + self.timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(CanError::Timeout);
            }
            self.sock.set_read_timeout(left)?;
            let f = match self.sock.read_frame() {
                Ok(f) => f,
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => return Err(CanError::Timeout),
                Err(e) => return Err(e.into()),
            };
            // Other traffic on the bus is skipped, never taken for the answer.
            if f.raw_id() != (ANS_BASE + self.node as u16) as u32 {
                continue;
            }
            let d = f.data();
            if d.len() < 7 || (d[1] as u32 | (d[2] as u32) << 8) != offset {
                continue;
            }
            return match d[0] {
                ST_REFUSED => Err(CanError::Refused),
                ST_READ if op == OP_READ => Ok(u32::from_le_bytes([d[3], d[4], d[5], d[6]])),
                ST_WRITE if op == OP_WRITE => Ok(0),
                _ => continue,
            };
        }
    }
}

impl RegisterBus for CanRegs {
    type Error = CanError;
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
        self.exchange(OP_READ, offset, 0)
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
        self.exchange(OP_WRITE, offset, value).map(|_| ())
    }
}

/// The part side, for twins and tests: answers requests to `node` from `model`.
pub fn serve(iface: &str, node: u8, model: SharedModel, stop: Arc<AtomicBool>) -> io::Result<()> {
    let sock = CanSocket::open(iface)?;
    sock.set_read_timeout(Duration::from_millis(50))?;
    while !stop.load(Ordering::Relaxed) {
        let f = match sock.read_frame() {
            Ok(f) => f,
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => continue,
            Err(e) => return Err(e),
        };
        if f.raw_id() != (REQ_BASE + node as u16) as u32 || f.data().len() < 7 {
            continue;
        }
        let d = f.data();
        let off = d[1] as u32 | (d[2] as u32) << 8;
        let mut m = model.lock().unwrap_or_else(|p| p.into_inner());
        let (st, val) = match d[0] {
            OP_READ => {
                m.reads += 1;
                (ST_READ, m.get(off))
            }
            OP_WRITE => {
                m.writes += 1;
                m.regs.insert(off, u32::from_le_bytes([d[3], d[4], d[5], d[6]]));
                (ST_WRITE, 0)
            }
            _ => (ST_REFUSED, 0),
        };
        drop(m);
        let v = val.to_le_bytes();
        let id = StandardId::new(ANS_BASE + node as u16).expect("node is 7 bits");
        let frame = CanFrame::new(id, &[st, d[1], d[2], v[0], v[1], v[2], v[3]]).expect("fits");
        sock.write_frame(&frame)?;
    }
    Ok(())
}
