//! Registers over a serial line: a small text protocol a part's firmware can
//! serve from a UART in a few hundred bytes, and that a person can type at a
//! terminal.
//!
//! ```text
//! gateway -> part   R 0000000c\n             read the register at 0x0c
//! part -> gateway   V 000001f4\n             its value
//! gateway -> part   W 0000000c 000001f4\n    write it
//! part -> gateway   K\n                      done
//! part -> gateway   E <reason>\n             refused or failed
//! ```
//!
//! Hex is fixed-width and lower case on output; input accepts either case.
//! One request is in flight at a time; the gateway waits for the answer (or
//! the port's timeout) before sending the next. Lines that are not answers are
//! skipped as the device's own output (logs and telemetry often share the
//! UART), so a device should start each answer on a fresh line.

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::sim::SharedModel;
use crate::bus::RegisterBus;

/// Why a register access over the serial line failed.
#[derive(Debug)]
pub enum UartError {
    /// The port failed, closed, or timed out before an answer.
    Io(io::Error),
    /// The part answered something that is not the protocol.
    Protocol(String),
    /// The part answered `E <reason>`.
    Refused(String),
}

impl From<io::Error> for UartError {
    fn from(e: io::Error) -> Self {
        UartError::Io(e)
    }
}

/// The gateway side: a [`RegisterBus`] over any byte stream with a read timeout.
pub struct UartRegs<P> {
    /// The byte stream: a serial port, a PTY, a TCP stream. Its read timeout is
    /// the answer timeout of every request.
    pub port: P,
}

// The longest line is the signature line of `D`: "S " + created (up to 20 digits) + " " + 88 base64 characters.
const MAX_LINE: usize = 160;

fn read_line<P: Read>(p: &mut P) -> io::Result<String> {
    let mut line = Vec::with_capacity(24);
    let mut b = [0u8; 1];
    loop {
        match p.read(&mut b) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the port closed")),
            Ok(_) => {
                if b[0] == b'\n' {
                    break;
                }
                if b[0] != b'\r' {
                    line.push(b[0]);
                }
                if line.len() > MAX_LINE {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "line too long"));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    String::from_utf8(line).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "not text"))
}

/// Lines skipped while waiting for one answer before giving up.
const MAX_SKIPPED_LINES: usize = 64;

impl<P: Read + Write> UartRegs<P> {
    /// The next protocol answer. A device often logs on the same UART as its
    /// register monitor, so a line that is not an answer (`V ...`, `K`, `E`,
    /// `E ...`) is skipped as the device's own output, never taken for one.
    fn answer(&mut self) -> Result<String, UartError> {
        for _ in 0..MAX_SKIPPED_LINES {
            let line = read_line(&mut self.port)?;
            if line == "E" {
                return Err(UartError::Refused(String::new()));
            }
            if let Some(r) = line.strip_prefix("E ") {
                return Err(UartError::Refused(r.to_string()));
            }
            if line == "K" || line.starts_with("V ") {
                return Ok(line);
            }
        }
        Err(UartError::Protocol(format!("no answer within {MAX_SKIPPED_LINES} lines of other output")))
    }
}

impl<P: Read + Write> RegisterBus for UartRegs<P> {
    type Error = UartError;
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
        writeln!(self.port, "R {offset:08x}")?;
        self.port.flush()?;
        let line = self.answer()?;
        let hex = line.strip_prefix("V ").ok_or_else(|| UartError::Protocol(line.clone()))?;
        u32::from_str_radix(hex.trim(), 16).map_err(|_| UartError::Protocol(line.clone()))
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
        writeln!(self.port, "W {offset:08x} {value:08x}")?;
        self.port.flush()?;
        match self.answer()?.as_str() {
            "K" => Ok(()),
            other => Err(UartError::Protocol(other.to_string())),
        }
    }
}

/// A device's signed tool description as read off the line.
#[derive(Debug, Clone)]
pub struct SignedDescription {
    /// The tool description, the exact bytes the device sent (and signed).
    pub tools: Vec<u8>,
    /// When the device signed it, as the device states it (Unix seconds, or 0
    /// for a device without a clock).
    pub created: u64,
    /// The Ed25519 signature, base64 (88 characters).
    pub signature: String,
}

impl<P: Read + Write> UartRegs<P> {
    /// `D`: asks the device for its signed tool description. The answer is
    /// `T <n>`, exactly n bytes, a newline, then `S <created> <signature>`. Device
    /// output before the `T` line is skipped, as for register answers.
    pub fn describe(&mut self) -> Result<SignedDescription, UartError> {
        writeln!(self.port, "D")?;
        self.port.flush()?;
        let mut n = None;
        for _ in 0..MAX_SKIPPED_LINES {
            let line = read_line(&mut self.port)?;
            if line == "E" || line.starts_with("E ") {
                return Err(UartError::Refused(line));
            }
            if let Some(len) = line.strip_prefix("T ") {
                n = Some(len.trim().parse::<usize>().map_err(|_| UartError::Protocol(line.clone()))?);
                break;
            }
        }
        let n = n.ok_or_else(|| UartError::Protocol("no tool description".into()))?;
        if n > 256 * 1024 {
            return Err(UartError::Protocol(format!("a {n}-byte tool description")));
        }
        let mut tools = vec![0u8; n];
        let mut got = 0;
        while got < n {
            match self.port.read(&mut tools[got..]) {
                Ok(0) => return Err(UartError::Io(io::Error::new(io::ErrorKind::UnexpectedEof, "the port closed"))),
                Ok(k) => got += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        let end = read_line(&mut self.port)?;
        if !end.is_empty() {
            return Err(UartError::Protocol(format!("expected the end of the description, got {end:?}")));
        }
        let s = read_line(&mut self.port)?;
        let mut parts = s.strip_prefix("S ").ok_or_else(|| UartError::Protocol(s.clone()))?.split(' ');
        let created = parts.next().and_then(|c| c.parse().ok()).ok_or_else(|| UartError::Protocol(s.clone()))?;
        let signature = parts.next().ok_or_else(|| UartError::Protocol(s.clone()))?.to_string();
        Ok(SignedDescription { tools, created, signature })
    }
}

/// What a twin answers to `D`: the description bytes, `created` and the signature.
pub type Described = Option<(Vec<u8>, u64, String)>;

/// The part side, for twins and tests: serves `model` on `port` until `stop`.
/// The port should have a read timeout so `stop` is noticed.
pub fn serve<P: Read + Write>(port: P, model: SharedModel, stop: Arc<AtomicBool>) -> io::Result<()> {
    serve_described(port, model, stop, None)
}

/// As [`serve`], also answering `D` with a signed tool description when one is given.
pub fn serve_described<P: Read + Write>(mut port: P, model: SharedModel, stop: Arc<AtomicBool>, described: Described) -> io::Result<()> {
    while !stop.load(Ordering::Relaxed) {
        let line = match read_line(&mut port) {
            Ok(l) => l,
            Err(e) if matches!(e.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock) => continue,
            Err(e) => return Err(e),
        };
        if line.trim() == "D" {
            match &described {
                Some((tools, created, sig)) => {
                    port.write_all(format!("T {}\n", tools.len()).as_bytes())?;
                    port.write_all(tools)?;
                    port.write_all(format!("\nS {created} {sig}\n").as_bytes())?;
                }
                None => port.write_all(b"E no signed description\n")?,
            }
            port.flush()?;
            continue;
        }
        let reply = handle(&line, &model);
        port.write_all(reply.as_bytes())?;
        port.flush()?;
    }
    Ok(())
}

fn handle(line: &str, model: &SharedModel) -> String {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let hex = |s: &str| u32::from_str_radix(s, 16).ok();
    let mut m = model.lock().unwrap_or_else(|p| p.into_inner());
    match parts.as_slice() {
        ["R", off] => match hex(off) {
            Some(o) => {
                m.reads += 1;
                format!("V {:08x}\n", m.get(o))
            }
            None => "E bad offset\n".into(),
        },
        ["W", off, val] => match (hex(off), hex(val)) {
            (Some(o), Some(v)) => {
                m.writes += 1;
                m.regs.insert(o, v);
                "K\n".into()
            }
            _ => "E bad number\n".into(),
        },
        _ => "E unknown request\n".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::sim::RegModel;

    /// A loopback "port": requests go to the handler, answers come back.
    struct Loop {
        model: SharedModel,
        pending: Vec<u8>,
        out: std::collections::VecDeque<u8>,
    }
    impl Write for Loop {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            for &c in b {
                if c == b'\n' {
                    let line = String::from_utf8(std::mem::take(&mut self.pending)).unwrap();
                    self.out.extend(handle(&line, &self.model).bytes());
                } else {
                    self.pending.push(c);
                }
            }
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl Read for Loop {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            match self.out.pop_front() {
                Some(c) => {
                    b[0] = c;
                    Ok(1)
                }
                None => Err(io::Error::new(io::ErrorKind::TimedOut, "empty")),
            }
        }
    }

    #[test]
    fn round_trip() {
        let model = std::sync::Arc::new(std::sync::Mutex::new(RegModel::default()));
        let mut bus = UartRegs { port: Loop { model: model.clone(), pending: vec![], out: Default::default() } };
        bus.write(0x0c, 32, 0x1f4).unwrap();
        assert_eq!(bus.read(0x0c, 32).unwrap(), 0x1f4);
        assert_eq!(handle("X", &model), "E unknown request\n");
        // Device output around an answer is skipped.
        bus.port.out.extend(b"S0012 D19\nboot\n".iter());
        bus.port.pending.clear();
        assert_eq!(bus.read(0x0c, 32).unwrap(), 0x1f4);
        assert_eq!(handle("W zz 1", &model), "E bad number\n");
    }
}
