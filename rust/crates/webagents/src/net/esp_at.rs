//! Driver for an ESP-AT network co-processor (Espressif's AT firmware on an
//! ESP32 or ESP8266) on a serial link: Wi-Fi join and up to five TCP links.
//! TLS is NOT delegated to the co-processor: the device runs its own TLS
//! session over the raw TCP link, so the co-processor never sees the key or
//! the plaintext.
//!
//! Receive uses passive mode (`AT+CIPRECVMODE=1`): data waits in the
//! co-processor until asked for with `AT+CIPRECVDATA`, so no data ever arrives
//! in the middle of a command response, and a small device never has to buffer
//! what it did not ask for.

use core::cell::RefCell;
use core::fmt::Write as _;

/// The serial link to the co-processor.
pub trait AtPort {
    /// Sends `bytes` to the co-processor, blocking until the UART has taken them.
    fn write(&mut self, bytes: &[u8]);
    /// The next received byte, or `None` when none is waiting. Must not block:
    /// the driver polls it against its deadlines.
    fn read_byte(&mut self) -> Option<u8>;
    /// Monotonic milliseconds.
    fn now_ms(&mut self) -> u32;
}

/// Why a co-processor command did not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtError {
    /// No answer within the command's deadline (see [`EspAt::cmd_timeout_ms`]).
    Timeout,
    /// The co-processor answered ERROR or FAIL.
    Refused,
    /// The TCP link is closed (the peer or the network ended it).
    Closed,
    /// An answer that is not ESP-AT's format, or more data than was asked for.
    Protocol,
}

/// The driver: one co-processor on one serial port. `no_std`, no heap; its
/// whole state is a 96-byte line buffer and a bit per link.
///
/// Link ids are 0 to 4, the five connections ESP-AT offers in multiple
/// connection mode. Every method blocks until the co-processor answers or
/// the deadline passes.
pub struct EspAt<P> {
    port: P,
    closed: u8,
    line: [u8; 96],
    len: usize,
    /// Deadline for an ordinary command, in milliseconds (default 5000). Joining
    /// the network and opening a link use their own, longer deadlines (20 s and
    /// 15 s).
    pub cmd_timeout_ms: u32,
}

const MAX_SEND: usize = 2048;

impl<P: AtPort> EspAt<P> {
    /// A driver over `port`. Nothing is sent until [`EspAt::init`]; every link
    /// starts closed.
    pub fn new(port: P) -> Self {
        EspAt { port, closed: 0x1f, line: [0; 96], len: 0, cmd_timeout_ms: 5_000 }
    }

    /// The serial port, for a caller that needs its clock or raw access.
    pub fn port(&mut self) -> &mut P {
        &mut self.port
    }

    fn cmd(&mut self, parts: &[&str]) {
        for p in parts {
            self.port.write(p.as_bytes());
        }
        self.port.write(b"\r\n");
    }

    fn byte(&mut self, deadline: u32) -> Result<u8, AtError> {
        loop {
            if let Some(b) = self.port.read_byte() {
                return Ok(b);
            }
            if (self.port.now_ms().wrapping_sub(deadline) as i32) >= 0 {
                return Err(AtError::Timeout);
            }
        }
    }

    /// Next non-empty line into `self.line[..self.len]`.
    fn read_line(&mut self, deadline: u32) -> Result<(), AtError> {
        loop {
            self.len = 0;
            loop {
                let b = self.byte(deadline)?;
                if b == b'\n' {
                    break;
                }
                if b != b'\r' && self.len < self.line.len() {
                    self.line[self.len] = b;
                    self.len += 1;
                }
            }
            if self.len > 0 {
                self.note();
                return Ok(());
            }
        }
    }

    fn text(&self) -> &str {
        core::str::from_utf8(&self.line[..self.len]).unwrap_or("")
    }

    /// Link state reports that can arrive between any two lines.
    fn note(&mut self) {
        let t = self.text();
        let id = t.as_bytes().first().map(|c| c.wrapping_sub(b'0')).filter(|d| *d < 5);
        if let Some(id) = id {
            if t[1..] == *",CLOSED" {
                self.closed |= 1 << id;
            } else if t[1..] == *",CONNECT" {
                self.closed &= !(1 << id);
            }
        }
    }

    fn deadline(&mut self, ms: u32) -> u32 {
        self.port.now_ms().wrapping_add(ms)
    }

    fn expect_ok(&mut self, ms: u32) -> Result<(), AtError> {
        let d = self.deadline(ms);
        loop {
            self.read_line(d)?;
            match self.text() {
                "OK" | "SEND OK" => return Ok(()),
                "ERROR" | "FAIL" | "SEND FAIL" => return Err(AtError::Refused),
                _ => {}
            }
        }
    }

    /// Reset echo, station mode, join the network, multiple links, passive receive.
    pub fn init(&mut self, ssid: &str, password: &str) -> Result<(), AtError> {
        let t = self.cmd_timeout_ms;
        for c in ["AT", "ATE0", "AT+CWMODE=1"] {
            self.cmd(&[c]);
            self.expect_ok(t)?;
        }
        self.cmd(&["AT+CWJAP=\"", ssid, "\",\"", password, "\""]);
        self.expect_ok(20_000)?;
        for c in ["AT+CIPMUX=1", "AT+CIPRECVMODE=1"] {
            self.cmd(&[c]);
            self.expect_ok(t)?;
        }
        Ok(())
    }

    /// Opens link `id` (0 to 4) as a TCP connection to `host:port`. `host` is a
    /// name or an IPv4 address; the co-processor resolves names. Waits up to 15 s.
    pub fn open(&mut self, id: u8, host: &str, port: u16) -> Result<(), AtError> {
        let mut p = heapless_str::<8>();
        let _ = write!(p, "{port}");
        let idc = [b'0' + id];
        self.cmd(&["AT+CIPSTART=", core::str::from_utf8(&idc).unwrap_or("0"), ",\"TCP\",\"", host, "\",", p.as_str()]);
        self.expect_ok(15_000)?;
        self.closed &= !(1 << id);
        Ok(())
    }

    /// Whether link `id` is closed, as last reported by the co-processor (its
    /// `<id>,CLOSED` and `<id>,CONNECT` lines are noted whenever they arrive).
    pub fn is_closed(&self, id: u8) -> bool {
        self.closed & (1 << id) != 0
    }

    /// Closes link `id`. The link counts as closed afterwards even when the
    /// co-processor answers with an error (closing a closed link).
    pub fn close(&mut self, id: u8) -> Result<(), AtError> {
        let idc = [b'0' + id];
        self.cmd(&["AT+CIPCLOSE=", core::str::from_utf8(&idc).unwrap_or("0")]);
        let r = self.expect_ok(self.cmd_timeout_ms);
        self.closed |= 1 << id;
        r
    }

    /// Sends `data` on link `id`, in chunks of at most 2048 bytes (ESP-AT's limit
    /// per `AT+CIPSEND`), each acknowledged before the next.
    pub fn send(&mut self, id: u8, data: &[u8]) -> Result<(), AtError> {
        for chunk in data.chunks(MAX_SEND) {
            if self.is_closed(id) {
                return Err(AtError::Closed);
            }
            let mut n = heapless_str::<8>();
            let _ = write!(n, "{}", chunk.len());
            let idc = [b'0' + id];
            self.cmd(&["AT+CIPSEND=", core::str::from_utf8(&idc).unwrap_or("0"), ",", n.as_str()]);
            // The prompt is a bare '>' with no line end; lines before it ("OK") are skipped.
            let d = self.deadline(self.cmd_timeout_ms);
            let mut last = 0u8;
            loop {
                let b = self.byte(d)?;
                if b == b'>' && (last == b'\n' || last == 0) {
                    break;
                }
                if b == b'E' && (last == b'\n' || last == 0) {
                    // "ERROR" at a line start: consume the line and fail. An 'E'
                    // inside another line ("0,CLOSED") is not an error.
                    self.read_line(d)?;
                    return Err(AtError::Refused);
                }
                last = b;
            }
            self.port.write(chunk);
            self.expect_ok(self.cmd_timeout_ms)?;
        }
        Ok(())
    }

    /// Bytes waiting on link `id`.
    pub fn available(&mut self, id: u8) -> Result<usize, AtError> {
        self.cmd(&["AT+CIPRECVLEN?"]);
        let d = self.deadline(self.cmd_timeout_ms);
        let mut n = 0usize;
        loop {
            self.read_line(d)?;
            let t = self.text();
            if let Some(list) = t.strip_prefix("+CIPRECVLEN:") {
                n = list.split(',').nth(id as usize).and_then(|x| x.trim().parse().ok()).unwrap_or(0);
            } else if t == "OK" {
                return Ok(n);
            } else if t == "ERROR" {
                return Err(AtError::Refused);
            }
        }
    }

    /// Reads up to `buf.len()` waiting bytes of link `id`.
    pub fn recv(&mut self, id: u8, buf: &mut [u8]) -> Result<usize, AtError> {
        let mut n = heapless_str::<8>();
        let _ = write!(n, "{}", buf.len().min(MAX_SEND));
        let idc = [b'0' + id];
        self.cmd(&["AT+CIPRECVDATA=", core::str::from_utf8(&idc).unwrap_or("0"), ",", n.as_str()]);
        let d = self.deadline(self.cmd_timeout_ms);
        // Header "+CIPRECVDATA:<len>," then <len> raw bytes, then OK.
        const P: &[u8] = b"+CIPRECVDATA:";
        let mut matched = 0;
        let mut last = b'\n';
        loop {
            let b = self.byte(d)?;
            if b == P[matched] {
                matched += 1;
                if matched == P.len() {
                    break;
                }
            } else if b == b'E' && matched == 0 && last == b'\n' {
                // "ERROR" at a line start (not an 'E' inside "0,CLOSED").
                self.read_line(d)?;
                return Err(AtError::Refused);
            } else {
                matched = usize::from(b == P[0]);
            }
            last = b;
        }
        let mut len = 0usize;
        loop {
            let b = self.byte(d)?;
            match b {
                b'0'..=b'9' => len = len * 10 + (b - b'0') as usize,
                b',' => break,
                _ => return Err(AtError::Protocol),
            }
        }
        if len > buf.len() {
            return Err(AtError::Protocol);
        }
        for slot in buf.iter_mut().take(len) {
            *slot = self.byte(d)?;
        }
        self.expect_ok(self.cmd_timeout_ms)?;
        Ok(len)
    }
}

/// A tiny fixed-capacity string for number formatting.
struct Small<const N: usize> {
    b: [u8; N],
    n: usize,
}
fn heapless_str<const N: usize>() -> Small<N> {
    Small { b: [0; N], n: 0 }
}
impl<const N: usize> Small<N> {
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.b[..self.n]).unwrap_or("")
    }
}
impl<const N: usize> core::fmt::Write for Small<N> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let b = s.as_bytes();
        if self.n + b.len() > N {
            return Err(core::fmt::Error);
        }
        self.b[self.n..self.n + b.len()].copy_from_slice(b);
        self.n += b.len();
        Ok(())
    }
}

/// One TCP link of a shared co-processor, as an `embedded-io` stream (so a TLS
/// session or a WebSocket can run over it). Reads poll until data arrives,
/// the link closes (end of stream) or `timeout_ms` passes.
pub struct Link<'a, P> {
    /// The shared driver. A `RefCell` because two links (the agent socket and a
    /// model call) take turns on one co-processor, on one core.
    pub at: &'a RefCell<EspAt<P>>,
    /// The link id, 0 to 4, already opened with [`EspAt::open`].
    pub id: u8,
    /// Longest wait for data in one `read`, in milliseconds.
    pub timeout_ms: u32,
}

/// The error of a [`Link`], wrapping the driver's so it implements
/// `embedded_io::Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkError(pub AtError);

impl core::fmt::Display for LinkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let what = match self.0 {
            AtError::Timeout => "timed out",
            AtError::Refused => "refused by the co-processor",
            AtError::Closed => "link closed",
            AtError::Protocol => "unexpected answer from the co-processor",
        };
        write!(f, "co-processor link: {what}")
    }
}

impl core::error::Error for LinkError {}

impl embedded_io::Error for LinkError {
    fn kind(&self) -> embedded_io::ErrorKind {
        match self.0 {
            AtError::Timeout => embedded_io::ErrorKind::TimedOut,
            AtError::Closed => embedded_io::ErrorKind::NotConnected,
            _ => embedded_io::ErrorKind::Other,
        }
    }
}

impl<P: AtPort> embedded_io::ErrorType for Link<'_, P> {
    type Error = LinkError;
}

impl<P: AtPort> embedded_io::Read for Link<'_, P> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut at = self.at.borrow_mut();
        let start = at.port().now_ms();
        loop {
            let n = at.available(self.id).map_err(LinkError)?;
            if n > 0 {
                let want = n.min(buf.len());
                return at.recv(self.id, &mut buf[..want]).map_err(LinkError);
            }
            if at.is_closed(self.id) {
                return Ok(0);
            }
            if at.port().now_ms().wrapping_sub(start) > self.timeout_ms {
                return Err(LinkError(AtError::Timeout));
            }
        }
    }
}

impl<P: AtPort> embedded_io::Write for Link<'_, P> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        self.at.borrow_mut().send(self.id, buf).map_err(LinkError)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}
