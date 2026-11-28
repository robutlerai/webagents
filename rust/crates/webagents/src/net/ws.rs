//! A WebSocket client (RFC 6455) for devices: the opening handshake and
//! framing, over any `embedded-io` stream, with caller-provided buffers.
//!
//! Client frames are masked with a key the caller supplies per frame (from the
//! device's random source); server frames are unmasked. Pings are answered
//! inside [`recv`]; a fragmented message is reassembled into the caller's
//! buffer, and a message larger than the buffer is reported, never truncated.

use embedded_io::{Read, Write};

/// Why a WebSocket operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsError {
    /// The underlying stream reported an error.
    Io,
    /// The server answered the upgrade with this HTTP status (401: bad key).
    Handshake(u16),
    /// The answer was not a valid upgrade (bad accept key, malformed head).
    BadUpgrade,
    /// A frame that breaks RFC 6455 (a control frame over 125 bytes).
    Protocol,
    /// The message does not fit the receive buffer; it was read and dropped.
    TooLarge(usize),
    /// The peer closed the connection (TCP level).
    Eof,
}

const GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

fn b64(input: &[u8], out: &mut [u8]) -> usize {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut n = 0;
    for c in input.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let v = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out[n] = T[(v >> 18) as usize & 63];
        out[n + 1] = T[(v >> 12) as usize & 63];
        out[n + 2] = if c.len() > 1 { T[(v >> 6) as usize & 63] } else { b'=' };
        out[n + 3] = if c.len() > 2 { T[v as usize & 63] } else { b'=' };
        n += 4;
    }
    n
}

fn write_all<S: Write>(s: &mut S, mut b: &[u8]) -> Result<(), WsError> {
    while !b.is_empty() {
        let n = s.write(b).map_err(|_| WsError::Io)?;
        if n == 0 {
            return Err(WsError::Eof);
        }
        b = &b[n..];
    }
    Ok(())
}

fn read_exact<S: Read>(s: &mut S, mut b: &mut [u8]) -> Result<(), WsError> {
    while !b.is_empty() {
        let n = s.read(b).map_err(|_| WsError::Io)?;
        if n == 0 {
            return Err(WsError::Eof);
        }
        b = &mut b[n..];
    }
    Ok(())
}

/// The opening handshake: `GET path` on `host`, with the 16-byte random `key`.
/// `scratch` holds the server's answer head (256 bytes is plenty).
pub fn connect<S: Read + Write>(s: &mut S, host: &str, path: &str, key: [u8; 16], scratch: &mut [u8]) -> Result<(), WsError> {
    let mut k = [0u8; 24];
    let kn = b64(&key, &mut k);
    let key_s = core::str::from_utf8(&k[..kn]).map_err(|_| WsError::Protocol)?;
    for part in [
        "GET ",
        path,
        " HTTP/1.1\r\nHost: ",
        host,
        "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ",
        key_s,
        "\r\nSec-WebSocket-Version: 13\r\n\r\n",
    ] {
        write_all(s, part.as_bytes())?;
    }
    s.flush().map_err(|_| WsError::Io)?;
    // Read the head byte by byte so no frame byte after it is consumed.
    let mut n = 0;
    loop {
        if n == scratch.len() {
            return Err(WsError::BadUpgrade);
        }
        read_exact(s, &mut scratch[n..n + 1])?;
        n += 1;
        if n >= 4 && &scratch[n - 4..n] == b"\r\n\r\n" {
            break;
        }
    }
    let head = core::str::from_utf8(&scratch[..n]).map_err(|_| WsError::BadUpgrade)?;
    let status: u16 = head.split(' ').nth(1).and_then(|c| c.parse().ok()).ok_or(WsError::BadUpgrade)?;
    if status != 101 {
        return Err(WsError::Handshake(status));
    }
    let mut sha = sha1_smol::Sha1::new();
    sha.update(key_s.as_bytes());
    sha.update(GUID);
    let mut want = [0u8; 28];
    let wn = b64(&sha.digest().bytes(), &mut want);
    let accepted = head.split("\r\n").any(|l| {
        let mut kv = l.splitn(2, ':');
        let (k, v) = (kv.next().unwrap_or(""), kv.next().unwrap_or("").trim());
        k.eq_ignore_ascii_case("sec-websocket-accept") && v.as_bytes() == &want[..wn]
    });
    if accepted {
        Ok(())
    } else {
        Err(WsError::BadUpgrade)
    }
}

/// Opcode of a text frame. UAMP frames are text.
pub const OP_TEXT: u8 = 0x1;
/// Opcode of a binary frame.
pub const OP_BINARY: u8 = 0x2;
/// Opcode of a close frame. Its payload, when present, starts with a 2-byte status code.
pub const OP_CLOSE: u8 = 0x8;
/// Opcode of a ping. [`recv`] answers pings itself.
pub const OP_PING: u8 = 0x9;
/// Opcode of a pong.
pub const OP_PONG: u8 = 0xA;

/// Sends one unfragmented frame, masked with `mask`.
pub fn send<S: Write>(s: &mut S, opcode: u8, payload: &[u8], mask: [u8; 4]) -> Result<(), WsError> {
    let mut h = [0u8; 14];
    h[0] = 0x80 | opcode;
    let len = payload.len();
    let mut hn = 2;
    if len < 126 {
        h[1] = 0x80 | len as u8;
    } else if len <= 0xffff {
        h[1] = 0x80 | 126;
        h[2..4].copy_from_slice(&(len as u16).to_be_bytes());
        hn = 4;
    } else {
        h[1] = 0x80 | 127;
        h[2..10].copy_from_slice(&(len as u64).to_be_bytes());
        hn = 10;
    }
    h[hn..hn + 4].copy_from_slice(&mask);
    write_all(s, &h[..hn + 4])?;
    let mut chunk = [0u8; 128];
    for (ci, part) in payload.chunks(chunk.len()).enumerate() {
        for (i, b) in part.iter().enumerate() {
            chunk[i] = b ^ mask[(ci * 128 + i) & 3];
        }
        write_all(s, &chunk[..part.len()])?;
    }
    s.flush().map_err(|_| WsError::Io)
}

/// A received message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Msg {
    /// A text message of this many bytes, at the start of the buffer.
    Text(usize),
    /// A binary message of this many bytes, at the start of the buffer.
    Binary(usize),
    /// The peer's close frame, with its status code when it sent one.
    Close(Option<u16>),
}

/// Receives the next data message into `buf`. Pings are answered (masked with
/// `mask()`), pongs dropped, continuation frames joined.
pub fn recv<S: Read + Write>(s: &mut S, buf: &mut [u8], mut mask: impl FnMut() -> [u8; 4]) -> Result<Msg, WsError> {
    let mut total = 0usize;
    let mut kind = 0u8;
    let mut overflow = false;
    loop {
        let mut h = [0u8; 2];
        read_exact(s, &mut h)?;
        let fin = h[0] & 0x80 != 0;
        let op = h[0] & 0x0f;
        let masked = h[1] & 0x80 != 0;
        let mut len = (h[1] & 0x7f) as u64;
        if len == 126 {
            let mut e = [0u8; 2];
            read_exact(s, &mut e)?;
            len = u16::from_be_bytes(e) as u64;
        } else if len == 127 {
            let mut e = [0u8; 8];
            read_exact(s, &mut e)?;
            len = u64::from_be_bytes(e);
        }
        let mut mk = [0u8; 4];
        if masked {
            read_exact(s, &mut mk)?;
        }
        let len = len as usize;
        if op >= 0x8 {
            // Control frames: at most 125 bytes, never fragmented.
            if len > 125 {
                return Err(WsError::Protocol);
            }
            let mut c = [0u8; 125];
            read_exact(s, &mut c[..len])?;
            if masked {
                for (i, b) in c[..len].iter_mut().enumerate() {
                    *b ^= mk[i & 3];
                }
            }
            match op {
                OP_PING => send(s, OP_PONG, &c[..len], mask())?,
                OP_CLOSE => return Ok(Msg::Close((len >= 2).then(|| u16::from_be_bytes([c[0], c[1]])))),
                _ => {}
            }
            continue;
        }
        if op != 0 {
            kind = op;
        }
        if total + len > buf.len() {
            overflow = true;
            // Drain the frame so the stream stays in sync.
            let mut sink = [0u8; 64];
            let mut left = len;
            while left > 0 {
                let n = left.min(sink.len());
                read_exact(s, &mut sink[..n])?;
                left -= n;
            }
        } else {
            read_exact(s, &mut buf[total..total + len])?;
            if masked {
                for (i, b) in buf[total..total + len].iter_mut().enumerate() {
                    *b ^= mk[i & 3];
                }
            }
        }
        total += len;
        if fin {
            if overflow {
                return Err(WsError::TooLarge(total));
            }
            return Ok(if kind == OP_BINARY { Msg::Binary(total) } else { Msg::Text(total) });
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    /// A stream with scripted input and recorded output.
    struct Pipe {
        input: Vec<u8>,
        pos: usize,
        output: Vec<u8>,
    }
    impl embedded_io::ErrorType for Pipe {
        type Error = core::convert::Infallible;
    }
    impl Read for Pipe {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
            let n = buf.len().min(self.input.len() - self.pos).min(7); // short reads on purpose
            buf[..n].copy_from_slice(&self.input[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }
    impl Write for Pipe {
        fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
            self.output.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn handshake_checks_the_accept_key() {
        // RFC 6455's example: key dGhlIHNhbXBsZSBub25jZQ== -> s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
        let key = *b"the sample nonce";
        let ok = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n\x81\x02hi";
        let mut p = Pipe { input: ok.to_vec(), pos: 0, output: Vec::new() };
        let mut scratch = [0u8; 256];
        connect(&mut p, "h", "/ws?token=t", key, &mut scratch).unwrap();
        let req = std::str::from_utf8(&p.output).unwrap();
        assert!(req.starts_with("GET /ws?token=t HTTP/1.1\r\nHost: h\r\n") && req.contains("Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"));
        let mut buf = [0u8; 16];
        assert_eq!(recv(&mut p, &mut buf, || [0; 4]).unwrap(), Msg::Text(2), "the frame after the head was not consumed");
        let bad = b"HTTP/1.1 101 OK\r\nSec-WebSocket-Accept: AAAA\r\n\r\n";
        assert_eq!(connect(&mut Pipe { input: bad.to_vec(), pos: 0, output: Vec::new() }, "h", "/", key, &mut scratch), Err(WsError::BadUpgrade));
        let denied = b"HTTP/1.1 401 Unauthorized\r\n\r\n";
        assert_eq!(connect(&mut Pipe { input: denied.to_vec(), pos: 0, output: Vec::new() }, "h", "/", key, &mut scratch), Err(WsError::Handshake(401)));
    }

    #[test]
    fn frames_round_trip_with_ping_fragments_and_close() {
        let mut out = Pipe { input: Vec::new(), pos: 0, output: Vec::new() };
        let payload: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        send(&mut out, OP_TEXT, &payload, [1, 2, 3, 4]).unwrap();
        assert_eq!(&out.output[..4], &[0x81, 0xfe, 0x01, 0x2c]);
        let unmasked: Vec<u8> = out.output[8..].iter().enumerate().map(|(i, b)| b ^ [1, 2, 3, 4][i & 3]).collect();
        assert_eq!(unmasked, payload);
        // Server side: ping, fragment "ab" + "cd", close 4001.
        let input = [&[0x89, 0x01, b'p'][..], &[0x01, 0x02, b'a', b'b'], &[0x80, 0x02, b'c', b'd'], &[0x88, 0x02, 0x0f, 0xa1]].concat();
        let mut p = Pipe { input, pos: 0, output: Vec::new() };
        let mut buf = [0u8; 8];
        assert_eq!(recv(&mut p, &mut buf, || [0; 4]).unwrap(), Msg::Text(4));
        assert_eq!(&buf[..4], b"abcd");
        assert_eq!(p.output[0], 0x8A, "ping answered with a pong");
        assert_eq!(recv(&mut p, &mut buf, || [0; 4]).unwrap(), Msg::Close(Some(4001)));
    }

    #[test]
    fn oversize_message_is_reported_and_stream_stays_in_sync() {
        let input = [&[0x81, 0x05][..], b"hello", &[0x81, 0x02], b"ok"].concat();
        let mut p = Pipe { input, pos: 0, output: Vec::new() };
        let mut buf = [0u8; 4];
        assert_eq!(recv(&mut p, &mut buf, || [0; 4]), Err(WsError::TooLarge(5)));
        assert_eq!(recv(&mut p, &mut buf, || [0; 4]).unwrap(), Msg::Text(2));
    }
}
