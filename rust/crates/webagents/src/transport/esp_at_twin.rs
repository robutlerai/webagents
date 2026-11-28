//! A twin of an ESP-AT network co-processor, for running device firmware
//! without the module: it answers the AT subset the device driver uses
//! ([`crate::net::esp_at`]) on its serial side and opens real TCP connections
//! on its network side. Responses follow Espressif's ESP-AT 2.x formats,
//! including passive receive mode.
//!
//! It is plain TCP: TLS, if any, is the device's own session running over it,
//! exactly as with the real module.

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct LinkState {
    tcp: TcpStream,
    rx: Arc<Mutex<VecDeque<u8>>>,
    eof: Arc<AtomicBool>,
    reported: bool,
}

/// Counters a test or a log can read.
#[derive(Debug, Default, Clone)]
pub struct TwinStats {
    /// AT commands answered.
    pub commands: u64,
    /// Bytes the device sent to the network (payload of `AT+CIPSEND`).
    pub bytes_up: u64,
    /// Bytes delivered to the device (payload of `AT+CIPRECVDATA`).
    pub bytes_down: u64,
    /// TCP connections opened for the device.
    pub connects: u64,
}

/// Serves the AT protocol on `serial` until it closes.
pub fn serve<S: Read + Write>(serial: S, stats: Arc<Mutex<TwinStats>>) -> io::Result<()> {
    let mut serial = BufReader::new(serial);
    let mut links: [Option<LinkState>; 5] = Default::default();
    let mut echo = true;
    let mut line = Vec::new();
    loop {
        line.clear();
        if serial.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        let cmd = String::from_utf8_lossy(&line).trim().to_string();
        if cmd.is_empty() {
            continue;
        }
        stats.lock().unwrap().commands += 1;
        let mut out: Vec<u8> = Vec::new();
        if echo {
            out.extend_from_slice(cmd.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        // Links the remote closed, once drained, are reported before the next answer.
        for (id, l) in links.iter_mut().enumerate() {
            if let Some(st) = l {
                if st.eof.load(Ordering::Relaxed) && st.rx.lock().unwrap().is_empty() && !st.reported {
                    st.reported = true;
                    out.extend_from_slice(format!("{id},CLOSED\r\n").as_bytes());
                }
            }
        }
        let ok = b"\r\nOK\r\n";
        let (verb, arg) = cmd.split_once('=').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((cmd.clone(), String::new()));
        match verb.as_str() {
            "AT" | "AT+CWMODE" | "AT+CIPMUX" | "AT+CIPRECVMODE" | "AT+RST" => out.extend_from_slice(ok),
            "ATE0" => {
                echo = false;
                out.extend_from_slice(ok);
            }
            "ATE1" => {
                echo = true;
                out.extend_from_slice(ok);
            }
            "AT+CWJAP" => out.extend_from_slice(b"WIFI CONNECTED\r\nWIFI GOT IP\r\n\r\nOK\r\n"),
            "AT+CIPSTART" => {
                let parts: Vec<String> = arg.split(',').map(|p| p.trim().trim_matches('"').to_string()).collect();
                let res = (|| -> Option<(usize, TcpStream)> {
                    let id: usize = parts.first()?.parse().ok().filter(|i| *i < 5)?;
                    if parts.get(1)? != "TCP" {
                        return None;
                    }
                    let addr = format!("{}:{}", parts.get(2)?, parts.get(3)?).to_socket_addrs().ok()?.next()?;
                    let tcp = TcpStream::connect_timeout(&addr, Duration::from_secs(10)).ok()?;
                    Some((id, tcp))
                })();
                match res {
                    Some((id, tcp)) => {
                        let rx = Arc::new(Mutex::new(VecDeque::new()));
                        let eof = Arc::new(AtomicBool::new(false));
                        let (mut reader, rx2, eof2, st2) = (tcp.try_clone()?, rx.clone(), eof.clone(), stats.clone());
                        std::thread::spawn(move || {
                            let mut b = [0u8; 4096];
                            loop {
                                match reader.read(&mut b) {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => {
                                        rx2.lock().unwrap().extend(&b[..n]);
                                        st2.lock().unwrap().bytes_down += n as u64;
                                    }
                                }
                            }
                            eof2.store(true, Ordering::Relaxed);
                        });
                        links[id] = Some(LinkState { tcp, rx, eof, reported: false });
                        stats.lock().unwrap().connects += 1;
                        out.extend_from_slice(format!("{id},CONNECT\r\n\r\nOK\r\n").as_bytes());
                    }
                    None => out.extend_from_slice(b"\r\nERROR\r\n"),
                }
            }
            "AT+CIPSEND" => {
                let mut it = arg.split(',');
                let id: Option<usize> = it.next().and_then(|x| x.trim().parse().ok());
                let len: Option<usize> = it.next().and_then(|x| x.trim().parse().ok());
                match (id, len) {
                    (Some(id), Some(len)) if id < 5 && links[id].as_ref().is_some_and(|l| !l.eof.load(Ordering::Relaxed)) && len <= 2048 => {
                        out.extend_from_slice(b"\r\nOK\r\n\r\n>");
                        serial.get_mut().write_all(&out)?;
                        serial.get_mut().flush()?;
                        out.clear();
                        let mut data = vec![0u8; len];
                        serial.read_exact(&mut data)?;
                        let st = links[id].as_mut().unwrap();
                        let sent = st.tcp.write_all(&data).is_ok();
                        stats.lock().unwrap().bytes_up += len as u64;
                        out.extend_from_slice(format!("\r\nRecv {len} bytes\r\n").as_bytes());
                        out.extend_from_slice(if sent { b"\r\nSEND OK\r\n" } else { b"\r\nSEND FAIL\r\n" });
                    }
                    _ => out.extend_from_slice(b"\r\nERROR\r\n"),
                }
            }
            "AT+CIPRECVLEN?" => {
                let lens: Vec<String> = links.iter().map(|l| l.as_ref().map_or(0, |s| s.rx.lock().unwrap().len()).to_string()).collect();
                out.extend_from_slice(format!("+CIPRECVLEN:{}\r\n\r\nOK\r\n", lens.join(",")).as_bytes());
            }
            "AT+CIPRECVDATA" => {
                let mut it = arg.split(',');
                let id: Option<usize> = it.next().and_then(|x| x.trim().parse().ok());
                let want: Option<usize> = it.next().and_then(|x| x.trim().parse().ok());
                match (id, want) {
                    (Some(id), Some(want)) if id < 5 && links[id].is_some() => {
                        let st = links[id].as_ref().unwrap();
                        let mut q = st.rx.lock().unwrap();
                        let n = want.min(q.len());
                        let data: Vec<u8> = q.drain(..n).collect();
                        out.extend_from_slice(format!("+CIPRECVDATA:{n},").as_bytes());
                        out.extend_from_slice(&data);
                        out.extend_from_slice(ok);
                    }
                    _ => out.extend_from_slice(b"\r\nERROR\r\n"),
                }
            }
            "AT+CIPCLOSE" => match arg.trim().parse::<usize>().ok().filter(|i| *i < 5).and_then(|i| links[i].take().map(|l| (i, l))) {
                Some((id, l)) => {
                    let _ = l.tcp.shutdown(Shutdown::Both);
                    out.extend_from_slice(format!("{id},CLOSED\r\n\r\nOK\r\n").as_bytes());
                }
                None => out.extend_from_slice(b"\r\nERROR\r\n"),
            },
            _ => out.extend_from_slice(b"\r\nERROR\r\n"),
        }
        serial.get_mut().write_all(&out)?;
        serial.get_mut().flush()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::esp_at::{AtPort, EspAt, Link};
    use std::cell::RefCell;
    use std::net::TcpListener;
    use std::os::unix::net::UnixStream;
    use std::time::Instant;

    struct Port(UnixStream, Instant);
    impl AtPort for Port {
        fn write(&mut self, b: &[u8]) {
            self.0.write_all(b).unwrap();
        }
        fn read_byte(&mut self) -> Option<u8> {
            let mut b = [0u8; 1];
            match self.0.read(&mut b) {
                Ok(1) => Some(b[0]),
                _ => None,
            }
        }
        fn now_ms(&mut self) -> u32 {
            self.1.elapsed().as_millis() as u32
        }
    }

    #[test]
    fn driver_and_twin_carry_tcp_both_ways() {
        // An echo server that upper-cases, then closes after one exchange.
        let srv = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = srv.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = srv.accept().unwrap();
            let mut b = [0u8; 5000];
            let mut got = Vec::new();
            while got.len() < 3000 {
                let n = s.read(&mut b).unwrap();
                got.extend_from_slice(&b[..n]);
            }
            s.write_all(&got.to_ascii_uppercase()).unwrap();
        });
        let (a, b) = UnixStream::pair().unwrap();
        std::thread::spawn(move || serve(b, Arc::default()));
        a.set_read_timeout(Some(Duration::from_millis(5))).unwrap();
        let at = RefCell::new(EspAt::new(Port(a, Instant::now())));
        at.borrow_mut().init("lab", "secret").unwrap();
        at.borrow_mut().open(0, "127.0.0.1", port).unwrap();
        let mut link = Link { at: &at, id: 0, timeout_ms: 3000 };
        let msg: Vec<u8> = (0..3000).map(|i| b"abcdefgh 0,CLOSED E"[i % 19]).collect();
        embedded_io::Write::write_all(&mut link, &msg).unwrap(); // more than one CIPSEND chunk
        let mut back = Vec::new();
        let mut buf = [0u8; 700];
        loop {
            let n = embedded_io::Read::read(&mut link, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            back.extend_from_slice(&buf[..n]);
        }
        assert_eq!(back, msg.to_ascii_uppercase());
        assert!(at.borrow().is_closed(0), "the remote close was reported");
        assert_eq!(at.borrow_mut().open(1, "127.0.0.1", 1), Err(crate::net::esp_at::AtError::Refused));
    }
}
