//! The W5500 driver against the register-level simulator with real TCP behind
//! it: more than one buffer of data each way (address wrap, flow control), end
//! of stream, a refused connection, and the device WebSocket client running
//! unchanged over a W5500 socket.
#![cfg(feature = "gateway")]

use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::OnceLock;
use std::time::Instant;

use embedded_hal::spi::{ErrorKind, ErrorType, Operation, SpiDevice};
use webagents::net::w5500::{Socket, W5500Error, W5500};
use webagents::net::ws;
use webagents::transport::w5500_sim::W5500Sim;

fn now() -> u32 {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now).elapsed().as_millis() as u32
}

const LOCAL: [u8; 4] = [127, 0, 0, 1];

fn chip() -> RefCell<W5500<W5500Sim>> {
    let mut c = W5500::new(W5500Sim::new(), now);
    c.timeout_ms = 3000;
    c.init([0x02, 0, 0, 0, 0, 1], [10, 0, 0, 2], [10, 0, 0, 1], [255, 255, 255, 0]).unwrap();
    RefCell::new(c)
}

#[test]
fn more_than_a_buffer_each_way() {
    let srv = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = srv.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut s, _) = srv.accept().unwrap();
        let mut got = Vec::new();
        let mut b = [0u8; 4096];
        while got.len() < 5000 {
            let n = s.read(&mut b).unwrap();
            got.extend_from_slice(&b[..n]);
        }
        s.write_all(&got.to_ascii_uppercase()).unwrap();
    });
    let c = chip();
    c.borrow_mut().connect(0, LOCAL, port, 40000).unwrap();
    let mut sock = Socket { chip: &c, sock: 0, timeout_ms: 3000 };
    let msg: Vec<u8> = (0..5000).map(|i| b"abcdefghij"[i % 10]).collect();
    embedded_io::Write::write_all(&mut sock, &msg).unwrap();
    let mut back = Vec::new();
    let mut buf = [0u8; 700];
    loop {
        let n = embedded_io::Read::read(&mut sock, &mut buf).unwrap();
        if n == 0 {
            break;
        }
        back.extend_from_slice(&buf[..n]);
    }
    assert_eq!(back, msg.to_ascii_uppercase());
}

#[test]
fn refused_connection_and_wrong_chip() {
    let free = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = free.local_addr().unwrap().port();
    drop(free);
    let c = chip();
    assert_eq!(c.borrow_mut().connect(1, LOCAL, port, 40001), Err(W5500Error::Closed));

    struct Nothing;
    impl ErrorType for Nothing {
        type Error = ErrorKind;
    }
    impl SpiDevice for Nothing {
        fn transaction(&mut self, ops: &mut [Operation<'_, u8>]) -> Result<(), ErrorKind> {
            for op in ops {
                if let Operation::Read(b) = op {
                    b.fill(0xff);
                }
            }
            Ok(())
        }
    }
    let mut w = W5500::new(Nothing, now);
    w.timeout_ms = 50;
    // 0xff in MR keeps the reset bit set: no chip answers, so the reset times out.
    assert_eq!(w.init([0; 6], [0; 4], [0; 4], [0; 4]), Err(W5500Error::Timeout));
}

#[test]
fn websocket_runs_over_a_w5500_socket() {
    use tokio_tungstenite::tungstenite::{self, protocol::frame::coding::CloseCode, protocol::CloseFrame, Message};
    let srv = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = srv.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (s, _) = srv.accept().unwrap();
        let mut ws = tungstenite::accept(s).unwrap();
        let m = ws.read().unwrap();
        let t = m.into_text().unwrap().to_uppercase();
        ws.send(Message::Text(t.into())).unwrap();
        ws.close(Some(CloseFrame { code: CloseCode::from(4001), reason: "Unauthorized".into() })).unwrap();
        let _ = ws.flush();
        while ws.read().is_ok() {}
    });
    let c = chip();
    c.borrow_mut().connect(2, LOCAL, port, 40002).unwrap();
    let mut sock = Socket { chip: &c, sock: 2, timeout_ms: 3000 };
    let mut scratch = [0u8; 512];
    ws::connect(&mut sock, "127.0.0.1", "/ws?token=placeholder", *b"0123456789abcdef", &mut scratch).unwrap();
    let body = "x".repeat(3000);
    ws::send(&mut sock, ws::OP_TEXT, format!("{{\"type\":\"session.create\",\"pad\":\"{body}\"}}").as_bytes(), [9, 8, 7, 6]).unwrap();
    let mut rx = vec![0u8; 8192];
    let ws::Msg::Text(n) = ws::recv(&mut sock, &mut rx, || [1, 2, 3, 4]).unwrap() else { panic!() };
    assert!(std::str::from_utf8(&rx[..n]).unwrap().starts_with("{\"TYPE\":\"SESSION.CREATE\""));
    assert_eq!(ws::recv(&mut sock, &mut rx, || [1, 2, 3, 4]).unwrap(), ws::Msg::Close(Some(4001)));
}
