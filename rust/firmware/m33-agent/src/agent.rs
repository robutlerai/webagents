//! The agent: one WebSocket to the platform's `/ws`, a turn loop that calls
//! the model on `/llm` (a second, short-lived link per model call, paid by the
//! turn's own token), and the part's envelope on every tool call.

use core::cell::RefCell;
use core::fmt::Write as _;

use embedded_io::{Read, Write};
use embedded_tls::blocking::{TlsConfig, TlsConnection, TlsContext};
use embedded_tls::pki::CertVerifier;
use embedded_tls::{Aes128GcmSha256, CryptoProvider, TlsVerifier};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use webagents::bus::Mmio;
use webagents::net::esp_at::{AtPort, EspAt, Link};
use webagents::net::{jsonv, uamp, ws};
use webagents::regspec::json::BufWriter;
use webagents::regspec::CallCtx;

use crate::uart::Uart;

use crate::pwm1;

const HOST: &str = env!("AGENT_HOST");
const PORT: &str = env!("AGENT_PORT");
const TOKEN: &str = env!("AGENT_TOKEN");
const TLS: bool = env_is(option_env!("AGENT_TLS"), "1");
const EXPECT_REFUSED: bool = env_is(option_env!("AGENT_EXPECT"), "refused");

const fn env_is(v: Option<&str>, want: &str) -> bool {
    let Some(v) = v else { return false };
    let (a, b) = (v.as_bytes(), want.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
const AGENT: &str = "pwm-m33";
const MAX_ROUNDS: usize = 6;
/// The single trust anchor for the platform's certificate chain.
const ROOT_CA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/root-ca.der"));

const INSTRUCTIONS: &str = "You operate the hardware through its tools. Every value is in the unit the tool states. \
A write outside a limit is refused with the reason and nothing is written; explain the limit instead of repeating the \
request. Some writes need a person's confirmation at the device; say so when a tool reports it.";

// ---- buffers (static so their size shows in .bss and the stack stays small) ----
const TLS_RX: usize = 16_640; // one full TLS record
const TLS_TX: usize = 4_096;
static mut TLS0_RX: [u8; TLS_RX] = [0; TLS_RX];
static mut TLS0_TX: [u8; TLS_TX] = [0; TLS_TX];
static mut TLS1_RX: [u8; TLS_RX] = [0; TLS_RX];
static mut TLS1_TX: [u8; TLS_TX] = [0; TLS_TX];
static mut WS_RX: [u8; 16_384] = [0; 16_384];
static mut REQ: [u8; 24_576] = [0; 24_576];
static mut LLM_RX: [u8; 12_288] = [0; 12_288];
static mut CONV: [u8; 8_192] = [0; 8_192];
static mut OUT: [u8; 4_096] = [0; 4_096];
/// The part: in emulation its registers are a RAM block reached through the
/// same memory-mapped bus adapter real silicon would use.
static mut PWM_REGS: [u32; 12] = [0; 12];

macro_rules! log {
    ($($t:tt)*) => {{ let mut c = Uart::console(); let _ = writeln!(c, $($t)*); }};
}

struct Port(Uart);
impl AtPort for Port {
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0.write_byte(*b);
        }
    }
    fn read_byte(&mut self) -> Option<u8> {
        self.0.try_read()
    }
    fn now_ms(&mut self) -> u32 {
        crate::now_ms()
    }
}

type Esp = RefCell<EspAt<Port>>;

struct Provider<'a> {
    rng: ChaCha20Rng,
    verifier: CertVerifier<'a, Aes128GcmSha256, embedded_tls::NoClock, 4096>,
}

impl CryptoProvider for Provider<'_> {
    type CipherSuite = Aes128GcmSha256;
    type Signature = &'static [u8];
    fn rng(&mut self) -> impl embedded_tls::CryptoRngCore {
        &mut self.rng
    }
    fn verifier(&mut self) -> Result<&mut impl TlsVerifier<Aes128GcmSha256>, embedded_tls::TlsError> {
        Ok(&mut self.verifier)
    }
}

struct Ctx<'e> {
    esp: &'e Esp,
    rng: ChaCha20Rng,
    ids: u32,
    engine: pwm1::Engine<Mmio>,
    turns_done: u32,
}

impl Ctx<'_> {
    fn id(&mut self) -> u32 {
        self.ids += 1;
        self.ids
    }
    fn mask(&mut self) -> [u8; 4] {
        self.rng.next_u32().to_le_bytes()
    }
    fn key(&mut self) -> [u8; 16] {
        let mut k = [0u8; 16];
        self.rng.fill_bytes(&mut k);
        k
    }
}

fn buf<const N: usize>(b: &'static mut [u8; N]) -> &'static mut [u8] {
    &mut b[..]
}

/// Takes a static buffer. Each is taken once, by one owner, on one core.
macro_rules! take {
    ($b:ident) => {
        // SAFETY: single core, no interrupts touch these, and each buffer has one user.
        buf(unsafe { &mut *core::ptr::addr_of_mut!($b) })
    };
}

pub fn run() -> bool {
    log!("m33-agent: boot; platform {HOST}:{PORT} tls={TLS} expect={}", if EXPECT_REFUSED { "refused" } else { "turn" });
    // The emulated board has no TRNG: the seed below is a fixed value mixed with
    // the timer, which is NOT secure. Real hardware seeds from its TRNG.
    let seed_mix = cortex_m::peripheral::SYST::get_current();
    let mut seed = *b"webagents-m33-emulation-seed!!!!";
    seed[..4].copy_from_slice(&seed_mix.to_le_bytes());
    let rng = ChaCha20Rng::from_seed(seed);

    // SAFETY: as for `take!`.
    let regs = unsafe { &mut *core::ptr::addr_of_mut!(PWM_REGS) };
    for r in pwm1::DEVICE.registers {
        regs[(r.offset / 4) as usize] = r.reset;
    }
    // SAFETY: PWM_REGS is the part's register block in this emulation and has no other user.
    let bus = unsafe { Mmio::new(regs.as_mut_ptr() as usize) };
    let engine = pwm1::Engine::new(&pwm1::DEVICE, bus, [0; pwm1::FIELD_COUNT]);

    let esp: Esp = RefCell::new(EspAt::new(Port(Uart::link())));
    if let Err(e) = esp.borrow_mut().init("lab", "emulated") {
        log!("co-processor init failed: {e:?}");
        return false;
    }
    log!("co-processor ready");
    let port: u16 = PORT.parse().unwrap_or(443);
    if let Err(e) = esp.borrow_mut().open(0, HOST, port) {
        log!("TCP to {HOST}:{port} failed: {e:?}");
        return false;
    }
    let mut cx = Ctx { esp: &esp, rng, ids: 0, engine, turns_done: 0 };
    let link = Link { at: &esp, id: 0, timeout_ms: 60_000 };
    if TLS {
        let mut tls = TlsConnection::new(link, take!(TLS0_RX), take!(TLS0_TX));
        let config = TlsConfig::new().with_server_name(HOST);
        let provider = Provider { rng: ChaCha20Rng::from_seed(cx.key().repeat2()), verifier: CertVerifier::new(embedded_tls::Certificate::X509(ROOT_CA)) };
        let t0 = crate::now_ms();
        if let Err(e) = tls.open(TlsContext::new(&config, provider)) {
            log!("TLS handshake failed: {e:?}");
            return false;
        }
        log!("TLS 1.3 up (AES-128-GCM, server chain verified to the pinned root, host {HOST}) in {} ms", crate::now_ms() - t0);
        session(&mut cx, &mut tls)
    } else {
        let mut l = link;
        session(&mut cx, &mut l)
    }
}

trait Repeat2 {
    fn repeat2(self) -> [u8; 32];
}
impl Repeat2 for [u8; 16] {
    fn repeat2(self) -> [u8; 32] {
        let mut o = [0u8; 32];
        o[..16].copy_from_slice(&self);
        o[16..].copy_from_slice(&self);
        o
    }
}

fn send_out<S: Read + Write>(cx: &mut Ctx, s: &mut S, n: usize) -> bool {
    let out = unsafe { &*core::ptr::addr_of!(OUT) };
    let m = cx.mask();
    ws::send(s, ws::OP_TEXT, &out[..n], m).is_ok()
}

/// Writes one frame into OUT with `f`, then sends it. A frame that does not fit is an error.
fn frame<S: Read + Write>(cx: &mut Ctx, s: &mut S, f: impl FnOnce(&mut BufWriter, u32) -> core::fmt::Result) -> bool {
    let id = cx.id();
    let out = unsafe { &mut *core::ptr::addr_of_mut!(OUT) };
    let mut w = BufWriter::new(&mut out[..]);
    if f(&mut w, id).is_err() {
        log!("frame too large for the output buffer");
        return false;
    }
    let Some(n) = w.finish() else { return false };
    send_out(cx, s, n)
}

fn session<S: Read + Write>(cx: &mut Ctx, s: &mut S) -> bool {
    let mut path = [0u8; 192];
    let mut pw = BufWriter::new(&mut path);
    let _ = write!(pw, "/ws?token={TOKEN}");
    let Some(pn) = pw.finish() else { return false };
    let path = core::str::from_utf8(&path[..pn]).unwrap_or("/ws");
    let mut scratch = [0u8; 512];
    let key = cx.key();
    match ws::connect(s, HOST, path, key, &mut scratch) {
        Ok(()) => log!("agent socket open (WebSocket upgrade accepted)"),
        Err(ws::WsError::Handshake(st)) => {
            log!("agent socket refused at the upgrade: HTTP {st}");
            return EXPECT_REFUSED && st == 401;
        }
        Err(e) => {
            log!("agent socket upgrade failed: {e:?}");
            return false;
        }
    }
    if !frame(cx, s, |w, id| uamp::agent_session_create(w, id, AGENT, TOKEN)) {
        return false;
    }
    let rx = take!(WS_RX);
    loop {
        let mut mk = cx.rng.next_u32().to_le_bytes();
        let msg = ws::recv(s, rx, || {
            mk.rotate_left(1);
            mk
        });
        match msg {
            Ok(ws::Msg::Text(n)) => {
                let Ok(f) = core::str::from_utf8(&rx[..n]) else { continue };
                match jsonv::get(f, "type").and_then(jsonv::as_str) {
                    Some("session.created") => log!("session.created"),
                    Some("session.error") => log!("session.error: {}", jsonv::path(f, &["error", "message"]).unwrap_or("?")),
                    Some("input.text") => {
                        if !turn(cx, s, f) {
                            return false;
                        }
                        cx.turns_done += 1;
                    }
                    _ => {}
                }
            }
            Ok(ws::Msg::Binary(_)) => {}
            Ok(ws::Msg::Close(code)) => {
                log!("agent socket closed by the platform: {:?}", code);
                return if EXPECT_REFUSED { code == Some(4001) } else { cx.turns_done >= 1 };
            }
            Err(ws::WsError::Eof) if !EXPECT_REFUSED && cx.turns_done >= 1 => {
                log!("agent socket ended after {} turn(s)", cx.turns_done);
                return true;
            }
            Err(e) => {
                log!("agent socket error: {e:?}");
                return false;
            }
        }
    }
}

/// One relayed turn. `f` is the `input.text` frame (it stays in WS_RX for the whole turn).
fn turn<S: Read + Write>(cx: &mut Ctx, s: &mut S, f: &str) -> bool {
    let Some(sid) = jsonv::get(f, "session_id").and_then(jsonv::as_str) else { return true };
    let rid = "resp_m33";
    log!("turn {sid}: started");
    let Some(token) = jsonv::get(f, "payment_token").and_then(jsonv::as_str) else {
        return frame(cx, s, |w, id| uamp::response_error(w, id, sid, rid, "model_error", "this turn carried no payment token, so the model cannot be called"));
    };
    let history = jsonv::get(f, "messages").unwrap_or("[]");
    let conv = take!(CONV);
    let mut conv_len = 0usize;
    for round in 0..MAX_ROUNDS {
        // ---- the request: instructions, the relayed history (plain turns only), this turn so far, the tools
        let req = take!(REQ);
        let id = cx.id();
        let mut w = BufWriter::new(&mut req[..]);
        let built = (|| -> core::fmt::Result {
            uamp::llm_request_open(&mut w, id, 1024)?;
            w.write_str("{\"role\":\"system\",\"content\":")?;
            webagents::regspec::json::write_str(&mut w, INSTRUCTIONS)?;
            w.write_char('}')?;
            for item in jsonv::items(history) {
                let role = jsonv::get(item, "role").and_then(jsonv::as_str).unwrap_or("");
                // Tool rows and non-text content from a relayed history never reach the model.
                if !matches!(role, "user" | "assistant" | "system" | "developer") {
                    continue;
                }
                let Some(content) = jsonv::get(item, "content").and_then(jsonv::as_str) else { continue };
                if content.trim().is_empty() {
                    continue;
                }
                write!(w, ",{{\"role\":\"{role}\",\"content\":\"{content}\"}}")?;
            }
            w.write_str(core::str::from_utf8(&conv[..conv_len]).unwrap_or(""))?;
            uamp::llm_request_tools(&mut w)?;
            webagents::regspec::engine::write_tools(&pwm1::DEVICE, &mut w)?;
            uamp::llm_request_close(&mut w)
        })();
        let Some(req_len) = (if built.is_ok() { w.finish() } else { None }) else {
            log!("turn {sid}: request too large");
            return frame(cx, s, |w, id| uamp::response_error(w, id, sid, rid, "device_limit", "the conversation is too long for this device"));
        };
        let reply = match llm_call(cx, token, req_len) {
            Ok(n) => n,
            Err(why) => {
                log!("turn {sid}: model call failed: {why}");
                return frame(cx, s, |w, id| uamp::response_error(w, id, sid, rid, "model_error", why));
            }
        };
        let rf = unsafe { &*core::ptr::addr_of!(LLM_RX) };
        let Ok(rf) = core::str::from_utf8(&rf[..reply]) else { return false };
        let output = jsonv::path(rf, &["response", "output"]).unwrap_or("[]");
        let mut calls = 0;
        for item in jsonv::items(output) {
            match jsonv::get(item, "type").and_then(jsonv::as_str) {
                Some("text") => {
                    if let Some(t) = jsonv::get(item, "text").and_then(jsonv::as_str) {
                        if !t.is_empty() && !frame(cx, s, |w, id| uamp::delta_text_json(w, id, sid, rid, t)) {
                            return false;
                        }
                    }
                }
                Some("tool_call") => calls += 1,
                _ => {}
            }
        }
        log!("turn {sid}: round {round}: model answered with {calls} tool call(s)");
        if calls == 0 {
            return frame(cx, s, |w, id| uamp::response_done(w, id, sid, rid));
        }
        // The assistant message with its calls, then one tool message per call.
        let mut cw = BufWriter::new(&mut conv[conv_len..]);
        let _ = cw.write_str(",{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[");
        let mut first = true;
        for item in jsonv::items(output).filter(|i| jsonv::get(i, "type") == Some("\"tool_call\"")) {
            let tc = jsonv::get(item, "tool_call").unwrap_or("{}");
            let (cid, name, args) =
                (jsonv::get(tc, "id").unwrap_or("\"\""), jsonv::get(tc, "name").unwrap_or("\"\""), jsonv::get(tc, "arguments").unwrap_or("\"{}\""));
            let _ = write!(cw, "{}{{\"id\":{cid},\"type\":\"function\",\"function\":{{\"name\":{name},\"arguments\":{args}}}}}", if first { "" } else { "," });
            first = false;
        }
        let _ = cw.write_str("]}");
        let Some(n) = cw.finish() else {
            return frame(cx, s, |w, id| uamp::response_error(w, id, sid, rid, "device_limit", "too many tool calls for this device"));
        };
        conv_len += n;
        for item in jsonv::items(output).filter(|i| jsonv::get(i, "type") == Some("\"tool_call\"")) {
            let tc = jsonv::get(item, "tool_call").unwrap_or("{}");
            let cid = jsonv::get(tc, "id").and_then(jsonv::as_str).unwrap_or("");
            let name_j = jsonv::get(tc, "name").and_then(jsonv::as_str).unwrap_or("");
            let args_j = jsonv::get(tc, "arguments").and_then(jsonv::as_str).unwrap_or("{}");
            if !frame(cx, s, |w, id| uamp::delta_tool_call(w, id, sid, rid, cid, name_j, args_j)) {
                return false;
            }
            let mut name_b = [0u8; 64];
            let mut args_b = [0u8; 512];
            let name = jsonv::unescape(name_j, &mut name_b).unwrap_or("");
            let args = jsonv::unescape(args_j, &mut args_b).unwrap_or("\u{0}");
            let mut res = [0u8; 1024];
            let ctx = CallCtx { now_ms: crate::now_ms() as u64, confirmed: false };
            let (len, is_error) = match cx.engine.call(name, args, &ctx, &mut res) {
                Ok(n) => (n, false),
                Err(e) => {
                    let mut ew = BufWriter::new(&mut res[..]);
                    let _ = e.write_json(&pwm1::DEVICE, &mut ew);
                    (ew.len(), true)
                }
            };
            let result = core::str::from_utf8(&res[..len]).unwrap_or("{}");
            log!("turn {sid}: {name} {args} -> {result}");
            if !frame(cx, s, |w, id| uamp::delta_tool_result(w, id, sid, rid, cid, result, is_error)) {
                return false;
            }
            let mut cw = BufWriter::new(&mut conv[conv_len..]);
            let _ = write!(cw, ",{{\"role\":\"tool\",\"tool_call_id\":\"{cid}\",\"content\":");
            let _ = webagents::regspec::json::write_str(&mut cw, result);
            let _ = cw.write_char('}');
            let Some(n) = cw.finish() else {
                return frame(cx, s, |w, id| uamp::response_error(w, id, sid, rid, "device_limit", "tool results too large for this device"));
            };
            conv_len += n;
        }
    }
    let _ = frame(cx, s, |w, id| uamp::delta_text(w, id, sid, rid, "Stopped after the round limit without a final answer."));
    frame(cx, s, |w, id| uamp::response_done(w, id, sid, rid))
}

/// One model call over a second link: open, (TLS,) upgrade to `/llm`,
/// `session.create` with the turn's token, send REQ[..req_len], wait for
/// `response.done` (left in LLM_RX), close. Returns the frame length.
fn llm_call(cx: &mut Ctx, token: &str, req_len: usize) -> Result<usize, &'static str> {
    let port: u16 = PORT.parse().unwrap_or(443);
    cx.esp.borrow_mut().open(1, HOST, port).map_err(|_| "cannot open the model link")?;
    let link = Link { at: cx.esp, id: 1, timeout_ms: 120_000 };
    let r = if TLS {
        let mut tls = TlsConnection::new(link, take!(TLS1_RX), take!(TLS1_TX));
        let config = TlsConfig::new().with_server_name(HOST);
        let provider = Provider { rng: ChaCha20Rng::from_seed(cx.key().repeat2()), verifier: CertVerifier::new(embedded_tls::Certificate::X509(ROOT_CA)) };
        tls.open(TlsContext::new(&config, provider)).map_err(|_| "TLS to the model socket failed")?;
        llm_over(cx, &mut tls, token, req_len)
    } else {
        let mut l = link;
        llm_over(cx, &mut l, token, req_len)
    };
    let _ = cx.esp.borrow_mut().close(1);
    r
}

fn llm_over<S: Read + Write>(cx: &mut Ctx, s: &mut S, token: &str, req_len: usize) -> Result<usize, &'static str> {
    let mut scratch = [0u8; 512];
    let key = cx.key();
    ws::connect(s, HOST, "/llm", key, &mut scratch).map_err(|_| "the model socket refused the upgrade")?;
    if !frame(cx, s, |w, id| uamp::llm_session_create(w, id, token)) {
        return Err("cannot send to the model socket");
    }
    let rx = take!(LLM_RX);
    loop {
        let mut mk = cx.mask();
        let m = ws::recv(s, rx, || {
            mk.rotate_left(1);
            mk
        });
        match m {
            Ok(ws::Msg::Text(n)) => {
                let f = core::str::from_utf8(&rx[..n]).map_err(|_| "not text")?;
                match jsonv::get(f, "type").and_then(jsonv::as_str) {
                    Some("session.created") => {
                        let req = unsafe { &*core::ptr::addr_of!(REQ) };
                        let mk = cx.mask();
                        ws::send(s, ws::OP_TEXT, &req[..req_len], mk).map_err(|_| "cannot send the request")?;
                    }
                    Some("response.done") => {
                        let mk = cx.mask();
                        let _ = ws::send(s, ws::OP_CLOSE, &1000u16.to_be_bytes(), mk);
                        return Ok(n);
                    }
                    Some("response.error") => return Err("the model call failed"),
                    Some("payment.required") => return Err("the turn's token does not cover the model call"),
                    _ => {}
                }
            }
            Ok(ws::Msg::Close(_)) => return Err("the model socket closed early"),
            Ok(_) => {}
            Err(ws::WsError::TooLarge(_)) => return Err("the model's answer is too large for this device"),
            Err(_) => return Err("the model socket failed"),
        }
    }
}
