//! The agent socket: one long-lived WebSocket to the platform's `/ws`, with the
//! behaviour of the TypeScript SDK's bridge:
//!
//! - a bounded handshake: a server that accepts TCP and never answers the
//!   upgrade fails the attempt instead of leaving the agent silently offline;
//! - reconnect with a delay and a budget that resets on every successful open;
//! - a ping every 55 s;
//! - one turn task per `input.text`, cancelled by `response.cancel` or
//!   `session.end`, never started by the broadcast chat envelope;
//! - a per-agent key only: a token without an `agent_id` claim is refused at
//!   start-up, because the platform accepts it and then never routes a turn.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::{self, Message};

use super::device::Devices;
use super::llm::ModelBackend;
use super::turn::{run_turn, TurnLimits};
use super::GatewayError;
use crate::uamp::{self, Inbound};

/// The agent socket's settings. [`PortalConfig::new`] fills the defaults; the
/// fields are public so a program can change them before [`run_portal`].
#[derive(Debug, Clone)]
pub struct PortalConfig {
    /// `wss://host/ws` (http(s) and a bare origin are accepted, see [`resolve_ws_url`]).
    pub url: String,
    /// The agent's name in code; a per-agent key's `agent_name` claim wins.
    pub agent: String,
    /// Per-agent key. Read from the environment by the caller, never from a file.
    pub token: String,
    /// Time between `ping` frames, which keep the socket from idling out
    /// (default 55 s).
    pub ping_interval: Duration,
    /// Longest wait for the WebSocket upgrade to complete (default 30 s).
    pub handshake_timeout: Duration,
    /// Wait between two connection attempts (default 5 s).
    pub reconnect_delay: Duration,
    /// Consecutive failed attempts before [`run_portal`] gives up (default 10).
    /// A connection that opens resets the count.
    pub max_reconnect_attempts: u32,
    /// Per-turn limits (model rounds).
    pub limits: TurnLimits,
    /// Signed in as a device (the agent's own key): each connection opens with a
    /// one-use ticket fetched by a signed request instead of a token, and
    /// `session.create` carries none.
    pub device: Option<Arc<super::signin::SignIn>>,
}

impl PortalConfig {
    /// Settings for `agent` at `url` with the per-agent key `token` and every
    /// default. `url` is resolved with [`resolve_ws_url`]. For a device sign-in,
    /// pass an empty token and set [`PortalConfig::device`].
    pub fn new(url: &str, agent: &str, token: &str) -> Self {
        PortalConfig {
            url: resolve_ws_url(url),
            agent: agent.into(),
            token: token.into(),
            ping_interval: Duration::from_secs(55),
            handshake_timeout: Duration::from_secs(30),
            reconnect_delay: Duration::from_secs(5),
            max_reconnect_attempts: 10,
            limits: TurnLimits::default(),
            device: None,
        }
    }
}

/// What happened on the socket, for logs and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortalEvent {
    /// A connection attempt is starting.
    Connecting {
        /// Failed attempts since the last connection that opened (0 for the first).
        attempt: u32,
    },
    /// The upgrade was refused with an HTTP status (401 for a bad key).
    HandshakeRefused {
        /// The HTTP status of the refusal.
        status: u16,
    },
    /// The upgrade did not complete: no answer in time, a TLS or network
    /// error, or (with a device sign-in) no ticket could be fetched.
    HandshakeFailed {
        /// What went wrong, as text.
        error: String,
    },
    /// The WebSocket is open; `session.create` is being sent.
    Open,
    /// The platform accepted the session: the agent is online.
    SessionCreated {
        /// The agent name the platform bound the socket to.
        agent: Option<String>,
    },
    /// The platform refused the session.
    SessionError {
        /// The platform's error code.
        code: String,
        /// The platform's explanation.
        message: String,
    },
    /// The socket closed; a reconnect follows unless shutting down.
    Closed {
        /// The close code, when the peer sent a close frame.
        code: Option<u16>,
        /// The peer's close reason, or the error that ended the stream.
        reason: String,
    },
    /// A relayed turn arrived and its task was started.
    TurnStarted {
        /// The turn's session.
        session_id: String,
    },
    /// The reconnect budget is spent; [`run_portal`] returns an error.
    GaveUp {
        /// How many consecutive attempts failed.
        attempts: u32,
    },
}

/// A callback that receives every [`PortalEvent`], for a log or a test. Called
/// on the gateway's own task, so it must not block.
pub type EventSink = Arc<dyn Fn(PortalEvent) + Send + Sync>;

/// `https://h` -> `wss://h/ws`; a URL with a path is kept.
pub fn resolve_ws_url(raw: &str) -> String {
    let raw = raw.trim();
    let mut s = if let Some(r) = raw.strip_prefix("https://") {
        format!("wss://{r}")
    } else if let Some(r) = raw.strip_prefix("http://") {
        format!("ws://{r}")
    } else {
        raw.to_string()
    };
    // A bare origin, with or without a trailing slash, gets `/ws`.
    let host_len = s.find("://").map_or(0, |i| i + 3);
    if !s[host_len..].trim_end_matches('/').contains('/') {
        while s.ends_with('/') {
            s.pop();
        }
        s.push_str("/ws");
    }
    s
}

/// The `/llm` URL next to a `/ws` URL.
pub fn llm_url_for(ws_url: &str) -> String {
    match ws_url.rsplit_once('/') {
        Some((head, _)) => format!("{head}/llm"),
        None => format!("{ws_url}/llm"),
    }
}

/// Claims of a JWT, without verifying it (the platform verifies; this only
/// catches the wrong KIND of key before connecting).
fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    serde_json::from_slice(&b64url_decode(payload)?).ok()
}

fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' | b'+' => 62,
            b'_' | b'/' => 63,
            b'=' => continue,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// Refuse an account-wide key (see the module documentation). Returns the platform's name for
/// the agent: a per-agent key's `agent_name` claim (`<owner>.<name>`), which the
/// platform resolves, else the configured name.
///
/// A key that is not a readable JWT is passed on for the platform to judge.
///
/// # Errors
/// [`GatewayError::Config`] for an empty key, or a JWT without an `agent_id`
/// claim.
pub fn check_token(cfg: &PortalConfig) -> Result<String, GatewayError> {
    if cfg.token.trim().is_empty() {
        return Err(GatewayError::Config("no per-agent key: set WEBAGENTS_AGENT_TOKEN to a key minted for this agent".into()));
    }
    let Some(claims) = jwt_claims(&cfg.token) else { return Ok(cfg.agent.clone()) }; // not a readable JWT: the platform judges it
    if claims.get("agent_id").is_none() && std::env::var("WEBAGENTS_ALLOW_UNBOUND_TOKEN").as_deref() != Ok("1") {
        return Err(GatewayError::Config(format!(
            "the key is not bound to an agent (subject {}, no agent_id claim): the platform would accept the connection and never route a turn to it. Mint a per-agent key for this agent",
            claims.get("sub").and_then(Value::as_str).unwrap_or("?")
        )));
    }
    Ok(claims.get("agent_name").and_then(Value::as_str).filter(|n| !n.is_empty()).map(str::to_string).unwrap_or_else(|| cfg.agent.clone()))
}

fn encode_query(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

/// Runs the agent socket until `shutdown` turns true or the reconnect budget is spent.
///
/// Each `input.text` starts one turn task over `devices` and `model`; a new
/// turn on a session cancels the one still running on it. On shutdown every
/// running turn is cancelled and the socket is closed cleanly.
///
/// # Errors
/// [`GatewayError::Config`] before connecting when the key is missing or not
/// bound to an agent; [`GatewayError::Portal`] after
/// [`PortalConfig::max_reconnect_attempts`] consecutive failed attempts.
pub async fn run_portal(
    cfg: PortalConfig,
    devices: Arc<Devices>,
    model: Arc<dyn ModelBackend>,
    mut shutdown: watch::Receiver<bool>,
    events: Option<EventSink>,
) -> Result<(), GatewayError> {
    super::init_tls();
    let agent = if cfg.device.is_some() { cfg.agent.clone() } else { check_token(&cfg)? };
    let emit = |e: PortalEvent| {
        if let Some(sink) = &events {
            sink(e);
        }
    };
    let mut attempts = 0u32;
    loop {
        if *shutdown.borrow() {
            return Ok(());
        }
        emit(PortalEvent::Connecting { attempt: attempts });
        let opened = one_connection(&cfg, &agent, &devices, &model, &mut shutdown, &emit).await;
        if *shutdown.borrow() {
            return Ok(());
        }
        if opened {
            attempts = 0; // a connection that opened resets the budget
        }
        if attempts >= cfg.max_reconnect_attempts {
            emit(PortalEvent::GaveUp { attempts });
            return Err(GatewayError::Portal(format!("gave up after {attempts} reconnect attempts to {}", cfg.url)));
        }
        attempts += 1;
        tokio::select! {
            _ = tokio::time::sleep(cfg.reconnect_delay) => {}
            _ = shutdown.changed() => {}
        }
    }
}

/// One socket lifetime. Returns whether the socket opened.
async fn one_connection(
    cfg: &PortalConfig,
    agent: &str,
    devices: &Arc<Devices>,
    model: &Arc<dyn ModelBackend>,
    shutdown: &mut watch::Receiver<bool>,
    emit: &impl Fn(PortalEvent),
) -> bool {
    let sep = if cfg.url.contains('?') { '&' } else { '?' };
    let url = match &cfg.device {
        // A one-use ticket per attempt (60 s), fetched by a signed request.
        Some(si) => {
            let si = si.clone();
            match tokio::task::spawn_blocking(move || si.ws_ticket()).await {
                Ok(Ok(t)) => format!("{}{sep}ticket={}", cfg.url, encode_query(&t)),
                Ok(Err(e)) => {
                    emit(PortalEvent::HandshakeFailed { error: e.to_string() });
                    return false;
                }
                Err(e) => {
                    emit(PortalEvent::HandshakeFailed { error: e.to_string() });
                    return false;
                }
            }
        }
        None => format!("{}{sep}token={}", cfg.url, encode_query(&cfg.token)),
    };
    let connect = tokio::time::timeout(cfg.handshake_timeout, tokio_tungstenite::connect_async(url.as_str()));
    let ws = match connect.await {
        Err(_) => {
            emit(PortalEvent::HandshakeFailed { error: format!("no handshake within {:?}", cfg.handshake_timeout) });
            return false;
        }
        Ok(Err(tungstenite::Error::Http(resp))) => {
            emit(PortalEvent::HandshakeRefused { status: resp.status().as_u16() });
            return false;
        }
        Ok(Err(e)) => {
            emit(PortalEvent::HandshakeFailed { error: e.to_string() });
            return false;
        }
        Ok(Ok((ws, _))) => ws,
    };
    emit(PortalEvent::Open);
    let (mut sink, mut stream) = ws.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
    let _ = tx.send(uamp::agent_session_create(agent, if cfg.device.is_some() { None } else { Some(&cfg.token) }));
    let mut ping = tokio::time::interval(cfg.ping_interval);
    ping.tick().await; // the first tick is immediate
    let mut turns: HashMap<String, Arc<AtomicBool>> = HashMap::new();

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    for c in turns.values() { c.store(true, Ordering::Relaxed); }
                    let _ = sink.send(Message::Close(None)).await;
                    return true;
                }
            }
            _ = ping.tick() => {
                if sink.send(Message::Text(uamp::ping().to_string().into())).await.is_err() { break; }
            }
            Some(frame) = rx.recv() => {
                if sink.send(Message::Text(frame.to_string().into())).await.is_err() { break; }
            }
            msg = stream.next() => {
                let text = match msg {
                    None => { emit(PortalEvent::Closed { code: None, reason: "stream ended".into() }); break; }
                    Some(Err(e)) => { emit(PortalEvent::Closed { code: None, reason: e.to_string() }); break; }
                    Some(Ok(Message::Close(c))) => {
                        emit(PortalEvent::Closed { code: c.as_ref().map(|c| u16::from(c.code)), reason: c.map(|c| c.reason.to_string()).unwrap_or_default() });
                        break;
                    }
                    Some(Ok(Message::Text(t))) => t.to_string(),
                    Some(Ok(_)) => continue, // the platform sends no binary frames on this socket
                };
                match uamp::parse_inbound(&text) {
                    None | Some(Inbound::Other(_)) => {}
                    Some(Inbound::SessionCreated { agent, .. }) => emit(PortalEvent::SessionCreated { agent }),
                    Some(Inbound::SessionError { code, message }) => {
                        eprintln!("[webagents-gateway] session.error {code}: {message}");
                        emit(PortalEvent::SessionError { code, message });
                    }
                    Some(Inbound::SessionEnd { session_id }) | Some(Inbound::Cancel { session_id }) if turns.contains_key(&session_id) => {
                        if let Some(c) = turns.remove(&session_id) { c.store(true, Ordering::Relaxed); }
                        let _ = tx.send(uamp::response_cancelled(&session_id));
                    }
                    Some(Inbound::SessionEnd { .. }) | Some(Inbound::Cancel { .. }) => {}
                    Some(Inbound::Input(input)) => {
                        emit(PortalEvent::TurnStarted { session_id: input.session_id.clone() });
                        let cancel = Arc::new(AtomicBool::new(false));
                        if let Some(old) = turns.insert(input.session_id.clone(), cancel.clone()) { old.store(true, Ordering::Relaxed); }
                        tokio::spawn(run_turn(devices.clone(), model.clone(), input, tx.clone(), cancel, cfg.limits.clone()));
                    }
                }
            }
        }
    }
    for c in turns.values() {
        c.store(true, Ordering::Relaxed);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls() {
        assert_eq!(resolve_ws_url("https://robutler.ai"), "wss://robutler.ai/ws");
        assert_eq!(resolve_ws_url("http://127.0.0.1:9/"), "ws://127.0.0.1:9/ws");
        assert_eq!(llm_url_for("wss://robutler.ai/ws"), "wss://robutler.ai/llm");
    }

    #[test]
    fn account_keys_are_refused() {
        // {"sub":"user_1"} and {"sub":"user_1","agent_id":"a1","agent_name":"me.dev"}, unsigned.
        let owner = "e30.eyJzdWIiOiJ1c2VyXzEifQ.x";
        let agent = "e30.eyJzdWIiOiJ1c2VyXzEiLCJhZ2VudF9pZCI6ImExIiwiYWdlbnRfbmFtZSI6Im1lLmRldiJ9.x";
        assert!(check_token(&PortalConfig::new("wss://h/ws", "dev", owner)).is_err());
        assert_eq!(check_token(&PortalConfig::new("wss://h/ws", "dev", agent)).unwrap(), "me.dev");
        assert_eq!(check_token(&PortalConfig::new("wss://h/ws", "dev", "opaque-key")).unwrap(), "dev");
        assert!(check_token(&PortalConfig::new("wss://h/ws", "dev", " ")).is_err());
    }
}
