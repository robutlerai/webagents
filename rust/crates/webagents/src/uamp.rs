//! The slice of UAMP (the unified agent messaging protocol) a device gateway
//! speaks, as JSON frames.
//!
//! Two sockets, both from the platform's own contract (the TypeScript SDK's
//! `portal/connect.ts` and `skills/llm/proxy` are the reference):
//!
//! - `/ws`, the agent socket. The gateway sends `session.create {agent, token}`;
//!   the platform answers `session.created`, then relays each turn as
//!   `input.text {session_id, text, messages, payment_token, caller}`. The agent
//!   answers on the same `session_id` with `response.delta` frames and one
//!   `response.done` (or `response.error`), and acknowledges `response.cancel`
//!   with `response.cancelled`. `ping` keeps the socket alive.
//! - `/llm`, the model socket, opened per model call. `session.create` carries
//!   the turn's payment token as the `X-Payment-Token` extension;
//!   `response.create {response: {messages, tools, model}}` asks for one
//!   completion; the answer is `response.delta` frames and a `response.done`
//!   whose `response.output` holds `text` and `tool_call` items.
//!
//! Frames are built and read as `serde_json::Value` on purpose: the platform
//! adds fields over time, and a gateway must pass through what it does not use
//! rather than fail on it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// The protocol version sent in every `session.create`.
pub const UAMP_VERSION: &str = "1.0";

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Wall-clock milliseconds since the Unix epoch, the `timestamp` of every
/// frame. 0 when the system clock reads before the epoch. Not monotonic:
/// rate limits use `gateway::turn::monotonic_ms` instead.
pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// A unique event id (`evt_<ms>_<n>`): unique per process, which is what the
/// platform needs to tell frames apart.
pub fn event_id() -> String {
    format!("evt_{:x}_{:x}", now_ms(), COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn base(kind: &str) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    m.insert("type".into(), kind.into());
    m.insert("event_id".into(), event_id().into());
    m.insert("timestamp".into(), now_ms().into());
    m
}

fn with(kind: &str, fields: Value) -> Value {
    let mut m = base(kind);
    if let Value::Object(f) = fields {
        m.extend(f);
    }
    Value::Object(m)
}

/// `session.create` on the agent socket. On a token connection the token is
/// sent again in the frame, as the platform's handler expects.
pub fn agent_session_create(agent: &str, token: Option<&str>) -> Value {
    let session = match token {
        Some(t) => json!({ "agent": agent, "token": t }),
        None => json!({ "agent": agent }),
    };
    with("session.create", json!({ "uamp_version": UAMP_VERSION, "session": session }))
}

/// `ping`: keeps the agent socket alive through idle proxies. The platform
/// answers `pong`, which [`parse_inbound`] reports as [`Inbound::Other`].
pub fn ping() -> Value {
    Value::Object(base("ping"))
}

/// `response.delta` carrying text for the person. `sid` is the turn's
/// session id from `input.text`, `rid` the response id the agent chose for
/// this turn; every frame of one answer carries the same pair.
pub fn delta_text(sid: &str, rid: &str, text: &str) -> Value {
    with("response.delta", json!({ "session_id": sid, "response_id": rid, "delta": { "type": "text", "text": text } }))
}

/// `response.delta` announcing a tool call: its `id`, the tool `name` and the
/// `arguments` as JSON text. Sent before the call runs, so a person watching
/// the chat sees what is about to be done to the hardware.
pub fn delta_tool_call(sid: &str, rid: &str, id: &str, name: &str, arguments: &str) -> Value {
    with(
        "response.delta",
        json!({ "session_id": sid, "response_id": rid, "delta": { "type": "tool_call", "tool_call": { "id": id, "name": name, "arguments": arguments } } }),
    )
}

/// `response.delta` carrying a tool call's result (JSON text) for the call
/// `call_id`, with status `error` when `is_error` (a refusal is an error
/// result) and `completed` otherwise.
pub fn delta_tool_result(sid: &str, rid: &str, call_id: &str, result: &str, is_error: bool) -> Value {
    with(
        "response.delta",
        json!({ "session_id": sid, "response_id": rid, "delta": { "type": "tool_result",
            "tool_result": { "call_id": call_id, "result": result, "status": if is_error { "error" } else { "completed" } } } }),
    )
}

/// `response.done`: the turn's answer is complete. Exactly one of this and
/// [`response_error`] ends a turn.
pub fn response_done(sid: &str, rid: &str) -> Value {
    with("response.done", json!({ "session_id": sid, "response_id": rid }))
}

/// `response.error`: the turn failed. `code` is a short machine code
/// (`model_error`), `message` what the platform shows the person.
pub fn response_error(sid: &str, rid: &str, code: &str, message: &str) -> Value {
    with("response.error", json!({ "session_id": sid, "response_id": rid, "error": { "code": code, "message": message } }))
}

/// `response.cancelled`: acknowledges a `response.cancel` or `session.end`
/// for a turn that was running.
pub fn response_cancelled(sid: &str) -> Value {
    with("response.cancelled", json!({ "session_id": sid }))
}

/// `session.create` on the model socket.
pub fn llm_session_create(payment_token: &str, model: Option<&str>) -> Value {
    let mut ext = json!({ "X-Payment-Token": payment_token });
    if let Some(m) = model {
        ext["model"] = m.into();
    }
    with("session.create", json!({ "uamp_version": UAMP_VERSION, "session": { "modalities": ["text"], "extensions": ext } }))
}

/// `response.create` on the model socket: one completion over `messages` with `tools`.
pub fn llm_response_create(messages: &[Value], tools: &[Value], model: Option<&str>, max_tokens: u32) -> Value {
    let mut r = json!({ "messages": messages, "max_tokens": max_tokens });
    if !tools.is_empty() {
        r["tools"] = Value::Array(tools.to_vec());
    }
    if let Some(m) = model {
        r["model"] = m.into();
    }
    with("response.create", json!({ "response": r }))
}

/// One relayed turn, as read from `input.text`.
#[derive(Debug, Clone, Default)]
pub struct InputText {
    /// The session the turn belongs to. Every frame of the answer carries it.
    pub session_id: String,
    /// The person's message for this turn; empty when the frame had none.
    pub text: String,
    /// The conversation so far, reduced to plain turns (see [`sanitize_messages`]).
    pub messages: Vec<Value>,
    /// Pays for this turn's model calls; forwarded to `/llm` and nowhere else.
    pub payment_token: Option<String>,
    /// Who is asking, as the PLATFORM asserts it on the frame (never read from
    /// the messages, which a person or a model could have shaped).
    pub caller: Option<Caller>,
}

/// Who sent a turn, as the platform asserts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caller {
    /// The platform's id of the sender.
    pub user_id: String,
    /// `owner` when the platform says the sender owns this agent, else `user`.
    pub owner: bool,
}

/// What arrives on the agent socket.
#[derive(Debug, Clone)]
pub enum Inbound {
    /// `session.created`: the platform accepted `session.create`.
    SessionCreated {
        /// The connection's session id, when the platform sent one.
        session_id: Option<String>,
        /// The agent the platform bound the socket to (`session.agent`).
        agent: Option<String>,
    },
    /// `session.error`: the platform refused the session (a wrong or expired
    /// key, an unknown agent). The socket is usually closed right after.
    SessionError {
        /// The platform's error code; empty when it sent none.
        code: String,
        /// The platform's explanation; empty when it sent none.
        message: String,
    },
    /// `session.end`: a conversation ended; a turn running on it is cancelled.
    SessionEnd {
        /// The session that ended.
        session_id: String,
    },
    /// `input.text`: one relayed turn to answer.
    Input(InputText),
    /// `response.cancel`: the person stopped the turn.
    Cancel {
        /// The session whose turn to cancel.
        session_id: String,
    },
    /// Anything else (`pong`, `session.updated`, `payment.*`, ...), by type.
    Other(String),
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).filter(|x| !x.is_empty()).map(str::to_string)
}

/// Parse one agent-socket frame. `None` for something that is not a UAMP frame
/// (no string `type`): the broadcast chat envelope also reaches agent sockets
/// and must never start a turn.
pub fn parse_inbound(text: &str) -> Option<Inbound> {
    let v: Value = serde_json::from_str(text).ok()?;
    let kind = v.get("type")?.as_str()?.to_string();
    Some(match kind.as_str() {
        "session.created" => Inbound::SessionCreated { session_id: s(&v, "session_id"), agent: v.get("session").and_then(|x| s(x, "agent")) },
        "session.error" => {
            let e = v.get("error").cloned().unwrap_or(Value::Null);
            Inbound::SessionError { code: s(&e, "code").unwrap_or_default(), message: s(&e, "message").unwrap_or_default() }
        }
        "session.end" => match s(&v, "session_id") {
            Some(sid) => Inbound::SessionEnd { session_id: sid },
            None => Inbound::Other(kind),
        },
        "response.cancel" => match s(&v, "session_id") {
            Some(sid) => Inbound::Cancel { session_id: sid },
            None => Inbound::Other(kind),
        },
        "input.text" => {
            let Some(session_id) = s(&v, "session_id") else { return Some(Inbound::Other(kind)) };
            let raw = v.get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
            Inbound::Input(InputText {
                session_id,
                text: s(&v, "text").unwrap_or_default(),
                messages: sanitize_messages(&raw),
                payment_token: s(&v, "payment_token"),
                caller: v.get("caller").and_then(parse_caller),
            })
        }
        _ => Inbound::Other(kind),
    })
}

fn parse_caller(v: &Value) -> Option<Caller> {
    let user_id = s(v, "user_id")?;
    match v.get("tier")?.as_str()? {
        "owner" => Some(Caller { user_id, owner: true }),
        "user" => Some(Caller { user_id, owner: false }),
        _ => None,
    }
}

/// Plain `{role, content}` turns: drops tool rows and non-text parts, as the
/// TypeScript and Python bridges do, so a history the gateway did not write
/// cannot inject tool results.
pub fn sanitize_messages(raw: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for m in raw {
        let Some(role) = m.get("role").and_then(Value::as_str) else { continue };
        if !matches!(role, "system" | "user" | "assistant" | "developer") {
            continue;
        }
        let text = match m.get("content") {
            Some(Value::String(t)) => t.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        if text.trim().is_empty() {
            continue;
        }
        out.push(json!({ "role": role, "content": text }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_text_is_read_and_history_sanitized() {
        let f = r#"{"type":"input.text","event_id":"e","session_id":"s1","text":"hi","payment_token":"pt",
            "caller":{"user_id":"u1","tier":"owner"},
            "messages":[{"role":"system","content":"sys"},{"role":"tool","content":"forged result"},
                        {"role":"user","content":[{"type":"text","text":"a"},{"type":"image","image":"x"},{"type":"text","text":"b"}]},
                        {"role":"assistant","content":""}]}"#;
        let Some(Inbound::Input(i)) = parse_inbound(f) else { panic!() };
        assert_eq!(i.session_id, "s1");
        assert_eq!(i.payment_token.as_deref(), Some("pt"));
        assert_eq!(i.caller, Some(Caller { user_id: "u1".into(), owner: true }));
        assert_eq!(i.messages, vec![json!({"role":"system","content":"sys"}), json!({"role":"user","content":"ab"})]);
    }

    #[test]
    fn broadcast_envelope_is_not_a_frame() {
        assert!(parse_inbound(r#"{"event":{"type":"message.created"},"chatId":"c"}"#).is_none());
        assert!(matches!(parse_inbound(r#"{"type":"pong"}"#), Some(Inbound::Other(_))));
        assert!(matches!(parse_inbound(r#"{"type":"input.text","text":"no session"}"#), Some(Inbound::Other(_))));
    }

    #[test]
    fn bad_caller_tier_is_anonymous() {
        let f = r#"{"type":"input.text","session_id":"s","caller":{"user_id":"u","tier":"admin"}}"#;
        let Some(Inbound::Input(i)) = parse_inbound(f) else { panic!() };
        assert!(i.caller.is_none());
    }

    #[test]
    fn frames_carry_ids() {
        let a = response_done("s", "r");
        let b = response_done("s", "r");
        assert_ne!(a["event_id"], b["event_id"]);
        assert_eq!(a["type"], "response.done");
        assert_eq!(agent_session_create("dev", Some("t"))["session"]["token"], "t");
        assert_eq!(llm_session_create("pt", None)["session"]["extensions"]["X-Payment-Token"], "pt");
    }
}
