//! UAMP frames a device writes, into any `fmt::Write` (a fixed buffer on a
//! microcontroller). Same frames as the `uamp` module on the host (feature `std`), without
//! `serde`. Arguments named `*_json` are JSON string CONTENT that is already
//! escaped (copied from a received frame); every other string is escaped here.

use core::fmt::{self, Write};

use crate::regspec::json::write_str as esc;

fn head<W: Write>(w: &mut W, kind: &str, id: u32) -> fmt::Result {
    write!(w, "{{\"type\":\"{kind}\",\"event_id\":\"evt_{id:x}\"")
}

/// `session.create` on the agent socket, with the per-agent key. `id` makes
/// the frame's `event_id` (`evt_<id in hex>`); use a counter, as for every
/// writer here.
pub fn agent_session_create<W: Write>(w: &mut W, id: u32, agent: &str, token: &str) -> fmt::Result {
    head(w, "session.create", id)?;
    w.write_str(",\"uamp_version\":\"1.0\",\"session\":{\"agent\":")?;
    esc(w, agent)?;
    w.write_str(",\"token\":")?;
    esc(w, token)?;
    w.write_str("}}")
}

/// `ping`: keeps the agent socket alive.
pub fn ping<W: Write>(w: &mut W, id: u32) -> fmt::Result {
    head(w, "ping", id)?;
    w.write_char('}')
}

fn delta_head<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str, kind: &str) -> fmt::Result {
    head(w, "response.delta", id)?;
    write!(w, ",\"session_id\":\"{sid_json}\",\"response_id\":")?;
    esc(w, rid)?;
    write!(w, ",\"delta\":{{\"type\":\"{kind}\"")
}

/// `response.delta` with text that is ALREADY JSON string content (copied
/// from the model's answer frame without unescaping it), so the device needs
/// no buffer for the unescaped text.
pub fn delta_text_json<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str, text_json: &str) -> fmt::Result {
    delta_head(w, id, sid_json, rid, "text")?;
    write!(w, ",\"text\":\"{text_json}\"}}}}")
}

/// `response.delta` with plain text, escaped here.
pub fn delta_text<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str, text: &str) -> fmt::Result {
    delta_head(w, id, sid_json, rid, "text")?;
    w.write_str(",\"text\":")?;
    esc(w, text)?;
    w.write_str("}}")
}

/// `response.delta` announcing a tool call. The call id, name and arguments
/// are JSON string content as received from the model, passed through.
pub fn delta_tool_call<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str, call_id_json: &str, name_json: &str, args_json: &str) -> fmt::Result {
    delta_head(w, id, sid_json, rid, "tool_call")?;
    write!(w, ",\"tool_call\":{{\"id\":\"{call_id_json}\",\"name\":\"{name_json}\",\"arguments\":\"{args_json}\"}}}}}}")
}

/// `response.delta` with a tool call's result. `result` is the engine's
/// JSON text, escaped here into a string; `is_error` marks a refusal.
pub fn delta_tool_result<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str, call_id_json: &str, result: &str, is_error: bool) -> fmt::Result {
    delta_head(w, id, sid_json, rid, "tool_result")?;
    write!(w, ",\"tool_result\":{{\"call_id\":\"{call_id_json}\",\"result\":")?;
    esc(w, result)?;
    write!(w, ",\"status\":\"{}\"}}}}}}", if is_error { "error" } else { "completed" })
}

/// `response.done`: the turn's answer is complete.
pub fn response_done<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str) -> fmt::Result {
    head(w, "response.done", id)?;
    write!(w, ",\"session_id\":\"{sid_json}\",\"response_id\":")?;
    esc(w, rid)?;
    w.write_char('}')
}

/// `response.error`: the turn failed, with a short code and a message for the person.
pub fn response_error<W: Write>(w: &mut W, id: u32, sid_json: &str, rid: &str, code: &str, message: &str) -> fmt::Result {
    head(w, "response.error", id)?;
    write!(w, ",\"session_id\":\"{sid_json}\",\"response_id\":")?;
    esc(w, rid)?;
    w.write_str(",\"error\":{\"code\":")?;
    esc(w, code)?;
    w.write_str(",\"message\":")?;
    esc(w, message)?;
    w.write_str("}}")
}

/// `response.cancelled`: acknowledges a cancelled turn.
pub fn response_cancelled<W: Write>(w: &mut W, id: u32, sid_json: &str) -> fmt::Result {
    head(w, "response.cancelled", id)?;
    write!(w, ",\"session_id\":\"{sid_json}\"}}")
}

/// `session.create` on the model socket, carrying the turn's payment token
/// (JSON string content, as received in `input.text`).
pub fn llm_session_create<W: Write>(w: &mut W, id: u32, payment_token_json: &str) -> fmt::Result {
    head(w, "session.create", id)?;
    write!(w, ",\"uamp_version\":\"1.0\",\"session\":{{\"modalities\":[\"text\"],\"extensions\":{{\"X-Payment-Token\":\"{payment_token_json}\"}}}}}}")
}

/// `response.create` is written in three parts so the messages can be streamed
/// from several buffers: [`llm_request_open`], the messages (comma-separated
/// objects), then [`llm_request_tools`] and [`llm_request_close`].
pub fn llm_request_open<W: Write>(w: &mut W, id: u32, max_tokens: u32) -> fmt::Result {
    head(w, "response.create", id)?;
    write!(w, ",\"response\":{{\"max_tokens\":{max_tokens},\"messages\":[")
}

/// Ends the messages and opens the tool list; write the tools (a JSON array,
/// as `regspec::engine::write_tools` emits) right after it.
pub fn llm_request_tools<W: Write>(w: &mut W) -> fmt::Result {
    w.write_str("],\"tools\":")
}

/// Closes the `response.create` frame after the tool list.
pub fn llm_request_close<W: Write>(w: &mut W) -> fmt::Result {
    w.write_str("}}")
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::string::String;

    #[test]
    fn frames_are_json() {
        let mut s = String::new();
        delta_tool_result(&mut s, 7, "s1", "r", "c1", "{\"ok\":true}", false).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["delta"]["tool_result"]["result"], "{\"ok\":true}");
        let mut s = String::new();
        delta_tool_call(&mut s, 8, "s1", "r", "c1", "pwm1_set_deadtime", r#"{\"deadtime\":1}"#).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["delta"]["tool_call"]["arguments"], "{\"deadtime\":1}");
        let mut s = String::new();
        llm_request_open(&mut s, 9, 512).unwrap();
        s.push_str(r#"{"role":"user","content":"hi"}"#);
        llm_request_tools(&mut s).unwrap();
        s.push_str("[]");
        llm_request_close(&mut s).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["response"]["messages"][0]["content"], "hi");
        let mut s = String::new();
        agent_session_create(&mut s, 1, "dev", "k\"").unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&s).unwrap()["session"]["token"], "k\"");
    }
}
