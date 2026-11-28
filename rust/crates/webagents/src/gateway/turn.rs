//! One turn: the model is asked, its tool calls run on the devices (through
//! each device's envelope), the results go back to the model, until it answers
//! in text or the round limit is reached.
//!
//! Every tool call and its result are also streamed to the chat as
//! `tool_call` / `tool_result` deltas, so a person watching sees exactly what
//! was done to the hardware, including what was refused and why.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

use super::device::Devices;
use super::llm::{ModelBackend, ModelRequest};
use crate::regspec::CallCtx;
use crate::uamp::{self, InputText};

/// Limits for one turn.
#[derive(Debug, Clone)]
pub struct TurnLimits {
    /// Model calls per turn. Each round may run several tool calls.
    pub max_rounds: usize,
}

impl Default for TurnLimits {
    fn default() -> Self {
        TurnLimits { max_rounds: 8 }
    }
}

/// The instruction the model gets. Agent-facing text: it names the devices
/// and how the tools behave, nothing about the platform.
pub fn instructions(devices: &Devices) -> String {
    let mut s = String::from("You operate the following hardware through its tools:\n");
    for (name, title) in devices.names() {
        s.push_str(&format!("- {name}: {title}\n"));
    }
    s.push_str(
        "Every value is in the unit the tool states. A write outside a limit is refused with the reason and nothing is \
         written; do not repeat a refused request unchanged, explain the limit instead. A result may report that a value was \
         rounded to the field's resolution: tell the person the applied value. Some writes need a person's confirmation; this \
         gateway does not take confirmations in chat, so say the change needs to be confirmed at the device. Read before you \
         change anything you are unsure about, and never guess a register's state.",
    );
    s
}

/// Runs one turn and sends its frames to `out`. Never returns an error: a
/// failure becomes a `response.error` on the turn's session, which is what the
/// platform shows the person.
pub async fn run_turn(
    devices: Arc<Devices>,
    model: Arc<dyn ModelBackend>,
    input: InputText,
    out: UnboundedSender<Value>,
    cancel: Arc<AtomicBool>,
    limits: TurnLimits,
) {
    let sid = input.session_id.clone();
    let rid = format!("resp_{}", uamp::event_id());
    let mut messages = vec![json!({ "role": "system", "content": instructions(&devices) })];
    if input.messages.is_empty() {
        messages.push(json!({ "role": "user", "content": input.text }));
    } else {
        messages.extend(input.messages.iter().cloned());
    }
    let tools = devices.tools();

    for _round in 0..limits.max_rounds {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let req = ModelRequest { messages: messages.clone(), tools: tools.clone(), payment_token: input.payment_token.clone() };
        let reply = match model.complete(req).await {
            Ok(r) => r,
            Err(e) => {
                let _ = out.send(uamp::response_error(&sid, &rid, "model_error", &e.to_string()));
                return;
            }
        };
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        if !reply.text.is_empty() {
            let _ = out.send(uamp::delta_text(&sid, &rid, &reply.text));
        }
        if reply.tool_calls.is_empty() {
            let _ = out.send(uamp::response_done(&sid, &rid));
            return;
        }
        let calls: Vec<Value> =
            reply.tool_calls.iter().map(|c| json!({ "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.arguments } })).collect();
        messages
            .push(json!({ "role": "assistant", "content": if reply.text.is_empty() { Value::Null } else { reply.text.clone().into() }, "tool_calls": calls }));
        for c in &reply.tool_calls {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let _ = out.send(uamp::delta_tool_call(&sid, &rid, &c.id, &c.name, &c.arguments));
            // Bus I/O blocks (a serial or CAN round trip), so it runs off the async threads.
            let (d, name, args) = (devices.clone(), c.name.clone(), c.arguments.clone());
            let ctx = CallCtx { now_ms: monotonic_ms(), confirmed: false };
            let outcome = match tokio::task::spawn_blocking(move || d.call(&name, &args, &ctx)).await {
                Ok(o) => o,
                Err(e) => super::device::ToolOutcome {
                    json: format!("{{\"ok\":false,\"error\":\"gateway_error\",\"message\":{}}}", Value::String(e.to_string())),
                    is_error: true,
                },
            };
            let _ = out.send(uamp::delta_tool_result(&sid, &rid, &c.id, &outcome.json, outcome.is_error));
            messages.push(json!({ "role": "tool", "tool_call_id": c.id, "content": outcome.json }));
        }
    }
    let _ = out.send(uamp::delta_text(&sid, &rid, &format!("Stopped after {} rounds of tool calls without a final answer.", limits.max_rounds)));
    let _ = out.send(uamp::response_done(&sid, &rid));
}

/// Milliseconds on a monotonic clock, for the engine's rate limits.
pub fn monotonic_ms() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}
