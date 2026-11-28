//! The model, behind one trait, with the platform's `/llm` socket as the first
//! backend.
//!
//! The gateway runs the tool loop itself (the tools touch hardware it owns), so
//! a backend only answers "given these messages and tools, what next?": text,
//! tool calls, or both. Other backends (a model on the device's own
//! accelerator) implement the same trait.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

use super::GatewayError;
use crate::uamp;

/// One request to the model.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// The conversation so far, OpenAI-style: `{role, content}` objects, with
    /// the gateway's instructions as the first `system` message, assistant
    /// messages carrying `tool_calls`, and `tool` messages carrying results.
    pub messages: Vec<Value>,
    /// The function tools on offer ([`Devices::tools`](super::Devices::tools)).
    pub tools: Vec<Value>,
    /// Pays for this call. The platform puts one on every relayed turn.
    pub payment_token: Option<String>,
}

/// One tool call the model asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// The call's id, unique within the turn; the result is matched to it.
    pub id: String,
    /// The tool's name, as in the tool list.
    pub name: String,
    /// The arguments as JSON text, exactly as the model wrote them. They are
    /// not trusted: the device's envelope checks them.
    pub arguments: String,
}

/// What a model answered to one request: text, tool calls, or both.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelReply {
    /// Text for the person; may be empty when there are tool calls.
    pub text: String,
    /// The calls to run before asking again. Empty means the turn is answered.
    pub tool_calls: Vec<ToolCall>,
}

/// The boxed future a [`ModelBackend`] returns, so the trait stays object safe
/// (`Arc<dyn ModelBackend>`) without an async-trait macro.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The model behind the gateway's tool loop.
///
/// A backend answers one question per call: given these messages and tools,
/// what next? It never runs a tool itself. The loop runs the calls it returns
/// through each device's envelope and asks again with the results, until a
/// reply has no calls or the round limit is reached.
///
/// A toy backend that scripts the model, which is also how to test a device
/// without a model: it asks for one write, then reports what the tool said.
///
/// ```
/// use std::sync::Arc;
/// use webagents::gateway::llm::BoxFuture;
/// use webagents::gateway::{Devices, GatewayError, ModelBackend, ModelReply, ModelRequest, ToolCall};
/// use webagents::regspec::{load::load_json, CallCtx, Engine};
/// use webagents::transport::sim::{RegModel, SimBus};
///
/// struct Scripted;
///
/// impl ModelBackend for Scripted {
///     fn complete(&self, req: ModelRequest) -> BoxFuture<'_, Result<ModelReply, GatewayError>> {
///         Box::pin(async move {
///             // After a tool result, answer in text; before one, ask for the write.
///             match req.messages.iter().rev().find(|m| m["role"] == "tool") {
///                 Some(result) => Ok(ModelReply { text: format!("The part answered: {}", result["content"]), tool_calls: vec![] }),
///                 None => Ok(ModelReply {
///                     text: String::new(),
///                     tool_calls: vec![ToolCall { id: "call_1".into(), name: "pwm1_set_deadtime".into(), arguments: r#"{"deadtime": 0.2}"#.into() }],
///                 }),
///             }
///         })
///     }
/// }
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), GatewayError> {
/// // One simulated part behind the gateway.
/// let dev = load_json(include_str!("../../../../fixtures/pwm1.regspec.json")).unwrap();
/// let mut devices = Devices::new();
/// devices.add(Box::new(Engine::new(dev, SimBus(RegModel::at_reset(dev)), vec![0u64; dev.field_count]))).unwrap();
///
/// // Two rounds by hand, as the gateway's turn loop does them.
/// let model: Arc<dyn ModelBackend> = Arc::new(Scripted);
/// let mut messages = vec![serde_json::json!({ "role": "user", "content": "Set the dead time to 0.2 us." })];
/// let first = model.complete(ModelRequest { messages: messages.clone(), tools: devices.tools(), payment_token: None }).await?;
/// let call = &first.tool_calls[0];
/// let outcome = devices.call(&call.name, &call.arguments, &CallCtx::default());
/// assert!(outcome.is_error, "0.2 us is below the 0.5 us limit, so the part refuses");
/// messages.push(serde_json::json!({ "role": "tool", "tool_call_id": call.id, "content": outcome.json }));
///
/// let last = model.complete(ModelRequest { messages, tools: devices.tools(), payment_token: None }).await?;
/// assert!(last.tool_calls.is_empty() && last.text.contains("out_of_range"));
/// # Ok(())
/// # }
/// ```
pub trait ModelBackend: Send + Sync {
    /// One completion over `req`. Called once per round of a turn, possibly
    /// from several turns at once.
    ///
    /// # Errors
    /// [`GatewayError::Model`] with a sentence for the person: the turn ends
    /// with it as a `response.error`.
    fn complete(&self, req: ModelRequest) -> BoxFuture<'_, Result<ModelReply, GatewayError>>;
}

/// The platform's model socket (`wss://<portal>/llm`).
///
/// One short-lived connection per completion: `session.create` with the
/// turn's payment token, one `response.create`, then deltas until
/// `response.done`. A request without a payment token fails before
/// connecting, because nothing would pay for the call.
#[derive(Debug, Clone)]
pub struct PortalLlm {
    /// The model socket, `wss://host/llm`.
    pub url: String,
    /// `None` lets the platform choose.
    pub model: Option<String>,
    /// The most tokens one completion may generate (default 4096).
    pub max_tokens: u32,
    /// Longest wait for the socket to open (default 15 s).
    pub connect_timeout: Duration,
    /// Longest silence between frames while a completion streams.
    pub idle_timeout: Duration,
}

impl PortalLlm {
    /// The platform's model socket at `url`, with the defaults above and the
    /// model left to the platform.
    pub fn new(url: impl Into<String>) -> Self {
        PortalLlm { url: url.into(), model: None, max_tokens: 4096, connect_timeout: Duration::from_secs(15), idle_timeout: Duration::from_secs(120) }
    }
}

impl ModelBackend for PortalLlm {
    fn complete(&self, req: ModelRequest) -> BoxFuture<'_, Result<ModelReply, GatewayError>> {
        Box::pin(self.complete_inner(req))
    }
}

impl PortalLlm {
    async fn complete_inner(&self, req: ModelRequest) -> Result<ModelReply, GatewayError> {
        // No quiet fallback: without the turn's token nothing pays for the call,
        // and the platform would refuse it with a bare 4001 anyway.
        super::init_tls();
        let token =
            req.payment_token.as_deref().ok_or_else(|| GatewayError::Model("this turn carried no payment token, so the model cannot be called".into()))?;
        let (mut ws, _) = tokio::time::timeout(self.connect_timeout, tokio_tungstenite::connect_async(self.url.as_str()))
            .await
            .map_err(|_| GatewayError::Model(format!("no answer from {} within {:?}", self.url, self.connect_timeout)))?
            .map_err(|e| GatewayError::Model(format!("cannot open {}: {e}", self.url)))?;
        let send = |v: Value| Message::Text(v.to_string().into());
        ws.send(send(uamp::llm_session_create(token, self.model.as_deref()))).await.map_err(|e| GatewayError::Model(e.to_string()))?;

        let mut created = false;
        let mut text = String::new();
        let mut streamed_calls: Vec<ToolCall> = Vec::new();
        loop {
            let next = tokio::time::timeout(self.idle_timeout, ws.next())
                .await
                .map_err(|_| GatewayError::Model(format!("the model socket was silent for {:?}", self.idle_timeout)))?;
            let msg = match next {
                None => return Err(GatewayError::Model("the model socket closed before the answer was complete".into())),
                Some(Err(e)) => return Err(GatewayError::Model(e.to_string())),
                Some(Ok(m)) => m,
            };
            let raw = match msg {
                Message::Text(t) => t.to_string(),
                Message::Close(c) => {
                    let why = c.map(|c| format!("{} {}", u16::from(c.code), c.reason)).unwrap_or_default();
                    return Err(GatewayError::Model(format!("the model socket closed: {why}")));
                }
                _ => continue,
            };
            let Ok(v) = serde_json::from_str::<Value>(&raw) else { continue };
            match v.get("type").and_then(Value::as_str).unwrap_or("") {
                "session.created" if !created => {
                    created = true;
                    let rc = uamp::llm_response_create(&req.messages, &req.tools, self.model.as_deref(), self.max_tokens);
                    ws.send(send(rc)).await.map_err(|e| GatewayError::Model(e.to_string()))?;
                }
                "response.delta" => {
                    let d = &v["delta"];
                    if let Some(t) = d.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                    if let Some(tc) = d.get("tool_call") {
                        if let Some(call) = tool_call_of(tc) {
                            // A call may stream twice (start, then final arguments): keep the last.
                            streamed_calls.retain(|c| c.id != call.id);
                            streamed_calls.push(call);
                        }
                    }
                }
                "tool.call" => {
                    if let (Some(id), Some(name)) = (v.get("call_id").and_then(Value::as_str), v.get("name").and_then(Value::as_str)) {
                        let args = v.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        streamed_calls.retain(|c| c.id != id);
                        streamed_calls.push(ToolCall { id: id.into(), name: name.into(), arguments: args.into() });
                    }
                }
                "response.done" => {
                    let _ = ws.close(None).await;
                    let out = v["response"]["output"].as_array().cloned().unwrap_or_default();
                    // The final output is authoritative when it has content.
                    let mut reply = ModelReply::default();
                    for item in &out {
                        match item.get("type").and_then(Value::as_str) {
                            Some("text") => reply.text.push_str(item.get("text").and_then(Value::as_str).unwrap_or("")),
                            Some("tool_call") => reply.tool_calls.extend(item.get("tool_call").and_then(tool_call_of)),
                            _ => {}
                        }
                    }
                    if reply.text.is_empty() {
                        reply.text = text;
                    }
                    if reply.tool_calls.is_empty() {
                        reply.tool_calls = streamed_calls;
                    }
                    if v["response"]["status"].as_str() == Some("failed") && reply.text.is_empty() && reply.tool_calls.is_empty() {
                        return Err(GatewayError::Model("the model reported a failed response".into()));
                    }
                    return Ok(reply);
                }
                "response.error" | "error" => {
                    let e = &v["error"];
                    return Err(GatewayError::Model(format!(
                        "{}: {}",
                        e.get("code").and_then(Value::as_str).unwrap_or("error"),
                        e.get("message").and_then(Value::as_str).unwrap_or("the model call failed")
                    )));
                }
                "payment.required" => {
                    return Err(GatewayError::Model("the model call needs more credit than this turn's token holds".into()));
                }
                _ => {}
            }
        }
    }
}

fn tool_call_of(v: &Value) -> Option<ToolCall> {
    let id = v.get("id").and_then(Value::as_str)?.to_string();
    let name = v.get("name").and_then(Value::as_str)?.to_string();
    let arguments = match v.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(other) if !other.is_null() => other.to_string(),
        _ => "{}".into(),
    };
    Some(ToolCall { id, name, arguments })
}
