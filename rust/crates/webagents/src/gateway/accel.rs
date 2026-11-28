//! The model on the board's own accelerator, as a [`ModelBackend`].
//!
//! One model call: the tool loop's messages and tools are rendered with the
//! model's chat template, tokenized, sent over the accelerator link, and the
//! generated ids are decoded and parsed into text and tool calls. No platform
//! model call is made and no payment token is used: the turn's compute is the
//! board's. The tool calls then run through the same device envelopes as with
//! the platform's model, so every limit and refusal is identical.

use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::llm::{BoxFuture, ModelBackend, ModelReply, ModelRequest, ToolCall};
use super::GatewayError;
use crate::model::{accel, bpe::Tokenizer, chat};

/// A [`ModelBackend`] that runs each completion on the accelerator at `link`.
///
/// Calls are serialized (the accelerator holds one sequence at a time), each
/// on a fresh TCP connection to the link. A generation that reaches `max_new`
/// without a stop token is returned with a `[stopped at N tokens]` note
/// appended to its text, so a cut-off answer is never taken for a whole one.
pub struct AccelBackend {
    link: String,
    tokenizer: Arc<Tokenizer>,
    stops: Vec<u32>,
    max_new: u32,
    /// One sequence at a time: the accelerator holds one KV cache.
    busy: Arc<Mutex<()>>,
    /// Longest wait for the next token (a software model of the accelerator is slow).
    pub token_timeout: Duration,
}

impl std::fmt::Debug for AccelBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccelBackend").field("link", &self.link).field("stops", &self.stops).field("max_new", &self.max_new).finish()
    }
}

impl AccelBackend {
    /// A backend for the accelerator link at `link` (`host:port`), with the
    /// model's own tokenizer and at most `max_new` generated tokens per call.
    /// Nothing is connected until the first call.
    ///
    /// # Errors
    /// [`GatewayError::Config`] when the tokenizer has neither `<|im_end|>`
    /// nor `<|endoftext|>`: without a stop token every answer would run to
    /// `max_new`.
    pub fn new(link: String, tokenizer: Tokenizer, max_new: u32) -> Result<Self, GatewayError> {
        let stops: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tokenizer.special(s)).collect();
        if stops.is_empty() {
            return Err(GatewayError::Config("the tokenizer has no <|im_end|> or <|endoftext|> token to stop on".into()));
        }
        Ok(AccelBackend { link, tokenizer: Arc::new(tokenizer), stops, max_new, busy: Arc::new(Mutex::new(())), token_timeout: Duration::from_secs(600) })
    }

    /// The whole call, blocking (it runs off the async threads).
    ///
    /// # Errors
    /// [`GatewayError::Model`] when the link cannot be opened, no token
    /// arrives within [`AccelBackend::token_timeout`], or the accelerator
    /// refuses or breaks the link protocol.
    pub fn complete_blocking(&self, req: &ModelRequest) -> Result<ModelReply, GatewayError> {
        let prompt = chat::render_qwen3(&req.messages, &req.tools);
        let ids = self.tokenizer.encode(&prompt);
        let _one = self.busy.lock().map_err(|_| GatewayError::Model("accelerator lock poisoned".into()))?;
        let mut s = TcpStream::connect(&self.link).map_err(|e| GatewayError::Model(format!("accelerator link {}: {e}", self.link)))?;
        s.set_read_timeout(Some(self.token_timeout)).map_err(|e| GatewayError::Model(e.to_string()))?;
        let _ = s.set_nodelay(true);
        let g = accel::generate(&mut s, &ids, self.max_new, &self.stops, |_| {}).map_err(|e| GatewayError::Model(e.to_string()))?;
        let body: Vec<u32> = g.tokens.iter().copied().filter(|t| !self.stops.contains(t)).collect();
        let completion = self.tokenizer.decode(&body);
        let (text, calls) = chat::parse_tool_calls(&completion);
        let text = if g.stopped { text } else { format!("{text}\n[stopped at {} tokens]", self.max_new).trim().to_string() };
        eprintln!("[webagents-gateway] accelerator: {} prompt tokens, {} generated, {} tool call(s)", ids.len(), g.tokens.len(), calls.len());
        Ok(ModelReply {
            text,
            tool_calls: calls
                .into_iter()
                .enumerate()
                .map(|(i, c)| ToolCall { id: format!("call_accel_{}_{i}", crate::uamp::event_id()), name: c.name, arguments: c.arguments })
                .collect(),
        })
    }
}

impl ModelBackend for AccelBackend {
    fn complete(&self, req: ModelRequest) -> BoxFuture<'_, Result<ModelReply, GatewayError>> {
        let me = AccelBackend {
            link: self.link.clone(),
            tokenizer: self.tokenizer.clone(),
            stops: self.stops.clone(),
            max_new: self.max_new,
            busy: self.busy.clone(),
            token_timeout: self.token_timeout,
        };
        Box::pin(async move { tokio::task::spawn_blocking(move || me.complete_blocking(&req)).await.map_err(|e| GatewayError::Model(e.to_string()))? })
    }
}
