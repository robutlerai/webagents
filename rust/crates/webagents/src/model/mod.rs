//! A model on the device's own accelerator, host side.
//!
//! An accelerator computes token ids from token ids. Everything textual stays
//! on the host that drives it: [`bpe`] (the model's tokenizer), [`chat`] (its
//! chat template with tools, and the parse of its tool calls) and [`accel`]
//! (the link: prompt ids out, generated ids back). The gateway's accelerator
//! backend joins them into the same `ModelBackend` the platform's `/llm` socket
//! implements, so the tool loop, the limits and the refusals are unchanged.

pub mod accel;
pub mod bpe;
pub mod chat;
