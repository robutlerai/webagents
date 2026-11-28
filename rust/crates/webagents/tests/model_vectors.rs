//! The accelerator host side against vectors made by Hugging Face's own libraries
//! (`tests/fixtures/qwen3-vectors.json`, see its `about`): the Qwen3 chat
//! template rendered byte for byte, and the tokenizer's ids for every text,
//! including whole rendered prompts.
//!
//! The tokenizer file itself (Apache-2.0, 11 MB) is not vendored. Point
//! `WEBAGENTS_QWEN3_TOKENIZER` at `Qwen/Qwen3-0.6B`'s `tokenizer.json` (the
//! revision in the vectors) to run the id checks; without it they say so and
//! pass, and the template checks still run.
#![cfg(all(feature = "model", feature = "auth"))]

use serde_json::Value;
use webagents::model::{bpe::Tokenizer, chat};

fn vectors() -> Value {
    serde_json::from_str(include_str!("fixtures/qwen3-vectors.json")).unwrap()
}

#[test]
fn chat_template_renders_byte_for_byte() {
    let v = vectors();
    let tools: Vec<Value> = v["tools"].as_array().unwrap().clone();
    for c in v["chats"].as_array().unwrap() {
        let msgs: Vec<Value> = c["messages"].as_array().unwrap().clone();
        let t: &[Value] = if c["with_tools"] == true { &tools } else { &[] };
        let got = chat::render_qwen3(&msgs, t);
        assert_eq!(got, c["rendered"].as_str().unwrap(), "chat {}", c["name"]);
    }
}

#[test]
fn tokenizer_ids_byte_for_byte() {
    let Ok(path) = std::env::var("WEBAGENTS_QWEN3_TOKENIZER") else {
        eprintln!("WEBAGENTS_QWEN3_TOKENIZER is not set: the tokenizer id vectors were not checked");
        return;
    };
    let v = vectors();
    let bytes = std::fs::read(&path).unwrap();
    let sha = {
        use sha2::{Digest, Sha256};
        Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>()
    };
    assert_eq!(sha, v["tokenizer_json_sha256"].as_str().unwrap(), "a different tokenizer.json from the one the vectors were made with");
    let t = Tokenizer::from_json(std::str::from_utf8(&bytes).unwrap()).unwrap();
    for c in v["tokens"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = t.encode(text);
        assert_eq!(got, want, "ids of {text:?}");
        assert_eq!(t.decode(&got), c["decoded"].as_str().unwrap(), "decode of {text:?}");
    }
}
