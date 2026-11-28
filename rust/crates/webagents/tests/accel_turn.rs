//! A whole tool-loop turn on the accelerator backend, with a scripted accelerator: the
//! "chip" answers the first call with the ids of a `<tool_call>` that asks for
//! a dead time below the part's limit, and the second (which must carry the
//! refusal as a tool response) with a sentence. Checked: the call is parsed
//! from token ids, refused by the device envelope with the limit, the register
//! keeps its value, the refusal reaches the model in the next prompt, the
//! person gets the model's text, and the chip saw prompts cut to its port.
//!
//! The tokenizer is a byte-level one written for this test (the Qwen3 one is
//! checked against Hugging Face in `model_vectors.rs`).
#![cfg(feature = "gateway")]

use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use webagents::gateway::accel::AccelBackend;
use webagents::gateway::turn::{run_turn, TurnLimits};
use webagents::gateway::Devices;
use webagents::model::accel::{serve_one, PROMPT_MAX};
use webagents::model::bpe::Tokenizer;
use webagents::regspec::{load::load_json, Engine};
use webagents::transport::sim::{RegModel, SimBus};
use webagents::uamp::InputText;

const SPECIALS: [&str; 7] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<think>", "</think>"];

fn byte_tokenizer() -> String {
    // GPT-2's byte alphabet as the whole vocabulary, no merges, the Qwen specials as added tokens.
    let mut vocab = serde_json::Map::new();
    let mut n = 0u32;
    for b in 0..=255u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
        vocab.insert(c.to_string(), b.into());
    }
    let added: Vec<Value> = SPECIALS.iter().enumerate().map(|(i, s)| json!({"id": 256 + i, "content": s})).collect();
    json!({
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": added
    })
    .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_call_from_the_accelerator_leaves_the_register_alone() {
    let tj = byte_tokenizer();
    let tk = Arc::new(Tokenizer::from_json(&tj).unwrap());
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let link = l.local_addr().unwrap().to_string();
    let prompts: Arc<Mutex<Vec<String>>> = Arc::default();
    let (tk2, p2) = (tk.clone(), prompts.clone());
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { return };
            let _ = serve_one(&mut s, &mut |prompt, _max_new, stops, emit| {
                let text = tk2.decode(prompt);
                p2.lock().unwrap().push(text.clone());
                let reply = if text.contains("<tool_response>") {
                    "The dead time cannot go below 0.5 microseconds, so nothing was changed."
                } else {
                    "<tool_call>\n{\"name\": \"pwm1_set_deadtime\", \"arguments\": {\"deadtime\": 0.3}}\n</tool_call>"
                };
                for id in tk2.encode(reply) {
                    emit(id)?;
                }
                emit(stops[0])?;
                Ok(true)
            });
        }
    });

    let dev = load_json(include_str!("../../../fixtures/pwm1.regspec.json")).unwrap();
    let regs = RegModel::at_reset(dev);
    let deadtime = dev.registers.iter().find(|r| r.name == "DEADTIME").unwrap().offset;
    let before = regs.lock().unwrap().get(deadtime);
    let mut devices = Devices::new();
    devices.add(Box::new(Engine::new(dev, SimBus(regs.clone()), vec![0u64; dev.field_count]))).unwrap();
    let model = Arc::new(AccelBackend::new(link, Tokenizer::from_json(&tj).unwrap(), 256).unwrap());

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let input = InputText { session_id: "s1".into(), text: "Set the dead time to 0.3 us.".into(), messages: vec![], payment_token: None, caller: None };
    run_turn(Arc::new(devices), model, input, tx, Arc::new(AtomicBool::new(false)), TurnLimits::default()).await;

    let mut frames = Vec::new();
    while let Ok(f) = rx.try_recv() {
        frames.push(f);
    }
    let kinds: Vec<String> = frames.iter().map(|f| f["delta"]["type"].as_str().or(f["type"].as_str()).unwrap_or("?").to_string()).collect();
    assert_eq!(kinds, ["tool_call", "tool_result", "text", "response.done"], "{frames:#?}");
    let call = &frames[0]["delta"]["tool_call"];
    assert_eq!(call["name"], "pwm1_set_deadtime");
    let result = &frames[1]["delta"]["tool_result"];
    assert_eq!(result["status"], "error", "{result}");
    assert!(result["result"].as_str().unwrap().contains("out_of_range"), "{result}");
    assert!(frames[2]["delta"]["text"].as_str().unwrap().contains("0.5 microseconds"));
    assert_eq!(regs.lock().unwrap().get(deadtime), before, "the refused write reached the register");
    assert_eq!(regs.lock().unwrap().writes, 0);

    let p = prompts.lock().unwrap();
    assert_eq!(p.len(), 2);
    assert!(p[0].contains("<tools>") && p[0].contains("\"pwm1_set_deadtime\"") && p[0].ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    assert!(p[1].contains("<tool_response>\n{") && p[1].contains("out_of_range"), "{}", p[1]);
    assert!(tk.encode(&p[0]).len() > PROMPT_MAX, "the prompt spans several prompt events");
}
