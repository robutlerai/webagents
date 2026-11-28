//! A stand-in for an AI accelerator: it serves the token link that the
//! gateway's accelerator backend (`[llm] kind = "accelerator"`) connects to,
//! with a scripted "model" in place of a neural network.
//!
//! ```sh
//! cargo run -p webagents --features model --example accel_stand_in -- 127.0.0.1:7100 rust/examples/accelerator/tokenizer.json
//! ```
//!
//! It is for trying the whole path on one machine, and for seeing what
//! crosses the link: prompt token ids in, generated token ids out, nothing
//! else. The script stands where the model would be:
//!
//! - a prompt whose last message is a person's request with a number in it is
//!   answered with a `<tool_call>` that asks for that dead time
//!   (`pwm1_set_deadtime`), and one without a number with a status read;
//! - a prompt that ends in a tool response is answered in text, with what the
//!   part said.
//!
//! The decision is made on the decoded prompt only because this is a script. A
//! real accelerator never sees text: the tokenizer and the chat template stay
//! on the gateway, which is the point of the split.
//!
//! `rust/examples/accelerator/README.md` runs it together with the gateway and
//! a mock of the platform.

use std::net::TcpListener;

use webagents::model::accel::serve_one;
use webagents::model::bpe::Tokenizer;

/// The text between the last `open` and the next `close` after it.
fn last_between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.rfind(open)? + open.len();
    let end = text[start..].find(close)? + start;
    Some(&text[start..end])
}

/// The first number in `text`, as written (`0.3` in "set it to 0.3 us").
fn first_number(text: &str) -> Option<&str> {
    let start = text.find(|c: char| c.is_ascii_digit())?;
    let len = text[start..].find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(text.len() - start);
    Some(text[start..start + len].trim_end_matches('.'))
}

/// The scripted model: the completion for a rendered prompt.
fn script(prompt: &str) -> String {
    // The conversation ends with whatever came last before the assistant's turn.
    let last_user = last_between(prompt, "<|im_start|>user\n", "<|im_end|>").unwrap_or("");
    if let Some(response) = last_between(last_user, "<tool_response>\n", "\n</tool_response>") {
        // A tool result came back: report it. A refusal carries its reason in `message`.
        return match serde_json::from_str::<serde_json::Value>(response) {
            Ok(v) if v["ok"] == false => format!("The part refused: {}", v["message"].as_str().unwrap_or("no reason given")),
            Ok(v) => format!("Done. The part reports {}.", v.get("applied").or(v.get("fields")).unwrap_or(&v)),
            Err(_) => "The part's answer could not be read.".to_string(),
        };
    }
    match first_number(last_user) {
        Some(n) => format!("<tool_call>\n{{\"name\": \"pwm1_set_deadtime\", \"arguments\": {{\"deadtime\": {n}}}}}\n</tool_call>"),
        None => "<tool_call>\n{\"name\": \"pwm1_read_status\", \"arguments\": {}}\n</tool_call>".to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, addr, tokenizer] = args.as_slice() else {
        eprintln!("usage: accel_stand_in <listen host:port> <tokenizer.json>");
        std::process::exit(2);
    };
    // The same tokenizer file the gateway is configured with: ids only mean
    // something to two sides that agree on it.
    let tokenizer = Tokenizer::from_file(std::path::Path::new(tokenizer))?;
    let listener = TcpListener::bind(addr)?;
    println!("accelerator stand-in on {addr}");

    // The gateway opens one connection per model call.
    for link in listener.incoming() {
        let mut link = link?;
        let served = serve_one(&mut link, &mut |prompt, max_new, stops, emit| {
            let reply = script(&tokenizer.decode(prompt));
            let ids = tokenizer.encode(&reply);
            println!("{} prompt tokens in, {} tokens out: {}", prompt.len(), ids.len() + 1, reply.replace('\n', " "));
            // One token at a time, as an accelerator produces them, within the budget.
            let budget = max_new as usize;
            for id in ids.iter().take(budget.saturating_sub(1)) {
                emit(*id)?;
            }
            if ids.len() < budget {
                // The stop token ends the answer; `true` reports that it did.
                emit(stops[0])?;
                Ok(true)
            } else {
                Ok(false)
            }
        });
        if let Err(e) = served {
            eprintln!("link: {e}");
        }
    }
    Ok(())
}
