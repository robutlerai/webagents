//! The Qwen3 chat template with tools, rendered in Rust exactly as Hugging Face
//! `apply_chat_template(messages, tools=..., add_generation_prompt=True,
//! enable_thinking=False)` renders the model's own Jinja template, and the
//! parse of the `<tool_call>` blocks the model answers with.
//!
//! A local model on an accelerator sees only token ids, so the host must build
//! the prompt itself; a prompt that differs from the one the model was trained
//! on by one byte (a separator in the tools JSON, a newline) degrades tool use
//! silently. The rendering is therefore pinned to vectors produced by
//! `transformers` itself, byte for byte.
//!
//! The tools are printed with Jinja's `tojson`, which in `transformers` is
//! Python's `json.dumps(x, ensure_ascii=False)`: `", "` and `": "`
//! separators, non-ASCII characters raw, and Python's float spelling.
//! [`write_py_json`] reproduces that.

use std::fmt::Write as _;

use serde_json::Value;

/// Python's `json.dumps(v, ensure_ascii=False)` (default separators).
pub fn write_py_json(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else {
                py_float(out, n.as_f64().unwrap_or(f64::NAN));
            }
        }
        Value::String(s) => py_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(out, x);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_str(out, k);
                out.push_str(": ");
                write_py_json(out, x);
            }
            out.push('}');
        }
    }
}

fn py_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, fixed notation for
/// decimal exponents -4 to 15, otherwise `d.ddde+XX` with at least two exponent digits.
fn py_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("NaN");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
        return;
    }
    let e = format!("{f:e}"); // shortest round-trip digits, e.g. "-1.25e-5"
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        out.push_str(&s);
        if !s.contains('.') {
            out.push_str(".0");
        }
    } else {
        out.push_str(mant);
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
}

fn content(m: &Value) -> &str {
    m["content"].as_str().unwrap_or("")
}

/// The prompt for the next assistant turn (`add_generation_prompt`, thinking off).
/// `messages` and `tools` are OpenAI-style, as the gateway's tool loop keeps them.
///
/// A leading `system` message is merged into the tools preamble; `tool`
/// messages become `<tool_response>` blocks inside a user turn; an assistant
/// message's `tool_calls` are written back as `<tool_call>` blocks. The
/// prompt always ends with the opening of the assistant's answer and an empty
/// thinking block.
///
/// ```
/// use serde_json::json;
/// use webagents::model::chat::render_qwen3;
///
/// let prompt = render_qwen3(&[json!({ "role": "user", "content": "Read the status." })], &[]);
/// assert_eq!(prompt, "<|im_start|>user\nRead the status.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
/// ```
pub fn render_qwen3(messages: &[Value], tools: &[Value]) -> String {
    let mut o = String::new();
    let first_system = messages.first().is_some_and(|m| m["role"] == "system");
    if !tools.is_empty() {
        o.push_str("<|im_start|>system\n");
        if first_system {
            o.push_str(content(&messages[0]));
            o.push_str("\n\n");
        }
        o.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for t in tools {
            o.push('\n');
            write_py_json(&mut o, t);
        }
        o.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    } else if first_system {
        let _ = write!(o, "<|im_start|>system\n{}<|im_end|>\n", content(&messages[0]));
    }
    // The last real user query (a user message that is not a wrapped tool response).
    let mut last_query = messages.len().saturating_sub(1);
    for (i, m) in messages.iter().enumerate().rev() {
        if m["role"] == "user" {
            if let Some(c) = m["content"].as_str() {
                if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                    last_query = i;
                    break;
                }
            }
        }
    }
    let n = messages.len();
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("");
        let mut c = content(m).to_string();
        if role == "user" || (role == "system" && i != 0) {
            let _ = write!(o, "<|im_start|>{role}\n{c}<|im_end|>\n");
        } else if role == "assistant" {
            let mut reasoning = m["reasoning_content"].as_str().map(str::to_string);
            if reasoning.is_none() {
                if let Some((before, after)) = c.split_once("</think>").map(|(a, _)| (a.to_string(), c.rsplit("</think>").next().unwrap_or("").to_string())) {
                    let r = before.trim_end_matches('\n');
                    let r = r.rsplit("<think>").next().unwrap_or(r).trim_start_matches('\n');
                    reasoning = Some(r.to_string());
                    c = after.trim_start_matches('\n').to_string();
                }
            }
            let reasoning = reasoning.unwrap_or_default();
            if i > last_query && (i == n - 1 || !reasoning.is_empty()) {
                let _ = write!(o, "<|im_start|>assistant\n<think>\n{}\n</think>\n\n{}", reasoning.trim_matches('\n'), c.trim_start_matches('\n'));
            } else {
                let _ = write!(o, "<|im_start|>assistant\n{c}");
            }
            if let Some(calls) = m["tool_calls"].as_array() {
                for (k, call) in calls.iter().enumerate() {
                    if (k == 0 && !c.is_empty()) || k > 0 {
                        o.push('\n');
                    }
                    let f = if call["function"].is_object() { &call["function"] } else { call };
                    let _ = write!(o, "<tool_call>\n{{\"name\": \"{}\", \"arguments\": ", f["name"].as_str().unwrap_or(""));
                    match &f["arguments"] {
                        Value::String(s) => o.push_str(s),
                        v => write_py_json(&mut o, v),
                    }
                    o.push_str("}\n</tool_call>");
                }
            }
            o.push_str("<|im_end|>\n");
        } else if role == "tool" {
            if i == 0 || messages[i - 1]["role"] != "tool" {
                o.push_str("<|im_start|>user");
            }
            let _ = write!(o, "\n<tool_response>\n{c}\n</tool_response>");
            if i == n - 1 || messages[i + 1]["role"] != "tool" {
                o.push_str("<|im_end|>\n");
            }
        }
    }
    o.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    o
}

/// One parsed call: the tool name and its arguments as a JSON object string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCall {
    /// The tool's name.
    pub name: String,
    /// The arguments as JSON text: the block's `arguments` object re-serialized,
    /// a string passed through as is, or `{}` when the model gave none.
    pub arguments: String,
}

/// Splits a completion into its text and its `<tool_call>` blocks. A block whose
/// JSON does not parse, or has no string `name`, stays in the text (the person
/// sees what the model wrote) and is not run.
///
/// ```
/// use webagents::model::chat::{parse_tool_calls, ParsedCall};
///
/// let completion = "Setting it now.\n<tool_call>\n{\"name\": \"pwm1_set_deadtime\", \"arguments\": {\"deadtime\": 1.5}}\n</tool_call>";
/// let (text, calls) = parse_tool_calls(completion);
/// assert_eq!(text, "Setting it now.");
/// assert_eq!(calls, [ParsedCall { name: "pwm1_set_deadtime".into(), arguments: r#"{"deadtime":1.5}"#.into() }]);
///
/// // Not JSON: nothing is run, and the person sees the block as written.
/// let (text, calls) = parse_tool_calls("<tool_call>set the dead time</tool_call>");
/// assert!(calls.is_empty());
/// assert_eq!(text, "<tool_call>set the dead time</tool_call>");
/// ```
pub fn parse_tool_calls(completion: &str) -> (String, Vec<ParsedCall>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut rest = completion;
    // A reply may still carry an empty think block.
    if let Some(after) = rest.trim_start().strip_prefix("<think>").and_then(|r| r.split_once("</think>")).map(|(_, a)| a) {
        rest = after;
    }
    while let Some(p) = rest.find("<tool_call>") {
        text.push_str(&rest[..p]);
        let body = &rest[p + "<tool_call>".len()..];
        let (inner, next) = match body.find("</tool_call>") {
            Some(e) => (&body[..e], &body[e + "</tool_call>".len()..]),
            None => (body, ""),
        };
        match serde_json::from_str::<Value>(inner.trim()) {
            Ok(v) if v["name"].is_string() => {
                let args = match &v["arguments"] {
                    Value::Null => "{}".to_string(),
                    Value::String(s) => s.clone(),
                    a => a.to_string(),
                };
                calls.push(ParsedCall { name: v["name"].as_str().unwrap_or_default().to_string(), arguments: args });
            }
            _ => {
                text.push_str("<tool_call>");
                text.push_str(inner);
                if next.len() < body.len() - inner.len() {
                    text.push_str("</tool_call>");
                }
            }
        }
        rest = next;
    }
    text.push_str(rest);
    (text.trim().to_string(), calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_json_matches_python_spelling() {
        let mut s = String::new();
        write_py_json(&mut s, &json!({"a": [1, 2.5, 1e-5, 1e16, 100.0, -0.0001], "b": "é\n\"x\"\u{1}", "c": null, "d": true}));
        assert_eq!(s, r#"{"a": [1, 2.5, 1e-05, 1e+16, 100.0, -0.0001], "b": "é\n\"x\"\u0001", "c": null, "d": true}"#);
    }

    #[test]
    fn tool_calls_parse_and_bad_blocks_stay_text() {
        let (t, c) = parse_tool_calls("<think>\n\n</think>\n\nOk.\n<tool_call>\n{\"name\": \"pwm1_read_ctrl\", \"arguments\": {}}\n</tool_call>");
        assert_eq!(t, "Ok.");
        assert_eq!(c, vec![ParsedCall { name: "pwm1_read_ctrl".into(), arguments: "{}".into() }]);
        let (t, c) = parse_tool_calls("<tool_call>{not json}</tool_call> done");
        assert!(c.is_empty());
        assert_eq!(t, "<tool_call>{not json}</tool_call> done");
    }
}
