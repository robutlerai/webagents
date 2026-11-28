//! A byte-level BPE tokenizer read from a Hugging Face `tokenizer.json` (the
//! GPT-2 / Qwen2 / Qwen3 family), in plain Rust so it cross-builds to any Linux
//! host the gateway runs on (no C library, no Python).
//!
//! The pipeline is the file's, in its order:
//! 1. added tokens (`<|im_start|>`, `<tool_call>`, ...) are matched literally,
//!    longest first, and become their own ids;
//! 2. the rest is NFC-normalized when the file says so;
//! 3. the `Split` pre-tokenizer's regex cuts it into words (the pattern is
//!    taken from the file, compiled with lookaround support);
//! 4. each word's UTF-8 bytes are mapped to the byte-level alphabet and merged
//!    by rank until no merge applies.
//!
//! Decoding reverses the byte map. Only what this family uses is accepted: any
//! other normalizer, pre-tokenizer or model type is refused by name at load,
//! never approximated.

use std::collections::HashMap;

use fancy_regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// Why a `tokenizer.json` was not accepted: unreadable, not JSON, or using a
/// model, normalizer or pre-tokenizer this tokenizer does not implement (named
/// in the message).
#[derive(Debug)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TokenizerError {}

fn err(m: impl Into<String>) -> TokenizerError {
    TokenizerError(m.into())
}

/// A loaded tokenizer. Immutable after loading, so one instance can be shared
/// between threads.
///
/// ```
/// use webagents::model::bpe::Tokenizer;
///
/// // A three-entry vocabulary with one merge and one added token. A model's
/// // own file has tens of thousands of entries and the same structure.
/// let tokenizer = Tokenizer::from_json(
///     r#"{ "model": { "type": "BPE", "vocab": { "h": 0, "i": 1, "hi": 2 }, "merges": ["h i"] },
///          "pre_tokenizer": { "type": "ByteLevel" },
///          "added_tokens": [ { "id": 3, "content": "<|im_end|>" } ] }"#,
/// )
/// .unwrap();
///
/// assert_eq!(tokenizer.encode("hi<|im_end|>"), [2, 3]);
/// assert_eq!(tokenizer.decode(&[2, 3]), "hi<|im_end|>");
/// assert_eq!(tokenizer.special("<|im_end|>"), Some(3));
/// ```
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    ranks: HashMap<(String, String), u32>,
    /// Added tokens, longest content first (so `<tool_call>` never loses to a prefix).
    added: Vec<(String, u32)>,
    added_ids: HashMap<u32, String>,
    split: Regex,
    nfc: bool,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("vocab", &self.vocab.len()).field("merges", &self.ranks.len()).field("added", &self.added.len()).finish()
    }
}

/// GPT-2's map from bytes to printable characters: the printable Latin-1 bytes
/// map to themselves, the other 68 to U+0100 onwards in byte order.
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

impl Tokenizer {
    /// Loads a Hugging Face `tokenizer.json` from `path`.
    pub fn from_file(path: &std::path::Path) -> Result<Self, TokenizerError> {
        let s = std::fs::read_to_string(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json(&s)
    }

    /// Loads a tokenizer from the text of a `tokenizer.json`. Accepted: a `BPE`
    /// model, no normalizer or `NFC`, and a pre-tokenizer that is either
    /// `ByteLevel` with its regex or a `Sequence` of an isolated `Split` and a
    /// `ByteLevel`.
    pub fn from_json(s: &str) -> Result<Self, TokenizerError> {
        let j: Value = serde_json::from_str(s).map_err(|e| err(format!("tokenizer.json: {e}")))?;
        let model = &j["model"];
        if model["type"] != "BPE" {
            return Err(err(format!("model type {} is not BPE", model["type"])));
        }
        let vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .ok_or_else(|| err("model.vocab is not an object"))?
            .iter()
            .map(|(k, v)| v.as_u64().map(|i| (k.clone(), i as u32)).ok_or_else(|| err(format!("vocab entry {k}"))))
            .collect::<Result<_, _>>()?;
        let mut ranks = HashMap::new();
        for (i, m) in model["merges"].as_array().ok_or_else(|| err("model.merges is not an array"))?.iter().enumerate() {
            // Two spellings exist: "a b" (older files) and ["a", "b"].
            let (a, b) = match m {
                Value::String(s) => {
                    let (a, b) = s.split_once(' ').ok_or_else(|| err(format!("merge {s:?}")))?;
                    (a.to_string(), b.to_string())
                }
                Value::Array(p) if p.len() == 2 => (p[0].as_str().unwrap_or_default().to_string(), p[1].as_str().unwrap_or_default().to_string()),
                _ => return Err(err(format!("merge {m}"))),
            };
            ranks.entry((a, b)).or_insert(i as u32);
        }
        let nfc = match &j["normalizer"] {
            Value::Null => false,
            n if n["type"] == "NFC" => true,
            n => return Err(err(format!("normalizer {} is not supported", n["type"]))),
        };
        let split = pre_tokenizer_regex(&j["pre_tokenizer"])?;
        let mut added: Vec<(String, u32)> = j["added_tokens"]
            .as_array()
            .map(|a| a.iter().filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32))).collect())
            .unwrap_or_default();
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let added_ids = added.iter().map(|(s, i)| (*i, s.clone())).collect();
        let byte_to_char = byte_alphabet();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let id_to_token = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Ok(Tokenizer { vocab, id_to_token, ranks, added, added_ids, split, nfc, byte_to_char, char_to_byte })
    }

    /// The id of an added token, such as `<|im_end|>`.
    pub fn special(&self, content: &str) -> Option<u32> {
        self.added.iter().find(|(s, _)| s == content).map(|(_, i)| *i)
    }

    /// Token ids of `text`. Added tokens (`<|im_start|>`) are matched literally
    /// wherever they occur and become their own ids; no token is added at the
    /// start or the end.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest added token, longest at that position.
            let hit = self.added.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id))).min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            match hit {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut ids);
                    ids.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.encode_plain(rest, &mut ids);
                    break;
                }
            }
        }
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let norm: String = if self.nfc { text.nfc().collect() } else { text.to_string() };
        let mut last = 0;
        for m in self.split.find_iter(&norm).flatten() {
            // Isolated behaviour: text between matches is its own piece too.
            if m.start() > last {
                self.bpe_word(&norm[last..m.start()], ids);
            }
            self.bpe_word(m.as_str(), ids);
            last = m.end();
        }
        if last < norm.len() {
            self.bpe_word(&norm[last..], ids);
        }
    }

    fn bpe_word(&self, word: &str, ids: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.bytes().map(|b| self.byte_to_char[b as usize].to_string()).collect();
        loop {
            let best = (0..parts.len().saturating_sub(1)).filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|r| (*r, i))).min();
            let Some((rank, _)) = best else { break };
            // Merge every non-overlapping occurrence of the best pair, left to right.
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) == Some(&rank) {
                    merged.push(format!("{}{}", parts[i], parts[i + 1]));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut parts[i]));
                    i += 1;
                }
            }
            parts = merged;
        }
        for p in parts {
            match self.vocab.get(&p) {
                Some(id) => ids.push(*id),
                // Every single byte is in a byte-level vocabulary; a gap is a broken file.
                None => ids.extend(p.chars().filter_map(|c| self.vocab.get(&c.to_string()).copied())),
            }
        }
    }

    /// Text of `ids`, added tokens included as their content. Bytes that do not
    /// form valid UTF-8 (a token boundary inside a character) become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(s) = self.added_ids.get(id) {
                bytes.extend_from_slice(s.as_bytes());
            } else if let Some(t) = self.id_to_token.get(id) {
                bytes.extend(t.chars().filter_map(|c| self.char_to_byte.get(&c)));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// The word-splitting regex from the file's pre-tokenizer. Accepted: a
/// `ByteLevel` with `use_regex` (GPT-2's pattern), or a `Sequence` of one
/// `Split` (Isolated) and a `ByteLevel` without its own regex (Qwen2/3).
fn pre_tokenizer_regex(p: &Value) -> Result<Regex, TokenizerError> {
    const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let compile = |s: &str| Regex::new(s).map_err(|e| err(format!("pre-tokenizer regex: {e}")));
    match p["type"].as_str() {
        Some("ByteLevel") if p["use_regex"] != false => compile(GPT2),
        Some("Sequence") => {
            let ps = p["pretokenizers"].as_array().ok_or_else(|| err("Sequence without pretokenizers"))?;
            let [split, bl] = ps.as_slice() else { return Err(err("a pre-tokenizer Sequence other than Split + ByteLevel")) };
            if split["type"] != "Split" || split["behavior"] != "Isolated" || split["invert"] == true {
                return Err(err(format!("pre-tokenizer {}", split["type"])));
            }
            if bl["type"] != "ByteLevel" || bl["use_regex"] == true || bl["add_prefix_space"] == true {
                return Err(err("a ByteLevel step with its own regex or prefix space"));
            }
            let pat = split["pattern"]["Regex"].as_str().ok_or_else(|| err("Split without a Regex pattern"))?;
            compile(pat)
        }
        t => Err(err(format!("pre-tokenizer {t:?} is not supported"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic byte-level BPE in the Qwen layout (written for this test).
    pub(crate) fn tiny() -> Tokenizer {
        let alpha = byte_alphabet();
        let mut vocab = serde_json::Map::new();
        for (i, c) in alpha.iter().enumerate() {
            vocab.insert(c.to_string(), (i as u64).into());
        }
        for (i, t) in ["ab", "abc", "\u{120}a", "\u{120}ab"].iter().enumerate() {
            vocab.insert(t.to_string(), (256 + i as u64).into());
        }
        let j = serde_json::json!({
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
            "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["ab", "c"], ["\u{120}", "a"], ["\u{120}a", "b"]]},
            "added_tokens": [{"id": 300, "content": "<x>"}, {"id": 301, "content": "<xy>"}]
        });
        Tokenizer::from_json(&j.to_string()).unwrap()
    }

    #[test]
    fn merges_by_rank_and_added_tokens_win() {
        let t = tiny();
        // " ab": the pair (a, b) outranks (Ġ, a), and (Ġ, ab) is no merge, so it stays two tokens.
        assert_eq!(t.encode("abc ab"), vec![257, 32, 256]);
        assert_eq!(t.encode(" a"), vec![258]);
        assert_eq!(t.encode("<xy>ab<x>"), vec![301, 256, 300]);
        assert_eq!(t.decode(&t.encode("abc <x> é 12")), "abc <x> é 12");
        // NFC: a decomposed e + combining acute encodes as the composed character.
        assert_eq!(t.encode("e\u{301}"), t.encode("\u{e9}"));
    }
}
