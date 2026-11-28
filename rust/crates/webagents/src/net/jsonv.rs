//! Reading values out of a JSON frame without a heap or a parse tree: find a
//! key's raw value, walk an array's raw items, unescape a string into a
//! buffer. Every function works on slices of the original text.
//!
//! A frame that is not well formed simply yields `None` where it breaks; the
//! device then treats the frame as unknown rather than guessing.
//!
//! ```
//! use webagents::net::jsonv;
//!
//! let frame = r#"{"type":"input.text","session_id":"s1","messages":[{"role":"user","content":"Set it to 1.5 \"us\""}]}"#;
//!
//! // A key's raw value, and the text inside a string value.
//! assert_eq!(jsonv::get(frame, "type"), Some(r#""input.text""#));
//! assert_eq!(jsonv::get(frame, "session_id").and_then(jsonv::as_str), Some("s1"));
//!
//! // The items of an array, each a slice of the frame.
//! let first = jsonv::items(jsonv::get(frame, "messages").unwrap()).next().unwrap();
//! let content = jsonv::get(first, "content").and_then(jsonv::as_str).unwrap();
//!
//! // String content stays escaped until it is unescaped into a buffer.
//! let mut buf = [0u8; 32];
//! assert_eq!(jsonv::unescape(content, &mut buf), Some(r#"Set it to 1.5 "us""#));
//! ```

fn ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// End (exclusive) of the value starting at `i` (after whitespace).
pub fn skip(s: &[u8], i: usize) -> Option<usize> {
    let mut i = ws(s, i);
    match *s.get(i)? {
        b'"' => {
            i += 1;
            while i < s.len() {
                match s[i] {
                    b'\\' => i += 2,
                    b'"' => return Some(i + 1),
                    _ => i += 1,
                }
            }
            None
        }
        b'{' | b'[' => {
            let mut depth = 0usize;
            while i < s.len() {
                match s[i] {
                    b'"' => {
                        i = skip(s, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            let st = i;
            while i < s.len() && !matches!(s[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                i += 1;
            }
            (i > st).then_some(i)
        }
    }
}

/// The raw value of `key` in the object `obj` (top level of that object only).
pub fn get<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let s = obj.as_bytes();
    let mut i = ws(s, 0);
    if s.get(i)? != &b'{' {
        return None;
    }
    i += 1;
    loop {
        i = ws(s, i);
        if s.get(i)? == &b'}' {
            return None;
        }
        let kend = skip(s, i)?;
        let k = obj.get(i + 1..kend - 1)?;
        i = ws(s, kend);
        if s.get(i)? != &b':' {
            return None;
        }
        let vstart = ws(s, i + 1);
        let vend = skip(s, vstart)?;
        if k == key {
            return obj.get(vstart..vend);
        }
        i = ws(s, vend);
        match s.get(i)? {
            b',' => i += 1,
            b'}' => return None,
            _ => return None,
        }
    }
}

/// `get` along a path of keys.
pub fn path<'a>(obj: &'a str, keys: &[&str]) -> Option<&'a str> {
    keys.iter().try_fold(obj, |o, k| get(o, k))
}

/// The escaped text inside a raw JSON string value (without the quotes).
pub fn as_str(raw: &str) -> Option<&str> {
    let r = raw.trim();
    (r.len() >= 2 && r.starts_with('"') && r.ends_with('"')).then(|| &r[1..r.len() - 1])
}

/// The raw items of a raw JSON array.
pub fn items(raw: &str) -> Items<'_> {
    let s = raw.as_bytes();
    let i = ws(s, 0);
    let ok = s.get(i) == Some(&b'[');
    Items { src: raw, i: if ok { i + 1 } else { s.len() }, done: !ok }
}

/// Iterator over the raw items of a JSON array; see [`items`]. Stops at the
/// first item that is not well formed.
pub struct Items<'a> {
    src: &'a str,
    i: usize,
    done: bool,
}

impl<'a> Iterator for Items<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        if self.done {
            return None;
        }
        let s = self.src.as_bytes();
        let i = ws(s, self.i);
        if s.get(i) == Some(&b']') || i >= s.len() {
            self.done = true;
            return None;
        }
        let Some(end) = skip(s, i) else {
            self.done = true;
            return None;
        };
        let j = ws(s, end);
        self.i = if s.get(j) == Some(&b',') { j + 1 } else { j };
        self.src.get(i..end)
    }
}

/// Unescapes JSON string content into `out`. `None` if it does not fit or is malformed.
pub fn unescape<'o>(esc: &str, out: &'o mut [u8]) -> Option<&'o str> {
    let b = esc.as_bytes();
    let (mut i, mut n) = (0, 0);
    let mut put = |c: &[u8], n: &mut usize| -> Option<()> {
        out.get_mut(*n..*n + c.len())?.copy_from_slice(c);
        *n += c.len();
        Some(())
    };
    while i < b.len() {
        if b[i] != b'\\' {
            put(&b[i..i + 1], &mut n)?;
            i += 1;
            continue;
        }
        let e = *b.get(i + 1)?;
        i += 2;
        match e {
            b'"' => put(b"\"", &mut n)?,
            b'\\' => put(b"\\", &mut n)?,
            b'/' => put(b"/", &mut n)?,
            b'n' => put(b"\n", &mut n)?,
            b'r' => put(b"\r", &mut n)?,
            b't' => put(b"\t", &mut n)?,
            b'b' => put(b"\x08", &mut n)?,
            b'f' => put(b"\x0c", &mut n)?,
            b'u' => {
                let h = core::str::from_utf8(b.get(i..i + 4)?).ok()?;
                let mut cp = u32::from_str_radix(h, 16).ok()?;
                i += 4;
                if (0xd800..0xdc00).contains(&cp) && b.get(i) == Some(&b'\\') && b.get(i + 1) == Some(&b'u') {
                    let lo = u32::from_str_radix(core::str::from_utf8(b.get(i + 2..i + 6)?).ok()?, 16).ok()?;
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo.checked_sub(0xdc00)?);
                    i += 6;
                }
                let ch = char::from_u32(cp)?;
                let mut tmp = [0u8; 4];
                put(ch.encode_utf8(&mut tmp).as_bytes(), &mut n)?;
            }
            _ => return None,
        }
    }
    core::str::from_utf8(&out[..n]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: &str = r#"{"type":"input.text","session_id":"s1","n":3,"messages":[{"role":"user","content":"a \"b\""},{"role":"tool","content":[1,{"x":"]"}]}],"payment_token":"pt"}"#;

    #[test]
    fn get_and_items() {
        assert_eq!(get(F, "type"), Some("\"input.text\""));
        assert_eq!(as_str(get(F, "session_id").unwrap()), Some("s1"));
        assert_eq!(get(F, "n"), Some("3"));
        assert_eq!(get(F, "missing"), None);
        let msgs: [&str; 2] = {
            let mut it = items(get(F, "messages").unwrap());
            [it.next().unwrap(), it.next().unwrap()]
        };
        assert_eq!(as_str(get(msgs[0], "content").unwrap()), Some(r#"a \"b\""#));
        assert_eq!(get(msgs[1], "role"), Some("\"tool\""));
        assert_eq!(items(get(F, "messages").unwrap()).count(), 2);
        assert_eq!(get(F, "payment_token"), Some("\"pt\""), "a ] inside a string does not end the array early");
        assert_eq!(path(r#"{"a":{"b":{"c":true}}}"#, &["a", "b", "c"]), Some("true"));
    }

    #[test]
    fn unescapes() {
        let mut o = [0u8; 32];
        assert_eq!(unescape(r#"{\"deadtime\": 1.0}\né😀"#, &mut o), Some("{\"deadtime\": 1.0}\né😀"));
        let mut tiny = [0u8; 2];
        assert_eq!(unescape("abc", &mut tiny), None);
    }
}
