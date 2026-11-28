//! The two pieces of JSON a device tool needs, with no heap: reading a FLAT
//! argument object (`{"en": true, "deadtime": 1.0, "mode": "sinusoidal"}`) and
//! writing results and tool descriptors into a caller's buffer.
//!
//! Tool arguments for register tools are flat by construction (one key per
//! field), so a full JSON parser would be flash spent on shapes that are refused
//! anyway. Nested values are an error, not something skipped: an argument the
//! tool did not read must never look accepted.

use core::fmt;

/// One argument value, borrowed from the argument text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'s> {
    /// `null`. In a `set` call it means "leave this field as it is".
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// The number exactly as written (valid JSON number grammar). It is converted
    /// with exact decimal arithmetic by the engine, never through a float.
    Number(&'s str),
    /// The raw text between the quotes. Escapes are left as written; register tool
    /// strings are enum names, which never need them.
    Str(&'s str),
}

/// Why an argument object was not accepted. The number is the byte offset in
/// the argument text where reading stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonError {
    /// Not a JSON object, or malformed at this byte.
    Syntax(usize),
    /// An object or array as a value: register tools take flat arguments.
    Nested(usize),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonError::Syntax(at) => write!(f, "invalid JSON at byte {at}"),
            JsonError::Nested(at) => write!(f, "nested value at byte {at}: arguments must be a flat object"),
        }
    }
}

/// Iterator over the members of a flat JSON object.
///
/// Each item is a `(key, value)` pair borrowed from the source text, or the
/// error that ended the walk. Keys are returned as written between the
/// quotes; field names never need escapes.
///
/// ```
/// use webagents::regspec::json::{FlatObject, JsonError, Value};
///
/// let mut args = FlatObject::parse(r#"{"en": true, "deadtime": 1.0, "mode": "sinusoidal"}"#).unwrap();
/// assert_eq!(args.next(), Some(Ok(("en", Value::Bool(true)))));
/// assert_eq!(args.next(), Some(Ok(("deadtime", Value::Number("1.0")))));
/// assert_eq!(args.next(), Some(Ok(("mode", Value::Str("sinusoidal")))));
/// assert_eq!(args.next(), None);
///
/// // A nested value is an error, with the byte where it starts.
/// assert_eq!(FlatObject::check(r#"{"a": {"b": 1}}"#), Err(JsonError::Nested(6)));
/// ```
pub struct FlatObject<'s> {
    s: &'s [u8],
    src: &'s str,
    i: usize,
    done: bool,
    first: bool,
}

impl<'s> FlatObject<'s> {
    /// Accepts `{...}` (surrounding whitespace allowed). An empty or all-blank
    /// input is read as `{}`, since some models send no arguments at all.
    pub fn parse(src: &'s str) -> Result<Self, JsonError> {
        let s = src.as_bytes();
        let mut i = skip_ws(s, 0);
        if i == s.len() {
            return Ok(FlatObject { s, src, i, done: true, first: true });
        }
        if s[i] != b'{' {
            return Err(JsonError::Syntax(i));
        }
        i += 1;
        Ok(FlatObject { s, src, i, done: false, first: true })
    }

    /// Validates the whole object without using it (the engine walks it twice:
    /// once to check every key is known, once to apply).
    pub fn check(src: &'s str) -> Result<(), JsonError> {
        for m in FlatObject::parse(src)? {
            m?;
        }
        Ok(())
    }
}

impl<'s> Iterator for FlatObject<'s> {
    type Item = Result<(&'s str, Value<'s>), JsonError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let s = self.s;
        let mut i = skip_ws(s, self.i);
        if i < s.len() && s[i] == b'}' {
            self.done = true;
            let j = skip_ws(s, i + 1);
            if j != s.len() {
                return Some(Err(JsonError::Syntax(j)));
            }
            return None;
        }
        if !self.first {
            if i >= s.len() || s[i] != b',' {
                self.done = true;
                return Some(Err(JsonError::Syntax(i)));
            }
            i = skip_ws(s, i + 1);
        }
        self.first = false;
        let r = (|| {
            let (key, j) = read_str(self.src, i)?;
            let j = skip_ws(s, j);
            if j >= s.len() || s[j] != b':' {
                return Err(JsonError::Syntax(j));
            }
            let j = skip_ws(s, j + 1);
            let (v, j) = read_value(self.src, j)?;
            Ok(((key, v), j))
        })();
        match r {
            Ok((kv, j)) => {
                self.i = j;
                Some(Ok(kv))
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

fn skip_ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

fn read_str(src: &str, i: usize) -> Result<(&str, usize), JsonError> {
    let s = src.as_bytes();
    if i >= s.len() || s[i] != b'"' {
        return Err(JsonError::Syntax(i));
    }
    let start = i + 1;
    let mut j = start;
    while j < s.len() {
        match s[j] {
            b'\\' => j += 2,
            b'"' => return Ok((&src[start..j], j + 1)),
            c if c < 0x20 => return Err(JsonError::Syntax(j)),
            _ => j += 1,
        }
    }
    Err(JsonError::Syntax(j))
}

fn read_value(src: &str, i: usize) -> Result<(Value<'_>, usize), JsonError> {
    let s = src.as_bytes();
    if i >= s.len() {
        return Err(JsonError::Syntax(i));
    }
    match s[i] {
        b'"' => read_str(src, i).map(|(v, j)| (Value::Str(v), j)),
        b't' if s[i..].starts_with(b"true") => Ok((Value::Bool(true), i + 4)),
        b'f' if s[i..].starts_with(b"false") => Ok((Value::Bool(false), i + 5)),
        b'n' if s[i..].starts_with(b"null") => Ok((Value::Null, i + 4)),
        b'{' | b'[' => Err(JsonError::Nested(i)),
        b'-' | b'0'..=b'9' => {
            let j = number_end(s, i).ok_or(JsonError::Syntax(i))?;
            Ok((Value::Number(&src[i..j]), j))
        }
        _ => Err(JsonError::Syntax(i)),
    }
}

/// `core::fmt::Write` into a fixed buffer. Overflow is sticky and reported by
/// [`BufWriter::finish`], so a truncated result is never sent as if whole.
pub struct BufWriter<'b> {
    buf: &'b mut [u8],
    len: usize,
    overflow: bool,
}

impl<'b> BufWriter<'b> {
    /// An empty writer over `buf`. The buffer's length is the limit.
    pub fn new(buf: &'b mut [u8]) -> Self {
        BufWriter { buf, len: 0, overflow: false }
    }
    /// Bytes written so far. After an overflow it stays at the last write that fitted.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The number of bytes written, or `None` if the buffer was too small.
    pub fn finish(self) -> Option<usize> {
        if self.overflow {
            None
        } else {
            Some(self.len)
        }
    }
}

impl fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let b = s.as_bytes();
        if self.overflow || self.len + b.len() > self.buf.len() {
            self.overflow = true;
            return Err(fmt::Error);
        }
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        Ok(())
    }
}

/// Writes `s` as a JSON string literal, quotes included: `"`, `\\` and
/// control characters are escaped, everything else (also non-ASCII) is
/// written as is.
pub fn write_str<W: fmt::Write + ?Sized>(w: &mut W, s: &str) -> fmt::Result {
    w.write_char('"')?;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        let esc: Option<&str> = match c {
            '"' => Some("\\\""),
            '\\' => Some("\\\\"),
            '\n' => Some("\\n"),
            '\r' => Some("\\r"),
            '\t' => Some("\\t"),
            c if (c as u32) < 0x20 => None,
            _ => continue,
        };
        w.write_str(&s[start..i])?;
        match esc {
            Some(e) => w.write_str(e)?,
            None => write!(w, "\\u{:04x}", c as u32)?,
        }
        start = i + c.len_utf8();
    }
    w.write_str(&s[start..])?;
    w.write_char('"')
}

/// End of a JSON number starting at `i`, if the grammar holds:
/// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
fn number_end(s: &[u8], mut i: usize) -> Option<usize> {
    let digits = |s: &[u8], mut i: usize| {
        let st = i;
        while i < s.len() && s[i].is_ascii_digit() {
            i += 1;
        }
        (i > st).then_some(i)
    };
    if s.get(i) == Some(&b'-') {
        i += 1;
    }
    match s.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits(s, i)?,
        _ => return None,
    }
    if s.get(i) == Some(&b'.') {
        i = digits(s, i + 1)?;
    }
    if matches!(s.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(s.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        i = digits(s, i)?;
    }
    Some(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_object() {
        extern crate std;
        use std::vec::Vec;
        let v: Vec<_> = FlatObject::parse(r#" {"a": 1, "b":true,"c":"x y", "d": -2.5e1, "e": null} "#).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(v.len(), 5);
        assert_eq!(v[0], ("a", Value::Number("1")));
        assert_eq!(v[3], ("d", Value::Number("-2.5e1")));
        assert_eq!(v[2], ("c", Value::Str("x y")));
    }

    #[test]
    fn refuses_nested_and_trailing() {
        assert!(FlatObject::check(r#"{"a": {"b": 1}}"#).is_err());
        assert!(FlatObject::check(r#"{"a": [1]}"#).is_err());
        assert!(FlatObject::check(r#"{"a": 1} x"#).is_err());
        assert!(FlatObject::check(r#"{"a": 1 "b": 2}"#).is_err());
        assert!(FlatObject::check(r#"{"a": 01}"#).is_err());
        assert!(FlatObject::check(r#"{"a": 1.}"#).is_err());
        assert!(FlatObject::check(r#"{"a": 1-2}"#).is_err());
        assert!(FlatObject::check("").is_ok());
        assert!(FlatObject::check("{}").is_ok());
    }

    #[test]
    fn writes_escaped() {
        let mut b = [0u8; 64];
        let mut w = BufWriter::new(&mut b);
        write_str(&mut w, "a\"b\\c\n\u{1}").unwrap();
        let n = w.finish().unwrap();
        assert_eq!(core::str::from_utf8(&b[..n]).unwrap(), r#""a\"b\\c\n\u0001""#);
    }

    #[test]
    fn overflow_is_reported() {
        let mut b = [0u8; 4];
        let mut w = BufWriter::new(&mut b);
        let _ = fmt::Write::write_str(&mut w, "hello");
        assert!(w.finish().is_none());
    }
}
