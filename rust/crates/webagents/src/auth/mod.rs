//! Agent sign-in by device authorization, the device side: the key, the codes
//! a person reads, and the RFC 9421 signature every request carries.
//!
//! The device makes the agent's own Ed25519 key; the owner allows it once on
//! the platform's `/device` page; from then on the device proves itself by
//! signing its requests with that key. Nothing long-lived but the private key
//! is kept, and it never leaves the device (or its secure element: only "sign
//! these bytes" and the public key are needed here).
//!
//! The wire form is the platform's agent sign-in contract (the
//! Web Bot Auth profile WITHOUT a `Signature-Agent` header): one label `sig1`,
//! always covering `@method`, `@authority`, `@path`, `@query`, plus
//! `content-digest` whenever there is a body and `authorization` whenever a
//! bearer is sent; `keyid` is the RFC 7638 thumbprint; `alg="ed25519"`, a
//! single-use `nonce`, `tag="web-bot-auth"`, and a window of at most 3600 s.
//! The shared test vectors (the TypeScript and Python SDKs check the same
//! file) are reproduced byte for byte by the tests.
//!
//! This module is `no_std` and heap-free: the caller supplies the buffers.
//!
//! # Signing a request
//!
//! ```
//! use webagents::auth::{DeviceKey, Request};
//!
//! // The seed comes from the device's secret store; this one is for the example.
//! let key = DeviceKey::from_seed(&[7u8; 32]);
//! assert_eq!(key.keyid().len(), 43);
//! assert_eq!(key.fingerprint().as_str().len(), 19); // XXXX-XXXX-XXXX-XXXX
//!
//! let req = Request {
//!     method: "POST",
//!     authority: "robutler.ai",
//!     path: "/api/agents/device/ws-ticket",
//!     query: None,
//!     body: Some(b"{}"),
//!     authorization: None,
//! };
//! // Valid for 60 seconds from `created`; the nonce is used once.
//! let mut scratch = [0u8; 1024];
//! let (headers, _) = key.sign(&req, 1_700_000_000, 1_700_000_060, "one-use-nonce", &mut scratch).unwrap();
//!
//! assert!(headers
//!     .signature_input
//!     .as_str()
//!     .starts_with(r#"sig1=("@method" "@authority" "@path" "@query" "content-digest");created=1700000000;expires=1700000060;keyid=""#));
//! assert!(headers.signature.as_str().starts_with("sig1=:"));
//! assert!(headers.content_digest.as_str().starts_with("sha-256=:"));
//! ```
//!
//! # Signing a tool description
//!
//! A part that holds its own key signs the exact bytes of its tool list, so a
//! gateway can check that the limits it is about to enforce are the limits the
//! part itself signed, before it offers a single tool.
//!
//! ```
//! use webagents::auth::{verify_tool_description, DeviceKey};
//!
//! let key = DeviceKey::from_seed(&[7u8; 32]);
//! let tools = br#"[{"type":"function","function":{"name":"pwm1_read_status"}}]"#;
//! let signature = key.sign_tool_description("pwm1", 1_700_000_000, tools);
//!
//! // The verifier holds the part's PUBLIC key, given to it out of band.
//! assert!(verify_tool_description(key.public_x(), "pwm1", 1_700_000_000, tools, signature.as_str()));
//! // One changed byte, another device name or another key: no.
//! assert!(!verify_tool_description(key.public_x(), "pwm1", 1_700_000_000, b"[]", signature.as_str()));
//! assert!(!verify_tool_description(key.public_x(), "pwm2", 1_700_000_000, tools, signature.as_str()));
//! let other = DeviceKey::from_seed(&[8u8; 32]);
//! assert!(!verify_tool_description(other.public_x(), "pwm1", 1_700_000_000, tools, signature.as_str()));
//! ```

use core::fmt::{self, Write};

use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};

use crate::regspec::json::BufWriter;

/// The `tag` parameter of every signature, naming the profile.
pub const TAG: &str = "web-bot-auth";
/// The one signature label used in `Signature-Input` and `Signature`.
pub const LABEL: &str = "sig1";
/// The longest window the platform accepts (`expires - created`).
pub const MAX_WINDOW_S: u64 = 3600;
/// The alphabet of user codes and fingerprints (RFC 8628 section 6.1).
pub const ALPHABET: &[u8; 20] = b"BCDFGHJKLMNPQRSTVWXZ";

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Base64 of `input` into `w`: standard with padding, or URL-safe without.
pub fn write_b64<W: Write + ?Sized>(w: &mut W, input: &[u8], url: bool) -> fmt::Result {
    let t = if url { B64URL } else { B64 };
    for c in input.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let v = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        w.write_char(t[(v >> 18) as usize & 63] as char)?;
        w.write_char(t[(v >> 12) as usize & 63] as char)?;
        if c.len() > 1 {
            w.write_char(t[(v >> 6) as usize & 63] as char)?;
        } else if !url {
            w.write_char('=')?;
        }
        if c.len() > 2 {
            w.write_char(t[v as usize & 63] as char)?;
        } else if !url {
            w.write_char('=')?;
        }
    }
    Ok(())
}

/// Decodes base64url (no padding) into `out`; returns the length.
pub fn b64url_decode(s: &str, out: &mut [u8]) -> Option<usize> {
    let (mut acc, mut bits, mut n) = (0u32, 0u32, 0usize);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            *out.get_mut(n)? = (acc >> bits) as u8;
            n += 1;
        }
    }
    Some(n)
}

/// A fixed-capacity text buffer: `N` bytes on the stack, written through
/// `core::fmt::Write`. A write that does not fit fails and leaves the text as
/// it was, so a header is never sent cut short.
#[derive(Clone)]
pub struct Text<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> Text<N> {
    /// An empty buffer.
    pub const fn new() -> Self {
        Text { buf: [0; N], len: 0 }
    }
    /// The text written so far.
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl<const N: usize> Default for Text<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> fmt::Write for Text<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let b = s.as_bytes();
        if self.len + b.len() > N {
            return Err(fmt::Error);
        }
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        Ok(())
    }
}

impl<const N: usize> fmt::Debug for Text<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize> fmt::Display for Text<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The RFC 7638 SHA-256 thumbprint of an Ed25519 public key given as its JWK
/// `x` (base64url): SHA-256 over `{"crv":"Ed25519","kty":"OKP","x":"<x>"}`,
/// exactly that member order and no whitespace.
pub fn thumbprint(x_b64url: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"{\"crv\":\"Ed25519\",\"kty\":\"OKP\",\"x\":\"");
    h.update(x_b64url.as_bytes());
    h.update(b"\"}");
    h.finalize().into()
}

/// The fingerprint a person compares on the device and on the approval page:
/// the first 16 bytes of the thumbprint, each as `ALPHABET[byte % 20]`, grouped
/// by four (`XXXX-XXXX-XXXX-XXXX`).
pub fn fingerprint(thumb: &[u8; 32]) -> Text<19> {
    let mut t = Text::new();
    for (i, b) in thumb[..16].iter().enumerate() {
        if i > 0 && i % 4 == 0 {
            let _ = t.write_char('-');
        }
        let _ = t.write_char(ALPHABET[(*b % 20) as usize] as char);
    }
    t
}

/// A user code as typed (any case, dashes or spaces) in its canonical
/// 8-letter form, or `None` when it is not a code.
pub fn normalize_user_code(input: &str) -> Option<[u8; 8]> {
    let mut out = [0u8; 8];
    let mut n = 0;
    for c in input.trim().bytes() {
        if c == b'-' || c == b' ' {
            continue;
        }
        let u = c.to_ascii_uppercase();
        if !ALPHABET.contains(&u) || n == 8 {
            return None;
        }
        out[n] = u;
        n += 1;
    }
    (n == 8).then_some(out)
}

/// The agent's device key: an Ed25519 key pair with its public `x` and key id
/// precomputed. The private key cannot be read back out, and the type has no
/// `Debug`, so it cannot reach a log by accident.
pub struct DeviceKey {
    key: SigningKey,
    x: Text<43>,
    keyid: Text<43>,
}

impl DeviceKey {
    /// From the 32-byte private seed (kept in the device's secret store or a secure element).
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let key = SigningKey::from_bytes(seed);
        let mut x = Text::new();
        let _ = write_b64(&mut x, key.verifying_key().as_bytes(), true);
        let mut keyid = Text::new();
        let _ = write_b64(&mut keyid, &thumbprint(x.as_str()), true);
        DeviceKey { key, x, keyid }
    }
    /// The public JWK's `x` (base64url).
    pub fn public_x(&self) -> &str {
        self.x.as_str()
    }
    /// `keyid`: the RFC 7638 thumbprint, base64url, 43 characters.
    pub fn keyid(&self) -> &str {
        self.keyid.as_str()
    }
    /// This key's fingerprint, `XXXX-XXXX-XXXX-XXXX`: what the device shows and
    /// the approval page repeats, so a person can tell that the page is about to
    /// allow THIS device and not another one asking at the same moment.
    pub fn fingerprint(&self) -> Text<19> {
        let mut t = [0u8; 32];
        b64url_decode(self.keyid(), &mut t);
        fingerprint(&t)
    }
    /// Writes the public JWK (`{"kty":"OKP","crv":"Ed25519","x":"..."}`).
    pub fn write_public_jwk<W: Write + ?Sized>(&self, w: &mut W) -> fmt::Result {
        write!(w, "{{\"kty\":\"OKP\",\"crv\":\"Ed25519\",\"x\":\"{}\"}}", self.x)
    }
}

/// The parts of a request the signature covers. They must be byte for byte
/// what is sent: the verifier rebuilds the same text from the request it
/// receives.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    /// Upper case (`GET`, `POST`).
    pub method: &'a str,
    /// The platform host, lower case, default port dropped.
    pub authority: &'a str,
    /// The path, starting with `/`, without the query.
    pub path: &'a str,
    /// The query without its `?`, or `None` for a URL with no query string.
    pub query: Option<&'a str>,
    /// The exact body bytes, when there is a body (also `{}`).
    pub body: Option<&'a [u8]>,
    /// The `Authorization` header value, when one is sent.
    pub authorization: Option<&'a str>,
}

/// The three headers of a signed request, ready to send as they are.
#[derive(Debug, Clone, Default)]
pub struct SignedHeaders {
    /// The `Signature-Input` header value: `sig1=(...)` with the covered
    /// components and the parameters.
    pub signature_input: Text<512>,
    /// The `Signature` header value: `sig1=:<base64 of the 64-byte signature>:`.
    pub signature: Text<100>,
    /// Empty when the request has no body.
    pub content_digest: Text<60>,
}

/// Why a request could not be signed. Nothing was signed in any of these cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignError {
    /// `expires` before `created`, or a window over 3600 s.
    Window,
    /// The nonce is empty or over 256 characters.
    Nonce,
    /// The signature base did not fit the scratch buffer.
    TooLarge,
}

impl DeviceKey {
    /// Signs `req` for the window `created..expires` with the single-use `nonce`.
    /// `scratch` holds the signature base (a request with a long bearer needs
    /// about 1.5 KB). Returns the headers and the signature base's length in
    /// `scratch` (for logging and tests).
    ///
    /// `created` and `expires` are Unix seconds on the PLATFORM's clock: a
    /// device without a real-time clock takes the time from the `Date` header
    /// of any answer first.
    ///
    /// # Errors
    /// See [`SignError`]. Nothing is signed on an error.
    pub fn sign(&self, req: &Request, created: u64, expires: u64, nonce: &str, scratch: &mut [u8]) -> Result<(SignedHeaders, usize), SignError> {
        if expires < created || expires - created > MAX_WINDOW_S {
            return Err(SignError::Window);
        }
        if nonce.is_empty() || nonce.len() > 256 {
            return Err(SignError::Nonce);
        }
        let mut out = SignedHeaders::default();
        if let Some(body) = req.body {
            let d: [u8; 32] = Sha256::digest(body).into();
            out.content_digest.write_str("sha-256=:").map_err(|_| SignError::TooLarge)?;
            write_b64(&mut out.content_digest, &d, false).map_err(|_| SignError::TooLarge)?;
            out.content_digest.write_char(':').map_err(|_| SignError::TooLarge)?;
        }
        // The component list and parameters, shared by the base and Signature-Input.
        let mut params: Text<400> = Text::new();
        let p = (|| -> fmt::Result {
            params.write_str("(\"@method\" \"@authority\" \"@path\" \"@query\"")?;
            if req.body.is_some() {
                params.write_str(" \"content-digest\"")?;
            }
            if req.authorization.is_some() {
                params.write_str(" \"authorization\"")?;
            }
            write!(params, ");created={created};expires={expires};keyid=\"{}\";alg=\"ed25519\";nonce=\"{nonce}\";tag=\"{TAG}\"", self.keyid)
        })();
        p.map_err(|_| SignError::TooLarge)?;
        let mut w = BufWriter::new(scratch);
        let b = (|| -> fmt::Result {
            write!(w, "\"@method\": {}\n\"@authority\": {}\n\"@path\": {}\n\"@query\": ?{}\n", req.method, req.authority, req.path, req.query.unwrap_or(""))?;
            if req.body.is_some() {
                writeln!(w, "\"content-digest\": {}", out.content_digest)?;
            }
            if let Some(a) = req.authorization {
                writeln!(w, "\"authorization\": {a}")?;
            }
            write!(w, "\"@signature-params\": {params}")
        })();
        if b.is_err() {
            return Err(SignError::TooLarge);
        }
        let n = w.finish().ok_or(SignError::TooLarge)?;
        let sig = self.key.sign(&scratch[..n]);
        write!(out.signature_input, "{LABEL}={params}").map_err(|_| SignError::TooLarge)?;
        out.signature.write_str("sig1=:").map_err(|_| SignError::TooLarge)?;
        write_b64(&mut out.signature, &sig.to_bytes(), false).map_err(|_| SignError::TooLarge)?;
        out.signature.write_char(':').map_err(|_| SignError::TooLarge)?;
        Ok((out, n))
    }
}

// ---- the signed tool description ------------------------------------------------------

/// The first line of the signed message.
pub const TOOL_DESCRIPTION_FORMAT: &str = "webagents-tool-description/1";

/// Decodes standard base64 (with or without padding) into `out`; returns the length.
pub fn b64_decode(s: &str, out: &mut [u8]) -> Option<usize> {
    let (mut acc, mut bits, mut n) = (0u32, 0u32, 0usize);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            *out.get_mut(n)? = (acc >> bits) as u8;
            n += 1;
        }
    }
    Some(n)
}

/// Writes the signed message of a tool description: five lines joined by `\n`,
/// no trailing newline (format, keyid, device, created, base64 SHA-256 of the
/// exact description bytes). The description itself is never re-serialized.
pub fn write_tool_description_message<W: Write + ?Sized>(w: &mut W, keyid: &str, device: &str, created: u64, tools: &[u8]) -> fmt::Result {
    write_tool_description_message_sha256(w, keyid, device, created, &Sha256::digest(tools).into())
}

/// [`write_tool_description_message`] from the SHA-256 of the description, for a
/// part that streams its description rather than holding it ([`ToolDigest`]).
pub fn write_tool_description_message_sha256<W: Write + ?Sized>(w: &mut W, keyid: &str, device: &str, created: u64, d: &[u8; 32]) -> fmt::Result {
    write!(w, "{TOOL_DESCRIPTION_FORMAT}\nkeyid: {keyid}\ndevice: {device}\ncreated: {created}\nsha-256: ")?;
    write_b64(w, d, false)
}

/// A `fmt::Write` sink that hashes and counts what is written to it. A part runs
/// its tool writer into this once at boot to sign, and again into its link to
/// send: the description (8.5 KB for the example device) is never held in RAM.
#[derive(Default)]
pub struct ToolDigest {
    h: Sha256,
    len: usize,
}

impl ToolDigest {
    /// An empty digest.
    pub fn new() -> Self {
        Self::default()
    }
    /// The byte count and SHA-256 of everything written.
    pub fn finish(self) -> (usize, [u8; 32]) {
        (self.len, self.h.finalize().into())
    }
}

impl Write for ToolDigest {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.h.update(s.as_bytes());
        self.len += s.len();
        Ok(())
    }
}

impl DeviceKey {
    /// Signs a tool description (the exact bytes the device sends) as `device` at `created`.
    /// Returns the base64 signature (88 characters).
    pub fn sign_tool_description(&self, device: &str, created: u64, tools: &[u8]) -> Text<88> {
        self.sign_tool_description_sha256(device, created, &Sha256::digest(tools).into())
    }

    /// [`Self::sign_tool_description`] from the description's SHA-256 ([`ToolDigest`]).
    pub fn sign_tool_description_sha256(&self, device: &str, created: u64, d: &[u8; 32]) -> Text<88> {
        let mut msg: Text<256> = Text::new();
        let _ = write_tool_description_message_sha256(&mut msg, self.keyid(), device, created, d);
        let sig = self.key.sign(msg.as_str().as_bytes());
        let mut out = Text::new();
        let _ = write_b64(&mut out, &sig.to_bytes(), false);
        out
    }
}

/// Verifies a tool description against the device's public key (`x`, base64url, as
/// the gateway holds it for that device). The keyid in the message is computed from
/// that key, so a description signed by any other key fails.
pub fn verify_tool_description(public_x: &str, device: &str, created: u64, tools: &[u8], signature_b64: &str) -> bool {
    let mut pk = [0u8; 32];
    if b64url_decode(public_x, &mut pk) != Some(32) {
        return false;
    }
    let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(&pk) else { return false };
    let mut sig = [0u8; 64];
    if b64_decode(signature_b64, &mut sig) != Some(64) {
        return false;
    }
    let mut keyid: Text<43> = Text::new();
    let _ = write_b64(&mut keyid, &thumbprint(public_x), true);
    let mut msg: Text<256> = Text::new();
    if write_tool_description_message(&mut msg, keyid.as_str(), device, created, tools).is_err() {
        return false;
    }
    vk.verify_strict(msg.as_str().as_bytes(), &ed25519_dalek::Signature::from_bytes(&sig)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_codes() {
        assert_eq!(normalize_user_code(" bcdf-ghjk "), Some(*b"BCDFGHJK"));
        assert_eq!(normalize_user_code("BCDF GHJK"), Some(*b"BCDFGHJK"));
        for bad in ["BCDF-GHJ", "BCDF-GHJKL", "ABCD-EFGH", "BCDF_GHJK", ""] {
            assert_eq!(normalize_user_code(bad), None, "{bad}");
        }
    }

    #[test]
    fn window_and_nonce_are_checked() {
        let k = DeviceKey::from_seed(&[7; 32]);
        let req = Request { method: "GET", authority: "h", path: "/", query: None, body: None, authorization: None };
        let mut s = [0u8; 512];
        assert_eq!(k.sign(&req, 10, 9, "n", &mut s).unwrap_err(), SignError::Window);
        assert_eq!(k.sign(&req, 0, 3601, "n", &mut s).unwrap_err(), SignError::Window);
        assert_eq!(k.sign(&req, 0, 60, "", &mut s).unwrap_err(), SignError::Nonce);
        assert!(k.sign(&req, 0, 3600, "n", &mut s).is_ok());
    }
}
