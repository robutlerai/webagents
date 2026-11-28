//! Agent sign-in by device authorization, the gateway's client: no browser.
//!
//! `robutler-device agent login @agent` makes the agent's Ed25519 key, asks the
//! platform for a code ([`SignIn::authorize`]), shows the code, the key's
//! fingerprint and the URL on the console, and polls ([`SignIn::poll`]) while
//! the owner allows it on a phone. From then on every call is signed by that
//! key ([`crate::auth`]): a WebSocket ticket ([`SignIn::ws_ticket`]) opens the
//! agent socket, a refresh ([`SignIn::refresh`]) gives a 60-minute token that
//! is accepted only beside a signature covering it, and a signed DELETE
//! ([`SignIn::logout`]) signs the device out. The contract is the platform's
//! agent sign-in API; the signature is checked byte for byte against its
//! shared vectors in `tests/auth_vectors.rs`.
//!
//! The private key is the only long-lived secret. It is written once, 0600 in a
//! 0700 directory, under `agent-device:<portal host>:<agent id>`, and never
//! printed. Tokens and tickets live in memory.
//!
//! Requests are blocking HTTP (`ureq`); the gateway calls them off its async
//! threads. Key storage uses Unix file permissions, so this module is for
//! Unix hosts.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::GatewayError;
use crate::auth::{write_b64, DeviceKey, Request};

/// Seconds a signature is valid for (the platform allows up to 3600).
const WINDOW_S: u64 = 60;

/// A platform base URL split into what the signature needs.
#[derive(Debug, Clone)]
pub struct Portal {
    /// `https://host[:port]`
    pub origin: String,
    /// Lower case, default port dropped: the `@authority` the platform expects.
    pub authority: String,
}

impl Portal {
    /// Splits a platform URL. `https`, `http`, `wss` and `ws` URLs are accepted
    /// (a socket URL names the same host); any path is dropped.
    ///
    /// # Errors
    /// [`GatewayError::Config`] for another scheme or a URL without a host.
    pub fn parse(url: &str) -> Result<Portal, GatewayError> {
        let u = url.trim().trim_end_matches('/');
        let (scheme, rest) = u.split_once("://").ok_or_else(|| GatewayError::Config(format!("{url} is not an http(s) URL")))?;
        let host = rest.split('/').next().unwrap_or("").to_ascii_lowercase();
        let (scheme, default_port) = match scheme {
            "https" | "wss" => ("https", ":443"),
            "http" | "ws" => ("http", ":80"),
            other => return Err(GatewayError::Config(format!("{other} is not http or https"))),
        };
        let authority = host.strip_suffix(default_port).unwrap_or(&host).to_string();
        if authority.is_empty() {
            return Err(GatewayError::Config(format!("{url} has no host")));
        }
        Ok(Portal { origin: format!("{scheme}://{authority}"), authority })
    }
}

/// The answer to [`SignIn::authorize`]: what the device shows the person.
#[derive(Debug, Clone)]
pub struct Pending {
    /// The secret the device polls with. Never shown to a person.
    pub device_code: String,
    /// The code the person types on the approval page (`XXXX-XXXX`).
    pub user_code: String,
    /// The approval page to open.
    pub verification_uri: String,
    /// The approval page with the code already filled in (what a QR code carries).
    pub verification_uri_complete: String,
    /// Seconds until the code expires (600 when the platform does not say).
    pub expires_in: u64,
    /// Seconds to wait between polls (5 when the platform does not say).
    pub interval: u64,
    /// This device key's fingerprint (`XXXX-XXXX-XXXX-XXXX`), computed here. The
    /// person checks that the approval page shows the same one.
    pub fingerprint: String,
    /// The platform's id of the agent being signed in to.
    pub agent_id: String,
    /// The agent's name on the platform.
    pub agent_username: String,
}

/// The answer to [`SignIn::poll`] or [`SignIn::refresh`].
#[derive(Debug, Clone)]
pub struct AccessToken {
    /// A bearer token for the agent's API calls. The platform accepts it only
    /// beside a signature by the device key that covers it.
    pub access_token: String,
    /// Seconds the token is good for (3600 when the platform does not say).
    pub expires_in: u64,
    /// The platform's id of the agent.
    pub agent_id: String,
    /// The agent's name on the platform.
    pub agent_username: String,
    /// The id of this device's sign-in, needed to sign it out.
    pub binding_id: String,
}

/// An HTTP answer, as the client reads it.
#[derive(Debug, Clone)]
pub struct Answer {
    /// The HTTP status.
    pub status: u16,
    /// The `Date` header, used to correct a wrong device clock.
    pub date: Option<String>,
    /// The `Retry-After` header in seconds, when present.
    pub retry_after: Option<u64>,
    /// The body as JSON; `Null` when it was empty or not JSON.
    pub body: Value,
}

/// The device's side of the sign-in.
pub struct SignIn {
    /// The platform this sign-in is with.
    pub portal: Portal,
    /// The device's key. Every request is signed with it.
    pub key: DeviceKey,
    /// Seconds to add to this host's clock (from the platform's `Date`).
    pub clock_offset: i64,
    http: ureq::Agent,
}

/// Shows the portal and the PUBLIC key id only: the private key never reaches a log.
impl std::fmt::Debug for SignIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignIn").field("portal", &self.portal.origin).field("keyid", &self.key.keyid()).field("clock_offset", &self.clock_offset).finish()
    }
}

fn now_s() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// 64 random bytes, base64: the nonce convention of the contract.
fn nonce() -> Result<String, GatewayError> {
    let mut b = [0u8; 64];
    getrandom::fill(&mut b).map_err(|e| GatewayError::Portal(format!("no random source: {e}")))?;
    let mut s = String::new();
    write_b64(&mut s, &b, false).expect("String");
    Ok(s)
}

/// A new private key seed from the system's random source.
pub fn new_seed() -> Result<[u8; 32], GatewayError> {
    let mut s = [0u8; 32];
    getrandom::fill(&mut s).map_err(|e| GatewayError::Portal(format!("no random source: {e}")))?;
    Ok(s)
}

/// Parses an HTTP date (`Wed, 26 Nov 2026 00:00:00 GMT`) to Unix seconds.
pub fn parse_http_date(s: &str) -> Option<i64> {
    let p: Vec<&str> = s.split_whitespace().collect();
    if p.len() < 5 {
        return None;
    }
    let day: i64 = p[1].parse().ok()?;
    let mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].iter().position(|m| *m == p[2])? as i64 + 1;
    let year: i64 = p[3].parse().ok()?;
    let hms: Vec<i64> = p[4].split(':').filter_map(|x| x.parse().ok()).collect();
    if hms.len() != 3 {
        return None;
    }
    // Days from civil (Howard Hinnant's algorithm).
    let (y, m) = if mon <= 2 { (year - 1, mon + 9) } else { (year, mon - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86400 + hms[0] * 3600 + hms[1] * 60 + hms[2])
}

impl SignIn {
    /// A client for `portal` signing with `key`, with a 30 s timeout per request
    /// and no clock correction yet.
    pub fn new(portal: Portal, key: DeviceKey) -> SignIn {
        let http: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(30))).build().into();
        SignIn { portal, key, clock_offset: 0, http }
    }

    fn now(&self) -> u64 {
        (now_s() + self.clock_offset).max(0) as u64
    }

    /// One unsigned GET to read the platform's clock: a device with no clock (or
    /// a wrong one) must sign inside the platform's 60 s skew from the first call.
    pub fn sync_clock(&mut self) -> Result<i64, GatewayError> {
        let r = self.http.get(&self.portal.origin).call().map_err(|e| GatewayError::Portal(format!("{}: {e}", self.portal.origin)))?;
        let date = r
            .headers()
            .get("date")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_http_date)
            .ok_or_else(|| GatewayError::Portal("the platform sent no Date header".into()))?;
        self.clock_offset = date - now_s();
        Ok(self.clock_offset)
    }

    /// A signed request. `path_q` is the path with its query, if any.
    /// `method` is `GET`, `DELETE` or `POST`; `body` is sent as JSON and
    /// covered by the signature; `bearer` adds an `Authorization` header,
    /// covered too. Each call uses a fresh nonce and a 60 s window.
    ///
    /// An HTTP error status is not an `Err`: read [`Answer::status`].
    ///
    /// # Errors
    /// [`GatewayError::Portal`] when the request cannot be signed or sent.
    pub fn signed(&self, method: &str, path_q: &str, body: Option<&[u8]>, bearer: Option<&str>) -> Result<Answer, GatewayError> {
        let (path, query) = match path_q.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (path_q, None),
        };
        let auth = bearer.map(|b| format!("Bearer {b}"));
        let req = Request { method, authority: &self.portal.authority, path, query, body, authorization: auth.as_deref() };
        let created = self.now();
        let n = nonce()?;
        let mut scratch = vec![0u8; 8192];
        let (h, _) = self.key.sign(&req, created, created + WINDOW_S, &n, &mut scratch).map_err(|e| GatewayError::Portal(format!("cannot sign: {e:?}")))?;
        let url = format!("{}{}", self.portal.origin, path_q);
        let r = match method {
            "GET" => {
                let mut rb = self.http.get(&url).header("Signature-Input", h.signature_input.as_str()).header("Signature", h.signature.as_str());
                if let Some(a) = &auth {
                    rb = rb.header("Authorization", a);
                }
                rb.call()
            }
            "DELETE" => {
                let mut rb = self.http.delete(&url).header("Signature-Input", h.signature_input.as_str()).header("Signature", h.signature.as_str());
                if let Some(a) = &auth {
                    rb = rb.header("Authorization", a);
                }
                rb.call()
            }
            _ => {
                let mut rb = self.http.post(&url).header("Signature-Input", h.signature_input.as_str()).header("Signature", h.signature.as_str());
                if let Some(a) = &auth {
                    rb = rb.header("Authorization", a);
                }
                match body {
                    Some(b) => rb.header("Content-Type", "application/json").header("Content-Digest", h.content_digest.as_str()).send(b),
                    None => rb.send_empty(),
                }
            }
        };
        let mut r = r.map_err(|e| GatewayError::Portal(format!("{url}: {e}")))?;
        let status = r.status().as_u16();
        let date = r.headers().get("date").and_then(|v| v.to_str().ok()).map(str::to_string);
        let retry_after = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok());
        let text = r.body_mut().read_to_string().unwrap_or_default();
        let body = serde_json::from_str(&text).unwrap_or(Value::Null);
        Ok(Answer { status, date, retry_after, body })
    }

    /// Asks for a code to bind this key to `agent` (`@name`, name or id).
    /// `device_name` is what the owner sees on the approval page and in the
    /// agent's list of devices. Also corrects the clock from the answer.
    ///
    /// # Errors
    /// [`GatewayError::Portal`] when the platform cannot be reached or
    /// refuses (an unknown agent, a bad signature).
    pub fn authorize(&mut self, agent: &str, device_name: &str) -> Result<Pending, GatewayError> {
        let mut jwk = String::new();
        self.key.write_public_jwk(&mut jwk).expect("String");
        let body = json!({
            "agent": agent,
            "public_key": serde_json::from_str::<Value>(&jwk).expect("jwk"),
            "device_name": device_name,
            "device_info": { "os": std::env::consts::OS, "sdk": concat!("webagents-rust/", env!("CARGO_PKG_VERSION")) },
        })
        .to_string();
        let a = self.signed("POST", "/api/agents/device/authorize", Some(body.as_bytes()), None)?;
        if let Some(d) = a.date.as_deref().and_then(parse_http_date) {
            self.clock_offset = d - now_s();
        }
        if a.status != 200 {
            return Err(refused("authorize", &a));
        }
        let s = |k: &str| a.body[k].as_str().unwrap_or("").to_string();
        Ok(Pending {
            device_code: s("device_code"),
            user_code: s("user_code"),
            verification_uri: s("verification_uri"),
            verification_uri_complete: s("verification_uri_complete"),
            expires_in: a.body["expires_in"].as_u64().unwrap_or(600),
            interval: a.body["interval"].as_u64().unwrap_or(5),
            fingerprint: self.key.fingerprint().as_str().to_string(),
            agent_id: a.body["agent"]["id"].as_str().unwrap_or("").to_string(),
            agent_username: a.body["agent"]["username"].as_str().unwrap_or("").to_string(),
        })
    }

    /// Polls for the owner's answer at the platform's interval. `sleep` is
    /// how the caller waits (a test passes a no-op). Blocks until the owner
    /// allows or denies the device or the code expires.
    ///
    /// # Errors
    /// [`GatewayError::Portal`] when the owner denies the device, the code
    /// expires or was already used, or the platform refuses.
    pub fn poll(&mut self, p: &Pending, sleep: &mut dyn FnMut(Duration)) -> Result<AccessToken, GatewayError> {
        let body = json!({ "device_code": p.device_code, "grant_type": "urn:ietf:params:oauth:grant-type:device_code" }).to_string();
        let mut interval = p.interval.max(1);
        let deadline = std::time::Instant::now() + Duration::from_secs(p.expires_in + 30);
        let mut first = true;
        loop {
            if !first {
                sleep(Duration::from_secs(interval));
            }
            first = false;
            if std::time::Instant::now() > deadline {
                return Err(GatewayError::Portal("the code expired before the owner answered".into()));
            }
            let a = self.signed("POST", "/api/agents/device/token", Some(body.as_bytes()), None)?;
            if a.status == 200 {
                return token_of(&a);
            }
            match a.body["error"].as_str().unwrap_or("") {
                "authorization_pending" => {}
                "slow_down" => interval = a.body["interval"].as_u64().unwrap_or(interval + 5).min(60),
                "access_denied" => return Err(GatewayError::Portal("the owner denied this device".into())),
                "expired_token" => return Err(GatewayError::Portal("the code expired or was already used; sign in again".into())),
                _ if a.status == 429 => interval = interval.max(a.retry_after.unwrap_or(interval)),
                _ => return Err(refused("token", &a)),
            }
        }
    }

    /// A fresh 60-minute token (accepted only beside a signature covering it).
    pub fn refresh(&self) -> Result<AccessToken, GatewayError> {
        let a = self.signed("POST", "/api/agents/device/refresh", Some(b"{}"), None)?;
        if a.status != 200 {
            return Err(refused("refresh", &a));
        }
        token_of(&a)
    }

    /// A one-use ticket for `/ws?ticket=`, good for 60 seconds. The gateway
    /// fetches one per connection attempt.
    pub fn ws_ticket(&self) -> Result<String, GatewayError> {
        let a = self.signed("POST", "/api/agents/device/ws-ticket", Some(b"{}"), None)?;
        match (a.status, a.body["ticket"].as_str()) {
            (200, Some(t)) => Ok(t.to_string()),
            _ => Err(refused("ws-ticket", &a)),
        }
    }

    /// Signs this device out, by the device itself. Already signed out is success.
    /// The stored key is left in place; it is no longer accepted.
    pub fn logout(&self, agent_id: &str, binding_id: &str) -> Result<(), GatewayError> {
        let a = self.signed("DELETE", &format!("/api/agents/{agent_id}/devices/{binding_id}"), None, None)?;
        if a.status == 200 || a.body["error_code"] == "device_signed_out" {
            Ok(())
        } else {
            Err(refused("sign out", &a))
        }
    }
}

fn token_of(a: &Answer) -> Result<AccessToken, GatewayError> {
    let s = |k: &str| a.body[k].as_str().unwrap_or("").to_string();
    Ok(AccessToken {
        access_token: s("access_token"),
        expires_in: a.body["expires_in"].as_u64().unwrap_or(3600),
        agent_id: a.body["agent"]["id"].as_str().unwrap_or("").to_string(),
        agent_username: a.body["agent"]["username"].as_str().unwrap_or("").to_string(),
        binding_id: s("binding_id"),
    })
}

/// A refusal in the platform's words (`error_code` for signature refusals, `error` otherwise), never its body whole.
fn refused(what: &str, a: &Answer) -> GatewayError {
    let code = a.body["error_code"].as_str().or(a.body["error"].as_str()).unwrap_or("?");
    let desc = a.body["error_description"].as_str().unwrap_or("");
    GatewayError::Portal(format!("{what}: HTTP {} {code} {desc}", a.status))
}

// ---- storage ---------------------------------------------------------------------------

/// Where device keys live: `$WEBAGENTS_HOME` or `~/.robutler`, directory 0700.
pub fn key_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WEBAGENTS_HOME") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".robutler")
}

/// `agent-device:<portal host>:<agent id>`.
pub fn key_name(portal: &Portal, agent_id: &str) -> String {
    format!("agent-device:{}:{agent_id}", portal.authority)
}

/// Writes the seed 0600 (created new, never overwritten) in a 0700 directory.
/// The file holds the seed as 64 hexadecimal characters. Returns its path.
///
/// # Errors
/// [`GatewayError::Config`] when the directory or file cannot be created,
/// including when a key of that name already exists.
pub fn store_seed(dir: &Path, name: &str, seed: &[u8; 32]) -> Result<PathBuf, GatewayError> {
    use std::io::Write;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| GatewayError::Config(format!("{}: {e}", dir.display())))?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| GatewayError::Config(e.to_string()))?;
    let path = dir.join(name);
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|e| GatewayError::Config(format!("{}: {e}", path.display())))?;
    let mut hex = String::with_capacity(64);
    for b in seed {
        hex.push_str(&format!("{b:02x}"));
    }
    f.write_all(hex.as_bytes()).map_err(|e| GatewayError::Config(e.to_string()))?;
    Ok(path)
}

/// Reads a stored seed, refusing a file anyone but its owner can read.
pub fn load_seed(path: &Path) -> Result<[u8; 32], GatewayError> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path).map_err(|e| GatewayError::Config(format!("{}: {e}", path.display())))?;
    if meta.permissions().mode() & 0o077 != 0 {
        return Err(GatewayError::Config(format!("{} can be read by others; refusing to use it", path.display())));
    }
    let text = std::fs::read_to_string(path).map_err(|e| GatewayError::Config(e.to_string()))?;
    let t = text.trim();
    if t.len() != 64 {
        return Err(GatewayError::Config(format!("{} is not a device key", path.display())));
    }
    let mut s = [0u8; 32];
    for (i, b) in s.iter_mut().enumerate() {
        *b = u8::from_str_radix(&t[2 * i..2 * i + 2], 16).map_err(|_| GatewayError::Config(format!("{} is not a device key", path.display())))?;
    }
    Ok(s)
}

/// The non-secret record beside the keys: `agents.json`, one entry per agent and portal.
pub fn remember(dir: &Path, portal: &Portal, t: &AccessToken) -> Result<(), GatewayError> {
    let path = dir.join("agents.json");
    let mut all: Value = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
    all[format!("{}:{}", portal.authority, t.agent_username)] = json!({ "agent_id": t.agent_id, "username": t.agent_username, "binding_id": t.binding_id, "portal": portal.origin, "key": key_name(portal, &t.agent_id) });
    std::fs::write(&path, serde_json::to_string_pretty(&all).expect("json")).map_err(|e| GatewayError::Config(e.to_string()))
}

/// The stored sign-in of `agent` (`@name` or `name`) at `portal_url`: its key and record.
/// Returns the client and the agent's name without the `@`.
///
/// # Errors
/// [`GatewayError::Config`] when the agent is not signed in on this device
/// (the message gives the command to run) or the key file is unreadable or
/// readable by others.
pub fn load(portal_url: &str, agent: &str) -> Result<(SignIn, String), GatewayError> {
    let portal = Portal::parse(portal_url)?;
    let dir = key_dir();
    let name = agent.trim_start_matches('@');
    let all: Value = std::fs::read_to_string(dir.join("agents.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
    let rec = &all[format!("{}:{name}", portal.authority)];
    let agent_id = rec["agent_id"].as_str().ok_or_else(|| {
        GatewayError::Config(format!("@{name} is not signed in on {}: run `robutler-device agent login @{name} --portal {}`", portal.authority, portal.origin))
    })?;
    let seed = load_seed(&dir.join(key_name(&portal, agent_id)))?;
    Ok((SignIn::new(portal, DeviceKey::from_seed(&seed)), name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portal_authority() {
        assert_eq!(Portal::parse("https://Robutler.ai/").unwrap().authority, "robutler.ai");
        assert_eq!(Portal::parse("https://h:443/ws").unwrap().authority, "h");
        assert_eq!(Portal::parse("http://127.0.0.1:8080").unwrap().authority, "127.0.0.1:8080");
        assert_eq!(Portal::parse("wss://x.ts.net/ws").unwrap().origin, "https://x.ts.net");
    }

    #[test]
    fn http_dates() {
        assert_eq!(parse_http_date("Thu, 26 Nov 2026 00:00:00 GMT"), Some(1795651200));
        assert_eq!(parse_http_date("Thu, 01 Jan 1970 00:00:10 GMT"), Some(10));
        assert_eq!(parse_http_date("nonsense"), None);
    }

    #[test]
    fn seed_storage_is_private_and_never_overwritten() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("wa-keys-{}", std::process::id()));
        let p = store_seed(&dir, "agent-device:h:a1", &[7; 32]).unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        assert_eq!(load_seed(&p).unwrap(), [7; 32]);
        assert!(store_seed(&dir, "agent-device:h:a1", &[8; 32]).is_err(), "an existing key is never overwritten");
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_seed(&p).is_err());
        std::fs::remove_file(&p).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
