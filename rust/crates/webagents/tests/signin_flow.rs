//! The device flow against a mock of the platform's sign-in API:
//! clock sync from `Date`, the authorize request signed by the key it binds,
//! token polling through `authorization_pending` and `slow_down` to the one
//! token, refresh, the WebSocket ticket, sign-out (and "already signed out" as
//! success). The mock checks
//! every request the way the platform does: it rebuilds the signature base from
//! the headers it received and verifies it with the key bound at authorize, refuses a
//! reused nonce, a body without `content-digest`, a wrong digest, a window over
//! 3600 s, and any `signature-agent`.
#![cfg(feature = "gateway")]

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use webagents::auth::{b64url_decode, DeviceKey};
use webagents::gateway::signin::{Portal, SignIn};

#[derive(Default)]
struct Platform {
    bound_x: Option<String>,
    nonces: HashSet<String>,
    polls: u32,
    signed_out: bool,
    log: Vec<String>,
    refusals: Vec<String>,
}

fn b64(s: &str) -> Vec<u8> {
    // Standard base64 with padding (Signature, Content-Digest).
    let t = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (mut acc, mut bits, mut out) = (0u32, 0u32, Vec::new());
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        let v = t.iter().position(|x| *x == c).unwrap() as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

/// Verifies a request as the platform does. Returns the keyid's x on success, or the error_code.
fn verify(p: &mut Platform, method: &str, target: &str, h: &[(String, String)], body: &[u8], key_x: Option<&str>) -> Result<(), &'static str> {
    let get = |n: &str| h.iter().find(|(k, _)| k == n).map(|(_, v)| v.as_str());
    let (Some(si), Some(sig)) = (get("signature-input"), get("signature")) else { return Err("missing_credential") };
    if get("signature-agent").is_some() || si.contains("signature-agent") {
        return Err("signature_coverage_insufficient");
    }
    let params = si.strip_prefix("sig1=").ok_or("signature_malformed")?;
    let comps = &params[1..params.find(')').ok_or("signature_malformed")?];
    let field = |name: &str| params.split(';').find_map(|kv| kv.strip_prefix(&format!("{name}="))).map(|v| v.trim_matches('"').to_string());
    let (created, expires) = (
        field("created").ok_or("signature_params_invalid")?.parse::<u64>().unwrap(),
        field("expires").ok_or("signature_params_invalid")?.parse::<u64>().unwrap(),
    );
    if expires < created || expires - created > 3600 || field("tag").as_deref() != Some("web-bot-auth") || field("alg").as_deref() != Some("ed25519") {
        return Err("signature_params_invalid");
    }
    let nonce = field("nonce").ok_or("signature_params_invalid")?;
    if !p.nonces.insert(nonce) {
        return Err("signature_replayed");
    }
    let has = |c: &str| comps.split(' ').any(|x| x == format!("\"{c}\""));
    for c in ["@method", "@authority", "@path", "@query"] {
        if !has(c) {
            return Err("signature_coverage_insufficient");
        }
    }
    if !body.is_empty() || get("content-digest").is_some() {
        if !has("content-digest") {
            return Err("signature_coverage_insufficient");
        }
        let d = Sha256::digest(body);
        let want = get("content-digest").ok_or("signature_coverage_insufficient")?;
        let got = want.strip_prefix("sha-256=:").and_then(|x| x.strip_suffix(':')).ok_or("content_digest_mismatch")?;
        if b64(got) != d.as_slice() {
            return Err("content_digest_mismatch");
        }
    }
    if get("authorization").is_some() && !has("authorization") {
        return Err("signature_coverage_insufficient");
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut base = format!("\"@method\": {method}\n\"@authority\": {}\n\"@path\": {path}\n\"@query\": ?{query}\n", get("host").unwrap_or(""));
    if has("content-digest") {
        base += &format!("\"content-digest\": {}\n", get("content-digest").unwrap());
    }
    if has("authorization") {
        base += &format!("\"authorization\": {}\n", get("authorization").unwrap());
    }
    base += &format!("\"@signature-params\": {params}");
    let x = key_x.ok_or("signature_key_unknown")?;
    let keyid = field("keyid").ok_or("signature_params_invalid")?;
    let mut thumb = [0u8; 32];
    b64url_decode(&keyid, &mut thumb).ok_or("signature_params_invalid")?;
    if thumb != webagents::auth::thumbprint(x) {
        return Err("signature_key_unknown");
    }
    let mut pk = [0u8; 32];
    b64url_decode(x, &mut pk).ok_or("signature_params_invalid")?;
    let vk = VerifyingKey::from_bytes(&pk).map_err(|_| "signature_invalid")?;
    let s = b64(sig.strip_prefix("sig1=:").and_then(|x| x.strip_suffix(':')).ok_or("signature_malformed")?);
    let s = Signature::from_slice(&s).map_err(|_| "signature_malformed")?;
    vk.verify(base.as_bytes(), &s).map_err(|_| "signature_invalid")
}

fn serve(listener: TcpListener, p: Arc<Mutex<Platform>>) {
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        let mut r = BufReader::new(s.try_clone().unwrap());
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
        let mut headers = Vec::new();
        loop {
            let mut l = String::new();
            r.read_line(&mut l).unwrap();
            let l = l.trim_end();
            if l.is_empty() {
                break;
            }
            let (k, v) = l.split_once(':').unwrap();
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
        let len: usize = headers.iter().find(|(k, _)| k == "content-length").map(|(_, v)| v.parse().unwrap()).unwrap_or(0);
        let mut body = vec![0u8; len];
        r.read_exact(&mut body).unwrap();
        let mut pl = p.lock().unwrap();
        pl.log.push(format!("{method} {target}"));
        let (status, reply): (u16, Value) = if method == "GET" && target == "/" {
            (200, json!({}))
        } else {
            let key = if target == "/api/agents/device/authorize" {
                serde_json::from_slice::<Value>(&body).ok().and_then(|v| v["public_key"]["x"].as_str().map(str::to_string))
            } else {
                pl.bound_x.clone()
            };
            match verify(&mut pl, &method, &target, &headers, &body, key.as_deref()) {
                Err(code) => {
                    pl.refusals.push(code.into());
                    (401, json!({ "error": "Unauthorized", "error_code": code, "error_description": "refused by the mock" }))
                }
                Ok(()) if pl.signed_out && target != "/api/agents/device/authorize" => {
                    (401, json!({ "error": "Unauthorized", "error_code": "device_signed_out" }))
                }
                Ok(()) => match target.as_str() {
                    "/api/agents/device/authorize" => {
                        pl.bound_x = key;
                        (
                            200,
                            json!({ "device_code": "dc_1", "user_code": "BCDF-GHJK", "verification_uri": "https://portal.test/device", "verification_uri_complete": "https://portal.test/device?code=BCDF-GHJK", "expires_in": 600, "interval": 5, "fingerprint": "ignored", "agent": { "id": "a-1", "username": "weather-station" } }),
                        )
                    }
                    "/api/agents/device/token" => {
                        pl.polls += 1;
                        match pl.polls {
                            1 => (400, json!({ "error": "authorization_pending", "interval": 5 })),
                            2 => (400, json!({ "error": "slow_down", "interval": 10 })),
                            3 => (400, json!({ "error": "authorization_pending", "interval": 10 })),
                            4 => (
                                200,
                                json!({ "access_token": "jwt.at.1", "token_type": "Bearer", "expires_in": 3600, "agent": { "id": "a-1", "username": "weather-station" }, "binding_id": "b-1" }),
                            ),
                            _ => (400, json!({ "error": "expired_token" })),
                        }
                    }
                    "/api/agents/device/refresh" => (
                        200,
                        json!({ "access_token": "jwt.at.2", "token_type": "Bearer", "expires_in": 3600, "agent": { "id": "a-1", "username": "weather-station" }, "binding_id": "b-1" }),
                    ),
                    "/api/agents/device/ws-ticket" => (200, json!({ "ticket": "wst_ticket", "expires_in": 60 })),
                    "/api/agents/a-1/devices/b-1" if method == "DELETE" => {
                        pl.signed_out = true;
                        (200, json!({ "ok": true, "signed_out": true }))
                    }
                    "/api/discovery?q=weather" => (200, json!({ "ok": true, "bearer": headers.iter().any(|(k, _)| k == "authorization") })),
                    _ => (404, json!({ "error": "not_found" })),
                },
            }
        };
        drop(pl);
        let b = reply.to_string();
        let _ = write!(s, "HTTP/1.1 {status} X\r\nDate: Thu, 26 Nov 2026 00:00:00 GMT\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
    }
}

#[test]
fn device_flow_end_to_end() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let p: Arc<Mutex<Platform>> = Arc::default();
    let p2 = p.clone();
    std::thread::spawn(move || serve(l, p2));

    let portal = Portal::parse(&format!("http://{addr}")).unwrap();
    let mut si = SignIn::new(portal, DeviceKey::from_seed(&[42; 32]));
    let off = si.sync_clock().unwrap();
    assert!(
        (off + (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64 - 1795651200)).abs() <= 2,
        "offset taken from Date"
    );

    let pending = si.authorize("@weather-station", "bench Pi").unwrap();
    assert_eq!(pending.user_code, "BCDF-GHJK");
    assert_eq!(pending.agent_id, "a-1");
    assert_eq!(pending.fingerprint, si.key.fingerprint().as_str(), "the fingerprint shown is computed on the device");

    let mut waits = Vec::new();
    let tok = si.poll(&pending, &mut |d| waits.push(d.as_secs())).unwrap();
    assert_eq!(tok.access_token, "jwt.at.1");
    assert_eq!(tok.binding_id, "b-1");
    assert_eq!(waits, vec![5, 10, 10], "polls at the interval, and slow_down's interval replaces it");

    assert_eq!(si.refresh().unwrap().access_token, "jwt.at.2");
    assert_eq!(si.ws_ticket().unwrap(), "wst_ticket");
    let with_bearer = si.signed("GET", "/api/discovery?q=weather", None, Some("jwt.at.2")).unwrap();
    assert_eq!((with_bearer.status, with_bearer.body["bearer"].clone()), (200, json!(true)), "a bearer travels only beside a signature covering it");
    si.logout("a-1", "b-1").unwrap();
    si.logout("a-1", "b-1").unwrap(); // already signed out: success
    assert!(si.ws_ticket().unwrap_err().to_string().contains("device_signed_out"));

    let pl = p.lock().unwrap();
    assert!(pl.refusals.is_empty() || pl.refusals == ["device_signed_out"], "every request verified: {:?}", pl.refusals);
    assert_eq!(pl.nonces.len(), pl.log.len() - 1, "one fresh nonce per signed request (the clock GET is unsigned)");
}

#[test]
fn a_different_key_cannot_use_the_binding() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let p: Arc<Mutex<Platform>> = Arc::default();
    let p2 = p.clone();
    std::thread::spawn(move || serve(l, p2));
    let portal = Portal::parse(&format!("http://{addr}")).unwrap();
    let mut good = SignIn::new(portal.clone(), DeviceKey::from_seed(&[1; 32]));
    good.sync_clock().unwrap();
    good.authorize("@weather-station", "a").unwrap();
    let mut other = SignIn::new(portal, DeviceKey::from_seed(&[2; 32]));
    other.clock_offset = good.clock_offset;
    let e = other.ws_ticket().unwrap_err().to_string();
    assert!(e.contains("signature_key_unknown"), "{e}");
}
