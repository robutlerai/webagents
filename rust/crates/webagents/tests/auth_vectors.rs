//! The shared agent sign-in vectors (`test-fixtures/agent-device/` at the
//! repository root, one set for every SDK in the repository): every case's
//! signature base and its three headers, reproduced byte for byte from the
//! case's key, request, created, expires and nonce; both test keys'
//! thumbprints and fingerprints; and the user-code rules.
#![cfg(feature = "auth")]

use serde_json::Value;
use webagents::auth::{normalize_user_code, DeviceKey, Request};

const VECTORS: &str = include_str!("../../../../test-fixtures/agent-device/sign-in-vectors.json");

fn seed(hex: &str) -> [u8; 32] {
    let mut s = [0u8; 32];
    for (i, b) in s.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    s
}

#[test]
fn keys_thumbprints_fingerprints() {
    let v: Value = serde_json::from_str(VECTORS).unwrap();
    for k in v["keys"].as_array().unwrap() {
        let key = DeviceKey::from_seed(&seed(k["private_seed_hex"].as_str().unwrap()));
        assert_eq!(key.public_x(), k["public_jwk"]["x"]);
        assert_eq!(key.keyid(), k["thumbprint"]);
        assert_eq!(key.fingerprint().as_str(), k["fingerprint"]);
    }
}

#[test]
fn user_codes() {
    let v: Value = serde_json::from_str(VECTORS).unwrap();
    let u = &v["user_codes"];
    let shown = u["shown"].as_str().unwrap().replace('-', "");
    for s in u["accepted_as_the_same"].as_array().unwrap() {
        assert_eq!(normalize_user_code(s.as_str().unwrap()).map(|c| String::from_utf8(c.to_vec()).unwrap()), Some(shown.clone()), "{s}");
    }
    for s in u["not_codes"].as_array().unwrap() {
        assert_eq!(normalize_user_code(s.as_str().unwrap()), None, "{s}");
    }
}

#[test]
fn every_case_byte_for_byte() {
    let v: Value = serde_json::from_str(VECTORS).unwrap();
    let keys = v["keys"].as_array().unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let k = keys.iter().find(|k| k["name"] == c["key"]).unwrap();
        let key = DeviceKey::from_seed(&seed(k["private_seed_hex"].as_str().unwrap()));
        assert_eq!(key.keyid(), c["keyid"], "{name}");
        let r = &c["request"];
        let url = r["url"].as_str().unwrap();
        let rest = url.split_once("://").unwrap().1;
        let (_, path_q) = rest.split_at(rest.find('/').unwrap());
        let (path, query) = match path_q.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (path_q, None),
        };
        let body = r["body"].as_str().map(str::as_bytes);
        let req = Request {
            method: r["method"].as_str().unwrap(),
            authority: r["headers"]["host"].as_str().unwrap(),
            path,
            query,
            body,
            authorization: r["headers"]["authorization"].as_str(),
        };
        let mut scratch = [0u8; 4096];
        let (h, n) = key.sign(&req, c["created"].as_u64().unwrap(), c["expires"].as_u64().unwrap(), c["nonce"].as_str().unwrap(), &mut scratch).unwrap();
        assert_eq!(std::str::from_utf8(&scratch[..n]).unwrap(), c["signature_base"].as_str().unwrap(), "{name}: signature base");
        let e = &c["expected"];
        assert_eq!(h.signature_input.as_str(), e["signature-input"], "{name}: Signature-Input");
        assert_eq!(h.signature.as_str(), e["signature"], "{name}: Signature");
        match e["content-digest"].as_str() {
            Some(d) => assert_eq!(h.content_digest.as_str(), d, "{name}: Content-Digest"),
            None => assert!(h.content_digest.as_str().is_empty(), "{name}: no body, no Content-Digest"),
        }
    }
}

// ---- the signed tool description, vectors made by Python `cryptography` ----

const TOOL_VECTORS: &str = include_str!("../../../../test-fixtures/agent-device/tool-description-vectors.json");

#[test]
fn tool_description_signatures_byte_for_byte() {
    let base: Value = serde_json::from_str(VECTORS).unwrap();
    let v: Value = serde_json::from_str(TOOL_VECTORS).unwrap();
    let k = base["keys"].as_array().unwrap().iter().find(|k| k["name"] == "device-1").unwrap();
    let key = DeviceKey::from_seed(&seed(k["private_seed_hex"].as_str().unwrap()));
    for c in v["cases"].as_array().unwrap() {
        let (device, created, tools) = (c["device"].as_str().unwrap(), c["created"].as_u64().unwrap(), c["tools"].as_str().unwrap().as_bytes());
        let mut msg = String::new();
        webagents::auth::write_tool_description_message(&mut msg, key.keyid(), device, created, tools).unwrap();
        assert_eq!(msg, c["message"].as_str().unwrap(), "{}: message", c["name"]);
        let sig = key.sign_tool_description(device, created, tools);
        assert_eq!(sig.as_str(), c["signature"].as_str().unwrap(), "{}: signature", c["name"]);
        assert!(webagents::auth::verify_tool_description(key.public_x(), device, created, tools, sig.as_str()));
        // A part that streams its description signs the same bytes from the running digest.
        let mut d = webagents::auth::ToolDigest::new();
        std::fmt::Write::write_str(&mut d, c["tools"].as_str().unwrap()).unwrap();
        let (n, sha) = d.finish();
        assert_eq!(n, tools.len());
        assert_eq!(key.sign_tool_description_sha256(device, created, &sha).as_str(), c["signature"].as_str().unwrap(), "{}: streamed", c["name"]);
    }
    let good = &v["cases"][0];
    let sig = good["signature"].as_str().unwrap();
    for t in v["tampered"]["variants"].as_array().unwrap() {
        let ok = webagents::auth::verify_tool_description(
            key.public_x(),
            t["device"].as_str().unwrap(),
            t["created"].as_u64().unwrap(),
            t["tools"].as_str().unwrap().as_bytes(),
            sig,
        );
        assert!(!ok, "must not verify: {}", t["what"]);
    }
    let wrong = v["wrong_key"]["public_x"].as_str().unwrap();
    assert!(
        !webagents::auth::verify_tool_description(wrong, "pwm1", good["created"].as_u64().unwrap(), good["tools"].as_str().unwrap().as_bytes(), sig),
        "the wrong key must not verify"
    );
    assert!(
        !webagents::auth::verify_tool_description(
            key.public_x(),
            "pwm1",
            good["created"].as_u64().unwrap(),
            good["tools"].as_str().unwrap().as_bytes(),
            "AAAA"
        ),
        "a malformed signature is refused"
    );
}
