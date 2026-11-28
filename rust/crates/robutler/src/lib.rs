//! # robutler
//!
//! Robutler's defaults for [`webagents`] device agents: where the platform is,
//! and how an agent finds its credential. Everything else is in `webagents`,
//! which this crate re-exports, so `robutler` is the one dependency a device
//! agent for the Robutler platform needs.
//!
//! ```
//! // An explicit value wins over the environment and the default; an origin
//! // becomes the agent socket URL.
//! assert_eq!(robutler::resolve_portal_ws_url(Some("https://robutler.ai")), "wss://robutler.ai/ws");
//! assert_eq!(robutler::DEFAULT_PORTAL_WS, "wss://robutler.ai/ws");
//!
//! // `webagents` is re-exported whole.
//! use robutler::webagents::regspec::SCHEMA;
//! assert_eq!(SCHEMA, "webagents.regspec/1");
//! ```
//!
//! # The `robutler-device` command
//!
//! `cargo install robutler` installs one binary, `robutler-device`. It has its
//! own name so that it never shadows the `robutler` command of the npm and
//! PyPI packages.
//!
//! ```text
//! robutler-device regspec check <spec.json>   validate a register spec and list every problem
//! robutler-device regspec tools <spec.json>   print the tool list a model sees (JSON)
//! robutler-device regspec gen   <spec.json>   print the typed Rust module for the spec
//! robutler-device regspec try   <spec.json> <tool> <args-json>...
//!                                             run tool calls against an in-memory copy of the part
//! robutler-device gateway check <config.toml> load the gateway configuration and open every device, then exit
//! robutler-device gateway run   <config.toml> put the configured devices on the platform as one agent
//! robutler-device agent login  <@agent> --portal <url> [--name <device name>]
//!                                             sign this device in as the agent; the key stays on this device
//! robutler-device agent logout <@agent> --portal <url>
//!                                             sign this device out
//! robutler-device twin <spec.json> uart --pty | uart <path> | can <iface> <node>
//!                                             serve a simulated part on a PTY, a serial port or SocketCAN
//! robutler-device twin esp-at --listen <addr> an ESP-AT network co-processor on a TCP serial line
//! ```
//!
//! The `regspec` verbs work on every host. The `gateway`, `agent` and `twin`
//! verbs are built on Linux only: they use serial ports, SocketCAN, i2c-dev
//! and spidev. An unknown command prints the usage text and exits with 2.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub use webagents;

/// The platform's agent socket. An agent opens it, authenticates, and serves
/// turns over it (UAMP, the unified agent messaging protocol).
pub const DEFAULT_PORTAL_WS: &str = "wss://robutler.ai/ws";

/// Environment variables read for the portal URL, in order.
pub const PORTAL_URL_ENV: [&str; 2] = ["WEBAGENTS_PORTAL_URL", "PORTAL_WS_URL"];

/// Environment variable holding a per-agent key (minted for one agent; an
/// account-wide key is refused because the platform would never route a turn
/// to it).
pub const AGENT_TOKEN_ENV: &str = "WEBAGENTS_AGENT_TOKEN";

/// Resolve the agent socket URL the way the TypeScript and Python SDKs do:
/// an explicit value, then the environment, then the default; `http(s)` is
/// turned into `ws(s)` and `/ws` is appended to a bare origin.
///
/// The environment variables are those in [`PORTAL_URL_ENV`], first one set
/// and not blank; the default is [`DEFAULT_PORTAL_WS`]. A URL that already
/// has a path is kept as it is.
pub fn resolve_portal_ws_url(configured: Option<&str>) -> String {
    let env = || PORTAL_URL_ENV.iter().find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()));
    let raw = configured.map(str::to_string).or_else(env).unwrap_or_else(|| DEFAULT_PORTAL_WS.to_string());
    normalize_ws_url(raw.trim())
}

fn normalize_ws_url(raw: &str) -> String {
    let mut s = if let Some(r) = raw.strip_prefix("https://") {
        format!("wss://{r}")
    } else if let Some(r) = raw.strip_prefix("http://") {
        format!("ws://{r}")
    } else {
        raw.to_string()
    };
    let rest = s.split_once("://").map_or(s.as_str(), |(_, r)| r);
    if !rest.contains('/') {
        while s.ends_with('/') {
            s.pop();
        }
        s.push_str("/ws");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::normalize_ws_url as n;

    #[test]
    fn urls_match_the_other_sdks() {
        assert_eq!(n("https://robutler.ai"), "wss://robutler.ai/ws");
        assert_eq!(n("http://localhost:3000"), "ws://localhost:3000/ws");
        assert_eq!(n("wss://robutler.ai/ws"), "wss://robutler.ai/ws");
        assert_eq!(n("https://dev.robutler.ai/custom"), "wss://dev.robutler.ai/custom");
    }
}
