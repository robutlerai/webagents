//! The host gateway: puts devices on the agent network as one agent.
//!
//! A gateway is the default home of a device agent: the part itself only
//! needs a bus (SPI, I2C, CAN, UART), and the gateway, a Linux host next to
//! it, holds the platform connection, the model loop and the envelope. A part
//! with a network of its own, or one whose host is not trusted, carries the
//! envelope itself (the same engine, built `no_std`).
//!
//! ```text
//! platform /ws  <--UAMP-->  gateway  --tool calls-->  Engine (limits, rules)  --bus-->  part
//!                              |
//!               platform /llm  <-- model calls, paid by the turn's token
//! ```
//!
//! Configuration is a TOML file ([`GatewayConfig`]). The per-agent key is read
//! from the environment variable the file names, never from the file.
//!
//! # Running a gateway
//!
//! [`run`] is the whole gateway: it loads the file, opens every device, signs
//! in, and serves turns until the shutdown channel turns true. It needs the
//! platform, so this example compiles but is not run as a test:
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//! use webagents::gateway::{self, PortalEvent};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), gateway::GatewayError> {
//!     // Sending `true` stops the gateway: running turns are cancelled and the
//!     // socket is closed. A real program sends it from its signal handler;
//!     // this one stops after an hour.
//!     let (stop, stopped) = tokio::sync::watch::channel(false);
//!     tokio::spawn(async move {
//!         tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
//!         let _ = stop.send(true);
//!     });
//!     // Optional: every connection event, for a log.
//!     let log = Arc::new(|e: PortalEvent| eprintln!("{e:?}"));
//!     gateway::run(Path::new("gateway.toml"), stopped, Some(log)).await
//! }
//! ```
//!
//! The pieces are public for a program that needs more than the file offers:
//! [`GatewayConfig::load`] and [`build_devices`] give the [`Devices`],
//! [`build_model`] the [`ModelBackend`], and [`run_portal`] serves them. A
//! backend of your own (a local model server, a scripted model in a test)
//! implements [`ModelBackend`] and is passed to [`run_portal`] in place of
//! the configured one.
//!
//! # What is fixed
//!
//! - One agent per gateway process, with every configured device's tools.
//! - Tool calls run with [`CallCtx::confirmed`](crate::regspec::CallCtx)
//!   false: the gateway does not take confirmations in chat, so a field marked
//!   `confirm` is refused and the model is told to say so.
//! - A turn's history is reduced to plain text turns before the model sees
//!   it, so a relayed conversation cannot inject a tool result.

pub mod accel;
pub mod device;
pub mod llm;
pub mod portal;
pub mod signin;
pub mod turn;

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use tokio::sync::watch;

pub use device::{Devices, ToolDevice, ToolOutcome};
pub use llm::{ModelBackend, ModelReply, ModelRequest, PortalLlm, ToolCall};
pub use portal::{run_portal, PortalConfig, PortalEvent};

/// Why the gateway could not start or had to stop. Each variant carries a
/// sentence for the operator; `Display` prefixes it with where it happened.
///
/// A failure inside a turn is not one of these: it is reported on the turn's
/// own session as a `response.error`, and the gateway keeps running.
#[derive(Debug)]
pub enum GatewayError {
    /// The configuration file, a register spec, the per-agent key or a stored
    /// device sign-in is missing or wrong. Nothing was connected.
    Config(String),
    /// The agent socket or the sign-in API: the platform refused, did not
    /// answer, or the reconnect budget was spent.
    Portal(String),
    /// A model call failed (the `/llm` socket or the accelerator link).
    Model(String),
    /// A device could not be opened, or its signed tool description did not verify.
    Transport(String),
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GatewayError::Config(m) => write!(f, "configuration: {m}"),
            GatewayError::Portal(m) => write!(f, "agent socket: {m}"),
            GatewayError::Model(m) => write!(f, "model: {m}"),
            GatewayError::Transport(m) => write!(f, "transport: {m}"),
        }
    }
}

impl std::error::Error for GatewayError {}

/// The gateway's TOML file.
///
/// ```toml
/// [portal]
/// url = "wss://robutler.ai/ws"
/// agent = "bench-gateway"
/// token_env = "WEBAGENTS_AGENT_TOKEN"   # the key is read from here, never written in this file
///
/// [llm]                                  # optional; defaults to the /llm next to the portal URL
/// model = "auto/balanced"
/// # or a model on the board's own accelerator (no platform model calls):
/// # kind = "accelerator", link = "127.0.0.1:7100", tokenizer = "qwen3/tokenizer.json"
///
/// [[device]]
/// spec = "pwm1.regspec.json"
/// transport = { kind = "uart", path = "/dev/ttyUSB0", baud = 115200 }
///
/// [[device]]
/// spec = "sensor.regspec.json"
/// transport = { kind = "i2c", bus = "/dev/i2c-1", address = 0x48 }
/// ```
///
/// An unknown key in any table is an error, so a misspelt setting is never
/// silently ignored. Relative paths (`spec`, `tokenizer`) are resolved against
/// the directory of the configuration file.
///
/// Loading a configuration and opening its devices needs no network. With a
/// simulated part the whole tool path runs in a test:
///
/// ```
/// use webagents::gateway::{build_devices, GatewayConfig};
/// use webagents::regspec::CallCtx;
///
/// let dir = std::env::temp_dir().join(format!("webagents-doc-{}", std::process::id()));
/// std::fs::create_dir_all(&dir)?;
/// std::fs::write(dir.join("pwm1.regspec.json"), include_str!("../../../../fixtures/pwm1.regspec.json"))?;
/// std::fs::write(
///     dir.join("gateway.toml"),
///     r#"
/// [portal]
/// url = "wss://robutler.ai/ws"
/// agent = "pwm-bench"
///
/// [[device]]
/// spec = "pwm1.regspec.json"
/// transport = { kind = "sim" }
/// "#,
/// )?;
///
/// let (cfg, base) = GatewayConfig::load(&dir.join("gateway.toml"))?;
/// assert_eq!(cfg.portal.token_env, "WEBAGENTS_AGENT_TOKEN");
/// let devices = build_devices(&cfg, &base)?;
/// assert_eq!(devices.names(), [("pwm1".to_string(), "example three-phase PWM driver".to_string())]);
/// // 12 registers can be read and 8 of them written: 20 tools.
/// assert_eq!(devices.tools().len(), 20);
///
/// // A tool call goes through the part's envelope, whatever asked for it.
/// let ctx = CallCtx { now_ms: 0, confirmed: false };
/// let ok = devices.call("pwm1_set_deadtime", r#"{"deadtime": 2}"#, &ctx);
/// assert!(!ok.is_error && ok.json.contains(r#""applied":{"DEADTIME":2}"#));
/// let refused = devices.call("pwm1_set_deadtime", r#"{"deadtime": 0.1}"#, &ctx);
/// assert!(refused.is_error && refused.json.contains(r#""error":"out_of_range""#));
/// # std::fs::remove_file(dir.join("gateway.toml"))?;
/// # std::fs::remove_file(dir.join("pwm1.regspec.json"))?;
/// # std::fs::remove_dir(&dir)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    /// `[portal]`: where the platform is and which agent this gateway is. Required.
    pub portal: PortalSection,
    /// `[llm]`: where the model runs. Optional; the default is the platform's
    /// `/llm` socket next to the portal URL, with the model the platform chooses.
    #[serde(default)]
    pub llm: LlmSection,
    /// `[[device]]`, one per part. At least one is required; all of them become
    /// tools of the one agent, and two devices may not share a name.
    #[serde(default, rename = "device")]
    pub devices: Vec<DeviceSection>,
}

/// The `[portal]` table of a gateway configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortalSection {
    /// The platform's agent socket: `wss://host/ws`. `https://host` and a bare
    /// origin are accepted and resolved to it.
    pub url: String,
    /// The agent's name on the platform (with `auth = "device"`, the name it
    /// was signed in as, with or without the `@`).
    pub agent: String,
    /// The NAME of the environment variable that holds the per-agent key
    /// (default `WEBAGENTS_AGENT_TOKEN`). The key itself never goes in the file.
    /// Not used with `auth = "device"`.
    #[serde(default = "default_token_env")]
    pub token_env: String,
    /// `"device"`: use this agent's device sign-in (`robutler-device agent login`) instead of a token.
    #[serde(default)]
    pub auth: Option<String>,
    /// Model calls per turn before the turn is stopped (default 8).
    #[serde(default)]
    pub max_rounds: Option<usize>,
    /// Consecutive failed connection attempts before giving up (default 10).
    #[serde(default)]
    pub max_reconnect_attempts: Option<u32>,
    /// Seconds between attempts (default 5).
    #[serde(default)]
    pub reconnect_delay_s: Option<u64>,
}

fn default_token_env() -> String {
    "WEBAGENTS_AGENT_TOKEN".into()
}

/// The `[llm]` table of a gateway configuration. Every key is optional.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmSection {
    /// `"portal"` (the default: the platform's /llm socket) or `"accelerator"` (the board's accelerator).
    #[serde(default)]
    pub kind: Option<String>,
    /// Portal backend: the model socket URL. Default: the portal URL with its
    /// last path segment replaced by `llm`.
    #[serde(default)]
    pub url: Option<String>,
    /// Portal backend: the model to ask for (`auto/balanced`). Default: the
    /// platform chooses.
    #[serde(default)]
    pub model: Option<String>,
    /// Accelerator: the accelerator link, `host:port`.
    #[serde(default)]
    pub link: Option<String>,
    /// Accelerator: the model's `tokenizer.json`, relative to the config file.
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    /// Accelerator: the most tokens one model call generates (default 512).
    #[serde(default)]
    pub max_new: Option<u32>,
    /// Accelerator: the longest wait for the next token, in seconds (default 600). A
    /// software model of the accelerator needs far more than silicon for a long prompt.
    #[serde(default)]
    pub token_timeout_s: Option<u64>,
}

/// One `[[device]]` table of a gateway configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceSection {
    /// Path to a `webagents.regspec/1` file, relative to the config file.
    pub spec: PathBuf,
    /// How the part's registers are reached: an inline table with a `kind`.
    pub transport: Transport,
    /// The device's own public key (Ed25519 JWK `x`, base64url) when the device holds
    /// one: its signed tool description is then verified against it
    /// before any of its tools are offered. Without it the tools are the gateway's own.
    #[serde(default)]
    pub device_key: Option<String>,
}

/// How the gateway reaches a device's registers.
///
/// In the file it is an inline table whose `kind` is the variant in lower
/// case: `{ kind = "uart", path = "/dev/ttyUSB0" }`. `sim` and `tcp` work on
/// any host; `i2c`, `spi`, `uart` and `can` need the `linux` feature.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Transport {
    /// An in-memory copy of the part at its reset values (no hardware).
    Sim,
    /// Linux i2c-dev; address and data layout from the spec's bus section unless given here.
    /// The spec's `bus` must be of kind `i2c`.
    I2c {
        /// The adapter's device node, such as `/dev/i2c-1`.
        bus: String,
        /// The target's 7-bit address, when it differs from the spec's
        /// (two parts of one type on one bus).
        address: Option<u8>,
    },
    /// Linux spidev. The spec's `bus` must be of kind `spi`; bit 7 of the
    /// first address byte marks a read. Mode and clock speed are the device
    /// node's current settings.
    Spi {
        /// The device node, such as `/dev/spidev0.0` (bus 0, chip select 0).
        path: String,
    },
    /// A serial port (or PTY) speaking the register line protocol ([`crate::transport::uart`]).
    Uart {
        /// The port, such as `/dev/ttyUSB0`, `/dev/serial0` or a PTY path.
        path: String,
        /// Baud rate (default 115200); 8 data bits, no parity, 1 stop bit.
        baud: Option<u32>,
        /// Answer timeout per request (default 200 ms).
        timeout_ms: Option<u64>,
    },
    /// A serial line carried over TCP (a terminal server, `ser2net`, or a
    /// simulated chip's UART bridge), same register line protocol.
    Tcp {
        /// `host:port` to connect to.
        addr: String,
        /// Answer timeout per request (default 2000 ms; a simulated chip may run slower than real time).
        timeout_ms: Option<u64>,
    },
    /// SocketCAN, register frames to a node id (`transport::can`, feature `linux`).
    Can {
        /// The SocketCAN interface, such as `can0` or `vcan0`.
        iface: String,
        /// The part's node id, 0 to 127. Requests go to CAN id `0x600 + node`,
        /// answers come from `0x580 + node`.
        node: u8,
    },
}

impl GatewayConfig {
    /// Reads and parses the file at `path`. Returns the configuration and the
    /// directory relative paths in it are resolved against (the file's own).
    ///
    /// # Errors
    /// [`GatewayError::Config`] when the file cannot be read, is not valid
    /// TOML for this structure (the message carries the line), or has no
    /// `[[device]]`. Nothing is opened: see [`build_devices`].
    pub fn load(path: &Path) -> Result<(GatewayConfig, PathBuf), GatewayError> {
        let text = std::fs::read_to_string(path).map_err(|e| GatewayError::Config(format!("cannot read {}: {e}", path.display())))?;
        let cfg: GatewayConfig = toml::from_str(&text).map_err(|e| GatewayError::Config(format!("{}: {e}", path.display())))?;
        if cfg.devices.is_empty() {
            return Err(GatewayError::Config("no [[device]] in the configuration".into()));
        }
        Ok((cfg, path.parent().map(Path::to_path_buf).unwrap_or_default()))
    }
}

/// Builds every device in the configuration.
///
/// For each `[[device]]`: load and validate its spec, open its transport, and
/// when it names a `device_key`, fetch the part's signed tool description and
/// verify it before the device is added. `base` is the directory relative
/// spec paths are resolved against, as returned by [`GatewayConfig::load`].
///
/// # Errors
/// [`GatewayError::Config`] for an unreadable or invalid spec (with every
/// problem in it) or two devices of one name; [`GatewayError::Transport`] when
/// a port, bus or interface cannot be opened or a signed description does not
/// verify. The first failure stops the build: a gateway does not start with
/// part of its hardware.
pub fn build_devices(cfg: &GatewayConfig, base: &Path) -> Result<Devices, GatewayError> {
    let mut devices = Devices::new();
    for d in &cfg.devices {
        let path = base.join(&d.spec);
        let src = std::fs::read_to_string(&path).map_err(|e| GatewayError::Config(format!("cannot read {}: {e}", path.display())))?;
        let dev = crate::regspec::load::load_json(&src).map_err(|e| GatewayError::Config(format!("{}: {e}", path.display())))?;
        if let Some(old) = crate::regspec::load::deprecated_schema(&src) {
            eprintln!("[webagents-gateway] {}: schema \"{old}\" is deprecated; change it to \"{}\"", path.display(), crate::regspec::SCHEMA);
        }
        let device = crate::transport::open_verified(dev, &d.transport, d.device_key.as_deref())?;
        devices.add(device).map_err(GatewayError::Config)?;
    }
    Ok(devices)
}

/// Selects the TLS crypto provider (ring) once per process. rustls refuses to
/// guess when its features do not pick exactly one, and panics at the first
/// TLS handshake instead, so every entry point that may open `wss://` calls this.
pub fn init_tls() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Loads a configuration and runs the gateway until `shutdown` turns true.
///
/// `events`, when given, is called with every [`PortalEvent`] (connection
/// attempts, refusals, turns starting), from the gateway's own task; keep it
/// quick. See the [module documentation](self) for a complete program.
///
/// # Errors
/// What [`GatewayConfig::load`], [`build_devices`] and [`build_model`]
/// return, before anything is connected; then [`GatewayError::Config`] for a
/// missing or account-wide key or a device that is not signed in, and
/// [`GatewayError::Portal`] once the reconnect budget is spent.
pub async fn run(config: &Path, shutdown: watch::Receiver<bool>, events: Option<portal::EventSink>) -> Result<(), GatewayError> {
    let (cfg, base) = GatewayConfig::load(config)?;
    let devices = Arc::new(build_devices(&cfg, &base)?);
    let mut pc = match cfg.portal.auth.as_deref() {
        Some("device") => {
            let (mut si, name) = signin::load(&cfg.portal.url, &cfg.portal.agent)?;
            let _ = si.sync_clock();
            let mut pc = PortalConfig::new(&cfg.portal.url, &name, "");
            pc.device = Some(Arc::new(si));
            pc
        }
        Some(other) => return Err(GatewayError::Config(format!("auth = \"{other}\" is not known (\"device\", or omit it for a token)"))),
        None => PortalConfig::new(&cfg.portal.url, &cfg.portal.agent, &std::env::var(&cfg.portal.token_env).unwrap_or_default()),
    };
    if let Some(n) = cfg.portal.max_rounds {
        pc.limits.max_rounds = n;
    }
    if let Some(n) = cfg.portal.max_reconnect_attempts {
        pc.max_reconnect_attempts = n;
    }
    if let Some(s) = cfg.portal.reconnect_delay_s {
        pc.reconnect_delay = std::time::Duration::from_secs(s);
    }
    let model = build_model(&cfg, &base, &pc.url)?;
    run_portal(pc, devices, model, shutdown, events).await
}

/// The model backend the `[llm]` section names.
///
/// `portal_url` is the resolved agent socket URL; the platform backend's
/// default model socket is derived from it. `base` is the directory the
/// accelerator's `tokenizer` path is resolved against.
///
/// # Errors
/// [`GatewayError::Config`] for an unknown `kind`, an accelerator backend
/// without `link` or `tokenizer`, or a tokenizer file that cannot be loaded.
pub fn build_model(cfg: &GatewayConfig, base: &Path, portal_url: &str) -> Result<Arc<dyn ModelBackend>, GatewayError> {
    match cfg.llm.kind.as_deref() {
        None | Some("portal") => {
            let mut llm = PortalLlm::new(cfg.llm.url.clone().unwrap_or_else(|| portal::llm_url_for(portal_url)));
            llm.model = cfg.llm.model.clone();
            Ok(Arc::new(llm))
        }
        Some("accelerator") => {
            let link = cfg.llm.link.clone().ok_or_else(|| GatewayError::Config("[llm] kind = \"accelerator\" needs link = \"host:port\"".into()))?;
            let tok = cfg
                .llm
                .tokenizer
                .as_ref()
                .ok_or_else(|| GatewayError::Config("[llm] kind = \"accelerator\" needs tokenizer = \"path/tokenizer.json\"".into()))?;
            let tokenizer = crate::model::bpe::Tokenizer::from_file(&base.join(tok)).map_err(|e| GatewayError::Config(e.to_string()))?;
            let mut b = accel::AccelBackend::new(link, tokenizer, cfg.llm.max_new.unwrap_or(512))?;
            if let Some(t) = cfg.llm.token_timeout_s {
                b.token_timeout = std::time::Duration::from_secs(t);
            }
            Ok(Arc::new(b))
        }
        Some(k) => Err(GatewayError::Config(format!("[llm] kind = \"{k}\" is not known (\"portal\" or \"accelerator\")"))),
    }
}
