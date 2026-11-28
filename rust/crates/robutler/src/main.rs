//! The `robutler-device` command. See the crate documentation for the verbs.

use std::process::ExitCode;

use robutler::webagents::bus::RegisterBus;
use robutler::webagents::regspec::{codegen, engine::write_tools, load, CallCtx, Engine};

const USAGE: &str = "usage:
  robutler-device regspec check <spec.json>   validate a register spec and list every problem
  robutler-device regspec tools <spec.json>   print the tool list a model sees (JSON)
  robutler-device regspec gen   <spec.json>   print the typed Rust module for the spec
  robutler-device regspec try   <spec.json> <tool> <args-json>...
                                       run tool calls in order against an in-memory copy of the
                                       part at its reset values (no hardware); prints each result
  robutler-device gateway check <config.toml> load the gateway configuration and open every device, then exit
  robutler-device gateway run   <config.toml> put the configured devices on the platform as one agent
                                       (the per-agent key is read from the variable the file names)
  robutler-device twin <spec.json> uart --pty serve a simulated part on a new PTY (prints its path)
  robutler-device twin <spec.json> uart <path> serve a simulated part on a serial port
  robutler-device twin <spec.json> can <iface> <node>   serve a simulated part on SocketCAN
  robutler-device agent login  <@agent> --portal <url> [--name <device name>]
                                       sign this device in as the agent (the owner allows it on the
                                       platform's /device page); the key stays on this device
  robutler-device agent logout <@agent> --portal <url>   sign this device out
  robutler-device twin esp-at --listen <addr> an ESP-AT network co-processor on a TCP serial line
                                       (QEMU: -serial tcp:<addr>); opens real TCP for the device";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    match a.as_slice() {
        ["regspec", verb @ ("check" | "tools" | "gen"), path] => regspec(verb, path),
        ["regspec", "try", path, calls @ ..] if !calls.is_empty() && calls.len() % 2 == 0 => try_calls(path, calls),
        #[cfg(target_os = "linux")]
        ["gateway", verb @ ("check" | "run"), path] => gateway::run(verb, path),
        #[cfg(target_os = "linux")]
        ["agent", verb @ ("login" | "logout"), agent, rest @ ..] => gateway::agent_signin(verb, agent, rest),
        #[cfg(target_os = "linux")]
        ["twin", "esp-at", "--listen", addr] => gateway::esp_at(addr),
        #[cfg(target_os = "linux")]
        ["twin", spec, rest @ ..] => gateway::twin(spec, rest),
        ["--version" | "-V"] => {
            println!("robutler {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// A spec with a deprecated schema id still loads; say so once, on stderr.
fn schema_note(src: &str) {
    if let Some(old) = load::deprecated_schema(src) {
        eprintln!("robutler-device: schema \"{old}\" is deprecated; change it to \"{}\"", robutler::webagents::regspec::SCHEMA);
    }
}

fn regspec(verb: &str, path: &str) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("robutler-device: cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    schema_note(&src);
    let dev = match load::load_json(&src) {
        Ok(d) => d,
        Err(e) => {
            eprint!("{e}");
            return ExitCode::FAILURE;
        }
    };
    match verb {
        "check" => {
            let writable = dev.registers.iter().filter(|r| r.fields.iter().any(|f| f.access.writable())).count();
            let rules: usize = dev.registers.iter().flat_map(|r| r.fields).map(|f| f.rules.len()).sum();
            println!(
                "{path}: {} ({}), {} registers, {} fields, {} writable registers, {} rules: OK",
                dev.name,
                dev.title,
                dev.registers.len(),
                dev.field_count,
                writable,
                rules
            );
        }
        "tools" => {
            let mut s = String::new();
            write_tools(dev, &mut s).expect("writing to a String cannot fail");
            println!("{s}");
        }
        _ => print!("{}", codegen::generate(dev)),
    }
    ExitCode::SUCCESS
}

/// Registers in a map, at their reset values: enough to see what a spec's tools
/// accept, refuse and report, before any hardware exists.
struct Twin(std::collections::HashMap<u32, u32>);

impl RegisterBus for Twin {
    type Error = std::convert::Infallible;
    fn read(&mut self, offset: u32, _width: u8) -> Result<u32, Self::Error> {
        Ok(self.0.get(&offset).copied().unwrap_or(0))
    }
    fn write(&mut self, offset: u32, _width: u8, value: u32) -> Result<(), Self::Error> {
        self.0.insert(offset, value);
        Ok(())
    }
}

fn try_calls(path: &str, calls: &[&str]) -> ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("robutler-device: cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    schema_note(&src);
    let dev = match load::load_json(&src) {
        Ok(d) => d,
        Err(e) => {
            eprint!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let twin = Twin(dev.registers.iter().map(|r| (r.offset, r.reset)).collect());
    let mut engine = Engine::new(dev, twin, vec![0u64; dev.field_count]);
    let ctx = CallCtx { now_ms: 0, confirmed: std::env::var_os("ROBUTLER_TRY_CONFIRMED").is_some() };
    let mut out = vec![0u8; 16 * 1024];
    for pair in calls.chunks(2) {
        let (tool, args) = (pair[0], pair[1]);
        match engine.call(tool, args, &ctx, &mut out) {
            Ok(n) => println!("{tool} {args}\n  -> {}", String::from_utf8_lossy(&out[..n])),
            Err(e) => {
                let mut s = String::new();
                e.write_json(dev, &mut s).expect("writing to a String cannot fail");
                println!("{tool} {args}\n  -> {s}");
            }
        }
    }
    ExitCode::SUCCESS
}

#[cfg(target_os = "linux")]
mod gateway {
    //! The gateway and twin verbs (Linux: they use serial ports, SocketCAN and i2c-dev).
    use std::path::Path;
    use std::process::ExitCode;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use robutler::webagents::gateway::{self, PortalEvent};
    use robutler::webagents::regspec::load;
    use robutler::webagents::transport::{self, sim::RegModel};

    pub fn run(verb: &str, path: &str) -> ExitCode {
        let p = Path::new(path);
        if verb == "check" {
            return match gateway::GatewayConfig::load(p).and_then(|(cfg, base)| gateway::build_devices(&cfg, &base).map(|d| (cfg, d))) {
                Ok((cfg, devices)) => {
                    for (name, title) in devices.names() {
                        println!("device {name}: {title}");
                    }
                    let set = std::env::var(&cfg.portal.token_env).map(|v| !v.is_empty()).unwrap_or(false);
                    println!(
                        "{} tools; agent {} at {}; key variable {} is {}",
                        devices.tools().len(),
                        cfg.portal.agent,
                        cfg.portal.url,
                        cfg.portal.token_env,
                        if set { "set" } else { "NOT set" }
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("robutler-device: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("robutler-device: {e}");
                return ExitCode::FAILURE;
            }
        };
        rt.block_on(async {
            let (tx, rx) = tokio::sync::watch::channel(false);
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                let _ = tx.send(true);
            });
            let log = Arc::new(|e: PortalEvent| eprintln!("[robutler-device gateway] {e:?}"));
            match gateway::run(p, rx, Some(log)).await {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("robutler-device: {e}");
                    ExitCode::FAILURE
                }
            }
        })
    }

    pub fn twin(spec: &str, rest: &[&str]) -> ExitCode {
        let src = match std::fs::read_to_string(spec) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("robutler-device: cannot read {spec}: {e}");
                return ExitCode::from(2);
            }
        };
        super::schema_note(&src);
        let dev = match load::load_json(&src) {
            Ok(d) => d,
            Err(e) => {
                eprint!("{e}");
                return ExitCode::FAILURE;
            }
        };
        let model = RegModel::at_reset(dev);
        let stop = Arc::new(AtomicBool::new(false));
        let s2 = stop.clone();
        let _ = ctrlc_flag(s2);
        let r = match rest {
            ["uart", "--pty"] => {
                use serialport::{SerialPort, TTYPort};
                match TTYPort::pair() {
                    Ok((mut master, slave)) => {
                        println!("{} twin on {}", dev.name, slave.name().unwrap_or_default());
                        // The gateway opens the slave path; the twin answers on the master.
                        // The slave stays open here for the twin's lifetime: with no slave
                        // descriptor open, reads on the master fail with EIO.
                        let _keep_slave_open = slave;
                        let _ = master.set_timeout(Duration::from_millis(100));
                        transport::uart::serve(master, model, stop)
                    }
                    Err(e) => Err(std::io::Error::other(e.to_string())),
                }
            }
            ["uart", path] => match serialport::new(*path, 115_200).timeout(Duration::from_millis(100)).open_native() {
                Ok(port) => {
                    println!("{} twin on {path}", dev.name);
                    transport::uart::serve(port, model, stop)
                }
                Err(e) => Err(std::io::Error::other(e.to_string())),
            },
            ["can", iface, node] => match node.parse::<u8>() {
                Ok(n) => {
                    println!("{} twin on {iface} node {n}", dev.name);
                    transport::can::serve(iface, n, model, stop)
                }
                Err(_) => Err(std::io::Error::other("node must be 0 to 127")),
            },
            _ => {
                eprintln!("usage: robutler-device twin <spec.json> uart --pty | uart <path> | can <iface> <node>");
                return ExitCode::from(2);
            }
        };
        match r {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("robutler-device: {e}");
                ExitCode::FAILURE
            }
        }
    }

    /// An ESP-AT co-processor for emulated firmware: accepts one serial
    /// connection (QEMU's `-serial tcp:<addr>`), then serves AT commands on it.
    pub fn esp_at(addr: &str) -> ExitCode {
        let l = match std::net::TcpListener::bind(addr) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("robutler-device: cannot listen on {addr}: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("esp-at twin listening on {addr}");
        let (serial, peer) = match l.accept() {
            Ok(x) => x,
            Err(e) => {
                eprintln!("robutler-device: {e}");
                return ExitCode::FAILURE;
            }
        };
        let _ = serial.set_nodelay(true);
        println!("esp-at twin: serial line from {peer}");
        let stats = Arc::new(std::sync::Mutex::new(transport::esp_at_twin::TwinStats::default()));
        let r = transport::esp_at_twin::serve(serial, stats.clone());
        let s = stats.lock().unwrap().clone();
        println!("esp-at twin: {} commands, {} connects, {} bytes up, {} bytes down", s.commands, s.connects, s.bytes_up, s.bytes_down);
        match r {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("robutler-device: {e}");
                ExitCode::FAILURE
            }
        }
    }

    /// `robutler-device agent login|logout`: the device flow, on the console (no browser).
    pub fn agent_signin(verb: &str, agent: &str, rest: &[&str]) -> ExitCode {
        use robutler::webagents::auth::DeviceKey;
        use robutler::webagents::gateway::signin;
        let opt = |k: &str| rest.windows(2).find(|w| w[0] == k).map(|w| w[1].to_string());
        let Some(portal_url) = opt("--portal") else {
            eprintln!("robutler-device: --portal <url> is required");
            return ExitCode::from(2);
        };
        robutler::webagents::gateway::init_tls();
        let r = if verb == "logout" {
            signin::load(&portal_url, agent).and_then(|(si, name)| {
                let dir = signin::key_dir();
                let all: serde_json::Value =
                    std::fs::read_to_string(dir.join("agents.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
                let rec = &all[format!("{}:{name}", si.portal.authority)];
                let agent_id = rec["agent_id"].as_str().unwrap_or("");
                si.logout(agent_id, rec["binding_id"].as_str().unwrap_or(""))?;
                // The key is useless once the platform has dropped its binding, and a
                // later `agent login` for the same agent needs the name free.
                let _ = std::fs::remove_file(dir.join(signin::key_name(&si.portal, agent_id)));
                println!("Signed out @{name} on {}.", si.portal.authority);
                Ok(())
            })
        } else {
            (|| {
                let portal = signin::Portal::parse(&portal_url)?;
                let seed = signin::new_seed()?;
                let mut si = signin::SignIn::new(portal.clone(), DeviceKey::from_seed(&seed));
                let _ = si.sync_clock();
                let host = std::fs::read_to_string("/etc/hostname").map(|h| h.trim().to_string()).unwrap_or_else(|_| "device".into());
                let name = opt("--name").unwrap_or(host);
                let p = si.authorize(agent, &name)?;
                let dir = signin::key_dir();
                let key_name = signin::key_name(&portal, &p.agent_id);
                if dir.join(&key_name).exists() {
                    // Keys are never overwritten. A sign-in that never completed (a
                    // denied or expired code) leaves its key behind; say how to clear it.
                    return Err(robutler::webagents::gateway::GatewayError::Config(format!(
                        "this device already holds a key for @{}: {}. Sign out first with `robutler-device agent logout @{} --portal {portal_url}`, or, if the earlier sign-in never completed, delete that file and run this again",
                        p.agent_username,
                        dir.join(&key_name).display(),
                        p.agent_username
                    )));
                }
                signin::store_seed(&dir, &key_name, &seed)?;
                println!("To sign this device in as @{}, open\n\n  {}\n\nand enter the code  {}\n\nCheck that the page shows this device's fingerprint  {}\n(or open {} directly, or scan this code with a phone):\n\n{}\nWaiting...", p.agent_username, p.verification_uri, p.user_code, p.fingerprint, p.verification_uri_complete, super::qr(&p.verification_uri_complete));
                let t = si.poll(&p, &mut |d| std::thread::sleep(d))?;
                signin::remember(&dir, &portal, &t)?;
                println!("Signed in as @{} on {}.", t.agent_username, portal.authority);
                Ok(())
            })()
        };
        match r {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("robutler-device: {e}");
                ExitCode::FAILURE
            }
        }
    }

    /// Ctrl-C stops the twin at its next read timeout.
    fn ctrlc_flag(stop: Arc<AtomicBool>) -> std::io::Result<()> {
        std::thread::Builder::new().name("ctrl-c".into()).spawn(move || {
            if let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() {
                rt.block_on(async {
                    let _ = tokio::signal::ctrl_c().await;
                });
                stop.store(true, Ordering::Relaxed);
            }
        })?;
        Ok(())
    }
}

/// The sign-in link as a QR code drawn with Unicode half blocks (two modules per
/// character cell) with the standard four-module quiet zone. Dark modules are drawn
/// as blanks, so on the usual dark terminal the code reads dark on light; scanners
/// also read the inverse on a light terminal. A
/// phone scans it straight off a headless device's terminal.
#[cfg(target_os = "linux")]
fn qr(link: &str) -> String {
    use qrcode::render::unicode::Dense1x2;
    match qrcode::QrCode::new(link.as_bytes()) {
        Ok(code) => code.render::<Dense1x2>().dark_color(Dense1x2::Light).light_color(Dense1x2::Dark).quiet_zone(true).build(),
        Err(_) => String::from("(the link is too long for a QR code)"),
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #[test]
    fn qr_is_a_square_block_with_quiet_zone() {
        let q = super::qr("https://robutler.ai/device?code=WDJB-MJHT");
        let rows: Vec<&str> = q.lines().collect();
        let w = rows[0].chars().count();
        assert!(rows.len() >= 12 && rows.iter().all(|r| r.chars().count() == w), "{q}");
        // Dark modules are blanks, so the quiet zone (light) is solid full blocks.
        assert!(rows[0].chars().all(|c| c == '\u{2588}'), "{q}");
        assert!(q.contains(' ') || q.contains('\u{2580}') || q.contains('\u{2584}'));
        // For an external decode check: render the code where asked.
        if let Ok(path) = std::env::var("ROBUTLER_QR_OUT") {
            std::fs::write(path, &q).unwrap();
        }
    }
}
