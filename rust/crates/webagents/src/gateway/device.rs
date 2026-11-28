//! Devices as the gateway sees them: something with a name, a tool list, and a
//! way to run one tool call. Every register device is an [`Engine`] over its
//! own bus, so the envelope is enforced here, on the gateway, whatever the
//! model sends.

use std::sync::Mutex;

use serde_json::Value;

use crate::bus::RegisterBus;
use crate::regspec::{engine::write_tools, CallCtx, Device, Engine};

/// The result of one tool call, as the model will read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutcome {
    /// The result as JSON text: the engine's result object, or
    /// `{"ok":false,"error":"<code>","message":"..."}` for a refusal or failure.
    pub json: String,
    /// True for a refusal or a failure. It becomes the `error` status of the
    /// `tool_result` delta, so the chat shows the call as not done.
    pub is_error: bool,
}

/// One device behind the gateway.
///
/// Every register device is one already (an [`Engine`] over a `Send` bus
/// with a `Vec<u64>` limit state). Implement it yourself for hardware that is
/// not a register map (a camera, a motion controller with its own protocol)
/// to put it behind the same agent; your `call` is then the envelope.
pub trait ToolDevice: Send {
    /// The device's name, the prefix of its tool names. Unique within a gateway.
    fn name(&self) -> &str;
    /// What a person calls the device; it goes into the model's instructions.
    fn title(&self) -> &str;
    /// Function tools (OpenAI shape).
    fn tools(&self) -> Vec<Value>;
    /// Whether this device owns `tool`.
    fn owns(&self, tool: &str) -> bool;
    /// Runs one call. `args` is the model's JSON text, unchecked: validating it
    /// is this method's job. A failure is an outcome with `is_error`, never a
    /// panic, so one bad call cannot stop the gateway. May block on bus I/O; the
    /// gateway calls it off its async threads.
    fn call(&mut self, tool: &str, args: &str, ctx: &CallCtx) -> ToolOutcome;
}

/// Output buffer for one result; a register result is a few hundred bytes.
const OUT: usize = 8 * 1024;

impl<B> ToolDevice for Engine<'static, B, Vec<u64>>
where
    B: RegisterBus + Send,
{
    fn name(&self) -> &str {
        self.dev.name
    }
    fn title(&self) -> &str {
        self.dev.title
    }
    fn tools(&self) -> Vec<Value> {
        tools_of(self.dev)
    }
    fn owns(&self, tool: &str) -> bool {
        self.resolve(tool).is_some()
    }
    fn call(&mut self, tool: &str, args: &str, ctx: &CallCtx) -> ToolOutcome {
        let mut out = vec![0u8; OUT];
        let dev = self.dev;
        match Engine::call(self, tool, args, ctx, &mut out) {
            Ok(n) => ToolOutcome { json: String::from_utf8_lossy(&out[..n]).into_owned(), is_error: false },
            Err(e) => {
                let mut s = String::new();
                let _ = e.write_json(dev, &mut s);
                ToolOutcome { json: s, is_error: true }
            }
        }
    }
}

/// The tool list of a device table as JSON values.
///
/// # Panics
/// Never for a table that loaded: the tool writer always produces a JSON array.
pub fn tools_of(dev: &Device) -> Vec<Value> {
    let mut s = String::new();
    write_tools(dev, &mut s).expect("writing to a String cannot fail");
    match serde_json::from_str(&s) {
        Ok(Value::Array(v)) => v,
        // The writer is tested to produce a JSON array; anything else is a bug.
        other => panic!("tool descriptor writer produced {other:?}"),
    }
}

/// Every device behind one gateway agent.
///
/// Each device sits behind its own lock, so calls to different devices run in
/// parallel and calls to one device run one at a time.
#[derive(Default)]
pub struct Devices {
    list: Vec<Mutex<Box<dyn ToolDevice>>>,
}

impl Devices {
    /// No devices yet; see [`Devices::add`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a device. Tool names are prefixed by the device name, so two devices
    /// with the same name would shadow each other: refused.
    pub fn add(&mut self, d: Box<dyn ToolDevice>) -> Result<(), String> {
        if self.list.iter().any(|x| x.lock().map(|x| x.name() == d.name()).unwrap_or(false)) {
            return Err(format!("two devices are named {}", d.name()));
        }
        self.list.push(Mutex::new(d));
        Ok(())
    }

    /// Whether no device has been added.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Every device's tools, in the order the devices were added: the list the
    /// model is given on each turn.
    pub fn tools(&self) -> Vec<Value> {
        self.list.iter().flat_map(|d| d.lock().map(|d| d.tools()).unwrap_or_default()).collect()
    }

    /// `(name, title)` of each device.
    pub fn names(&self) -> Vec<(String, String)> {
        self.list.iter().filter_map(|d| d.lock().ok().map(|d| (d.name().to_string(), d.title().to_string()))).collect()
    }

    /// Runs one call on the device that owns the tool. Calls to one device are
    /// serialized by its lock, so two turns can never interleave a
    /// read-modify-write on the same register.
    pub fn call(&self, tool: &str, args: &str, ctx: &CallCtx) -> ToolOutcome {
        for d in &self.list {
            let mut d = match d.lock() {
                Ok(d) => d,
                Err(p) => p.into_inner(),
            };
            if d.owns(tool) {
                return d.call(tool, args, ctx);
            }
        }
        ToolOutcome {
            json: format!("{{\"ok\":false,\"error\":\"unknown_tool\",\"message\":\"No device has a tool named {}.\"}}", json_escape(tool)),
            is_error: true,
        }
    }
}

fn json_escape(s: &str) -> String {
    let v = Value::String(s.to_string()).to_string();
    v[1..v.len() - 1].to_string()
}
