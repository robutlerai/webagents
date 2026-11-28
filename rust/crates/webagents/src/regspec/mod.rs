//! Register specs: the description of a part's registers that typed, limit-checked
//! tools are generated from.
//!
//! WHY THIS SHAPE. A model that drives hardware must never be the thing that keeps
//! the hardware inside its limits. The limits belong to the part, so they are
//! written once, next to the register map, by whoever designed the part (a design
//! tool can export them), and every layer that touches the bus enforces the same table:
//! the generated firmware driver, the host gateway and the tool dispatcher. A
//! request outside the envelope is REFUSED with the reason, never clamped quietly:
//! a clamped "set dead time to 0.5 us" that silently writes 1.0 us is a different
//! command from the one the model thinks it issued.
//!
//! The tables here are plain `&'a` data so the same engine runs from `static`
//! tables emitted by `codegen` (feature `std`) on a microcontroller
//! (no heap) and from a spec loaded at run time on a gateway (`std`).
//!
//! The JSON form is `webagents.regspec/1`, a superset of the register list a
//! datasheet already carries (`{offset, name, access, desc, fields: [{bits, name,
//! desc}]}`): a datasheet register list loads as is, and gains units, ranges and
//! safety rules field by field.
//!
//! # From a spec to a tool call
//!
//! Load a spec (feature `std`), put an [`Engine`] over the device's bus, and
//! every register is a pair of tools a model can call. The spec here is the
//! example PWM driver of this repository's tests:
//!
//! ```
//! # #[cfg(feature = "std")] {
//! use webagents::bus::MemBus;
//! use webagents::regspec::{engine::write_tools, load::load_json, CallCtx, Engine};
//!
//! let dev = load_json(include_str!("../../../../fixtures/pwm1.regspec.json")).unwrap();
//! assert_eq!((dev.name, dev.registers.len(), dev.field_count), ("pwm1", 12, 24));
//!
//! // The tool list a model is given: OpenAI-style function tools, as JSON.
//! let mut tools = String::new();
//! write_tools(dev, &mut tools).unwrap();
//! assert!(tools.contains(r#""name":"pwm1_read_status""#));
//! assert!(tools.contains(r#""name":"pwm1_set_deadtime""#));
//!
//! // One call. The bus is 32 words of memory; a gateway would pass a serial,
//! // CAN, I2C or SPI bus here, and firmware a memory-mapped one.
//! let mut engine = Engine::new(dev, MemBus::<32>::new(), vec![0u64; dev.field_count]);
//! let ctx = CallCtx { now_ms: 0, confirmed: false };
//! let mut out = [0u8; 512];
//! let n = engine.call("pwm1_set_deadtime", r#"{"deadtime": 1.5}"#, &ctx, &mut out).unwrap();
//! assert_eq!(
//!     core::str::from_utf8(&out[..n]).unwrap(),
//!     r#"{"ok":true,"register":"DEADTIME","written":"0x0000000f","applied":{"DEADTIME":1.5},"units":{"DEADTIME":"us"}}"#
//! );
//! // 1.5 us at 0.1 us per count is 15 counts, at offset 0x10.
//! assert_eq!(engine.bus.get(0x10), 15);
//! # }
//! ```
//!
//! A write outside the envelope is refused before the bus is touched; see
//! [`Refusal`] for an example and every reason.
//!
//! # Without a heap
//!
//! Everything in this module except `load` and `codegen` (feature `std`) is
//! `no_std` and allocation-free. Firmware does not parse JSON: it compiles the
//! tables that `robutler-device regspec gen` (or `codegen::generate` from a
//! `build.rs`) emits, and runs the same [`Engine`] over them with a `[u64; N]`
//! limit state.

pub mod dec;
pub mod engine;
pub mod json;

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod codegen;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod load;

pub use dec::Dec;
pub use engine::{CallCtx, Engine, Refusal, ToolError};

/// The JSON schema tag this crate reads and writes.
pub const SCHEMA: &str = "webagents.regspec/1";

/// Earlier names of [`SCHEMA`], still read for one release and then removed.
///
/// A spec that carries one of these loads exactly as if it carried [`SCHEMA`];
/// `load::deprecated_schema` (feature `std`) tells a tool to ask for the
/// file to be updated.
pub const DEPRECATED_SCHEMAS: &[&str] = &["reda.regspec/1"];

/// One part (a chip or a block on a chip) and its registers.
///
/// A `Device` is plain borrowed data. It comes from
/// `load::load_json` at run time (feature `std`, leaked to `'static`) or
/// from a `static` emitted by `codegen::generate` at build time; a table
/// written by hand works too, but then nothing has validated it, and the
/// engine trusts its tables (field positions inside the register, at most 32
/// fields per register, `first_field` and `field_count` consistent).
///
/// ```
/// use webagents::regspec::{Access, Bus, Dec, Device, Encoding, Field, Register};
///
/// // One 8-bit read/write register with one field, written out by hand.
/// static FIELDS: [Field<'static>; 1] = [Field {
///     name: "LEVEL",
///     desc: "Output level",
///     lsb: 0,
///     width: 8,
///     access: Access::RW,
///     encoding: Encoding::Unsigned,
///     scale: Dec::ONE,
///     offset: Dec::ZERO,
///     unit: "",
///     min: 0,
///     max: 200,
///     enums: &[],
///     rules: &[],
///     confirm: false,
///     min_interval_ms: 0,
/// }];
/// static REGISTERS: [Register<'static>; 1] = [Register {
///     name: "LEVEL",
///     desc: "Output level",
///     offset: 0x00,
///     width: 8,
///     access: Access::RW,
///     reset: 0,
///     read_side_effect: false,
///     fields: &FIELDS,
///     first_field: 0,
/// }];
/// static DAC: Device<'static> = Device { name: "dac", title: "example DAC", bus: Bus::Custom, registers: &REGISTERS, field_count: 1 };
///
/// assert_eq!(DAC.registers[0].fields[0].mask(), 0xff);
/// assert!(DAC.registers[0].access.writable());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Device<'a> {
    /// Lower-case identifier, used as the tool name prefix (`pwm1_set_ctrl`).
    pub name: &'a str,
    /// Human title, used in tool descriptions.
    pub title: &'a str,
    /// How the registers are reached.
    pub bus: Bus,
    /// The registers, in spec order. Tool lists and generated code follow
    /// this order, so it is part of what a signed tool description covers.
    pub registers: &'a [Register<'a>],
    /// Total number of fields across all registers; sizes the limit state.
    pub field_count: usize,
}

/// How a device's registers are addressed. The engine never touches the bus
/// itself: a [`RegisterBus`](crate::bus::RegisterBus) implementation does,
/// and this tells it what it is talking to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bus {
    /// Memory-mapped at `base`.
    Mmio {
        /// Address of the register block; register offsets are added to it.
        base: u32,
    },
    /// I2C target at a 7-bit address; register address width and data width in bytes.
    I2c {
        /// The target's 7-bit address (0x00 to 0x7f).
        address: u8,
        /// Bytes of register address on the wire, 1 to 4.
        addr_bytes: u8,
        /// Bytes of register value on the wire, 1 to 4.
        data_bytes: u8,
        /// The value's byte order: most significant byte first when true.
        big_endian: bool,
    },
    /// SPI device; register address width and data width in bytes.
    Spi {
        /// Bytes of register address on the wire, 1 to 4.
        addr_bytes: u8,
        /// Bytes of register value on the wire, 1 to 4.
        data_bytes: u8,
        /// The value's byte order: most significant byte first when true.
        big_endian: bool,
    },
    /// Not addressed by this crate (UART protocols, CAN objects): the spec still
    /// generates tools and limits, the transport is supplied by the integrator.
    Custom,
}

/// Register or field access as a datasheet states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read only.
    R,
    /// Read and write.
    RW,
    /// Write only (reads return nothing meaningful).
    W,
    /// Write 1 to trigger an action; reads 0. Never re-written by a read-modify-write.
    W1,
}

impl Access {
    /// Whether a read returns the value: true for `R` and `RW`. A register
    /// gets a `read` tool only when it is readable.
    pub const fn readable(self) -> bool {
        matches!(self, Access::R | Access::RW)
    }
    /// Whether a write is accepted: true for `RW`, `W` and `W1`. A register
    /// gets a `set` tool when at least one of its fields is writable.
    pub const fn writable(self) -> bool {
        matches!(self, Access::RW | Access::W | Access::W1)
    }
    /// The datasheet spelling: `R`, `R/W`, `W` or `W1`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Access::R => "R",
            Access::RW => "R/W",
            Access::W => "W",
            Access::W1 => "W1",
        }
    }
}

/// One register: where it is, how wide, and the fields inside it.
#[derive(Debug, Clone, Copy)]
pub struct Register<'a> {
    /// The datasheet name (`CTRL`). Lower-cased, it is the last part of the
    /// tool names (`pwm1_read_ctrl`, `pwm1_set_ctrl`); matched without case.
    pub name: &'a str,
    /// One line on what the register is, quoted in the tool descriptions.
    pub desc: &'a str,
    /// Byte offset from the device base (or the register address on I2C/SPI).
    pub offset: u32,
    /// Register width in bits (8, 16 or 32).
    pub width: u8,
    /// Access of the register as a whole. A field may narrow it (a read-only
    /// status bit in a read/write register) but the register's own access
    /// decides whether it is read before a partial write.
    pub access: Access,
    /// The value after reset. It is the base word of a write to a register
    /// that cannot be read, and what simulated parts start from.
    pub reset: u32,
    /// A read changes state (clears a flag, pops a FIFO). Such a register is never
    /// read behind the caller's back, so it is never read-modify-written.
    pub read_side_effect: bool,
    /// The fields, at most 32, not overlapping. A register declared without
    /// fields has one field of its own name spanning the whole register.
    pub fields: &'a [Field<'a>],
    /// Index of this register's first field in the device-wide field numbering.
    pub first_field: usize,
}

/// How the bits of a field encode an integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// A plain binary count, 0 to `2^width - 1`.
    Unsigned,
    /// Two's complement.
    Signed,
    /// Top bit is the sign, the rest the magnitude.
    SignMagnitude,
    /// One bit, presented as true/false.
    Bool,
}

/// One field of a register: its bits, how they read as a value, and the
/// limits every write of it must pass.
///
/// The engine works on the field's INTEGER value (its count). A scaled field
/// is shown to the model in physical units, `count * scale + offset`, and a
/// physical value it sends is converted back with exact decimal arithmetic
/// ([`Dec`]), never through a float.
#[derive(Debug, Clone, Copy)]
pub struct Field<'a> {
    /// The datasheet name (`DEADTIME`). Lower-cased, it is the argument name
    /// in the register's `set` tool; matched without case.
    pub name: &'a str,
    /// One line on what the field does, quoted in the tool's schema.
    pub desc: &'a str,
    /// Position of the field's least significant bit in the register.
    pub lsb: u8,
    /// Width in bits, 1 to 32; `lsb + width` does not exceed the register's width.
    pub width: u8,
    /// Access of this field, which may differ from its register's.
    pub access: Access,
    /// How the bits read as an integer.
    pub encoding: Encoding,
    /// Physical value = integer value * scale + offset, exactly (see [`Dec`]).
    pub scale: Dec,
    /// The physical value of count 0, in `unit` (see `scale`).
    pub offset: Dec,
    /// Physical unit, empty for a dimensionless count or a flag.
    pub unit: &'a str,
    /// Inclusive limits on the INTEGER value a write may carry (the physical range,
    /// converted once when the table is built, so the check needs no float).
    pub min: i64,
    /// The inclusive upper limit, as a count (see `min`).
    pub max: i64,
    /// Named values, for mode fields. A write may name the value instead of the number.
    pub enums: &'a [EnumValue<'a>],
    /// Cross-field rules checked on every write of this field.
    pub rules: &'a [Rule<'a>],
    /// The host must have a human's confirmation for this write (a trip level, a
    /// safety interlock). The model cannot grant it: see [`CallCtx::confirmed`].
    pub confirm: bool,
    /// Minimum milliseconds between two writes of this field (0 = no limit).
    pub min_interval_ms: u32,
}

impl<'a> Field<'a> {
    /// The field's bits within its register word, in place (a 4-bit field at
    /// bit 8 gives `0x0f00`).
    pub const fn mask(&self) -> u32 {
        if self.width >= 32 {
            u32::MAX
        } else {
            ((1u32 << self.width) - 1) << self.lsb
        }
    }
    /// Whether this field is presented in physical units rather than as a count.
    pub fn is_scaled(&self) -> bool {
        self.scale != Dec::ONE || self.offset != Dec::ZERO
    }
}

/// One named value of a mode field (`sinusoidal` = 1).
#[derive(Debug, Clone, Copy)]
pub struct EnumValue<'a> {
    /// The name a tool call may send and a read reports; an identifier,
    /// matched without case.
    pub name: &'a str,
    /// The count the name stands for.
    pub value: i64,
    /// What the value selects, added to the tool's schema; may be empty.
    pub desc: &'a str,
}

/// A field of the same device, by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldRef {
    /// Index into [`Device::registers`].
    pub reg: u16,
    /// Index into that register's [`Register::fields`].
    pub field: u16,
}

/// One side of a rule's comparison. Every operand evaluates to a count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// The value being written to the field that owns the rule.
    New,
    /// Another field's integer value: its new value if the same write sets it,
    /// otherwise read from the device.
    Field(FieldRef),
    /// A constant, already a count: the loader converts a physical constant
    /// (`20` in a rule on a field in microseconds) once, and refuses one that
    /// is not a whole step of the field.
    Const(i64),
}

/// The comparison of a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl Op {
    /// Whether `a OP b` holds, on counts.
    pub fn eval(self, a: i64, b: i64) -> bool {
        match self {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
        }
    }
    /// The operator as written in a spec (`<=`), used when a refusal quotes
    /// the comparison that failed.
    pub const fn as_str(self) -> &'static str {
        match self {
            Op::Eq => "==",
            Op::Ne => "!=",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
        }
    }
}

/// One comparison, `lhs OP rhs`, on counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cond {
    /// The left side.
    pub lhs: Operand,
    /// The comparison.
    pub op: Op,
    /// The right side.
    pub rhs: Operand,
}

/// `when` holds (or is absent) => `require` must hold, else the write is refused with `why`.
///
/// A rule belongs to one field but guards every field it names: the engine
/// re-checks it on a write of ANY field in its `when` or `require`, so a limit
/// written once cannot be walked around through the other operand. In a spec:
///
/// ```json
/// { "when": "CTRL.MODE == sinusoidal", "require": "self >= 20",
///   "why": "sinusoidal commutation needs a PWM period of at least 20 us" }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rule<'a> {
    /// The condition under which the rule applies; `None` means always.
    pub when: Option<Cond>,
    /// What must hold for the write to go ahead.
    pub require: Cond,
    /// The reason, in the part's own terms. It is what the model is told on a
    /// refusal and what the tool's schema lists as a limit, so it is required.
    pub why: &'a str,
}
