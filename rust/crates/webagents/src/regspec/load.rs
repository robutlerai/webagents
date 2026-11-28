//! `webagents.regspec/1` JSON -> a validated [`Device`] table (host side, `std`).
//!
//! Every problem in a spec is collected and reported together, with the
//! register and field it is in, because the person fixing a spec exported from a
//! design tool wants the whole list at once, not one round trip per mistake.
//! A spec with any error produces no table: a half-understood limit is not a
//! limit.
//!
//! Rules are written as short comparisons so they read like the datasheet:
//! `"require": "self <= PERIOD.PERIOD"`, `"when": "CTRL.MODE == sinusoidal"`.
//! An operand is `self` (the value being written), `REG.FIELD`, an enum name of
//! the field on the other side, or a number in the PHYSICAL units of the field on
//! the other side (so `"self >= 1.0"` on a field in microseconds means 1.0 us).
//!
//! The table is leaked to `'static`: a gateway loads its device specs once at
//! start-up and keeps them for the life of the process, and `'static` tables are
//! what the generated firmware code uses too, so both run the same engine.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::string::{String, ToString};
use std::vec::Vec;

use serde::Deserialize;

use super::dec::{self, Dec, Round};
use super::{Access, Bus, Cond, Device, Encoding, EnumValue, Field, FieldRef, Op, Operand, Register, Rule, DEPRECATED_SCHEMAS, SCHEMA};

/// One problem in a spec, with where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecIssue {
    /// Where the problem is, as a path a person can follow in the file:
    /// `registers[2] PERIOD.PERIOD.rules[0]`, `bus.address`, `schema`, or
    /// `spec` when the file is not valid JSON or has an unknown key.
    pub at: String,
    /// What is wrong, in a sentence.
    pub message: String,
}

impl fmt::Display for SpecIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.at, self.message)
    }
}

/// Every problem found in one spec. Its `Display` is the list a person
/// reads: a count line, then one line per [`SpecIssue`].
///
/// ```
/// use webagents::regspec::load::load_json;
///
/// // Two mistakes in one file: both are reported, not just the first.
/// let err = load_json(
///     r#"{ "device": { "name": "pump" },
///          "registers": [ { "offset": 0, "name": "RATE", "width": 12,
///            "fields": [ { "bits": "7:0", "name": "RATE", "scale": 0.5 } ] } ] }"#,
/// )
/// .unwrap_err();
/// let text = err.to_string();
/// assert_eq!(err.0.len(), 2);
/// assert!(text.contains("width 12 must be 8, 16 or 32"));
/// assert!(text.contains("a scaled field needs a unit"));
/// ```
#[derive(Debug, Clone)]
pub struct SpecError(pub Vec<SpecIssue>);

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} problem(s) in the register spec:", self.0.len())?;
        for i in &self.0 {
            writeln!(f, "  {i}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SpecError {}

// ---- the JSON document -------------------------------------------------------------------

/// A `webagents.regspec/1` file as parsed, before validation. The types
/// below mirror the JSON one to one, and an unknown key anywhere is an error
/// (a misspelt `"range"` must not load as "no limit").
///
/// ```json
/// {
///   "schema": "webagents.regspec/1",
///   "device": { "name": "pwm1", "title": "example three-phase PWM driver" },
///   "bus": { "kind": "mmio", "base": "0x40010000" },
///   "width": 32,
///   "registers": [
///     { "offset": "0x10", "name": "DEADTIME", "access": "R/W", "desc": "Dead time", "reset": 10,
///       "fields": [ { "bits": "7:0", "name": "DEADTIME", "unit": "us", "scale": 0.1, "range": { "min": 0.5 } } ] }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecDoc {
    /// The schema id, [`SCHEMA`]. Optional: a file without it is read as
    /// this schema. Any other id is an error, except the deprecated ones.
    #[serde(default)]
    pub schema: Option<String>,
    /// The part the file describes.
    pub device: DeviceDoc,
    /// How the registers are addressed; absent means `custom`.
    #[serde(default)]
    pub bus: Option<BusDoc>,
    /// Default register width in bits.
    #[serde(default)]
    pub width: Option<u8>,
    /// The registers, in the order tools and generated code will list them.
    pub registers: Vec<RegisterDoc>,
    /// Free text: where the limits came from. Kept in the doc, not in the table.
    #[serde(default)]
    pub provenance: Option<serde_json::Value>,
}

/// The `device` object.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceDoc {
    /// An identifier (a letter, then letters, digits or `_`). Lower-cased, it
    /// prefixes every tool name, and `<device>_read_<register>` may not exceed
    /// 64 characters, the limit function-calling models put on a tool name.
    pub name: String,
    /// What a person calls the part; used in tool descriptions. Defaults to `name`.
    #[serde(default)]
    pub title: Option<String>,
    /// A longer description for readers of the file. Not in the table.
    #[serde(default)]
    pub desc: Option<String>,
}

/// The `bus` object, selected by its `kind`: `mmio`, `i2c`, `spi` or `custom`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum BusDoc {
    /// Memory-mapped registers.
    Mmio {
        /// Address of the register block; must fit 32 bits.
        base: Num,
    },
    /// An I2C target.
    I2c {
        /// The 7-bit address, 0 to 0x7f.
        address: Num,
        /// Bytes of register address, 1 to 4 (default 1).
        #[serde(default = "one")]
        addr_bytes: u8,
        /// Bytes of register value, 1 to 4 (default 1). It is also the
        /// default register width, as `data_bytes * 8` bits.
        #[serde(default = "one")]
        data_bytes: u8,
        /// Whether the value is sent most significant byte first (default true).
        #[serde(default = "yes")]
        big_endian: bool,
    },
    /// A SPI device.
    Spi {
        /// Bytes of register address, 1 to 4 (default 1).
        #[serde(default = "one")]
        addr_bytes: u8,
        /// Bytes of register value, 1 to 4 (default 1). It is also the
        /// default register width, as `data_bytes * 8` bits.
        #[serde(default = "one")]
        data_bytes: u8,
        /// Whether the value is sent most significant byte first (default true).
        #[serde(default = "yes")]
        big_endian: bool,
    },
    /// A transport this crate does not frame (a serial protocol, CAN).
    Custom,
}

fn one() -> u8 {
    1
}
fn yes() -> bool {
    true
}

/// A number written as JSON number or as a string (`"0x40000000"`, `"0b101"`, `"12"`).
///
/// A string may be hexadecimal, binary or decimal, with `_` between digits.
/// Offsets, addresses, resets and enum values must be integers; scales,
/// offsets and range bounds are decimals, read exactly as written.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Num {
    /// A JSON integer.
    Int(i64),
    /// A JSON number with a fraction or an exponent.
    Float(f64),
    /// A number in a string.
    Text(String),
}

impl Num {
    fn as_i64(&self) -> Result<i64, String> {
        match self {
            Num::Int(v) => Ok(*v),
            Num::Float(f) if f.fract() == 0.0 => Ok(*f as i64),
            Num::Float(f) => Err(format!("{f} is not an integer")),
            Num::Text(s) => parse_int(s),
        }
    }
    /// The exact decimal the author wrote. A JSON float arrives as f64; its
    /// shortest round-trip form is the text that was written (for the 15 or so
    /// significant digits a datasheet value has), so it converts exactly.
    fn as_dec(&self) -> Result<Dec, String> {
        match self {
            Num::Int(v) => Ok(Dec::from_int(*v)),
            Num::Float(f) => Dec::parse(&normalize_float_text(*f)).ok_or_else(|| format!("{f} has more precision than a register can use")),
            Num::Text(s) => parse_int(s).map(Dec::from_int).or_else(|_| Dec::parse(s.trim()).ok_or_else(|| format!("'{s}' is not a number"))),
        }
    }
}

/// `{f}` prints the shortest round-trip form; JSON wants no `inf` and an exponent as `e`.
fn normalize_float_text(f: f64) -> String {
    let t = format!("{f}");
    if t.contains('e') || t.contains('E') {
        format!("{f:e}")
    } else {
        t
    }
}

fn parse_int(s: &str) -> Result<i64, String> {
    let t = s.trim().replace('_', "");
    let (neg, t) = match t.strip_prefix('-') {
        Some(r) => (true, r.to_string()),
        None => (false, t),
    };
    let v = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i64::from_str_radix(h, 16)
    } else if let Some(b) = t.strip_prefix("0b") {
        i64::from_str_radix(b, 2)
    } else {
        t.parse::<i64>()
    }
    .map_err(|_| format!("'{s}' is not an integer"))?;
    Ok(if neg { -v } else { v })
}

/// One entry of `registers`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterDoc {
    /// Byte offset from the device base, or the register address on I2C and
    /// SPI. Unique within the device; 0 to `0xffff_ffff`.
    pub offset: Num,
    /// The datasheet name, an identifier, unique without regard to case.
    pub name: String,
    /// `R` (or `RO`), `R/W` (or `RW`), `W` (or `WO`), `W1` (also `W1S`,
    /// `W1C`, `WO1`). Default `R/W`.
    #[serde(default)]
    pub access: Option<String>,
    /// One line on what the register is; quoted in the tool descriptions.
    #[serde(default)]
    pub desc: Option<String>,
    /// Width in bits: 8, 16 or 32. Defaults to the file's `width`, else to
    /// the bus's data width, else 32.
    #[serde(default)]
    pub width: Option<u8>,
    /// The value after reset. When fields carry resets too, the two must agree.
    /// Default: assembled from the field resets (0 where a field has none).
    #[serde(default)]
    pub reset: Option<Num>,
    /// True when a read changes the part's state. The register is then never
    /// read before a write, so a write must name every writable field.
    #[serde(default)]
    pub read_side_effect: bool,
    /// The fields, at most 32. Empty means one unsigned field of the
    /// register's name spanning the register.
    #[serde(default)]
    pub fields: Vec<FieldDoc>,
}

/// One entry of a register's `fields`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldDoc {
    /// The bit or bit range, as a datasheet writes it: `"3"` or `"11:8"`
    /// (high bit first). Fields of one register must not overlap.
    pub bits: String,
    /// The datasheet name, an identifier, unique within the register.
    pub name: String,
    /// One line on what the field does; quoted in the tool's schema.
    #[serde(default)]
    pub desc: Option<String>,
    /// As the register's `access`; defaults to the register's.
    #[serde(default)]
    pub access: Option<String>,
    /// `unsigned`, `signed` (or `twos`), `sign_magnitude` or `bool`. Default:
    /// `bool` for a 1-bit field with no enum and no scale, else `unsigned`.
    #[serde(default)]
    pub encoding: Option<String>,
    /// The physical unit (`us`, `mV`, `deg`). Required when the field is scaled.
    #[serde(default)]
    pub unit: Option<String>,
    /// Physical value per count (default 1). Not zero; negative is allowed.
    #[serde(default)]
    pub scale: Option<Num>,
    /// Physical value of count 0 (default 0).
    #[serde(default)]
    pub offset: Option<Num>,
    /// Limits on what a write may carry, in physical units. Absent: whatever
    /// the field's bits can hold.
    #[serde(default)]
    pub range: Option<RangeDoc>,
    /// Named values of a mode field. A field with an enum cannot be scaled.
    #[serde(default, rename = "enum")]
    pub enums: Option<EnumDoc>,
    /// Rules checked on every write that can affect them.
    #[serde(default)]
    pub rules: Vec<RuleDoc>,
    /// True when a write of this field needs a person's confirmation.
    #[serde(default)]
    pub confirm: bool,
    /// Minimum milliseconds between two writes of this field (0: no limit).
    #[serde(default)]
    pub min_interval_ms: u32,
    /// The field's value after reset, as a count.
    #[serde(default)]
    pub reset: Option<Num>,
    /// Where this field's limits came from (free text, not in the table).
    #[serde(default)]
    pub source: Option<String>,
}

/// A field's `range`: inclusive bounds in the field's physical units. A bound
/// between two steps is rounded inwards, and a range the field's bits cannot
/// hold, or an empty one, is an error.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeDoc {
    /// The lowest value a write may carry; absent means the field's own minimum.
    #[serde(default)]
    pub min: Option<Num>,
    /// The highest value a write may carry; absent means the field's own maximum.
    #[serde(default)]
    pub max: Option<Num>,
}

/// A field's `enum`, in either of two spellings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EnumDoc {
    /// `[{"name": "six_step", "value": 0, "desc": "..."}]`: the full form,
    /// with a description per value.
    List(Vec<EnumItemDoc>),
    /// `{"0": "six_step", "1": "sinusoidal"}`
    Map(BTreeMap<String, String>),
}

/// One named value in the list form of an `enum`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumItemDoc {
    /// The name, an identifier, unique within the field without regard to case.
    pub name: String,
    /// The count it stands for; must fit the field.
    pub value: Num,
    /// What the value selects; added to the tool's schema.
    #[serde(default)]
    pub desc: Option<String>,
}

/// One entry of a field's `rules`.
///
/// `when` and `require` are comparisons with one of `==`, `!=`, `<`, `<=`,
/// `>`, `>=`. Each side is `self` (the value being written to this field),
/// `REG.FIELD`, an enum name of the field on the other side, `true`/`false`
/// against a flag, or a number in the physical units of the field on the
/// other side. At least one side must be a field, two fields compared with
/// each other must share scale and offset, and a constant must be a whole
/// step of its field. A field read from another register must be readable
/// without a side effect.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleDoc {
    /// The condition under which the rule applies; absent means always.
    #[serde(default)]
    pub when: Option<String>,
    /// What must hold, or the write is refused.
    pub require: String,
    /// The reason given to the model on a refusal and listed in the tool's
    /// schema. Required, and not blank.
    pub why: String,
    /// Where the rule came from (free text, not in the table).
    #[serde(default)]
    pub source: Option<String>,
}

// ---- loading -----------------------------------------------------------------------------

/// Parse and validate. The result lives for the rest of the process.
///
/// On success the table is leaked to `'static` (a few kilobytes per spec), so
/// load each spec once and keep the reference; do not call this in a loop. On
/// failure every problem is in the [`SpecError`] and nothing is leaked for
/// the caller to use.
///
/// ```
/// use webagents::regspec::{load::load_json, Bus, Encoding};
///
/// let dev = load_json(include_str!("../../../../fixtures/pwm1.regspec.json")).unwrap();
/// assert_eq!(dev.bus, Bus::Mmio { base: 0x4001_0000 });
///
/// let deadtime = &dev.registers[4].fields[0];
/// assert_eq!((deadtime.name, deadtime.unit, deadtime.encoding), ("DEADTIME", "us", Encoding::Unsigned));
/// // The range is kept as counts: 0.5 us at 0.1 us per count is 5, and 8 bits hold up to 255.
/// assert_eq!((deadtime.min, deadtime.max), (5, 255));
/// ```
pub fn load_json(src: &str) -> Result<&'static Device<'static>, SpecError> {
    let doc: SpecDoc = serde_json::from_str(src).map_err(|e| SpecError(vec![SpecIssue { at: "spec".into(), message: e.to_string() }]))?;
    compile(&doc)
}

fn leak(s: &str) -> &'static str {
    String::from(s).leak()
}

fn parse_access(s: Option<&str>, default: Access) -> Result<Access, String> {
    Ok(match s.map(|s| s.trim().to_ascii_uppercase()) {
        None => default,
        Some(s) => match s.as_str() {
            "R" | "RO" => Access::R,
            "R/W" | "RW" => Access::RW,
            "W" | "WO" => Access::W,
            "W1" | "W1S" | "W1C" | "WO1" => Access::W1,
            other => return Err(format!("access '{other}' is not one of R, R/W, W, W1")),
        },
    })
}

fn parse_bits(s: &str) -> Result<(u8, u8), String> {
    let t = s.trim();
    let (hi, lo) = match t.split_once(':') {
        Some((h, l)) => (h.trim().parse::<u8>(), l.trim().parse::<u8>()),
        None => (t.parse::<u8>(), t.parse::<u8>()),
    };
    match (hi, lo) {
        (Ok(h), Ok(l)) if h >= l => Ok((l, h - l + 1)),
        _ => Err(format!("bits '{s}' is not 'N' or 'HI:LO'")),
    }
}

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(ch) if ch.is_ascii_alphabetic()) && s.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn representable(enc: Encoding, width: u8) -> (i64, i64) {
    let w = width as u32;
    match enc {
        Encoding::Unsigned | Encoding::Bool => (0, ((1u64 << w) - 1) as i64),
        Encoding::Signed => (-(1i64 << (w - 1)), (1i64 << (w - 1)) - 1),
        Encoding::SignMagnitude => (-((1i64 << (w - 1)) - 1), (1i64 << (w - 1)) - 1),
    }
}

struct Pending {
    reg: usize,
    field: usize,
    at: String,
    rules: Vec<RuleDoc>,
}

/// The deprecated schema id (one of [`DEPRECATED_SCHEMAS`]) that a spec's
/// source carries, if it carries one. Such a spec still loads; a tool should
/// tell its user to change the file's `schema` to [`SCHEMA`].
///
/// ```
/// use webagents::regspec::{load::deprecated_schema, SCHEMA};
/// assert_eq!(deprecated_schema(r#"{"schema": "reda.regspec/1"}"#), Some("reda.regspec/1"));
/// assert_eq!(deprecated_schema(&format!(r#"{{"schema": "{SCHEMA}"}}"#)), None);
/// ```
pub fn deprecated_schema(src: &str) -> Option<&'static str> {
    let v: serde_json::Value = serde_json::from_str(src).ok()?;
    let s = v.get("schema")?.as_str()?;
    DEPRECATED_SCHEMAS.iter().copied().find(|d| *d == s)
}

/// Validate a parsed doc and build the table.
///
/// The second half of [`load_json`], for a caller that built or edited the
/// [`SpecDoc`] itself (a design tool's exporter). Same checks, same leak.
pub fn compile(doc: &SpecDoc) -> Result<&'static Device<'static>, SpecError> {
    let mut issues: Vec<SpecIssue> = Vec::new();
    let mut err = |at: &str, m: String| issues.push(SpecIssue { at: at.to_string(), message: m });

    if let Some(s) = &doc.schema {
        if s != SCHEMA && !DEPRECATED_SCHEMAS.contains(&s.as_str()) {
            err("schema", format!("'{s}' is not {SCHEMA}"));
        }
    }
    let dname = doc.device.name.to_ascii_lowercase();
    if !is_ident(&dname) {
        err("device.name", format!("'{}' must be an identifier (letters, digits, _), it prefixes every tool name", doc.device.name));
    }
    let bus = match &doc.bus {
        None => Bus::Custom,
        Some(BusDoc::Custom) => Bus::Custom,
        Some(BusDoc::Mmio { base }) => match base.as_i64() {
            Ok(b) if (0..=u32::MAX as i64).contains(&b) => Bus::Mmio { base: b as u32 },
            Ok(b) => {
                err("bus.base", format!("{b:#x} does not fit 32 bits"));
                Bus::Custom
            }
            Err(e) => {
                err("bus.base", e);
                Bus::Custom
            }
        },
        Some(BusDoc::I2c { address, addr_bytes, data_bytes, big_endian }) => {
            let a = address.as_i64().unwrap_or(-1);
            if !(0..=0x7f).contains(&a) {
                err("bus.address", "I2C address must be 7-bit".into());
            }
            if !(1..=4).contains(addr_bytes) || !(1..=4).contains(data_bytes) {
                err("bus", "addr_bytes and data_bytes must be 1 to 4".into());
            }
            Bus::I2c { address: a.clamp(0, 0x7f) as u8, addr_bytes: *addr_bytes, data_bytes: *data_bytes, big_endian: *big_endian }
        }
        Some(BusDoc::Spi { addr_bytes, data_bytes, big_endian }) => {
            if !(1..=4).contains(addr_bytes) || !(1..=4).contains(data_bytes) {
                err("bus", "addr_bytes and data_bytes must be 1 to 4".into());
            }
            Bus::Spi { addr_bytes: *addr_bytes, data_bytes: *data_bytes, big_endian: *big_endian }
        }
    };
    let default_width = doc.width.unwrap_or(match bus {
        Bus::I2c { data_bytes, .. } | Bus::Spi { data_bytes, .. } => data_bytes * 8,
        _ => 32,
    });

    // Pass 1: registers and fields without rules.
    let mut regs: Vec<(Register<'static>, Vec<Field<'static>>)> = Vec::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut reg_names: HashMap<String, usize> = HashMap::new();
    let mut offsets: HashMap<i64, String> = HashMap::new();
    let mut first_field = 0usize;

    for (ri, r) in doc.registers.iter().enumerate() {
        let at = format!("registers[{ri}] {}", r.name);
        if !is_ident(&r.name) {
            err(&at, "register name must be an identifier".into());
        }
        if reg_names.insert(r.name.to_ascii_uppercase(), ri).is_some() {
            err(&at, "duplicate register name".into());
        }
        let offset = match r.offset.as_i64() {
            Ok(o) if (0..=u32::MAX as i64).contains(&o) => o,
            Ok(o) => {
                err(&at, format!("offset {o:#x} out of range"));
                0
            }
            Err(e) => {
                err(&at, format!("offset: {e}"));
                0
            }
        };
        if let Some(prev) = offsets.insert(offset, r.name.clone()) {
            err(&at, format!("offset {offset:#x} also used by {prev}"));
        }
        let width = r.width.unwrap_or(default_width);
        if ![8u8, 16, 32].contains(&width) {
            err(&at, format!("width {width} must be 8, 16 or 32"));
        }
        let access = match parse_access(r.access.as_deref(), Access::RW) {
            Ok(a) => a,
            Err(e) => {
                err(&at, e);
                Access::RW
            }
        };
        // `_read_` is the longer of the two infixes, so it bounds both tool names.
        let tool_len = dname.len() + "_read_".len() + r.name.len();
        if tool_len > 64 {
            err(&at, format!("tool name {dname}_read_{} is longer than 64 characters", r.name.to_ascii_lowercase()));
        }

        // A register without fields is one field spanning the register.
        let synth;
        let fdocs: &[FieldDoc] = if r.fields.is_empty() {
            synth = vec![FieldDoc {
                bits: format!("{}:0", width.saturating_sub(1)),
                name: r.name.clone(),
                desc: r.desc.clone(),
                access: None,
                encoding: Some("unsigned".into()),
                unit: None,
                scale: None,
                offset: None,
                range: None,
                enums: None,
                rules: vec![],
                confirm: false,
                min_interval_ms: 0,
                reset: None,
                source: None,
            }];
            &synth
        } else {
            &r.fields
        };
        if fdocs.len() > 32 {
            err(&at, "more than 32 fields".into());
        }

        let mut fields: Vec<Field<'static>> = Vec::new();
        let mut used: u64 = 0;
        let mut fnames: HashMap<String, ()> = HashMap::new();
        let mut reset_from_fields: u32 = 0;
        for (fi, fd) in fdocs.iter().enumerate() {
            let fat = format!("{at}.{}", fd.name);
            if !is_ident(&fd.name) {
                err(&fat, "field name must be an identifier".into());
            }
            if fnames.insert(fd.name.to_ascii_uppercase(), ()).is_some() {
                err(&fat, "duplicate field name".into());
            }
            let (lsb, fw) = match parse_bits(&fd.bits) {
                Ok(b) => b,
                Err(e) => {
                    err(&fat, e);
                    (0, 1)
                }
            };
            if lsb as u32 + fw as u32 > width as u32 {
                err(&fat, format!("bits {} exceed the {width}-bit register", fd.bits));
            }
            let m: u64 = (((1u128 << fw) - 1) as u64) << lsb;
            if used & m != 0 {
                err(&fat, format!("bits {} overlap another field", fd.bits));
            }
            used |= m;
            let faccess = match parse_access(fd.access.as_deref(), access) {
                Ok(a) => a,
                Err(e) => {
                    err(&fat, e);
                    access
                }
            };
            let has_enum = fd.enums.is_some();
            let encoding = match fd.encoding.as_deref() {
                None if fw == 1 && !has_enum && fd.scale.is_none() => Encoding::Bool,
                None => Encoding::Unsigned,
                Some("unsigned") => Encoding::Unsigned,
                Some("signed") | Some("twos") => Encoding::Signed,
                Some("sign_magnitude") => Encoding::SignMagnitude,
                Some("bool") => Encoding::Bool,
                Some(other) => {
                    err(&fat, format!("encoding '{other}' is not unsigned, signed, sign_magnitude or bool"));
                    Encoding::Unsigned
                }
            };
            if encoding == Encoding::Bool && fw != 1 {
                err(&fat, "bool encoding needs a 1-bit field".into());
            }
            if matches!(encoding, Encoding::Signed | Encoding::SignMagnitude) && fw < 2 {
                err(&fat, "a signed field needs at least 2 bits".into());
            }
            let scale = match fd.scale.as_ref().map(|s| s.as_dec()).transpose() {
                Ok(v) => v.unwrap_or(Dec::ONE),
                Err(e) => {
                    err(&fat, format!("scale: {e}"));
                    Dec::ONE
                }
            };
            if scale == Dec::ZERO {
                err(&fat, "scale must not be zero".into());
            }
            let offset_v = match fd.offset.as_ref().map(|s| s.as_dec()).transpose() {
                Ok(v) => v.unwrap_or(Dec::ZERO),
                Err(e) => {
                    err(&fat, format!("offset: {e}"));
                    Dec::ZERO
                }
            };
            let scaled = scale != Dec::ONE || offset_v != Dec::ZERO;
            if scaled && fd.unit.as_deref().unwrap_or("").is_empty() {
                err(&fat, "a scaled field needs a unit".into());
            }
            if scaled && (encoding == Encoding::Bool || has_enum) {
                err(&fat, "a flag or a mode field cannot be scaled".into());
            }
            let (rmin, rmax) = representable(encoding, fw.min(63));
            let safe_scale = if scale == Dec::ZERO { Dec::ONE } else { scale };
            let (mut min, mut max) = (rmin, rmax);
            if let Some(rg) = &fd.range {
                // Allowed counts c satisfy lo <= c * scale + offset <= hi. With a
                // positive scale lo bounds c from below; a negative scale swaps them.
                let pos = !safe_scale.is_negative();
                for (bound, is_lo) in [(&rg.min, true), (&rg.max, false)] {
                    let Some(n) = bound else { continue };
                    let p = match n.as_dec() {
                        Ok(p) => p,
                        Err(e) => {
                            err(&fat, format!("range: {e}"));
                            continue;
                        }
                    };
                    let lower_count = is_lo == pos;
                    let mode = if lower_count { Round::Ceil } else { Round::Floor };
                    match dec::to_count(p, safe_scale, offset_v, mode) {
                        Some((c, _)) if lower_count => min = c,
                        Some((c, _)) => max = c,
                        None => err(&fat, format!("range bound {p} is far outside the field")),
                    }
                }
                if min < rmin || max > rmax {
                    err(&fat, format!("range is outside what {fw} bits can hold (counts {rmin} to {rmax}, asked {min} to {max})"));
                }
                if min > max {
                    err(&fat, "range is empty".into());
                }
            }
            let enums: Vec<EnumValue<'static>> = match &fd.enums {
                None => vec![],
                Some(EnumDoc::List(items)) => items
                    .iter()
                    .filter_map(|e| match e.value.as_i64() {
                        Ok(v) => Some(EnumValue { name: leak(&e.name), value: v, desc: leak(e.desc.as_deref().unwrap_or("")) }),
                        Err(m) => {
                            err(&fat, format!("enum {}: {m}", e.name));
                            None
                        }
                    })
                    .collect(),
                Some(EnumDoc::Map(m)) => m
                    .iter()
                    .filter_map(|(k, v)| match parse_int(k) {
                        Ok(n) => Some(EnumValue { name: leak(v), value: n, desc: "" }),
                        Err(e) => {
                            err(&fat, format!("enum key: {e}"));
                            None
                        }
                    })
                    .collect(),
            };
            for e in &enums {
                if !is_ident(e.name) {
                    err(&fat, format!("enum name '{}' must be an identifier", e.name));
                }
                if e.value < rmin || e.value > rmax {
                    err(&fat, format!("enum {} = {} does not fit the field", e.name, e.value));
                }
            }
            if !enums.is_empty() {
                let mut seen = HashMap::new();
                for e in &enums {
                    if seen.insert(e.name.to_ascii_lowercase(), ()).is_some() {
                        err(&fat, format!("enum name {} repeated", e.name));
                    }
                }
            }
            if let Some(rs) = &fd.reset {
                match rs.as_i64() {
                    Ok(v) => reset_from_fields |= ((v as u64 as u32) << lsb) & (m as u32),
                    Err(e) => err(&fat, format!("reset: {e}")),
                }
            }
            if (fd.confirm || fd.min_interval_ms > 0 || !fd.rules.is_empty() || fd.range.is_some()) && !faccess.writable() {
                err(&fat, "limits on a field that cannot be written".into());
            }
            if !fd.rules.is_empty() {
                pending.push(Pending { reg: ri, field: fi, at: fat.clone(), rules: fd.rules.clone() });
            }
            fields.push(Field {
                name: leak(&fd.name),
                desc: leak(fd.desc.as_deref().unwrap_or("")),
                lsb,
                width: fw,
                access: faccess,
                encoding,
                scale: safe_scale,
                offset: offset_v,
                unit: leak(fd.unit.as_deref().unwrap_or("")),
                min,
                max,
                enums: enums.leak(),
                rules: &[],
                confirm: fd.confirm,
                min_interval_ms: fd.min_interval_ms,
            });
        }
        let reset = match &r.reset {
            Some(n) => match n.as_i64() {
                Ok(v) => {
                    if r.fields.iter().any(|f| f.reset.is_some()) && (v as u32) != reset_from_fields {
                        err(&at, format!("register reset {v:#x} disagrees with the field resets ({reset_from_fields:#x})"));
                    }
                    v as u32
                }
                Err(e) => {
                    err(&at, format!("reset: {e}"));
                    0
                }
            },
            None => reset_from_fields,
        };
        let nf = fields.len();
        regs.push((
            Register {
                name: leak(&r.name),
                desc: leak(r.desc.as_deref().unwrap_or("")),
                offset: offset as u32,
                width,
                access,
                reset,
                read_side_effect: r.read_side_effect,
                fields: &[],
                first_field,
            },
            fields,
        ));
        first_field += nf;
    }

    // Pass 2: rules, now that every field can be named.
    let lookup = |name: &str| -> Option<FieldRef> {
        let (rn, fname) = name.split_once('.')?;
        let ri = *reg_names.get(&rn.to_ascii_uppercase())?;
        let fi = regs[ri].1.iter().position(|f| f.name.eq_ignore_ascii_case(fname))?;
        Some(FieldRef { reg: ri as u16, field: fi as u16 })
    };
    let mut rules_by_field: HashMap<(usize, usize), Vec<Rule<'static>>> = HashMap::new();
    for p in &pending {
        let owner = FieldRef { reg: p.reg as u16, field: p.field as u16 };
        for (k, rd) in p.rules.iter().enumerate() {
            let rat = format!("{}.rules[{k}]", p.at);
            let (when, require) = {
                let mut parse = |text: &str, what: &str| -> Option<Cond> {
                    match parse_cond(text, owner, &lookup, &regs) {
                        Ok(c) => Some(c),
                        Err(m) => {
                            err(&rat, format!("{what} '{text}': {m}"));
                            None
                        }
                    }
                };
                // Outer None = the `when` text did not parse; Some(None) = no `when`.
                let when: Option<Option<Cond>> = match rd.when.as_deref() {
                    None => Some(None),
                    Some(t) => parse(t, "when").map(Some),
                };
                (when, parse(&rd.require, "require"))
            };
            // Every field a rule reads from ANOTHER register is read from the
            // device at write time, so it must be safe to read.
            let conds: Vec<Cond> = when.iter().flatten().copied().chain(require.iter().copied()).collect();
            for c in &conds {
                for op in [c.lhs, c.rhs] {
                    if let Operand::Field(fr) = op {
                        let (oreg, ofields) = &regs[fr.reg as usize];
                        if fr.reg as usize != p.reg && oreg.read_side_effect {
                            err(&rat, format!("refers to {} whose read has a side effect", oreg.name));
                        }
                        if fr.reg as usize != p.reg && !ofields[fr.field as usize].access.readable() {
                            err(&rat, format!("refers to {}.{} which cannot be read", oreg.name, ofields[fr.field as usize].name));
                        }
                    }
                }
            }
            if rd.why.trim().is_empty() {
                err(&rat, "every rule needs a 'why' (it is what the model is told)".into());
            }
            if let (Some(when), Some(require)) = (when, require) {
                rules_by_field.entry((p.reg, p.field)).or_default().push(Rule { when, require, why: leak(&rd.why) });
            }
        }
    }

    if !issues.is_empty() {
        return Err(SpecError(issues));
    }

    let mut registers: Vec<Register<'static>> = Vec::new();
    for (ri, (mut reg, mut fields)) in regs.into_iter().enumerate() {
        for (fi, f) in fields.iter_mut().enumerate() {
            if let Some(rs) = rules_by_field.remove(&(ri, fi)) {
                f.rules = rs.leak();
            }
        }
        reg.fields = fields.leak();
        registers.push(reg);
    }
    let dev = Device {
        name: leak(&dname),
        title: leak(doc.device.title.as_deref().unwrap_or(&doc.device.name)),
        bus,
        registers: registers.leak(),
        field_count: first_field,
    };
    Ok(Box::leak(Box::new(dev)))
}

/// `a OP b` where each side is `self`, `REG.FIELD`, an enum name or a number.
fn parse_cond(
    text: &str,
    owner: FieldRef,
    lookup: &dyn Fn(&str) -> Option<FieldRef>,
    regs: &[(Register<'static>, Vec<Field<'static>>)],
) -> Result<Cond, String> {
    const OPS: [(&str, Op); 6] = [("<=", Op::Le), (">=", Op::Ge), ("==", Op::Eq), ("!=", Op::Ne), ("<", Op::Lt), (">", Op::Gt)];
    let (pos, tok, op) = OPS
        .iter()
        .filter_map(|(t, o)| text.find(t).map(|p| (p, *t, *o)))
        .min_by_key(|(p, t, _)| (*p, usize::MAX - t.len()))
        .ok_or("no comparison (one of == != < <= > >=)")?;
    let (a, b) = (text[..pos].trim(), text[pos + tok.len()..].trim());
    if a.is_empty() || b.is_empty() {
        return Err("a side of the comparison is empty".into());
    }
    let field_of = |s: &str| -> Option<Result<(Operand, FieldRef), String>> {
        if s.eq_ignore_ascii_case("self") {
            return Some(Ok((Operand::New, owner)));
        }
        if s.contains('.') && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
            return Some(lookup(s).map(|r| (Operand::Field(r), r)).ok_or_else(|| format!("no field {s}")));
        }
        None
    };
    let fa = field_of(a).transpose()?;
    let fb = field_of(b).transpose()?;
    let fld = |r: FieldRef| &regs[r.reg as usize].1[r.field as usize];
    let constant = |s: &str, other: Option<FieldRef>| -> Result<Operand, String> {
        let f = other.map(fld).ok_or("a comparison needs a field on at least one side")?;
        if let Some(e) = f.enums.iter().find(|e| e.name.eq_ignore_ascii_case(s)) {
            return Ok(Operand::Const(e.value));
        }
        if f.encoding == Encoding::Bool {
            match s {
                "true" => return Ok(Operand::Const(1)),
                "false" => return Ok(Operand::Const(0)),
                _ => {}
            }
        }
        let p = Dec::parse(s).ok_or_else(|| format!("'{s}' is not a number, an enum name of {} or a field", f.name))?;
        match dec::to_count(p, f.scale, f.offset, Round::Nearest) {
            Some((c, true)) => Ok(Operand::Const(c)),
            Some((_, false)) => Err(format!("{s} is not a whole step of {} (resolution {})", f.name, f.scale)),
            None => Err(format!("{s} is far outside {}", f.name)),
        }
    };
    let lhs = match fa {
        Some((o, _)) => o,
        None => constant(a, fb.map(|x| x.1))?,
    };
    let rhs = match fb {
        Some((o, _)) => o,
        None => constant(b, fa.map(|x| x.1))?,
    };
    if let (Some((_, x)), Some((_, y))) = (fa, fb) {
        let (fx, fy) = (fld(x), fld(y));
        if fx.scale != fy.scale || fx.offset != fy.offset {
            return Err(format!("{} and {} have different scales; compare a field with a constant instead", fx.name, fy.name));
        }
    }
    Ok(Cond { lhs, op, rhs })
}
