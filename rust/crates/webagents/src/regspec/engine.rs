//! The tool engine: turns a [`Device`] table into model-callable tools and
//! enforces the envelope on every write before the bus sees it.
//!
//! Two tools per register: `<device>_read_<register>` (readable registers) and
//! `<device>_set_<register>` (registers with writable fields). One tool per
//! register rather than per field keeps a part with 30 registers at a tool list a
//! model can hold, and lets one call set fields that must change together (a mode
//! bit and its enable) with the rules judged on the combined new value.
//!
//! ORDER OF CHECKS ON A WRITE, every one before any bus write:
//! 1. the arguments are a flat object whose keys are writable fields of the
//!    register, each named once, each of the field's type;
//! 2. each value converts to a whole count (physical values are rounded to the
//!    field's resolution and the rounding is REPORTED in the result);
//! 3. each count is inside the field's range;
//! 4. confirmation (granted by the host, never by the model) and rate limits;
//! 5. every rule that can be affected: the rules of the fields being written,
//!    AND the rules of any other field, in any register, that refer to a field
//!    being written. So `DUTY <= PERIOD` written once on DUTY also refuses a
//!    PERIOD write that would put PERIOD under the present DUTY.
//!
//! Fields not named in a set call keep their present value (read-modify-write),
//! except trigger (`W1`) fields, which are written as 0, and write-only fields,
//! which are written at their reset value. A register whose read has a side
//! effect is never read behind the caller's back: every writable field of such a
//! register must be named.

use core::fmt::{self, Write};

use super::dec::{self, Dec, Round};
use super::json::{self, FlatObject, JsonError, Value};
use super::{Access, Device, Encoding, Field, FieldRef, Operand, Register, Rule};
use crate::bus::RegisterBus;

/// What the HOST knows about a call that the model cannot assert.
///
/// Both values come from the code that runs the engine, never from a tool
/// call's arguments: a model that could say "confirmed" or set the clock
/// would be granting itself its own limits.
///
/// ```
/// # #[cfg(feature = "std")] {
/// use webagents::bus::MemBus;
/// use webagents::regspec::{load::load_json, CallCtx, Engine, Refusal, ToolError};
///
/// // A valve position that may move at most once a second.
/// let dev = load_json(
///     r#"{ "device": { "name": "valve" },
///          "registers": [ { "offset": 0, "name": "POS",
///            "fields": [ { "bits": "7:0", "name": "POS", "min_interval_ms": 1000 } ] } ] }"#,
/// )
/// .unwrap();
/// let mut engine = Engine::new(dev, MemBus::<1>::new(), [0u64; 1]);
/// let mut out = [0u8; 256];
///
/// let at = |now_ms| CallCtx { now_ms, confirmed: false };
/// engine.call("valve_set_pos", r#"{"pos": 10}"#, &at(0), &mut out).unwrap();
/// // 400 ms later is too soon; the refusal says how long to wait.
/// let err = engine.call("valve_set_pos", r#"{"pos": 20}"#, &at(400), &mut out).unwrap_err();
/// assert!(matches!(err, ToolError::Refused(Refusal::RateLimited { wait_ms: 600, .. })));
/// assert_eq!(engine.bus.get(0), 10);
/// engine.call("valve_set_pos", r#"{"pos": 20}"#, &at(1000), &mut out).unwrap();
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CallCtx {
    /// Monotonic milliseconds, for rate limits. Any clock that never goes
    /// backwards works (time since boot on a microcontroller); it is only
    /// compared with the time of earlier calls on the same engine.
    pub now_ms: u64,
    /// A person confirmed this specific call (the gateway asked them). Fields
    /// marked `confirm` are refused without it. Never set this from the model's
    /// own arguments.
    pub confirmed: bool,
}

/// A requested value as reported back: the model's own text, or a count from typed code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requested<'x> {
    /// The value exactly as the tool call's arguments spelled it (`0.2`),
    /// borrowed from the argument text, so the refusal quotes what was asked.
    Text(&'x str),
    /// A count that came from typed code ([`Engine::write_word`]) or from a
    /// JSON `true`/`false`; rendered in the field's physical units.
    Count(i64),
}

/// Why a write was refused. Rendered for the model by [`ToolError::write_json`].
///
/// Every refusal is decided before the bus write, so the register keeps its
/// value: the rendered message ends with "Nothing was written." Each variant
/// names the field by position ([`FieldRef`]); the renderer turns that into
/// `REGISTER.FIELD` and the limits into the field's units.
///
/// ```
/// # #[cfg(feature = "std")] {
/// use webagents::bus::MemBus;
/// use webagents::regspec::{load::load_json, CallCtx, Engine, Refusal, ToolError};
///
/// let dev = load_json(include_str!("../../../../fixtures/pwm1.regspec.json")).unwrap();
/// let mut engine = Engine::new(dev, MemBus::<32>::new(), vec![0u64; dev.field_count]);
/// let ctx = CallCtx::default();
/// let mut out = [0u8; 512];
/// let render = |e: ToolError| {
///     let mut s = String::new();
///     e.write_json(dev, &mut s).unwrap();
///     s
/// };
///
/// // Outside a range: the dead time may not go below 0.5 us.
/// let err = engine.call("pwm1_set_deadtime", r#"{"deadtime": 0.2}"#, &ctx, &mut out).unwrap_err();
/// assert!(matches!(err, ToolError::Refused(Refusal::OutOfRange { .. })));
/// assert_eq!(
///     render(err),
///     r#"{"ok":false,"error":"out_of_range","message":"DEADTIME = 0.2 is outside its limits 0.5 to 25.5 us. Nothing was written."}"#
/// );
///
/// // Against a rule: the on-time may not exceed the period (50 us here).
/// engine.call("pwm1_set_period", r#"{"period": 50}"#, &ctx, &mut out).unwrap();
/// let err = engine.call("pwm1_set_duty", r#"{"duty": 60}"#, &ctx, &mut out).unwrap_err();
/// assert!(matches!(err, ToolError::Refused(Refusal::Rule { lhs: 600, rhs: 500, .. })));
/// assert_eq!(
///     render(err),
///     r#"{"ok":false,"error":"rule","message":"the on-time must not exceed the PWM period (60 us is not <= 50 us). Nothing was written."}"#
/// );
///
/// // The rule guards both of its fields: shrinking the period under the
/// // present on-time is refused by the same rule, written once on DUTY.
/// engine.call("pwm1_set_duty", r#"{"duty": 40}"#, &ctx, &mut out).unwrap();
/// let err = engine.call("pwm1_set_period", r#"{"period": 30}"#, &ctx, &mut out).unwrap_err();
/// assert!(matches!(err, ToolError::Refused(Refusal::Rule { .. })));
///
/// // A field marked `confirm` needs the host's word that a person agreed.
/// let err = engine.call("pwm1_set_uart_div", r#"{"div": 100}"#, &ctx, &mut out).unwrap_err();
/// assert!(matches!(err, ToolError::Refused(Refusal::NeedsConfirmation { .. })));
/// let confirmed = CallCtx { now_ms: 0, confirmed: true };
/// engine.call("pwm1_set_uart_div", r#"{"div": 100}"#, &confirmed, &mut out).unwrap();
///
/// // Three writes went through; none of the refused calls reached the bus.
/// assert_eq!(engine.bus.writes, 3);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal<'x> {
    /// The value is below the field's minimum or above its maximum (the
    /// spec's `range`, or what the field's bits can hold). Code `out_of_range`.
    OutOfRange {
        /// The field that was asked to take the value.
        field: FieldRef,
        /// What was asked for, quoted back in the message.
        requested: Requested<'x>,
    },
    /// A count field (no scale) was given a fraction. Scaled fields round to
    /// their resolution and report it; a count has nothing to round to.
    /// Code `not_a_whole_count`.
    NotAWholeCount {
        /// The field that counts whole steps.
        field: FieldRef,
        /// The number as the arguments spelled it.
        requested: &'x str,
    },
    /// Not a number this field can use (more than 18 significant digits).
    /// Code `bad_number`.
    BadNumber {
        /// The field the number was meant for.
        field: FieldRef,
    },
    /// A mode field was given a name, or a number, that is not one of its
    /// named values. The message lists the names. Code `bad_value`.
    BadEnum {
        /// The mode field.
        field: FieldRef,
    },
    /// The JSON type does not fit the field: a boolean for a number, a number
    /// other than 0 or 1 for a flag, an explicit type the field cannot read.
    /// Code `wrong_type`.
    WrongType {
        /// The field that was given the value.
        field: FieldRef,
    },
    /// The arguments name a field that cannot be written. Code `read_only`.
    ReadOnly {
        /// The read-only field.
        field: FieldRef,
    },
    /// The field is marked `confirm` and [`CallCtx::confirmed`] was false.
    /// Code `needs_confirmation`.
    NeedsConfirmation {
        /// The field that needs a person's confirmation.
        field: FieldRef,
    },
    /// The field was written less than its `min_interval_ms` ago.
    /// Code `rate_limited`.
    RateLimited {
        /// The rate-limited field.
        field: FieldRef,
        /// Milliseconds until a write of this field is accepted again.
        wait_ms: u64,
    },
    /// `owner`'s rule number `rule` did not hold; `lhs`/`rhs` are the integer values compared.
    /// The owner need not be the field being written: a rule is checked on a
    /// write of any field it names. Code `rule`.
    Rule {
        /// The field whose rule list holds the rule.
        owner: FieldRef,
        /// Index into the owner's [`Field::rules`].
        rule: u16,
        /// The left side of the rule's `require`, as a count, as it would be
        /// after the write.
        lhs: i64,
        /// The right side, likewise.
        rhs: i64,
    },
    /// The register's read has a side effect, so every writable field must be named.
    /// Code `must_name_all_fields`.
    MustNameAll {
        /// Index of the register in [`Device::registers`].
        reg: u16,
    },
}

/// Why a tool call did not produce a result.
///
/// [`ToolError::Refused`] is the envelope saying no; the other variants are a
/// call that could not be understood or carried out. Whatever the variant,
/// [`ToolError::write_json`] renders it as the tool result the model reads,
/// `{"ok":false,"error":"<code>","message":"..."}`, and nothing was written
/// to the bus, with two exceptions the message states: after
/// [`ToolError::Bus`] the register's state is unknown, and
/// [`ToolError::OutputTooSmall`] is raised after a write was done.
///
/// The lifetime `'x` is that of the argument text: names and numbers in an
/// error are borrowed from it, so no error allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolError<'x> {
    /// The name is not `<device>_read_<register>` or `<device>_set_<register>`
    /// for a register of this device that can be read, or written, that way.
    /// Code `unknown_tool`.
    UnknownTool,
    /// The arguments are not a flat JSON object. Code `bad_arguments`.
    Json(JsonError),
    /// A key that is not a field of the register (or any key at all on a
    /// `read` tool, which takes none). Code `unknown_argument`.
    UnknownArg(&'x str),
    /// The same field named twice in one call. Code `duplicate_argument`.
    DuplicateArg(&'x str),
    /// A `set` call that names no field, or only fields set to `null`.
    /// Code `no_arguments`.
    NoArgs,
    /// The envelope refused the write; see [`Refusal`] for the reasons.
    Refused(Refusal<'x>),
    /// The transport failed. The engine does not know what was sent, so it says so.
    /// Code `bus_error`.
    Bus,
    /// `out` was too small for the result. Code `output_too_small`.
    OutputTooSmall,
}

/// The engine over one device.
///
/// `B` is the bus the registers are reached through and `S` the limit state,
/// one `u64` per field of the device (the time of its last write, for rate
/// limits): a `[u64; N]` on a microcontroller, a `Vec<u64>` on a host. Nothing
/// else is kept between calls, and nothing is allocated by a call.
///
/// One engine serves one device. It is not synchronized: a caller that runs
/// calls from several threads puts it behind a lock (the gateway does), so two
/// calls can never interleave a read-modify-write of one register.
pub struct Engine<'d, B, S> {
    /// The device table the engine enforces.
    pub dev: &'d Device<'d>,
    /// The bus. Public so the owner can read a register directly or inspect a
    /// test bus; a write made here bypasses every check.
    pub bus: B,
    /// Last write time per field (device-wide numbering); `u64::MAX` = never.
    last_write: S,
}

const NEVER: u64 = u64::MAX;
const MAX_FIELDS_PER_REG: usize = 32;

impl<'d, B: RegisterBus, S: AsMut<[u64]> + AsRef<[u64]>> Engine<'d, B, S> {
    /// `state` must hold at least `dev.field_count` slots (`[0u64; N]` or a `Vec`).
    /// Its contents are overwritten: every field starts as "never written".
    ///
    /// # Panics
    /// If `state` is shorter than `dev.field_count`.
    pub fn new(dev: &'d Device<'d>, bus: B, mut state: S) -> Self {
        assert!(state.as_ref().len() >= dev.field_count, "limit state smaller than the device's field count");
        for s in state.as_mut().iter_mut() {
            *s = NEVER;
        }
        Engine { dev, bus, last_write: state }
    }

    // ---- integer <-> bits <-> physical ---------------------------------------------------

    /// The count of field `f` in the register word `raw`: the field's bits,
    /// sign-extended or sign-and-magnitude decoded as its encoding says.
    pub fn decode(f: &Field, raw: u32) -> i64 {
        let bits = (raw & f.mask()) >> f.lsb;
        match f.encoding {
            Encoding::Unsigned | Encoding::Bool => bits as i64,
            Encoding::Signed => {
                let shift = 64 - f.width as u32;
                ((bits as i64) << shift) >> shift
            }
            Encoding::SignMagnitude => {
                let sign = 1u32 << (f.width - 1);
                let mag = (bits & (sign - 1)) as i64;
                if bits & sign != 0 {
                    -mag
                } else {
                    mag
                }
            }
        }
    }

    /// Bits of `v` for field `f`, placed at the field's position. `v` must be in range.
    pub fn encode(f: &Field, v: i64) -> u32 {
        let bits: u32 = match f.encoding {
            Encoding::Unsigned | Encoding::Bool | Encoding::Signed => v as u32,
            Encoding::SignMagnitude => {
                let sign = 1u32 << (f.width - 1);
                if v < 0 {
                    sign | ((-v) as u32)
                } else {
                    v as u32
                }
            }
        };
        (bits << f.lsb) & f.mask()
    }

    /// The physical value of count `v`, exactly, when it fits a [`Dec`].
    pub fn physical(f: &Field, v: i64) -> Option<Dec> {
        dec::physical(v, f.scale, f.offset)
    }

    fn reg(&self, i: usize) -> &'d Register<'d> {
        &self.dev.registers[i]
    }
    fn field(&self, r: FieldRef) -> &'d Field<'d> {
        &self.dev.registers[r.reg as usize].fields[r.field as usize]
    }

    fn read_reg(&mut self, i: usize) -> Result<u32, ToolError<'static>> {
        let r = self.reg(i);
        self.bus.read(r.offset, r.width).map_err(|_| ToolError::Bus)
    }

    // ---- tool names ----------------------------------------------------------------------

    /// `(register index, is_set)` for a tool name, or `None`.
    ///
    /// A name is `<device>_read_<register>` or `<device>_set_<register>`,
    /// matched without regard to case. `None` also for a `read` of a register
    /// that cannot be read and a `set` of one with no writable field, so a
    /// name resolves exactly when [`write_tools`] lists it.
    pub fn resolve(&self, tool: &str) -> Option<(usize, bool)> {
        let rest = strip_prefix_ci(tool, self.dev.name)?.strip_prefix('_')?;
        let (is_set, reg) = match rest.strip_prefix("read_") {
            Some(r) => (false, r),
            None => (true, rest.strip_prefix("set_")?),
        };
        let i = self.dev.registers.iter().position(|r| r.name.eq_ignore_ascii_case(reg))?;
        let r = self.reg(i);
        let ok = if is_set { r.fields.iter().any(|f| f.access.writable()) } else { r.access.readable() };
        ok.then_some((i, is_set))
    }

    // ---- calls ---------------------------------------------------------------------------

    /// Run one tool call. On success the JSON result is in `out[..n]`. On error,
    /// render it with [`ToolError::write_json`] and return it to the model as an
    /// error result: it says what was refused and why, in the part's units.
    ///
    /// `args` is the call's arguments as JSON text: a flat object keyed by
    /// field name for a `set` tool (`{"deadtime": 1.5}`), empty or `{}` for a
    /// `read` tool. A result is a few hundred bytes for a typical register;
    /// 512 bytes of `out` holds any register of up to a dozen fields.
    ///
    /// A read answers `{"register", "raw", "fields", "units"}`; a write answers
    /// `{"ok": true, "register", "written", "applied", "rounded", "units"}`,
    /// where `rounded` appears only when a physical value was rounded to the
    /// field's resolution and `units` only for fields that have one.
    ///
    /// ```
    /// # #[cfg(feature = "std")] {
    /// use webagents::bus::MemBus;
    /// use webagents::regspec::{load::load_json, CallCtx, Engine};
    ///
    /// let dev = load_json(include_str!("../../../../fixtures/pwm1.regspec.json")).unwrap();
    /// let mut engine = Engine::new(dev, MemBus::<32>::new(), vec![0u64; dev.field_count]);
    /// let ctx = CallCtx::default();
    /// let mut out = [0u8; 512];
    ///
    /// // 1.23 us is between two 0.1 us steps: the nearest is applied and reported.
    /// let n = engine.call("pwm1_set_deadtime", r#"{"deadtime": 1.23}"#, &ctx, &mut out).unwrap();
    /// assert_eq!(
    ///     core::str::from_utf8(&out[..n]).unwrap(),
    ///     r#"{"ok":true,"register":"DEADTIME","written":"0x0000000c","applied":{"DEADTIME":1.2},"rounded":{"DEADTIME":{"requested":1.23,"applied":1.2}},"units":{"DEADTIME":"us"}}"#
    /// );
    ///
    /// let n = engine.call("pwm1_read_deadtime", "{}", &ctx, &mut out).unwrap();
    /// assert_eq!(
    ///     core::str::from_utf8(&out[..n]).unwrap(),
    ///     r#"{"register":"DEADTIME","raw":"0x0000000c","fields":{"DEADTIME":1.2},"units":{"DEADTIME":"us"}}"#
    /// );
    /// # }
    /// ```
    pub fn call<'x>(&mut self, tool: &str, args: &'x str, ctx: &CallCtx, out: &mut [u8]) -> Result<usize, ToolError<'x>> {
        let (ri, is_set) = self.resolve(tool).ok_or(ToolError::UnknownTool)?;
        if is_set {
            self.set(ri, args, ctx, out)
        } else {
            FlatObject::parse(args).map_err(ToolError::Json)?.next().map_or(Ok(()), |m| match m {
                Ok((k, _)) => Err(ToolError::UnknownArg(k)),
                Err(e) => Err(ToolError::Json(e)),
            })?;
            let raw = self.read_reg(ri).map_err(widen)?;
            let mut w = json::BufWriter::new(out);
            self.write_read_result(&mut w, ri, raw).map_err(|_| ToolError::OutputTooSmall)?;
            w.finish().ok_or(ToolError::OutputTooSmall)
        }
    }

    fn set<'x>(&mut self, ri: usize, args: &'x str, ctx: &CallCtx, out: &mut [u8]) -> Result<usize, ToolError<'x>> {
        let reg = self.reg(ri);
        let n = reg.fields.len();
        debug_assert!(n <= MAX_FIELDS_PER_REG);
        let mut new: [Option<i64>; MAX_FIELDS_PER_REG] = [None; MAX_FIELDS_PER_REG];
        // The model's text for fields whose value was rounded to the resolution.
        let mut rounded: [Option<&'x str>; MAX_FIELDS_PER_REG] = [None; MAX_FIELDS_PER_REG];
        let mut any = false;

        // 1-3: parse, convert, range.
        for m in FlatObject::parse(args).map_err(ToolError::Json)? {
            let (key, val) = m.map_err(ToolError::Json)?;
            let fi = reg.fields.iter().position(|f| f.name.eq_ignore_ascii_case(key)).ok_or(ToolError::UnknownArg(key))?;
            let fref = FieldRef { reg: ri as u16, field: fi as u16 };
            let f = &reg.fields[fi];
            if !f.access.writable() {
                return Err(ToolError::Refused(Refusal::ReadOnly { field: fref }));
            }
            if new[fi].is_some() {
                return Err(ToolError::DuplicateArg(key));
            }
            if val == Value::Null {
                continue; // an explicit null is "leave as is", same as absent
            }
            let (count, exact) = to_count(f, fref, val)?;
            if count < f.min || count > f.max {
                return Err(ToolError::Refused(Refusal::OutOfRange { field: fref, requested: requested(val) }));
            }
            new[fi] = Some(count);
            if !exact {
                if let Value::Number(t) = val {
                    rounded[fi] = Some(t);
                }
            }
            any = true;
        }
        if !any {
            return Err(ToolError::NoArgs);
        }

        let word = self.commit(ri, &new, ctx)?;
        let mut w = json::BufWriter::new(out);
        self.write_set_result(&mut w, ri, word, &new, &rounded).map_err(|_| ToolError::OutputTooSmall)?;
        w.finish().ok_or(ToolError::OutputTooSmall)
    }

    /// Write register `ri` with the whole word `word`, as typed firmware code does:
    /// every writable field counts as named (a trigger field only when its bit is
    /// set) and goes through the same range, confirmation, rate and rule checks as
    /// a tool call.
    ///
    /// `ri` is the register's index in [`Device::registers`]. A read/write
    /// field whose bits equal the register's present value is left out of the
    /// checks, so an unchanged confirm-only or rate-limited field does not
    /// block a write of its neighbours. This is what the generated `write_*`
    /// methods call, and what a part behind a gateway runs on every raw write
    /// it receives, so the limits hold even when the gateway is bypassed.
    ///
    /// # Panics
    /// If `ri` is not a register of the device.
    pub fn write_word(&mut self, ri: usize, word: u32, ctx: &CallCtx) -> Result<(), ToolError<'static>> {
        let reg = self.reg(ri);
        // Fields whose value does not change are not "named", so an unchanged
        // confirm-only or rate-limited field does not block a write of its
        // neighbours. That needs the present value, read only when safe to read.
        let current = if reg.access.readable() && !reg.read_side_effect { Some(self.read_reg(ri)?) } else { None };
        let mut new: [Option<i64>; MAX_FIELDS_PER_REG] = [None; MAX_FIELDS_PER_REG];
        for (fi, f) in reg.fields.iter().enumerate() {
            if !f.access.writable() || (f.access == Access::W1 && word & f.mask() == 0) {
                continue;
            }
            if f.access == Access::RW && current.is_some_and(|c| (c ^ word) & f.mask() == 0) {
                continue;
            }
            let v = Self::decode(f, word);
            if v < f.min || v > f.max {
                let field = FieldRef { reg: ri as u16, field: fi as u16 };
                return Err(ToolError::Refused(Refusal::OutOfRange { field, requested: Requested::Count(v) }));
            }
            new[fi] = Some(v);
        }
        self.commit(ri, &new, ctx).map(|_| ())
    }

    /// Steps 4 and 5 and the bus write, for already range-checked values.
    fn commit<'x>(&mut self, ri: usize, new: &[Option<i64>; MAX_FIELDS_PER_REG], ctx: &CallCtx) -> Result<u32, ToolError<'x>> {
        let reg = self.reg(ri);
        let n = reg.fields.len();
        // 4: confirmation and rate limits.
        for (fi, f) in reg.fields.iter().enumerate() {
            if new[fi].is_none() {
                continue;
            }
            let fref = FieldRef { reg: ri as u16, field: fi as u16 };
            if f.confirm && !ctx.confirmed {
                return Err(ToolError::Refused(Refusal::NeedsConfirmation { field: fref }));
            }
            if f.min_interval_ms > 0 {
                let last = self.last_write.as_ref()[reg.first_field + fi];
                if last != NEVER && ctx.now_ms < last.saturating_add(f.min_interval_ms as u64) {
                    let wait_ms = last + f.min_interval_ms as u64 - ctx.now_ms;
                    return Err(ToolError::Refused(Refusal::RateLimited { field: fref, wait_ms }));
                }
            }
        }

        // The base word the named fields are merged into.
        let unnamed_writable = reg.fields.iter().enumerate().any(|(i, f)| new[i].is_none() && f.access == Access::RW);
        let mut base = if reg.access.readable() && unnamed_writable {
            if reg.read_side_effect {
                return Err(ToolError::Refused(Refusal::MustNameAll { reg: ri as u16 }));
            }
            self.read_reg(ri).map_err(widen)?
        } else {
            reg.reset
        };
        for (i, f) in reg.fields.iter().enumerate() {
            match f.access {
                Access::W1 => base &= !f.mask(),
                Access::W if new[i].is_none() => base = (base & !f.mask()) | (reg.reset & f.mask()),
                _ => {}
            }
        }
        let mut word = base;
        for (i, f) in reg.fields.iter().enumerate() {
            if let Some(v) = new[i] {
                word = (word & !f.mask()) | Self::encode(f, v);
            }
        }

        // 5: rules.
        self.check_rules(ri, new, word)?;

        self.bus.write(reg.offset, reg.width, word).map_err(|_| ToolError::Bus)?;
        for (i, v) in new.iter().enumerate().take(n) {
            if v.is_some() {
                self.last_write.as_mut()[reg.first_field + i] = ctx.now_ms;
            }
        }
        Ok(word)
    }

    /// Value of `r` as it would be after writing `word` to register `ri`.
    fn value_after(&mut self, r: FieldRef, ri: usize, word: u32, cache: &mut RegCache) -> Result<i64, ToolError<'static>> {
        let f = self.field(r);
        if r.reg as usize == ri {
            return Ok(Self::decode(f, word));
        }
        let raw = match cache.get(r.reg) {
            Some(v) => v,
            None => {
                let v = self.read_reg(r.reg as usize)?;
                cache.put(r.reg, v);
                v
            }
        };
        Ok(Self::decode(f, raw))
    }

    fn check_rules<'x>(&mut self, ri: usize, new: &[Option<i64>; MAX_FIELDS_PER_REG], word: u32) -> Result<(), ToolError<'x>> {
        let dev = self.dev;
        let touches = |op: &Operand| matches!(op, Operand::Field(r) if r.reg as usize == ri && new[r.field as usize].is_some());
        let mut cache = RegCache::default();
        for (gri, reg) in dev.registers.iter().enumerate() {
            for (fi, f) in reg.fields.iter().enumerate() {
                let owner = FieldRef { reg: gri as u16, field: fi as u16 };
                let owner_written = gri == ri && new[fi].is_some();
                for (k, rule) in f.rules.iter().enumerate() {
                    let affected = owner_written
                        || rule.when.is_some_and(|c| touches(&c.lhs) || touches(&c.rhs))
                        || touches(&rule.require.lhs)
                        || touches(&rule.require.rhs);
                    if !affected {
                        continue;
                    }
                    if let Some((lhs, rhs)) = self.eval_rule(rule, owner, ri, word, &mut cache).map_err(widen)? {
                        return Err(ToolError::Refused(Refusal::Rule { owner, rule: k as u16, lhs, rhs }));
                    }
                }
            }
        }
        Ok(())
    }

    /// `Some((lhs, rhs))` when the rule FAILS.
    fn eval_rule(&mut self, rule: &Rule, owner: FieldRef, ri: usize, word: u32, cache: &mut RegCache) -> Result<Option<(i64, i64)>, ToolError<'static>> {
        if let Some(c) = rule.when {
            let a = self.operand(c.lhs, owner, ri, word, cache)?;
            let b = self.operand(c.rhs, owner, ri, word, cache)?;
            if !c.op.eval(a, b) {
                return Ok(None);
            }
        }
        let c = rule.require;
        let a = self.operand(c.lhs, owner, ri, word, cache)?;
        let b = self.operand(c.rhs, owner, ri, word, cache)?;
        Ok(if c.op.eval(a, b) { None } else { Some((a, b)) })
    }

    fn operand(&mut self, op: Operand, owner: FieldRef, ri: usize, word: u32, cache: &mut RegCache) -> Result<i64, ToolError<'static>> {
        match op {
            Operand::Const(v) => Ok(v),
            Operand::New => self.value_after(owner, ri, word, cache),
            Operand::Field(r) => self.value_after(r, ri, word, cache),
        }
    }

    // ---- results -------------------------------------------------------------------------

    fn write_field_value<W: Write>(&self, w: &mut W, f: &Field, v: i64) -> fmt::Result {
        if f.encoding == Encoding::Bool {
            return w.write_str(if v != 0 { "true" } else { "false" });
        }
        if let Some(e) = f.enums.iter().find(|e| e.value == v) {
            return json::write_str(w, e.name);
        }
        if f.is_scaled() {
            dec::write_physical(w, v, f.scale, f.offset)
        } else {
            write!(w, "{v}")
        }
    }

    fn write_read_result<W: Write>(&self, w: &mut W, ri: usize, raw: u32) -> fmt::Result {
        let reg = self.reg(ri);
        w.write_str("{\"register\":")?;
        json::write_str(w, reg.name)?;
        write!(w, ",\"raw\":\"0x{:0width$x}\",\"fields\":{{", raw, width = (reg.width as usize).div_ceil(4))?;
        let mut first = true;
        for f in reg.fields.iter().filter(|f| f.access.readable()) {
            if !first {
                w.write_char(',')?;
            }
            first = false;
            json::write_str(w, f.name)?;
            w.write_char(':')?;
            self.write_field_value(w, f, Self::decode(f, raw))?;
        }
        w.write_char('}')?;
        self.write_units(w, reg, |_| true)?;
        w.write_char('}')
    }

    fn write_units<W: Write>(&self, w: &mut W, reg: &Register, pick: impl Fn(usize) -> bool) -> fmt::Result {
        let mut any = false;
        for (i, f) in reg.fields.iter().enumerate() {
            if f.unit.is_empty() || !pick(i) {
                continue;
            }
            w.write_str(if any { "," } else { ",\"units\":{" })?;
            any = true;
            json::write_str(w, f.name)?;
            w.write_char(':')?;
            json::write_str(w, f.unit)?;
        }
        if any {
            w.write_char('}')?;
        }
        Ok(())
    }

    fn write_set_result<W: Write>(
        &self,
        w: &mut W,
        ri: usize,
        word: u32,
        new: &[Option<i64>; MAX_FIELDS_PER_REG],
        rounded: &[Option<&str>; MAX_FIELDS_PER_REG],
    ) -> fmt::Result {
        let reg = self.reg(ri);
        w.write_str("{\"ok\":true,\"register\":")?;
        json::write_str(w, reg.name)?;
        write!(w, ",\"written\":\"0x{:0width$x}\",\"applied\":{{", word, width = (reg.width as usize).div_ceil(4))?;
        let mut first = true;
        for (i, f) in reg.fields.iter().enumerate() {
            let Some(v) = new[i] else { continue };
            if !first {
                w.write_char(',')?;
            }
            first = false;
            json::write_str(w, f.name)?;
            w.write_char(':')?;
            self.write_field_value(w, f, v)?;
        }
        w.write_char('}')?;
        // Rounding to the field's resolution is reported, never silent.
        let mut any = false;
        for (i, f) in reg.fields.iter().enumerate() {
            let (Some(v), Some(text)) = (new[i], rounded[i]) else { continue };
            w.write_str(if any { "," } else { ",\"rounded\":{" })?;
            any = true;
            json::write_str(w, f.name)?;
            // `text` passed the JSON number grammar, so it is written as is.
            w.write_str(":{\"requested\":")?;
            w.write_str(text)?;
            w.write_str(",\"applied\":")?;
            dec::write_physical(w, v, f.scale, f.offset)?;
            w.write_char('}')?;
        }
        if any {
            w.write_char('}')?;
        }
        self.write_units(w, reg, |i| new[i].is_some())?;
        w.write_char('}')
    }

    // ---- descriptors ---------------------------------------------------------------------

    /// The tool list as OpenAI-style function tools (the shape UAMP's
    /// `session.tools` and the `/llm` proxy take). Deterministic: the same table
    /// always gives the same bytes, so it can be hashed and signed.
    ///
    /// For each register, in spec order: a `<device>_read_<register>` tool with
    /// no parameters when it is readable, then a `<device>_set_<register>` tool
    /// with one optional parameter per writable field. A field's schema states
    /// its type (`boolean`, a string `enum`, `integer` or `number`), its
    /// `minimum` and `maximum` in physical units, its unit as `x-unit`, and a
    /// description that carries the resolution, every rule that bounds it
    /// (also rules declared on other fields), and whether it needs
    /// confirmation or is rate limited.
    pub fn write_tools<W: Write>(&self, w: &mut W) -> fmt::Result {
        write_tools(self.dev, w)
    }
}

/// See [`Engine::write_tools`]; usable without a bus.
///
/// The output of the example device is about 8.5 KB. A part with little RAM
/// writes it straight into its link, or into a hash, instead of a buffer.
///
/// ```
/// # #[cfg(feature = "std")] {
/// use webagents::regspec::{engine::write_tools, load::load_json};
///
/// let dev = load_json(include_str!("../../../../fixtures/pwm1.regspec.json")).unwrap();
/// let mut json = String::new();
/// write_tools(dev, &mut json).unwrap();
///
/// let tools: serde_json::Value = serde_json::from_str(&json).unwrap();
/// let names: Vec<&str> = tools.as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
/// // STATUS is read only, so it has no set tool.
/// assert_eq!(&names[..4], ["pwm1_read_ctrl", "pwm1_set_ctrl", "pwm1_read_status", "pwm1_read_period"]);
///
/// let deadtime = &tools[8]["function"]["parameters"]["properties"]["deadtime"];
/// assert_eq!(tools[8]["function"]["name"], "pwm1_set_deadtime");
/// assert_eq!(deadtime["type"], "number");
/// assert_eq!(deadtime["minimum"], 0.5);
/// assert_eq!(deadtime["maximum"], 25.5);
/// assert_eq!(deadtime["x-unit"], "us");
/// # }
/// ```
pub fn write_tools<W: Write>(dev: &Device, w: &mut W) -> fmt::Result {
    w.write_char('[')?;
    let mut first = true;
    for (ri, reg) in dev.registers.iter().enumerate() {
        if reg.access.readable() {
            if !first {
                w.write_char(',')?;
            }
            first = false;
            write_read_tool(dev, reg, w)?;
        }
        if reg.fields.iter().any(|f| f.access.writable()) {
            if !first {
                w.write_char(',')?;
            }
            first = false;
            write_set_tool(dev, ri, reg, w)?;
        }
    }
    w.write_char(']')
}

fn write_name<W: Write>(w: &mut W, dev: &Device, verb: &str, reg: &Register) -> fmt::Result {
    w.write_char('"')?;
    for c in dev.name.chars().chain("_".chars()).chain(verb.chars()).chain("_".chars()).chain(reg.name.chars()) {
        w.write_char(c.to_ascii_lowercase())?;
    }
    w.write_char('"')
}

fn write_read_tool<W: Write>(dev: &Device, reg: &Register, w: &mut W) -> fmt::Result {
    w.write_str("{\"type\":\"function\",\"function\":{\"name\":")?;
    write_name(w, dev, "read", reg)?;
    w.write_str(",\"description\":")?;
    let mut d = DescBuf::<'_, W>::start(w)?;
    write!(d, "Read {} ({}) of the {}.", reg.name, trim_dot(reg.desc), dev.title)?;
    if reg.read_side_effect {
        d.write_str(" Reading it changes the device's state (see the description), so read it only when that is intended.")?;
    }
    d.end()?;
    w.write_str(",\"parameters\":{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}}}")
}

fn write_set_tool<W: Write>(dev: &Device, ri: usize, reg: &Register, w: &mut W) -> fmt::Result {
    w.write_str("{\"type\":\"function\",\"function\":{\"name\":")?;
    write_name(w, dev, "set", reg)?;
    w.write_str(",\"description\":")?;
    let mut d = DescBuf::<'_, W>::start(w)?;
    write!(d, "Write {} ({}) of the {}. Name only the fields to change; the others keep their value.", reg.name, trim_dot(reg.desc), dev.title)?;
    if reg.read_side_effect {
        d.write_str(" Every field must be named, because this register is not read before writing.")?;
    }
    d.write_str(" A value outside a field's limits is refused with the reason, not adjusted.")?;
    d.end()?;
    w.write_str(",\"parameters\":{\"type\":\"object\",\"properties\":{")?;
    let mut first = true;
    for (fi, f) in reg.fields.iter().enumerate() {
        if !f.access.writable() {
            continue;
        }
        if !first {
            w.write_char(',')?;
        }
        first = false;
        let lower = LowerName(f.name);
        write!(w, "\"{}\":{{", lower)?;
        if f.encoding == Encoding::Bool {
            w.write_str("\"type\":\"boolean\"")?;
        } else if !f.enums.is_empty() {
            w.write_str("\"type\":\"string\",\"enum\":[")?;
            for (i, e) in f.enums.iter().filter(|e| e.value >= f.min && e.value <= f.max).enumerate() {
                if i > 0 {
                    w.write_char(',')?;
                }
                json::write_str(w, e.name)?;
            }
            w.write_char(']')?;
        } else if f.is_scaled() {
            let (lo, hi) = physical_limits(f);
            w.write_str("\"type\":\"number\",\"minimum\":")?;
            dec::write_physical(w, lo, f.scale, f.offset)?;
            w.write_str(",\"maximum\":")?;
            dec::write_physical(w, hi, f.scale, f.offset)?;
            if !f.unit.is_empty() {
                w.write_str(",\"x-unit\":")?;
                json::write_str(w, f.unit)?;
            }
        } else {
            write!(w, "\"type\":\"integer\",\"minimum\":{},\"maximum\":{}", f.min, f.max)?;
        }
        w.write_str(",\"description\":")?;
        let mut d = DescBuf::<'_, W>::start(w)?;
        d.write_str(trim_dot(f.desc))?;
        d.write_char('.')?;
        if f.is_scaled() {
            write!(d, " In {}, resolution ", if f.unit.is_empty() { "units" } else { f.unit })?;
            f.scale.abs().write(&mut d)?;
            d.write_char('.')?;
        }
        for e in f.enums.iter().filter(|e| !e.desc.is_empty()) {
            write!(d, " {}: {}.", e.name, trim_dot(e.desc))?;
        }
        for rule in f.rules {
            write!(d, " Limit: {}.", trim_dot(rule.why))?;
        }
        // Rules on OTHER fields that name this one also bound it.
        for (ori, oreg) in dev.registers.iter().enumerate() {
            for (ofi, of) in oreg.fields.iter().enumerate() {
                if ori == ri && ofi == fi {
                    continue;
                }
                for rule in of.rules {
                    let me = FieldRef { reg: ri as u16, field: fi as u16 };
                    let names = |c: &super::Cond| c.lhs == Operand::Field(me) || c.rhs == Operand::Field(me);
                    if rule.when.as_ref().is_some_and(names) || names(&rule.require) {
                        write!(d, " Limit: {}.", trim_dot(rule.why))?;
                    }
                }
            }
        }
        if f.confirm {
            d.write_str(" Needs a person's confirmation; ask before calling.")?;
        }
        if f.min_interval_ms > 0 {
            write!(d, " At most one write every {} ms.", f.min_interval_ms)?;
        }
        if f.access == Access::W1 {
            d.write_str(" Write true to trigger; it reads back false.")?;
        }
        d.end()?;
        w.write_char('}')?;
    }
    w.write_str("},\"additionalProperties\":false,\"minProperties\":1}}}")
}

/// `REG.FIELD`, or just `REG` for a register that is one field of the same name.
struct FieldName<'a>(&'a str, &'a str);
impl fmt::Display for FieldName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.eq_ignore_ascii_case(self.1) {
            f.write_str(self.0)
        } else {
            write!(f, "{}.{}", self.0, self.1)
        }
    }
}

/// A count as the field shows it: an enum name, true/false, or the physical value with its unit.
fn write_value_in<W: Write>(w: &mut W, f: Option<&Field>, v: i64) -> fmt::Result {
    let Some(f) = f else { return write!(w, "{v}") };
    if let Some(e) = f.enums.iter().find(|e| e.value == v) {
        return w.write_str(e.name);
    }
    if f.encoding == Encoding::Bool {
        return w.write_str(if v != 0 { "true" } else { "false" });
    }
    dec::write_physical(w, v, f.scale, f.offset)?;
    if !f.unit.is_empty() {
        write!(w, " {}", f.unit)?;
    }
    Ok(())
}

fn trim_dot(s: &str) -> &str {
    s.trim().trim_end_matches('.')
}

struct LowerName<'a>(&'a str);
impl fmt::Display for LowerName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in self.0.chars() {
            f.write_char(c.to_ascii_lowercase())?;
        }
        Ok(())
    }
}

/// Writes a JSON string literal piecewise through `fmt::Write`, escaping as it goes.
struct DescBuf<'w, W: Write> {
    w: &'w mut W,
}
impl<'w, W: Write> DescBuf<'w, W> {
    fn start(w: &'w mut W) -> Result<Self, fmt::Error> {
        w.write_char('"')?;
        Ok(DescBuf { w })
    }
    fn end(self) -> fmt::Result {
        self.w.write_char('"')
    }
}
impl<W: Write> Write for DescBuf<'_, W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let mut tmp = EscapeInner(self.w);
        json::write_str(&mut tmp, s)
    }
}
/// Drops the surrounding quotes `json::write_str` adds.
struct EscapeInner<'a, W: Write>(&'a mut W);
impl<W: Write> Write for EscapeInner<'_, W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if s == "\"" {
            return Ok(());
        }
        self.0.write_str(s)
    }
    fn write_char(&mut self, c: char) -> fmt::Result {
        if c == '"' {
            return Ok(());
        }
        self.0.write_char(c)
    }
}

fn strip_prefix_ci<'a>(s: &'a str, p: &str) -> Option<&'a str> {
    if s.len() >= p.len() && s.is_char_boundary(p.len()) && s[..p.len()].eq_ignore_ascii_case(p) {
        Some(&s[p.len()..])
    } else {
        None
    }
}

fn widen<'x>(e: ToolError<'static>) -> ToolError<'x> {
    e
}

/// The counts at the low and high PHYSICAL limits (swapped for a negative scale).
fn physical_limits(f: &Field) -> (i64, i64) {
    if f.scale.is_negative() {
        (f.max, f.min)
    } else {
        (f.min, f.max)
    }
}

fn requested(v: Value<'_>) -> Requested<'_> {
    match v {
        Value::Number(t) => Requested::Text(t),
        Value::Bool(b) => Requested::Count(b as i64),
        Value::Str(t) => Requested::Text(t),
        Value::Null => Requested::Text("null"),
    }
}

/// Converts one argument to the field's integer value, and says whether that
/// was exact (a physical value between two steps is rounded to the nearest and
/// the rounding is reported).
fn to_count<'x>(f: &Field, fref: FieldRef, v: Value<'x>) -> Result<(i64, bool), ToolError<'x>> {
    let wrong = ToolError::Refused(Refusal::WrongType { field: fref });
    match v {
        Value::Bool(b) => {
            if f.encoding == Encoding::Bool || (f.width == 1 && f.access == Access::W1) {
                Ok((b as i64, true))
            } else {
                Err(wrong)
            }
        }
        Value::Str(s) => {
            let e = f.enums.iter().find(|e| e.name.eq_ignore_ascii_case(s)).ok_or(ToolError::Refused(Refusal::BadEnum { field: fref }))?;
            Ok((e.value, true))
        }
        Value::Number(text) => {
            let x = match Dec::try_parse(text) {
                Ok(x) => x,
                Err(dec::DecError::Huge) => return Err(ToolError::Refused(Refusal::OutOfRange { field: fref, requested: Requested::Text(text) })),
                Err(_) => return Err(ToolError::Refused(Refusal::BadNumber { field: fref })),
            };
            if f.encoding == Encoding::Bool {
                // 0/1 accepted for a flag; anything else is a type error.
                return match x {
                    Dec::ZERO => Ok((0, true)),
                    Dec::ONE => Ok((1, true)),
                    _ => Err(wrong),
                };
            }
            if !f.enums.is_empty() && !f.is_scaled() {
                // A mode field may be given by number too, but only a named value.
                return match dec::to_count(x, Dec::ONE, Dec::ZERO, Round::Nearest) {
                    Some((c, true)) if f.enums.iter().any(|e| e.value == c) => Ok((c, true)),
                    _ => Err(ToolError::Refused(Refusal::BadEnum { field: fref })),
                };
            }
            let (c, exact) = dec::to_count(x, f.scale, f.offset, Round::Nearest)
                .ok_or(ToolError::Refused(Refusal::OutOfRange { field: fref, requested: Requested::Text(text) }))?;
            if !f.is_scaled() && !exact {
                return Err(ToolError::Refused(Refusal::NotAWholeCount { field: fref, requested: text }));
            }
            Ok((c, exact))
        }
        Value::Null => Err(wrong),
    }
}

#[derive(Default)]
struct RegCache {
    slots: [(u16, u32); 4],
    n: usize,
}
impl RegCache {
    fn get(&self, reg: u16) -> Option<u32> {
        self.slots[..self.n].iter().find(|s| s.0 == reg).map(|s| s.1)
    }
    fn put(&mut self, reg: u16, v: u32) {
        if self.n < self.slots.len() {
            self.slots[self.n] = (reg, v);
            self.n += 1;
        }
    }
}

impl<'x> ToolError<'x> {
    /// The error as a tool result for the model: `{"ok":false,"error":code,"message":...}`.
    ///
    /// `dev` must be the device the error came from: field positions in the
    /// error are looked up in it. The message is a full sentence in the part's
    /// own names and units, written for a model to act on; `error` is a stable
    /// code for programs. Fails only when `w` does (a full buffer).
    pub fn write_json<W: Write>(&self, dev: &Device, w: &mut W) -> fmt::Result {
        let fname = |r: FieldRef| {
            let reg = &dev.registers[r.reg as usize];
            FieldName(reg.name, reg.fields[r.field as usize].name)
        };
        let fld = |r: FieldRef| &dev.registers[r.reg as usize].fields[r.field as usize];
        let code = match self {
            ToolError::UnknownTool => "unknown_tool",
            ToolError::Json(_) => "bad_arguments",
            ToolError::UnknownArg(_) => "unknown_argument",
            ToolError::DuplicateArg(_) => "duplicate_argument",
            ToolError::NoArgs => "no_arguments",
            ToolError::Refused(r) => match r {
                Refusal::OutOfRange { .. } => "out_of_range",
                Refusal::NotAWholeCount { .. } => "not_a_whole_count",
                Refusal::BadNumber { .. } => "bad_number",
                Refusal::BadEnum { .. } => "bad_value",
                Refusal::WrongType { .. } => "wrong_type",
                Refusal::ReadOnly { .. } => "read_only",
                Refusal::NeedsConfirmation { .. } => "needs_confirmation",
                Refusal::RateLimited { .. } => "rate_limited",
                Refusal::Rule { .. } => "rule",
                Refusal::MustNameAll { .. } => "must_name_all_fields",
            },
            ToolError::Bus => "bus_error",
            ToolError::OutputTooSmall => "output_too_small",
        };
        w.write_str("{\"ok\":false,\"error\":\"")?;
        w.write_str(code)?;
        w.write_str("\",\"message\":")?;
        let mut d = DescBuf::<'_, W>::start(w)?;
        match *self {
            ToolError::UnknownTool => d.write_str("No such tool on this device.")?,
            ToolError::Json(e) => write!(d, "{e}.")?,
            ToolError::UnknownArg(k) => write!(d, "{k} is not a field of this register.")?,
            ToolError::DuplicateArg(k) => write!(d, "{k} is named twice.")?,
            ToolError::NoArgs => d.write_str("Name at least one field to write.")?,
            ToolError::Bus => d.write_str("The bus transfer failed; the register's state is unknown. Read it back before retrying.")?,
            ToolError::OutputTooSmall => d.write_str("The result did not fit the output buffer. The write, if any, was done.")?,
            ToolError::Refused(r) => match r {
                Refusal::OutOfRange { field, requested } => {
                    let f = fld(field);
                    let nm = fname(field);
                    let (lo, hi) = physical_limits(f);
                    write!(d, "{nm} = ")?;
                    match requested {
                        Requested::Text(t) => d.write_str(t)?,
                        Requested::Count(c) => dec::write_physical(&mut d, c, f.scale, f.offset)?,
                    }
                    d.write_str(" is outside its limits ")?;
                    dec::write_physical(&mut d, lo, f.scale, f.offset)?;
                    d.write_str(" to ")?;
                    dec::write_physical(&mut d, hi, f.scale, f.offset)?;
                    if !f.unit.is_empty() {
                        write!(d, " {}", f.unit)?;
                    }
                    d.write_str(". Nothing was written.")?;
                }
                Refusal::NotAWholeCount { field, requested } => {
                    let nm = fname(field);
                    write!(d, "{nm} counts whole steps; {requested} is not one. Nothing was written.")?;
                }
                Refusal::BadEnum { field } => {
                    let nm = fname(field);
                    let enums = fld(field).enums;
                    if enums.is_empty() {
                        // A string given to a field with no named values.
                        write!(d, "{nm} takes a number, not a name.")?;
                    } else {
                        write!(d, "{nm} takes one of:")?;
                        for e in enums {
                            write!(d, " {}", e.name)?;
                        }
                        d.write_char('.')?;
                    }
                    d.write_str(" Nothing was written.")?;
                }
                Refusal::BadNumber { field } => {
                    let nm = fname(field);
                    write!(d, "{nm} was given a number with more precision than any register can use. Nothing was written.")?;
                }
                Refusal::WrongType { field } => {
                    let nm = fname(field);
                    write!(d, "{nm} was given a value of the wrong type. Nothing was written.")?;
                }
                Refusal::ReadOnly { field } => {
                    let nm = fname(field);
                    write!(d, "{nm} is read only. Nothing was written.")?;
                }
                Refusal::NeedsConfirmation { field } => {
                    let nm = fname(field);
                    write!(
                        d,
                        "{nm} needs a person's confirmation for this write. Ask them, then the host retries with their confirmation. Nothing was written."
                    )?;
                }
                Refusal::RateLimited { field, wait_ms } => {
                    let nm = fname(field);
                    write!(d, "{nm} was written too recently; wait {wait_ms} ms. Nothing was written.")?;
                }
                Refusal::Rule { owner, rule, lhs, rhs } => {
                    let r = &fld(owner).rules[rule as usize];
                    // The compared values in the units of the fields they belong to.
                    let side = |op: Operand, other: Operand| match op {
                        Operand::New => Some(owner),
                        Operand::Field(f) => Some(f),
                        Operand::Const(_) => match other {
                            Operand::New => Some(owner),
                            Operand::Field(f) => Some(f),
                            Operand::Const(_) => None,
                        },
                    };
                    write!(d, "{} (", trim_dot(r.why))?;
                    write_value_in(&mut d, side(r.require.lhs, r.require.rhs).map(fld), lhs)?;
                    write!(d, " is not {} ", r.require.op.as_str())?;
                    write_value_in(&mut d, side(r.require.rhs, r.require.lhs).map(fld), rhs)?;
                    d.write_str("). Nothing was written.")?;
                }
                Refusal::MustNameAll { reg } => {
                    write!(
                        d,
                        "{} is not read before writing (its read has a side effect), so name every writable field. Nothing was written.",
                        dev.registers[reg as usize].name
                    )?;
                }
            },
        }
        d.end()?;
        w.write_char('}')
    }
}
