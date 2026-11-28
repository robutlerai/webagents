//! Exact decimal numbers for units: `value = m * 10^e`.
//!
//! WHY NOT f64. A device envelope compares what a model asked for with a limit a
//! person wrote, and reports what was applied. With binary floats, "0.1 us steps"
//! are not exact, `150 mV` can come back as `149.99999999999997`, and a request on
//! the limit can land one count either side of it. The float parse and format
//! code alone is about 30 KB of a 56 KB Cortex-M33 image (that core has no
//! double-precision FPU, so every f64 operation is software).
//! Datasheet scales are decimal (`2.2 mV`, `0.25 us`, `2^-12 turn` written out),
//! so they are exact as decimals, and every conversion here is integer math with
//! the rounding mode stated.

use core::fmt;

/// `m * 10^e`, kept normalized (no trailing zeros in `m`, zero is `0e0`) so
/// equality is equality of value.
///
/// Build one with [`Dec::new`], [`Dec::from_int`] or [`Dec::parse`] rather
/// than by its fields, so it is normalized. At most 18 significant digits and
/// exponents from -128 to 127: more than any register field can use.
///
/// ```
/// use webagents::regspec::{dec, Dec};
///
/// let step = Dec::parse("0.1").unwrap(); // 0.1 us per count
/// assert_eq!(step, Dec::new(1, -1));
/// // 1.5 us is exactly count 15; the physical value of count 15 prints as 1.5.
/// assert_eq!(dec::to_count(Dec::parse("1.5").unwrap(), step, Dec::ZERO, dec::Round::Nearest), Some((15, true)));
/// assert_eq!(dec::physical(15, step, Dec::ZERO).unwrap().to_string(), "1.5");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dec {
    /// The mantissa: the value's significant digits as an integer, with its sign.
    pub m: i64,
    /// The power of ten the mantissa is multiplied by (`-2` for hundredths).
    pub e: i8,
}

/// Why a number did not parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecError {
    /// Not the JSON number grammar (`01`, `1.`, `.5`, `+1`, `0x10`, an empty string).
    Syntax,
    /// More than 18 significant digits.
    TooPrecise,
    /// Magnitude beyond 10^127: outside every field.
    Huge,
    /// Non-zero but below 10^-128.
    Tiny,
}

/// How a quotient that is not whole becomes a count.
///
/// A tool call's value rounds to the nearest step (and the result reports
/// it); a range bound rounds inwards, so a limit of 0.55 on a field with 0.1
/// steps never admits 0.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Round {
    /// Half away from zero.
    Nearest,
    /// Towards negative infinity. Used for an upper range bound, so the limit
    /// is never rounded outwards.
    Floor,
    /// Towards positive infinity. Used for a lower range bound.
    Ceil,
}

const POW10: [i128; 39] = {
    let mut t = [1i128; 39];
    let mut i = 1;
    while i < 39 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

fn pow10(n: i32) -> Option<i128> {
    if (0..39).contains(&n) {
        Some(POW10[n as usize])
    } else {
        None
    }
}

impl Dec {
    /// Zero. Also the offset of a field that has none.
    pub const ZERO: Dec = Dec { m: 0, e: 0 };
    /// One. Also the scale of a field that is a plain count.
    pub const ONE: Dec = Dec { m: 1, e: 0 };

    /// `m * 10^e`, normalized.
    pub const fn new(m: i64, e: i8) -> Dec {
        let mut m = m;
        let mut e = e;
        if m == 0 {
            return Dec::ZERO;
        }
        while m % 10 == 0 && e < i8::MAX {
            m /= 10;
            e += 1;
        }
        Dec { m, e }
    }

    /// The integer `v`, normalized (`1500` is stored as `15e2`).
    pub const fn from_int(v: i64) -> Dec {
        Dec::new(v, 0)
    }

    /// Whether the value is below zero. A negative scale swaps which count is
    /// the physical minimum.
    pub const fn is_negative(self) -> bool {
        self.m < 0
    }

    /// The magnitude. A field's resolution is `scale.abs()`.
    pub const fn abs(self) -> Dec {
        Dec { m: if self.m < 0 { -self.m } else { self.m }, e: self.e }
    }

    /// Parses a JSON number (`-12`, `0.25`, `1.5e-3`). `None` for anything else,
    /// or for more than 18 significant digits (more precision than any register
    /// field can use, refused rather than silently cut).
    pub fn parse(s: &str) -> Option<Dec> {
        Dec::try_parse(s).ok()
    }

    /// As [`Dec::parse`], saying why a number was not accepted.
    pub fn try_parse(s: &str) -> Result<Dec, DecError> {
        use DecError::*;
        let b = s.as_bytes();
        let mut i = 0;
        let neg = b.first() == Some(&b'-');
        if neg {
            i = 1;
        }
        let mut m: i64 = 0;
        let mut digits = 0u32;
        let mut exp: i32 = 0;
        let mut any = false;
        // Integer part: JSON forbids leading zeros except a lone 0.
        if i < b.len() && b[i] == b'0' {
            i += 1;
            any = true;
            if i < b.len() && b[i].is_ascii_digit() {
                return Err(Syntax);
            }
        }
        while i < b.len() && b[i].is_ascii_digit() {
            any = true;
            if m != 0 || b[i] != b'0' {
                digits += 1;
                if digits > 18 {
                    return Err(TooPrecise);
                }
            }
            m = m * 10 + (b[i] - b'0') as i64;
            i += 1;
        }
        if !any {
            return Err(Syntax);
        }
        if i < b.len() && b[i] == b'.' {
            i += 1;
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                if m != 0 || b[i] != b'0' {
                    digits += 1;
                    if digits > 18 {
                        // Trailing zeros past 18 digits carry no value; anything else is too precise.
                        if b[i] == b'0' {
                            i += 1;
                            continue;
                        }
                        return Err(TooPrecise);
                    }
                }
                m = m * 10 + (b[i] - b'0') as i64;
                exp -= 1;
                i += 1;
            }
            if i == start {
                return Err(Syntax);
            }
        }
        if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
            i += 1;
            let eneg = match b.get(i) {
                Some(b'-') => {
                    i += 1;
                    true
                }
                Some(b'+') => {
                    i += 1;
                    false
                }
                _ => false,
            };
            let start = i;
            let mut x: i32 = 0;
            while i < b.len() && b[i].is_ascii_digit() {
                x = x.saturating_mul(10).saturating_add((b[i] - b'0') as i32);
                i += 1;
            }
            if i == start {
                return Err(Syntax);
            }
            exp = exp.saturating_add(if eneg { -x } else { x });
        }
        if i != b.len() {
            return Err(Syntax);
        }
        if m == 0 {
            return Ok(Dec::ZERO);
        }
        // Normalize before the exponent range check, so 1000e-130 still fits.
        while m % 10 == 0 {
            m /= 10;
            exp += 1;
        }
        if exp > i8::MAX as i32 {
            return Err(Huge);
        }
        if exp < i8::MIN as i32 {
            return Err(Tiny);
        }
        Ok(Dec { m: if neg { -m } else { m }, e: exp as i8 })
    }

    /// The mantissa scaled to exponent `e` (which must not exceed `self.e`).
    fn at(self, e: i32) -> Option<i128> {
        (self.m as i128).checked_mul(pow10(self.e as i32 - e)?)
    }

    /// Writes the plain decimal form (`0.25`, `-45`, `132.4`), never an exponent.
    pub fn write<W: fmt::Write + ?Sized>(self, w: &mut W) -> fmt::Result {
        write_wide(w, self.m as i128, self.e as i32)
    }
}

impl fmt::Display for Dec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write(f)
    }
}

/// Writes `m * 10^e` as a plain decimal.
fn write_wide<W: fmt::Write + ?Sized>(w: &mut W, m: i128, e: i32) -> fmt::Result {
    let (mut m, mut e) = (m, e);
    while m != 0 && m % 10 == 0 {
        m /= 10;
        e += 1;
    }
    if m < 0 {
        w.write_char('-')?;
    }
    let u = m.unsigned_abs();
    let mut buf = [0u8; 40];
    let mut n = 0;
    let mut x = u;
    loop {
        buf[n] = b'0' + (x % 10) as u8;
        n += 1;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    // buf holds the digits reversed; n of them.
    let digits = |w: &mut W, from: usize, to: usize| -> fmt::Result {
        for k in (to..from).rev() {
            w.write_char(buf[k] as char)?;
        }
        Ok(())
    };
    if e >= 0 {
        digits(w, n, 0)?;
        for _ in 0..e {
            w.write_char('0')?;
        }
        return Ok(());
    }
    let frac = (-e) as usize;
    if n > frac {
        digits(w, n, frac)?;
        w.write_char('.')?;
        digits(w, frac, 0)
    } else {
        w.write_str("0.")?;
        for _ in 0..(frac - n) {
            w.write_char('0')?;
        }
        digits(w, n, 0)
    }
}

/// `num / den` rounded as asked; `den != 0`.
fn div_round(num: i128, den: i128, mode: Round) -> (i128, bool) {
    let q = num / den;
    let r = num % den;
    if r == 0 {
        return (q, true);
    }
    // The true quotient lies strictly between q and q + sign.
    let sign: i128 = if (num < 0) != (den < 0) { -1 } else { 1 };
    let q = match mode {
        Round::Nearest => {
            if 2 * r.unsigned_abs() >= den.unsigned_abs() {
                q + sign
            } else {
                q
            }
        }
        Round::Floor => {
            if sign < 0 {
                q - 1
            } else {
                q
            }
        }
        Round::Ceil => {
            if sign > 0 {
                q + 1
            } else {
                q
            }
        }
    };
    (q, false)
}

/// The count for physical value `v`: `(v - offset) / scale`, rounded as asked.
/// Returns the count and whether it was exact. `None` when the value is too
/// large to be any field's count (refuse it as out of range).
pub fn to_count(v: Dec, scale: Dec, offset: Dec, mode: Round) -> Option<(i64, bool)> {
    let e = (v.e as i32).min(scale.e as i32).min(offset.e as i32);
    let num = v.at(e)?.checked_sub(offset.at(e)?)?;
    let den = scale.at(e)?;
    if den == 0 {
        return None;
    }
    let (q, exact) = div_round(num, den, mode);
    i64::try_from(q).ok().map(|q| (q, exact))
}

/// Writes the physical value of `count`: `count * scale + offset`, exactly.
pub fn write_physical<W: fmt::Write + ?Sized>(w: &mut W, count: i64, scale: Dec, offset: Dec) -> fmt::Result {
    let e = (scale.e as i32).min(offset.e as i32);
    let v = (|| (count as i128).checked_mul(scale.at(e)?)?.checked_add(offset.at(e)?))();
    match v {
        Some(m) => write_wide(w, m, e),
        None => w.write_str("null"),
    }
}

/// The physical value of `count` as a `Dec`, when it fits.
pub fn physical(count: i64, scale: Dec, offset: Dec) -> Option<Dec> {
    let e = (scale.e as i32).min(offset.e as i32);
    let mut m = (count as i128).checked_mul(scale.at(e)?)?.checked_add(offset.at(e)?)?;
    let mut e = e;
    while m != 0 && m % 10 == 0 {
        m /= 10;
        e += 1;
    }
    if m == 0 {
        return Some(Dec::ZERO);
    }
    Some(Dec { m: i64::try_from(m).ok()?, e: i8::try_from(e).ok()? })
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::string::String;

    fn s(d: Dec) -> String {
        let mut o = String::new();
        d.write(&mut o).unwrap();
        o
    }

    #[test]
    fn parse_and_print() {
        assert_eq!(Dec::parse("0.25"), Some(Dec { m: 25, e: -2 }));
        assert_eq!(Dec::parse("132.40"), Some(Dec { m: 1324, e: -1 }));
        assert_eq!(Dec::parse("-45"), Some(Dec { m: -45, e: 0 }));
        assert_eq!(Dec::parse("1500"), Some(Dec { m: 15, e: 2 }));
        assert_eq!(Dec::parse("1.5e-3"), Some(Dec { m: 15, e: -4 }));
        assert_eq!(Dec::parse("0"), Some(Dec::ZERO));
        assert_eq!(Dec::parse("-0.0"), Some(Dec::ZERO));
        for bad in ["", "-", "01", "1.", ".5", "1e", "1e+", "+1", "1x", "0x10", "1234567890123456789", "1e999"] {
            assert_eq!(Dec::parse(bad), None, "{bad}");
        }
        assert_eq!(Dec::parse("0.10000000000000000000000"), Some(Dec { m: 1, e: -1 }));
        assert_eq!(s(Dec::parse("0.0054931640625").unwrap()), "0.0054931640625");
        assert_eq!(s(Dec::new(1500, 0)), "1500");
        assert_eq!(s(Dec::new(-5, -3)), "-0.005");
    }

    #[test]
    fn counts_round_as_asked() {
        let q = Dec::parse("0.25").unwrap();
        assert_eq!(to_count(Dec::parse("1.0").unwrap(), q, Dec::ZERO, Round::Nearest), Some((4, true)));
        assert_eq!(to_count(Dec::parse("1.1").unwrap(), q, Dec::ZERO, Round::Nearest), Some((4, false)));
        assert_eq!(to_count(Dec::parse("1.125").unwrap(), q, Dec::ZERO, Round::Nearest), Some((5, false)));
        assert_eq!(to_count(Dec::parse("-1.125").unwrap(), q, Dec::ZERO, Round::Nearest), Some((-5, false)));
        assert_eq!(to_count(Dec::parse("1.1").unwrap(), q, Dec::ZERO, Round::Floor), Some((4, false)));
        assert_eq!(to_count(Dec::parse("1.1").unwrap(), q, Dec::ZERO, Round::Ceil), Some((5, false)));
        assert_eq!(to_count(Dec::parse("-1.1").unwrap(), q, Dec::ZERO, Round::Floor), Some((-5, false)));
        assert_eq!(to_count(Dec::parse("-1.1").unwrap(), q, Dec::ZERO, Round::Ceil), Some((-4, false)));
        // 2.2 mV steps from 132.4 mV: 150 mV is exactly code 8.
        let (s22, o) = (Dec::parse("2.2").unwrap(), Dec::parse("132.4").unwrap());
        assert_eq!(to_count(Dec::parse("150").unwrap(), s22, o, Round::Nearest), Some((8, true)));
        // A negative scale flips floor and ceil in count space correctly.
        let neg = Dec::parse("-0.5").unwrap();
        assert_eq!(to_count(Dec::parse("1.2").unwrap(), neg, Dec::ZERO, Round::Floor), Some((-3, false)));
        assert_eq!(to_count(Dec::parse("1.2").unwrap(), neg, Dec::ZERO, Round::Ceil), Some((-2, false)));
        assert_eq!(to_count(Dec::new(9, 100), q, Dec::ZERO, Round::Nearest), None);
    }

    #[test]
    fn physical_is_exact() {
        let deg = Dec::parse("0.0054931640625").unwrap();
        let mut o = String::new();
        write_physical(&mut o, 16384, deg, Dec::ZERO).unwrap();
        assert_eq!(o, "90");
        o.clear();
        write_physical(&mut o, 10, Dec::parse("2.2").unwrap(), Dec::parse("132.4").unwrap()).unwrap();
        assert_eq!(o, "154.4");
        assert_eq!(physical(15, Dec::parse("2.2").unwrap(), Dec::parse("132.4").unwrap()), Some(Dec { m: 1654, e: -1 }));
    }
}
