//! IEEE 754 helpers with A64 NaN semantics, on raw bit patterns.
//!
//! Host floating point is used only for non-NaN operands, where IEEE 754 fixes the result.
//! NaN inputs and NaN results follow the Arm ARM (`FPProcessNaNs`, default NaN) explicitly,
//! because NaN bit patterns from Wasm (and some hosts) are not deterministic. Rounding is
//! always round-to-nearest-even (FPCR.RMode = 0), denormals are not flushed (FPCR.FZ = 0), and
//! the cumulative exception flags in FPSR are not tracked.

/// Floating-point formats used here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fmt {
    S,
    D,
}

impl Fmt {
    pub(crate) const fn bits(self) -> u32 {
        match self {
            Self::S => 32,
            Self::D => 64,
        }
    }
    const fn frac_bits(self) -> u32 {
        match self {
            Self::S => 23,
            Self::D => 52,
        }
    }
    const fn exp_mask(self) -> u64 {
        match self {
            Self::S => 0x7f80_0000,
            Self::D => 0x7ff0_0000_0000_0000,
        }
    }
    const fn frac_mask(self) -> u64 {
        (1 << self.frac_bits()) - 1
    }
    const fn quiet_bit(self) -> u64 {
        1 << (self.frac_bits() - 1)
    }
    /// The default NaN.
    pub(crate) const fn default_nan(self) -> u64 {
        self.exp_mask() | self.quiet_bit()
    }
    pub(crate) const fn sign_bit(self) -> u64 {
        1 << (self.bits() - 1)
    }
    pub(crate) const fn is_nan(self, x: u64) -> bool {
        x & self.exp_mask() == self.exp_mask() && x & self.frac_mask() != 0
    }
    const fn is_snan(self, x: u64) -> bool {
        self.is_nan(x) && x & self.quiet_bit() == 0
    }
    const fn is_inf(self, x: u64) -> bool {
        x & !self.sign_bit() == self.exp_mask()
    }
    const fn is_zero(self, x: u64) -> bool {
        x & !self.sign_bit() == 0
    }
    pub(crate) fn to_f64(self, x: u64) -> f64 {
        match self {
            Self::S => f64::from(f32::from_bits(x as u32)),
            Self::D => f64::from_bits(x),
        }
    }
    fn encode_f64(self, v: f64) -> u64 {
        match self {
            Self::S => u64::from((v as f32).to_bits()),
            Self::D => v.to_bits(),
        }
    }
}

/// `FPProcessNaNs` for two operands: `Some(result)` if either is a NaN.
fn nans2(f: Fmt, a: u64, b: u64, dn: bool) -> Option<u64> {
    let pick = |x: u64| {
        Some(if dn {
            f.default_nan()
        } else {
            x | f.quiet_bit()
        })
    };
    if f.is_snan(a) {
        pick(a)
    } else if f.is_snan(b) {
        pick(b)
    } else if f.is_nan(a) {
        pick(a)
    } else if f.is_nan(b) {
        pick(b)
    } else {
        None
    }
}

/// `FPProcessNaNs3` (order: `a`, `b`, `c`).
fn nans3(f: Fmt, a: u64, b: u64, c: u64, dn: bool) -> Option<u64> {
    let pick = |x: u64| {
        Some(if dn {
            f.default_nan()
        } else {
            x | f.quiet_bit()
        })
    };
    for x in [a, b, c] {
        if f.is_snan(x) {
            return pick(x);
        }
    }
    for x in [a, b, c] {
        if f.is_nan(x) {
            return pick(x);
        }
    }
    None
}

/// A NaN result of an operation on non-NaN inputs (an invalid operation) is the default NaN.
fn finish(f: Fmt, r: u64) -> u64 {
    if f.is_nan(r) { f.default_nan() } else { r }
}

/// Binary arithmetic operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op2 {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNum,
    MinNum,
    /// `FMULX`: like `Mul`, but `0 * inf = ±2.0`.
    MulX,
    /// `FABD`.
    Abd,
}

pub(crate) fn op2(f: Fmt, op: Op2, a: u64, b: u64, dn: bool) -> u64 {
    // MaxNum/MinNum: a single quiet NaN loses against a number.
    if matches!(op, Op2::MaxNum | Op2::MinNum) {
        let (qa, qb) = (f.is_nan(a) && !f.is_snan(a), f.is_nan(b) && !f.is_snan(b));
        if qa && !f.is_nan(b) {
            return op2(
                f,
                if op == Op2::MaxNum {
                    Op2::Max
                } else {
                    Op2::Min
                },
                b,
                b,
                dn,
            );
        }
        if qb && !f.is_nan(a) {
            return op2(
                f,
                if op == Op2::MaxNum {
                    Op2::Max
                } else {
                    Op2::Min
                },
                a,
                a,
                dn,
            );
        }
    }
    if let Some(n) = nans2(f, a, b, dn) {
        return n;
    }
    if op == Op2::MulX && ((f.is_inf(a) && f.is_zero(b)) || (f.is_zero(a) && f.is_inf(b))) {
        let two = f.encode_f64(2.0);
        return two | ((a ^ b) & f.sign_bit());
    }
    let (x, y) = (f.to_f64(a), f.to_f64(b));
    let r = match f {
        Fmt::S => {
            let (x, y) = (x as f32, y as f32);
            let v = match op {
                Op2::Add => x + y,
                Op2::Sub => x - y,
                Op2::Mul | Op2::MulX => x * y,
                Op2::Div => x / y,
                Op2::Abd => (x - y).abs(),
                Op2::Max | Op2::MaxNum => return max_min(f, a, b, true),
                Op2::Min | Op2::MinNum => return max_min(f, a, b, false),
            };
            u64::from(v.to_bits())
        }
        Fmt::D => {
            let v = match op {
                Op2::Add => x + y,
                Op2::Sub => x - y,
                Op2::Mul | Op2::MulX => x * y,
                Op2::Div => x / y,
                Op2::Abd => (x - y).abs(),
                Op2::Max | Op2::MaxNum => return max_min(f, a, b, true),
                Op2::Min | Op2::MinNum => return max_min(f, a, b, false),
            };
            v.to_bits()
        }
    };
    finish(f, r)
}

/// `FMAX`/`FMIN` on non-NaN inputs: `-0 < +0`.
fn max_min(f: Fmt, a: u64, b: u64, max: bool) -> u64 {
    if f.is_zero(a) && f.is_zero(b) {
        let neg = if max { a & b } else { a | b } & f.sign_bit();
        return neg;
    }
    let (x, y) = (f.to_f64(a), f.to_f64(b));
    if (x > y) == max { a } else { b }
}

/// `FMADD`-style fused multiply-add: `addend + x * y` with one rounding.
pub(crate) fn fma(f: Fmt, addend: u64, x: u64, y: u64, dn: bool) -> u64 {
    let inf_times_zero = (f.is_inf(x) && f.is_zero(y)) || (f.is_zero(x) && f.is_inf(y));
    if f.is_nan(addend) && !f.is_snan(addend) && inf_times_zero {
        return f.default_nan();
    }
    if let Some(n) = nans3(f, addend, x, y, dn) {
        return n;
    }
    let r = match f {
        Fmt::S => u64::from(
            libm::fmaf(
                f32::from_bits(x as u32),
                f32::from_bits(y as u32),
                f32::from_bits(addend as u32),
            )
            .to_bits(),
        ),
        Fmt::D => libm::fma(f64::from_bits(x), f64::from_bits(y), f64::from_bits(addend)).to_bits(),
    };
    finish(f, r)
}

/// `FSQRT`.
pub(crate) fn sqrt(f: Fmt, a: u64, dn: bool) -> u64 {
    if let Some(n) = nans2(f, a, a, dn) {
        return n;
    }
    let r = match f {
        Fmt::S => u64::from(libm::sqrtf(f32::from_bits(a as u32)).to_bits()),
        Fmt::D => libm::sqrt(f64::from_bits(a)).to_bits(),
    };
    finish(f, r)
}

/// Rounding modes of `FRINT*` and `FCVT*` (int).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Round {
    /// To nearest, ties to even (`N`, and the FPCR default for `X`/`I`).
    Nearest,
    /// Ties away from zero (`A`).
    Away,
    /// Towards +inf (`P`).
    Up,
    /// Towards -inf (`M`).
    Down,
    /// Towards zero (`Z`).
    Zero,
}

fn round_f64(v: f64, mode: Round) -> f64 {
    match mode {
        Round::Nearest => libm::rint(v),
        Round::Away => libm::round(v),
        Round::Up => libm::ceil(v),
        Round::Down => libm::floor(v),
        Round::Zero => libm::trunc(v),
    }
}

fn round_f32(v: f32, mode: Round) -> f32 {
    match mode {
        Round::Nearest => libm::rintf(v),
        Round::Away => libm::roundf(v),
        Round::Up => libm::ceilf(v),
        Round::Down => libm::floorf(v),
        Round::Zero => libm::truncf(v),
    }
}

/// `FRINT*`.
pub(crate) fn round_int(f: Fmt, a: u64, mode: Round, dn: bool) -> u64 {
    if let Some(n) = nans2(f, a, a, dn) {
        return n;
    }
    match f {
        Fmt::S => u64::from(round_f32(f32::from_bits(a as u32), mode).to_bits()),
        Fmt::D => round_f64(f64::from_bits(a), mode).to_bits(),
    }
}

/// `FCVT*` to an integer of `width` bits: saturating, NaN to 0.
pub(crate) fn to_int(f: Fmt, a: u64, mode: Round, signed: bool, width: u32) -> u64 {
    if f.is_nan(a) {
        return 0;
    }
    let v = round_f64(f.to_f64(a), mode);
    match (signed, width) {
        (true, 32) => u64::from(v as i32 as u32),
        (true, _) => v as i64 as u64,
        (false, 32) => u64::from(v as u32),
        (false, _) => v as u64,
    }
}

/// `SCVTF`/`UCVTF` from an integer of `width` bits (round to nearest even).
pub(crate) fn from_int(f: Fmt, x: u64, signed: bool, width: u32) -> u64 {
    match (f, signed, width) {
        (Fmt::S, true, 32) => u64::from((x as i32 as f32).to_bits()),
        (Fmt::S, true, _) => u64::from((x as i64 as f32).to_bits()),
        (Fmt::S, false, 32) => u64::from((x as u32 as f32).to_bits()),
        (Fmt::S, false, _) => u64::from((x as f32).to_bits()),
        (Fmt::D, true, 32) => (f64::from(x as i32)).to_bits(),
        (Fmt::D, true, _) => (x as i64 as f64).to_bits(),
        (Fmt::D, false, 32) => (f64::from(x as u32)).to_bits(),
        (Fmt::D, false, _) => (x as f64).to_bits(),
    }
}

/// `FCVT` between single and double, with A64 NaN conversion.
pub(crate) fn convert(from: Fmt, to: Fmt, a: u64, dn: bool) -> u64 {
    if from == to {
        return a;
    }
    if from.is_nan(a) {
        if dn {
            return to.default_nan();
        }
        let sign = u64::from(a & from.sign_bit() != 0) << (to.bits() - 1);
        let payload = (a & from.frac_mask()) | from.quiet_bit();
        let frac = if to.frac_bits() > from.frac_bits() {
            payload << (to.frac_bits() - from.frac_bits())
        } else {
            payload >> (from.frac_bits() - to.frac_bits())
        };
        return sign | to.exp_mask() | frac | to.quiet_bit();
    }
    to.encode_f64(from.to_f64(a))
}

/// `FCMP` result as NZCV (bits 3:0).
pub(crate) fn compare(f: Fmt, a: u64, b: u64) -> u8 {
    if f.is_nan(a) || f.is_nan(b) {
        return 0b0011;
    }
    let (x, y) = (f.to_f64(a), f.to_f64(b));
    if x == y {
        0b0110
    } else if x < y {
        0b1000
    } else {
        0b0010
    }
}

/// Vector compare predicates (`FCMEQ`, `FCMGE`, `FCMGT`, and the absolute `FACGE`/`FACGT`).
pub(crate) fn cmp_pred(f: Fmt, a: u64, b: u64, op: CmpOp) -> bool {
    if f.is_nan(a) || f.is_nan(b) {
        return false;
    }
    let (mut x, mut y) = (f.to_f64(a), f.to_f64(b));
    if matches!(op, CmpOp::AbsGe | CmpOp::AbsGt) {
        x = x.abs();
        y = y.abs();
    }
    match op {
        CmpOp::Eq => x == y,
        CmpOp::Ge | CmpOp::AbsGe => x >= y,
        CmpOp::Gt | CmpOp::AbsGt => x > y,
        CmpOp::Le => x <= y,
        CmpOp::Lt => x < y,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    AbsGe,
    AbsGt,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_rules() {
        let snan = 0x7ff0_0000_0000_0001u64;
        let qnan = 0x7ff8_0000_0000_0002u64;
        let one = 1.0f64.to_bits();
        assert_eq!(op2(Fmt::D, Op2::Add, one, snan, false), snan | (1 << 51));
        assert_eq!(op2(Fmt::D, Op2::Add, qnan, snan, false), snan | (1 << 51));
        assert_eq!(op2(Fmt::D, Op2::Add, qnan, one, true), Fmt::D.default_nan());
        let inf = f64::INFINITY.to_bits();
        assert_eq!(op2(Fmt::D, Op2::Sub, inf, inf, false), Fmt::D.default_nan());
        assert_eq!(op2(Fmt::D, Op2::MaxNum, qnan, one, false), one);
        assert_eq!(op2(Fmt::D, Op2::Max, (-0.0f64).to_bits(), 0, false), 0);
        assert_eq!(
            op2(Fmt::D, Op2::Min, (-0.0f64).to_bits(), 0, false),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            convert(Fmt::S, Fmt::D, 0x7f80_0001, false),
            0x7ff8_0000_2000_0000
        );
        assert_eq!(
            to_int(Fmt::D, (-1.5f64).to_bits(), Round::Nearest, true, 64),
            (-2i64) as u64
        );
        assert_eq!(
            to_int(Fmt::D, 1e30f64.to_bits(), Round::Zero, true, 32),
            i32::MAX as u32 as u64
        );
        assert_eq!(
            to_int(Fmt::D, (-3.0f64).to_bits(), Round::Zero, false, 64),
            0
        );
        assert_eq!(
            compare(
                Fmt::S,
                u64::from(1.0f32.to_bits()),
                u64::from(2.0f32.to_bits())
            ),
            0b1000
        );
    }
}
