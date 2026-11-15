//! A tiny A64 assembler for code patches: the handful of
//! instructions a patch needs to make a function return early, skip a branch or replace a
//! call, so a controller can write `mov x0, #0; ret` instead of instruction words.
//!
//! Not a general assembler. Statements are separated by `;` or newlines:
//!
//! | statement | encodes |
//! |---|---|
//! | `nop` | `NOP` |
//! | `ret`, `ret xN`, `br xN`, `blr xN` | branches to a register (`ret` uses `x30`; `lr` is `x30`) |
//! | `b T`, `bl T` | a direct branch; `T` is an address (`0x4005d0`) or relative to this instruction (`.+8`, `.-0x10`) |
//! | `mov Rd, #imm` | one `MOVZ` or `MOVN` (an error if `imm` needs more than one instruction) |
//! | `movz`/`movn`/`movk Rd, #imm16[, lsl #s]` | the move-wide instructions themselves |
//! | `brk #imm`, `svc #imm` | exception generation |
//! | `.word W` | a raw instruction word |
//!
//! `Rd` is `x0`–`x30`, `w0`–`w30`, `xzr` or `wzr`; numbers are decimal or `0x` hex, `#` optional.
//! The encodings are checked against [`crate::decode`] by the tests.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Why a statement did not assemble.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsmError {
    /// The statement (trimmed).
    pub statement: String,
    pub message: String,
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`: {}", self.statement, self.message)
    }
}

impl core::error::Error for AsmError {}

/// Assembles `text`, whose first instruction will sit at address `at` (direct branches are
/// relative to their own address).
pub fn assemble(text: &str, at: u64) -> Result<Vec<u32>, AsmError> {
    let mut out = Vec::new();
    for stmt in text
        .split([';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let pc = at.wrapping_add(4 * out.len() as u64);
        let word = statement(stmt, pc).map_err(|message| AsmError {
            statement: stmt.into(),
            message,
        })?;
        out.push(word);
    }
    Ok(out)
}

/// [`assemble`] as little-endian bytes (what a memory write takes).
pub fn assemble_bytes(text: &str, at: u64) -> Result<Vec<u8>, AsmError> {
    Ok(assemble(text, at)?
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect())
}

const NOP: u32 = 0xd503_201f;

fn statement(stmt: &str, pc: u64) -> Result<u32, String> {
    let lower = stmt.to_ascii_lowercase();
    let (op, rest) = lower
        .split_once(char::is_whitespace)
        .map_or((lower.as_str(), ""), |(o, r)| (o, r.trim()));
    let args: Vec<&str> = if rest.is_empty() {
        Vec::new()
    } else {
        rest.split(',').map(str::trim).collect()
    };
    let arity = |n: usize| {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!("{op} takes {n} operand(s)"))
        }
    };
    match op {
        "nop" => arity(0).map(|()| NOP),
        "ret" if args.is_empty() => Ok(0xd65f_0000 | 30 << 5),
        "ret" | "br" | "blr" => {
            arity(1)?;
            let (sf, n) = reg(args[0])?;
            if !sf || n == 31 {
                return Err("needs an x register".into());
            }
            let base = match op {
                "ret" => 0xd65f_0000,
                "br" => 0xd61f_0000,
                _ => 0xd63f_0000,
            };
            Ok(base | u32::from(n) << 5)
        }
        "b" | "bl" => {
            arity(1)?;
            let target = target(args[0], pc)?;
            let delta = target.wrapping_sub(pc) as i64;
            if delta % 4 != 0 || !(-(1 << 27)..(1 << 27)).contains(&delta) {
                return Err(format!(
                    "{target:#x} is not a 4-byte aligned target within 128 MiB"
                ));
            }
            let base = if op == "b" { 0x1400_0000 } else { 0x9400_0000 };
            Ok(base | ((delta >> 2) as u32 & 0x03ff_ffff))
        }
        "mov" => {
            arity(2)?;
            let (sf, rd) = reg(args[0])?;
            let v = signed(args[1])?;
            let bits = if sf { 64 } else { 32 };
            if !sf && !(i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(&v) {
                return Err("does not fit a w register".into());
            }
            let mask = if sf { u64::MAX } else { 0xffff_ffff };
            let v = v as u64 & mask;
            for (op, value) in [(0b10, v), (0b00, !v & mask)] {
                for hw in 0..bits / 16 {
                    let shift = 16 * hw;
                    if value & !(0xffff << shift) == 0 {
                        return Ok(move_wide(sf, op, hw, (value >> shift) as u16, rd));
                    }
                }
            }
            Err("needs more than one instruction (use movz and movk)".into())
        }
        "movz" | "movn" | "movk" => {
            if !(2..=3).contains(&args.len()) {
                return Err(format!("{op} takes 2 or 3 operands"));
            }
            let (sf, rd) = reg(args[0])?;
            let imm = unsigned(args[1])?;
            let imm = u16::try_from(imm).map_err(|_| "the immediate is 16 bits")?;
            let shift = match args.get(2) {
                None => 0,
                Some(s) => unsigned(s.strip_prefix("lsl").ok_or("the shift is `lsl #s`")?.trim())?,
            };
            if shift % 16 != 0 || shift >= if sf { 64 } else { 32 } {
                return Err("the shift is 0, 16 (w) or 0, 16, 32, 48 (x)".into());
            }
            let opc = match op {
                "movn" => 0b00,
                "movz" => 0b10,
                _ => 0b11,
            };
            Ok(move_wide(sf, opc, (shift / 16) as u32, imm, rd))
        }
        "brk" | "svc" => {
            arity(1)?;
            let imm = u16::try_from(unsigned(args[0])?).map_err(|_| "the immediate is 16 bits")?;
            let base = if op == "brk" {
                0xd420_0000
            } else {
                0xd400_0001
            };
            Ok(base | u32::from(imm) << 5)
        }
        ".word" => {
            arity(1)?;
            u32::try_from(unsigned(args[0])?).map_err(|_| "a word is 32 bits".into())
        }
        _ => Err(
            "not one of nop, ret, br, blr, b, bl, mov, movz, movn, movk, brk, svc, .word".into(),
        ),
    }
}

fn move_wide(sf: bool, opc: u32, hw: u32, imm16: u16, rd: u8) -> u32 {
    u32::from(sf) << 31
        | opc << 29
        | 0b100101 << 23
        | hw << 21
        | u32::from(imm16) << 5
        | u32::from(rd)
}

/// `(is x, number)`; the zero register is 31.
fn reg(s: &str) -> Result<(bool, u8), String> {
    match s {
        "xzr" => return Ok((true, 31)),
        "wzr" => return Ok((false, 31)),
        "lr" => return Ok((true, 30)),
        _ => {}
    }
    let (sf, n) = match s.split_at_checked(1) {
        Some(("x", n)) => (true, n),
        Some(("w", n)) => (false, n),
        _ => return Err(format!("{s} is not a register")),
    };
    match n.parse::<u8>() {
        Ok(n) if n <= 30 => Ok((sf, n)),
        _ => Err(format!("{s} is not a register")),
    }
}

fn unsigned(s: &str) -> Result<u64, String> {
    let s = s.strip_prefix('#').unwrap_or(s).trim();
    let r = match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    };
    r.map_err(|_| format!("{s} is not a number"))
}

fn signed(s: &str) -> Result<i64, String> {
    let t = s.strip_prefix('#').unwrap_or(s).trim();
    match t.strip_prefix('-') {
        Some(m) => {
            let v = unsigned(m)?;
            if v > 1 << 63 {
                return Err(format!("{s} is out of range"));
            }
            Ok((v as i64).wrapping_neg())
        }
        None => unsigned(t).map(|v| v as i64),
    }
}

/// An address, or `.+N` / `.-N` relative to `pc`.
fn target(s: &str, pc: u64) -> Result<u64, String> {
    if let Some(off) = s.strip_prefix(".+") {
        Ok(pc.wrapping_add(unsigned(off)?))
    } else if let Some(off) = s.strip_prefix(".-") {
        Ok(pc.wrapping_sub(unsigned(off)?))
    } else {
        unsigned(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BranchRegOp, Inst, MovOp, decode};

    fn one(s: &str, at: u64) -> u32 {
        let w = assemble(s, at).unwrap();
        assert_eq!(w.len(), 1, "{s}");
        w[0]
    }

    #[test]
    fn known_encodings() {
        // Words from LLVM's assembler.
        assert_eq!(one("nop", 0), 0xd503201f);
        assert_eq!(one("ret", 0), 0xd65f03c0);
        assert_eq!(one("ret x1", 0), 0xd65f0020);
        assert_eq!(one("br x16", 0), 0xd61f0200);
        assert_eq!(one("blr x8", 0), 0xd63f0100);
        assert_eq!(one("mov x0, #0", 0), 0xd2800000);
        assert_eq!(one("mov w0, #1", 0), 0x52800020);
        assert_eq!(one("mov x0, #-1", 0), 0x92800000);
        assert_eq!(one("mov x0, #-104", 0), 0x92800ce0);
        assert_eq!(one("mov w0, #-1", 0), 0x12800000);
        assert_eq!(one("mov x1, #0x10000", 0), 0xd2a00021);
        assert_eq!(one("movk x2, #0xbeef, lsl #16", 0), 0xf2b7dde2);
        assert_eq!(one("brk #0", 0), 0xd4200000);
        assert_eq!(one("svc #0", 0), 0xd4000001);
        assert_eq!(one("b .+8", 0x1000), 0x14000002);
        assert_eq!(one("b .-4", 0x1000), 0x17ffffff);
        assert_eq!(one("bl 0x2000", 0x1000), 0x94000400);
        assert_eq!(one(".word 0xd503201f", 0), NOP);
    }

    #[test]
    fn several_statements_and_relative_branches() {
        let w = assemble("mov x0, #0\n ret; b .-8", 0x4000).unwrap();
        assert_eq!(w, [0xd2800000, 0xd65f03c0, 0x17fffffe]);
        let b = assemble_bytes("nop; ret", 0).unwrap();
        assert_eq!(b, [0x1f, 0x20, 0x03, 0xd5, 0xc0, 0x03, 0x5f, 0xd6]);
    }

    #[test]
    fn round_trips_through_the_decoder() {
        assert_eq!(decode(one("nop", 0)), Inst::Nop);
        assert_eq!(
            decode(one("ret", 0)),
            Inst::BranchReg {
                op: BranchRegOp::Ret,
                rn: 30
            }
        );
        assert_eq!(
            decode(one("bl .-0x100", 0x10000)),
            Inst::Branch {
                link: true,
                offset: -0x100
            }
        );
        assert_eq!(
            decode(one("mov x3, #-2", 0)),
            Inst::MovWide {
                sf: true,
                op: MovOp::N,
                rd: 3,
                imm16: 1,
                shift: 0
            }
        );
        assert_eq!(
            decode(one("mov w5, #0xffff0000", 0)),
            Inst::MovWide {
                sf: false,
                op: MovOp::Z,
                rd: 5,
                imm16: 0xffff,
                shift: 16
            }
        );
        assert_eq!(decode(one("svc #0", 0)), Inst::Svc { imm: 0 });
        assert_eq!(decode(one("brk #7", 0)), Inst::Brk { imm: 7 });
    }

    #[test]
    fn errors_name_the_statement() {
        let e = assemble("nop; mov x0, #0x12345", 0).unwrap_err();
        assert_eq!(e.statement, "mov x0, #0x12345");
        assert!(e.message.contains("more than one"), "{e}");
        assert!(assemble("b .+2", 0).is_err());
        assert!(assemble("ret w1", 0).is_err());
        assert!(assemble("mov w0, #0x100000000", 0).is_err());
        assert!(assemble("frobnicate", 0).is_err());
    }
}
