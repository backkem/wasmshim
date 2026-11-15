# The A64 CPU: coverage and floating point

`wasmshim-aarch64` implements the user-mode (EL0) view of an AArch64 CPU: X0–X30, SP, PC, NZCV,
V0–V31, FPCR, FPSR, `TPIDR_EL0`, `TPIDRRO_EL0` and the virtual counter. There are no exception
levels and no MMU (the `GuestMemory` is the address space), memory is single-copy atomic (one
host thread), and the exclusive monitor is cleared whenever `run_until_event` returns. This
note is derived from [`decode.rs`](../../crates/cpu-aarch64/src/decode.rs),
[`simd.rs`](../../crates/cpu-aarch64/src/simd.rs),
[`exec.rs`](../../crates/cpu-aarch64/src/exec.rs),
[`simd_exec.rs`](../../crates/cpu-aarch64/src/simd_exec.rs) and
[`fp.rs`](../../crates/cpu-aarch64/src/fp.rs). Translated code follows the interpreter exactly
([engines-and-translation.md](engines-and-translation.md)).

**Guarantee.** An instruction that is not implemented raises `Fault::Undefined`, which the
Linux personality delivers as `SIGILL` (`ILL_ILLOPC`). It is never executed approximately.

## What the CPU reports

| Register or value | Reads as |
|---|---|
| `MIDR_EL1` | `0x410f_d083` (Arm Cortex-A72 r0p3); `/proc/cpuinfo` and `/sys/.../midr_el1` agree |
| `CTR_EL0` | `0x8444_c004` (64-byte cache lines) |
| `DCZID_EL0` | `4`: `DC ZVA` zeroes 64 bytes |
| `CNTFRQ_EL0` | 1 GHz; `CNTVCT_EL0` is virtual time in ns, fixed for the length of one run |
| `TPIDRRO_EL0` | always 0 |
| `AT_HWCAP` | `FP | ASIMD` only; `AT_HWCAP2` is 0 |

`MRS` reads `NZCV`, `FPCR`, `FPSR`, `TPIDR_EL0`, `TPIDRRO_EL0`, `CNTVCT_EL0`, `CNTFRQ_EL0`,
`CTR_EL0`, `DCZID_EL0` and `MIDR_EL1`. `MSR` writes `NZCV`, `FPCR`, `FPSR` and `TPIDR_EL0`.
Every other system register is `Undefined`, including the `ID_AA64*` registers (`HWCAP_CPUID`
is not advertised). LSE atomics are implemented but not advertised in `AT_HWCAP`, so libraries
that check it use `LDXR`/`STXR`.

## Integer instructions

Implemented:

- **Immediate:** `ADR`, `ADRP`, `ADD`/`SUB`(`S`), logical immediates, `MOVN`/`MOVZ`/`MOVK`,
  `SBFM`/`BFM`/`UBFM`, `EXTR`.
- **Register:** logical (shifted, with inversion), `ADD`/`SUB` shifted and extended,
  `ADC`/`SBC`(`S`), `CCMP`/`CCMN` (register and immediate), `CSEL`/`CSINC`/`CSINV`/`CSNEG`,
  `RBIT`, `REV16`, `REV32`, `REV`, `CLZ`, `CLS`, `UDIV`, `SDIV`, `LSLV`, `LSRV`, `ASRV`, `RORV`,
  `MADD`, `MSUB`, `SMADDL`, `SMSUBL`, `UMADDL`, `UMSUBL`, `SMULH`, `UMULH`.
- **Branches:** `B`, `BL`, `B.cond` (and `BC.cond`), `CBZ`/`CBNZ`, `TBZ`/`TBNZ`, and the plain
  `BR`, `BLR`, `RET`.
- **Loads and stores:** all sizes, signed and unsigned; unsigned offset, unscaled, unprivileged
  (as unscaled), pre- and post-index, register offset with extend; literal loads; `LDP`/`STP`,
  `LDNP`/`STNP`, `LDPSW`; `PRFM` (no effect); SIMD&FP registers `B`/`H`/`S`/`D`/`Q`.
- **Exclusives and atomics:** `LDXR`/`STXR`, `LDAXR`/`STLXR`, pair forms, `LDAR`/`STLR`
  (`LDLAR`/`STLLR`), `LDAPR`, `CAS*`, `CASP*`, `LDADD`, `LDCLR`, `LDEOR`, `LDSET`,
  `LD{S,U}{MAX,MIN}`, `SWP`, all orderings (ordering has no effect on one core).
  These fault with `Fault::Alignment` on an address that is not naturally aligned. A load or
  store (scalar, pair or SIMD structure) whose base is SP faults if SP is not 16-byte aligned.
- **System:** `SVC #imm` (an `Event::Syscall`), `BRK` (`Fault::Breakpoint`, `SIGTRAP`), `NOP`,
  `YIELD`, `WFE`, `WFI`, `SEV` and the hint-space `PAC*`/`AUT*`/`BTI` instructions as `NOP`,
  `DSB`/`DMB`/`ISB`/`SB` as `NOP`, `CLREX`, `DC ZVA`, and `DC CVAC`/`CVAU`/`CVAP`/`CVADP`/
  `CIVAC` and `IC IVAU` as `NOP` (memory is coherent).

`Undefined`: `CRC32*`/`CRC32C*`, `PACGA`, the non-hint pointer-authentication instructions
(`BRAA`, `RETAA`, `LDRAA`, `PACIA` with a register, ...), CSSC (`ABS`, `CNT`, `CTZ`,
`SMAX` ...), `RMIF`/`SETF8`/`SETF16`, MTE (`ADDG`, `SUBG`, `STGP`, ...), `MSR` (immediate)
including `SMSTART`, `LDAPUR`/`STLUR`, LSE atomics on SIMD&FP registers, memory copy and set
(`CPY*`, `SET*`), SVE and SME, `UDF`.

## SIMD and floating point

Implemented: scalar FP in single and double precision (`FMOV`, `FABS`, `FNEG`, `FSQRT`,
`FRINT{N,P,M,Z,A,X,I}`, `FCVT` between S and D, `FADD`, `FSUB`, `FMUL`, `FNMUL`, `FDIV`,
`FMAX`/`FMIN`(`NM`), `FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`, `FCMP`(`E`), `FCCMP`(`E`), `FCSEL`,
`FMOV` immediate, conversions to and from integers and fixed point); and Advanced SIMD three
same, three different, two-register miscellaneous, across lanes, scalar pairwise, copy
(`DUP`, `INS`, `SMOV`, `UMOV`), modified immediate, shift by immediate, `UZP`/`TRN`/`ZIP`,
`EXT`, `TBL`/`TBX`, by-element forms, and `LD1`–`LD4`/`ST1`–`ST4` (multiple, single lane,
replicate). Within those groups, selector combinations the interpreter does not handle are
`Undefined` when executed.

`Undefined`:

- **Half precision** everywhere: scalar FP with `ftype` = half, the FP16 vector forms, and
  `FCVT`, `FCVTL`, `FCVTN` to or from half. No BFloat16.
- **Crypto:** `AES*`, `SHA1*`, `SHA256*`, `SHA512*`, SHA-3, SM3, SM4, and `PMULL` on 64-bit
  lanes (`PMULL` on bytes is implemented).
- **Estimates:** `FRECPE`, `FRSQRTE`, `FRECPS`, `FRSQRTS`, `FRECPX`, `URECPE`, `URSQRTE`.
- Dot products (`SDOT`, `UDOT`, `USDOT`), `SQRDMLAH`/`SQRDMLSH`, `FCMLA`/`FCADD`, I8MM,
  `SQDMULL`/`SQDMLAL`/`SQDMLSL`, `SUQADD`/`USQADD`, `FCVTXN`, `FRINT32*`/`FRINT64*`,
  `FJCVTZS`.

## The floating-point model

- **Rounding** is always to nearest, ties to even, whatever `FPCR.RMode` says. `FRINTX` and
  `FRINTI` round to nearest too. Instructions with their own rounding (`FRINTP`, `FCVTZS`,
  `FCVTAS`, ...) use it.
- **No flushing:** `FPCR.FZ` is ignored; denormals are computed exactly.
- **`FPCR`** keeps the bits user code may write (mask `0x07ff_9f00`) and reads them back; of
  them only `DN` (default NaN, bit 25) has an effect.
- **`FPSR`:** only `QC` (bit 27, cumulative saturation) is ever set. The exception flags
  (`IOC`, `DZC`, `OFC`, `UFC`, `IXC`, `IDC`) are never set.
- **NaNs** follow the Arm ARM explicitly, because host and Wasm NaN bit patterns are not
  deterministic: a signalling NaN operand wins over a quiet one, operands are taken in order,
  the chosen NaN is quieted, `DN` substitutes the default NaN, and an invalid operation on
  non-NaN operands gives the default NaN. Non-NaN results are IEEE 754 results computed on
  the host.
- **Fused multiply-add** rounds once (`libm::fma`/`fmaf`).
- **Conversions to integer** saturate, and a NaN converts to 0. Fixed-point `SCVTF`/`UCVTF`
  from integers above 2^53 in magnitude round twice.

## Known deviations

- A writable `FPCR.RMode` or `FZ` is accepted silently instead of taking effect.
