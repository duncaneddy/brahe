/*!
IERS Conventions (2010) Solid Earth Tide — Step 2 frequency-dependent
geopotential coefficient correction tables (Tables 6.5a, 6.5b, 6.5c).

Source: IERS Conventions (2010), TN36, Chapter 6, Tables 6.5a (p.9),
6.5b/6.5c (p.10).
URL: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>

Each table entry provides a tidal constituent's Delaunay multipliers (in
the order `[l, l', F, D, Ω]`) and the in-phase / out-of-phase amplitude
corrections to the corresponding degree-2 geopotential Love number. The
`Amp.` columns are in units of 1e-12 (dimensionless); the implementor
multiplies by `scale = 1e-12` before folding into ΔC̄nm / ΔS̄nm.

To verify a row: open the PDF above, navigate to the indicated page, and
match the row by Doodson number. The Delaunay columns are labeled
`ℓ ℓ' F D Ω` immediately to the right of the Doodson multipliers; the
amplitude columns are labeled `Amp.` (in-phase, then out-of-phase where
applicable).

- **TABLE_M1** (Table 6.5a, m=1, 48 rows): diurnal band, k21 corrections.
- **TABLE_M0** (Table 6.5b, m=0, 21 rows): long-period/zonal band, k20 corrections.
- **TABLE_M2** (Table 6.5c, m=2, 2 rows): semidiurnal band, k22 corrections.
  Table 6.5c has NO out-of-phase column; `amp_out_of_phase` is stored as 0.0.
*/

/// One line of Tables 6.5a / 6.5b / 6.5c.
///
/// `delaunay` is `[l, l', F, D, Ω]` (five integers).
/// `amp_in_phase` and `amp_out_of_phase` are in units of 1e-12.
/// For Table 6.5c (m=2) there is no out-of-phase correction; it is 0.0.
#[derive(Debug, Clone, Copy)]
pub struct Step2Line {
    /// Delaunay multipliers `[l, l', F, D, Ω]`.
    pub delaunay: [i8; 5],
    /// In-phase amplitude correction (units 1e-12).
    pub amp_in_phase: f64,
    /// Out-of-phase amplitude correction (units 1e-12); 0.0 for Table 6.5c.
    pub amp_out_of_phase: f64,
}

/// Table 6.5a — Diurnal band (m=1), 48 rows.
/// Source: IERS Conventions (2010) Ch.6 PDF, p.9.
pub const TABLE_M1: [Step2Line; 48] = [
    // 2Q1  | 125,755 | [2, 0, 2, 0, 2]
    Step2Line {
        delaunay: [2, 0, 2, 0, 2],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // σ1   | 127,555 | [0, 0, 2, 2, 2]
    Step2Line {
        delaunay: [0, 0, 2, 2, 2],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 135,645 | [1, 0, 2, 0, 1]
    Step2Line {
        delaunay: [1, 0, 2, 0, 1],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // Q1   | 135,655 | [1, 0, 2, 0, 2]
    Step2Line {
        delaunay: [1, 0, 2, 0, 2],
        amp_in_phase: -0.7,
        amp_out_of_phase: 0.1,
    },
    // ρ1   | 137,455 | [-1, 0, 2, 2, 2]
    Step2Line {
        delaunay: [-1, 0, 2, 2, 2],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 145,545 | [0, 0, 2, 0, 1]
    Step2Line {
        delaunay: [0, 0, 2, 0, 1],
        amp_in_phase: -1.3,
        amp_out_of_phase: 0.1,
    },
    // O1   | 145,555 | [0, 0, 2, 0, 2]
    Step2Line {
        delaunay: [0, 0, 2, 0, 2],
        amp_in_phase: -6.8,
        amp_out_of_phase: 0.6,
    },
    // τ1   | 147,555 | [0, 0, 0, 2, 0]
    Step2Line {
        delaunay: [0, 0, 0, 2, 0],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // Nτ1  | 153,655 | [1, 0, 2, -2, 2]
    Step2Line {
        delaunay: [1, 0, 2, -2, 2],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 155,445 | [-1, 0, 2, 0, 1]
    Step2Line {
        delaunay: [-1, 0, 2, 0, 1],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // Lk1  | 155,455 | [-1, 0, 2, 0, 2]
    Step2Line {
        delaunay: [-1, 0, 2, 0, 2],
        amp_in_phase: 0.4,
        amp_out_of_phase: 0.0,
    },
    // No1  | 155,655 | [1, 0, 0, 0, 0]
    Step2Line {
        delaunay: [1, 0, 0, 0, 0],
        amp_in_phase: 1.3,
        amp_out_of_phase: -0.1,
    },
    // —    | 155,665 | [1, 0, 0, 0, 1]
    Step2Line {
        delaunay: [1, 0, 0, 0, 1],
        amp_in_phase: 0.3,
        amp_out_of_phase: 0.0,
    },
    // χ1   | 157,455 | [-1, 0, 0, 2, 0]
    Step2Line {
        delaunay: [-1, 0, 0, 2, 0],
        amp_in_phase: 0.3,
        amp_out_of_phase: 0.0,
    },
    // —    | 157,465 | [-1, 0, 0, 2, 1]
    Step2Line {
        delaunay: [-1, 0, 0, 2, 1],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // π1   | 162,556 | [0, 1, 2, -2, 2]
    Step2Line {
        delaunay: [0, 1, 2, -2, 2],
        amp_in_phase: -1.9,
        amp_out_of_phase: 0.1,
    },
    // —    | 163,545 | [0, 0, 2, -2, 1]
    Step2Line {
        delaunay: [0, 0, 2, -2, 1],
        amp_in_phase: 0.5,
        amp_out_of_phase: 0.0,
    },
    // P1   | 163,555 | [0, 0, 2, -2, 2]
    Step2Line {
        delaunay: [0, 0, 2, -2, 2],
        amp_in_phase: -43.4,
        amp_out_of_phase: 2.9,
    },
    // —    | 164,554 | [0, -1, 2, -2, 2]
    Step2Line {
        delaunay: [0, -1, 2, -2, 2],
        amp_in_phase: 0.6,
        amp_out_of_phase: 0.0,
    },
    // S1   | 164,556 | [0, 1, 0, 0, 0]
    Step2Line {
        delaunay: [0, 1, 0, 0, 0],
        amp_in_phase: 1.6,
        amp_out_of_phase: -0.1,
    },
    // —    | 165,345 | [-2, 0, 2, 0, 1]
    Step2Line {
        delaunay: [-2, 0, 2, 0, 1],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 165,535 | [0, 0, 0, 0, -2]
    Step2Line {
        delaunay: [0, 0, 0, 0, -2],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 165,545 | [0, 0, 0, 0, -1]
    Step2Line {
        delaunay: [0, 0, 0, 0, -1],
        amp_in_phase: -8.8,
        amp_out_of_phase: 0.5,
    },
    // K1   | 165,555 | [0, 0, 0, 0, 0]
    Step2Line {
        delaunay: [0, 0, 0, 0, 0],
        amp_in_phase: 470.9,
        amp_out_of_phase: -30.2,
    },
    // —    | 165,565 | [0, 0, 0, 0, 1]
    Step2Line {
        delaunay: [0, 0, 0, 0, 1],
        amp_in_phase: 68.1,
        amp_out_of_phase: -4.6,
    },
    // —    | 165,575 | [0, 0, 0, 0, 2]
    Step2Line {
        delaunay: [0, 0, 0, 0, 2],
        amp_in_phase: -1.6,
        amp_out_of_phase: 0.1,
    },
    // —    | 166,455 | [-1, 0, 0, 1, 0]
    Step2Line {
        delaunay: [-1, 0, 0, 1, 0],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 166,544 | [0, -1, 0, 0, -1]
    Step2Line {
        delaunay: [0, -1, 0, 0, -1],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // ψ1   | 166,554 | [0, -1, 0, 0, 0]
    Step2Line {
        delaunay: [0, -1, 0, 0, 0],
        amp_in_phase: -20.6,
        amp_out_of_phase: -0.3,
    },
    // —    | 166,556 | [0, 1, -2, 2, -2]
    Step2Line {
        delaunay: [0, 1, -2, 2, -2],
        amp_in_phase: 0.3,
        amp_out_of_phase: 0.0,
    },
    // —    | 166,564 | [0, -1, 0, 0, 1]
    Step2Line {
        delaunay: [0, -1, 0, 0, 1],
        amp_in_phase: -0.3,
        amp_out_of_phase: 0.0,
    },
    // —    | 167,355 | [-2, 0, 0, 2, 0]
    Step2Line {
        delaunay: [-2, 0, 0, 2, 0],
        amp_in_phase: -0.2,
        amp_out_of_phase: 0.0,
    },
    // —    | 167,365 | [-2, 0, 0, 2, 1]
    Step2Line {
        delaunay: [-2, 0, 0, 2, 1],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // φ1   | 167,555 | [0, 0, -2, 2, -2]
    Step2Line {
        delaunay: [0, 0, -2, 2, -2],
        amp_in_phase: -5.0,
        amp_out_of_phase: 0.3,
    },
    // —    | 167,565 | [0, 0, -2, 2, -1]
    Step2Line {
        delaunay: [0, 0, -2, 2, -1],
        amp_in_phase: 0.2,
        amp_out_of_phase: 0.0,
    },
    // —    | 168,554 | [0, -1, -2, 2, -2]
    Step2Line {
        delaunay: [0, -1, -2, 2, -2],
        amp_in_phase: -0.2,
        amp_out_of_phase: 0.0,
    },
    // θ1   | 173,655 | [1, 0, 0, -2, 0]
    Step2Line {
        delaunay: [1, 0, 0, -2, 0],
        amp_in_phase: -0.5,
        amp_out_of_phase: 0.0,
    },
    // —    | 173,665 | [1, 0, 0, -2, 1]
    Step2Line {
        delaunay: [1, 0, 0, -2, 1],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 175,445 | [-1, 0, 0, 0, -1]
    Step2Line {
        delaunay: [-1, 0, 0, 0, -1],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.0,
    },
    // J1   | 175,455 | [-1, 0, 0, 0, 0]
    Step2Line {
        delaunay: [-1, 0, 0, 0, 0],
        amp_in_phase: -2.1,
        amp_out_of_phase: 0.1,
    },
    // —    | 175,465 | [-1, 0, 0, 0, 1]
    Step2Line {
        delaunay: [-1, 0, 0, 0, 1],
        amp_in_phase: -0.4,
        amp_out_of_phase: 0.0,
    },
    // So1  | 183,555 | [0, 0, 0, -2, 0]
    Step2Line {
        delaunay: [0, 0, 0, -2, 0],
        amp_in_phase: -0.2,
        amp_out_of_phase: 0.0,
    },
    // —    | 185,355 | [-2, 0, 0, 0, 0]
    Step2Line {
        delaunay: [-2, 0, 0, 0, 0],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // Oo1  | 185,555 | [0, 0, -2, 0, -2]
    Step2Line {
        delaunay: [0, 0, -2, 0, -2],
        amp_in_phase: -0.6,
        amp_out_of_phase: 0.0,
    },
    // —    | 185,565 | [0, 0, -2, 0, -1]
    Step2Line {
        delaunay: [0, 0, -2, 0, -1],
        amp_in_phase: -0.4,
        amp_out_of_phase: 0.0,
    },
    // —    | 185,575 | [0, 0, -2, 0, 0]
    Step2Line {
        delaunay: [0, 0, -2, 0, 0],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // ν1   | 195,455 | [-1, 0, -2, 0, -2]
    Step2Line {
        delaunay: [-1, 0, -2, 0, -2],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
    // —    | 195,465 | [-1, 0, -2, 0, -1]
    Step2Line {
        delaunay: [-1, 0, -2, 0, -1],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.0,
    },
];

/// Table 6.5b — Long-period / zonal band (m=0), 21 rows.
/// Source: IERS Conventions (2010) Ch.6 PDF, p.10.
pub const TABLE_M0: [Step2Line; 21] = [
    // —    | 55,565 | [0, 0, 0, 0, 1]
    Step2Line {
        delaunay: [0, 0, 0, 0, 1],
        amp_in_phase: 16.6,
        amp_out_of_phase: -6.7,
    },
    // —    | 55,575 | [0, 0, 0, 0, 2]
    Step2Line {
        delaunay: [0, 0, 0, 0, 2],
        amp_in_phase: -0.1,
        amp_out_of_phase: 0.1,
    },
    // Sa   | 56,554 | [0, -1, 0, 0, 0]
    Step2Line {
        delaunay: [0, -1, 0, 0, 0],
        amp_in_phase: -1.2,
        amp_out_of_phase: 0.8,
    },
    // Ssa  | 57,555 | [0, 0, -2, 2, -2]
    Step2Line {
        delaunay: [0, 0, -2, 2, -2],
        amp_in_phase: -5.5,
        amp_out_of_phase: 4.3,
    },
    // —    | 57,565 | [0, 0, -2, 2, -1]
    Step2Line {
        delaunay: [0, 0, -2, 2, -1],
        amp_in_phase: 0.1,
        amp_out_of_phase: -0.1,
    },
    // —    | 58,554 | [0, -1, -2, 2, -2]
    Step2Line {
        delaunay: [0, -1, -2, 2, -2],
        amp_in_phase: -0.3,
        amp_out_of_phase: 0.2,
    },
    // Msm  | 63,655 | [1, 0, 0, -2, 0]
    Step2Line {
        delaunay: [1, 0, 0, -2, 0],
        amp_in_phase: -0.3,
        amp_out_of_phase: 0.7,
    },
    // —    | 65,445 | [-1, 0, 0, 0, -1]
    Step2Line {
        delaunay: [-1, 0, 0, 0, -1],
        amp_in_phase: 0.1,
        amp_out_of_phase: -0.2,
    },
    // Mm   | 65,455 | [-1, 0, 0, 0, 0]
    Step2Line {
        delaunay: [-1, 0, 0, 0, 0],
        amp_in_phase: -1.2,
        amp_out_of_phase: 3.7,
    },
    // —    | 65,465 | [-1, 0, 0, 0, 1]
    Step2Line {
        delaunay: [-1, 0, 0, 0, 1],
        amp_in_phase: 0.1,
        amp_out_of_phase: -0.2,
    },
    // —    | 65,655 | [1, 0, -2, 0, -2]
    Step2Line {
        delaunay: [1, 0, -2, 0, -2],
        amp_in_phase: 0.1,
        amp_out_of_phase: -0.2,
    },
    // Msf  | 73,555 | [0, 0, 0, -2, 0]
    Step2Line {
        delaunay: [0, 0, 0, -2, 0],
        amp_in_phase: 0.0,
        amp_out_of_phase: 0.6,
    },
    // Mf   | 75,355 | [-2, 0, 0, 0, 0]
    Step2Line {
        delaunay: [-2, 0, 0, 0, 0],
        amp_in_phase: 0.0,
        amp_out_of_phase: 0.3,
    },
    // Mf   | 75,555 | [0, 0, -2, 0, -2]
    Step2Line {
        delaunay: [0, 0, -2, 0, -2],
        amp_in_phase: 0.6,
        amp_out_of_phase: 6.3,
    },
    // —    | 75,565 | [0, 0, -2, 0, -1]
    Step2Line {
        delaunay: [0, 0, -2, 0, -1],
        amp_in_phase: 0.2,
        amp_out_of_phase: 2.6,
    },
    // —    | 75,575 | [0, 0, -2, 0, 0]
    Step2Line {
        delaunay: [0, 0, -2, 0, 0],
        amp_in_phase: 0.2,
        amp_out_of_phase: 0.2,
    },
    // Mstm | 83,655 | [1, 0, -2, -2, -2]
    Step2Line {
        delaunay: [1, 0, -2, -2, -2],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.2,
    },
    // Mtm  | 85,455 | [-1, 0, -2, 0, -2]
    Step2Line {
        delaunay: [-1, 0, -2, 0, -2],
        amp_in_phase: 0.4,
        amp_out_of_phase: 1.1,
    },
    // —    | 85,465 | [-1, 0, -2, 0, -1]
    Step2Line {
        delaunay: [-1, 0, -2, 0, -1],
        amp_in_phase: 0.2,
        amp_out_of_phase: 0.5,
    },
    // Msqm | 93,555 | [0, 0, -2, -2, -2]
    Step2Line {
        delaunay: [0, 0, -2, -2, -2],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.2,
    },
    // Mqm  | 95,355 | [-2, 0, -2, 0, -2]
    Step2Line {
        delaunay: [-2, 0, -2, 0, -2],
        amp_in_phase: 0.1,
        amp_out_of_phase: 0.1,
    },
];

/// Table 6.5c — Semidiurnal band (m=2), 2 rows.
/// Source: IERS Conventions (2010) Ch.6 PDF, p.10.
/// There is NO out-of-phase column in Table 6.5c; `amp_out_of_phase` is 0.0.
pub const TABLE_M2: [Step2Line; 2] = [
    // N2 | 245,655 | [1, 0, 2, 0, 2]
    Step2Line {
        delaunay: [1, 0, 2, 0, 2],
        amp_in_phase: -0.3,
        amp_out_of_phase: 0.0,
    },
    // M2 | 255,555 | [0, 0, 2, 0, 2]
    Step2Line {
        delaunay: [0, 0, 2, 0, 2],
        amp_in_phase: -1.2,
        amp_out_of_phase: 0.0,
    },
];
