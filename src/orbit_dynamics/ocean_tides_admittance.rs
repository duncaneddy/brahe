/*!
IERS TN36 Table 6.7: astronomical amplitudes Hf of the FES2004 main waves and
the secondary (admittance) waves with their linear-interpolation pivot waves.

Source: IERS Conventions (2010), TN36, Chapter 6, pp. 92-93.
URL: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>

Transcription notes (verified against the PDF at 600 DPI):
- Row `117.655`'s pivot 1 is printed in the source as `135.455`, which is not
  a Doodson number of any FES2004 file constituent. This is an erratum in the
  original document; the correct pivot is `135.655` (Q1), consistent with the
  surrounding diurnal-band rows that all pivot on Q1/O1.
- Rows `245.555` and `245.645`'s pivot 1 is printed as `237.755`, likewise not
  a file constituent. The correct pivot is `235.755` (2N2), consistent with
  the surrounding semidiurnal-band rows.
- Row `135.645` is printed with the same Darwin symbol (`σ1`) as row
  `127.555`; this duplication is a genuine quirk of the source document, not
  an extraction artifact, so the name is left blank here to avoid asserting
  an incorrect label.
- Row `272.556` (T2) is bold-typeset like a main wave but is not one of the
  18 FES2004 file constituents (verified against Task 5's constituent list),
  so it is treated as a secondary (interpolated) wave per the functional rule
  in the table caption, not by boldface.
*/

/// One Table 6.7 row. `pivots = None` marks a main wave (or S1, excluded per
/// TN36 §6.3.2 radiational-tide note).
pub(crate) struct AdmittanceRow {
    /// Doodson number, e.g. `"165.555"` (K1).
    pub doodson: &'static str,
    /// Darwin symbol, e.g. `"K1"`; empty for unnamed minor waves.
    pub name: &'static str,
    /// Astronomical amplitude Hf \[m\].
    pub hf: f64,
    /// `(pivot wave 1, pivot wave 2)` Doodson numbers for linear admittance
    /// interpolation (Eq. 6.16); `None` for main waves (and S1).
    pub pivots: Option<(&'static str, &'static str)>,
}

/// Table 6.7 — all 82 rows (18 FES2004 main waves + S1 + 63 secondary
/// admittance waves).
pub(crate) static ADMITTANCE_TABLE: &[AdmittanceRow] = &[
    // ---- long-period band ----
    AdmittanceRow {
        doodson: "55.565",
        name: "Om1",
        hf: 0.02793,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "55.575",
        name: "Om2",
        hf: -0.00027,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "56.554",
        name: "Sa",
        hf: -0.00492,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "57.555",
        name: "Ssa",
        hf: -0.03100,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "58.554",
        name: "Sta",
        hf: -0.00181,
        pivots: Some(("57.555", "65.455")),
    },
    AdmittanceRow {
        doodson: "63.655",
        name: "Msm",
        hf: -0.00673,
        pivots: Some(("57.555", "65.455")),
    },
    AdmittanceRow {
        doodson: "65.445",
        name: "",
        hf: 0.00231,
        pivots: Some(("57.555", "65.455")),
    },
    AdmittanceRow {
        doodson: "65.455",
        name: "Mm",
        hf: -0.03518,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "65.465",
        name: "",
        hf: 0.00229,
        pivots: Some(("65.455", "75.555")),
    },
    AdmittanceRow {
        doodson: "65.555",
        name: "",
        hf: -0.00375,
        pivots: Some(("65.455", "75.555")),
    },
    AdmittanceRow {
        doodson: "65.655",
        name: "",
        hf: 0.00188,
        pivots: Some(("65.455", "75.555")),
    },
    AdmittanceRow {
        doodson: "73.555",
        name: "Msf",
        hf: -0.00583,
        pivots: Some(("65.455", "75.555")),
    },
    AdmittanceRow {
        doodson: "75.355",
        name: "",
        hf: -0.00288,
        pivots: Some(("65.455", "75.555")),
    },
    AdmittanceRow {
        doodson: "75.555",
        name: "Mf",
        hf: -0.06663,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "75.565",
        name: "",
        hf: -0.02762,
        pivots: Some(("75.555", "85.455")),
    },
    AdmittanceRow {
        doodson: "75.575",
        name: "",
        hf: -0.00258,
        pivots: Some(("75.555", "85.455")),
    },
    AdmittanceRow {
        doodson: "83.655",
        name: "Mstm",
        hf: -0.00242,
        pivots: Some(("75.555", "85.455")),
    },
    AdmittanceRow {
        doodson: "83.665",
        name: "",
        hf: -0.00100,
        pivots: Some(("75.555", "85.455")),
    },
    AdmittanceRow {
        doodson: "85.455",
        name: "Mtm",
        hf: -0.01276,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "85.465",
        name: "",
        hf: -0.00529,
        pivots: Some(("85.455", "93.555")),
    },
    AdmittanceRow {
        doodson: "93.555",
        name: "Msqm",
        hf: -0.00204,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "95.355",
        name: "",
        hf: -0.00169,
        pivots: Some(("85.455", "93.555")),
    },
    // ---- diurnal band ----
    AdmittanceRow {
        doodson: "117.655",
        name: "",
        hf: -0.00194,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "125.755",
        name: "2Q1",
        hf: -0.00664,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "127.555",
        name: "sigma1",
        hf: -0.00802,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "135.645",
        name: "",
        hf: -0.00947,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "135.655",
        name: "Q1",
        hf: -0.05020,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "137.445",
        name: "",
        hf: -0.00180,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "137.455",
        name: "rho1",
        hf: -0.00954,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "145.545",
        name: "",
        hf: -0.04946,
        pivots: Some(("135.655", "145.555")),
    },
    AdmittanceRow {
        doodson: "145.555",
        name: "O1",
        hf: -0.26221,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "145.755",
        name: "",
        hf: 0.00170,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "147.555",
        name: "tau1",
        hf: 0.00343,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "153.655",
        name: "",
        hf: 0.00194,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "155.455",
        name: "",
        hf: 0.00741,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "155.555",
        name: "",
        hf: -0.00399,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "155.655",
        name: "M1",
        hf: 0.02062,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "155.665",
        name: "",
        hf: 0.00414,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "157.455",
        name: "chi1",
        hf: 0.00394,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "162.556",
        name: "pi1",
        hf: -0.00714,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "163.555",
        name: "P1",
        hf: -0.12203,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "164.556",
        name: "S1",
        hf: 0.00289,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "165.545",
        name: "K1-",
        hf: -0.00730,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "165.555",
        name: "K1",
        hf: 0.36878,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "165.565",
        name: "K1+",
        hf: 0.05001,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "166.554",
        name: "psi1",
        hf: 0.00293,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "167.555",
        name: "phi1",
        hf: 0.00525,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "173.655",
        name: "theta1",
        hf: 0.00395,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "175.455",
        name: "J1",
        hf: 0.02062,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "175.465",
        name: "",
        hf: 0.00409,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "183.555",
        name: "So1",
        hf: 0.00342,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "185.355",
        name: "",
        hf: 0.00169,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "185.555",
        name: "Oo1",
        hf: 0.01129,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "185.565",
        name: "",
        hf: 0.00723,
        pivots: Some(("145.555", "165.555")),
    },
    AdmittanceRow {
        doodson: "195.455",
        name: "nu1",
        hf: 0.00216,
        pivots: Some(("145.555", "165.555")),
    },
    // ---- semi-diurnal band ----
    AdmittanceRow {
        doodson: "225.855",
        name: "3N2",
        hf: 0.00180,
        pivots: Some(("235.755", "245.655")),
    },
    AdmittanceRow {
        doodson: "227.655",
        name: "eps2",
        hf: 0.00467,
        pivots: Some(("235.755", "245.655")),
    },
    AdmittanceRow {
        doodson: "235.755",
        name: "2N2",
        hf: 0.01601,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "237.555",
        name: "mu2",
        hf: 0.01932,
        pivots: Some(("235.755", "245.655")),
    },
    AdmittanceRow {
        doodson: "245.555",
        name: "",
        hf: -0.00389,
        pivots: Some(("235.755", "245.655")),
    },
    AdmittanceRow {
        doodson: "245.645",
        name: "",
        hf: -0.00451,
        pivots: Some(("235.755", "245.655")),
    },
    AdmittanceRow {
        doodson: "245.655",
        name: "N2",
        hf: 0.12099,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "247.455",
        name: "nu2",
        hf: 0.02298,
        pivots: Some(("245.655", "255.555")),
    },
    AdmittanceRow {
        doodson: "253.755",
        name: "gamma2",
        hf: -0.00190,
        pivots: Some(("245.655", "255.555")),
    },
    AdmittanceRow {
        doodson: "254.556",
        name: "alpha2",
        hf: -0.00218,
        pivots: Some(("245.655", "255.555")),
    },
    AdmittanceRow {
        doodson: "255.545",
        name: "",
        hf: -0.02358,
        pivots: Some(("245.655", "255.555")),
    },
    AdmittanceRow {
        doodson: "255.555",
        name: "M2",
        hf: 0.63192,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "256.554",
        name: "beta2",
        hf: 0.00192,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "263.655",
        name: "lambda2",
        hf: -0.00466,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "265.455",
        name: "L2",
        hf: -0.01786,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "265.555",
        name: "",
        hf: 0.00359,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "265.655",
        name: "",
        hf: 0.00447,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "265.665",
        name: "",
        hf: 0.00197,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "272.556",
        name: "T2",
        hf: 0.01720,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "273.555",
        name: "S2",
        hf: 0.29400,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "274.554",
        name: "R2",
        hf: -0.00246,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "275.555",
        name: "K2",
        hf: 0.07996,
        pivots: None,
    },
    AdmittanceRow {
        doodson: "275.565",
        name: "K2+",
        hf: 0.02383,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "275.575",
        name: "K2++",
        hf: 0.00259,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "285.455",
        name: "eta2",
        hf: 0.00447,
        pivots: Some(("255.555", "275.555")),
    },
    AdmittanceRow {
        doodson: "285.465",
        name: "",
        hf: 0.00195,
        pivots: Some(("255.555", "275.555")),
    },
    // ---- quarter-diurnal ----
    AdmittanceRow {
        doodson: "455.555",
        name: "M4",
        hf: 0.0,
        pivots: None,
    },
];
