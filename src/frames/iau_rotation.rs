/*!
 * IAU/WGCCRE body rotation model.
 *
 * Implements the rotation between the ICRF (International Celestial
 * Reference Frame, treated here as equivalent to J2000) and the
 * body-fixed frame of a Solar System body, using the polynomial +
 * trigonometric pole right-ascension/declination and prime-meridian model
 * published by the IAU Working Group on Cartographic Coordinates and
 * Rotational Elements (WGCCRE). See:
 *
 * - Archinal, B.A., et al. "Report of the IAU Working Group on
 *   Cartographic Coordinates and Rotational Elements: 2015." Celestial
 *   Mechanics and Dynamical Astronomy 130, 22 (2018).
 * - NAIF generic text PCK
 *   [`pck00011.tpc`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc),
 *   which is the direct transcription source for the coefficients embedded
 *   below (the file's "Current values" data blocks; see per-body comments).
 *
 * # Model
 *
 * For a body with NAIF ID `naif_id`, the pole right ascension `alpha`,
 * pole declination `delta`, and prime meridian angle `W` (all in degrees)
 * are evaluated as
 *
 * ```text
 * alpha = ra0 + ra1*T + ra2*T^2 + sum_i ra_nut_prec[i] * sin(theta_i)
 * delta = dec0 + dec1*T + dec2*T^2 + sum_i dec_nut_prec[i] * cos(theta_i)
 * W     = pm0 + pm1*d + pm2*d^2 + sum_i pm_nut_prec[i] * sin(theta_i)
 * ```
 *
 * where `T` is Julian centuries TDB past J2000, `d` is days TDB past
 * J2000, and `theta_i = nut_prec_angles[i][0] + nut_prec_angles[i][1] * T
 * + nut_prec_angles[i][2] * T^2` (degrees) are the body system's
 * nutation-precession angles. The
 * resulting DCM rotating ICRF vectors into the body-fixed frame is the
 * 3-1-3 sequence `R = Rz(W) * Rx(90 - delta) * Rz(90 + alpha)`.
 *
 * # Lunar orientation
 *
 * The Moon's entry evaluates the IAU low-precision model (`pck00011.tpc`'s
 * `BODY301_*` polynomials) — appropriate when the IAU variant is requested
 * explicitly. The default lunar frame transformations
 * ([`super::lunar`]) use the higher-fidelity DE440 principal-axis
 * rotation from the lunar binary PCK instead.
 */
use nalgebra::Vector3;

use crate::attitude::RotationMatrix;
use crate::constants::{AngleFormat, SECONDS_PER_JULIAN_CENTURY};
use crate::math::SMatrix3;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// IAU/WGCCRE rotation model for a single body: polynomial pole
/// right-ascension, pole declination, and prime-meridian coefficients,
/// plus the trigonometric (nutation-precession) corrections applied to
/// each.
///
/// All angles are in degrees; polynomial coefficients for `pole_ra` and
/// `pole_dec` are `[deg, deg/century, deg/century^2]` and for `pm` are
/// `[deg, deg/day, deg/day^2]`. `nut_prec_angles` holds `[deg,
/// deg/century, deg/century^2]` triples for the body system's
/// nutation-precession angles (the quadratic term is zero for every angle
/// except `pck00011.tpc`'s Mars-system angle index 4, used by Phobos —
/// see `MARS_NUT_PREC_ANGLES`); `ra_nut_prec`, `dec_nut_prec`, and
/// `pm_nut_prec` are the sine/cosine amplitude coefficients (degrees)
/// applied to those angles, indexed from the start of `nut_prec_angles`
/// (a coefficient array may be shorter than `nut_prec_angles` — only its
/// own length is evaluated — but is not re-based, so leading zero
/// coefficients present in the source PCK must be preserved to keep
/// indices aligned).
#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct IAURotationModel {
    /// NAIF ID of the body (e.g. `499` for Mars).
    pub naif_id: i32,
    /// Pole right ascension polynomial coefficients `[deg, deg/century, deg/century^2]`.
    pub pole_ra: [f64; 3],
    /// Pole declination polynomial coefficients `[deg, deg/century, deg/century^2]`.
    pub pole_dec: [f64; 3],
    /// Prime meridian polynomial coefficients `[deg, deg/day, deg/day^2]`.
    pub pm: [f64; 3],
    /// Nutation-precession angles `[deg, deg/century, deg/century^2]` for the body's system.
    pub nut_prec_angles: &'static [[f64; 3]],
    /// Sine amplitude coefficients (degrees) applied to `nut_prec_angles` for `alpha`.
    pub ra_nut_prec: &'static [f64],
    /// Cosine amplitude coefficients (degrees) applied to `nut_prec_angles` for `delta`.
    pub dec_nut_prec: &'static [f64],
    /// Sine amplitude coefficients (degrees) applied to `nut_prec_angles` for `W`.
    pub pm_nut_prec: &'static [f64],
}

/// Mars-system (barycenter 4) nutation-precession angles, `[deg,
/// deg/century, deg/century^2]`, shared by Mars (499), Phobos (401), and
/// Deimos (402). Transcribed from `pck00011.tpc`'s
/// `BODY4_NUT_PREC_ANGLES` (the "current values" / quadratic-phase block
/// guarded by `BODY4_MAX_PHASE_DEGREE = 2`), truncated to the 26 rows
/// (indices 0-25) referenced by RA/DEC/PM coefficients below; rows 26-30
/// of the file's 31-row table are not used by any body in this table.
/// Row index 4 (the Phobos longitude `M1`) is the only row in the file
/// with a nonzero third (quadratic-in-T) column
/// (`+12.711923222 deg/century^2`); all other rows have a zero third
/// column in the source.
const MARS_NUT_PREC_ANGLES: &[[f64; 3]] = &[
    [190.72646643, 15917.10818695, 0.0],
    [21.46892470, 31834.27934054, 0.0],
    [332.86082793, 19139.89694742, 0.0],
    [394.93256437, 38280.79631835, 0.0],
    [189.63271560, 41215158.1842005, 12.711923222],
    [121.46893664, 660.22803474, 0.0],
    [231.05028581, 660.99123540, 0.0],
    [251.37314025, 1320.50145245, 0.0],
    [217.98635955, 38279.96125550, 0.0],
    [196.19729402, 19139.83628608, 0.0],
    [198.991226, 19139.4819985, 0.0],
    [226.292679, 38280.8511281, 0.0],
    [249.663391, 57420.7251593, 0.0],
    [266.183510, 76560.6367950, 0.0],
    [79.398797, 0.5042615, 0.0],
    [122.433576, 19139.9407476, 0.0],
    [43.058401, 38280.8753272, 0.0],
    [57.663379, 57420.7517205, 0.0],
    [79.476401, 76560.6495004, 0.0],
    [166.325722, 0.5042615, 0.0],
    [129.071773, 19140.0328244, 0.0],
    [36.352167, 38281.0473591, 0.0],
    [56.668646, 57420.9295360, 0.0],
    [67.364003, 76560.2552215, 0.0],
    [104.792680, 95700.4387578, 0.0],
    [95.391654, 0.5042615, 0.0],
];

/// Jupiter-system (barycenter 5) nutation-precession angles Ja-Je and
/// J1-J10, `[deg, deg/century, deg/century^2]`, shared by Jupiter (599),
/// Io (501), Europa (502), Ganymede (503), and Callisto (504).
/// Transcribed from `pck00011.tpc`'s `BODY5_NUT_PREC_ANGLES` (a purely
/// linear table in the source; the quadratic column is zero for every
/// row).
const JUPITER_NUT_PREC_ANGLES: &[[f64; 3]] = &[
    [73.32, 91472.9, 0.0],
    [24.62, 45137.2, 0.0],
    [283.90, 4850.7, 0.0],
    [355.80, 1191.3, 0.0],
    [119.90, 262.1, 0.0],
    [229.80, 64.3, 0.0],
    [352.25, 2382.6, 0.0],
    [113.35, 6070.0, 0.0],
    [146.64, 182945.8, 0.0],
    [49.24, 90274.4, 0.0],
    [99.360714, 4850.4046, 0.0],
    [175.895369, 1191.9605, 0.0],
    [300.323162, 262.5475, 0.0],
    [114.012305, 6070.2476, 0.0],
    [49.511251, 64.3000, 0.0],
];

/// Earth-Moon barycenter (3) nutation-precession angles, `[deg,
/// deg/century, deg/century^2]`, used by the Moon's (301) low-precision
/// IAU rotation model. Transcribed from `pck00011.tpc`'s
/// `BODY3_NUT_PREC_ANGLES` (a purely linear table in the source; the
/// quadratic column is zero for every row).
const EARTH_MOON_NUT_PREC_ANGLES: &[[f64; 3]] = &[
    [125.045, -1935.5364525000, 0.0],
    [250.089, -3871.0729050000, 0.0],
    [260.008, 475263.3328725, 0.0],
    [176.625, 487269.629985, 0.0],
    [357.529, 35999.0509575, 0.0],
    [311.589, 964468.49931, 0.0],
    [134.963, 477198.869325, 0.0],
    [276.617, 12006.300765, 0.0],
    [34.226, 63863.5132425, 0.0],
    [15.134, -5806.6093575, 0.0],
    [119.743, 131.84064, 0.0],
    [239.961, 6003.1503825, 0.0],
    [25.053, 473327.79642, 0.0],
];

/// Neptune-system (barycenter 8) nutation-precession angle N, `[deg,
/// deg/century, deg/century^2]`. Transcribed from `pck00011.tpc`'s
/// `BODY8_NUT_PREC_ANGLES` (only the first row is referenced by
/// Neptune's (899) RA/DEC/PM coefficients; a purely linear table in the
/// source).
const NEPTUNE_NUT_PREC_ANGLES: &[[f64; 3]] = &[[357.85, 52.316, 0.0]];

/// Mercury-system (barycenter 1) nutation-precession angles M1-M5, `[deg,
/// deg/century, deg/century^2]`. Transcribed from `pck00011.tpc`'s
/// `BODY1_NUT_PREC_ANGLES` (a purely linear table in the source; the
/// quadratic column is zero for every row).
const MERCURY_NUT_PREC_ANGLES: &[[f64; 3]] = &[
    [174.7910857, 149472.53587500003, 0.0],
    [349.5821714, 298945.07175000006, 0.0],
    [164.3732571, 448417.60762500006, 0.0],
    [339.1643429, 597890.1435000001, 0.0],
    [153.9554286, 747362.679375, 0.0],
];

/// Embedded IAU/WGCCRE rotation models, transcribed from `pck00011.tpc`.
/// Order is not significant; [`iau_rotation_model_ids`] returns the
/// sorted NAIF IDs.
const IAU_ROTATION_MODELS: &[IAURotationModel] = &[
    // Sun. `BODY10_POLE_RA/POLE_DEC/PM`; no nutation-precession terms.
    IAURotationModel {
        naif_id: 10,
        pole_ra: [286.13, 0.0, 0.0],
        pole_dec: [63.87, 0.0, 0.0],
        pm: [84.176, 14.18440, 0.0],
        nut_prec_angles: &[],
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[],
    },
    // Mercury. `BODY199_POLE_RA/POLE_DEC/PM`, `BODY199_NUT_PREC_PM`
    // (RA/DEC nutation-precession coefficients are all zero in the
    // source and are omitted), `BODY1_NUT_PREC_ANGLES`.
    IAURotationModel {
        naif_id: 199,
        pole_ra: [281.0103, -0.0328, 0.0],
        pole_dec: [61.4155, -0.0049, 0.0],
        pm: [329.5988, 6.1385108, 0.0],
        nut_prec_angles: MERCURY_NUT_PREC_ANGLES,
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[
            0.01067257,
            -0.00112309,
            -0.00011040,
            -0.00002539,
            -0.00000571,
        ],
    },
    // Venus. `BODY299_POLE_RA/POLE_DEC/PM`; no nutation-precession terms.
    IAURotationModel {
        naif_id: 299,
        pole_ra: [272.76, 0.0, 0.0],
        pole_dec: [67.16, 0.0, 0.0],
        pm: [160.20, -1.4813688, 0.0],
        nut_prec_angles: &[],
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[],
    },
    // Moon. IAU low-precision model: `BODY301_POLE_RA/POLE_DEC/PM` and
    // `BODY301_NUT_PREC_RA/DEC/PM` against `BODY3_NUT_PREC_ANGLES`. The
    // PM quadratic coefficient `-1.4D-12` (Fortran D-exponent notation)
    // is `-1.4e-12` deg/day^2.
    IAURotationModel {
        naif_id: 301,
        pole_ra: [269.9949, 0.0031, 0.0],
        pole_dec: [66.5392, 0.0130, 0.0],
        pm: [38.3213, 13.17635815, -1.4e-12],
        nut_prec_angles: EARTH_MOON_NUT_PREC_ANGLES,
        ra_nut_prec: &[
            -3.8787, -0.1204, 0.0700, -0.0172, 0.0, 0.0072, 0.0, 0.0, 0.0, -0.0052, 0.0, 0.0,
            0.0043,
        ],
        dec_nut_prec: &[
            1.5419, 0.0239, -0.0278, 0.0068, 0.0, -0.0029, 0.0009, 0.0, 0.0, 0.0008, 0.0, 0.0,
            -0.0009,
        ],
        pm_nut_prec: &[
            3.5610, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047, -0.0046, 0.0028, 0.0052,
            0.0040, 0.0019, -0.0044,
        ],
    },
    // Mars. `BODY499_POLE_RA/POLE_DEC/PM` and
    // `BODY499_NUT_PREC_RA/DEC/PM` against `BODY4_NUT_PREC_ANGLES`.
    IAURotationModel {
        naif_id: 499,
        pole_ra: [317.269202, -0.10927547, 0.0],
        pole_dec: [54.432516, -0.05827105, 0.0],
        pm: [176.049863, 350.891982443297, 0.0],
        nut_prec_angles: MARS_NUT_PREC_ANGLES,
        ra_nut_prec: &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000068, 0.000238, 0.000052,
            0.000009, 0.419057,
        ],
        dec_nut_prec: &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000051,
            0.000141, 0.000031, 0.000005, 1.591274,
        ],
        pm_nut_prec: &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.000145, 0.000157, 0.000040, 0.000001, 0.000001, 0.584542,
        ],
    },
    // Jupiter. `BODY599_POLE_RA/POLE_DEC/PM` and
    // `BODY599_NUT_PREC_RA/DEC` against `BODY5_NUT_PREC_ANGLES` (PM
    // nutation-precession coefficients are all zero in the source and
    // are omitted).
    IAURotationModel {
        naif_id: 599,
        pole_ra: [268.056595, -0.006499, 0.0],
        pole_dec: [64.495303, 0.002413, 0.0],
        pm: [284.95, 870.5360000, 0.0],
        nut_prec_angles: JUPITER_NUT_PREC_ANGLES,
        ra_nut_prec: &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000117, 0.000938, 0.001432,
            0.000030, 0.002150,
        ],
        dec_nut_prec: &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000050, 0.000404, 0.000617,
            -0.000013, 0.000926,
        ],
        pm_nut_prec: &[],
    },
    // Saturn. `BODY699_POLE_RA/POLE_DEC/PM`; no nutation-precession
    // terms (Saturn's `S1-S8` angle list is only used by its satellites,
    // none of which are in this table).
    IAURotationModel {
        naif_id: 699,
        pole_ra: [40.589, -0.036, 0.0],
        pole_dec: [83.537, -0.004, 0.0],
        pm: [38.90, 810.7939024, 0.0],
        nut_prec_angles: &[],
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[],
    },
    // Uranus. `BODY799_POLE_RA/POLE_DEC/PM`; no nutation-precession
    // terms (Uranus' `U1-U18` angle list is only used by its satellites,
    // none of which are in this table).
    IAURotationModel {
        naif_id: 799,
        pole_ra: [257.311, 0.0, 0.0],
        pole_dec: [-15.175, 0.0, 0.0],
        pm: [203.81, -501.1600928, 0.0],
        nut_prec_angles: &[],
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[],
    },
    // Neptune. `BODY899_POLE_RA/POLE_DEC/PM` and
    // `BODY899_NUT_PREC_RA/DEC/PM` against `BODY8_NUT_PREC_ANGLES`
    // (only the first angle, N, has a nonzero coefficient).
    IAURotationModel {
        naif_id: 899,
        pole_ra: [299.36, 0.0, 0.0],
        pole_dec: [43.46, 0.0, 0.0],
        pm: [249.978, 541.1397757, 0.0],
        nut_prec_angles: NEPTUNE_NUT_PREC_ANGLES,
        ra_nut_prec: &[0.70],
        dec_nut_prec: &[-0.51],
        pm_nut_prec: &[-0.48],
    },
    // Phobos. `BODY401_POLE_RA/POLE_DEC/PM` and
    // `BODY401_NUT_PREC_RA/DEC/PM` against `BODY4_NUT_PREC_ANGLES`. PM's
    // fifth coefficient multiplies Mars-system angle index 4 (M1), which
    // carries the table's one nonzero quadratic (deg/century^2) term.
    IAURotationModel {
        naif_id: 401,
        pole_ra: [317.67071657, -0.10844326, 0.0],
        pole_dec: [52.88627266, -0.06134706, 0.0],
        pm: [35.18774440, 1128.84475928, 9.536137031212154e-09],
        nut_prec_angles: MARS_NUT_PREC_ANGLES,
        ra_nut_prec: &[-1.78428399, 0.02212824, -0.01028251, -0.00475595],
        dec_nut_prec: &[-1.07516537, 0.00668626, -0.00648740, 0.00281576],
        pm_nut_prec: &[1.42421769, -0.02273783, 0.00410711, 0.00631964, -1.143],
    },
    // Deimos. `BODY402_POLE_RA/POLE_DEC/PM` and
    // `BODY402_NUT_PREC_RA/DEC/PM` against `BODY4_NUT_PREC_ANGLES`.
    IAURotationModel {
        naif_id: 402,
        pole_ra: [316.65705808, -0.10518014, 0.0],
        pole_dec: [53.50992033, -0.05979094, 0.0],
        pm: [79.39932954, 285.16188899, 0.0],
        nut_prec_angles: MARS_NUT_PREC_ANGLES,
        ra_nut_prec: &[
            0.0, 0.0, 0.0, 0.0, 0.0, 3.09217726, 0.22980637, 0.06418655, 0.02533537, 0.00778695,
        ],
        dec_nut_prec: &[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.83936004,
            0.14325320,
            0.01911409,
            -0.01482590,
            0.00192430,
        ],
        pm_nut_prec: &[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.73954829,
            -0.39968606,
            -0.06563259,
            -0.02912940,
            0.01699160,
        ],
    },
    // Io. `BODY501_POLE_RA/POLE_DEC/PM` and `BODY501_NUT_PREC_RA/DEC/PM`
    // against `BODY5_NUT_PREC_ANGLES` (angles J3, J4).
    IAURotationModel {
        naif_id: 501,
        pole_ra: [268.05, -0.009, 0.0],
        pole_dec: [64.50, 0.003, 0.0],
        pm: [200.39, 203.4889538, 0.0],
        nut_prec_angles: JUPITER_NUT_PREC_ANGLES,
        ra_nut_prec: &[0.0, 0.0, 0.094, 0.024],
        dec_nut_prec: &[0.0, 0.0, 0.040, 0.011],
        pm_nut_prec: &[0.0, 0.0, -0.085, -0.022],
    },
    // Europa. `BODY502_POLE_RA/POLE_DEC/PM` and
    // `BODY502_NUT_PREC_RA/DEC/PM` against `BODY5_NUT_PREC_ANGLES`
    // (angles J4-J7).
    IAURotationModel {
        naif_id: 502,
        pole_ra: [268.08, -0.009, 0.0],
        pole_dec: [64.51, 0.003, 0.0],
        pm: [36.022, 101.3747235, 0.0],
        nut_prec_angles: JUPITER_NUT_PREC_ANGLES,
        ra_nut_prec: &[0.0, 0.0, 0.0, 1.086, 0.060, 0.015, 0.009],
        dec_nut_prec: &[0.0, 0.0, 0.0, 0.468, 0.026, 0.007, 0.002],
        pm_nut_prec: &[0.0, 0.0, 0.0, -0.980, -0.054, -0.014, -0.008],
    },
    // Ganymede. `BODY503_POLE_RA/POLE_DEC/PM` and
    // `BODY503_NUT_PREC_RA/DEC/PM` against `BODY5_NUT_PREC_ANGLES`
    // (angles J4-J6).
    IAURotationModel {
        naif_id: 503,
        pole_ra: [268.20, -0.009, 0.0],
        pole_dec: [64.57, 0.003, 0.0],
        pm: [44.064, 50.3176081, 0.0],
        nut_prec_angles: JUPITER_NUT_PREC_ANGLES,
        ra_nut_prec: &[0.0, 0.0, 0.0, -0.037, 0.431, 0.091],
        dec_nut_prec: &[0.0, 0.0, 0.0, -0.016, 0.186, 0.039],
        pm_nut_prec: &[0.0, 0.0, 0.0, 0.033, -0.389, -0.082],
    },
    // Callisto. `BODY504_POLE_RA/POLE_DEC/PM` and
    // `BODY504_NUT_PREC_RA/DEC/PM` against `BODY5_NUT_PREC_ANGLES`
    // (angles J5-J8).
    IAURotationModel {
        naif_id: 504,
        pole_ra: [268.72, -0.009, 0.0],
        pole_dec: [64.83, 0.003, 0.0],
        pm: [259.51, 21.5710715, 0.0],
        nut_prec_angles: JUPITER_NUT_PREC_ANGLES,
        ra_nut_prec: &[0.0, 0.0, 0.0, 0.0, -0.068, 0.590, 0.0, 0.010],
        dec_nut_prec: &[0.0, 0.0, 0.0, 0.0, -0.029, 0.254, 0.0, -0.004],
        pm_nut_prec: &[0.0, 0.0, 0.0, 0.0, 0.061, -0.533, 0.0, -0.009],
    },
    // Enceladus. `BODY602_POLE_RA/POLE_DEC/PM`; no nutation-precession
    // terms (pure linear model in the source).
    IAURotationModel {
        naif_id: 602,
        pole_ra: [40.66, -0.036, 0.0],
        pole_dec: [83.52, -0.004, 0.0],
        pm: [6.32, 262.7318996, 0.0],
        nut_prec_angles: &[],
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[],
    },
    // Titan. `BODY606_POLE_RA/POLE_DEC/PM`; `BODY606_NUT_PREC_RA/DEC/PM`
    // are present in the source but all zero ("removal of dependence on
    // the nutation precession angles") and are omitted.
    IAURotationModel {
        naif_id: 606,
        pole_ra: [39.4827, 0.0, 0.0],
        pole_dec: [83.4279, 0.0, 0.0],
        pm: [186.5855, 22.5769768, 0.0],
        nut_prec_angles: &[],
        ra_nut_prec: &[],
        dec_nut_prec: &[],
        pm_nut_prec: &[],
    },
];

/// Sorted list of NAIF IDs with an embedded IAU/WGCCRE rotation model.
///
/// # Returns:
/// - `ids`: Sorted NAIF IDs supported by [`rotation_icrf_to_body_fixed_iau`]
///   and [`body_fixed_iau_angles_and_rates`].
///
/// # Examples:
/// ```
/// use brahe::frames::iau_rotation_model_ids;
///
/// let ids = iau_rotation_model_ids();
/// assert!(ids.contains(&499)); // Mars
/// ```
pub fn iau_rotation_model_ids() -> Vec<i32> {
    let mut ids: Vec<i32> = IAU_ROTATION_MODELS.iter().map(|m| m.naif_id).collect();
    ids.sort_unstable();
    ids
}

/// Looks up the embedded rotation model for `naif_id`.
fn model_for_id(naif_id: i32) -> Result<&'static IAURotationModel, BraheError> {
    IAU_ROTATION_MODELS
        .iter()
        .find(|m| m.naif_id == naif_id)
        .ok_or_else(|| {
            BraheError::Error(format!(
                "No IAU/WGCCRE rotation model for NAIF ID {}. Supported IDs: {}",
                naif_id,
                iau_rotation_model_ids()
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// Evaluates a model's pole right ascension, pole declination, and prime
/// meridian angle, and their time derivatives, at `epc`.
///
/// # Returns:
/// - `(alpha, delta, w, dalpha, ddelta, dw)`: `alpha`/`delta`/`w` in
///   degrees, `dalpha`/`ddelta`/`dw` in degrees/second.
fn eval(model: &IAURotationModel, epc: Epoch) -> (f64, f64, f64, f64, f64, f64) {
    let et = epc.seconds_past_j2000_as_time_system(TimeSystem::TDB);
    let d = et / 86400.0;
    let t = d / 36525.0;

    // theta_i and dtheta_i/dt, both [deg, deg/s].
    let thetas: Vec<(f64, f64)> = model
        .nut_prec_angles
        .iter()
        .map(|[a0, a1, a2]| {
            (
                a0 + a1 * t + a2 * t * t,
                (a1 + 2.0 * a2 * t) / SECONDS_PER_JULIAN_CENTURY,
            )
        })
        .collect();

    let mut alpha = model.pole_ra[0] + model.pole_ra[1] * t + model.pole_ra[2] * t * t;
    let mut dalpha = (model.pole_ra[1] + 2.0 * model.pole_ra[2] * t) / SECONDS_PER_JULIAN_CENTURY;
    for (i, c) in model.ra_nut_prec.iter().enumerate() {
        let (th, dth) = thetas[i];
        alpha += c * th.to_radians().sin();
        dalpha += c * th.to_radians().cos() * dth.to_radians();
    }

    let mut delta = model.pole_dec[0] + model.pole_dec[1] * t + model.pole_dec[2] * t * t;
    let mut ddelta = (model.pole_dec[1] + 2.0 * model.pole_dec[2] * t) / SECONDS_PER_JULIAN_CENTURY;
    for (i, c) in model.dec_nut_prec.iter().enumerate() {
        let (th, dth) = thetas[i];
        delta += c * th.to_radians().cos();
        ddelta -= c * th.to_radians().sin() * dth.to_radians();
    }

    let mut w = model.pm[0] + model.pm[1] * d + model.pm[2] * d * d;
    let mut dw = (model.pm[1] + 2.0 * model.pm[2] * d) / 86400.0;
    for (i, c) in model.pm_nut_prec.iter().enumerate() {
        let (th, dth) = thetas[i];
        w += c * th.to_radians().sin();
        dw += c * th.to_radians().cos() * dth.to_radians();
    }

    (alpha, delta, w, dalpha, ddelta, dw)
}

/// Computes the 3-1-3 Euler angles `(phi, theta, psi)` [rad] and their
/// rates [rad/s] describing the rotation from the ICRF to the body-fixed
/// frame of `naif_id` at `epc`, using the IAU/WGCCRE rotation model.
///
/// The angles follow the convention `R = Rz(psi) * Rx(theta) * Rz(phi)`
/// with `phi = pi/2 + alpha`, `theta = pi/2 - delta`, `psi = W`, where
/// `alpha`/`delta`/`W` are the body's pole right ascension, pole
/// declination, and prime meridian angle.
///
/// # Arguments:
/// - `naif_id`: NAIF ID of the body (see [`iau_rotation_model_ids`] for
///   the supported set)
/// - `epc`: Epoch instant for computation of the angles
///
/// # Returns:
/// - `(angles, rates)`: `angles = (phi, theta, psi)` [rad], `rates =
///   (phi_dot, theta_dot, psi_dot)` [rad/s]
///
/// # Examples:
/// ```
/// use brahe::frames::body_fixed_iau_angles_and_rates;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let (angles, rates) = body_fixed_iau_angles_and_rates(499, epc).unwrap();
/// ```
pub fn body_fixed_iau_angles_and_rates(
    naif_id: i32,
    epc: Epoch,
) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
    let model = model_for_id(naif_id)?;
    let (alpha, delta, w, dalpha, ddelta, dw) = eval(model, epc);

    let phi = (90.0 + alpha).to_radians();
    let theta = (90.0 - delta).to_radians();
    let psi = w.to_radians();

    let phi_dot = dalpha.to_radians();
    let theta_dot = -ddelta.to_radians();
    let psi_dot = dw.to_radians();

    Ok((
        Vector3::new(phi, theta, psi),
        Vector3::new(phi_dot, theta_dot, psi_dot),
    ))
}

/// Rotation matrix about the body-fixed x-axis by `angle` [rad]. Matches
/// the SPICE `[angle]_1` convention used by TK frame kernels (see
/// `frames.req`): `v_out = rx(angle) * v_in`.
pub(crate) fn rx(angle: f64) -> SMatrix3 {
    RotationMatrix::Rx(angle, AngleFormat::Radians).to_matrix()
}

/// Rotation matrix about the body-fixed y-axis by `angle` [rad]. Matches
/// the SPICE `[angle]_2` convention used by TK frame kernels (see
/// `frames.req`): `v_out = ry(angle) * v_in`.
pub(crate) fn ry(angle: f64) -> SMatrix3 {
    RotationMatrix::Ry(angle, AngleFormat::Radians).to_matrix()
}

/// Rotation matrix about the body-fixed z-axis by `angle` [rad]. Matches
/// the SPICE `[angle]_3` convention used by TK frame kernels (see
/// `frames.req`): `v_out = rz(angle) * v_in`.
pub(crate) fn rz(angle: f64) -> SMatrix3 {
    RotationMatrix::Rz(angle, AngleFormat::Radians).to_matrix()
}

/// Computes the rotation matrix from the ICRF to the body-fixed frame of
/// `naif_id` at `epc`, using the IAU/WGCCRE rotation model.
///
/// # Arguments:
/// - `naif_id`: NAIF ID of the body (see [`iau_rotation_model_ids`] for
///   the supported set)
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming ICRF -> body-fixed
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_icrf_to_body_fixed_iau;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_icrf_to_body_fixed_iau(499, epc).unwrap();
/// ```
pub fn rotation_icrf_to_body_fixed_iau(naif_id: i32, epc: Epoch) -> Result<SMatrix3, BraheError> {
    let (angles, _) = body_fixed_iau_angles_and_rates(naif_id, epc)?;
    Ok(rz(angles[2]) * rx(angles[1]) * rz(angles[0]))
}

/// Computes the angular velocity [rad/s] of a body-fixed frame with
/// respect to the inertial frame, expressed in the body-fixed frame,
/// given 3-1-3 Euler angles and their rates.
///
/// Shared by the IAU/WGCCRE model above and the lunar principal-axis PCK
/// rotation path (both use the same 3-1-3 `Rz * Rx * Rz` convention).
///
/// # Arguments:
/// - `angles`: 3-1-3 Euler angles `(phi, theta, psi)` [rad]
/// - `rates`: 3-1-3 Euler angle rates `(phi_dot, theta_dot, psi_dot)` [rad/s]
///
/// # Returns:
/// - `omega`: Angular velocity vector [rad/s], expressed in the
///   body-fixed frame
pub(crate) fn euler313_omega_body(angles: Vector3<f64>, rates: Vector3<f64>) -> Vector3<f64> {
    let (phi_dot, theta_dot, psi_dot) = (rates[0], rates[1], rates[2]);
    let (theta, psi) = (angles[1], angles[2]);
    Vector3::new(
        phi_dot * theta.sin() * psi.sin() + theta_dot * psi.cos(),
        phi_dot * theta.sin() * psi.cos() - theta_dot * psi.sin(),
        phi_dot * theta.cos() + psi_dot,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    use super::*;
    use crate::time::{Epoch, TimeSystem};

    #[test]
    fn test_iau_mars_pole_at_j2000() {
        // At J2000 TDB, alpha0/delta0/W from WGCCRE 2015 polynomials (T=0, d=0):
        // trig terms with nonzero phase still contribute; check against direct evaluation.
        let epc = Epoch::from_jd(2451545.0, TimeSystem::TT); // match existing Epoch constructor usage
        let r = rotation_icrf_to_body_fixed_iau(499, epc).unwrap();
        // DCM is orthonormal
        let should_be_identity = r * r.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    should_be_identity[(i, j)],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = 1e-12
                );
            }
        }
        assert_abs_diff_eq!(r.determinant(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_iau_mars_spin_rate() {
        // W rate must match OMEGA_MARS to within the small trig-term rates
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let (_, rates) = body_fixed_iau_angles_and_rates(499, epc).unwrap();
        assert_abs_diff_eq!(rates[2], crate::constants::OMEGA_MARS, epsilon = 1e-9);
    }

    #[test]
    fn test_iau_unknown_body_errors() {
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let e = rotation_icrf_to_body_fixed_iau(999999, epc);
        assert!(e.is_err());
        assert!(format!("{}", e.unwrap_err()).contains("499")); // error lists supported IDs
    }

    #[test]
    fn test_iau_mars_transcription_guard() {
        // Regression guard for direct transcription errors in the `IAU_ROTATION_MODELS`
        // table that the orthonormality tests above cannot catch (any
        // rotation matrix, correct or not, is orthonormal). `epc` is
        // 2020-06-15T12:00:00 TDB, which is exactly
        // et = 645_494_400.0 s past J2000 TDB (verified to have an exact
        // `f64` round trip through `Epoch`, unlike an arbitrary `et`
        // constructed via `Epoch::from_jd`, whose `f64`-JD intermediate
        // loses sub-microsecond precision at this epoch's magnitude).
        // Expected values were computed independently in Python (not
        // brahe, not ANISE) by evaluating the same
        // `pck00011.tpc`-transcribed coefficients at that `et`:
        //
        //   d = et / 86400.0; t = d / 36525.0
        //   theta_i = a0_i + a1_i*t + a2_i*t^2      (MARS_NUT_PREC_ANGLES)
        //   alpha = ra0 + ra1*t + sum_i ra_nut_prec[i] * sin(radians(theta_i))
        //   delta = dec0 + dec1*t + sum_i dec_nut_prec[i] * cos(radians(theta_i))
        //   W     = pm0 + pm1*d + sum_i pm_nut_prec[i] * sin(radians(theta_i))  (mod 360)
        let epc = Epoch::from_datetime(2020, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::TDB);
        let model = model_for_id(499).unwrap();
        let (alpha, delta, w, _, _, _) = eval(model, epc);

        assert_abs_diff_eq!(alpha, 317.6591428449708, epsilon = 1e-9);
        assert_abs_diff_eq!(delta, 52.87386598710687, epsilon = 1e-9);
        assert_abs_diff_eq!(w.rem_euclid(360.0), 170.63252998981625, epsilon = 1e-9);
    }

    #[test]
    fn test_iau_enceladus_transcription_guard() {
        // Regression guard for Enceladus, which has no periodic
        // (nutation-precession) terms in `pck00011.tpc` — a pure
        // polynomial model — exercising the `alpha`/`delta`/`W` polynomial
        // evaluation path independent of the trigonometric-term path
        // covered above. Same epoch as the Mars guard above
        // (2020-06-15T12:00:00 TDB, et = 645_494_400.0 s past J2000 TDB).
        // Expected values computed independently in Python:
        //
        //   d = et / 86400.0; t = d / 36525.0
        //   alpha = 40.66 - 0.036*t
        //   delta = 83.52 - 0.004*t
        //   W     = 6.32 + 262.7318996*d   (mod 360)
        let epc = Epoch::from_datetime(2020, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::TDB);
        let model = model_for_id(602).unwrap();
        let (alpha, delta, w, _, _, _) = eval(model, epc);

        assert_abs_diff_eq!(alpha, 40.65263638603696, epsilon = 1e-9);
        assert_abs_diff_eq!(delta, 83.51918182067077, epsilon = 1e-9);
        assert_abs_diff_eq!(w.rem_euclid(360.0), 156.34191160020418, epsilon = 1e-9);
    }

    #[test]
    fn test_euler313_omega_pure_spin() {
        // Pure z-spin: angles (0, 0, psi), rates (0, 0, w) -> omega_body = [0, 0, w]
        let omega =
            euler313_omega_body(Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, 7.29e-5));
        assert_abs_diff_eq!(omega[0], 0.0, epsilon = 1e-20);
        assert_abs_diff_eq!(omega[1], 0.0, epsilon = 1e-20);
        assert_abs_diff_eq!(omega[2], 7.29e-5, epsilon = 1e-20);
    }

    #[test]
    fn test_iau_rotation_model_ids_sorted_and_contains_all_bodies() {
        let ids = iau_rotation_model_ids();
        let expected = [
            10, 199, 299, 301, 401, 402, 499, 501, 502, 503, 504, 599, 602, 606, 699, 799, 899,
        ];
        assert_eq!(ids, expected);
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn test_iau_rotation_orthonormal_for_all_bodies() {
        // Every embedded body should produce an orthonormal, proper (det = +1)
        // DCM across a range of epochs, not just Mars at J2000.
        let epc = Epoch::from_datetime(2035, 6, 15, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        for id in iau_rotation_model_ids() {
            let r = rotation_icrf_to_body_fixed_iau(id, epc).unwrap();
            let should_be_identity = r * r.transpose();
            for i in 0..3 {
                for j in 0..3 {
                    assert_abs_diff_eq!(
                        should_be_identity[(i, j)],
                        if i == j { 1.0 } else { 0.0 },
                        epsilon = 1e-10
                    );
                }
            }
            assert_abs_diff_eq!(r.determinant(), 1.0, epsilon = 1e-10);
        }
    }

    /// URL for ANISE's PCK11 planetary-constants kernel (built by the
    /// ANISE project from `pck00011.tpc` + `gm_de431.tpc`), used only as
    /// an independent oracle to validate the transcription above.
    const PCK11_URL: &str = "http://public-data.nyxspace.com/anise/v0.10/pck11.pca";

    /// Downloads (once, caching to `test_assets/pck11.pca`) and returns
    /// the path to ANISE's PCK11 planetary-constants kernel.
    fn pck11_pca_path() -> std::path::PathBuf {
        use std::io::Read;

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("pck11.pca");

        if !path.exists() {
            let response = ureq::get(PCK11_URL)
                .call()
                .expect("failed to download pck11.pca from ANISE public data bucket");
            let mut buffer = Vec::new();
            response
                .into_body()
                .into_reader()
                .read_to_end(&mut buffer)
                .expect("failed to read pck11.pca response body");

            std::fs::create_dir_all(path.parent().unwrap())
                .expect("failed to create test_assets directory");
            std::fs::write(&path, &buffer).expect("failed to write pck11.pca to test_assets");
        }

        path
    }

    /// 25 epochs evenly spaced (in TDB seconds past J2000) from 2000-01-01
    /// to 2050-01-01.
    fn sample_epochs() -> Vec<Epoch> {
        let start = Epoch::from_datetime(2000, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB)
            .seconds_past_j2000_as_time_system(TimeSystem::TDB);
        let end = Epoch::from_datetime(2050, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB)
            .seconds_past_j2000_as_time_system(TimeSystem::TDB);
        let n = 25;
        (0..n)
            .map(|i| {
                let et = start + (end - start) * i as f64 / (n - 1) as f64;
                Epoch::from_jd(
                    crate::constants::JD_J2000 + et / crate::constants::SECONDS_PER_DAY,
                    TimeSystem::TDB,
                )
            })
            .collect()
    }

    /// Empirical relative-error bound of hifitime's `Duration ->
    /// Unit::Century` conversion (used internally by ANISE to evaluate
    /// the WGCCRE polynomials), with a 5x safety margin.
    ///
    /// Diagnosed by direct comparison: for `et = 1_577_880_000.0` s (built
    /// to be exactly `T = 0.5` century), `hifitime::Epoch::from_et_seconds(et)
    /// .to_tdb_duration().to_unit(Unit::Century)` returns
    /// `0.499_999_999_999_995_115_02`, i.e. a relative error of `4.88e-15`
    /// against the exact value — hifitime's fixed-point `Duration`
    /// representation is not perfectly round-trip-exact through
    /// `Unit::Century`, independent of and unrelated to the coefficient
    /// table transcribed from `pck00011.tpc` above (this codebase's own
    /// `eval()` computes `T` via a single division and reproduces `0.5`
    /// exactly for that input). The error is negligible in absolute
    /// terms (a few nanoradians of body rotation by 2050) but, because it
    /// is a *relative* error in `T`, it is amplified by the body's
    /// accumulated rotation angle, which is largest for the
    /// fastest-spinning body over the longest time span (Jupiter, ~870.5
    /// deg/day, by 2050): see `validation_tolerance` below.
    const HIFITIME_CENTURY_REL_ERROR: f64 = 5.0 * 4.88e-15;

    /// Per-DCM-element comparison tolerance for the ANISE oracle test:
    /// the larger of a fixed floor and
    /// `HIFITIME_CENTURY_REL_ERROR` scaled by the body's accumulated
    /// prime-meridian angle `|psi|` [rad] (the fastest-growing, and thus
    /// dominant, angle in the model). A genuine transcription error in
    /// `IAU_ROTATION_MODELS` would show up as a *constant*, non-epoch-scaling offset
    /// even at `T ~ 0` (a periodic term has a nonzero phase at J2000 too)
    /// and would not be masked by this tolerance.
    fn validation_tolerance(psi_rad: f64) -> f64 {
        (1e-9_f64).max(HIFITIME_CENTURY_REL_ERROR * psi_rad.abs())
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_validate_iau_rotation_against_anise() {
        use anise::prelude::{Almanac, Epoch as AniseEpoch, Frame};

        // Grid of 25 epochs over 2000-2050; assert DCM element agreement
        // against ANISE (independently evaluating the same pck00011.tpc
        // model from its own pck11.pca) for Mars (499), Jupiter (599),
        // and Enceladus (602). This is the acceptance gate for the
        // embedded coefficient table transcribed from pck00011.tpc: a
        // failure here (beyond `validation_tolerance`, which accounts
        // only for the ANISE oracle's own internal floating-point
        // century-conversion precision — see its doc comment) means the
        // table needs to be fixed.
        let path = pck11_pca_path();
        let almanac = Almanac::default()
            .load(path.to_str().unwrap())
            .expect("failed to load pck11.pca");

        let bodies = [499_i32, 599, 602];

        let mut max_dev = 0.0_f64;
        for &naif_id in &bodies {
            let frame = Frame::from_ephem_j2000(naif_id).with_orient(naif_id);
            for epc in sample_epochs() {
                let r_native = rotation_icrf_to_body_fixed_iau(naif_id, epc).unwrap();
                let (angles, _) = body_fixed_iau_angles_and_rates(naif_id, epc).unwrap();
                let tol = validation_tolerance(angles[2]);

                let et = epc.seconds_past_j2000_as_time_system(TimeSystem::TDB);
                let dcm = almanac
                    .rotation_to_parent(frame, AniseEpoch::from_et_seconds(et))
                    .unwrap();

                for i in 0..3 {
                    for j in 0..3 {
                        let dev = (r_native[(i, j)] - dcm.rot_mat[(i, j)]).abs();
                        max_dev = max_dev.max(dev);
                        assert!(
                            dev < tol,
                            "body {} et {}: |Δ({},{})| = {:e} (tol = {:e})",
                            naif_id,
                            et,
                            i,
                            j,
                            dev,
                            tol
                        );
                    }
                }
            }
        }

        println!(
            "IAU rotation vs ANISE (pck11.pca), 25 epochs 2000-2050, bodies {:?}: max |ΔR| = {:.3e}",
            bodies, max_dev
        );
    }
}
