/*!
 * Chebyshev-polynomial kernel segments: SPK Type 2 (position), SPK Type 3
 * (position and velocity), and binary PCK Type 2 (Euler angles).
 *
 * All three share one record layout: fixed-size records of Chebyshev
 * coefficients over uniformly spaced time intervals, followed by a 4-word
 * trailer `[INIT, INTLEN, RSIZE, N]`.
 *
 * # References
 * - NAIF SPK Required Reading, "Type 2" and "Type 3":
 *   <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html>
 * - NAIF PCK Required Reading:
 *   <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html>
 */

use nalgebra::Vector3;

use crate::utils::BraheError;

use super::daf::{DAFFile, DAFSummary};

/// One Chebyshev-interpolated kernel segment with coefficients resident in
/// memory. Covers SPK types 2/3 and binary PCK type 2.
#[derive(Debug)]
pub(crate) struct ChebyshevSegment {
    /// SPK: target NAIF body ID. PCK: body-frame class ID.
    pub target: i32,
    /// SPK: center NAIF body ID. PCK: reference frame ID.
    pub center: i32,
    /// Reference frame ID of the segment data (1 = J2000/ICRF).
    #[allow(dead_code)]
    pub frame: i32,
    /// SPK/PCK data type (2 or 3).
    pub data_type: i32,
    /// Components per record: 3 (type 2 / PCK) or 6 (type 3).
    #[allow(dead_code)]
    pub ncomp: usize,
    /// Segment coverage start, TDB seconds past J2000.
    pub start_et: f64,
    /// Segment coverage end, TDB seconds past J2000.
    pub end_et: f64,
    /// Start epoch of the first record. Units: [s] (TDB past J2000)
    pub init: f64,
    /// Length of each record's interval. Units: [s]
    pub intlen: f64,
    /// Words per record (2 + ncomp*(degree+1)).
    pub rsize: usize,
    /// Number of records.
    pub n: usize,
    /// Chebyshev polynomial degree.
    pub degree: usize,
    /// All record data, `n * rsize` words.
    pub coeffs: Vec<f64>,
}

/// Evaluate a Chebyshev series with the Clenshaw recurrence.
///
/// # Arguments
/// - `a`: Coefficients `a_0..a_n`
/// - `s`: Normalized argument in [-1, 1]
///
/// # Returns
/// - `p(s) = sum_{k=0}^n a_k T_k(s)`
fn chebyshev_value(a: &[f64], s: f64) -> f64 {
    let n = a.len() - 1;
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for k in (1..=n).rev() {
        let b0 = 2.0 * s * b1 - b2 + a[k];
        b2 = b1;
        b1 = b0;
    }
    s * b1 - b2 + a[0]
}

/// Build the derivative-series coefficients of a Chebyshev series `a`, in
/// the same (unhalved-`c_0`) convention accepted by [`chebyshev_value`].
///
/// # Arguments
/// - `a`: Coefficients `a_0..a_n` of the series being differentiated
///
/// # Returns
/// - Coefficients `c_0..c_{n-1}` of `p'(s)` (a single `[0.0]` when `n = 0`)
fn chebyshev_derivative_series(a: &[f64]) -> Vec<f64> {
    let n = a.len() - 1;
    if n == 0 {
        return vec![0.0];
    }
    // c_k coefficients of p'(s): c_{n-1} = 2n·a_n;
    // c_k = c_{k+2} + 2(k+1)·a_{k+1} for k = n-2..1; c_0 = c_2/2 + a_1.
    //
    // `c_0` always uses the dedicated formula above, never the general
    // recurrence: for n=1 the general "c_{n-1} = 2n·a_n" formula computes
    // the same slot (c_0) but in the halved-a_0 convention used internally
    // by the recurrence, which is off by a factor of 2 from the unhalved
    // convention `chebyshev_value` expects. The `saturating_sub` keeps the
    // loop range empty (a no-op) for n < 3 instead of underflowing.
    let mut c = vec![0.0; n];
    c[n - 1] = 2.0 * n as f64 * a[n];
    for k in (1..=n.saturating_sub(2)).rev() {
        c[k] = c.get(k + 2).copied().unwrap_or(0.0) + 2.0 * (k + 1) as f64 * a[k + 1];
    }
    c[0] = c.get(2).copied().unwrap_or(0.0) / 2.0 + a[1];
    c
}

/// Evaluate the derivative (d/ds) of a Chebyshev series at `s` by building
/// the derivative-series coefficients and applying Clenshaw.
///
/// # Arguments
/// - `a`: Coefficients `a_0..a_n` of the series being differentiated
/// - `s`: Normalized argument in [-1, 1]
///
/// # Returns
/// - `p'(s)`, the derivative with respect to `s` (not `et`)
fn chebyshev_derivative(a: &[f64], s: f64) -> f64 {
    chebyshev_value(&chebyshev_derivative_series(a), s)
}

/// Evaluate the second derivative (d²/ds²) of a Chebyshev series at `s` by
/// applying the derivative-series construction twice.
///
/// # Arguments
/// - `a`: Coefficients `a_0..a_n` of the series being differentiated
/// - `s`: Normalized argument in [-1, 1]
///
/// # Returns
/// - `p''(s)`, the second derivative with respect to `s` (not `et`)
fn chebyshev_second_derivative(a: &[f64], s: f64) -> f64 {
    chebyshev_value(
        &chebyshev_derivative_series(&chebyshev_derivative_series(a)),
        s,
    )
}

/// Validate that `v` is a finite, nonnegative, integer-valued count and
/// return it as `usize`. Rejects `NaN`, negative, and fractional values
/// that would otherwise silently truncate or wrap on cast.
///
/// # Arguments
/// - `v`: Value to validate
/// - `field`: Field name, used in the error message
/// - `kind`: Kernel kind (`"SPK"` or `"PCK"`), used in the error message
/// - `name`: Segment name, used in the error message
///
/// # Returns
/// - `v` as `usize`, or `BraheError::IoError` if `v` is not a valid count
fn validate_count(v: f64, field: &str, kind: &str, name: &str) -> Result<usize, BraheError> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
        return Err(BraheError::IoError(format!(
            "{} segment '{}' has invalid {} {}",
            kind, name, field, v
        )));
    }
    Ok(v as usize)
}

/// Distinctive fragment identifying an out-of-coverage error built by
/// [`coverage_error`]. Used by [`is_coverage_error`] to recognize such
/// errors; kept in one place so construction and detection cannot drift.
const COVERAGE_ERROR_MARKER: &str = "outside segment coverage";

/// Build the canonical out-of-coverage error for an epoch that no segment
/// (or record) covers. All coverage-error construction sites must use
/// this so [`is_coverage_error`] reliably identifies them.
///
/// # Arguments
/// - `et`: Queried epoch. Units: [s] (TDB past J2000)
/// - `start_et`: Coverage start of the (first candidate) segment. Units: [s]
/// - `end_et`: Coverage end of the (first candidate) segment. Units: [s]
/// - `target`: SPK target body ID, or PCK body-frame class ID
/// - `center`: SPK center body ID, or PCK reference frame ID
///
/// # Returns
/// - `BraheError::Error` naming the epoch, coverage interval, and body pair
pub(crate) fn coverage_error(
    et: f64,
    start_et: f64,
    end_et: f64,
    target: i32,
    center: i32,
) -> BraheError {
    BraheError::Error(format!(
        "Epoch ET {} {} [{}, {}] (target {}, center {})",
        et, COVERAGE_ERROR_MARKER, start_et, end_et, target, center
    ))
}

/// Build the out-of-coverage error for a chain link with multiple candidate
/// segments for the same body pair, none of which covers the queried
/// epoch. Reports the union of all candidates' coverage intervals and how
/// many were checked, since with multiple segments per pair the first
/// candidate alone can understate the union and give a misleadingly narrow
/// bound.
///
/// # Arguments
/// - `et`: Queried epoch. Units: [s] (TDB past J2000)
/// - `start_et`: Union coverage start (minimum over all candidates). Units: [s]
/// - `end_et`: Union coverage end (maximum over all candidates). Units: [s]
/// - `target`: SPK target body ID, or PCK body-frame class ID
/// - `center`: SPK center body ID, or PCK reference frame ID
/// - `count`: Number of candidate segments checked
///
/// # Returns
/// - `BraheError::Error` naming the epoch, union coverage interval, body
///   pair, and candidate count
pub(crate) fn coverage_error_multi(
    et: f64,
    start_et: f64,
    end_et: f64,
    target: i32,
    center: i32,
    count: usize,
) -> BraheError {
    BraheError::Error(format!(
        "Epoch ET {} {} [{}, {}] across {} candidate segments (target {}, center {})",
        et, COVERAGE_ERROR_MARKER, start_et, end_et, count, target, center
    ))
}

/// Distinctive fragment identifying a PCK frame-lookup miss: a frame class
/// ID absent from a kernel's segments, or present but lacking coverage at
/// the queried epoch. Built by [`pck_frame_not_found_error`] and
/// [`pck_out_of_coverage_error`] (used by [`super::pck::BPCK`]); recognized
/// by [`is_coverage_error`] alongside [`COVERAGE_ERROR_MARKER`] so callers
/// searching multiple loaded PCK kernels (`registry::pck_query`) can tell
/// "try the next kernel" misses apart from genuine data errors (e.g. a
/// corrupt record) that must propagate immediately instead of being masked.
const PCK_MISS_MARKER: &str = "not available from this PCK kernel";

/// Build the error for a PCK frame class ID absent from a kernel's
/// segments.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID that was not found
///
/// # Returns
/// - `BraheError::Error` naming the frame ID, recognized by
///   [`is_coverage_error`]
pub(crate) fn pck_frame_not_found_error(frame_id: i32) -> BraheError {
    BraheError::Error(format!(
        "Frame class ID {} {} (frame not present in this kernel)",
        frame_id, PCK_MISS_MARKER
    ))
}

/// Build the error for a PCK frame class ID present in a kernel but
/// lacking segment coverage at the queried epoch.
///
/// # Arguments
/// - `et`: Queried epoch. Units: [s] (TDB past J2000)
/// - `frame_id`: Body-frame class ID that lacks coverage at `et`
///
/// # Returns
/// - `BraheError::Error` naming the epoch and frame ID, recognized by
///   [`is_coverage_error`]
pub(crate) fn pck_out_of_coverage_error(et: f64, frame_id: i32) -> BraheError {
    BraheError::Error(format!(
        "Epoch ET {} {} for frame class ID {} (out of segment coverage)",
        et, PCK_MISS_MARKER, frame_id
    ))
}

/// True if `err` is an out-of-coverage error built by [`coverage_error`],
/// [`coverage_error_multi`], [`pck_frame_not_found_error`], or
/// [`pck_out_of_coverage_error`].
///
/// Used to gate the epoch-aware chain fallback
/// (`spk::evaluate_with_epoch_fallback`) and the multi-kernel PCK search
/// (`registry::pck_query`): only a coverage/frame-lookup miss justifies
/// moving on (re-resolving the chain, or trying the next kernel); any other
/// evaluation error (e.g. corrupt record data) must propagate unchanged
/// rather than being masked.
///
/// # Arguments
/// - `err`: Error to classify
///
/// # Returns
/// - `true` if `err` was produced by one of the coverage-error
///   constructors above
pub(crate) fn is_coverage_error(err: &BraheError) -> bool {
    matches!(err, BraheError::Error(msg) if msg.contains(COVERAGE_ERROR_MARKER) || msg.contains(PCK_MISS_MARKER))
}

impl ChebyshevSegment {
    /// Build a segment from an SPK summary
    /// (ints = `[target, center, frame, type, start_addr, end_addr]`).
    ///
    /// # Arguments
    /// - `daf`: Parsed DAF container holding the segment's word data
    /// - `summary`: SPK segment summary (6 ints, at least 2 doubles)
    ///
    /// # Returns
    /// - Parsed `ChebyshevSegment`, or `BraheError` on malformed data or an
    ///   unsupported segment type
    pub fn from_spk_summary(daf: &DAFFile, summary: &DAFSummary) -> Result<Self, BraheError> {
        if summary.ints.len() < 6 || summary.doubles.len() < 2 {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' summary has {} ints and {} doubles, expected 6 ints and at least 2 doubles",
                summary.name,
                summary.ints.len(),
                summary.doubles.len()
            )));
        }
        Self::from_summary(
            daf,
            summary,
            summary.ints[0],
            summary.ints[1],
            summary.ints[2],
            summary.ints[3],
            summary.ints[4],
            summary.ints[5],
            "SPK",
        )
    }

    /// Build a segment from a binary PCK summary
    /// (ints = `[frame_class_id, reference_frame, type, start_addr, end_addr]`).
    ///
    /// # Arguments
    /// - `daf`: Parsed DAF container holding the segment's word data
    /// - `summary`: PCK segment summary (5 ints, at least 2 doubles)
    ///
    /// # Returns
    /// - Parsed `ChebyshevSegment`, or `BraheError` on malformed data or an
    ///   unsupported segment type
    pub fn from_pck_summary(daf: &DAFFile, summary: &DAFSummary) -> Result<Self, BraheError> {
        if summary.ints.len() < 5 || summary.doubles.len() < 2 {
            return Err(BraheError::IoError(format!(
                "PCK segment '{}' summary has {} ints and {} doubles, expected 5 ints and at least 2 doubles",
                summary.name,
                summary.ints.len(),
                summary.doubles.len()
            )));
        }
        Self::from_summary(
            daf,
            summary,
            summary.ints[0],
            summary.ints[1],
            summary.ints[1],
            summary.ints[2],
            summary.ints[3],
            summary.ints[4],
            "PCK",
        )
    }

    /// Shared summary parsing for both SPK and PCK: reads the segment's
    /// word range, validates the trailer, and extracts the coefficient data.
    ///
    /// # Arguments
    /// - `daf`: Parsed DAF container holding the segment's word data
    /// - `summary`: Segment summary, used for its name and coverage doubles
    /// - `target`: SPK target body ID, or PCK body-frame class ID
    /// - `center`: SPK center body ID, or PCK reference frame ID
    /// - `frame`: Reference frame ID of the segment data
    /// - `data_type`: SPK/PCK data type (2 or 3)
    /// - `start_addr`: First word address of the segment (1-based)
    /// - `end_addr`: Last word address of the segment (1-based, inclusive)
    /// - `kind`: Kernel kind (`"SPK"` or `"PCK"`), used in error messages
    ///
    /// # Returns
    /// - Parsed `ChebyshevSegment`, or `BraheError` on malformed data, an
    ///   unsupported segment type, or a record directory that does not
    ///   cover the descriptor's coverage interval
    #[allow(clippy::too_many_arguments)]
    fn from_summary(
        daf: &DAFFile,
        summary: &DAFSummary,
        target: i32,
        center: i32,
        frame: i32,
        data_type: i32,
        start_addr: i32,
        end_addr: i32,
        kind: &str,
    ) -> Result<Self, BraheError> {
        let (ncomp, supported) = match data_type {
            2 => (3, "2"),
            3 if kind == "SPK" => (6, "2, 3"),
            _ => (0, if kind == "SPK" { "2, 3" } else { "2" }),
        };
        if ncomp == 0 {
            return Err(BraheError::Error(format!(
                "{} segment type {} not supported (segment '{}'); supported types: {}",
                kind, data_type, summary.name, supported
            )));
        }

        // The reference frame of the segment data is stored (`self.frame`)
        // but everywhere else in the crate (frame biases, GCRF-compatible
        // output, etc.) assumes ICRF axes. A segment in another frame (e.g.
        // ECLIPJ2000 = 17, seen in some Horizons/generic small-body SPKs)
        // would otherwise parse successfully and be summed into chains as
        // if it were J2000/ICRF, silently producing wrong answers.
        if frame != 1 {
            return Err(BraheError::Error(format!(
                "{} segment '{}' has reference frame {}; only frame 1 (J2000/ICRF) supported",
                kind, summary.name, frame
            )));
        }

        if start_addr < 1 || end_addr < start_addr {
            return Err(BraheError::IoError(format!(
                "{} segment '{}' has invalid address range [{}, {}]",
                kind, summary.name, start_addr, end_addr
            )));
        }
        let words = daf.words(start_addr as usize, end_addr as usize)?;
        if words.len() < 4 {
            return Err(BraheError::IoError(format!(
                "{} segment '{}' too short ({} words)",
                kind,
                summary.name,
                words.len()
            )));
        }
        let trailer = &words[words.len() - 4..];
        let (init, intlen) = (trailer[0], trailer[1]);
        let rsize = validate_count(trailer[2], "RSIZE", kind, &summary.name)?;
        let n = validate_count(trailer[3], "N", kind, &summary.name)?;

        let word_count = n.checked_mul(rsize);
        if !init.is_finite()
            || !intlen.is_finite()
            || intlen <= 0.0
            || rsize < 2 + ncomp
            || !(rsize - 2).is_multiple_of(ncomp)
            || n == 0
            || word_count
                .and_then(|wc| wc.checked_add(4))
                .is_none_or(|total| words.len() < total)
        {
            return Err(BraheError::IoError(format!(
                "{} segment '{}' has inconsistent directory (INIT={}, INTLEN={}, RSIZE={}, N={}, words={})",
                kind,
                summary.name,
                init,
                intlen,
                rsize,
                n,
                words.len()
            )));
        }
        let degree = (rsize - 2) / ncomp - 1;

        // The record directory [INIT, INIT + N*INTLEN] must cover the
        // descriptor's coverage interval [start_et, end_et]; otherwise
        // `record()`'s index clamp would silently evaluate the wrong (or
        // an out-of-range) record for epochs inside the descriptor but
        // outside the directory, rather than reporting an error.
        let (start_et, end_et) = (summary.doubles[0], summary.doubles[1]);
        let dir_end = init + (n as f64) * intlen;
        if init > start_et || dir_end < end_et {
            return Err(BraheError::IoError(format!(
                "{} segment '{}' record directory [{}, {}] does not cover descriptor interval [{}, {}]",
                kind, summary.name, init, dir_end, start_et, end_et
            )));
        }

        Ok(ChebyshevSegment {
            target,
            center,
            frame,
            data_type,
            ncomp,
            start_et,
            end_et,
            init,
            intlen,
            rsize,
            n,
            degree,
            coeffs: words[..word_count.unwrap()].to_vec(),
        })
    }

    /// True if `et` lies within this segment's descriptor coverage.
    ///
    /// # Arguments
    /// - `et`: Epoch to test. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - `true` if `et` is within `[start_et, end_et]`, inclusive
    pub fn covers(&self, et: f64) -> bool {
        et >= self.start_et && et <= self.end_et
    }

    /// Locate the record covering `et` and return `(record_words, s)` where
    /// `s` is the normalized Chebyshev argument.
    ///
    /// # Arguments
    /// - `et`: Epoch to look up. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - The record's words and the normalized argument `s`, or
    ///   `BraheError` if `et` is out of coverage or the record's `RADIUS`
    ///   is not a positive, finite number
    fn record(&self, et: f64) -> Result<(&[f64], f64), BraheError> {
        if !self.covers(et) {
            return Err(coverage_error(
                et,
                self.start_et,
                self.end_et,
                self.target,
                self.center,
            ));
        }
        let idx = (((et - self.init) / self.intlen).floor() as usize).min(self.n - 1);
        let rec = &self.coeffs[idx * self.rsize..(idx + 1) * self.rsize];
        let (mid, radius) = (rec[0], rec[1]);
        if !radius.is_finite() || radius <= 0.0 {
            return Err(BraheError::IoError(format!(
                "Segment record {} has invalid RADIUS {} (target {}, center {})",
                idx, radius, self.target, self.center
            )));
        }
        Ok((rec, (et - mid) / radius))
    }

    /// Return the Chebyshev coefficients for component `c` (0-based) of
    /// `rec`, a single record's words.
    ///
    /// # Arguments
    /// - `rec`: A single record's words (`MID, RADIUS`, then `ncomp`
    ///   groups of `degree + 1` coefficients)
    /// - `degree`: Chebyshev polynomial degree
    /// - `c`: 0-based component index
    ///
    /// # Returns
    /// - The `degree + 1` coefficients `a_0..a_degree` for component `c`
    fn component(rec: &[f64], degree: usize, c: usize) -> &[f64] {
        &rec[2 + c * (degree + 1)..2 + (c + 1) * (degree + 1)]
    }

    /// Evaluate three consecutive components of `rec` at `s`.
    ///
    /// # Arguments
    /// - `rec`: A single record's words
    /// - `degree`: Chebyshev polynomial degree
    /// - `start`: 0-based index of the first of three consecutive components
    /// - `s`: Normalized Chebyshev argument in [-1, 1]
    ///
    /// # Returns
    /// - The three evaluated components
    fn eval_triple(rec: &[f64], degree: usize, start: usize, s: f64) -> Vector3<f64> {
        Vector3::new(
            chebyshev_value(Self::component(rec, degree, start), s),
            chebyshev_value(Self::component(rec, degree, start + 1), s),
            chebyshev_value(Self::component(rec, degree, start + 2), s),
        )
    }

    /// Evaluate the derivative (d/det) of three consecutive components of
    /// `rec` at `s`.
    ///
    /// # Arguments
    /// - `rec`: A single record's words
    /// - `degree`: Chebyshev polynomial degree
    /// - `start`: 0-based index of the first of three consecutive components
    /// - `s`: Normalized Chebyshev argument in [-1, 1]
    /// - `radius`: Record `RADIUS`, used to scale d/ds to d/det. Units: [s]
    ///
    /// # Returns
    /// - The three evaluated derivatives
    fn eval_triple_derivative(
        rec: &[f64],
        degree: usize,
        start: usize,
        s: f64,
        radius: f64,
    ) -> Vector3<f64> {
        Vector3::new(
            chebyshev_derivative(Self::component(rec, degree, start), s) / radius,
            chebyshev_derivative(Self::component(rec, degree, start + 1), s) / radius,
            chebyshev_derivative(Self::component(rec, degree, start + 2), s) / radius,
        )
    }

    /// Evaluate the second time derivative (d²/det²) of three consecutive
    /// components of `rec` at `s`.
    ///
    /// # Arguments
    /// - `rec`: A single record's words
    /// - `degree`: Chebyshev polynomial degree
    /// - `start`: 0-based index of the first of three consecutive components
    /// - `s`: Normalized Chebyshev argument in [-1, 1]
    /// - `radius`: Record `RADIUS`, used to scale d²/ds² to d²/det². Units: [s]
    ///
    /// # Returns
    /// - The three evaluated second derivatives
    fn eval_triple_second_derivative(
        rec: &[f64],
        degree: usize,
        start: usize,
        s: f64,
        radius: f64,
    ) -> Vector3<f64> {
        let r2 = radius * radius;
        Vector3::new(
            chebyshev_second_derivative(Self::component(rec, degree, start), s) / r2,
            chebyshev_second_derivative(Self::component(rec, degree, start + 1), s) / r2,
            chebyshev_second_derivative(Self::component(rec, degree, start + 2), s) / r2,
        )
    }

    /// Evaluate the first three components at `et`.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - Position in kernel-natural units (km for SPK, rad for PCK), or
    ///   `BraheError` if `et` is out of coverage
    pub fn position(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        let (rec, s) = self.record(et)?;
        Ok(Self::eval_triple(rec, self.degree, 0, s))
    }

    /// Evaluate the time derivative of the first three components at `et`.
    ///
    /// Type 2 differentiates the position polynomials analytically; type 3
    /// evaluates the stored velocity polynomials.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - Velocity in kernel-natural units (km/s for SPK, rad/s for PCK), or
    ///   `BraheError` if `et` is out of coverage
    pub fn velocity(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        let (rec, s) = self.record(et)?;
        let radius = rec[1];
        Ok(match self.data_type {
            3 => Self::eval_triple(rec, self.degree, 3, s),
            _ => Self::eval_triple_derivative(rec, self.degree, 0, s, radius),
        })
    }

    /// Evaluate position and velocity with a single record lookup.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - `(position, velocity)` in kernel-natural units (km, km/s for SPK;
    ///   rad, rad/s for PCK), or `BraheError` if `et` is out of coverage
    pub fn state(&self, et: f64) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
        let (rec, s) = self.record(et)?;
        let radius = rec[1];
        let r = Self::eval_triple(rec, self.degree, 0, s);
        let v = match self.data_type {
            3 => Self::eval_triple(rec, self.degree, 3, s),
            _ => Self::eval_triple_derivative(rec, self.degree, 0, s, radius),
        };
        Ok((r, v))
    }

    /// Evaluate the second time derivative of the first three components at
    /// `et`.
    ///
    /// Type 2 differentiates the position polynomials twice analytically;
    /// type 3 differentiates the stored velocity polynomials once.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - Acceleration in kernel-natural units (km/s² for SPK, rad/s² for
    ///   PCK), or `BraheError` if `et` is out of coverage
    pub fn acceleration(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        let (rec, s) = self.record(et)?;
        let radius = rec[1];
        Ok(match self.data_type {
            3 => Self::eval_triple_derivative(rec, self.degree, 3, s, radius),
            _ => Self::eval_triple_second_derivative(rec, self.degree, 0, s, radius),
        })
    }
}

/// Maximum difference-table dimension (terms per Cartesian component) a
/// type-21 record may declare. Matches CSPICE's `MAXTRM` from `spk21.inc`;
/// a larger value trips `SPICE(DIFFLINETOOLARGE)` there. Guards against a
/// corrupt `MAXDIM` forcing an absurd working-array allocation.
const TYPE21_MAXTRM: usize = 25;

/// One SPK type-21 (Extended Modified Difference Array) segment with all
/// records resident in memory.
///
/// Type 21 stores, per record, a reference epoch, a reference
/// position/velocity, and modified divided-difference arrays (MDAs) whose
/// per-component term count `MAXDIM` is variable (unlike the fixed-degree
/// Chebyshev types). Horizons emits type 21 for small bodies (asteroids,
/// comets). Evaluation is a direct port of CSPICE `spke21.f`/`spke21.c`
/// (via the `spktype21` Python package, itself a `spke21.f` transcription);
/// see [`Type21Segment::spke21`].
#[derive(Debug)]
pub(crate) struct Type21Segment {
    /// Target NAIF body ID.
    pub target: i32,
    /// Center NAIF body ID (Horizons small-body SPKs are Sun-centered, 10).
    pub center: i32,
    /// Reference frame ID of the segment data (1 = J2000/ICRF).
    #[allow(dead_code)]
    pub frame: i32,
    /// Segment coverage start, TDB seconds past J2000.
    pub start_et: f64,
    /// Segment coverage end, TDB seconds past J2000.
    pub end_et: f64,
    /// Difference-table dimension `MAXDIM`: terms per Cartesian component.
    pub maxdim: usize,
    /// Words per record, `4 * MAXDIM + 11`.
    pub dlsize: usize,
    /// Number of records.
    pub n: usize,
    /// All record data, `n * dlsize` words.
    pub records: Vec<f64>,
    /// Per-record upper-boundary epochs (the epoch directory), `n` words,
    /// ascending. Units: [s] (TDB past J2000)
    pub epochs: Vec<f64>,
}

impl Type21Segment {
    /// Build a type-21 segment from an SPK summary
    /// (ints = `[target, center, frame, type, start_addr, end_addr]`).
    ///
    /// # Arguments
    /// - `daf`: Parsed DAF container holding the segment's word data
    /// - `summary`: SPK segment summary (6 ints, at least 2 doubles)
    ///
    /// # Returns
    /// - Parsed `Type21Segment`, or `BraheError` on malformed data, a
    ///   non-J2000 frame, or an inconsistent record directory
    pub fn from_spk_summary(daf: &DAFFile, summary: &DAFSummary) -> Result<Self, BraheError> {
        if summary.ints.len() < 6 || summary.doubles.len() < 2 {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' summary has {} ints and {} doubles, expected 6 ints and at least 2 doubles",
                summary.name,
                summary.ints.len(),
                summary.doubles.len()
            )));
        }
        let (target, center, frame) = (summary.ints[0], summary.ints[1], summary.ints[2]);
        let (start_addr, end_addr) = (summary.ints[4], summary.ints[5]);
        let (start_et, end_et) = (summary.doubles[0], summary.doubles[1]);

        // Same frame constraint as the Chebyshev path: the rest of the crate
        // assumes ICRF axes, so a segment in another frame (e.g. ECLIPJ2000
        // = 17) must be rejected rather than silently summed into chains.
        if frame != 1 {
            return Err(BraheError::Error(format!(
                "SPK segment '{}' has reference frame {}; only frame 1 (J2000/ICRF) supported",
                summary.name, frame
            )));
        }
        if start_addr < 1 || end_addr < start_addr {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' has invalid address range [{}, {}]",
                summary.name, start_addr, end_addr
            )));
        }
        let words = daf.words(start_addr as usize, end_addr as usize)?;

        // The segment's final two words are, in order, MAXDIM (the
        // difference-table dimension) and N (the record count). CSPICE's
        // "SPK Required Reading" nominally documents DLSIZE in the
        // penultimate slot, but Horizons kernels store MAXDIM there;
        // `spktype21` documents the same discrepancy.
        if words.len() < 2 {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' too short ({} words)",
                summary.name,
                words.len()
            )));
        }
        let maxdim = validate_count(words[words.len() - 2], "MAXDIM", "SPK", &summary.name)?;
        let n = validate_count(words[words.len() - 1], "N", "SPK", &summary.name)?;
        if maxdim == 0 || maxdim > TYPE21_MAXTRM {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' has type-21 MAXDIM {} outside [1, {}]",
                summary.name, maxdim, TYPE21_MAXTRM
            )));
        }
        let dlsize = 4 * maxdim + 11;

        // The segment layout is: N records of DLSIZE words, then N boundary
        // epochs (the epoch directory), then N/100 coarse directory entries,
        // then [MAXDIM, N]. Validate the record + epoch region fits with
        // checked arithmetic so a corrupt N cannot overflow or index past
        // the segment.
        let records_len = n.checked_mul(dlsize);
        let epoch_end = records_len.and_then(|rl| rl.checked_add(n));
        if n == 0
            || epoch_end
                .and_then(|ee| ee.checked_add(n / 100 + 2))
                .is_none_or(|total| words.len() < total)
        {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' has inconsistent type-21 directory (MAXDIM={}, N={}, words={})",
                summary.name,
                maxdim,
                n,
                words.len()
            )));
        }
        let records_len = records_len.unwrap();
        let epoch_end = epoch_end.unwrap();
        let records = words[..records_len].to_vec();
        let epochs = words[records_len..epoch_end].to_vec();

        // The epoch directory must be finite and ascending and cover the
        // descriptor's coverage interval; otherwise record selection (a
        // binary search) could silently pick the wrong record for epochs
        // inside the descriptor. A non-finite entry is called out explicitly
        // because NaN comparisons are false and would slip past the ordering
        // check.
        if epochs.iter().any(|e| !e.is_finite()) || epochs.windows(2).any(|w| w[1] < w[0]) {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' has a non-finite or non-ascending type-21 epoch directory",
                summary.name
            )));
        }
        if start_et > epochs[0] || epochs[n - 1] < end_et {
            return Err(BraheError::IoError(format!(
                "SPK segment '{}' type-21 epoch directory [{}, {}] does not cover descriptor interval [{}, {}]",
                summary.name,
                epochs[0],
                epochs[n - 1],
                start_et,
                end_et
            )));
        }

        Ok(Type21Segment {
            target,
            center,
            frame,
            start_et,
            end_et,
            maxdim,
            dlsize,
            n,
            records,
            epochs,
        })
    }

    /// True if `et` lies within this segment's descriptor coverage.
    ///
    /// # Arguments
    /// - `et`: Epoch to test. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - `true` if `et` is within `[start_et, end_et]`, inclusive
    pub fn covers(&self, et: f64) -> bool {
        et >= self.start_et && et <= self.end_et
    }

    /// Locate the record covering `et` and return its `dlsize` words.
    ///
    /// Records partition coverage by their upper-boundary epochs: record `i`
    /// covers `(epochs[i-1], epochs[i]]`. The first record whose boundary
    /// exceeds `et` is selected (binary search over the ascending directory).
    ///
    /// # Arguments
    /// - `et`: Epoch to look up. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - The selected record's words, or `BraheError` if `et` is out of
    ///   coverage
    fn record(&self, et: f64) -> Result<&[f64], BraheError> {
        if !self.covers(et) {
            return Err(coverage_error(
                et,
                self.start_et,
                self.end_et,
                self.target,
                self.center,
            ));
        }
        let idx = self.epochs.partition_point(|&e| e <= et).min(self.n - 1);
        Ok(&self.records[idx * self.dlsize..(idx + 1) * self.dlsize])
    }

    /// Evaluate a single type-21 record at `et`, returning position and
    /// velocity in kernel-natural units (km, km/s).
    ///
    /// Direct port of CSPICE `spke21.f`/`spke21.c` (the Extended Modified
    /// Difference Array evaluator, a generalization of `spke01` to variable
    /// difference-line size), transcribed via the `spktype21` Python
    /// package. The record layout is: reference epoch `TL` (1 word), the
    /// stepsize function vector `G` (`MAXDIM`), reference position/velocity
    /// interleaved as x,ẋ,y,ẏ,z,ż (6), the modified divided-difference
    /// arrays `DT` (`MAXDIM × 3`, component-major), `KQMAX1` (1), and the
    /// per-component integration-order array `KQ` (3).
    ///
    /// # Arguments
    /// - `record`: One record's `dlsize` words
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - `(position, velocity)` in km, km/s, or `BraheError` if the record's
    ///   metadata is malformed (bad `KQMAX1`/`KQ`, or a zero stepsize)
    fn spke21(&self, record: &[f64], et: f64) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
        let maxdim = self.maxdim;
        let tl = record[0];
        let g = &record[1..maxdim + 1];
        let refpos = Vector3::new(record[maxdim + 1], record[maxdim + 3], record[maxdim + 5]);
        let refvel = Vector3::new(record[maxdim + 2], record[maxdim + 4], record[maxdim + 6]);

        // DT component `c` occupies a contiguous MAXDIM-word block; term `j`
        // of component `c` is at `dt_base + c*maxdim + j`.
        let dt_base = maxdim + 7;
        let dt = |c: usize, j: usize| record[dt_base + c * maxdim + j];

        let kqmax1 = validate_count(record[4 * maxdim + 7], "KQMAX1", "SPK", "type-21 record")?;
        let kq = [
            validate_count(record[4 * maxdim + 8], "KQ", "SPK", "type-21 record")?,
            validate_count(record[4 * maxdim + 9], "KQ", "SPK", "type-21 record")?,
            validate_count(record[4 * maxdim + 10], "KQ", "SPK", "type-21 record")?,
        ];
        // Bound the orders before they drive indexing and allocation. The
        // evaluator assumes the maximum integration order KQMAX1-1 is at
        // least 2 (KQMAX1 >= 3); the stepsize loop reads G up to index
        // KQMAX1-3, so KQMAX1 must not exceed MAXDIM+1; and each component
        // order KQ[c] must be a genuine integration order (< KQMAX1), which
        // also bounds its MDA term access below MAXDIM. A record violating
        // any of these is corrupt, and without these checks a malformed
        // record could index out of bounds or silently sum unrefined terms.
        if kqmax1 < 3 || kqmax1 > maxdim + 1 || kq.iter().any(|&k| k >= kqmax1) {
            return Err(BraheError::IoError(format!(
                "SPK type-21 record has invalid orders KQMAX1={}, KQ={:?} (MAXDIM={})",
                kqmax1, kq, maxdim
            )));
        }

        // Working coefficient arrays. Sized generously (all indices used are
        // provably below this bound) to avoid per-index bookkeeping.
        let wsize = maxdim + kqmax1 + 4;
        let mut fc = vec![0.0f64; wsize];
        fc[0] = 1.0;
        let mut wc = vec![0.0f64; wsize];
        let mut w = vec![0.0f64; wsize];

        let delta = et - tl;
        let mut tp = delta;
        let mq2 = kqmax1 - 2;
        for j in 1..=mq2 {
            if g[j - 1] == 0.0 {
                return Err(BraheError::IoError(format!(
                    "SPK type-21 record has a zero stepsize at G[{}]",
                    j - 1
                )));
            }
            fc[j] = tp / g[j - 1];
            wc[j - 1] = delta / g[j - 1];
            tp = delta + g[j - 1];
        }

        // Collect KQMAX1 reciprocals into W.
        for j in 1..=kqmax1 {
            w[j - 1] = 1.0 / j as f64;
        }

        // Refine the W coefficients for the position interpolation. KS
        // (``maximum integration'') starts at KQMAX1-1 and steps down to 1;
        // the invariant KS1 == KS-1 holds throughout.
        let mut jx = 0usize;
        let mut ks = kqmax1 - 1;
        let mut ks1 = ks - 1;
        while ks >= 2 {
            jx += 1;
            for j in 1..=jx {
                w[j + ks - 1] = fc[j] * w[j + ks1 - 1] - wc[j - 1] * w[j + ks - 1];
            }
            ks = ks1;
            ks1 -= 1;
        }

        // Position interpolation (KS == 1 here).
        let mut pos = Vector3::zeros();
        for c in 0..3 {
            let mut sum = 0.0;
            for j in (1..=kq[c]).rev() {
                sum += dt(c, j - 1) * w[j + ks - 1];
            }
            pos[c] = refpos[c] + delta * (refvel[c] + delta * sum);
        }

        // Refine the W coefficients once more for the velocity
        // interpolation (KS == 1, KS1 == 0 here), then drop KS to 0.
        for j in 1..=jx {
            w[j + ks - 1] = fc[j] * w[j + ks1 - 1] - wc[j - 1] * w[j + ks - 1];
        }
        ks -= 1;

        // Velocity interpolation (KS == 0 here).
        let mut vel = Vector3::zeros();
        for c in 0..3 {
            let mut sum = 0.0;
            for j in (1..=kq[c]).rev() {
                sum += dt(c, j - 1) * w[j + ks - 1];
            }
            vel[c] = refvel[c] + delta * sum;
        }

        Ok((pos, vel))
    }

    /// Evaluate position at `et`. Units: km.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - Position in km, or `BraheError` if `et` is out of coverage
    pub fn position(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        let rec = self.record(et)?;
        Ok(self.spke21(rec, et)?.0)
    }

    /// Evaluate velocity at `et`. Units: km/s.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - Velocity in km/s, or `BraheError` if `et` is out of coverage
    pub fn velocity(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        let rec = self.record(et)?;
        Ok(self.spke21(rec, et)?.1)
    }

    /// Evaluate position and velocity with a single record lookup. Units:
    /// km, km/s.
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - `(position, velocity)` in km, km/s, or `BraheError` if `et` is out
    ///   of coverage
    pub fn state(&self, et: f64) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
        let rec = self.record(et)?;
        self.spke21(rec, et)
    }

    /// Evaluate acceleration at `et`. Units: km/s².
    ///
    /// Type-21 records store position and velocity only, so acceleration is
    /// a central finite difference of the analytically evaluated velocity,
    /// taken within the single covering record (so the estimate stays inside
    /// the record's valid polynomial domain even at coverage boundaries).
    ///
    /// # Arguments
    /// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
    ///
    /// # Returns
    /// - Acceleration in km/s², or `BraheError` if `et` is out of coverage
    pub fn acceleration(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        let rec = self.record(et)?;
        let h = 1.0;
        let (_, vp) = self.spke21(rec, et + h)?;
        let (_, vm) = self.spke21(rec, et - h)?;
        Ok((vp - vm) / (2.0 * h))
    }
}

/// An SPK segment of any supported data type. A resolved chain may mix
/// segment types (e.g. a Type 21 small-body segment alongside Type 2/3
/// planetary segments), so segments are stored behind this enum rather
/// than the concrete Chebyshev type.
#[derive(Debug)]
pub(crate) enum SpkSegment {
    Chebyshev(ChebyshevSegment),
    Type21(Type21Segment),
}

impl SpkSegment {
    /// Build an SPK segment from a DAF summary, dispatching on data type
    /// (ints\[3\]): type 21 to [`Type21Segment`], all others to
    /// [`ChebyshevSegment`] (which supports types 2/3 and raises the
    /// unsupported-type error otherwise).
    ///
    /// # Arguments
    /// - `daf`: Parsed DAF container holding the segment's word data
    /// - `summary`: SPK segment summary (6 ints, at least 2 doubles)
    ///
    /// # Returns
    /// - Parsed `SpkSegment`, or `BraheError` on malformed data or an
    ///   unsupported segment type
    pub(crate) fn from_spk_summary(
        daf: &DAFFile,
        summary: &DAFSummary,
    ) -> Result<Self, BraheError> {
        if summary.ints.len() >= 6 && summary.ints[3] == 21 {
            Ok(SpkSegment::Type21(Type21Segment::from_spk_summary(
                daf, summary,
            )?))
        } else {
            Ok(SpkSegment::Chebyshev(ChebyshevSegment::from_spk_summary(
                daf, summary,
            )?))
        }
    }

    /// SPK target NAIF body ID.
    pub(crate) fn target(&self) -> i32 {
        match self {
            SpkSegment::Chebyshev(s) => s.target,
            SpkSegment::Type21(s) => s.target,
        }
    }

    /// SPK center NAIF body ID.
    pub(crate) fn center(&self) -> i32 {
        match self {
            SpkSegment::Chebyshev(s) => s.center,
            SpkSegment::Type21(s) => s.center,
        }
    }

    /// Segment coverage start, TDB seconds past J2000.
    pub(crate) fn start_et(&self) -> f64 {
        match self {
            SpkSegment::Chebyshev(s) => s.start_et,
            SpkSegment::Type21(s) => s.start_et,
        }
    }

    /// Segment coverage end, TDB seconds past J2000.
    pub(crate) fn end_et(&self) -> f64 {
        match self {
            SpkSegment::Chebyshev(s) => s.end_et,
            SpkSegment::Type21(s) => s.end_et,
        }
    }

    /// True if `et` lies within this segment's descriptor coverage.
    pub(crate) fn covers(&self, et: f64) -> bool {
        match self {
            SpkSegment::Chebyshev(s) => s.covers(et),
            SpkSegment::Type21(s) => s.covers(et),
        }
    }

    /// Evaluate position at `et`. Units: kernel-natural (km).
    pub(crate) fn position(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        match self {
            SpkSegment::Chebyshev(s) => s.position(et),
            SpkSegment::Type21(s) => s.position(et),
        }
    }

    /// Evaluate velocity at `et`. Units: kernel-natural (km/s).
    pub(crate) fn velocity(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        match self {
            SpkSegment::Chebyshev(s) => s.velocity(et),
            SpkSegment::Type21(s) => s.velocity(et),
        }
    }

    /// Evaluate position and velocity with a single record lookup. Units:
    /// kernel-natural (km, km/s).
    pub(crate) fn state(&self, et: f64) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
        match self {
            SpkSegment::Chebyshev(s) => s.state(et),
            SpkSegment::Type21(s) => s.state(et),
        }
    }

    /// Evaluate acceleration at `et`. Units: kernel-natural (km/s²).
    pub(crate) fn acceleration(&self, et: f64) -> Result<Vector3<f64>, BraheError> {
        match self {
            SpkSegment::Chebyshev(s) => s.acceleration(et),
            SpkSegment::Type21(s) => s.acceleration(et),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;

    /// Build a synthetic 2-record type-2 segment where each component is an
    /// exact polynomial in s: x = T0 + 2T1 + 3T2, y = 4T1, z = 5T0 + 6T3.
    /// Record 0 covers et in [0, 100], record 1 covers [100, 200].
    fn synthetic_segment() -> ChebyshevSegment {
        let degree = 3usize;
        let rsize = 2 + 3 * (degree + 1);
        let mut coeffs = Vec::new();
        for rec in 0..2 {
            let mid = 50.0 + 100.0 * rec as f64;
            coeffs.extend_from_slice(&[mid, 50.0]); // MID, RADIUS
            coeffs.extend_from_slice(&[1.0, 2.0, 3.0, 0.0]); // x
            coeffs.extend_from_slice(&[0.0, 4.0, 0.0, 0.0]); // y
            coeffs.extend_from_slice(&[5.0, 0.0, 0.0, 6.0]); // z
        }
        ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et: 0.0,
            end_et: 200.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 2,
            degree,
            coeffs,
        }
    }

    #[test]
    fn test_chebyshev_position_exact() {
        let seg = synthetic_segment();
        // et=75 -> record 0, s = 0.5
        // T0=1, T1=0.5, T2=2*0.25-1=-0.5, T3=4*0.125-3*0.5=-1.0
        let p = seg.position(75.0).unwrap();
        assert_abs_diff_eq!(p[0], 1.0 + 2.0 * 0.5 + 3.0 * (-0.5), epsilon = 1e-12);
        assert_abs_diff_eq!(p[1], 4.0 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(p[2], 5.0 - 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chebyshev_velocity_exact() {
        let seg = synthetic_segment();
        // d/ds: x' = 2·T0' etc. T1'=1, T2'=4s, T3'=12s²-3. At s=0.5:
        // x' = 2*1 + 3*(4*0.5) = 8; y' = 4; z' = 6*(12*0.25-3) = 0
        // d/det = d/ds / RADIUS = /50
        let v = seg.velocity(75.0).unwrap();
        assert_abs_diff_eq!(v[0], 8.0 / 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[1], 4.0 / 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chebyshev_velocity_degree_one() {
        // Regression test: degree-1 records (n=1 in the derivative
        // recurrence) occur in real SPK data (e.g. a Mercury/Mercury
        // barycenter segment in de440s.bsp). p(s) = a0 + a1*s has the
        // constant derivative p'(s) = a1, independent of s.
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1);
        let mut coeffs = Vec::new();
        coeffs.extend_from_slice(&[50.0, 50.0]); // MID, RADIUS
        coeffs.extend_from_slice(&[3.0, 5.0]); // x = 3 + 5s
        coeffs.extend_from_slice(&[1.0, -2.0]); // y = 1 - 2s
        coeffs.extend_from_slice(&[0.0, 0.0]); // z = 0
        let seg = ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et: 0.0,
            end_et: 100.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 1,
            degree,
            coeffs,
        };
        let v = seg.velocity(75.0).unwrap();
        assert_abs_diff_eq!(v[0], 5.0 / 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[1], -2.0 / 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chebyshev_acceleration_exact() {
        let seg = synthetic_segment();
        // d²/ds²: T0''=T1''=0, T2''=4, T3''=24s. At s=0.5 (et=75):
        // x'' = 3*4 = 12; y'' = 0; z'' = 6*(24*0.5) = 72
        // d²/det² = d²/ds² / RADIUS² = /2500
        let a = seg.acceleration(75.0).unwrap();
        assert_abs_diff_eq!(a[0], 12.0 / 2500.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[2], 72.0 / 2500.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chebyshev_acceleration_degree_one() {
        // p(s) = a0 + a1*s has zero second derivative for all s.
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1);
        let mut coeffs = Vec::new();
        coeffs.extend_from_slice(&[50.0, 50.0]); // MID, RADIUS
        coeffs.extend_from_slice(&[3.0, 5.0]); // x = 3 + 5s
        coeffs.extend_from_slice(&[1.0, -2.0]); // y = 1 - 2s
        coeffs.extend_from_slice(&[0.0, 0.0]); // z = 0
        let seg = ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et: 0.0,
            end_et: 100.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 1,
            degree,
            coeffs,
        };
        let a = seg.acceleration(75.0).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(a[i], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_chebyshev_acceleration_type_3() {
        // Type 3: components 0-2 position, 3-5 velocity polynomials.
        // Acceleration = d/det of the stored velocity polynomials.
        let degree = 3usize;
        let rsize = 2 + 6 * (degree + 1);
        let mut coeffs = Vec::new();
        coeffs.extend_from_slice(&[50.0, 50.0]); // MID, RADIUS
        coeffs.extend_from_slice(&[1.0, 2.0, 3.0, 0.0]); // x (unused here)
        coeffs.extend_from_slice(&[0.0, 4.0, 0.0, 0.0]); // y
        coeffs.extend_from_slice(&[5.0, 0.0, 0.0, 6.0]); // z
        coeffs.extend_from_slice(&[1.0, 2.0, 3.0, 0.0]); // vx = T0+2T1+3T2
        coeffs.extend_from_slice(&[0.0, 4.0, 0.0, 0.0]); // vy = 4T1
        coeffs.extend_from_slice(&[5.0, 0.0, 0.0, 6.0]); // vz = 5T0+6T3
        let seg = ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 3,
            ncomp: 6,
            start_et: 0.0,
            end_et: 100.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 1,
            degree,
            coeffs,
        };
        // Same series as test_chebyshev_velocity_exact, differentiated once:
        // vx' = 8, vy' = 4, vz' = 0 at s=0.5; /RADIUS = /50
        let a = seg.acceleration(75.0).unwrap();
        assert_abs_diff_eq!(a[0], 8.0 / 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[1], 4.0 / 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_record_selection_and_bounds() {
        let seg = synthetic_segment();
        // Second record: et=150 -> s=0.0 -> x = 1 + 3*(-1) = -2 (T2(0)=-1)
        let p = seg.position(150.0).unwrap();
        assert_abs_diff_eq!(p[0], 1.0 - 3.0, epsilon = 1e-12);
        // Boundary et exactly at record edge (100.0) uses record 1 (floor)
        assert!(seg.position(100.0).is_ok());
        // Segment edges are inclusive
        assert!(seg.position(0.0).is_ok());
        assert!(seg.position(200.0).is_ok());
        // Out of range errors mention the ET and coverage
        let err = seg.position(200.5).unwrap_err();
        assert!(format!("{}", err).contains("200.5"));
        assert!(seg.position(-0.1).is_err());
    }

    #[test]
    fn test_zero_radius_record_is_rejected() {
        // A record with RADIUS=0.0 would otherwise divide-by-zero into
        // NaN/Inf rather than a clean error.
        let degree = 0usize;
        let rsize = 2 + 3 * (degree + 1);
        let mut coeffs = vec![50.0, 0.0]; // MID, RADIUS=0
        coeffs.extend_from_slice(&[1.0, 2.0, 3.0]);
        let seg = ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et: 0.0,
            end_et: 100.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 1,
            degree,
            coeffs,
        };
        assert!(seg.position(50.0).is_err());
    }

    #[test]
    fn test_state_matches_position_and_velocity() {
        let seg = synthetic_segment();
        let (r, v) = seg.state(42.0).unwrap();
        assert_eq!(r, seg.position(42.0).unwrap());
        assert_eq!(v, seg.velocity(42.0).unwrap());
    }

    #[test]
    fn test_from_spk_summary_de440s() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        for summary in &daf.summaries {
            let seg = ChebyshevSegment::from_spk_summary(&daf, summary).unwrap();
            assert_eq!(seg.data_type, 2);
            assert_eq!(seg.ncomp, 3);
            assert_eq!(seg.rsize, 2 + 3 * (seg.degree + 1));
            assert_eq!(seg.coeffs.len(), seg.n * seg.rsize);
            assert!(seg.intlen > 0.0);
            // Records must cover the descriptor interval
            assert!(seg.init <= seg.start_et);
            assert!(seg.init + seg.intlen * seg.n as f64 >= seg.end_et);
            // Sanity: mid-span position is finite
            let et_mid = 0.5 * (seg.start_et + seg.end_et);
            let p = seg.position(et_mid).unwrap();
            assert!(p.iter().all(|x| x.is_finite()));
            // Mercury (199 wrt barycenter 1) and Venus (299 wrt barycenter
            // 2) have no satellites, so each planet coincides with its
            // barycenter and DE440s stores identically zero coefficients
            // for those two segments. All other segments must be nonzero.
            if matches!((seg.target, seg.center), (199, 1) | (299, 2)) {
                assert_eq!(p.norm(), 0.0);
            } else {
                assert!(p.norm() > 0.0);
            }
        }
    }

    #[test]
    fn test_unsupported_type_error_names_type() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        let mut summary = daf.summaries[0].clone();
        summary.ints[3] = 13; // pretend type 13
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("13"));
        assert!(msg.contains("2, 3"));
    }

    #[test]
    fn test_pck_type_3_error_does_not_claim_type_3_supported() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        // PCK ints layout: [frame_class_id, reference_frame, type, start, end].
        // Type 3 is not supported for PCK (only SPK); the error must not
        // claim otherwise.
        let summary = DAFSummary {
            name: "TEST_PCK".to_string(),
            doubles: vec![0.0, 1.0],
            ints: vec![10, 1, 3, 1, 1],
        };
        let err = ChebyshevSegment::from_pck_summary(&daf, &summary).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains('3'));
        assert!(msg.contains("supported types: 2"));
        assert!(!msg.contains("2, 3"));
    }

    #[test]
    fn test_from_spk_summary_rejects_non_j2000_frame() {
        // A Type 2/3 segment in a non-J2000 frame (e.g. ECLIPJ2000 = 17,
        // seen in some Horizons/generic small-body SPKs) must be rejected
        // rather than silently summed into chains as if it were ICRF.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        let mut summary = daf.summaries[0].clone();
        summary.ints[2] = 17; // ECLIPJ2000
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains(&summary.name));
        assert!(msg.contains("17"));
        assert!(msg.contains("frame 1"));
    }

    #[test]
    fn test_from_pck_summary_rejects_non_j2000_frame() {
        // Same guard, mirrored for PCK: ints[1] is the reference frame.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        let summary = DAFSummary {
            name: "TEST_PCK".to_string(),
            doubles: vec![0.0, 1.0],
            ints: vec![10, 17, 2, 1, 1],
        };
        let err = ChebyshevSegment::from_pck_summary(&daf, &summary).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("17"));
        assert!(msg.contains("frame 1"));
    }

    #[test]
    fn test_from_spk_summary_rejects_short_ints() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        let mut summary = daf.summaries[0].clone();
        summary.ints.truncate(5); // one short of the required 6
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        assert!(format!("{}", err).contains('5'));
    }

    #[test]
    fn test_is_coverage_error_classification() {
        // `is_coverage_error` must recognize exactly the errors built by
        // `coverage_error` (the epoch-aware chain fallback is gated on it)
        // and reject other error kinds, including IoError variants like
        // the invalid-RADIUS record error.
        assert!(is_coverage_error(&coverage_error(300.0, 0.0, 200.0, 10, 0)));
        assert!(!is_coverage_error(&BraheError::Error(
            "No ephemeris path from target 10 to center 0".to_string()
        )));
        assert!(!is_coverage_error(&BraheError::IoError(
            "Segment record 0 has invalid RADIUS 0 (target 10, center 0)".to_string()
        )));
    }

    #[test]
    fn test_is_coverage_error_classifies_pck_and_multi_variants() {
        // `is_coverage_error` must also recognize the multi-candidate SPK
        // variant and both PCK frame-lookup misses (used by
        // `registry::pck_query` to distinguish "try the next kernel" from
        // genuine data errors), while still rejecting unrelated errors.
        assert!(is_coverage_error(&coverage_error_multi(
            300.0, 0.0, 200.0, 10, 0, 3
        )));
        assert!(is_coverage_error(&pck_frame_not_found_error(31006)));
        assert!(is_coverage_error(&pck_out_of_coverage_error(2000.0, 31006)));
        assert!(!is_coverage_error(&BraheError::Error(
            "Frame class ID 31006 not found in loaded binary PCK data".to_string()
        )));
    }

    #[test]
    fn test_from_summary_rejects_directory_shorter_than_descriptor_end() {
        // Regression test: a malformed kernel could claim descriptor
        // coverage [start_et, end_et] wider than what the record directory
        // [INIT, INIT + N*INTLEN] actually spans. `record()`'s index clamp
        // would otherwise silently evaluate the wrong record for epochs
        // beyond the directory but still inside the (bogus) descriptor
        // interval, rather than erroring.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        let mut summary = daf.summaries[0].clone();
        // Claim the descriptor covers far beyond the record directory.
        summary.doubles[1] += 1.0e9;
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains(&summary.name));
        assert!(msg.contains("does not cover"));
    }

    #[test]
    fn test_chebyshev_derivative_degree_zero_is_zero() {
        // A degree-0 (single-coefficient) record has a constant position, so
        // its analytic velocity is exactly zero. Exercises the n == 0 early
        // return in `chebyshev_derivative` via `velocity`.
        let degree = 0usize;
        let rsize = 2 + 3 * (degree + 1); // 5
        let coeffs = vec![50.0, 50.0, 7.0, 8.0, 9.0]; // MID, RADIUS, x0, y0, z0
        let seg = ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et: 0.0,
            end_et: 100.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 1,
            degree,
            coeffs,
        };
        let v = seg.velocity(50.0).unwrap();
        assert_eq!(v, Vector3::zeros());
        // Position returns the constant coefficients.
        let p = seg.position(50.0).unwrap();
        assert_abs_diff_eq!(p[0], 7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p[1], 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p[2], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_validate_count_rejects_invalid_values() {
        // `validate_count`'s error branch rejects NaN, negative, and
        // fractional counts; a finite nonnegative integer is accepted.
        assert!(validate_count(f64::NAN, "N", "SPK", "seg").is_err());
        assert!(validate_count(-1.0, "RSIZE", "SPK", "seg").is_err());
        assert!(validate_count(2.5, "N", "PCK", "seg").is_err());
        assert_eq!(validate_count(4.0, "N", "SPK", "seg").unwrap(), 4);
    }

    #[test]
    fn test_type3_velocity_and_state_read_stored_velocity_polynomials() {
        // SPK Type 3 stores velocity coefficients directly (rather than
        // differentiating position), so `velocity` and `state` must read
        // components 3..6 rather than the analytic derivative path.
        let degree = 1usize;
        let rsize = 2 + 6 * (degree + 1); // 14
        let mut coeffs = vec![50.0, 50.0]; // MID, RADIUS
        // Position components x, y, z (each a0 + a1*T1(s)).
        coeffs.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Velocity components vx, vy, vz (read directly, not differentiated).
        coeffs.extend_from_slice(&[7.0, 8.0, 9.0, 0.0, 10.0, 0.0]);
        let seg = ChebyshevSegment {
            target: 10,
            center: 0,
            frame: 1,
            data_type: 3,
            ncomp: 6,
            start_et: 0.0,
            end_et: 100.0,
            init: 0.0,
            intlen: 100.0,
            rsize,
            n: 1,
            degree,
            coeffs,
        };
        // et=75 -> s=0.5. Velocity comes from the stored velocity polynomials.
        let v = seg.velocity(75.0).unwrap();
        assert_abs_diff_eq!(v[0], 7.0 + 8.0 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(v[1], 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v[2], 10.0, epsilon = 1e-12);
        // State shares the record lookup: position from comps 0..3, velocity
        // from comps 3..6.
        let (r, vs) = seg.state(75.0).unwrap();
        assert_abs_diff_eq!(r[0], 1.0 + 2.0 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], 3.0 + 4.0 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 5.0 + 6.0 * 0.5, epsilon = 1e-12);
        assert_eq!(vs, v);
    }

    #[test]
    fn test_from_pck_summary_rejects_short_summary() {
        // `from_pck_summary`'s guard: fewer than 5 ints must be rejected
        // before any word access, so the DAF contents are irrelevant.
        let daf = DAFFile::from_bytes(&crate::utils::testing::synthetic_spk_kernel_bytes(&[(
            10, 0, 1.0,
        )]))
        .unwrap();
        let summary = DAFSummary {
            name: "SHORT_PCK".to_string(),
            doubles: vec![0.0, 1.0],
            ints: vec![10, 1, 2, 1], // one short of the required 5
        };
        let err = ChebyshevSegment::from_pck_summary(&daf, &summary).unwrap_err();
        assert!(format!("{}", err).contains('4'));
    }

    #[test]
    fn test_from_summary_rejects_invalid_address_range() {
        // A start address of 0 (below the 1-based minimum) is rejected.
        let daf = DAFFile::from_bytes(&crate::utils::testing::synthetic_spk_kernel_bytes(&[(
            10, 0, 1.0,
        )]))
        .unwrap();
        let mut summary = daf.summaries[0].clone();
        summary.ints[4] = 0; // start_addr < 1
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        assert!(format!("{}", err).contains("invalid address range"));
    }

    #[test]
    fn test_from_summary_rejects_segment_too_short() {
        // A one-word address range cannot hold the 4-word trailer.
        let daf = DAFFile::from_bytes(&crate::utils::testing::synthetic_spk_kernel_bytes(&[(
            10, 0, 1.0,
        )]))
        .unwrap();
        let mut summary = daf.summaries[0].clone();
        summary.ints[4] = 1;
        summary.ints[5] = 1; // single word -> too short
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        assert!(format!("{}", err).contains("too short"));
    }

    #[test]
    fn test_from_summary_rejects_inconsistent_directory() {
        // Pointing the segment at the record's first four words makes the
        // trailer read RSIZE=1, N=0 -- an internally inconsistent directory.
        let daf = DAFFile::from_bytes(&crate::utils::testing::synthetic_spk_kernel_bytes(&[(
            10, 0, 1.0,
        )]))
        .unwrap();
        let mut summary = daf.summaries[0].clone();
        // Data begins at word 385; [385, 388] are [MID, RADIUS, x0, x1].
        summary.ints[4] = 385;
        summary.ints[5] = 388;
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        assert!(format!("{}", err).contains("inconsistent directory"));
    }

    // --- SPK Type 21 (Extended Modified Difference Array) ------------------
    //
    // Fixture `test_assets/ceres_horizons_type21.bsp` is a real 74 KB
    // Horizons SPK for Ceres (target 20000001, center 10 = Sun, frame 1 =
    // J2000/ICRF, type 21) covering 2015-12..2016-03. Note the center is the
    // Sun (10), not the SSB (0): Horizons small-body SPKs are Sun-centered.

    /// Ceres Horizons type-21 fixture path, or `None` if not present.
    fn ceres_type21_path() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("ceres_horizons_type21.bsp");
        p.exists().then_some(p)
    }

    /// Load the single type-21 segment from the Ceres fixture, verifying it
    /// dispatches through `SpkSegment` to the `Type21` arm.
    fn load_ceres_type21() -> Option<Type21Segment> {
        let path = ceres_type21_path()?;
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        assert_eq!(daf.summaries.len(), 1, "fixture has one segment");
        let seg = SpkSegment::from_spk_summary(&daf, &daf.summaries[0]).unwrap();
        match seg {
            SpkSegment::Type21(s) => Some(s),
            SpkSegment::Chebyshev(_) => panic!("type-21 segment dispatched to Chebyshev arm"),
        }
    }

    /// A record's stored reference epoch and reference position/velocity.
    fn record_reference(seg: &Type21Segment, idx: usize) -> (f64, Vector3<f64>, Vector3<f64>) {
        let rec = &seg.records[idx * seg.dlsize..(idx + 1) * seg.dlsize];
        let md = seg.maxdim;
        (
            rec[0],
            Vector3::new(rec[md + 1], rec[md + 3], rec[md + 5]),
            Vector3::new(rec[md + 2], rec[md + 4], rec[md + 6]),
        )
    }

    #[test]
    #[serial]
    fn test_type21_load_and_metadata() {
        // Gate 1: the fixture loads as a Type21 segment with self-consistent
        // metadata (target/center/frame, MAXDIM/N, and a directory that
        // matches the record count and spans the descriptor coverage).
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        assert_eq!(seg.target, 20000001);
        assert_eq!(seg.center, 10); // Sun, not SSB
        assert_eq!(seg.frame, 1);
        assert_eq!(seg.maxdim, 20);
        assert_eq!(seg.dlsize, 4 * 20 + 11);
        assert_eq!(seg.n, 12);
        assert_eq!(seg.records.len(), seg.n * seg.dlsize);
        assert_eq!(seg.epochs.len(), seg.n);
        // Coverage spans the 2015-12..2016-03 window (ET seconds past J2000).
        assert_abs_diff_eq!(seg.start_et, 502200000.0, epsilon = 1.0);
        assert_abs_diff_eq!(seg.end_et, 510062400.0, epsilon = 1.0);
        // Directory ascending and covering the descriptor.
        assert!(seg.epochs.windows(2).all(|w| w[1] >= w[0]));
        assert!(seg.epochs[0] >= seg.start_et);
        assert!(seg.epochs[seg.n - 1] >= seg.end_et);
    }

    #[test]
    #[serial]
    fn test_type21_exact_at_reference_epoch() {
        // Gate 2: at a record's reference epoch DELTA = 0, so every
        // difference term vanishes and the evaluated state must reduce to the
        // stored reference position/velocity. Validates parsing + the
        // interpolation base case, deterministically, for several records.
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        let mut max_rel = 0.0f64;
        for idx in [0usize, 4, seg.n / 2, seg.n - 1] {
            let rec = &seg.records[idx * seg.dlsize..(idx + 1) * seg.dlsize];
            let (tl, refpos, refvel) = record_reference(&seg, idx);
            let (p, v) = seg.spke21(rec, tl).unwrap();
            let rp = (p - refpos).norm() / refpos.norm();
            let rv = (v - refvel).norm() / refvel.norm();
            max_rel = max_rel.max(rp).max(rv);
        }
        // Exact up to floating-point round-off (well under the 1e-6 target).
        assert!(
            max_rel < 1e-12,
            "max relative error at ref epoch {max_rel:e}"
        );
    }

    #[test]
    #[serial]
    fn test_type21_inter_record_continuity() {
        // Gate 3: at an interior record boundary the records on each side
        // must agree. Record i covers up to epochs[i]; record i+1 begins
        // there. Evaluate both at that shared epoch.
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        let k = 5usize;
        let boundary = seg.epochs[k];
        let rec_a = &seg.records[k * seg.dlsize..(k + 1) * seg.dlsize];
        let rec_b = &seg.records[(k + 1) * seg.dlsize..(k + 2) * seg.dlsize];
        let (pa, va) = seg.spke21(rec_a, boundary).unwrap();
        let (pb, vb) = seg.spke21(rec_b, boundary).unwrap();
        let dp_m = (pa - pb).norm() * 1.0e3; // km -> m
        let dv_mm = (va - vb).norm() * 1.0e6; // km/s -> mm/s
        assert!(dp_m < 1.0, "position discontinuity {dp_m:e} m");
        assert!(dv_mm < 1.0, "velocity discontinuity {dv_mm:e} mm/s");
    }

    #[test]
    #[serial]
    fn test_type21_heliocentric_distance() {
        // Gate 4: Ceres's distance from its center (the Sun) at a mid-span
        // epoch must be ~2.77 AU (Ceres semimajor axis). Catches unit/scale
        // errors such as a missing km->m factor or a mis-sized record.
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        const AU_KM: f64 = 1.495978707e8;
        let et_mid = 0.5 * (seg.start_et + seg.end_et);
        let d_au = seg.position(et_mid).unwrap().norm() / AU_KM;
        assert!(
            (2.5..=3.1).contains(&d_au),
            "heliocentric distance {d_au} AU"
        );
    }

    #[test]
    #[serial]
    fn test_type21_transcription_fidelity_vs_cspice() {
        // Transcription-fidelity gate: reproduce JPL CSPICE N0067
        // `spkgeo(20000001, et, "J2000", 10)` (Ceres relative to the Sun,
        // J2000 frame, km / km·s⁻¹) evaluated on this exact fixture. CSPICE
        // is the definitive type-21 evaluator, but this port and CSPICE share
        // algorithm lineage (both descend from `spke21.f`), so bit-exact
        // agreement proves the transcription is faithful — not, on its own,
        // that the algorithm is physically correct. Independent correctness
        // comes from `test_type21_heliocentric_distance` (physical scale) and
        // `test_type21_horizons_vectors_cross_check` (an independent JPL
        // Horizons solution). Our reader reproduces CSPICE to floating-point
        // round-off (far under the 1 km / 1e-3 km·s⁻¹ gate).
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        // (et [s past J2000 TDB], [x, y, z, vx, vy, vz] km, km/s from CSPICE)
        let cases: [(f64, [f64; 6]); 3] = [
            (
                502300000.0,
                [
                    364552647.0778291,
                    -194445550.31376046,
                    -165919199.86730012,
                    9.170483825674097,
                    13.14603077087047,
                    4.329207575774189,
                ],
            ),
            (
                505000000.0,
                [
                    387273518.5807745,
                    -157952245.1120478,
                    -153343872.20668784,
                    7.644831886609603,
                    13.86119251302727,
                    4.977180911103915,
                ],
            ),
            (
                509000000.0,
                [
                    413088373.59470785,
                    -100835751.21701588,
                    -131676362.58027884,
                    5.235826381462796,
                    14.640142027052732,
                    5.835197035168514,
                ],
            ),
        ];
        let mut max_dp = 0.0f64;
        let mut max_dv = 0.0f64;
        for (et, expected) in cases {
            let (p, v) = seg.state(et).unwrap();
            let dp = (p - Vector3::new(expected[0], expected[1], expected[2])).norm();
            let dv = (v - Vector3::new(expected[3], expected[4], expected[5])).norm();
            max_dp = max_dp.max(dp);
            max_dv = max_dv.max(dv);
        }
        assert!(max_dp < 1.0, "position error vs CSPICE {max_dp:e} km");
        assert!(max_dv < 1.0e-3, "velocity error vs CSPICE {max_dv:e} km/s");
    }

    #[test]
    #[serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_type21_horizons_vectors_cross_check() {
        // Independent cross-check against a JPL Horizons VECTORS solution
        // (a different code path and orbit fit than the fixture kernel, so
        // this — unlike the CSPICE transcription gate — does not share
        // lineage with the port). Integration-gated: it asserts hardcoded
        // reference numbers rather than making a network call, but it is
        // segregated from the default gate as an on-demand independent check.
        //
        // Reference states from a live query on 2026-07-21 (Ceres relative to
        // the Sun, J2000/ICRF, km / km·s⁻¹):
        //   https://ssd.jpl.nasa.gov/api/horizons.api?format=text
        //     &COMMAND=2000001&EPHEM_TYPE=VECTORS&CENTER=500@10
        //     &REF_PLANE=FRAME&REF_SYSTEM=J2000&OUT_UNITS=KM-S&VEC_TABLE=2
        //     &TLIST=2457358.657407407,2457389.907407407,2457436.203703704
        //
        // The reader agrees to ~2.9 km / ~1e-7 km·s⁻¹ — a uniform ~7e-9
        // relative offset in BOTH position and velocity. That is the
        // signature of a solution-vintage difference (the committed fixture
        // predates Horizons' current `dawn_final` fit), NOT an interpolation
        // error: exact-at-reference-epoch is bit-exact, and the reader's
        // position is the exact antiderivative of its velocity. The
        // tolerances below are set with margin around that physical offset.
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        // (et [s past J2000 TDB], [x, y, z, vx, vy, vz] km, km/s from Horizons)
        let cases: [(f64, [f64; 6]); 3] = [
            (
                502300000.0,
                [
                    3.645526491447987e8,
                    -1.944455484833567e8,
                    -1.65919200671872e8,
                    9.170483728168245e0,
                    1.314603079056034e1,
                    4.329207566312728e0,
                ],
            ),
            (
                505000000.0,
                [
                    3.872735203899039e8,
                    -1.579522432469829e8,
                    -1.533438730395891e8,
                    7.644831793452791e0,
                    1.386119251920006e1,
                    4.977180899644783e0,
                ],
            ),
            (
                509000000.0,
                [
                    4.130883750499314e8,
                    -1.008357493617283e8,
                    -1.316763634622429e8,
                    5.235826297983635e0,
                    1.464014201601533e1,
                    5.835197022125968e0,
                ],
            ),
        ];
        let mut max_dp = 0.0f64;
        let mut max_dv = 0.0f64;
        for (et, expected) in cases {
            let (p, v) = seg.state(et).unwrap();
            let dp = (p - Vector3::new(expected[0], expected[1], expected[2])).norm();
            let dv = (v - Vector3::new(expected[3], expected[4], expected[5])).norm();
            max_dp = max_dp.max(dp);
            max_dv = max_dv.max(dv);
        }
        // Actual residual ~2.9 km / ~1e-7 km·s⁻¹; margined tolerances.
        assert!(max_dp < 5.0, "position error vs Horizons {max_dp:e} km");
        assert!(
            max_dv < 1.0e-6,
            "velocity error vs Horizons {max_dv:e} km/s"
        );
    }

    #[test]
    #[serial]
    fn test_type21_out_of_coverage_errors() {
        // Epochs outside the descriptor interval yield a coverage error that
        // the chain fallback can recognize.
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        let err = seg.position(seg.start_et - 1.0).unwrap_err();
        assert!(is_coverage_error(&err));
        assert!(seg.position(seg.end_et + 1.0).is_err());
        // In-coverage endpoints evaluate.
        assert!(seg.state(seg.start_et).is_ok());
        assert!(seg.state(seg.end_et).is_ok());
    }

    #[test]
    #[serial]
    fn test_type21_rejects_corrupt_record_orders() {
        // A record whose KQMAX1 exceeds MAXDIM+1 would drive out-of-bounds
        // indexing of the stepsize/difference arrays; it must be rejected
        // cleanly rather than panicking.
        let Some(mut seg) = load_ceres_type21() else {
            return;
        };
        let slot = 4 * seg.maxdim + 7; // KQMAX1 word of record 0
        seg.records[slot] = 999.0;
        let et = seg.start_et + 1.0; // inside record 0
        let err = seg.position(et).unwrap_err();
        assert!(format!("{}", err).contains("invalid orders"));
    }

    #[test]
    #[serial]
    fn test_type21_state_matches_position_velocity_and_acceleration() {
        // `state` agrees with separate `position`/`velocity` calls, and the
        // finite-difference acceleration matches a central difference of
        // `velocity` across neighboring records.
        let Some(seg) = load_ceres_type21() else {
            return;
        };
        let et = 0.5 * (seg.start_et + seg.end_et);
        let (p, v) = seg.state(et).unwrap();
        assert_eq!(p, seg.position(et).unwrap());
        assert_eq!(v, seg.velocity(et).unwrap());
        // Acceleration ~ GM_sun / r^2 for Ceres at ~2.98 AU: ~6.7e-7 km/s^2.
        let a = seg.acceleration(et).unwrap();
        assert!(
            (1.0e-8..1.0e-5).contains(&a.norm()),
            "accel {} km/s^2",
            a.norm()
        );
    }

    #[test]
    fn test_from_summary_rejects_directory_starting_after_descriptor_start() {
        // Same defect as above, mirrored on the start side: INIT after
        // start_et means the directory doesn't cover the descriptor's
        // earliest claimed epochs.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let daf = crate::spice::daf::DAFFile::from_file(&path).unwrap();
        let mut summary = daf.summaries[0].clone();
        // Claim the descriptor starts far before the record directory.
        summary.doubles[0] -= 1.0e9;
        let err = ChebyshevSegment::from_spk_summary(&daf, &summary).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains(&summary.name));
        assert!(msg.contains("does not cover"));
    }
}
