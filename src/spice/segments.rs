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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;

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
