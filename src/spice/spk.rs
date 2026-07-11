/*!
 * Native SPK (Spacecraft and Planet Kernel) ephemeris reader.
 *
 * An [`SPK`] holds one parsed kernel with all Chebyshev coefficients
 * resident in memory. Queries resolve a signed chain of segments linking
 * `target` to `center` (e.g. Sun rel. Earth = +Sun/SSB − EMB/SSB −
 * Earth/EMB), cache the chain per pair, and evaluate with O(1) record
 * lookup. Output states are in the kernel's inertial frame (J2000 label,
 * ICRF axes for DE4xx kernels) in SI units.
 *
 * The cached chain is resolved by topology alone (shortest hop count),
 * without regard to epoch; this is correct and fast for the common case
 * of one full-span segment per body pair. If the cached chain's segments
 * don't cover a queried epoch while a different (longer) path does,
 * queries transparently fall back to an epoch-aware re-resolution — see
 * `resolve_chain_for_epoch` and `evaluate_with_epoch_fallback`.
 */

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, RwLock};

use nalgebra::{Vector3, Vector6};

use crate::utils::BraheError;

use super::daf::DAFFile;
use super::segments::{ChebyshevSegment, coverage_error_multi, is_coverage_error};

/// One link in a resolved target→center chain: the candidate segments for a
/// single body pair (in input order; last-listed takes precedence) and the
/// traversal sign.
#[derive(Debug, Clone)]
pub(crate) struct ChainLink {
    /// Candidate segments for this pair, kept in input order; evaluation
    /// checks them last-to-first for ET coverage (last-listed wins).
    pub segments: Vec<Arc<ChebyshevSegment>>,
    /// +1.0 when the segment's (target rel center) direction matches the
    /// traversal; −1.0 when traversed in reverse.
    pub sign: f64,
}

/// Breadth-first search over the segment connectivity graph, returning the
/// signed chain of links whose sum gives `target` rel `center`.
///
/// # Arguments
/// - `segments`: Candidate segments to search (may span multiple kernels)
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
///
/// # Returns
/// - Chain of [`ChainLink`]s from `target` down to `center`, or
///   `BraheError` if no path connects them
pub(crate) fn resolve_chain(
    segments: &[Arc<ChebyshevSegment>],
    target: i32,
    center: i32,
) -> Result<Vec<ChainLink>, BraheError> {
    resolve_chain_filtered(segments, target, center, None)
}

/// Epoch-aware variant of [`resolve_chain`]: the same breadth-first search,
/// but restricted to segments that cover `et`.
///
/// Chain topology is resolved once per `(target, center)` pair and cached
/// (see [`SPK::chain`] / the registry's `global_chain`), which is correct
/// and fast for the common case of one full-span segment per body pair
/// (e.g. DE44x kernels). It can, however, pick a shorter path (fewer
/// hops) whose segments don't cover a given epoch while a longer path
/// does — possible with multiple kernels or multi-segment kernels of
/// partial temporal coverage. This function is the epoch-aware fallback
/// for that rare case: an edge exists between two bodies only if at least
/// one candidate segment for that pair covers `et`, and each resulting
/// [`ChainLink`] keeps only the `et`-covering candidates (still in input
/// precedence order). Callers should try the cached topology-only chain
/// first and only call this on evaluation failure; see
/// [`evaluate_with_epoch_fallback`].
///
/// # Arguments
/// - `segments`: Candidate segments to search (may span multiple kernels)
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `et`: Epoch that every segment in the resolved chain must cover.
///   Units: [s] (TDB past J2000)
///
/// # Returns
/// - Chain of [`ChainLink`]s from `target` down to `center`, using only
///   `et`-covering segments, or `BraheError` (naming `target`, `center`,
///   and `et`) if no such path exists
pub(crate) fn resolve_chain_for_epoch(
    segments: &[Arc<ChebyshevSegment>],
    target: i32,
    center: i32,
    et: f64,
) -> Result<Vec<ChainLink>, BraheError> {
    resolve_chain_filtered(segments, target, center, Some(et))
}

/// Shared breadth-first search backing both [`resolve_chain`] and
/// [`resolve_chain_for_epoch`].
///
/// # Arguments
/// - `segments`: Candidate segments to search (may span multiple kernels)
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `et`: When `Some`, only segments covering this epoch participate in
///   the graph (both for adjacency discovery and as chain-link
///   candidates), and the "no path" error names the epoch. When `None`,
///   the search is purely topological (all segments participate).
///
/// # Returns
/// - Chain of [`ChainLink`]s from `target` down to `center`, or
///   `BraheError` if no path connects them
fn resolve_chain_filtered(
    segments: &[Arc<ChebyshevSegment>],
    target: i32,
    center: i32,
    et: Option<f64>,
) -> Result<Vec<ChainLink>, BraheError> {
    if target == center {
        return Ok(Vec::new());
    }

    // Adjacency: body -> neighboring bodies. Pair key is (target, center)
    // as stored in segments; candidates kept in input order (evaluation
    // selects the last covering candidate, so later input wins).
    let mut pair_segments: HashMap<(i32, i32), Vec<Arc<ChebyshevSegment>>> = HashMap::new();
    let mut adjacency: HashMap<i32, Vec<i32>> = HashMap::new();
    for seg in segments {
        if et.is_some_and(|et| !seg.covers(et)) {
            continue;
        }
        let key = (seg.target, seg.center);
        let entry = pair_segments.entry(key).or_default();
        if entry.is_empty() {
            adjacency.entry(seg.target).or_default().push(seg.center);
            adjacency.entry(seg.center).or_default().push(seg.target);
        }
        entry.push(Arc::clone(seg));
    }

    // BFS from center to target
    let mut visited: HashSet<i32> = HashSet::from([center]);
    let mut parent: HashMap<i32, i32> = HashMap::new();
    let mut queue = VecDeque::from([center]);
    while let Some(node) = queue.pop_front() {
        if node == target {
            break;
        }
        for &next in adjacency.get(&node).into_iter().flatten() {
            if visited.insert(next) {
                parent.insert(next, node);
                queue.push_back(next);
            }
        }
    }

    if !parent.contains_key(&target) {
        return Err(BraheError::Error(match et {
            Some(et) => format!(
                "No ephemeris path from target {} to center {} with segment coverage at epoch ET {} in loaded SPK data",
                target, center, et
            ),
            None => format!(
                "No ephemeris path from target {} to center {} in loaded SPK data",
                target, center
            ),
        }));
    }

    // Walk back target -> center, emitting one link per edge.
    let mut chain = Vec::new();
    let mut node = target;
    while node != center {
        let prev = parent[&node];
        // Edge prev -> node. A stored segment (target=node, center=prev)
        // gives node rel prev directly (sign +1); (target=prev, center=node)
        // gives the reverse (sign -1).
        if let Some(segs) = pair_segments.get(&(node, prev)) {
            chain.push(ChainLink {
                segments: segs.clone(),
                sign: 1.0,
            });
        } else {
            let segs = pair_segments
                .get(&(prev, node))
                .expect("edge implies a stored pair in one direction");
            chain.push(ChainLink {
                segments: segs.clone(),
                sign: -1.0,
            });
        }
        node = prev;
    }
    Ok(chain)
}

/// Select the last segment in `link` covering `et` (SPICE convention: the
/// most recently loaded/last-listed segment takes precedence).
///
/// On a miss, the error reports the union of every candidate segment's
/// coverage interval (not just the first candidate's, which can be far
/// narrower than the union when a pair has multiple partial-coverage
/// segments) and how many candidates were checked.
fn covering_segment(link: &ChainLink, et: f64) -> Result<&Arc<ChebyshevSegment>, BraheError> {
    link.segments
        .iter()
        .rev()
        .find(|s| s.covers(et))
        .ok_or_else(|| {
            let start_et = link
                .segments
                .iter()
                .map(|s| s.start_et)
                .fold(f64::INFINITY, f64::min);
            let end_et = link
                .segments
                .iter()
                .map(|s| s.end_et)
                .fold(f64::NEG_INFINITY, f64::max);
            let s0 = &link.segments[0];
            coverage_error_multi(
                et,
                start_et,
                end_et,
                s0.target,
                s0.center,
                link.segments.len(),
            )
        })
}

/// Sum link positions along a chain. Units: kernel-natural (km).
///
/// # Arguments
/// - `chain`: Resolved chain from [`resolve_chain`]
/// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
///
/// # Returns
/// - Summed position. Units: [km]
pub(crate) fn evaluate_chain_position(
    chain: &[ChainLink],
    et: f64,
) -> Result<Vector3<f64>, BraheError> {
    let mut r = Vector3::zeros();
    for link in chain {
        r += covering_segment(link, et)?.position(et)? * link.sign;
    }
    Ok(r)
}

/// Sum link velocities along a chain. Units: kernel-natural (km/s).
///
/// # Arguments
/// - `chain`: Resolved chain from [`resolve_chain`]
/// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
///
/// # Returns
/// - Summed velocity. Units: [km/s]
pub(crate) fn evaluate_chain_velocity(
    chain: &[ChainLink],
    et: f64,
) -> Result<Vector3<f64>, BraheError> {
    let mut v = Vector3::zeros();
    for link in chain {
        v += covering_segment(link, et)?.velocity(et)? * link.sign;
    }
    Ok(v)
}

/// Sum link states along a chain. Units: kernel-natural (km, km/s).
///
/// # Arguments
/// - `chain`: Resolved chain from [`resolve_chain`]
/// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
///
/// # Returns
/// - Summed `(position, velocity)`. Units: [km], [km/s]
pub(crate) fn evaluate_chain_state(
    chain: &[ChainLink],
    et: f64,
) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
    let mut r = Vector3::zeros();
    let mut v = Vector3::zeros();
    for link in chain {
        let (lr, lv) = covering_segment(link, et)?.state(et)?;
        r += lr * link.sign;
        v += lv * link.sign;
    }
    Ok((r, v))
}

/// Evaluate a cached (topology-only) `chain` at `et` via `eval`; on an
/// out-of-coverage failure, fall back to an epoch-aware chain resolved
/// over `segments_for_fallback()` and retry.
///
/// The cached chain is resolved once per `(target, center)` pair without
/// regard to epoch, which is correct and O(1) for the common case (one
/// full-span segment per pair, e.g. DE44x kernels). It can pick a
/// shorter/direct path that doesn't cover `et` while a longer path does
/// (multiple kernels, or multi-segment kernels with partial temporal
/// coverage); [`resolve_chain_for_epoch`] recovers that case by
/// re-resolving the chain using only `et`-covering segments. This
/// fallback is invoked only after the primary `eval` call has already
/// failed, so it does not add cost to the hot (single-segment-per-pair)
/// path. `segments_for_fallback` is a closure (rather than an eagerly
/// gathered slice) so that callers whose segment list is expensive to
/// assemble (e.g. the global registry, which aggregates across all
/// loaded kernels) only pay that cost on this rare-path retry.
///
/// The fallback chain reflects only `et`'s coverage and is intentionally
/// never cached: caching it would require a validity interval per
/// `(target, center)` pair (since the correct chain can vary by epoch),
/// which is not worth the added complexity for what should be a rare
/// event in practice.
///
/// Only out-of-coverage errors (identified via
/// [`is_coverage_error`](super::segments::is_coverage_error)) trigger the
/// fallback: a coverage miss is the one failure a different chain can
/// legitimately fix. Any other evaluation error (e.g. a corrupt record's
/// invalid `RADIUS`) propagates unchanged — re-routing around corrupt
/// data would silently mask the true cause.
///
/// # Arguments
/// - `chain`: Cached topology-only chain for `(target, center)`
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `et`: Epoch to evaluate at. Units: [s] (TDB past J2000)
/// - `segments_for_fallback`: Lazily produces the full candidate segment
///   list to re-resolve against, only called if `eval(chain, et)` fails
/// - `eval`: Evaluator (one of [`evaluate_chain_position`],
///   [`evaluate_chain_velocity`], [`evaluate_chain_state`])
///
/// # Returns
/// - The evaluated result; the original error if it was not a coverage
///   miss; or `BraheError` if neither the cached chain nor the
///   epoch-aware fallback chain covers `et`
pub(crate) fn evaluate_with_epoch_fallback<T>(
    chain: &[ChainLink],
    target: i32,
    center: i32,
    et: f64,
    segments_for_fallback: impl FnOnce() -> Vec<Arc<ChebyshevSegment>>,
    eval: impl Fn(&[ChainLink], f64) -> Result<T, BraheError>,
) -> Result<T, BraheError> {
    match eval(chain, et) {
        Ok(value) => Ok(value),
        Err(err) if is_coverage_error(&err) => {
            let segments = segments_for_fallback();
            let fallback = resolve_chain_for_epoch(&segments, target, center, et)?;
            eval(&fallback, et)
        }
        Err(err) => Err(err),
    }
}

/// Per-pair resolved chain cache, keyed by `(target, center)`.
type ChainCache = RwLock<HashMap<(i32, i32), Arc<Vec<ChainLink>>>>;

/// A loaded SPK ephemeris kernel with in-memory Chebyshev coefficients and
/// cached per-pair segment chains.
#[derive(Debug)]
pub struct SPK {
    segments: Vec<Arc<ChebyshevSegment>>,
    pub(crate) chain_cache: ChainCache,
}

impl SPK {
    /// Load an SPK kernel from a file.
    ///
    /// # Arguments
    /// - `path`: Path to a `.bsp` file (SPK types 2 and 3 supported)
    ///
    /// # Returns
    /// - Loaded kernel, or an error naming any unsupported segment type
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::spice::SPK;
    ///
    /// let spk = SPK::from_file(Path::new("de440s.bsp")).unwrap();
    /// ```
    pub fn from_file(path: &Path) -> Result<Self, BraheError> {
        let daf = DAFFile::from_file(path)?;
        Self::from_daf(daf)
    }

    /// Load an SPK kernel from an in-memory byte buffer.
    ///
    /// # Arguments
    /// - `bytes`: Raw bytes of a binary SPICE kernel
    ///
    /// # Returns
    /// - Loaded kernel, or an error naming any unsupported segment type
    ///
    /// # Example
    /// ```
    /// use brahe::spice::SPK;
    ///
    /// let bytes = std::fs::read("test_assets/de440s.bsp").unwrap_or_default();
    /// let _ = SPK::from_bytes(&bytes); // no-op if the test asset is absent
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BraheError> {
        Self::from_daf(DAFFile::from_bytes(bytes)?)
    }

    pub(crate) fn from_daf(daf: DAFFile) -> Result<Self, BraheError> {
        if daf.id_word != "DAF/SPK" {
            return Err(BraheError::IoError(format!(
                "Not an SPK kernel: ID word is '{}', expected 'DAF/SPK'",
                daf.id_word
            )));
        }
        let mut segments = Vec::with_capacity(daf.summaries.len());
        for summary in &daf.summaries {
            segments.push(Arc::new(ChebyshevSegment::from_spk_summary(&daf, summary)?));
        }
        Ok(SPK {
            segments,
            chain_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Segments in file order (for the global registry).
    pub(crate) fn segments(&self) -> &[Arc<ChebyshevSegment>] {
        &self.segments
    }

    fn chain(&self, target: i32, center: i32) -> Result<Arc<Vec<ChainLink>>, BraheError> {
        if let Some(chain) = self.chain_cache.read().unwrap().get(&(target, center)) {
            return Ok(Arc::clone(chain));
        }
        let chain = Arc::new(resolve_chain(&self.segments, target, center)?);
        self.chain_cache
            .write()
            .unwrap()
            .insert((target, center), Arc::clone(&chain));
        Ok(chain)
    }

    /// Position of `target` relative to `center` at ET `et`.
    ///
    /// Resolves the segment chain once per `(target, center)` pair and
    /// caches it. If the cached chain's segments don't cover `et` (e.g. a
    /// cached direct link with only partial temporal coverage, while a
    /// longer path through other segments does cover `et`), transparently
    /// falls back to an epoch-aware re-resolution; see
    /// `resolve_chain_for_epoch`. This fallback is not cached.
    ///
    /// # Arguments
    /// - `target`: NAIF ID of the target body
    /// - `center`: NAIF ID of the observing/center body
    /// - `et`: TDB seconds past J2000
    ///
    /// # Returns
    /// - Position in the kernel's inertial frame (ICRF axes). Units: [m]
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::spice::SPK;
    ///
    /// let spk = SPK::from_file(Path::new("de440s.bsp")).unwrap();
    /// let r_sun = spk.position(10, 399, 0.0).unwrap(); // Sun rel Earth at J2000
    /// ```
    pub fn position(&self, target: i32, center: i32, et: f64) -> Result<Vector3<f64>, BraheError> {
        let chain = self.chain(target, center)?;
        Ok(evaluate_with_epoch_fallback(
            &chain,
            target,
            center,
            et,
            || self.segments.clone(),
            evaluate_chain_position,
        )? * 1.0e3)
    }

    /// Velocity of `target` relative to `center` at ET `et`.
    ///
    /// Resolves the segment chain once per `(target, center)` pair and
    /// caches it. If the cached chain's segments don't cover `et` (e.g. a
    /// cached direct link with only partial temporal coverage, while a
    /// longer path through other segments does cover `et`), transparently
    /// falls back to an epoch-aware re-resolution; see
    /// `resolve_chain_for_epoch`. This fallback is not cached.
    ///
    /// # Arguments
    /// - `target`: NAIF ID of the target body
    /// - `center`: NAIF ID of the observing/center body
    /// - `et`: TDB seconds past J2000
    ///
    /// # Returns
    /// - Velocity in the kernel's inertial frame (ICRF axes). Units: [m/s]
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::spice::SPK;
    ///
    /// let spk = SPK::from_file(Path::new("de440s.bsp")).unwrap();
    /// let v_sun = spk.velocity(10, 399, 0.0).unwrap(); // Sun rel Earth at J2000
    /// ```
    pub fn velocity(&self, target: i32, center: i32, et: f64) -> Result<Vector3<f64>, BraheError> {
        let chain = self.chain(target, center)?;
        Ok(evaluate_with_epoch_fallback(
            &chain,
            target,
            center,
            et,
            || self.segments.clone(),
            evaluate_chain_velocity,
        )? * 1.0e3)
    }

    /// Position and velocity of `target` relative to `center` at ET `et`.
    ///
    /// Resolves the segment chain once per `(target, center)` pair and
    /// caches it. If the cached chain's segments don't cover `et` (e.g. a
    /// cached direct link with only partial temporal coverage, while a
    /// longer path through other segments does cover `et`), transparently
    /// falls back to an epoch-aware re-resolution; see
    /// `resolve_chain_for_epoch`. This fallback is not cached.
    ///
    /// # Arguments
    /// - `target`: NAIF ID of the target body
    /// - `center`: NAIF ID of the observing/center body
    /// - `et`: TDB seconds past J2000
    ///
    /// # Returns
    /// - State `[x, y, z, vx, vy, vz]` in the kernel's inertial frame
    ///   (ICRF axes). Units: [m, m/s]
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::spice::SPK;
    ///
    /// let spk = SPK::from_file(Path::new("de440s.bsp")).unwrap();
    /// let x_sun = spk.state(10, 399, 0.0).unwrap(); // Sun rel Earth at J2000
    /// ```
    pub fn state(&self, target: i32, center: i32, et: f64) -> Result<Vector6<f64>, BraheError> {
        let chain = self.chain(target, center)?;
        let (r, v) = evaluate_with_epoch_fallback(
            &chain,
            target,
            center,
            et,
            || self.segments.clone(),
            evaluate_chain_state,
        )?;
        Ok(Vector6::new(
            r[0] * 1.0e3,
            r[1] * 1.0e3,
            r[2] * 1.0e3,
            v[0] * 1.0e3,
            v[1] * 1.0e3,
            v[2] * 1.0e3,
        ))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn load_de440s() -> Option<SPK> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        path.exists().then(|| SPK::from_file(&path).unwrap())
    }

    // ET for 2025-01-01 00:00 TDB, safely inside DE440s coverage
    const ET_2025: f64 = 788_961_600.0;

    /// Build a synthetic single-record type-2 segment whose x-position is
    /// the constant `x` (km) over `[start_et, end_et]`; y = z = 0.
    fn constant_segment(
        target: i32,
        center: i32,
        start_et: f64,
        end_et: f64,
        x: f64,
    ) -> Arc<ChebyshevSegment> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1);
        let mid = (start_et + end_et) / 2.0;
        let radius = (end_et - start_et) / 2.0;
        let coeffs = vec![mid, radius, x, 0.0, 0.0, 0.0, 0.0, 0.0];
        Arc::new(ChebyshevSegment {
            target,
            center,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et,
            end_et,
            init: start_et,
            intlen: end_et - start_et,
            rsize,
            n: 1,
            degree,
            coeffs,
        })
    }

    #[test]
    fn test_covering_segment_error_reports_union_and_candidate_count() {
        // Regression test: when a pair has multiple candidate segments with
        // disjoint coverage and none covers the queried epoch, the error
        // must report the union of all candidates' intervals (not just
        // `segments[0]`'s, which would be far narrower here) and how many
        // candidates were checked.
        let seg_a = constant_segment(10, 0, 0.0, 100.0, 1.0);
        let seg_b = constant_segment(10, 0, 200.0, 300.0, 2.0);
        let chain = resolve_chain(&[seg_a, seg_b], 10, 0).unwrap();

        let err = evaluate_chain_position(&chain, 150.0).unwrap_err();
        let msg = format!("{}", err);
        // Union is [0, 300], not segments[0]'s narrower [0, 100].
        assert!(msg.contains('0') && msg.contains("300"), "msg: {}", msg);
        assert!(msg.contains("2 candidate segments"), "msg: {}", msg);
    }

    #[test]
    fn test_chain_overlapping_segments_last_wins() {
        // SPICE precedence: when multiple same-pair segments cover an epoch,
        // the LAST one in input (load) order wins. Regression test for the
        // segment-selection direction in `covering_segment`.
        let seg_old = constant_segment(10, 0, 0.0, 200.0, 1.0);
        let seg_new = constant_segment(10, 0, 0.0, 100.0, 2.0);
        let chain = resolve_chain(&[seg_old, seg_new], 10, 0).unwrap();

        // Both segments cover et=50: the later-listed one takes precedence.
        let r = evaluate_chain_position(&chain, 50.0).unwrap();
        assert_abs_diff_eq!(r[0], 2.0, epsilon = 1e-12);

        // Only the earlier segment covers et=150: falls back to it.
        let r = evaluate_chain_position(&chain, 150.0).unwrap();
        assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_resolve_chain_for_epoch_excludes_non_covering_segments() {
        // Direct A rel B [0,100] plus a two-hop A rel C [0,200] / C rel B
        // [0,200] path. At et=150 only the two-hop path covers, so the
        // epoch-aware resolver must route through C rather than reusing
        // the (non-covering) direct link picked by topology-only BFS.
        let direct = constant_segment(10, 0, 0.0, 100.0, 7.0);
        let leg_ac = constant_segment(10, 3, 0.0, 200.0, 2.0);
        let leg_cb = constant_segment(3, 0, 0.0, 200.0, 3.0);
        let segments = [direct, leg_ac, leg_cb];

        // Topology-only resolution always prefers the 1-hop direct link.
        let topo_chain = resolve_chain(&segments, 10, 0).unwrap();
        assert_eq!(topo_chain.len(), 1);

        // et=150: direct link doesn't cover it; epoch-aware resolution
        // must find the two-hop path (2.0 + 3.0 = 5.0), not error.
        let chain = resolve_chain_for_epoch(&segments, 10, 0, 150.0).unwrap();
        assert_eq!(chain.len(), 2);
        let r = evaluate_chain_position(&chain, 150.0).unwrap();
        assert_abs_diff_eq!(r[0], 5.0, epsilon = 1e-12);

        // et=300: no segment (direct or two-hop) covers it; error must
        // name target, center, and the epoch.
        let err = resolve_chain_for_epoch(&segments, 10, 0, 300.0).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("10") && msg.contains("300"));
    }

    #[test]
    fn test_spk_position_falls_back_to_epoch_aware_chain() {
        // End-to-end SPK-level regression for the cached-chain-misses-
        // coverage defect: a direct segment [0,100] plus a two-hop
        // alternative [0,200]/[0,200]. The cached chain (resolved once,
        // topology-only) is the direct link; querying an epoch only the
        // two-hop path covers must transparently fall back rather than
        // erroring, while queries the direct link covers keep using it.
        let direct = constant_segment(10, 0, 0.0, 100.0, 7.0);
        let leg_ac = constant_segment(10, 3, 0.0, 200.0, 2.0);
        let leg_cb = constant_segment(3, 0, 0.0, 200.0, 3.0);
        let spk = SPK {
            segments: vec![direct, leg_ac, leg_cb],
            chain_cache: RwLock::new(HashMap::new()),
        };

        // et=50: covered by the (cached) direct link -> direct value.
        let r = spk.position(10, 0, 50.0).unwrap();
        assert_abs_diff_eq!(r[0], 7.0e3, epsilon = 1e-9);

        // et=150: direct link out of coverage -> falls back to the
        // two-hop path (2.0 + 3.0 = 5.0 km).
        let r = spk.position(10, 0, 150.0).unwrap();
        assert_abs_diff_eq!(r[0], 5.0e3, epsilon = 1e-9);

        // The cached chain itself is untouched by the fallback (still the
        // direct 1-hop link), confirming the fallback isn't cached.
        assert_eq!(
            spk.chain_cache.read().unwrap().get(&(10, 0)).unwrap().len(),
            1
        );

        // et=300: neither path covers -> error names target, center, epoch.
        let err = spk.position(10, 0, 300.0).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("10") && msg.contains("0") && msg.contains("300"));
    }

    /// Build a synthetic single-record type-2 segment covering
    /// `[start_et, end_et]` whose record `RADIUS` is 0, so any evaluation
    /// inside its coverage fails with the invalid-RADIUS `IoError` (a
    /// non-coverage error).
    fn corrupt_radius_segment(
        target: i32,
        center: i32,
        start_et: f64,
        end_et: f64,
    ) -> Arc<ChebyshevSegment> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1);
        let mid = (start_et + end_et) / 2.0;
        Arc::new(ChebyshevSegment {
            target,
            center,
            frame: 1,
            data_type: 2,
            ncomp: 3,
            start_et,
            end_et,
            init: start_et,
            intlen: end_et - start_et,
            rsize,
            n: 1,
            degree,
            coeffs: vec![mid, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
    }

    #[test]
    fn test_spk_corrupt_record_error_propagates_end_to_end() {
        // End-to-end via SPK::position: a corrupt covering segment (record
        // RADIUS=0 -> invalid-RADIUS IoError) must surface its error, with
        // a valid alternate two-hop path also loaded. Note this setup alone
        // does not discriminate the is_coverage_error guard (the corrupt
        // segment covers et, so an unguarded fallback would re-resolve to
        // the same corrupt 1-hop chain and fail identically); the
        // discriminating unit test is
        // `test_fallback_guard_ignores_non_coverage_error_even_when_fallback_would_succeed`.
        let corrupt_direct = corrupt_radius_segment(10, 0, 0.0, 200.0);
        let leg_ac = constant_segment(10, 3, 0.0, 200.0, 2.0);
        let leg_cb = constant_segment(3, 0, 0.0, 200.0, 3.0);
        let spk = SPK {
            segments: vec![corrupt_direct, leg_ac, leg_cb],
            chain_cache: RwLock::new(HashMap::new()),
        };

        // et=50 is inside the corrupt direct segment's coverage: the
        // invalid-RADIUS error is not a coverage miss, so no fallback.
        let err = spk.position(10, 0, 50.0).unwrap_err();
        assert!(!is_coverage_error(&err));
        let msg = format!("{}", err);
        assert!(msg.contains("RADIUS"), "unexpected error: {}", msg);
    }

    #[test]
    fn test_fallback_guard_ignores_non_coverage_error_even_when_fallback_would_succeed() {
        // Discriminating regression test for the is_coverage_error gate in
        // `evaluate_with_epoch_fallback`: the cached chain is built from a
        // corrupt (10,0) segment (record RADIUS=0, covering et), while the
        // injected fallback segment pool contains ONLY a valid (10,0)
        // segment that would evaluate successfully. Under the old
        // unguarded `Err(_)` behavior the fallback would engage, resolve
        // the valid pool, and return Ok -- silently masking the corrupt
        // data. With the guard, the non-coverage RADIUS error propagates.
        let corrupt = corrupt_radius_segment(10, 0, 0.0, 200.0);
        let valid = constant_segment(10, 0, 0.0, 200.0, 5.0);

        let cached_chain = resolve_chain(std::slice::from_ref(&corrupt), 10, 0).unwrap();
        let fallback_pool = vec![Arc::clone(&valid)];

        // (a) Corruption propagates: the fallback must NOT rescue it.
        let err = evaluate_with_epoch_fallback(
            &cached_chain,
            10,
            0,
            50.0,
            || fallback_pool.clone(),
            evaluate_chain_position,
        )
        .unwrap_err();
        assert!(!is_coverage_error(&err));
        let msg = format!("{}", err);
        assert!(msg.contains("RADIUS"), "unexpected error: {}", msg);

        // (b) Prove the setup discriminates: had the fallback engaged, it
        // WOULD have succeeded over this pool at this epoch.
        let rescue_chain = resolve_chain_for_epoch(&fallback_pool, 10, 0, 50.0).unwrap();
        let r = evaluate_chain_position(&rescue_chain, 50.0).unwrap();
        assert_abs_diff_eq!(r[0], 5.0, epsilon = 1e-12);

        // (c) Sanity: a genuine coverage miss on the same cached chain DOES
        // engage the fallback and reach the valid pool (guard lets
        // coverage errors through). et=300 is outside the corrupt chain's
        // [0,200] coverage; use a pool segment covering it to confirm the
        // rescue path.
        let wide_valid = constant_segment(10, 0, 0.0, 400.0, 5.0);
        let wide_pool = vec![wide_valid];
        let r = evaluate_with_epoch_fallback(
            &cached_chain,
            10,
            0,
            300.0,
            || wide_pool.clone(),
            evaluate_chain_position,
        )
        .unwrap();
        assert_abs_diff_eq!(r[0], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_spk_direct_segment_query() {
        let Some(spk) = load_de440s() else { return };
        // Moon rel EMB is a direct segment (301, 3)
        let r = spk.position(301, 3, ET_2025).unwrap();
        let d = r.norm();
        // Earth-Moon distance from EMB: roughly (356e6..407e6) * m_E/(m_E+m_M) share;
        // Moon-to-EMB distance is Moon-Earth minus Earth-EMB, ~ 3.4e8..4.1e8 * 0.9878
        assert!(d > 3.4e8 && d < 4.1e8, "unexpected |r| = {}", d);
    }

    #[test]
    fn test_spk_two_hop_chain() {
        let Some(spk) = load_de440s() else { return };
        // Moon rel Earth = +(301 rel 3) - (399 rel 3)
        let r_moon_earth = spk.position(301, 399, ET_2025).unwrap();
        let expected =
            spk.position(301, 3, ET_2025).unwrap() - spk.position(399, 3, ET_2025).unwrap();
        assert_abs_diff_eq!((r_moon_earth - expected).norm(), 0.0, epsilon = 1.0e-6);
        let d = r_moon_earth.norm();
        assert!(d > 3.5e8 && d < 4.1e8);
    }

    #[test]
    fn test_spk_three_hop_chain_sun_earth() {
        let Some(spk) = load_de440s() else { return };
        // Sun rel Earth = +(10 rel 0) - (3 rel 0) - (399 rel 3)
        let r = spk.position(10, 399, ET_2025).unwrap();
        let d = r.norm();
        // 1 AU ± 2%
        assert!(d > 1.44e11 && d < 1.55e11, "unexpected |r_sun| = {}", d);
    }

    #[test]
    fn test_spk_reverse_direction_is_negation() {
        let Some(spk) = load_de440s() else { return };
        let a = spk.position(399, 10, ET_2025).unwrap();
        let b = spk.position(10, 399, ET_2025).unwrap();
        assert_abs_diff_eq!((a + b).norm(), 0.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_spk_identity_is_zero() {
        let Some(spk) = load_de440s() else { return };
        let r = spk.position(399, 399, ET_2025).unwrap();
        assert_eq!(r.norm(), 0.0);
    }

    #[test]
    fn test_spk_state_consistent_and_units() {
        let Some(spk) = load_de440s() else { return };
        let x = spk.state(301, 399, ET_2025).unwrap();
        let r = spk.position(301, 399, ET_2025).unwrap();
        let v = spk.velocity(301, 399, ET_2025).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-15);
        // Moon orbital speed ~ 1.0e3 m/s (0.95..1.1 km/s)
        let speed = v.norm();
        assert!(speed > 9.0e2 && speed < 1.2e3, "unexpected |v| = {}", speed);
    }

    #[test]
    fn test_spk_velocity_matches_finite_difference() {
        let Some(spk) = load_de440s() else { return };
        let h = 10.0; // seconds
        let v = spk.velocity(10, 399, ET_2025).unwrap();
        let fd = (spk.position(10, 399, ET_2025 + h).unwrap()
            - spk.position(10, 399, ET_2025 - h).unwrap())
            / (2.0 * h);
        // Central difference O(h^2): agreement to ~1e-4 m/s at these scales
        assert!((v - fd).norm() < 1.0e-2, "|v - fd| = {}", (v - fd).norm());
    }

    #[test]
    fn test_spk_no_path_error() {
        let Some(spk) = load_de440s() else { return };
        let err = spk.position(301, 12345, ET_2025).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("301") && msg.contains("12345"));
    }

    #[test]
    fn test_spk_out_of_coverage_error() {
        let Some(spk) = load_de440s() else { return };
        // DE440s covers ~1849..2150; ET far outside
        let err = spk.position(10, 399, 1.0e13).unwrap_err();
        assert!(format!("{}", err).contains("coverage"));
    }

    #[test]
    fn test_spk_chain_cache_reused() {
        let Some(spk) = load_de440s() else { return };
        let _ = spk.position(10, 399, ET_2025).unwrap();
        assert!(spk.chain_cache.read().unwrap().contains_key(&(10, 399)));
        // Second query hits the cache and matches
        let a = spk.position(10, 399, ET_2025).unwrap();
        let b = spk.position(10, 399, ET_2025).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_spk_rejects_pck_file() {
        // A DAF/PCK ID word must be rejected by SPK::from_bytes. The DAF
        // must still be structurally valid (file record + one empty summary
        // record + its name record) so the failure is the SPK-level ID word
        // check, not an earlier DAFFile parse error.
        let mut file = vec![0u8; 3 * 1024];
        file[..8].copy_from_slice(b"DAF/PCK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes());
        file[12..16].copy_from_slice(&5i32.to_le_bytes());
        file[76..80].copy_from_slice(&2i32.to_le_bytes());
        file[80..84].copy_from_slice(&2i32.to_le_bytes());
        file[84..88].copy_from_slice(&100i32.to_le_bytes());
        file[88..96].copy_from_slice(b"LTL-IEEE");
        // Summary record (record 2): NEXT=0, PREV=0, NSUM=0 (no segments).
        let rec = 1024;
        file[rec..rec + 8].copy_from_slice(&0f64.to_le_bytes());
        file[rec + 8..rec + 16].copy_from_slice(&0f64.to_le_bytes());
        file[rec + 16..rec + 24].copy_from_slice(&0f64.to_le_bytes());
        let err = SPK::from_bytes(&file).unwrap_err();
        assert!(format!("{}", err).contains("DAF/PCK"));
    }
}
