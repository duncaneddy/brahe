/*!
 * Native SPK (Spacecraft and Planet Kernel) ephemeris reader.
 *
 * An [`SPK`] holds one parsed kernel with all Chebyshev coefficients
 * resident in memory. Queries resolve a signed chain of segments linking
 * `target` to `center` (e.g. Sun rel. Earth = +Sun/SSB − EMB/SSB −
 * Earth/EMB), cache the chain per pair, and evaluate with O(1) record
 * lookup. Output states are in the kernel's inertial frame (J2000 label,
 * ICRF axes for DE4xx kernels) in SI units.
 */

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, RwLock};

use nalgebra::{Vector3, Vector6};

use crate::utils::BraheError;

use super::daf::DafFile;
use super::segments::ChebyshevSegment;

/// One link in a resolved target→center chain: the candidate segments for a
/// single body pair (in precedence order) and the traversal sign.
#[derive(Debug, Clone)]
pub(crate) struct ChainLink {
    /// Candidate segments for this pair, checked in order for ET coverage.
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
    if target == center {
        return Ok(Vec::new());
    }

    // Adjacency: body -> neighboring bodies. Pair key is (target, center)
    // as stored in segments; candidates kept in input (precedence) order.
    let mut pair_segments: HashMap<(i32, i32), Vec<Arc<ChebyshevSegment>>> = HashMap::new();
    let mut adjacency: HashMap<i32, Vec<i32>> = HashMap::new();
    for seg in segments {
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
        return Err(BraheError::Error(format!(
            "No ephemeris path from target {} to center {} in loaded SPK data",
            target, center
        )));
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
fn covering_segment(link: &ChainLink, et: f64) -> Result<&Arc<ChebyshevSegment>, BraheError> {
    link.segments
        .iter()
        .rev()
        .find(|s| s.covers(et))
        .ok_or_else(|| {
            let s0 = &link.segments[0];
            BraheError::Error(format!(
                "Epoch ET {} outside segment coverage [{}, {}] (target {}, center {})",
                et, s0.start_et, s0.end_et, s0.target, s0.center
            ))
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
        let daf = DafFile::from_file(path)?;
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
        Self::from_daf(DafFile::from_bytes(bytes)?)
    }

    pub(crate) fn from_daf(daf: DafFile) -> Result<Self, BraheError> {
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
        Ok(evaluate_chain_position(&self.chain(target, center)?, et)? * 1.0e3)
    }

    /// Velocity of `target` relative to `center` at ET `et`.
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
        Ok(evaluate_chain_velocity(&self.chain(target, center)?, et)? * 1.0e3)
    }

    /// Position and velocity of `target` relative to `center` at ET `et`.
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
        let (r, v) = evaluate_chain_state(&self.chain(target, center)?, et)?;
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
        // check, not an earlier DafFile parse error.
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
