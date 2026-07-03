/*!
Implement central-body gravity force models.
 */

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use nalgebra::{DMatrix, Vector3};

use crate::math::{traits::IntoPosition, SMatrix3};
use once_cell::sync::Lazy;

use rayon::prelude::*;

use crate::constants::{GM_EARTH, J2_EARTH, J3_EARTH, J4_EARTH, J5_EARTH, J6_EARTH, R_EARTH};
use crate::math::kronecker_delta;
use crate::utils::threading::get_thread_pool;
use crate::utils::BraheError;

/// Packaged EGM2008_360 Data File
static PACKAGED_EGM2008_360: &[u8] = include_bytes!("../../data/gravity_models/EGM2008_360.gfc");

/// Packaged GGM05S Data File
static PACKAGED_GGM05S: &[u8] = include_bytes!("../../data/gravity_models/GGM05S.gfc");

/// Packaged JGM3
static PACKAGED_JGM3: &[u8] = include_bytes!("../../data/gravity_models/JGM3.gfc");
static GLOBAL_GRAVITY_MODEL: Lazy<Arc<RwLock<Box<GravityModel>>>> =
    Lazy::new(|| Arc::new(RwLock::new(Box::new(GravityModel::new()))));

/// Process-wide cache mapping a `GravityModelType` to the most recently loaded
/// `Arc<GravityModel>` for that type. Backs `GravityModel::from_model_type` and
/// `GravityModel::shared` so that repeated requests for the same model avoid
/// the ~60 ms file-parse cost paid on a cold load.
///
/// Hit path: read-lock + `Arc::clone`. Miss path: load from disk under no
/// lock, then double-check + insert under a write lock so concurrent misses
/// for the same type only pay the load cost once.
///
/// Use `clear_gravity_model_cache()` to drop all cached entries — useful
/// after replacing a `FromFile` source on disk, or in tests that want to
/// exercise the cold-load path.
///
// TODO: add an eviction policy. The cache is currently unbounded: each
// distinct `GravityModelType` ever loaded stays resident for the process
// lifetime. For the three packaged variants this is fine (~2 MB total at
// most), but `FromFile(path)` lets long-running programs accumulate one
// entry per unique path. A small LRU would bound memory,
// until then `clear_gravity_model_cache()` is the manual escape hatch.
static GRAVITY_MODEL_CACHE: Lazy<RwLock<HashMap<GravityModelType, Arc<GravityModel>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Drop every entry from the process-wide gravity-model cache.
///
/// The next call to [`GravityModel::from_model_type`] or [`GravityModel::shared`]
/// for any previously cached type will re-read and re-parse the underlying
/// `.gfc` file. Typical uses:
///
/// - tests that need to measure the cold-load path
/// - long-running programs that have swapped a `FromFile(path)` on disk and
///   want subsequent loads to pick up the new contents
/// - any consumer needing a deterministic memory baseline
///
/// # Example
///
/// ```
/// use brahe::gravity::{GravityModel, GravityModelType, clear_gravity_model_cache};
///
/// // First call populates the cache.
/// let _ = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
///
/// // Clear and the next load goes back to disk.
/// clear_gravity_model_cache();
/// let _ = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
/// ```
pub fn clear_gravity_model_cache() {
    GRAVITY_MODEL_CACHE.write().unwrap().clear();
}

/// Set the global gravity model to a new gravity model. A global gravity model is useful as it
/// allows for a single gravity model to be used throughout a program. This is useful when multiple
/// objects are being propagated, as it allows for the gravity model to only be loaded into memory
/// once, reducing memory usage and improving performance.
///
/// # Arguments
///
/// - `gravity_model` : New gravity model to set as the global gravity model.
///
/// # Examples
///
/// ```
/// use brahe::gravity::{GravityModel, set_global_gravity_model, GravityModelType};
///
/// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
/// set_global_gravity_model(gravity_model);
/// ```
pub fn set_global_gravity_model(gravity_model: GravityModel) {
    **GLOBAL_GRAVITY_MODEL.write().unwrap() = gravity_model;
}

/// Get the global gravity model.
///
/// # Returns
///
/// - `GravityModel`: Gravity model object.
///
/// # Examples
///
/// ```
/// use brahe::gravity::{GravityModel, set_global_gravity_model, get_global_gravity_model, GravityModelType};
///
/// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
/// set_global_gravity_model(gravity_model);
///
/// let model = get_global_gravity_model();
///
/// assert_eq!(model.model_name, "EGM2008");
/// ```
pub fn get_global_gravity_model() -> RwLockReadGuard<'static, Box<GravityModel>> {
    GLOBAL_GRAVITY_MODEL.read().unwrap()
}

/// Helper function to aid in denormalization of gravity field coefficients.
/// This method computes the factorial ratio (n-m)!/(n+m)! in an efficient
/// manner, without computing the full factorial products.
fn factorial_product(n: usize, m: usize) -> f64 {
    let mut p = 1.0;

    // TODO: Confirm range termination of n+m+1 vs n+m
    for i in n - m + 1..n + m + 1 {
        p /= i as f64;
    }

    p
}

/// Compute the acceleration due to point-mass gravity.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_object`.
/// When a state vector is provided, only the position component is used.
///
/// # Arguments
///
/// - `r_object`: Position vector of the object, or state vector (position + velocity).
/// - `r_central_body`: Position vector of the central body. If the central body is at the origin, this is the zero vector.
/// - `gm`: Product of the gravitational parameter and the mass of the central body.
///
/// # Returns
///
/// - `a_grav` : Acceleration due to gravity of the central body.
///
/// # Examples
///
/// Using a position vector:
/// ```
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbit_dynamics::accel_point_mass_gravity;
/// use nalgebra::Vector3;
///
/// let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
/// let r_central_body = Vector3::new(0.0, 0.0, 0.0);
///
/// let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);
///
/// // Acceleration should be in the negative x-direction and magnitude should be GM_EARTH / R_EARTH^2
/// // Roughly -9.81 m/s^2
/// assert!((a_grav - Vector3::new(-GM_EARTH / R_EARTH.powi(2), 0.0, 0.0)).norm() < 1e-12);
/// ```
///
/// Using a state vector:
/// ```
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbit_dynamics::accel_point_mass_gravity;
/// use nalgebra::{SVector, Vector3};
///
/// let x_object = SVector::<f64, 6>::new(R_EARTH, 0.0, 0.0, 0.0, 7500.0, 0.0);
/// let r_central_body = Vector3::new(0.0, 0.0, 0.0);
///
/// let a_grav = accel_point_mass_gravity(x_object, r_central_body, GM_EARTH);
///
/// // Acceleration should be in the negative x-direction
/// assert!((a_grav - Vector3::new(-GM_EARTH / R_EARTH.powi(2), 0.0, 0.0)).norm() < 1e-12);
/// ```
///
/// # References
///
/// - TODO: Add references
pub fn accel_point_mass_gravity<P: IntoPosition>(
    r_object: P,
    r_central_body: Vector3<f64>,
    gm: f64,
) -> Vector3<f64> {
    let r_obj = r_object.position();
    let d = r_obj - r_central_body;

    let d_norm = d.norm();
    let r_central_body_norm = r_central_body.norm();

    if r_central_body_norm != 0.0 {
        -gm * (d / d_norm.powi(3) + r_central_body / r_central_body_norm.powi(3))
    } else {
        -gm * d / d_norm.powi(3)
    }
}

/// Compute the gravitational acceleration due to Earth's zonal harmonic terms.
///
/// Evaluates the closed-form perturbation acceleration arising from the zonal
/// (axially-symmetric, order-zero) coefficients J₂ through J₆ of Earth's gravity
/// field, summed onto the central two-body acceleration. The position is assumed
/// to already be expressed in an Earth-fixed frame whose z-axis is aligned with
/// the rotation pole; callers responsible for that ECI→ECEF rotation.
///
/// This routine is a hand-rolled, branch-on-degree implementation that mirrors
/// the result of [`accel_gravity_spherical_harmonics`] with `m = 0`, but evaluates
/// each term as an explicit polynomial in `z/r` instead of via Legendre-polynomial
/// recursion. Benchmarks show ~1.5–2x faster evaluation than the general
/// spherical-harmonic routine for the same axially-symmetric expansion, with
/// agreement to <3 km over a 24 h LEO propagation when both routines are driven
/// in the same Earth-fixed frame.
///
/// # Arguments
///
/// - `r_object`: Earth-fixed position vector of the object (or 6D state vector;
///   only the position component is used). The frame must have z aligned with
///   Earth's rotation axis.
/// - `n`: Maximum zonal degree to include (clamped to the supported range
///   `2..=6`). For `n < 2` only the two-body term is returned.
///
/// # Returns
///
/// - `a_grav`: Acceleration vector in the same frame as `r_object`.
///
/// # Examples
///
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::orbit_dynamics::accel_earth_zonal_gravity;
/// use nalgebra::Vector3;
///
/// // Equatorial position in an Earth-fixed frame.
/// let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
///
/// // Acceleration including the J2 zonal term.
/// let a_grav = accel_earth_zonal_gravity(r_object, 2);
///
/// // Magnitude is dominated by the two-body pull (~9.8 m/s²) with a small J2 perturbation.
/// assert!((a_grav.norm() - 9.8).abs() < 1e-1);
/// ```
///
/// # References
///
/// - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., pp. 593.
pub fn accel_earth_zonal_gravity<P: IntoPosition>(r_object: P, n: usize) -> Vector3<f64> {
    let pos = r_object.position();

    let i = pos[0];
    let j = pos[1];
    let k = pos[2];

    let r = r_object.position().norm();

    let k_r = k / r;
    let k_r2 = k_r * k_r;
    let k_r4 = k_r2 * k_r2;
    let k_r6 = k_r4 * k_r2;

    // Two-body acceleration
    let mut accel = accel_point_mass_gravity(r_object, Vector3::zeros(), GM_EARTH);

    if n < 2 {
        return accel;
    }

    // Explicit math based on [Vallado, p.593ff], leave optimization to compiler
    if n >= 2 {
        // Shared term for J2: (-3 * J2 * GM *  (R_e / r)^2 ) / 2
        let j2_coeff = (-3. * J2_EARTH * GM_EARTH * R_EARTH.powi(2)) / (2. * r.powi(5));

        let j2_xy_factor = 1.0 - 5.0 * k_r2;
        let j2_z_factor = 3.0 - 5.0 * k_r2;

        accel.x += j2_coeff * j2_xy_factor * i;
        accel.y += j2_coeff * j2_xy_factor * j;
        accel.z += j2_coeff * j2_z_factor * k;
    }

    if n >= 3 {
        let j3_coeff = (-5. * J3_EARTH * GM_EARTH * R_EARTH.powi(3)) / (2. * r.powi(7));

        let j3_xy_factor = 3. * k - (7. * k * k_r2);
        let j3_z_factor = 6. * k.powi(2) - (7. * k.powi(2) * k_r2) - (3. / 5. * r.powi(2));

        accel.x += j3_coeff * j3_xy_factor * i;
        accel.y += j3_coeff * j3_xy_factor * j;
        accel.z += j3_coeff * j3_z_factor;
    }

    if n >= 4 {
        let j4_coeff = (15. * J4_EARTH * GM_EARTH * R_EARTH.powi(4)) / (8. * r.powi(7));

        let j4_xy_factor = 1. - (14. * k_r2) + (21. * k_r4);
        let j4_z_factor = 5. - (70. / 3. * k_r2) + (21. * k_r4);
        accel.x += j4_coeff * j4_xy_factor * i;
        accel.y += j4_coeff * j4_xy_factor * j;
        accel.z += j4_coeff * j4_z_factor * k;
    }

    if n >= 5 {
        let re_5 = R_EARTH.powi(5);
        let j5_coeff = (3. * J5_EARTH * GM_EARTH * re_5) / (8. * r.powi(9));

        let j5_xy_factor = 35. - (210. * k_r2) + (231. * k_r4);
        let j5_z_factor = 105. - (315. * k_r2) + (231. * k_r4);
        accel.x += j5_coeff * j5_xy_factor * i * k;
        accel.y += j5_coeff * j5_xy_factor * j * k;
        accel.z += j5_coeff * j5_z_factor * k.powi(2)
            - (15. * J5_EARTH * GM_EARTH * re_5 / (8. * r.powi(7)));
    }

    if n >= 6 {
        let j6_coeff = (-J6_EARTH * GM_EARTH * R_EARTH.powi(6)) / (16. * r.powi(9));

        let j6_xy_factor = 35. - (945. * k_r2) + (3465. * k_r4) - (3003. * k_r6);
        let j6_z_factor = 245. - (2205. * k_r2) + (4851. * k_r4) - (3003. * k_r6);

        accel.x += j6_coeff * j6_xy_factor * i;
        accel.y += j6_coeff * j6_xy_factor * j;
        accel.z += j6_coeff * j6_z_factor * k;
    }

    accel
}

/// Enumeration of the tide system used in a gravity model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GravityModelTideSystem {
    /// Zero-tide system: includes permanent tidal deformation from Sun and Moon.
    /// C₂₀ coefficient includes indirect effect of Earth's centrifugal potential.
    ZeroTide,
    /// Tide-free system: permanent tidal effects removed. Most commonly used in
    /// modern gravity models. Represents Earth's shape without tidal deformation.
    TideFree,
    /// Mean-tide system: includes time-averaged permanent tide. Historical convention
    /// used in older gravity models and some geodetic applications.
    MeanTide,
    /// Unknown or unspecified tide system. Default value when tide system is not declared.
    Unknown,
}

/// Enumeration of the error handling used in a gravity model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GravityModelErrors {
    /// No error estimates provided with gravity coefficients. Model provides only
    /// nominal C and S coefficient values without uncertainty information.
    No,
    /// Calibrated error estimates: empirically derived uncertainties based on
    /// comparison with independent data sources or solution comparisons.
    Calibrated,
    /// Formal errors: statistical uncertainties from least-squares adjustment or
    /// estimation process. May underestimate true uncertainties.
    Formal,
    /// Both calibrated and formal error estimates provided, allowing comparison
    /// between statistical and empirical uncertainty assessments.
    CalibratedAndFormal,
}

/// Enumeration of the normalization used in a gravity model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GravityModelNormalization {
    /// Fully normalized spherical harmonics (4π normalization). Standard in modern
    /// gravity models (EGM2008, GRACE, etc.). Coefficients have similar magnitudes.
    FullyNormalized,
    /// Unnormalized spherical harmonics. Used in older models and some theoretical
    /// applications. Coefficients decrease rapidly with increasing degree/order.
    Unnormalized,
}

/// Execution policy for the spherical-harmonic acceleration computation.
///
/// Controls whether [`GravityModel::compute_spherical_harmonics_cunningham_with_workspace`]
/// and its callers parallelize the recurrence column-fill and the acceleration
/// accumulation across Brahe's managed thread pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum ParallelMode {
    /// Parallelize only when `n_max >= PARALLEL_THRESHOLD_NMAX`. Below that, the
    /// rayon dispatch overhead outweighs the gain, so the serial path is used.
    #[default]
    Auto,
    /// Always parallelize (via the configured global thread pool).
    Always,
    /// Always run serially.
    Never,
}

/// Benchmark-calibrated crossover degree. At or above this `n_max`, parallel
/// evaluation beats serial on a typical multi-core host. Machine-approximate;
/// see `benchmarks/gravity_benchmarks.rs`.
///
/// Measured on a 10-core Apple M1 Max after the serial-path optimization
/// (precomputed reciprocal recurrence coefficients + bounds-check-free column
/// slices in the recurrence and accumulation). Serial vs parallel median times:
///
/// | n_max | serial    | parallel  |
/// |-------|-----------|-----------|
/// |     2 |   114 ns  |  27.6 µs  |
/// |    20 |  1.51 µs  |  38.4 µs  |
/// |    50 |  8.75 µs  |  55.3 µs  |
/// |    90 |  32.6 µs  |   107 µs  |
/// |   120 |  59.7 µs  |   127 µs  |
/// |   180 |   121 µs  |   153 µs  |
/// |   210 |   156 µs  |   154 µs  |
/// |   240 |   212 µs  |   177 µs  |
/// |   360 |   495 µs  |   310 µs  |
///
/// The serial path is now ~1.6× faster than before, which pushed the crossover
/// out: parallel breaks even at n≈210 and wins reliably from n=240 (1.20×).
/// Set to 210 — the break-even point — so parallel only engages where it is at
/// least as fast as serial. This value is machine-approximate.
pub(crate) const PARALLEL_THRESHOLD_NMAX: usize = 210;

/// Decide whether to run the spherical-harmonic computation in parallel.
///
/// `Auto` parallelizes only for large expansions (`n_max >= PARALLEL_THRESHOLD_NMAX`)
/// AND only when not already executing on a rayon worker thread. The latter check
/// prevents nested parallelism: batch propagation (`par_propagate_to_*`) already
/// saturates the managed thread pool with one propagation per worker, so an inner
/// `install` per gravity eval would only add split/reduce overhead. `Always` still
/// forces parallelism (an explicit opt-in escape hatch); `Never` always serial.
pub(crate) fn should_parallelize(mode: ParallelMode, n_max: usize) -> bool {
    match mode {
        ParallelMode::Always => true,
        ParallelMode::Never => false,
        ParallelMode::Auto => {
            n_max >= PARALLEL_THRESHOLD_NMAX && rayon::current_thread_index().is_none()
        }
    }
}

/// Benchmark-calibrated crossover degree for the Clenshaw kernel. At or above
/// this `n_max`, parallel evaluation of the Clenshaw kernel beats serial on a
/// typical multi-core host. Machine-approximate; see
/// `benchmarks/gravity_benchmarks.rs`.
///
/// Measured on a 10-core Apple M1 Max. Serial vs parallel median times for
/// the Clenshaw kernel, alongside serial Cunningham for reference (Cunningham
/// overflows above n=120 at this benchmark's altitude/latitude, so no
/// Cunningham entries above that degree):
///
/// | n_max | cunningham serial | clenshaw serial | clenshaw parallel |
/// |-------|--------------------|------------------|--------------------|
/// |     2 |            50.6 ns |          45.8 ns |            11.1 µs |
/// |    20 |           755.5 ns |         787.1 ns |            16.5 µs |
/// |    50 |            5.37 µs |          4.15 µs |            21.9 µs |
/// |    90 |            19.5 µs |          12.9 µs |            24.2 µs |
/// |   120 |            35.1 µs |          22.6 µs |            23.2 µs |
/// |   180 |                  — |          50.2 µs |            29.9 µs |
/// |   240 |                  — |          88.7 µs |            48.9 µs |
/// |   360 |                  — |           198 µs |            83.4 µs |
///
/// The Clenshaw serial path is consistently faster than serial Cunningham
/// where both are valid (1.30–1.55× at n=50–120, and the recurrence's flatter
/// per-degree cost means the gap keeps widening past Cunningham's overflow
/// ceiling). Serial vs parallel for Clenshaw itself: at n=120 serial is still
/// (barely) ahead of parallel (22.6 µs vs 23.2 µs); at n=180 parallel wins
/// decisively (29.9 µs vs 50.2 µs) and continues to win at 240/360. The true
/// crossover lies between 120 and 180 with no benchmarked point in between;
/// set to 180 — the first measured size where parallel is not slower than
/// serial — so `Auto` only parallelizes once the win is unambiguous. This
/// value is machine-approximate.
pub(crate) const CLENSHAW_PARALLEL_THRESHOLD_NMAX: usize = 180;

/// Clenshaw-kernel counterpart of [`should_parallelize`]: same `Always` /
/// `Never` semantics and the same nested-parallelism guard, but with the
/// Clenshaw-specific auto threshold.
pub(crate) fn should_parallelize_clenshaw(mode: ParallelMode, n_max: usize) -> bool {
    match mode {
        ParallelMode::Always => true,
        ParallelMode::Never => false,
        ParallelMode::Auto => {
            n_max >= CLENSHAW_PARALLEL_THRESHOLD_NMAX && rayon::current_thread_index().is_none()
        }
    }
}

/// Overflow-guard scale applied to every coefficient as it enters the
/// Clenshaw sweep and removed in the final combine. Equal to 2^-614 —
/// GeographicLib's `SphericalEngine::scale()` for IEEE binary64 — which
/// keeps the backward-recurrence accumulators far from overflow at very
/// high degree without ever denormalizing the coefficients.
const CLENSHAW_SCALE: f64 = f64::from_bits(0x1990_0000_0000_0000); // 2^-614

/// Type of spherical harmonic gravity model
///
/// Specifies which gravity model to load and use for orbit propagation.
/// Models can either be packaged with Brahe or loaded from external files.
#[derive(Debug, PartialEq, Eq, Hash, Clone, serde::Serialize, serde::Deserialize)]
pub enum GravityModelType {
    /// Earth Gravitational Model 2008, truncated to degree/order 360. High-accuracy
    /// global model developed by NGA. Best for precision orbit determination.
    EGM2008_360,
    /// Goddard Earth Model from GRACE mission, degree/order 180. Derived from
    /// satellite gravity measurements. Good balance of accuracy and computation speed.
    GGM05S,
    /// Joint Gravity Model 3, degree/order 70. Legacy model from 1990s. Included
    /// for compatibility and applications not requiring modern accuracy.
    JGM3,
    /// Load gravity model from custom file
    ///
    /// Allows using custom gravity models. File must be in standard GFC format.
    FromFile(String),
    /// Load a gravity model from ICGEM by body + model name.
    ///
    /// `name` is either an exact ICGEM model name (auto-resolves to the largest
    /// available degree variant) or a `name-DEGREE` suffix to pick a specific
    /// variant. The model is downloaded on first use and cached under
    /// `$BRAHE_CACHE/icgem/models/<body>/<name>-<degree>.gfc`.
    ICGEMModel {
        /// Celestial body whose gravity model to load.
        body: crate::datasets::icgem::ICGEMBody,
        /// ICGEM model name (e.g. `"JGM3"`) or `"name-DEGREE"` for a specific variant.
        name: String,
    },
}

impl GravityModelType {
    /// Create a GravityModelType from a file path, validating the file exists.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the gravity model file in GFC format
    ///
    /// # Returns
    ///
    /// * `Ok(GravityModelType)` - If the file exists
    /// * `Err(BraheError)` - If the file does not exist or is not a file
    pub fn from_file<P: AsRef<Path>>(filepath: P) -> Result<Self, BraheError> {
        let path = filepath.as_ref();
        if !path.exists() {
            return Err(BraheError::IoError(format!(
                "Gravity model file not found: {}",
                path.display()
            )));
        }
        if !path.is_file() {
            return Err(BraheError::IoError(format!(
                "Gravity model path is not a file: {}",
                path.display()
            )));
        }
        Ok(GravityModelType::FromFile(
            path.to_string_lossy().to_string(),
        ))
    }
}

/// Selects which precomputed evaluation tables a gravity model builds at load.
///
/// The Clenshaw and Cunningham kernels require different precomputed values
/// (normalized packed coefficients vs denormalized dense matrices), so each
/// kernel can only run when its table set is present. The default is
/// `Clenshaw` — the main evaluation APIs use the Clenshaw kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GravityTables {
    /// Clenshaw kernel tables only (default).
    #[default]
    Clenshaw,
    /// Cunningham V/W kernel tables only.
    Cunningham,
    /// Both table sets (needed to call both kernels on one model).
    Both,
}

/// Precomputed tables for the Clenshaw-summation kernel
/// ([`GravityModel::compute_spherical_harmonics_clenshaw`]).
///
/// Coefficients are stored **fully normalized** (no denormalization, so there
/// is no high-degree overflow ceiling) in GeographicLib's triangular packing:
/// column-major by order `m`, degree `n = m..=n_max` contiguous within a
/// column. The only auxiliary table is the O(N) square-root table; the
/// Legendre recurrence factors are computed on the fly from it, which keeps
/// the per-evaluation memory traffic to one pass over the coefficients.
///
/// Reference: Holmes & Featherstone (2002); GeographicLib v1.52
/// `SphericalEngine.cpp`.
#[derive(Clone)]
pub(crate) struct ClenshawTables {
    /// Storage-layout degree (the model `n_max` at build time). Governs the
    /// packing stride; requests truncated below it read the same layout.
    n_stride: usize,
    /// Fully-normalized cosine coefficients, `c[index(n, m)]`.
    c: Vec<f64>,
    /// Fully-normalized sine coefficients. The `m = 0` column is omitted
    /// (sine terms vanish): `s[index(n, m) - (n_stride + 1)]` for `m >= 1`.
    s: Vec<f64>,
    /// `sqrt_table[k] = sqrt(k)` for `k = 0..=max(2 * n_stride + 5, 15)`.
    sqrt_table: Vec<f64>,
}

impl ClenshawTables {
    /// One-dimensional index of coefficient `(n, m)` in the triangular
    /// packing: `m * n_stride - m * (m - 1) / 2 + n`, computed in an
    /// underflow-safe form.
    #[inline]
    fn index(&self, n: usize, m: usize) -> usize {
        m * (2 * self.n_stride - m + 1) / 2 + n
    }
}

/// Per-order partial sums produced by one inner Clenshaw sweep over degree.
///
/// `wc`/`ws` are the value sums `Sc[m]`/`Ss[m]` of Holmes & Featherstone;
/// `wrc`/`wrs` the radial-derivative sums; `wtc`/`wts` the colatitude
/// (theta) derivative sums **including** the `m * (t/u) * Sc[m]` diagonal
/// terms, so the outer combine consumes them directly.
#[derive(Clone, Copy, Default)]
struct PerOrderSums {
    wc: f64,
    ws: f64,
    wrc: f64,
    wrs: f64,
    wtc: f64,
    wts: f64,
}

/// One inner Clenshaw sweep for order `m`: a single backward pass over
/// degree `n = n_max..=m` touching each packed coefficient exactly once.
/// Port of the inner loop of GeographicLib v1.52
/// `SphericalEngine::Value<true, FULL, 1>` (SphericalEngine.cpp:191–229),
/// with the sectoral `P'[m,m]` terms folded in at the end so per-order
/// results are self-contained.
///
/// Geometry inputs: `t = cos(theta)`, `u = sin(theta)` (pole-guarded),
/// `q = radius / r`, `q2 = q^2`, `tu = t / u`.
#[allow(clippy::too_many_arguments)]
fn clenshaw_inner_sweep(
    tables: &ClenshawTables,
    m: usize,
    n_max: usize,
    t: f64,
    u: f64,
    q: f64,
    q2: f64,
    tu: f64,
) -> PerOrderSums {
    let root = &tables.sqrt_table;
    let (mut wc, mut wc2, mut ws, mut ws2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut wrc, mut wrc2, mut wrs, mut wrs2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut wtc, mut wtc2, mut wts, mut wts2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let s_offset = tables.n_stride + 1;

    let mut k = tables.index(n_max, m) + 1;
    for n in (m..=n_max).rev() {
        // Fully-normalized recurrence factors (Holmes & Featherstone Eq. 11)
        // built from the cache-resident sqrt table:
        //   alpha[l]   = t * q * sqrt((2n+1)(2n+3) / ((n-m+1)(n+m+1)))
        //   beta[l+1]  = -q^2 * sqrt((n-m+1)(n+m+1)(2n+5) / ((n-m+2)(n+m+2)(2n+1)))
        let w = root[2 * n + 1] / (root[n - m + 1] * root[n + m + 1]);
        let ax = q * w * root[2 * n + 3];
        let a = t * ax;
        let b = -q2 * root[2 * n + 5] / (w * root[n - m + 2] * root[n + m + 2]);

        k -= 1;
        let rc = tables.c[k] * CLENSHAW_SCALE;
        let next = a * wc + b * wc2 + rc;
        wc2 = wc;
        wc = next;
        // Radial derivative: coefficient weighted by (n + 1).
        let next = a * wrc + b * wrc2 + (n as f64 + 1.0) * rc;
        wrc2 = wrc;
        wrc = next;
        // Theta derivative: d(alpha)/d(theta) = -u * ax, applied to the
        // just-shifted value accumulator (wc2 now holds y[n + 1]).
        let next = a * wtc + b * wtc2 - u * ax * wc2;
        wtc2 = wtc;
        wtc = next;

        if m > 0 {
            let rs = tables.s[k - s_offset] * CLENSHAW_SCALE;
            let next = a * ws + b * ws2 + rs;
            ws2 = ws;
            ws = next;
            let next = a * wrs + b * wrs2 + (n as f64 + 1.0) * rs;
            wrs2 = wrs;
            wrs = next;
            let next = a * wts + b * wts2 - u * ax * ws2;
            wts2 = wts;
            wts = next;
        }
    }

    // Fold in Sc[m] * P'[m,m] / P[m,m] = m * (t/u) * Sc[m] (zero for m = 0).
    let mf = m as f64;
    PerOrderSums {
        wc,
        ws,
        wrc,
        wrs,
        wtc: wtc + mf * tu * wc,
        wts: wts + mf * tu * ws,
    }
}

/// The `GravityModel` struct is for storing spherical harmonic gravity models.
///
/// # Fields
///
/// - `data` : Matrix of gravity model coefficients. The matrix is (n_max + 1) x (m_max + 1) in size.
/// - `tide_system` : Tide system used in the gravity model.
/// - `n_max` : Maximum degree of the gravity model.
/// - `m_max` : Maximum order of the gravity model.
/// - `gm` : Gravitational parameter of the central body. Units are m^3/s^2.
/// - `radius` : Radius of the central body. Units are meters.
/// - `model_name` : Name of the gravity model.
/// - `model_errors` : Error handling used in the gravity model.
/// - `normalization` : Normalization used in the gravity model.
#[derive(Clone)]
pub struct GravityModel {
    data: DMatrix<f64>,
    /// Tide system convention used in the model (zero-tide, tide-free, or mean-tide).
    pub tide_system: GravityModelTideSystem,
    /// Maximum degree (n) of spherical harmonic expansion. Higher degrees capture
    /// finer spatial resolution of gravity field. Typical values: 70-2190.
    pub n_max: usize,
    /// Maximum order (m) of spherical harmonic expansion. Often equal to n_max.
    /// Determines resolution in longitude direction.
    pub m_max: usize,
    /// Gravitational parameter (GM) of the central body. Product of gravitational
    /// constant and body mass. Units: m³/s².
    pub gm: f64,
    /// Reference radius of the central body used for normalization of spherical harmonics.
    /// For Earth models, typically the equatorial radius. Units: meters.
    pub radius: f64,
    /// Human-readable name of the gravity model (e.g., "EGM2008", "GGM05S").
    pub model_name: String,
    /// Type of error/uncertainty estimates included with the model coefficients.
    pub model_errors: GravityModelErrors,
    /// Normalization convention used for spherical harmonic coefficients (fully normalized or unnormalized).
    pub normalization: GravityModelNormalization,
    /// Cunningham V/W kernel tables; `None` until precomputed.
    cunningham: Option<CunninghamTables>,
    /// Clenshaw kernel tables; `None` until precomputed.
    clenshaw: Option<ClenshawTables>,
}

/// Precomputed tables for the Cunningham V/W kernel
/// ([`GravityModel::compute_spherical_harmonics_cunningham`]).
///
/// Holds denormalized coefficients and recurrence multipliers sized to the
/// model's `n_max`/`m_max`. Present only when the model was loaded with
/// `GravityTables::Cunningham`/`Both` or after
/// [`GravityModel::precompute_cunningham_tables`].
#[derive(Clone)]
pub(crate) struct CunninghamTables {
    /// Denormalized cosine coefficients `C_{n,m}` indexed as `coeff_c[(n, m)]`.
    /// Precomputed from `data` and `normalization` to hoist the per-call sqrt
    /// and factorial work out of the spherical harmonic acceleration loop.
    /// Rebuilt by `precompute_cunningham_tables` whenever `data`, `n_max`, `m_max`,
    /// or `normalization` change.
    coeff_c: DMatrix<f64>,
    /// Denormalized sine coefficients `S_{n,m}` indexed as `coeff_s[(n, m)]`.
    /// Entries with `m == 0` are unused (sine terms vanish for zonal harmonics).
    /// See `coeff_c` for invalidation rules.
    coeff_s: DMatrix<f64>,
    /// Precomputed degree/order factor `0.5 * (n - m + 1) * (n - m + 2)` used
    /// by the tesseral/sectoral branch of the acceleration recursion. Only
    /// entries with `m >= 1` are populated and read. See `coeff_c` for
    /// invalidation rules.
    fac: DMatrix<f64>,
    /// Precomputed V/W recurrence multiplier `(2n - 1) / (n - m)` indexed as
    /// `rec_a[(n, m)]`. Hoists the integer→float casts and — critically — the
    /// floating-point division out of the column-fill inner loop, which sits on
    /// a serial dependency chain (`V[n]` needs `V[n-1]`, `V[n-2]`) where the
    /// division latency cannot be hidden. The same coefficients serve the zonal
    /// column (m = 0, where `(n+m-1) = n-1` and `(n-m) = n`) and the tesseral
    /// columns. Sized `(n_max + 2) × (n_max + 2)` to cover the recurrence's
    /// `n_max + 1` top row. Only entries with `n > m` are populated/read.
    /// See `coeff_c` for invalidation rules.
    rec_a: DMatrix<f64>,
    /// Precomputed V/W recurrence multiplier `(n + m - 1) / (n - m)` indexed as
    /// `rec_b[(n, m)]`. Companion to `rec_a`; see it for details.
    rec_b: DMatrix<f64>,
}

impl GravityModel {
    /// Create a new gravity model. This is an internal method used to aid in initializing the
    /// global gravity model. It should not be used directly.
    fn new() -> Self {
        Self {
            data: DMatrix::zeros(1, 1),
            tide_system: GravityModelTideSystem::Unknown,
            n_max: 0,
            m_max: 0,
            gm: 0.0,
            radius: 0.0,
            model_name: String::from("Unknown"),
            model_errors: GravityModelErrors::No,
            normalization: GravityModelNormalization::FullyNormalized,
            cunningham: None,
            clenshaw: None,
        }
    }

    /// Precompute denormalized `C_{n,m}`, `S_{n,m}`, and degree/order factor
    /// matrices used by the Cunningham V/W kernel's acceleration recursion.
    ///
    /// For fully-normalized models this applies the denormalization
    /// factor `sqrt((2 - δ_{0,m}) * (2n + 1) * (n - m)! / (n + m)!)` once
    /// per coefficient so the hot propagation loop in
    /// `compute_spherical_harmonics_cunningham` can read the denormalized
    /// values directly. The `fac` matrix caches the recursion term
    /// `0.5 * (n - m + 1) * (n - m + 2)`.
    ///
    /// Sizes the output matrices to `(n_max + 1) × (n_max + 1)` against the
    /// model's current `data`, `n_max`, `m_max`, and `normalization`.
    fn build_cunningham_tables(&self) -> CunninghamTables {
        let size = self.n_max + 1;
        let mut coeff_c = DMatrix::zeros(size, size);
        let mut coeff_s = DMatrix::zeros(size, size);
        let mut fac = DMatrix::zeros(size, size);

        for n in 0..=self.n_max {
            let nf = n as f64;
            // m = 0
            let c_raw = self.data[(n, 0)];
            coeff_c[(n, 0)] = if self.normalization == GravityModelNormalization::FullyNormalized {
                (2.0 * nf + 1.0).sqrt() * c_raw
            } else {
                c_raw
            };

            // m > 0
            for m in 1..=n.min(self.m_max) {
                let mf = m as f64;
                let c_raw = self.data[(n, m)];
                let s_raw = self.data[(m - 1, n)];
                let (c, s) = if self.normalization == GravityModelNormalization::FullyNormalized {
                    let norm = ((2 - kronecker_delta(0, m)) as f64
                        * (2.0 * nf + 1.0)
                        * factorial_product(n, m))
                    .sqrt();
                    (norm * c_raw, norm * s_raw)
                } else {
                    (c_raw, s_raw)
                };
                coeff_c[(n, m)] = c;
                coeff_s[(n, m)] = s;
                fac[(n, m)] = 0.5 * (nf - mf + 1.0) * (nf - mf + 2.0);
            }
        }

        // V/W recurrence multipliers. These depend only on the integer indices
        // (n, m), not on position or coefficients, so hoisting them here removes
        // a float division and two int→float casts from every inner-loop cell.
        // Size to (n_max + 2) so the recurrence's top row (index n_max + 1) is
        // covered; only entries with n > m are ever read.
        let rsize = self.n_max + 2;
        let mut rec_a = DMatrix::zeros(rsize, rsize);
        let mut rec_b = DMatrix::zeros(rsize, rsize);
        for m in 0..rsize {
            for n in (m + 1)..rsize {
                let nf = n as f64;
                let mf = m as f64;
                let inv = 1.0 / (nf - mf);
                rec_a[(n, m)] = (2.0 * nf - 1.0) * inv;
                rec_b[(n, m)] = (nf + mf - 1.0) * inv;
            }
        }

        CunninghamTables {
            coeff_c,
            coeff_s,
            fac,
            rec_a,
            rec_b,
        }
    }

    /// Precompute the Cunningham V/W kernel tables (denormalized `C_{n,m}`,
    /// `S_{n,m}`, and recurrence multipliers) used by
    /// [`GravityModel::compute_spherical_harmonics_cunningham`].
    ///
    /// No-op if the tables are already present. Loading a model via
    /// `GravityTables::Cunningham`/`Both` calls this automatically; call it
    /// directly if the model was loaded without those tables and you need to
    /// use the Cunningham kernel.
    ///
    /// # Examples
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let mut gravity_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
    /// gravity_model.precompute_cunningham_tables();
    /// ```
    pub fn precompute_cunningham_tables(&mut self) {
        if self.cunningham.is_none() {
            self.cunningham = Some(self.build_cunningham_tables());
        }
    }

    fn build_clenshaw_tables(&self) -> ClenshawTables {
        let n_stride = self.n_max;
        // index(n_max, m_max) + 1 entries; the m = 0 column has n_max + 1.
        let c_len = self.m_max * (2 * n_stride - self.m_max + 1) / 2 + self.n_max + 1;
        let s_len = c_len.saturating_sub(n_stride + 1);
        let mut c = vec![0.0; c_len];
        let mut s = vec![0.0; s_len];

        for m in 0..=self.m_max {
            let col0 = m * (2 * n_stride - m + 1) / 2;
            for n in m..=self.n_max {
                let idx = col0 + n;
                let c_raw = self.data[(n, m)];
                let s_raw = if m > 0 { self.data[(m - 1, n)] } else { 0.0 };
                let (c_norm, s_norm) = match self.normalization {
                    GravityModelNormalization::FullyNormalized => (c_raw, s_raw),
                    GravityModelNormalization::Unnormalized => {
                        // Invert the standard denormalization factor. Only
                        // valid to moderate degree (the factor overflows f64
                        // near n ~ 170); unnormalized models are low-degree
                        // in practice.
                        let nf = n as f64;
                        let f = ((2 - kronecker_delta(0, m)) as f64
                            * (2.0 * nf + 1.0)
                            * factorial_product(n, m))
                        .sqrt();
                        (c_raw / f, s_raw / f)
                    }
                };
                c[idx] = c_norm;
                if m > 0 {
                    s[idx - (n_stride + 1)] = s_norm;
                }
            }
        }

        let sqrt_len = (2 * n_stride + 5).max(15) + 1;
        let sqrt_table = (0..sqrt_len).map(|k| (k as f64).sqrt()).collect();

        ClenshawTables {
            n_stride,
            c,
            s,
            sqrt_table,
        }
    }

    /// Precompute the Clenshaw-summation kernel tables (fully-normalized
    /// coefficients in triangular packing + square-root table) used by
    /// [`GravityModel::compute_spherical_harmonics_clenshaw`].
    ///
    /// No-op if the tables are already present. Loading a model calls this
    /// automatically; call it directly if you need to use the Clenshaw kernel
    /// after the model was loaded.
    ///
    /// # Arguments
    ///
    /// (none)
    ///
    /// # Returns
    ///
    /// (none)
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
    /// model.precompute_clenshaw_tables();
    /// ```
    pub fn precompute_clenshaw_tables(&mut self) {
        if self.clenshaw.is_none() {
            self.clenshaw = Some(self.build_clenshaw_tables());
        }
    }

    /// Apply a [`GravityTables`] load configuration: precompute the
    /// requested table set(s) and drop whichever set is not requested.
    fn apply_tables(&mut self, tables: GravityTables) {
        match tables {
            GravityTables::Clenshaw => {
                self.precompute_clenshaw_tables();
                self.drop_cunningham_tables();
            }
            GravityTables::Cunningham => {
                self.precompute_cunningham_tables();
                self.drop_clenshaw_tables();
            }
            GravityTables::Both => {
                self.precompute_clenshaw_tables();
                self.precompute_cunningham_tables();
            }
        }
    }

    /// Drop the Clenshaw tables, freeing their memory. The Clenshaw kernel
    /// errors until [`Self::precompute_clenshaw_tables`] is called again.
    ///
    /// # Returns
    ///
    /// (none)
    ///
    /// # Examples
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
    /// model.drop_clenshaw_tables();
    /// assert!(!model.has_clenshaw_tables());
    /// ```
    pub fn drop_clenshaw_tables(&mut self) {
        self.clenshaw = None;
    }

    /// Drop the Cunningham tables, freeing their memory (five dense
    /// model-sized matrices). The Cunningham kernel errors until
    /// [`Self::precompute_cunningham_tables`] is called again.
    ///
    /// # Returns
    ///
    /// (none)
    ///
    /// # Examples
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
    /// model.precompute_cunningham_tables();
    /// model.drop_cunningham_tables();
    /// assert!(!model.has_cunningham_tables());
    /// ```
    pub fn drop_cunningham_tables(&mut self) {
        self.cunningham = None;
    }

    /// Check whether the Clenshaw kernel's tables are present.
    ///
    /// # Returns
    ///
    /// - `bool` : `true` if [`Self::compute_spherical_harmonics_clenshaw`] can be called without error.
    ///
    /// # Examples
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
    /// assert!(model.has_clenshaw_tables());
    /// ```
    pub fn has_clenshaw_tables(&self) -> bool {
        self.clenshaw.is_some()
    }

    /// Check whether the Cunningham kernel's tables are present.
    ///
    /// # Returns
    ///
    /// - `bool` : `true` if [`Self::compute_spherical_harmonics_cunningham`] can be called without error.
    ///
    /// # Examples
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
    /// assert!(!model.has_cunningham_tables());
    /// ```
    pub fn has_cunningham_tables(&self) -> bool {
        self.cunningham.is_some()
    }

    fn from_bufreader<T: Read>(reader: BufReader<T>) -> Result<Self, BraheError> {
        let mut lines = reader.lines();

        // Read the header
        let mut line = lines.next().unwrap()?;

        let mut model_name = String::from("Unknown");
        let mut gm = 0.0;
        let mut radius = 0.0;
        let mut n_max = 0;
        let mut m_max = 0;
        let mut tide_system = GravityModelTideSystem::Unknown;
        let mut model_errors = GravityModelErrors::No;
        let mut normalization = GravityModelNormalization::FullyNormalized;

        while !line.starts_with("end_of_head") {
            // Read the header line
            line = lines.next().unwrap()?;

            // Parse the header
            if line.starts_with("modelname") {
                model_name = String::from(line.split_whitespace().last().unwrap());
            } else if line.starts_with("earth_gravity_constant") {
                gm = line.split_whitespace().last().unwrap().parse::<f64>()?;
            } else if line.starts_with("radius") {
                radius = line.split_whitespace().last().unwrap().parse::<f64>()?;
            } else if line.starts_with("max_degree") {
                n_max = line.split_whitespace().last().unwrap().parse::<usize>()?;
                m_max = n_max;
            } else if line.starts_with("tide_system") {
                tide_system = match line.split_whitespace().last().unwrap() {
                    "zero_tide" => GravityModelTideSystem::ZeroTide,
                    "tide_free" => GravityModelTideSystem::TideFree,
                    "mean_tide" => GravityModelTideSystem::MeanTide,
                    _ => GravityModelTideSystem::Unknown,
                };
            } else if line.starts_with("errors") {
                model_errors = match line.split_whitespace().last().unwrap() {
                    "no" => GravityModelErrors::No,
                    "calibrated" => GravityModelErrors::Calibrated,
                    "formal" => GravityModelErrors::Formal,
                    "calibrated_and_formal" => GravityModelErrors::CalibratedAndFormal,
                    _ => {
                        return Err(BraheError::ParseError(format!(
                            "Invalid model_errors value: \"{}\". Expected \"no\", \"calibrated\", \"formal\", or \"calibrated_and_formal\".",
                            line.split_whitespace().last().unwrap()
                        )));
                    }
                };
            } else if line.starts_with("normalization") {
                normalization = match line.split_whitespace().last().unwrap() {
                    "fully_normalized" => GravityModelNormalization::FullyNormalized,
                    "unnormalized" => GravityModelNormalization::Unnormalized,
                    _ => {
                        return Err(BraheError::ParseError(format!(
                            "Invalid normalization value: \"{}\". Expected \"fully_normalized\" or \"unnormalized\".",
                            line.split_whitespace().last().unwrap()
                        )));
                    }
                };
            }
        }

        // Confirm that the header contained all required fields
        if gm == 0.0 {
            return Err(BraheError::ParseError(
                "Gravity model file header missing required field: \"earth_gravity_constant\""
                    .to_string(),
            ));
        }
        if radius == 0.0 {
            return Err(BraheError::ParseError(
                "Gravity model file header missing required field: \"radius\"".to_string(),
            ));
        }
        if n_max == 0 {
            return Err(BraheError::ParseError(
                "Gravity model file header missing required field: \"max_degree\"".to_string(),
            ));
        }

        // Read the data
        let mut data = DMatrix::zeros(n_max + 1, m_max + 1);

        for line in lines {
            let l = line?.replace("D", "e").replace("d", "e");
            let mut values = l.split_whitespace();

            // Convert values from string to numeric types
            values.next(); // Skip the first value
            let n = values.next().unwrap().parse::<usize>()?;
            let m = values.next().unwrap().parse::<usize>()?;
            let c = values.next().unwrap().parse::<f64>()?;
            let s = values.next().unwrap().parse::<f64>()?;
            // let sig_C = values.next().unwrap().parse::<f64>()?;
            // let sig_S = values.next().unwrap().parse::<f64>()?;

            // Store the values in the data matrix
            data[(n, m)] = c;

            if m > 0 {
                data[(m - 1, n)] = s;
            }
        }

        let mut model = Self {
            data,
            tide_system,
            n_max,
            m_max,
            gm,
            radius,
            model_name,
            model_errors,
            normalization,
            cunningham: None,
            clenshaw: None,
        };
        model.precompute_clenshaw_tables();
        Ok(model)
    }

    /// Load a gravity model from a file.
    ///
    /// # Arguments
    ///
    /// - `filepath` : Path to the gravity model file.
    ///
    /// # Returns
    ///
    /// - `Self` : Gravity model object.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use brahe::gravity::GravityModel;
    /// use std::path::Path;
    ///
    /// let filepath = Path::new("./data/gravity_models/EGM2008_360.gfc");
    /// let gravity_model = GravityModel::from_file(filepath).unwrap();
    /// ```
    pub fn from_file(filepath: &Path) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);

        Self::from_bufreader(reader)
    }

    /// Load a gravity model from a file with an explicit [`GravityTables`]
    /// load configuration, instead of the Clenshaw-only default.
    ///
    /// # Arguments
    ///
    /// - `filepath` : Path to the gravity model file.
    /// - `tables` : Which precomputed evaluation table set(s) to build.
    ///
    /// # Returns
    ///
    /// - `Result<Self, BraheError>` : Loaded gravity model, or error if file loading fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use brahe::gravity::{GravityModel, GravityTables};
    /// use std::path::Path;
    ///
    /// let filepath = Path::new("./data/gravity_models/EGM2008_360.gfc");
    /// let model = GravityModel::from_file_with_tables(filepath, GravityTables::Both).unwrap();
    /// ```
    pub fn from_file_with_tables(
        filepath: &Path,
        tables: GravityTables,
    ) -> Result<Self, BraheError> {
        let mut m = Self::from_file(filepath)?;
        m.apply_tables(tables);
        Ok(m)
    }

    /// Load a gravity model from packaged models or file.
    ///
    /// The available packaged models are:
    /// - `EGM2008_360` - a truncated 360x360 version of the full 2190x2190 EGM2008 model.
    /// - `GGM05S` - The full 180x180 GGM05S model.
    /// - `JGM3` - The full 70x70 JGM3 model.
    ///
    /// Or load a custom model from file using `FromFile(path)`.
    ///
    /// Loads are backed by a process-wide cache: the first call for a given
    /// `GravityModelType` parses the underlying `.gfc` data once, and every
    /// subsequent call returns an owned clone of the cached model (~1 ms
    /// memcpy instead of ~60 ms disk parse for EGM2008_360). Use
    /// [`clear_gravity_model_cache`] to drop cached entries, or
    /// [`Self::load_uncached`] to bypass the cache entirely (useful when
    /// profiling cold-load behavior or asserting deterministic memory).
    ///
    /// # Caution: cache is unbounded
    ///
    /// Every distinct `GravityModelType` ever passed in here stays resident
    /// for the process lifetime — there is no eviction policy yet (see the
    /// For only loading a few models this is fine. but
    /// programs that loop over many distinct [`GravityModelType::FromFile`]
    /// paths will see the cache grow without bound — each unique path
    /// retains its own ~0.1-2 MB allocation depending on degree/order.
    ///
    /// If you iterate over many file-backed models, prefer
    /// [`Self::load_uncached`] (which doesn't touch the cache) or call
    /// [`clear_gravity_model_cache`] between batches.
    ///
    /// # Arguments
    ///
    /// - `model` : Gravity model type to load. This is a `GravityModelType` enum.
    ///
    /// # Returns
    ///
    /// - `Result<Self, BraheError>` : Loaded gravity model, or error if file loading fails.
    pub fn from_model_type(model: &GravityModelType) -> Result<Self, BraheError> {
        let arc = Self::shared(model)?;
        Ok((*arc).clone())
    }

    /// Load a gravity model from packaged models or file with an explicit
    /// [`GravityTables`] load configuration, instead of the Clenshaw-only
    /// default.
    ///
    /// Still goes through the process-wide cache (see [`Self::from_model_type`])
    /// to get the underlying coefficients, then adjusts the returned owned
    /// model's table set(s) — the cached entry itself is unaffected and stays
    /// at the Clenshaw-only configuration.
    ///
    /// # Arguments
    ///
    /// - `model` : Gravity model type to load.
    /// - `tables` : Which precomputed evaluation table set(s) to build.
    ///
    /// # Returns
    ///
    /// - `Result<Self, BraheError>` : Loaded gravity model, or error if file loading fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType, GravityTables};
    ///
    /// let model = GravityModel::from_model_type_with_tables(
    ///     &GravityModelType::JGM3,
    ///     GravityTables::Both,
    /// )
    /// .unwrap();
    /// assert!(model.has_clenshaw_tables() && model.has_cunningham_tables());
    /// ```
    pub fn from_model_type_with_tables(
        model: &GravityModelType,
        tables: GravityTables,
    ) -> Result<Self, BraheError> {
        let mut m = Self::from_model_type(model)?;
        m.apply_tables(tables);
        Ok(m)
    }

    /// Internal: get a shared, cached `Arc<GravityModel>` for a packaged or
    /// file-backed model type. Backs [`Self::from_model_type`] — repeated
    /// calls for the same `GravityModelType` return an `Arc` pointing at the
    /// same allocation.
    ///
    /// Models from the shared cache are fixed at the Clenshaw-only table
    /// configuration — use [`Self::from_model_type_with_tables`] for an
    /// owned model with a different configuration.
    pub(crate) fn shared(model: &GravityModelType) -> Result<Arc<GravityModel>, BraheError> {
        // Fast path: existing entry, no allocation, no load.
        {
            let cache = GRAVITY_MODEL_CACHE.read().unwrap();
            if let Some(existing) = cache.get(model) {
                return Ok(Arc::clone(existing));
            }
        }

        // Cold path: parse the model with no cache lock held so concurrent
        // hits on other types aren't blocked, then double-check on insert
        // so a racing thread loading the same type doesn't waste the work.
        let loaded = Self::load_uncached(model)?;
        let arc = Arc::new(loaded);

        let mut cache = GRAVITY_MODEL_CACHE.write().unwrap();
        if let Some(existing) = cache.get(model) {
            return Ok(Arc::clone(existing));
        }
        cache.insert(model.clone(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Parse a `GravityModelType` directly from its underlying source,
    /// bypassing the process-wide cache.
    ///
    /// Most callers should prefer [`Self::from_model_type`], which is
    /// cache-backed and avoids the ~60 ms disk parse on repeat calls. Reach
    /// for `load_uncached` when you need:
    /// - deterministic memory behavior (each call allocates its own coefficients)
    /// - to profile or compare cold-load performance against the cached path
    /// - to re-read a `FromFile(path)` source whose contents have changed
    ///   on disk (the alternative is [`clear_gravity_model_cache`] followed
    ///   by `from_model_type`, which also works for the packaged variants)
    ///
    /// # Arguments
    ///
    /// - `model` : Gravity model type to load.
    ///
    /// # Returns
    ///
    /// - `Result<Self, BraheError>` : Freshly parsed gravity model, or load error.
    pub fn load_uncached(model: &GravityModelType) -> Result<Self, BraheError> {
        match model {
            GravityModelType::EGM2008_360 => {
                let reader = BufReader::new(PACKAGED_EGM2008_360);
                Self::from_bufreader(reader)
            }
            GravityModelType::GGM05S => {
                let reader = BufReader::new(PACKAGED_GGM05S);
                Self::from_bufreader(reader)
            }
            GravityModelType::JGM3 => {
                let reader = BufReader::new(PACKAGED_JGM3);
                Self::from_bufreader(reader)
            }
            GravityModelType::FromFile(path) => Self::from_file(Path::new(path)),
            GravityModelType::ICGEMModel { body, name } => {
                let path = crate::datasets::icgem::download_icgem_model(body.clone(), name, None)?;
                Self::from_file(&path)
            }
        }
    }

    /// Parse a `GravityModelType` directly from its underlying source with
    /// an explicit [`GravityTables`] load configuration, bypassing the
    /// process-wide cache. Combines [`Self::load_uncached`] and
    /// [`Self::from_model_type_with_tables`]'s table selection.
    ///
    /// # Arguments
    ///
    /// - `model` : Gravity model type to load.
    /// - `tables` : Which precomputed evaluation table set(s) to build.
    ///
    /// # Returns
    ///
    /// - `Result<Self, BraheError>` : Freshly parsed gravity model, or load error.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType, GravityTables};
    ///
    /// let model = GravityModel::load_uncached_with_tables(
    ///     &GravityModelType::JGM3,
    ///     GravityTables::Cunningham,
    /// )
    /// .unwrap();
    /// assert!(model.has_cunningham_tables() && !model.has_clenshaw_tables());
    /// ```
    pub fn load_uncached_with_tables(
        model: &GravityModelType,
        tables: GravityTables,
    ) -> Result<Self, BraheError> {
        let mut m = Self::load_uncached(model)?;
        m.apply_tables(tables);
        Ok(m)
    }

    /// Get the gravity model coefficients for a given degree and order.
    ///
    /// # Arguments
    ///
    /// - `n` : Degree of the gravity model.
    /// - `m` : Order of the gravity model.
    ///
    /// # Returns
    ///
    /// - `Result<(f64, f64), BraheError>` : Tuple of the gravity model coefficients. The first value
    ///   is the C coefficient and the second value is the S coefficient.
    pub fn get(&self, n: usize, m: usize) -> Result<(f64, f64), BraheError> {
        if n > self.n_max || m > self.m_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n={}, m={}) are out of bounds (n_max={}, m_max={}).",
                n, m, self.n_max, self.m_max
            )));
        }

        if m == 0 {
            Ok((self.data[(n, m)], 0.0))
        } else {
            Ok((self.data[(n, m)], self.data[(m - 1, n)]))
        }
    }

    /// Truncate the gravity model to a smaller degree and order.
    ///
    /// This reduces memory usage by discarding higher-degree/order coefficients
    /// that won't be used. The operation is irreversible - coefficients beyond
    /// the new limits are permanently removed.
    ///
    /// # Arguments
    ///
    /// * `n` - New maximum degree (must be <= current n_max)
    /// * `m` - New maximum order (must be <= n and <= current m_max)
    ///
    /// # Returns
    ///
    /// * `Ok(())` if truncation succeeded
    /// * `Err(BraheError)` if validation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::gravity::{GravityModel, GravityModelType};
    ///
    /// let mut model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
    /// assert_eq!(model.n_max, 360);
    ///
    /// // Reduce from 360×360 to 70×70 to save memory
    /// model.set_max_degree_order(70, 70).unwrap();
    /// assert_eq!(model.n_max, 70);
    /// assert_eq!(model.m_max, 70);
    /// ```
    pub fn set_max_degree_order(&mut self, n: usize, m: usize) -> Result<(), BraheError> {
        // Validate: m <= n
        if m > n {
            return Err(BraheError::Error(format!(
                "Maximum order (m={}) cannot exceed maximum degree (n={})",
                m, n
            )));
        }

        // Validate: new limits don't exceed current model
        if n > self.n_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested degree (n={}) exceeds model's maximum degree (n_max={})",
                n, self.n_max
            )));
        }
        if m > self.m_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested order (m={}) exceeds model's maximum order (m_max={})",
                m, self.m_max
            )));
        }

        // Skip if no resize needed
        if n == self.n_max && m == self.m_max {
            return Ok(());
        }

        // Resize matrix in-place using nalgebra's resize_mut
        // The data matrix stores:
        //   - C coefficients at data[(n, m)] for the lower triangle
        //   - S coefficients at data[(m-1, n)] for m > 0 (upper triangle shifted)
        // Both fit within the (n+1) x (n+1) square when n == m (typical case)
        let new_size = n + 1;

        // resize_mut preserves existing values at their (row, col) positions
        // and fills new cells with the provided value (0.0 here, but we're shrinking)
        self.data.resize_mut(new_size, new_size, 0.0);

        // Update model limits
        self.n_max = n;
        self.m_max = m;

        // Rebuild whichever precomputed table sets exist against the resized data.
        if self.cunningham.is_some() {
            self.cunningham = Some(self.build_cunningham_tables());
        }
        if self.clenshaw.is_some() {
            self.clenshaw = Some(self.build_clenshaw_tables());
        }

        Ok(())
    }

    /// Compute gravitational acceleration from spherical harmonic expansion
    /// using the Cunningham (Montenbruck & Gill) V/W recursion.
    ///
    /// Evaluates gravity field using recursively-computed associated Legendre functions.
    /// Higher degrees/orders provide more accurate representation of Earth's gravitational
    /// field but increase computational cost.
    ///
    /// Each call heap-allocates two `(n_max + 2) × (n_max + 2)` matrices for
    /// the recurrence calcualtion. For hot-path workloads (numerical
    /// propagation, where the integrator calls this 4-17 times per step) use
    /// [`Self::compute_spherical_harmonics_cunningham_with_workspace`] to reuse the
    /// allocations across calls.
    ///
    /// # Arguments
    /// - `r_body`: Position vector in body-fixed frame (e.g., ECEF). Units: meters.
    /// - `n_max`: Maximum degree of expansion (zonal terms). Must not exceed model's n_max.
    /// - `m_max`: Maximum order of expansion (tesseral/sectoral terms). Must satisfy m_max <= n_max.
    ///
    /// # Returns
    /// Acceleration vector in body-fixed frame. Units: m/s².
    ///
    /// # Errors
    /// - `BraheError::Error` if the Cunningham tables are not precomputed for this model.
    /// - `BraheError::OutOfBoundsError` if requested n_max or m_max exceeds loaded model's limits
    /// - `BraheError::OutOfBoundsError` if m_max > n_max
    /// - `BraheError::Error` if the denormalized recursion overflows and produces a
    ///   non-finite result (see `# Numerical limits` below).
    ///
    /// # Numerical limits
    /// This kernel denormalizes coefficients into an unnormalized V/W
    /// recursion, which is not degree-stable: `V(m, m)` grows like
    /// `(2m-1)!!`, and at LEO-altitude radii (where `q = radius / r` is not
    /// small) that growth overflows `f64` around degree ~150, in which case
    /// this function returns `Err` rather than a non-finite result. Below
    /// that ceiling, accuracy still degrades progressively above roughly
    /// degree 120 near the equator, where the O(n) sequential-division
    /// coefficient denormalization amplifies rounding error through the
    /// near-total cancellation in the acceleration's off-axis components.
    /// Use [`Self::compute_spherical_harmonics_clenshaw`] for high-degree work.
    pub fn compute_spherical_harmonics_cunningham(
        &self,
        r_body: Vector3<f64>,
        n_max: usize,
        m_max: usize,
        parallel: ParallelMode,
    ) -> Result<Vector3<f64>, BraheError> {
        let mut v_workspace = DMatrix::<f64>::zeros(n_max + 2, n_max + 2);
        let mut w_workspace = DMatrix::<f64>::zeros(n_max + 2, n_max + 2);
        self.compute_spherical_harmonics_cunningham_with_workspace(
            r_body,
            n_max,
            m_max,
            parallel,
            &mut v_workspace,
            &mut w_workspace,
        )
    }

    /// Variant of [`Self::compute_spherical_harmonics_cunningham`] that operates on
    /// caller-supplied work matrices.
    ///
    /// Designed for hot paths (numerical propagation) where avoiding the
    /// per-call `DMatrix::zeros((n_max + 2)²)` allocation is worthwhile.
    /// Callers typically construct the workspace once at the highest
    /// `n_max` they'll need, then reuse it across many calls — the
    /// recurrence only writes-then-reads cells in `[0..n_max+2, 0..m_max+2]`,
    /// so leftover values in unused cells from previous calls are never
    /// observed and the workspace doesn't need to be zeroed between calls.
    ///
    /// If the workspace is smaller than required it is resized in-place
    /// (`DMatrix::resize_mut`); the resize allocates only when growing, so
    /// steady-state use at a stable `n_max` is allocation-free.
    ///
    /// See [`Self::compute_spherical_harmonics_cunningham`]'s `# Numerical
    /// limits` section — this variant shares the same unnormalized V/W
    /// recursion and its high-degree overflow/precision ceiling.
    ///
    /// # Arguments
    /// - `r_body`: Position vector in body-fixed frame.
    /// - `n_max`, `m_max`: Same constraints as [`Self::compute_spherical_harmonics_cunningham`].
    /// - `v_workspace`, `w_workspace`: Mutable references to the V and W
    ///   recurrence buffers. Must be at least `(n_max + 2) × (n_max + 2)`;
    ///   will be grown if smaller.
    #[allow(non_snake_case)]
    pub fn compute_spherical_harmonics_cunningham_with_workspace(
        &self,
        r_body: Vector3<f64>,
        n_max: usize,
        m_max: usize,
        parallel: ParallelMode,
        v_workspace: &mut DMatrix<f64>,
        w_workspace: &mut DMatrix<f64>,
    ) -> Result<Vector3<f64>, BraheError> {
        let tables = self.cunningham.as_ref().ok_or_else(|| {
            BraheError::Error(
                "Cunningham tables not precomputed for this gravity model. Load with \
                 GravityTables::Cunningham or GravityTables::Both, or call \
                 precompute_cunningham_tables() first."
                    .to_string(),
            )
        })?;

        if n_max > self.n_max || m_max > self.m_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n_max={}, m_max={}) are out of bounds for the input model (n_max={}, m_max={}).",
                n_max, m_max, self.n_max, self.m_max
            )));
        }

        if m_max > n_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n_max={}, m_max={}) are out of bounds. m_max must be less than or equal to n_max.",
                n_max, m_max
            )));
        }

        // Auxiliary quantities
        let r_sqr = r_body.dot(&r_body); // Square of distance
        let rho = self.radius * self.radius / r_sqr;
        // Normalized coordinates
        let x0 = self.radius * r_body[0] / r_sqr;
        let y0 = self.radius * r_body[1] / r_sqr;
        let z0 = self.radius * r_body[2] / r_sqr;

        // Grow workspace if needed. The recurrence only reads cells it has
        // explicitly written this call, so any pre-existing values in larger
        // workspaces are harmless.
        let needed = n_max + 2;
        if v_workspace.nrows() < needed || v_workspace.ncols() < needed {
            v_workspace.resize_mut(needed, needed, 0.0);
        }
        if w_workspace.nrows() < needed || w_workspace.ncols() < needed {
            w_workspace.resize_mut(needed, needed, 0.0);
        }
        let V = v_workspace;
        let W = w_workspace;

        let run_parallel = should_parallelize(parallel, n_max);

        // ---- Phase 1: V/W recurrence ----
        // Zonal column m = 0 (independent, sequential either way).
        V[(0, 0)] = self.radius / r_sqr.sqrt();
        W[(0, 0)] = 0.0;
        V[(1, 0)] = z0 * V[(0, 0)];
        W[(1, 0)] = 0.0;
        for n in 2..(n_max + 2) {
            V[(n, 0)] = tables.rec_a[(n, 0)] * z0 * V[(n - 1, 0)]
                - tables.rec_b[(n, 0)] * rho * V[(n - 2, 0)];
            W[(n, 0)] = 0.0;
        }

        if run_parallel {
            // Sequential diagonal chain (the only cross-column dependency) plus
            // the first sub-diagonal seed for each column.
            for m in 1..m_max + 2 {
                let mf = m as f64;
                V[(m, m)] = (2.0 * mf - 1.0) * (x0 * V[(m - 1, m - 1)] - y0 * W[(m - 1, m - 1)]);
                W[(m, m)] = (2.0 * mf - 1.0) * (x0 * W[(m - 1, m - 1)] + y0 * V[(m - 1, m - 1)]);
                if m <= n_max {
                    V[(m + 1, m)] = (2.0 * mf + 1.0) * z0 * V[(m, m)];
                    W[(m + 1, m)] = (2.0 * mf + 1.0) * z0 * W[(m, m)];
                }
            }

            // Parallel column fill. DMatrix is column-major, so each column is a
            // contiguous slice of length `stride`. The first `m_max + 2` columns
            // occupy the first `(m_max + 2) * stride` contiguous elements; slice
            // to exactly those so `par_chunks_mut` yields one chunk per column.
            // (rayon's parallel iterator has no `take`, so bound via the slice.)
            let stride = V.nrows();
            let ncols_used = m_max + 2;
            let v_slice = &mut V.as_mut_slice()[..ncols_used * stride];
            let w_slice = &mut W.as_mut_slice()[..ncols_used * stride];
            let rec_a = &tables.rec_a;
            let rec_b = &tables.rec_b;
            get_thread_pool().install(|| {
                v_slice
                    .par_chunks_mut(stride)
                    .zip(w_slice.par_chunks_mut(stride))
                    .enumerate()
                    .for_each(|(m, (v_col, w_col))| {
                        if m == 0 {
                            return; // zonal column already filled
                        }
                        for n in m + 2..n_max + 2 {
                            let a = rec_a[(n, m)];
                            let b = rec_b[(n, m)];
                            v_col[n] = a * z0 * v_col[n - 1] - b * rho * v_col[n - 2];
                            w_col[n] = a * z0 * w_col[n - 1] - b * rho * w_col[n - 2];
                        }
                    });
            });
        } else {
            // Column strides differ: the V/W workspace may be oversized from a
            // prior larger call, while rec_a/rec_b are sized to the model. Each
            // column is sliced to exactly `col_len = n_max + 2` elements so the
            // inner-loop index `n < n_max + 2` is provably in-bounds and LLVM
            // elides the per-access bounds checks on the serial hot path.
            let col_len = n_max + 2;
            let v_stride = V.nrows();
            let r_stride = tables.rec_a.nrows();
            let rec_a = tables.rec_a.as_slice();
            let rec_b = tables.rec_b.as_slice();
            for m in 1..m_max + 2 {
                let mf = m as f64;
                // Diagonal + sub-diagonal seeds read column m-1 (cross-column),
                // so they stay on the matrix API rather than the column slices.
                V[(m, m)] = (2.0 * mf - 1.0) * (x0 * V[(m - 1, m - 1)] - y0 * W[(m - 1, m - 1)]);
                W[(m, m)] = (2.0 * mf - 1.0) * (x0 * W[(m - 1, m - 1)] + y0 * V[(m - 1, m - 1)]);
                if m <= n_max {
                    V[(m + 1, m)] = (2.0 * mf + 1.0) * z0 * V[(m, m)];
                    W[(m + 1, m)] = (2.0 * mf + 1.0) * z0 * W[(m, m)];
                }
                let v_lo = m * v_stride;
                let r_lo = m * r_stride;
                let v_col = &mut V.as_mut_slice()[v_lo..v_lo + col_len];
                let w_col = &mut W.as_mut_slice()[v_lo..v_lo + col_len];
                let ra = &rec_a[r_lo..r_lo + col_len];
                let rb = &rec_b[r_lo..r_lo + col_len];
                for n in m + 2..n_max + 2 {
                    let a = ra[n];
                    let b = rb[n];
                    v_col[n] = a * z0 * v_col[n - 1] - b * rho * v_col[n - 2];
                    w_col[n] = a * z0 * w_col[n - 1] - b * rho * w_col[n - 2];
                }
            }
        }

        // ---- Phase 2: acceleration accumulation ----
        // Reborrow the workspaces immutably so the accumulation closure is
        // `Sync` (a closure capturing `&mut DMatrix` is not, and rayon's
        // `par_iter` requires `Sync`). The Phase 1 mutable borrows have ended.
        let v_ref: &DMatrix<f64> = V;
        let w_ref: &DMatrix<f64> = W;
        // Column-slice the V/W workspaces and coefficient matrices so the
        // accumulation reads are bounds-check-free (same technique as the
        // recurrence above). The accumulation touches columns m-1, m, m+1 at
        // row n+1, and coefficient column m at row n. `vw_len = n_max + 2`
        // bounds the V/W row index n+1; `cf_len = n_max + 1` bounds the
        // coefficient row index n. Strides may differ between the (possibly
        // oversized) workspaces and the model-sized coefficient matrices.
        let v_stride = v_ref.nrows();
        let c_stride = tables.coeff_c.nrows();
        let v_buf = v_ref.as_slice();
        let w_buf = w_ref.as_slice();
        let c_buf = tables.coeff_c.as_slice();
        let s_buf = tables.coeff_s.as_slice();
        let f_buf = tables.fac.as_slice();
        let vw_len = n_max + 2;
        let cf_len = n_max + 1;
        let accumulate_m = |m: usize| -> (f64, f64, f64) {
            let mf = m as f64;
            let (mut ax, mut ay, mut az) = (0.0, 0.0, 0.0);
            if m == 0 {
                let c_col = &c_buf[0..cf_len];
                let v0 = &v_buf[0..vw_len];
                let v1 = &v_buf[v_stride..v_stride + vw_len];
                let w1 = &w_buf[v_stride..v_stride + vw_len];
                for n in 0..n_max + 1 {
                    let nf = n as f64;
                    let c = c_col[n];
                    ax -= c * v1[n + 1];
                    ay -= c * w1[n + 1];
                    az -= (nf + 1.0) * c * v0[n + 1];
                }
            } else {
                let cbase = m * c_stride;
                let c_col = &c_buf[cbase..cbase + cf_len];
                let s_col = &s_buf[cbase..cbase + cf_len];
                let f_col = &f_buf[cbase..cbase + cf_len];
                // Columns m-1, m, m+1 are contiguous starting at (m-1)*stride.
                let vbase = (m - 1) * v_stride;
                let v_lo = &v_buf[vbase..vbase + vw_len];
                let v_mid = &v_buf[vbase + v_stride..vbase + v_stride + vw_len];
                let v_hi = &v_buf[vbase + 2 * v_stride..vbase + 2 * v_stride + vw_len];
                let w_lo = &w_buf[vbase..vbase + vw_len];
                let w_mid = &w_buf[vbase + v_stride..vbase + v_stride + vw_len];
                let w_hi = &w_buf[vbase + 2 * v_stride..vbase + 2 * v_stride + vw_len];
                for n in m..n_max + 1 {
                    let nf = n as f64;
                    let c = c_col[n];
                    let s = s_col[n];
                    let fac = f_col[n];
                    let p = n + 1;
                    ax += 0.5 * (-c * v_hi[p] - s * w_hi[p]) + fac * (c * v_lo[p] + s * w_lo[p]);
                    ay += 0.5 * (-c * w_hi[p] + s * v_hi[p]) + fac * (-c * w_lo[p] + s * v_lo[p]);
                    az += (nf - mf + 1.0) * (-c * v_mid[p] - s * w_mid[p]);
                }
            }
            (ax, ay, az)
        };

        let (ax, ay, az) = if run_parallel {
            get_thread_pool().install(|| {
                (0..m_max + 1)
                    .into_par_iter()
                    .map(accumulate_m)
                    .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
            })
        } else {
            (0..m_max + 1).fold((0.0, 0.0, 0.0), |acc, m| {
                let (ax, ay, az) = accumulate_m(m);
                (acc.0 + ax, acc.1 + ay, acc.2 + az)
            })
        };

        // Body-fixed acceleration
        let a = (self.gm / (self.radius * self.radius)) * Vector3::new(ax, ay, az);
        if !(a[0].is_finite() && a[1].is_finite() && a[2].is_finite()) {
            return Err(BraheError::Error(format!(
                "Cunningham spherical-harmonic kernel produced a non-finite result at \
                 n_max={}, m_max={} (denormalized V/W overflow at high degree and low \
                 altitude). Use the Clenshaw kernel (the default) for high-degree \
                 evaluations.",
                n_max, m_max
            )));
        }
        Ok(a)
    }

    /// Compute gravitational acceleration from spherical harmonic expansion
    /// using the Clenshaw-summation algorithm of Holmes & Featherstone
    /// (2002), ported from GeographicLib v1.52 `SphericalEngine`.
    ///
    /// Unlike [`Self::compute_spherical_harmonics_cunningham`], which builds
    /// explicit `(n_max + 2) × (n_max + 2)` Legendre recurrence matrices, this
    /// kernel evaluates the field with a pair of nested Clenshaw backward
    /// recurrences (inner over degree per order, outer over order) that never
    /// materialize the Legendre functions themselves. Coefficients are
    /// consumed directly from the fully-normalized triangular packing in
    /// [`ClenshawTables`], so there is no per-call heap allocation and no
    /// denormalization overflow ceiling — the model is usable to arbitrarily
    /// high degree (subject to the packed-table memory footprint).
    ///
    /// The two kernels agree to better than `1e-10` relative accuracy across
    /// degrees 2–200, including positions on the polar axis.
    ///
    /// # Arguments
    /// - `r_body`: Position vector in body-fixed frame (e.g., ECEF). Units: meters.
    /// - `n_max`: Maximum degree of expansion (zonal terms). Must not exceed model's n_max.
    /// - `m_max`: Maximum order of expansion (tesseral/sectoral terms). Must satisfy m_max <= n_max.
    /// - `parallel`: Execution policy. The serial path is used regardless of
    ///   this setting until the parallel-over-order dispatch lands.
    ///
    /// # Returns
    /// Acceleration vector in body-fixed frame. Units: m/s².
    ///
    /// # Errors
    /// - `BraheError::Error` if the model was loaded without Clenshaw tables
    ///   (see [`Self::precompute_clenshaw_tables`]).
    /// - `BraheError::OutOfBoundsError` if requested n_max or m_max exceeds loaded model's limits
    /// - `BraheError::OutOfBoundsError` if m_max > n_max
    ///
    /// # Examples
    /// ```
    /// use nalgebra::Vector3;
    /// use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
    /// use brahe::R_EARTH;
    ///
    /// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
    /// let r_body = Vector3::new(R_EARTH + 500.0e3, 0.0, 0.0);
    /// let a_grav = gravity_model
    ///     .compute_spherical_harmonics_clenshaw(r_body, 20, 20, ParallelMode::Auto)
    ///     .unwrap();
    /// ```
    pub fn compute_spherical_harmonics_clenshaw(
        &self,
        r_body: Vector3<f64>,
        n_max: usize,
        m_max: usize,
        parallel: ParallelMode,
    ) -> Result<Vector3<f64>, BraheError> {
        let tables = self.clenshaw.as_ref().ok_or_else(|| {
            BraheError::Error(
                "Clenshaw tables not precomputed for this gravity model. Load with \
                 GravityTables::Clenshaw or GravityTables::Both, or call \
                 precompute_clenshaw_tables() first."
                    .to_string(),
            )
        })?;

        if n_max > self.n_max || m_max > self.m_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n_max={}, m_max={}) are out of bounds for the input model (n_max={}, m_max={}).",
                n_max, m_max, self.n_max, self.m_max
            )));
        }
        if m_max > n_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n_max={}, m_max={}) are out of bounds. m_max must be less than or equal to n_max.",
                n_max, m_max
            )));
        }

        // Geometry (GeographicLib SphericalEngine.cpp:161–173). theta is
        // colatitude; lambda longitude. The pole guard nudges u = sin(theta)
        // away from zero — the apparent 1/u singularities cancel in the
        // final Cartesian assembly.
        let (x, y, z) = (r_body[0], r_body[1], r_body[2]);
        let p = x.hypot(y);
        let cl = if p != 0.0 { x / p } else { 1.0 };
        let sl = if p != 0.0 { y / p } else { 0.0 };
        let r = z.hypot(p);
        let t = if r != 0.0 { z / r } else { 0.0 };
        let pole_eps = f64::EPSILON * f64::EPSILON.sqrt();
        let u = if r != 0.0 { (p / r).max(pole_eps) } else { 1.0 };
        let q = self.radius / r;
        let q2 = q * q;
        let uq = u * q;
        let uq2 = uq * uq;
        let tu = t / u;

        // Independent inner sweeps per order. Serial in this task; Task 4
        // adds the rayon branch (identical results either way).
        let sums: Vec<PerOrderSums> = if should_parallelize_clenshaw(parallel, n_max) {
            get_thread_pool().install(|| {
                (0..=m_max)
                    .into_par_iter()
                    .map(|m| clenshaw_inner_sweep(tables, m, n_max, t, u, q, q2, tu))
                    .collect()
            })
        } else {
            (0..=m_max)
                .map(|m| clenshaw_inner_sweep(tables, m, n_max, t, u, q, q2, tu))
                .collect()
        };

        // Sequential outer Clenshaw over order in cos/sin(m * lambda)
        // (SphericalEngine.cpp:232–284), consuming the buffer high-to-low.
        let root = &tables.sqrt_table;
        let (mut vc, mut vc2, mut vs, mut vs2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut vrc, mut vrc2, mut vrs, mut vrs2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut vtc, mut vtc2, mut vts, mut vts2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut vlc, mut vlc2, mut vls, mut vls2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

        for m in (1..=m_max).rev() {
            let sm = &sums[m];
            let v = root[2] * root[2 * m + 3] / root[m + 1];
            let a = cl * v * uq;
            let b = -v * root[2 * m + 5] / (root[8] * root[m + 2]) * uq2;
            let mf = m as f64;
            let next = a * vc + b * vc2 + sm.wc;
            vc2 = vc;
            vc = next;
            let next = a * vs + b * vs2 + sm.ws;
            vs2 = vs;
            vs = next;
            let next = a * vrc + b * vrc2 + sm.wrc;
            vrc2 = vrc;
            vrc = next;
            let next = a * vrs + b * vrs2 + sm.wrs;
            vrs2 = vrs;
            vrs = next;
            let next = a * vtc + b * vtc2 + sm.wtc;
            vtc2 = vtc;
            vtc = next;
            let next = a * vts + b * vts2 + sm.wts;
            vts2 = vts;
            vts = next;
            let next = a * vlc + b * vlc2 + mf * sm.ws;
            vlc2 = vlc;
            vlc = next;
            let next = a * vls + b * vls2 - mf * sm.wc;
            vls2 = vls;
            vls = next;
        }

        // m = 0 closure: F[0] = q, F[1] = sqrt(3) * u * q^2 * cos(lambda),
        // beta[1] = -sqrt(15)/2 * u^2 * q^2 (SphericalEngine.cpp:259–284).
        let s0 = &sums[0];
        let a = root[3] * uq;
        let b = -root[15] / 2.0 * uq2;
        let qs = q / CLENSHAW_SCALE / r;
        // Spherical gradient components: vr = dV/dr, vt = (1/r) dV/dtheta,
        // vl = 1/(r*u) dV/dlambda — all still missing the GM/radius factor.
        let vr = -qs * (s0.wrc + a * (cl * vrc + sl * vrs) + b * vrc2);
        let vt = qs * (s0.wtc + a * (cl * vtc + sl * vts) + b * vtc2);
        let vl = qs / u * (a * (cl * vlc + sl * vls) + b * vlc2);

        // Rotate into body-fixed Cartesian coordinates.
        let gx = cl * (u * vr + t * vt) - sl * vl;
        let gy = sl * (u * vr + t * vt) + cl * vl;
        let gz = t * vr - u * vt;

        Ok((self.gm / self.radius) * Vector3::new(gx, gy, gz))
    }

    /// Compute gravitational acceleration from spherical harmonic expansion.
    ///
    /// Main entry point: dispatches to whichever kernel this model has
    /// tables for. Clenshaw-first — if [`Self::has_clenshaw_tables`] is
    /// true (the load default), evaluates with
    /// [`Self::compute_spherical_harmonics_clenshaw`]. Otherwise falls back
    /// to [`Self::compute_spherical_harmonics_cunningham`] if
    /// [`Self::has_cunningham_tables`] is true. Returns an error if neither
    /// table set is present.
    ///
    /// # Arguments
    /// - `r_body`: Position vector in body-fixed frame (e.g., ECEF). Units: meters.
    /// - `n_max`: Maximum degree of expansion (zonal terms). Must not exceed model's n_max.
    /// - `m_max`: Maximum order of expansion (tesseral/sectoral terms). Must satisfy m_max <= n_max.
    ///
    /// # Returns
    /// Acceleration vector in body-fixed frame. Units: m/s².
    ///
    /// # Errors
    /// - `BraheError::Error` if the model has neither Clenshaw nor
    ///   Cunningham tables (see [`Self::precompute_clenshaw_tables`] /
    ///   [`Self::precompute_cunningham_tables`]).
    /// - `BraheError::OutOfBoundsError` if requested n_max or m_max exceeds loaded model's limits
    /// - `BraheError::OutOfBoundsError` if m_max > n_max
    ///
    /// # Examples
    /// ```
    /// use nalgebra::Vector3;
    /// use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
    /// use brahe::R_EARTH;
    ///
    /// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
    /// let r_body = Vector3::new(R_EARTH + 500.0e3, 0.0, 0.0);
    /// let a_grav = gravity_model
    ///     .compute_spherical_harmonics(r_body, 20, 20, ParallelMode::Auto)
    ///     .unwrap();
    /// ```
    pub fn compute_spherical_harmonics(
        &self,
        r_body: Vector3<f64>,
        n_max: usize,
        m_max: usize,
        parallel: ParallelMode,
    ) -> Result<Vector3<f64>, BraheError> {
        if self.clenshaw.is_some() {
            self.compute_spherical_harmonics_clenshaw(r_body, n_max, m_max, parallel)
        } else if self.cunningham.is_some() {
            self.compute_spherical_harmonics_cunningham(r_body, n_max, m_max, parallel)
        } else {
            Err(BraheError::Error(
                "No precomputed gravity tables on this model. Call \
                 precompute_clenshaw_tables() or precompute_cunningham_tables()."
                    .to_string(),
            ))
        }
    }
}

impl std::fmt::Display for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GravityModel: {}", self.model_name)
    }
}

impl std::fmt::Debug for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GravityModel")
            .field("model_name", &self.model_name)
            .field("gm", &self.gm)
            .field("radius", &self.radius)
            .field("n_max", &self.n_max)
            .field("m_max", &self.m_max)
            .field("tide_system", &self.tide_system)
            .field("model_errors", &self.model_errors)
            .field("normalization", &self.normalization)
            .finish()
    }
}

/// Compute the acceleration due to gravity using a spherical harmonic gravity model. The gravity
/// model is defined by the `GravityModel` struct. The acceleration is computed in the body-fixed
/// frame of the central body, and returned in the inertial frame.
///
/// Routes through [`GravityModel::compute_spherical_harmonics`], the
/// Clenshaw-first dispatcher: the underlying kernel is Clenshaw by default,
/// falling back to Cunningham only if `gravity_model` was loaded with
/// [`GravityTables::Cunningham`] (no Clenshaw tables present).
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_eci`.
/// When a state vector is provided, only the position component is used.
///
/// # Arguments
///
/// - `r_eci` : Position vector of the object in the ECI frame, or state vector (position + velocity).
/// - `R_i2b` : Transformation matrix from the ECI frame to the body-fixed frame of the central body.
/// - `gravity_model` : Gravity model to use for the computation.
/// - `n_max` : Maximum degree of the gravity model to evaluate.
/// - `m_max` : Maximum order of the gravity model to evaluate.
///
/// # Returns
///
/// - `Vector3<f64>` : Acceleration due to gravity in the ECI frame.
///
/// # Panics
///
/// Panics if [`GravityModel::compute_spherical_harmonics`] returns an
/// error: `n_max`/`m_max` out of bounds for `gravity_model`, no precomputed
/// tables present, or (Cunningham fallback only) a non-finite result from
/// high-degree denormalized-coefficient overflow.
///
/// # Examples
///
/// Using a position vector:
/// ```
/// use nalgebra::{Vector3, Vector6};
/// use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_koe_to_eci, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model
/// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
///
/// // Compute the acceleration due to gravity
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_koe_to_eci(oe, AngleFormat::Degrees);
/// let r_eci: Vector3<f64> = x_eci.fixed_rows::<3>(0).into();
///
/// // Compute the acceleration due to gravity
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics(r_eci, R_i2b, &gravity_model, 20, 20, ParallelMode::Auto);
/// ```
///
/// Using a state vector:
/// ```
/// use nalgebra::Vector6;
/// use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_koe_to_eci, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model
/// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
///
/// // Compute the acceleration due to gravity using state vector directly
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_koe_to_eci(oe, AngleFormat::Degrees);
///
/// // Pass state vector directly - no need to extract position
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics(x_eci, R_i2b, &gravity_model, 20, 20, ParallelMode::Auto);
/// ```
#[allow(non_snake_case)]
pub fn accel_gravity_spherical_harmonics<P: IntoPosition>(
    r_eci: P,
    R_i2b: SMatrix3,
    gravity_model: &GravityModel,
    n_max: usize,
    m_max: usize,
    parallel: ParallelMode,
) -> Vector3<f64> {
    let r = r_eci.position();
    let r_bf = R_i2b * r;
    let a_ecef = gravity_model
        .compute_spherical_harmonics(r_bf, n_max, m_max, parallel)
        .unwrap();
    R_i2b.transpose() * a_ecef
}

/// Compute the acceleration due to gravity using the Cunningham (Montenbruck
/// & Gill) V/W spherical harmonic recursion. The gravity model is defined by
/// the `GravityModel` struct. The acceleration is computed in the body-fixed
/// frame of the central body, and returned in the inertial frame.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_eci`.
/// When a state vector is provided, only the position component is used.
///
/// # Arguments
///
/// - `r_eci` : Position vector of the object in the ECI frame, or state vector (position + velocity).
/// - `R_i2b` : Transformation matrix from the ECI frame to the body-fixed frame of the central body.
/// - `gravity_model` : Gravity model to use for the computation.
/// - `n_max` : Maximum degree of the gravity model to evaluate.
/// - `m_max` : Maximum order of the gravity model to evaluate.
///
/// # Returns
///
/// - `Vector3<f64>` : Acceleration due to gravity in the ECI frame.
///
/// # Panics
///
/// Panics if `gravity_model` has no precomputed Cunningham tables (see
/// [`GravityModel::precompute_cunningham_tables`] / [`GravityTables`]),
/// `n_max`/`m_max` are out of bounds, or the denormalized recursion
/// overflows to a non-finite result at high degree and low altitude.
///
/// # Examples
///
/// ```
/// use nalgebra::{Vector3, Vector6};
/// use brahe::gravity::{GravityModel, GravityModelType, GravityTables, ParallelMode};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_koe_to_eci, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model with Cunningham tables (not built by default)
/// let gravity_model = GravityModel::from_model_type_with_tables(
///     &GravityModelType::EGM2008_360,
///     GravityTables::Cunningham,
/// )
/// .unwrap();
///
/// // Compute the acceleration due to gravity
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_koe_to_eci(oe, AngleFormat::Degrees);
/// let r_eci: Vector3<f64> = x_eci.fixed_rows::<3>(0).into();
///
/// // Compute the acceleration due to gravity
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics_cunningham(r_eci, R_i2b, &gravity_model, 20, 20, ParallelMode::Auto);
/// ```
#[allow(non_snake_case)]
pub fn accel_gravity_spherical_harmonics_cunningham<P: IntoPosition>(
    r_eci: P,
    R_i2b: SMatrix3,
    gravity_model: &GravityModel,
    n_max: usize,
    m_max: usize,
    parallel: ParallelMode,
) -> Vector3<f64> {
    // Extract position and compute body-fixed position
    let r = r_eci.position();
    let r_bf = R_i2b * r;

    // Compute spherical harmonic acceleration
    let a_ecef = gravity_model
        .compute_spherical_harmonics_cunningham(r_bf, n_max, m_max, parallel)
        .unwrap();

    // Inertial acceleration
    R_i2b.transpose() * a_ecef
}

/// Compute the acceleration due to gravity using the Clenshaw-summation
/// spherical harmonic kernel (see
/// [`GravityModel::compute_spherical_harmonics_clenshaw`]). The gravity
/// model is defined by the `GravityModel` struct. The acceleration is
/// computed in the body-fixed frame of the central body, and returned in the
/// inertial frame.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_eci`.
/// When a state vector is provided, only the position component is used.
///
/// # Arguments
///
/// - `r_eci` : Position vector of the object in the ECI frame, or state vector (position + velocity).
/// - `R_i2b` : Transformation matrix from the ECI frame to the body-fixed frame of the central body.
/// - `gravity_model` : Gravity model to use for the computation.
/// - `n_max` : Maximum degree of the gravity model to evaluate.
/// - `m_max` : Maximum order of the gravity model to evaluate.
///
/// # Returns
///
/// - `Vector3<f64>` : Acceleration due to gravity in the ECI frame.
///
/// # Examples
///
/// ```
/// use nalgebra::{Vector3, Vector6};
/// use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_koe_to_eci, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model
/// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
///
/// // Compute the acceleration due to gravity
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_koe_to_eci(oe, AngleFormat::Degrees);
/// let r_eci: Vector3<f64> = x_eci.fixed_rows::<3>(0).into();
///
/// // Compute the acceleration due to gravity
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics_clenshaw(r_eci, R_i2b, &gravity_model, 20, 20, ParallelMode::Auto);
/// ```
///
/// Using a state vector:
/// ```
/// use nalgebra::Vector6;
/// use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_koe_to_eci, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model
/// let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
///
/// // Compute the acceleration due to gravity using state vector directly
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_koe_to_eci(oe, AngleFormat::Degrees);
///
/// // Pass state vector directly - no need to extract position
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics_clenshaw(x_eci, R_i2b, &gravity_model, 20, 20, ParallelMode::Auto);
/// ```
#[allow(non_snake_case)]
pub fn accel_gravity_spherical_harmonics_clenshaw<P: IntoPosition>(
    r_eci: P,
    R_i2b: SMatrix3,
    gravity_model: &GravityModel,
    n_max: usize,
    m_max: usize,
    parallel: ParallelMode,
) -> Vector3<f64> {
    let r = r_eci.position();
    let r_bf = R_i2b * r;
    let a_ecef = gravity_model
        .compute_spherical_harmonics_clenshaw(r_bf, n_max, m_max, parallel)
        .unwrap();
    R_i2b.transpose() * a_ecef
}

/// Variant of [`accel_gravity_spherical_harmonics_cunningham`] that reuses
/// caller-supplied V and W work matrices to avoid per-call heap allocation.
///
/// Functionally identical to [`accel_gravity_spherical_harmonics_cunningham`] — see
/// that function's docs for argument semantics. The only difference is that the
/// inner `compute_spherical_harmonics_cunningham_with_workspace` call routes through
/// caller-supplied workspaces instead of allocating fresh ones each invocation.
/// Hot-path callers (the numerical propagator's dynamics closure) capture a
/// pair of `DMatrix<f64>` once at construction and reuse them across every
/// integrator stage.
///
/// # Panics
///
/// Panics if `gravity_model` has no precomputed Cunningham tables (see
/// [`GravityModel::precompute_cunningham_tables`] / [`GravityTables`]),
/// `n_max`/`m_max` are out of bounds, or the denormalized recursion
/// overflows to a non-finite result at high degree and low altitude.
///
/// # Examples
///
/// ```
/// use nalgebra::{DMatrix, Vector3};
/// use brahe::gravity::{GravityModel, GravityModelType, GravityTables, ParallelMode};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, TimeSystem};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
/// // Cunningham tables are not built by default; request them explicitly.
/// let gravity_model = GravityModel::from_model_type_with_tables(
///     &GravityModelType::EGM2008_360,
///     GravityTables::Cunningham,
/// )
/// .unwrap();
///
/// let r_eci = Vector3::new(R_EARTH + 500.0e3, 0.0, 0.0);
/// let mut v_workspace = DMatrix::<f64>::zeros(22, 22);
/// let mut w_workspace = DMatrix::<f64>::zeros(22, 22);
///
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics_cunningham_with_workspace(
///     r_eci, R_i2b, &gravity_model, 20, 20, ParallelMode::Auto, &mut v_workspace, &mut w_workspace,
/// );
/// ```
#[allow(non_snake_case)]
// Workspace-reuse variant: the V/W buffers plus model/degree/order/mode are all
// load-bearing, so the 8-arg count is intentional.
#[allow(clippy::too_many_arguments)]
pub fn accel_gravity_spherical_harmonics_cunningham_with_workspace<P: IntoPosition>(
    r_eci: P,
    R_i2b: SMatrix3,
    gravity_model: &GravityModel,
    n_max: usize,
    m_max: usize,
    parallel: ParallelMode,
    v_workspace: &mut DMatrix<f64>,
    w_workspace: &mut DMatrix<f64>,
) -> Vector3<f64> {
    let r = r_eci.position();
    let r_bf = R_i2b * r;

    let a_ecef = gravity_model
        .compute_spherical_harmonics_cunningham_with_workspace(
            r_bf,
            n_max,
            m_max,
            parallel,
            v_workspace,
            w_workspace,
        )
        .unwrap();

    R_i2b.transpose() * a_ecef
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::DVector;
    use rstest::rstest;
    use std::io::BufReader;

    use crate::constants::{GM_EARTH, R_EARTH};
    use crate::traits::DStatePropagator;
    use crate::utils::testing::setup_global_test_eop;
    use crate::{
        set_global_eop_provider, set_global_space_weather_provider, state_koe_to_eci, AngleFormat,
        DNumericalOrbitPropagator, EOPExtrapolation, Epoch, FileEOPProvider,
        FileSpaceWeatherProvider, ForceModelConfig, FrameTransformationModel, GravityConfiguration,
        GravityModelSource, NumericalPropagationConfig, SVector6, TimeSystem, ZonalHarmonicsDegree,
    };

    use super::*;

    #[test]
    fn test_gravity_model_from_file() {
        let filepath = Path::new("data/gravity_models/EGM2008_360.gfc");
        let gravity_model = GravityModel::from_file(filepath).unwrap();

        assert_eq!(gravity_model.model_name, "EGM2008");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 360);
        assert_eq!(gravity_model.m_max, 360);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::TideFree);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Calibrated);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_from_model_type_egm2008_360() {
        let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();

        assert_eq!(gravity_model.model_name, "EGM2008");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 360);
        assert_eq!(gravity_model.m_max, 360);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::TideFree);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Calibrated);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_from_model_type_ggm05s() {
        let gravity_model = GravityModel::from_model_type(&GravityModelType::GGM05S).unwrap();

        assert_eq!(gravity_model.model_name, "GGM05S");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 180);
        assert_eq!(gravity_model.m_max, 180);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::ZeroTide);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Calibrated);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_from_model_type_jgm3() {
        let gravity_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        assert_eq!(gravity_model.model_name, "JGM3");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 70);
        assert_eq!(gravity_model.m_max, 70);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::Unknown);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Formal);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    // ----- Process-wide gravity model cache (`shared`, `from_model_type`) -----
    //
    // These tests touch `GRAVITY_MODEL_CACHE`, which is module-global state.
    // They use `#[serial]` to avoid interleaving with each other or with
    // unrelated tests that happen to load gravity models concurrently.
    //
    // The fundamental property we want is "the cached model is structurally
    // identical to a fresh-load of the same model" — we do NOT assert
    // anything about timing, only about identity and data correctness.

    /// Helper: assert two models have the same identifying header fields.
    /// Stops short of comparing every coefficient — `test_cache_data_matches_uncached_load`
    /// covers data-matrix equality below.
    fn assert_models_equivalent(a: &GravityModel, b: &GravityModel) {
        assert_eq!(a.model_name, b.model_name);
        assert_eq!(a.gm, b.gm);
        assert_eq!(a.radius, b.radius);
        assert_eq!(a.n_max, b.n_max);
        assert_eq!(a.m_max, b.m_max);
        assert_eq!(a.tide_system, b.tide_system);
        assert_eq!(a.model_errors, b.model_errors);
        assert_eq!(a.normalization, b.normalization);
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_shared_returns_same_arc() {
        // After two calls for the same type, both Arcs should point at the
        // identical allocation — this is the proof that the cache is doing
        // what it claims and not just round-tripping through disk twice.
        clear_gravity_model_cache();
        let a = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        let b = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "shared() should return the same Arc on repeated calls for the same type"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_shared_per_type_isolation() {
        // Caching one type must not corrupt or alias entries for other types.
        // We deliberately load all three packaged types and check pairwise
        // distinctness via both Arc-pointer inequality and model-name inequality.
        clear_gravity_model_cache();
        let egm = GravityModel::shared(&GravityModelType::EGM2008_360).unwrap();
        let ggm = GravityModel::shared(&GravityModelType::GGM05S).unwrap();
        let jgm = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        assert!(!Arc::ptr_eq(&egm, &ggm));
        assert!(!Arc::ptr_eq(&egm, &jgm));
        assert!(!Arc::ptr_eq(&ggm, &jgm));
        assert_eq!(egm.model_name, "EGM2008");
        assert_eq!(ggm.model_name, "GGM05S");
        assert_eq!(jgm.model_name, "JGM3");
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_from_model_type_returns_independent_owners() {
        // Owned-clone API: each call returns a distinct, mutable model.
        // Truncating one must not bleed into the other (i.e. the cache must
        // hold the canonical Arc and `from_model_type` must clone out).
        clear_gravity_model_cache();
        let mut a = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let b = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        assert_eq!(a.n_max, 70);
        assert_eq!(b.n_max, 70);
        a.set_max_degree_order(30, 30).unwrap();
        assert_eq!(a.n_max, 30);
        assert_eq!(
            b.n_max, 70,
            "mutation on one owned clone must not affect another"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_from_model_type_independent_of_shared() {
        // Mutating a `from_model_type` clone must not contaminate the cached
        // Arc — otherwise the next `shared()` caller would see a truncated
        // model. Verifies the cache holds the canonical full-resolution copy.
        clear_gravity_model_cache();
        let _seed = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        let mut owned = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        owned.set_max_degree_order(10, 10).unwrap();

        let again = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        assert_eq!(
            again.n_max, 70,
            "cached Arc must remain at full resolution after truncation of a clone"
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_cache_data_matches_uncached_load() {
        // The whole point of the cache is to be a transparent acceleration.
        // Cross-check that the cached model and a fresh uncached load agree
        // on every coefficient — not just the header fields.
        clear_gravity_model_cache();
        let cached = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        let uncached = GravityModel::load_uncached(&GravityModelType::JGM3).unwrap();
        assert_models_equivalent(&cached, &uncached);
        // Spot-check a few coefficients that are stable across JGM3 versions.
        // J2 is by far the dominant term so a bug-typo in the cache would
        // show up here immediately.
        let (c_cached, s_cached) = cached.get(2, 0).unwrap();
        let (c_uncached, s_uncached) = uncached.get(2, 0).unwrap();
        assert_eq!(c_cached, c_uncached);
        assert_eq!(s_cached, s_uncached);
    }

    #[test]
    #[serial_test::serial]
    fn test_clear_gravity_model_cache_forces_reload() {
        // After clear(), the next shared() call must produce a fresh Arc
        // (different allocation) — confirming clear() actually drops entries
        // rather than just no-op'ing.
        clear_gravity_model_cache();
        let first = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        clear_gravity_model_cache();
        let second = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "after clear(), shared() must allocate a new Arc"
        );
        // But the data should still match — clear doesn't change disk contents.
        assert_models_equivalent(&first, &second);
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_cache_from_file_keys_by_path() {
        // The FromFile variant must hit the cache when the same path is
        // requested twice, just like the packaged variants.
        clear_gravity_model_cache();
        let path_str = "data/gravity_models/JGM3.gfc";
        let model_type = GravityModelType::FromFile(path_str.to_string());
        let a = GravityModel::shared(&model_type).unwrap();
        let b = GravityModel::shared(&model_type).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "FromFile sources should cache by their path string"
        );
        // And it should be a different cache slot from the packaged JGM3,
        // even though the file content is identical — the user-facing key
        // is the `GravityModelType` variant, not the parsed data.
        let packaged = GravityModel::shared(&GravityModelType::JGM3).unwrap();
        assert!(
            !Arc::ptr_eq(&a, &packaged),
            "FromFile(\"...JGM3.gfc\") and GravityModelType::JGM3 are distinct cache keys"
        );
        // But the data should be identical.
        assert_models_equivalent(&a, &packaged);
    }

    #[test]
    fn test_gravity_model_type_is_hash_eq() {
        // Static check: the cache won't compile (and hence neither will the
        // crate) without GravityModelType: Hash + Eq, but make the contract
        // explicit at the test level so a future refactor that drops these
        // derives fails here loudly rather than far downstream.
        fn assert_hash_eq<T: std::hash::Hash + Eq>() {}
        assert_hash_eq::<GravityModelType>();
    }

    #[test]
    fn test_gravity_model_get() {
        let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();

        let (c, s) = gravity_model.get(2, 0).unwrap();
        assert_abs_diff_eq!(c, -0.484165143790815e-03, epsilon = 1e-12);
        assert_abs_diff_eq!(s, 0.0, epsilon = 1e-12);

        let (c, s) = gravity_model.get(3, 3).unwrap();
        assert_abs_diff_eq!(c, 0.721321757121568e-06, epsilon = 1e-12);
        assert_abs_diff_eq!(s, 0.141434926192941e-05, epsilon = 1e-12);

        let (c, s) = gravity_model.get(360, 360).unwrap();
        assert_abs_diff_eq!(c, 0.200046056782130e-10, epsilon = 1e-12);
        assert_abs_diff_eq!(s, -0.958653755280305e-10, epsilon = 1e-12);

        let result = gravity_model.get(361, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_accel_point_mass_gravity() {
        let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_central_body = Vector3::new(0.0, 0.0, 0.0);

        let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);

        // Acceleration should be in the negative x-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);

        let r_object = Vector3::new(0.0, R_EARTH, 0.0);
        let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);

        // Acceleration should be in the negative y-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);

        let r_object = Vector3::new(0.0, 0.0, R_EARTH);
        let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);

        // Acceleration should be in the negative z-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);
    }

    #[test]
    fn test_accel_earth_zonal_gravity() {
        let earth_center = Vector3::new(0.0, 0.0, 0.0);

        // Equatorial surface position (z=0, so k/r=0)
        let r_equator = Vector3::new(R_EARTH, 0.0, 0.0);
        // Polar surface position (z=r, so k/r=1)
        let r_pole_radius = 6356752.0;
        let r_polar = Vector3::new(0.0, 0.0, r_pole_radius);

        // J2 should modify the acceleration relative to pure two-body.
        let a_point_mass_equator = accel_point_mass_gravity(r_equator, earth_center, GM_EARTH);
        let a_j2_equator = accel_earth_zonal_gravity(r_equator, 2);

        // At the equator the xy-factor (1 - 5(z/r)^2) is +1 and the J2 coefficient is negative,
        // so J2 strengthens the inward pull along x (a_j2 is more negative than the point-mass value).
        assert!(a_j2_equator[0] < a_point_mass_equator[0]);

        let a_point_mass_polar = accel_point_mass_gravity(r_polar, earth_center, GM_EARTH);
        let a_j2_polar = accel_earth_zonal_gravity(r_polar, 2);

        // At the pole the z-factor (3 - 5(z/r)^2) = -2 flips the sign, so J2 *weakens* the inward
        // pull along z (a_j2 is less negative than the point-mass value).
        assert!(a_j2_polar[2] > a_point_mass_polar[2]);

        // The z-factor magnitude at the pole (2) is twice the xy-factor magnitude at the equator (1),
        // so the |J2 perturbation| should be larger at the pole than at the equator at comparable r.
        let pole_pert = (a_j2_polar[2] - a_point_mass_polar[2]).abs();
        let equator_pert = (a_j2_equator[0] - a_point_mass_equator[0]).abs();
        assert!(pole_pert > equator_pert);
    }

    #[test]
    fn test_gravity_model_compute_spherical_harmonics_cunningham() {
        setup_global_test_eop();

        let gravity_model = GravityModel::from_model_type_with_tables(
            &GravityModelType::EGM2008_360,
            GravityTables::Both,
        )
        .unwrap();
        let r_body = Vector3::new(R_EARTH, 0.0, 0.0);

        // Simple test confirming point-mass equivalence
        let a_grav = gravity_model
            .compute_spherical_harmonics_cunningham(r_body, 0, 0, ParallelMode::Auto)
            .unwrap();
        assert_abs_diff_eq!(a_grav[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);

        // Test a more complex case
        let a_grav = gravity_model
            .compute_spherical_harmonics_cunningham(r_body, 60, 60, ParallelMode::Auto)
            .unwrap();
        assert_abs_diff_eq!(a_grav[0], -9.81433239, epsilon = 1e-8);
        assert_abs_diff_eq!(a_grav[1], 1.813976e-6, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], -7.29925652190e-5, epsilon = 1e-12);
    }

    #[test]
    fn test_clenshaw_matches_cunningham() {
        let model = GravityModel::from_model_type_with_tables(
            &GravityModelType::EGM2008_360,
            GravityTables::Both,
        )
        .unwrap();

        let positions = [
            Vector3::new(6.5e6, 1.2e6, 3.1e6),       // mid-latitude LEO
            Vector3::new(R_EARTH + 500e3, 0.0, 0.0), // equatorial LEO
            Vector3::new(0.0, 0.0, 7.0e6),           // exactly on the polar axis
            Vector3::new(1.0, 1.0, 7.0e6),           // 1 m off the polar axis
            Vector3::new(4.2164e7, 1.0e6, -2.0e6),   // GEO radius
        ];
        let degrees: [(usize, usize); 11] = [
            (2, 2),
            (5, 5),
            (10, 10),
            (20, 20),
            (40, 40),
            (80, 80),
            (90, 45),
            (120, 120),
            (160, 160),
            (200, 200),
            (200, 100),
        ];

        // Cunningham's unnormalized V/W recursion is not a valid reference at
        // every grid point: `V(m, m)` grows like `(2m-1)!!` while `q = radius / r`
        // is not small enough at LEO altitude to compensate, so it overflows to
        // a non-finite result (surfaced as an error, not silent NaN — see
        // `test_cunningham_high_degree_overflow_errors`) near n >~ 150
        // (positions 0 and 1) and has already lost several digits of accuracy
        // above n ~ 120 at exact-equator geometry (position 1), where the
        // acceleration's y-component is a near-total cancellation and
        // Cunningham's O(n) sequential-division coefficient denormalization
        // amplifies its own rounding error through that cancellation. Confirmed
        // against an independent 40-digit mpmath evaluation: Clenshaw matches the
        // high-precision reference to 4e-18..1.5e-16 at every one of these
        // points, while Cunningham reproduces exactly these residuals (or the
        // non-finite-overflow error). Those points are pinned against the
        // mpmath reference in `test_clenshaw_high_precision_reference` instead;
        // the 1e-10 bound below is unchanged (and still enforced) for every
        // other point in the grid.
        let excluded: &[(usize, usize, usize)] = &[
            // (position index, n, m)
            (0, 160, 160), // mid-lat LEO: cunningham returns a non-finite-overflow error
            (0, 200, 200), // mid-lat LEO: cunningham returns a non-finite-overflow error
            (1, 120, 120), // equatorial LEO: cunningham loses precision near equator
            (1, 160, 160), // equatorial LEO: cunningham returns a non-finite-overflow error
            (1, 200, 200), // equatorial LEO: cunningham returns a non-finite-overflow error
            (1, 200, 100), // equatorial LEO: cunningham loses precision near equator
        ];

        let mut worst_rel = 0.0_f64;
        for (pi, r_body) in positions.into_iter().enumerate() {
            for (n, m) in degrees {
                if excluded.contains(&(pi, n, m)) {
                    continue;
                }
                let a_cun = model
                    .compute_spherical_harmonics_cunningham(r_body, n, m, ParallelMode::Never)
                    .unwrap();
                let a_cl = model
                    .compute_spherical_harmonics_clenshaw(r_body, n, m, ParallelMode::Never)
                    .unwrap();
                let rel = (a_cun - a_cl).norm() / a_cun.norm();
                worst_rel = worst_rel.max(rel);
                assert!(
                    rel < 1e-10,
                    "r={r_body:?} n={n} m={m}: clenshaw/cunningham mismatch rel={rel:e}\n  cun={a_cun:?}\n  cl ={a_cl:?}"
                );
            }
        }
        eprintln!("test_clenshaw_matches_cunningham: worst-case relative residual = {worst_rel:e}");
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_clenshaw_high_precision_reference() {
        // Reference accelerations from an independent mpmath (40-digit)
        // evaluation of the same truncated EGM2008_360 sums (forward-column
        // normalized-ALF recurrence + analytic theta-derivative identity),
        // validated against the existing 80x80 characterization golden to
        // ~8e-16. Pins the Clenshaw kernel at exactly the (position, degree)
        // points excluded from `test_clenshaw_matches_cunningham` because the
        // Cunningham reference is degraded or NaN there and cannot validate
        // them itself.
        let model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
        let mid_lat_leo = Vector3::new(6.5e6, 1.2e6, 3.1e6);
        let equatorial_leo = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let cases: [(Vector3<f64>, usize, usize, Vector3<f64>); 6] = [
            (
                mid_lat_leo,
                160,
                160,
                Vector3::new(
                    -6.6591315830515751841,
                    -1.2294146745011928897,
                    -3.1837468168152788115,
                ),
            ),
            (
                mid_lat_leo,
                200,
                200,
                Vector3::new(
                    -6.6591315830515740167,
                    -1.2294146745011972695,
                    -3.1837468168152737831,
                ),
            ),
            (
                equatorial_leo,
                120,
                120,
                Vector3::new(
                    -8.4373560608626760559,
                    -2.3378538552167687078e-5,
                    3.0066034851689928739e-5,
                ),
            ),
            (
                equatorial_leo,
                160,
                160,
                Vector3::new(
                    -8.4373560595369940899,
                    -2.3377716801670622287e-5,
                    3.0068172954120995242e-5,
                ),
            ),
            (
                equatorial_leo,
                200,
                100,
                Vector3::new(
                    -8.4373560583252716778,
                    -2.3376084094462712969e-5,
                    3.0067260172799924015e-5,
                ),
            ),
            (
                equatorial_leo,
                200,
                200,
                Vector3::new(
                    -8.4373560595550198218,
                    -2.3377609235339850357e-5,
                    3.0068241451976491269e-5,
                ),
            ),
        ];

        let mut worst_rel = 0.0_f64;
        for (r_body, n, m, a_ref) in cases {
            let a_cl = model
                .compute_spherical_harmonics_clenshaw(r_body, n, m, ParallelMode::Never)
                .unwrap();
            let rel = (a_ref - a_cl).norm() / a_ref.norm();
            worst_rel = worst_rel.max(rel);
            assert!(
                rel < 1e-13,
                "r={r_body:?} n={n} m={m}: clenshaw vs mpmath mismatch rel={rel:e}\n  ref={a_ref:?}\n  cl ={a_cl:?}"
            );
        }
        eprintln!(
            "test_clenshaw_high_precision_reference: worst-case relative residual = {worst_rel:e}"
        );
    }

    #[test]
    fn test_clenshaw_point_mass_equivalence() {
        let model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
        let r_body = Vector3::new(R_EARTH, 0.0, 0.0);
        let a = model
            .compute_spherical_harmonics_clenshaw(r_body, 0, 0, ParallelMode::Never)
            .unwrap();
        assert_abs_diff_eq!(a[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_clenshaw_bounds_errors_match_cunningham() {
        let model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let r = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
        assert!(model
            .compute_spherical_harmonics_clenshaw(r, 100, 50, ParallelMode::Never)
            .is_err());
        assert!(model
            .compute_spherical_harmonics_clenshaw(r, 10, 20, ParallelMode::Never)
            .is_err());
    }

    #[test]
    fn test_spherical_harmonic_agrees_with_fast_zonal() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);

        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        set_global_space_weather_provider(sw);

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0);
        let state = state_koe_to_eci(oe, AngleFormat::Degrees);
        let dstate = DVector::from_column_slice(state.as_slice());
        let target = epoch + 86400.0;
        let degree = ZonalHarmonicsDegree::J6;

        let mut prop_spherical = DNumericalOrbitPropagator::new(
            epoch,
            dstate.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfig {
                gravity: GravityConfiguration::SphericalHarmonic {
                    source: GravityModelSource::default(),
                    degree: (&degree).into(),
                    order: 0,
                    parallel: crate::orbit_dynamics::ParallelMode::Auto,
                },
                drag: None,
                srp: None,
                third_body: None,
                relativity: false,
                mass: None,
                frame_transform: FrameTransformationModel::FullEarthRotation,
            },
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let mut prop_zonal_fast = DNumericalOrbitPropagator::new(
            epoch,
            dstate.clone(),
            NumericalPropagationConfig::default(),
            ForceModelConfig {
                gravity: GravityConfiguration::EarthZonal {
                    degree: ZonalHarmonicsDegree::J6,
                },
                drag: None,
                srp: None,
                third_body: None,
                relativity: false,
                mass: None,
                frame_transform: FrameTransformationModel::FullEarthRotation,
            },
            None,
            None,
            None,
            None,
        )
        .unwrap();

        prop_spherical.propagate_to(target);
        prop_zonal_fast.propagate_to(target);
        let s_state = prop_spherical.current_state();
        let z_state = prop_zonal_fast.current_state();
        print!("{} <> {}", s_state, z_state);

        // Sanity-only: both now rotate correctly so any remaining gap must be
        // smaller than the 3 km that the missing rotation used to cause.
        let eps_pos = 3_000.;
        let eps_v = 10.;

        assert_abs_diff_eq!(s_state[0], z_state[0], epsilon = eps_pos);
        assert_abs_diff_eq!(s_state[1], z_state[1], epsilon = eps_pos);
        assert_abs_diff_eq!(s_state[2], z_state[2], epsilon = eps_pos);
        assert_abs_diff_eq!(s_state[3], z_state[3], epsilon = eps_v);
        assert_abs_diff_eq!(s_state[4], z_state[4], epsilon = eps_v);
        assert_abs_diff_eq!(s_state[5], z_state[5], epsilon = eps_v);
    }

    /// Regression test: with `FullEarthRotation` both propagators transform to the same
    /// Earth-fixed frame before evaluating the harmonics, so the only remaining difference
    /// should be floating-point round-off between the explicit-formula and Cunningham V/W
    /// recursion implementations of J2–J6.
    #[serial_test::serial]
    fn test_fast_zonal_full_rotation_agrees_with_spherical_harmonic() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);

        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        set_global_space_weather_provider(sw);

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0);
        let state = state_koe_to_eci(oe, AngleFormat::Degrees);
        let dstate = DVector::from_column_slice(state.as_slice());
        let target = epoch + 3600.0;
        let degree = ZonalHarmonicsDegree::J6;

        let make_prop = |gravity| {
            DNumericalOrbitPropagator::new(
                epoch,
                dstate.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig {
                    gravity,
                    drag: None,
                    srp: None,
                    third_body: None,
                    relativity: false,
                    mass: None,
                    frame_transform: FrameTransformationModel::FullEarthRotation,
                },
                None,
                None,
                None,
                None,
            )
            .unwrap()
        };

        let mut prop_spherical = make_prop(GravityConfiguration::SphericalHarmonic {
            source: GravityModelSource::default(),
            degree: (&degree).into(),
            order: 0,
            parallel: crate::orbit_dynamics::ParallelMode::Auto,
        });
        let mut prop_zonal = make_prop(GravityConfiguration::EarthZonal { degree });

        prop_spherical.propagate_to(target);
        prop_zonal.propagate_to(target);

        let s = prop_spherical.current_state();
        let z = prop_zonal.current_state();

        let eps_pos = 1e-3;
        let eps_v = 1e-3;

        assert_abs_diff_eq!(s[0], z[0], epsilon = eps_pos);
        assert_abs_diff_eq!(s[1], z[1], epsilon = eps_pos);
        assert_abs_diff_eq!(s[2], z[2], epsilon = eps_pos);
        assert_abs_diff_eq!(s[3], z[3], epsilon = eps_v);
        assert_abs_diff_eq!(s[4], z[4], epsilon = eps_v);
        assert_abs_diff_eq!(s[5], z[5], epsilon = eps_v);
    }

    #[rstest]
    #[case(2, 2, - 6.97922756436, - 1.8292810538, - 2.69001658552)]
    #[case(3, 3, - 6.97926211185, - 1.82929165145, - 2.68998602761)]
    #[case(4, 4, - 6.97931189287, - 1.82931487069, - 2.6899914012)]
    #[case(5, 5, - 6.9792700471, - 1.82929795164, - 2.68997917147)]
    #[case(6, 6, - 6.979220667, - 1.8292787808, - 2.68997263887)]
    #[case(7, 7, - 6.97925478463, - 1.82926946742, - 2.68999296889)]
    #[case(8, 8, - 6.97927699747, - 1.82928186346, - 2.68998582282)]
    #[case(9, 9, - 6.97925893036, - 1.82928170212, - 2.68997442046)]
    #[case(10, 10, - 6.97924447943, - 1.82928331386, - 2.68997524437)]
    #[case(11, 11, - 6.9792517591, - 1.82928094754, - 2.68998382906)]
    #[case(12, 12, - 6.97924725688, - 1.82928130662, - 2.68998625958)]
    #[case(13, 13, - 6.97924858679, - 1.82928591192, - 2.6899891726)]
    #[case(14, 14, - 6.97924919386, - 1.82928546814, - 2.68999164569)]
    #[case(15, 15, - 6.97925490319, - 1.82928469874, - 2.68999376747)]
    #[case(16, 16, - 6.97926211023, - 1.82928438361, - 2.68999719587)]
    #[case(17, 17, - 6.97926308133, - 1.82928484644, - 2.68999716187)]
    #[case(18, 18, - 6.97926208121, - 1.829284918, - 2.6899952379)]
    #[case(19, 19, - 6.97926229494, - 1.82928369323, - 2.68999256236)]
    #[case(20, 20, - 6.979261862, - 1.82928315091, - 2.68999053339)]
    fn test_accel_gravity_jgm3_validation(
        #[case] n: usize,
        #[case] m: usize,
        #[case] ax: f64,
        #[case] ay: f64,
        #[case] az: f64,
    ) {
        let rot = SMatrix3::identity();

        let gravity_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let r_body = Vector3::new(6525.919e3, 1710.416e3, 2508.886e3);

        let a_grav = accel_gravity_spherical_harmonics(
            r_body,
            rot,
            &gravity_model,
            n,
            m,
            ParallelMode::Auto,
        );

        // This could potentially be validated to a higher degree of accuracy, but currently the
        // parameters provided by the Satellite Orbits book are only accurate to seven decimal
        // places, so without using the exact same parameters, it's difficult to validate to a higher
        // degree of accuracy.
        let tol = 1e-7;
        assert_abs_diff_eq!(a_grav[0], ax, epsilon = tol);
        assert_abs_diff_eq!(a_grav[1], ay, epsilon = tol);
        assert_abs_diff_eq!(a_grav[2], az, epsilon = tol);
    }

    #[test]
    fn test_set_max_degree_order_basic() {
        // Load JGM3 (70x70) and truncate to 20x20
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        assert_eq!(model.n_max, 70);
        assert_eq!(model.m_max, 70);

        model.set_max_degree_order(20, 20).unwrap();

        assert_eq!(model.n_max, 20);
        assert_eq!(model.m_max, 20);
        // Matrix should be resized to (21, 21)
        assert_eq!(model.data.nrows(), 21);
        assert_eq!(model.data.ncols(), 21);
    }

    #[test]
    fn test_set_max_degree_order_coefficient_preservation() {
        // Load JGM3 and verify coefficients are preserved after truncation
        let original_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let mut truncated_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        // Get some coefficients before truncation
        let (c_2_0_orig, s_2_0_orig) = original_model.get(2, 0).unwrap();
        let (c_3_3_orig, s_3_3_orig) = original_model.get(3, 3).unwrap();
        let (c_10_5_orig, s_10_5_orig) = original_model.get(10, 5).unwrap();
        let (c_20_20_orig, s_20_20_orig) = original_model.get(20, 20).unwrap();

        // Truncate to 20x20
        truncated_model.set_max_degree_order(20, 20).unwrap();

        // Verify coefficients are preserved
        let (c_2_0, s_2_0) = truncated_model.get(2, 0).unwrap();
        assert_abs_diff_eq!(c_2_0, c_2_0_orig, epsilon = 1e-15);
        assert_abs_diff_eq!(s_2_0, s_2_0_orig, epsilon = 1e-15);

        let (c_3_3, s_3_3) = truncated_model.get(3, 3).unwrap();
        assert_abs_diff_eq!(c_3_3, c_3_3_orig, epsilon = 1e-15);
        assert_abs_diff_eq!(s_3_3, s_3_3_orig, epsilon = 1e-15);

        let (c_10_5, s_10_5) = truncated_model.get(10, 5).unwrap();
        assert_abs_diff_eq!(c_10_5, c_10_5_orig, epsilon = 1e-15);
        assert_abs_diff_eq!(s_10_5, s_10_5_orig, epsilon = 1e-15);

        let (c_20_20, s_20_20) = truncated_model.get(20, 20).unwrap();
        assert_abs_diff_eq!(c_20_20, c_20_20_orig, epsilon = 1e-15);
        assert_abs_diff_eq!(s_20_20, s_20_20_orig, epsilon = 1e-15);

        // Verify coefficients beyond truncation limit are now inaccessible
        assert!(truncated_model.get(21, 0).is_err());
        assert!(truncated_model.get(70, 70).is_err());
    }

    #[test]
    fn test_set_max_degree_order_validation_m_greater_than_n() {
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        // m > n should error
        let result = model.set_max_degree_order(10, 15);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BraheError::Error(_)));
    }

    #[test]
    fn test_set_max_degree_order_validation_n_exceeds_max() {
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        // n > n_max (70 for JGM3) should error
        let result = model.set_max_degree_order(100, 100);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BraheError::OutOfBoundsError(_)
        ));
    }

    #[test]
    fn test_set_max_degree_order_validation_m_exceeds_max() {
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        // First truncate to (50, 40) - n_max=50, m_max=40
        model.set_max_degree_order(50, 40).unwrap();
        assert_eq!(model.n_max, 50);
        assert_eq!(model.m_max, 40);

        // Now try to set m > m_max (40) but m <= n (valid m <= n)
        let result = model.set_max_degree_order(50, 45);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BraheError::OutOfBoundsError(_)
        ));
    }

    #[test]
    fn test_set_max_degree_order_no_change() {
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let original_size = model.data.nrows();

        // Setting same values should succeed without error
        model.set_max_degree_order(70, 70).unwrap();

        // Size should be unchanged
        assert_eq!(model.data.nrows(), original_size);
        assert_eq!(model.n_max, 70);
        assert_eq!(model.m_max, 70);
    }

    #[test]
    fn test_set_max_degree_order_asymmetric() {
        // Test with m < n (less common but valid)
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        model.set_max_degree_order(30, 20).unwrap();

        assert_eq!(model.n_max, 30);
        assert_eq!(model.m_max, 20);
        // Matrix is sized based on n
        assert_eq!(model.data.nrows(), 31);
        assert_eq!(model.data.ncols(), 31);

        // Can still access coefficients within the truncated range
        let (c, s) = model.get(20, 20).unwrap();
        assert!(c.is_finite());
        assert!(s.is_finite());

        // Can access coefficients with n > m_max but m <= m_max
        let (c, s) = model.get(30, 15).unwrap();
        assert!(c.is_finite());
        assert!(s.is_finite());
    }

    #[test]
    fn test_set_max_degree_order_computation_after_truncation() {
        setup_global_test_eop();

        // Load full model and truncate
        let mut truncated_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        truncated_model.set_max_degree_order(20, 20).unwrap();

        // Load fresh model for comparison
        let full_model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();

        let r_body = Vector3::new(6525.919e3, 1710.416e3, 2508.886e3);

        // Compute spherical harmonics up to 20x20 on both
        let a_truncated = truncated_model
            .compute_spherical_harmonics(r_body, 20, 20, ParallelMode::Auto)
            .unwrap();
        let a_full = full_model
            .compute_spherical_harmonics(r_body, 20, 20, ParallelMode::Auto)
            .unwrap();

        // Results should be identical
        assert_abs_diff_eq!(a_truncated[0], a_full[0], epsilon = 1e-15);
        assert_abs_diff_eq!(a_truncated[1], a_full[1], epsilon = 1e-15);
        assert_abs_diff_eq!(a_truncated[2], a_full[2], epsilon = 1e-15);
    }

    #[test]
    fn test_gravity_model_type_from_file_valid_path() {
        let model_type =
            GravityModelType::from_file("data/gravity_models/EGM2008_360.gfc").unwrap();
        assert!(matches!(model_type, GravityModelType::FromFile(_)));
    }

    #[test]
    fn test_gravity_model_type_from_file_nonexistent_path() {
        let result = GravityModelType::from_file("/nonexistent/path/to/model.gfc");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_gravity_model_type_from_file_directory_path() {
        let result = GravityModelType::from_file("data/gravity_models");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a file"));
    }

    #[test]
    #[serial_test::serial]
    fn test_gravity_model_type_icgem_variant_loads_jgm3() {
        use crate::datasets::icgem::ICGEMBody;

        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

        // Seed the icgem cache with a manually-placed gfc file so no network
        // fetch is required.
        let cache_dir =
            std::path::PathBuf::from(crate::utils::cache::get_icgem_cache_dir().unwrap());
        let model_dir = cache_dir.join("models").join("earth");
        std::fs::create_dir_all(&model_dir).unwrap();
        let gfc = std::fs::read("data/gravity_models/JGM3.gfc").unwrap();
        std::fs::write(model_dir.join("JGM3-70-x.gfc"), &gfc).unwrap();

        // Seed a fresh index so list/refresh doesn't fetch.
        let idx = crate::datasets::icgem::index::IndexFile {
            fetched_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            entries: vec![crate::datasets::icgem::IndexEntry {
                body: ICGEMBody::Earth,
                name: "JGM3".into(),
                year: Some(1996),
                degree: 70,
                download_path: "/getmodel/gfc/x/JGM3.gfc".into(),
            }],
        };
        let idx_path = crate::datasets::icgem::index::index_path_for(&ICGEMBody::Earth).unwrap();
        crate::datasets::icgem::index::write_index_file(&idx_path, &idx).unwrap();

        let mt = GravityModelType::ICGEMModel {
            body: ICGEMBody::Earth,
            name: "JGM3".into(),
        };
        let model = GravityModel::from_model_type(&mt).unwrap();
        assert_eq!(model.n_max, 70);

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_numerical_propagator_with_icgem_jgm3_model() {
        // End-to-end: a DNumericalOrbitPropagator built from a ForceModelConfig
        // whose gravity source is `GravityModelType::ICGEMModel { Earth, "JGM3" }`
        // must initialize cleanly and advance the state.
        use crate::datasets::icgem::ICGEMBody;
        use crate::propagators::{
            DNumericalOrbitPropagator, ForceModelConfig, GravityConfiguration, GravityModelSource,
            NumericalPropagationConfig,
        };

        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);

        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

        // Seed the ICGEM cache with JGM3 so no network fetch is required.
        let cache_dir =
            std::path::PathBuf::from(crate::utils::cache::get_icgem_cache_dir().unwrap());
        let model_dir = cache_dir.join("models").join("earth");
        std::fs::create_dir_all(&model_dir).unwrap();
        let gfc = std::fs::read("data/gravity_models/JGM3.gfc").unwrap();
        std::fs::write(model_dir.join("JGM3-70-x.gfc"), &gfc).unwrap();

        let idx = crate::datasets::icgem::index::IndexFile {
            fetched_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            entries: vec![crate::datasets::icgem::IndexEntry {
                body: ICGEMBody::Earth,
                name: "JGM3".into(),
                year: Some(1996),
                degree: 70,
                download_path: "/getmodel/gfc/x/JGM3.gfc".into(),
            }],
        };
        let idx_path = crate::datasets::icgem::index::index_path_for(&ICGEMBody::Earth).unwrap();
        crate::datasets::icgem::index::write_index_file(&idx_path, &idx).unwrap();

        // Drop the process-wide gravity cache so the propagator forces a fresh
        // load of the ICGEM model under BRAHE_CACHE rather than reusing a model
        // a prior test loaded under a different cache root.
        clear_gravity_model_cache();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0);
        let state = state_koe_to_eci(oe, AngleFormat::Degrees);
        let dstate = DVector::from_column_slice(state.as_slice());

        let icgem_model = GravityModelType::ICGEMModel {
            body: ICGEMBody::Earth,
            name: "JGM3".into(),
        };
        let force_model = ForceModelConfig {
            gravity: GravityConfiguration::SphericalHarmonic {
                source: GravityModelSource::ModelType(icgem_model),
                degree: 20,
                order: 20,
                parallel: crate::orbit_dynamics::ParallelMode::Auto,
            },
            drag: None,
            srp: None,
            third_body: None,
            relativity: false,
            mass: None,
            frame_transform: FrameTransformationModel::FullEarthRotation,
        };

        // Construction must succeed: the propagator has to be able to resolve
        // the ICGEMModel variant through download_icgem_model → from_gfc_file.
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            dstate.clone(),
            NumericalPropagationConfig::default(),
            force_model,
            None,
            None,
            None,
            None,
        )
        .expect("propagator construction with ICGEMModel must succeed");

        // Step once to confirm the propagator actually runs and the gravity
        // model is wired in (a wiring failure typically shows up here as a
        // panic or NaN state).
        let target = epoch + 60.0;
        prop.propagate_to(target);
        let final_state = prop.current_state();

        assert_eq!(final_state.len(), 6);
        for i in 0..6 {
            assert!(
                final_state[i].is_finite(),
                "state element {} is non-finite after 60 s ICGEM-JGM3 propagation",
                i
            );
        }
        // Position should have moved by at least ~100 km in 60 s at LEO orbital
        // speed (~7.6 km/s).
        let dx = final_state[0] - dstate[0];
        let dy = final_state[1] - dstate[1];
        let dz = final_state[2] - dstate[2];
        let drift = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(drift > 100e3, "state barely moved: drift = {} m", drift);

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }

    #[test]
    fn test_should_parallelize_modes() {
        // Always / Never ignore n_max
        assert!(should_parallelize(ParallelMode::Always, 0));
        assert!(should_parallelize(ParallelMode::Always, 1000));
        assert!(!should_parallelize(ParallelMode::Never, 0));
        assert!(!should_parallelize(ParallelMode::Never, 1000));

        // Runs on the main thread (off-pool), so Auto's nested-parallelism guard
        // is satisfied and Auto reduces to the degree-threshold check.
        assert!(!should_parallelize(
            ParallelMode::Auto,
            PARALLEL_THRESHOLD_NMAX - 1
        ));
        assert!(should_parallelize(
            ParallelMode::Auto,
            PARALLEL_THRESHOLD_NMAX
        ));
        assert!(should_parallelize(
            ParallelMode::Auto,
            PARALLEL_THRESHOLD_NMAX + 1
        ));
    }

    #[test]
    fn test_should_parallelize_clenshaw_modes() {
        assert!(should_parallelize_clenshaw(ParallelMode::Always, 0));
        assert!(!should_parallelize_clenshaw(ParallelMode::Never, 1000));
        assert!(!should_parallelize_clenshaw(
            ParallelMode::Auto,
            CLENSHAW_PARALLEL_THRESHOLD_NMAX - 1
        ));
        assert!(should_parallelize_clenshaw(
            ParallelMode::Auto,
            CLENSHAW_PARALLEL_THRESHOLD_NMAX
        ));
    }

    #[test]
    fn test_clenshaw_parallel_bitwise_matches_serial() {
        let model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
        let positions = [
            Vector3::new(6.5e6, 1.2e6, 3.1e6),
            Vector3::new(0.0, 0.0, 7.0e6),
        ];
        for r_body in positions {
            for &(n, m) in &[(10usize, 10usize), (60, 60), (90, 45), (240, 240)] {
                let serial = model
                    .compute_spherical_harmonics_clenshaw(r_body, n, m, ParallelMode::Never)
                    .unwrap();
                let parallel = model
                    .compute_spherical_harmonics_clenshaw(r_body, n, m, ParallelMode::Always)
                    .unwrap();
                // Per-order sweeps are independent and the outer combine is
                // identical, so results must be bit-for-bit equal.
                assert_eq!(
                    serial.as_slice(),
                    parallel.as_slice(),
                    "r={r_body:?} n={n} m={m}: parallel result not bitwise equal"
                );
            }
        }
    }

    #[test]
    fn test_parallel_mode_default_is_auto() {
        assert_eq!(ParallelMode::default(), ParallelMode::Auto);
    }

    #[test]
    fn test_parallel_matches_serial_spherical_harmonics() {
        let model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();

        // Off-axis and on-axis (x0=y0=0 zeroes the sectoral/tesseral seeds).
        let positions = [
            Vector3::new(6.5e6, 1.2e6, 3.1e6),
            Vector3::new(0.0, 0.0, 7.0e6),
        ];

        for r_body in positions {
            // Cover below-threshold and above-threshold sizes, square and m<n.
            for &(n, m) in &[(10usize, 10usize), (60, 60), (90, 45), (120, 120)] {
                let serial = model
                    .compute_spherical_harmonics(r_body, n, m, ParallelMode::Never)
                    .unwrap();
                let parallel = model
                    .compute_spherical_harmonics(r_body, n, m, ParallelMode::Always)
                    .unwrap();
                let rel = (serial - parallel).norm() / serial.norm();
                assert!(
                    rel < 1e-12,
                    "r={r_body:?} n={n} m={m}: parallel/serial mismatch rel={rel:e}"
                );
            }
        }
    }

    #[test]
    fn test_serial_spherical_harmonics_characterization() {
        // Golden values captured from the reference (pre-optimization) serial
        // implementation. This guards the optimized serial recurrence and
        // accumulation against numerical regressions at the small degrees
        // common to LEO propagation (5, 20, 80). The relative tolerance (1e-12)
        // is loose enough to absorb the ~1e-15 FP-reordering introduced by
        // replacing the inner-loop division with a precomputed reciprocal
        // multiply, but far tighter than any real algorithmic bug would survive.
        // Specifically characterizes the Cunningham (V/W recursion) serial path.
        let model = GravityModel::from_model_type_with_tables(
            &GravityModelType::EGM2008_360,
            GravityTables::Both,
        )
        .unwrap();

        // (position, n_max==m_max, [expected ax, ay, az])
        let cases: [(Vector3<f64>, usize, [f64; 3]); 6] = [
            (
                Vector3::new(6.5e6, 1.2e6, 3.1e6),
                5,
                [
                    -6.659_157_617_724_3,
                    -1.229_429_462_451_389_3,
                    -3.183_740_444_887_946_3,
                ],
            ),
            (
                Vector3::new(6.5e6, 1.2e6, 3.1e6),
                20,
                [
                    -6.659_131_615_693_394,
                    -1.229_413_486_834_888,
                    -3.183_743_631_333_252,
                ],
            ),
            (
                Vector3::new(6.5e6, 1.2e6, 3.1e6),
                80,
                [
                    -6.659_131_583_250_136_5,
                    -1.229_414_674_519_354_8,
                    -3.183_746_816_300_017_5,
                ],
            ),
            (
                Vector3::new(0.0, 0.0, 7.0e6),
                5,
                [
                    4.832_073_418_678_780_7e-5,
                    -2.144_848_603_164_800_2e-5,
                    -8.112_882_851_092_754,
                ],
            ),
            (
                Vector3::new(0.0, 0.0, 7.0e6),
                20,
                [
                    8.160_586_191_674_914e-5,
                    -1.987_917_874_597_063e-5,
                    -8.112_905_372_087_614,
                ],
            ),
            (
                Vector3::new(0.0, 0.0, 7.0e6),
                80,
                [
                    8.241_307_452_961_249e-5,
                    -1.812_120_022_312_316e-5,
                    -8.112_900_111_910_852,
                ],
            ),
        ];

        for (r_body, n, expected) in cases {
            let a = model
                .compute_spherical_harmonics_cunningham(r_body, n, n, ParallelMode::Never)
                .unwrap();
            let exp = Vector3::new(expected[0], expected[1], expected[2]);
            let rel = (a - exp).norm() / a.norm();
            assert!(
                rel < 1e-12,
                "n={n} r={r_body:?}: got [{:.17e}, {:.17e}, {:.17e}] rel={rel:e}",
                a[0],
                a[1],
                a[2]
            );
        }
    }

    #[test]
    fn test_clenshaw_tables_packing_and_values() {
        // JGM3 is fully normalized: packed Clenshaw values must equal the raw
        // stored coefficients (no conversion applied).
        let model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let tables = model
            .clenshaw
            .as_ref()
            .expect("clenshaw tables built at load");

        assert_eq!(tables.n_stride, model.n_max);
        // C(0,0) = 1 sits at index 0.
        assert_eq!(tables.index(0, 0), 0);
        assert_abs_diff_eq!(tables.c[tables.index(0, 0)], 1.0, epsilon = 1e-15);

        // Spot-check a zonal, a tesseral, and a sectoral coefficient against
        // GravityModel::get (which returns the raw normalized values).
        for &(n, m) in &[(2usize, 0usize), (2, 1), (2, 2), (10, 5), (70, 70)] {
            let (c, s) = model.get(n, m).unwrap();
            let idx = tables.index(n, m);
            assert_abs_diff_eq!(tables.c[idx], c, epsilon = 1e-15);
            if m > 0 {
                assert_abs_diff_eq!(tables.s[idx - (tables.n_stride + 1)], s, epsilon = 1e-15);
            }
        }

        // sqrt table covers max(2*n_max + 5, 15) inclusive and holds sqrt(k).
        assert_eq!(tables.sqrt_table.len(), (2 * model.n_max + 5).max(15) + 1);
        assert_abs_diff_eq!(tables.sqrt_table[4], 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_clenshaw_tables_unnormalized_model_normalizes() {
        // A tiny unnormalized model: C20 = -1.0826e-3 (unnormalized J2-like).
        // The Clenshaw table must store C20 / sqrt(5) (the n=2, m=0
        // denormalization factor is sqrt(2n+1) = sqrt(5)).
        let gfc = "begin_of_head\n\
                   modelname test_unnorm\n\
                   earth_gravity_constant 3.986004415e14\n\
                   radius 6378136.3\n\
                   max_degree 2\n\
                   normalization unnormalized\n\
                   errors no\n\
                   end_of_head\n\
                   gfc 0 0 1.0 0.0\n\
                   gfc 2 0 -1.0826e-3 0.0\n";
        let model = GravityModel::from_bufreader(BufReader::new(gfc.as_bytes())).unwrap();
        let tables = model.clenshaw.as_ref().unwrap();
        assert_abs_diff_eq!(
            tables.c[tables.index(2, 0)],
            -1.0826e-3 / 5.0_f64.sqrt(),
            epsilon = 1e-18
        );
    }

    #[test]
    fn test_clenshaw_characterization() {
        // Golden values captured from the Clenshaw kernel behind the main
        // (compute_spherical_harmonics) dispatch API — the Clenshaw twin of
        // `test_serial_spherical_harmonics_characterization`, which
        // characterizes the Cunningham serial path directly.
        let model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();

        // (position, n_max==m_max, [expected ax, ay, az])
        let cases: [(Vector3<f64>, usize, [f64; 3]); 6] = [
            (
                Vector3::new(6.5e6, 1.2e6, 3.1e6),
                5,
                [
                    -6.659_157_617_724_302,
                    -1.229_429_462_451_39,
                    -3.183_740_444_887_948,
                ],
            ),
            (
                Vector3::new(6.5e6, 1.2e6, 3.1e6),
                20,
                [
                    -6.659_131_615_693_393_5,
                    -1.229_413_486_834_886_3,
                    -3.183_743_631_333_254,
                ],
            ),
            (
                Vector3::new(6.5e6, 1.2e6, 3.1e6),
                80,
                [
                    -6.659_131_583_250_132,
                    -1.229_414_674_519_354,
                    -3.183_746_816_300_018,
                ],
            ),
            (
                Vector3::new(0.0, 0.0, 7.0e6),
                5,
                [
                    4.832_073_418_678_783_4e-5,
                    -2.144_848_603_164_801_5e-5,
                    -8.112_882_851_092_756,
                ],
            ),
            (
                Vector3::new(0.0, 0.0, 7.0e6),
                20,
                [
                    8.160_586_191_674_913e-5,
                    -1.987_917_874_597_062_8e-5,
                    -8.112_905_372_087_614,
                ],
            ),
            (
                Vector3::new(0.0, 0.0, 7.0e6),
                80,
                [
                    8.241_307_452_961_25e-5,
                    -1.812_120_022_312_313_4e-5,
                    -8.112_900_111_910_857,
                ],
            ),
        ];

        for (r_body, n, expected) in cases {
            let a = model
                .compute_spherical_harmonics(r_body, n, n, ParallelMode::Never)
                .unwrap();
            let exp = Vector3::new(expected[0], expected[1], expected[2]);
            let rel = (a - exp).norm() / a.norm();
            assert!(
                rel < 1e-12,
                "n={n} r={r_body:?}: got [{:.17e}, {:.17e}, {:.17e}] rel={rel:e}",
                a[0],
                a[1],
                a[2]
            );
        }
    }

    #[test]
    fn test_gravity_model_compute_spherical_harmonics() {
        // Twin of the Cunningham variant above: guards the main (Clenshaw)
        // dispatch path with goldens captured from the Clenshaw kernel.
        setup_global_test_eop();
        let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
        let r_body = Vector3::new(R_EARTH, 0.0, 0.0);

        let a_grav = gravity_model
            .compute_spherical_harmonics(r_body, 0, 0, ParallelMode::Auto)
            .unwrap();
        assert_abs_diff_eq!(a_grav[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);

        let a_grav = gravity_model
            .compute_spherical_harmonics(r_body, 60, 60, ParallelMode::Auto)
            .unwrap();
        assert_abs_diff_eq!(a_grav[0], -9.814_332_397_930_517, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 1.813_976_428_513_033_4e-6, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], -7.299_256_521_901_194e-5, epsilon = 1e-12);
    }

    #[test]
    fn test_gravity_tables_default_is_clenshaw_only() {
        let model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        assert!(model.has_clenshaw_tables());
        assert!(!model.has_cunningham_tables());

        let r = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
        // Main API works (dispatches to Clenshaw)...
        assert!(
            model
                .compute_spherical_harmonics(r, 10, 10, ParallelMode::Never)
                .is_ok()
        );
        // ...explicit Cunningham does not.
        let err = model
            .compute_spherical_harmonics_cunningham(r, 10, 10, ParallelMode::Never)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Cunningham tables not precomputed")
        );
    }

    #[test]
    fn test_gravity_tables_load_variants() {
        let both =
            GravityModel::from_model_type_with_tables(&GravityModelType::JGM3, GravityTables::Both)
                .unwrap();
        assert!(both.has_clenshaw_tables() && both.has_cunningham_tables());

        let cun = GravityModel::from_model_type_with_tables(
            &GravityModelType::JGM3,
            GravityTables::Cunningham,
        )
        .unwrap();
        assert!(!cun.has_clenshaw_tables() && cun.has_cunningham_tables());
        // Main API falls back to Cunningham when it is the only table set.
        let r = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
        assert!(
            cun.compute_spherical_harmonics(r, 10, 10, ParallelMode::Never)
                .is_ok()
        );
    }

    #[test]
    fn test_gravity_tables_precompute_drop_roundtrip() {
        let mut model = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let r = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
        let a_clenshaw = model
            .compute_spherical_harmonics(r, 10, 10, ParallelMode::Never)
            .unwrap();

        model.precompute_cunningham_tables();
        assert!(model.has_cunningham_tables());
        let a_cun = model
            .compute_spherical_harmonics_cunningham(r, 10, 10, ParallelMode::Never)
            .unwrap();
        assert!((a_clenshaw - a_cun).norm() / a_cun.norm() < 1e-10);

        model.drop_clenshaw_tables();
        assert!(!model.has_clenshaw_tables());
        // Dispatch now falls back to Cunningham.
        assert!(
            model
                .compute_spherical_harmonics(r, 10, 10, ParallelMode::Never)
                .is_ok()
        );

        model.drop_cunningham_tables();
        let err = model
            .compute_spherical_harmonics(r, 10, 10, ParallelMode::Never)
            .unwrap_err();
        assert!(err.to_string().contains("No precomputed gravity tables"));
    }

    #[test]
    fn test_gravity_tables_truncation_rebuilds_existing_sets() {
        let mut model = GravityModel::from_model_type_with_tables(
            &GravityModelType::EGM2008_360,
            GravityTables::Both,
        )
        .unwrap();
        model.set_max_degree_order(70, 70).unwrap();
        assert!(model.has_clenshaw_tables() && model.has_cunningham_tables());
        let r = Vector3::new(6.5e6, 1.2e6, 3.1e6);
        let a_cl = model
            .compute_spherical_harmonics_clenshaw(r, 70, 70, ParallelMode::Never)
            .unwrap();
        let a_cun = model
            .compute_spherical_harmonics_cunningham(r, 70, 70, ParallelMode::Never)
            .unwrap();
        assert!((a_cl - a_cun).norm() / a_cun.norm() < 1e-10);
    }

    #[test]
    fn test_cunningham_high_degree_overflow_errors() {
        // Degree 160 at LEO altitude overflows the denormalized V/W recursion;
        // the kernel must surface a descriptive error, not silent NaN.
        let model = GravityModel::from_model_type_with_tables(
            &GravityModelType::EGM2008_360,
            GravityTables::Both,
        )
        .unwrap();
        let r = Vector3::new(6.5e6, 1.2e6, 3.1e6);
        let err = model
            .compute_spherical_harmonics_cunningham(r, 160, 160, ParallelMode::Never)
            .unwrap_err();
        assert!(err.to_string().contains("non-finite"));
        // GEO radius at the same degree stays finite (q^n decay wins): no error.
        let r_geo = Vector3::new(4.2164e7, 1.0e6, -2.0e6);
        assert!(
            model
                .compute_spherical_harmonics_cunningham(r_geo, 160, 160, ParallelMode::Never)
                .is_ok()
        );
    }
}
