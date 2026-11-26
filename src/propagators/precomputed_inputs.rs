/*!
 * Precomputed time-only inputs for the numerical orbit propagator.
 *
 * Tables are sampled on an absolute-epoch grid anchored at J2000 TT, so the
 * node at index `k` is always at `anchor + k * step` regardless of the
 * propagation start epoch. Tables hold only the nodes their interpolation
 * method reads for requested spans and extend contiguously in place.
 */

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;

use nalgebra::{SVector, Vector3};
use rayon::prelude::*;

use crate::earth_models::SpaceWeatherLookup;
use crate::frames::{bias_precession_nutation, earth_rotation, polar_motion};
use crate::math::SMatrix3;
use crate::math::interpolation::{InterpolationMethod, interpolate_hermite_cubic_svector6};
use crate::orbit_dynamics::ephemerides::{moon_position, sun_position};
use crate::orbit_dynamics::third_body::{
    third_body_position, third_body_position_geocentric, third_body_position_velocity,
    third_body_velocity_geocentric,
};
use crate::propagators::{
    AtmosphericModel, CentralBody, EclipseModel, EphemerisSource, ForceModelConfig,
    FrameTransformationModel, GravityConfiguration, ThirdBody,
};
use crate::space_weather::{
    FileSpaceWeatherProvider, get_global_ap_all, get_global_ap_daily, get_global_f107_obs_avg81,
    get_global_f107_observed,
};
use crate::spice::registry::ensure_bodies_loadable;
use crate::spice::{
    SPICEKernel, moon_position_spice, moon_velocity_spice, spk_acceleration, spk_position,
    spk_state, spk_velocity, sun_position_spice, sun_velocity_spice,
};
use crate::time::{Epoch, TimeSystem};
use crate::utils::errors::BraheError;
use crate::utils::threading::get_thread_pool;

/// Epoch of node index zero for every precomputed grid (J2000, TT).
pub(crate) fn grid_anchor() -> Epoch {
    Epoch::from_jd(2451545.0, TimeSystem::TT)
}

/// Tolerance, in grid steps, within which an epoch offset snaps to the nearest node.
const NODE_SNAP_TOLERANCE: f64 = 1e-9;

/// Inclusive range of grid node indices on a uniform step.
///
/// The range is empty when `k_min > k_max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NodeRange {
    /// Grid spacing in seconds.
    pub(crate) step: f64,
    /// First node index.
    pub(crate) k_min: i64,
    /// Last node index (inclusive).
    pub(crate) k_max: i64,
}

impl NodeRange {
    /// Node range whose nodes bracket the span between two epochs.
    ///
    /// The epochs may be given in either order.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `NodeRange`: Nodes `floor(x_first)` through `floor(x_last) + 1`, where `x` is the
    ///   epoch offset from the grid anchor in units of `step`
    pub(crate) fn bracketing(step: f64, start: Epoch, end: Epoch) -> NodeRange {
        let (first, last) = if end < start {
            (end, start)
        } else {
            (start, end)
        };
        let (k_first, _) = Self::locate(step, first);
        let (k_last, _) = Self::locate(step, last);
        NodeRange {
            step,
            k_min: k_first,
            k_max: k_last + 1,
        }
    }

    /// Epoch of node `k`.
    ///
    /// # Arguments
    /// - `k`: Node index
    ///
    /// # Returns
    /// - `Epoch`: `anchor + k * step`
    pub(crate) fn epoch(&self, k: i64) -> Epoch {
        grid_anchor() + k as f64 * self.step
    }

    /// Interval index and fractional position of an epoch.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    /// - `epoch`: Epoch to locate
    ///
    /// # Returns
    /// - `(i64, f64)`: Index `k = floor(x)` and fraction `s = x - k` in `[0, 1)`, where `x` is
    ///   the epoch offset from the grid anchor in steps, snapped to the nearest integer when
    ///   within `1e-9` steps so node epochs return `s = 0` exactly
    pub(crate) fn locate(step: f64, epoch: Epoch) -> (i64, f64) {
        let mut x = (epoch - grid_anchor()) / step;
        let nearest = x.round();
        if (x - nearest).abs() <= NODE_SNAP_TOLERANCE {
            x = nearest;
        }
        let k = x.floor();
        (k as i64, x - k)
    }

    fn contains(&self, other: &NodeRange) -> bool {
        self.k_min <= other.k_min && other.k_max <= self.k_max
    }
}

fn lerp(a: &SMatrix3, b: &SMatrix3, s: f64) -> SMatrix3 {
    a + (b - a) * s
}

/// Interpolation method of a table and the grid nodes it reads around a
/// query epoch.
///
/// A query at grid offset `x = k + s` (`k = floor(x)`, `s` in `[0, 1)`)
/// reads nodes `k` and `k + 1` for `Linear`, `HermiteCubic` and
/// `HermiteQuintic`. `Lagrange { degree: d }` reads the `d + 1` nodes centred
/// on the query: `k - (d - 1) / 2 ..= k + (d + 1) / 2` for odd `d`, and for
/// even `d` the `d + 1` nodes centred on the nearest node, `k` when
/// `s <= 0.5` and `k + 1` otherwise.
///
/// Lagrange values use the barycentric formula on the uniform node offsets
/// `0..=d`, whose weights `(-1)^j * C(d, j)` depend only on the degree. They
/// are computed once here, so a lookup costs `d + 1` divisions and no
/// allocation, where the general
/// [`interpolate_lagrange_svector`](crate::math::interpolation::interpolate_lagrange_svector)
/// allocates and recomputes the weights on every call.
#[derive(Debug, Clone)]
struct Stencil {
    method: InterpolationMethod,
    weights: Vec<f64>,
}

impl Stencil {
    fn new(method: InterpolationMethod) -> Self {
        let weights = match method {
            InterpolationMethod::Lagrange { degree } => {
                let mut weights = Vec::with_capacity(degree + 1);
                let mut w = 1.0;
                for j in 0..=degree {
                    if j > 0 {
                        w = -w * (degree - j + 1) as f64 / j as f64;
                    }
                    weights.push(w);
                }
                weights
            }
            _ => Vec::new(),
        };
        Stencil { method, weights }
    }

    /// Node offsets `(lo, hi)` from `k = floor(x)` that a query anywhere in
    /// `[k, k + 1)` may read.
    fn reach(&self) -> (i64, i64) {
        match self.method {
            InterpolationMethod::Lagrange { degree } => {
                let d = degree as i64;
                if d % 2 == 1 {
                    (-(d - 1) / 2, (d + 1) / 2)
                } else {
                    (-d / 2, d / 2 + 1)
                }
            }
            _ => (0, 1),
        }
    }

    /// First of the `degree + 1` nodes a `Lagrange { degree }` query at
    /// `k + s` reads.
    fn lagrange_first(degree: usize, k: i64, s: f64) -> i64 {
        let d = degree as i64;
        if d % 2 == 1 {
            k - (d - 1) / 2
        } else if s > 0.5 {
            k + 1 - d / 2
        } else {
            k - d / 2
        }
    }

    /// Nodes every query between two epochs reads.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `NodeRange`: [`NodeRange::bracketing`] widened by the stencil's reach
    fn span(&self, step: f64, start: Epoch, end: Epoch) -> NodeRange {
        let (lo, hi) = self.reach();
        let bracket = NodeRange::bracketing(step, start, end);
        NodeRange {
            step,
            k_min: bracket.k_min + lo,
            k_max: bracket.k_max - 1 + hi,
        }
    }

    /// Barycentric Lagrange terms `w_j / (u - j)` for a query at offset `u`
    /// from the first node, with `u` not a node offset.
    fn lagrange_terms(&self, u: f64) -> impl Iterator<Item = f64> + '_ {
        self.weights
            .iter()
            .enumerate()
            .map(move |(j, w)| w / (u - j as f64))
    }
}

/// Evaluate `f` at every node of `range` in parallel on the brahe thread pool.
///
/// # Arguments
/// - `range`: Nodes to evaluate
/// - `f`: Function of the node epoch
///
/// # Returns
/// - `Vec<T>`: Values in node index order
///
/// # Examples
///
/// ```ignore
/// let range = NodeRange::bracketing(60.0, epoch, epoch + 3600.0);
/// let bpn: Vec<SMatrix3> = compute_parallel(range, bias_precession_nutation);
/// ```
fn compute_parallel<T: Send>(range: NodeRange, f: impl Fn(Epoch) -> T + Sync) -> Vec<T> {
    get_thread_pool().install(|| {
        (range.k_min..=range.k_max)
            .into_par_iter()
            .map(|k| f(range.epoch(k)))
            .collect()
    })
}

/// Contiguous run of node values on a uniform grid, from node `k_min` to
/// `k_min + nodes.len() - 1`.
#[derive(Debug, Clone)]
struct NodeStore<T> {
    step: f64,
    k_min: i64,
    nodes: VecDeque<T>,
}

impl<T> NodeStore<T> {
    fn new(step: f64) -> Self {
        NodeStore {
            step,
            k_min: 0,
            nodes: VecDeque::new(),
        }
    }

    /// Node range currently held, or `None` when empty.
    fn coverage(&self) -> Option<NodeRange> {
        if self.nodes.is_empty() {
            None
        } else {
            Some(NodeRange {
                step: self.step,
                k_min: self.k_min,
                k_max: self.k_min + self.nodes.len() as i64 - 1,
            })
        }
    }

    /// Whether the store holds every node of a range.
    ///
    /// # Arguments
    /// - `need`: Node range to check
    ///
    /// # Returns
    /// - `bool`: `true` when every node of `need` is held
    fn holds(&self, need: NodeRange) -> bool {
        self.coverage().is_some_and(|cov| cov.contains(&need))
    }

    /// Position in the store of node `first`, when nodes `first` through
    /// `first + n - 1` are all held.
    ///
    /// # Arguments
    /// - `first`: Index of the first node
    /// - `n`: Number of nodes
    ///
    /// # Returns
    /// - `Option<usize>`: Position of node `first` in `nodes`, or `None` when any of the nodes
    ///   is not held
    fn window(&self, first: i64, n: usize) -> Option<usize> {
        let i = first.checked_sub(self.k_min)?;
        if i < 0 || i as usize + n > self.nodes.len() {
            return None;
        }
        Some(i as usize)
    }

    /// Extend the store to cover `need`, computing only missing nodes.
    ///
    /// Missing nodes are appended or prepended in place. When `need` is disjoint
    /// from the current coverage, the nodes in the gap are computed too and the
    /// store stays contiguous, provided the gap is no longer than the current
    /// coverage (both in nodes). A larger gap replaces the contents.
    ///
    /// # Arguments
    /// - `need`: Node range that must be held after the call
    /// - `compute`: Values of the nodes of a range, in node index order
    ///
    /// # Returns
    /// - `Result<(), E>`: The first error from `compute`; the store then holds the nodes
    ///   computed before it
    fn ensure<E>(
        &mut self,
        need: NodeRange,
        mut compute: impl FnMut(NodeRange) -> Result<Vec<T>, E>,
    ) -> Result<(), E> {
        let Some(cov) = self.coverage() else {
            return self.replace(need, compute);
        };
        let len = self.nodes.len() as i64;
        if need.k_min - cov.k_max - 1 > len || cov.k_min - need.k_max - 1 > len {
            return self.replace(need, compute);
        }
        if need.k_max > cov.k_max {
            let new = compute(NodeRange {
                step: self.step,
                k_min: cov.k_max + 1,
                k_max: need.k_max,
            })?;
            self.nodes.extend(new);
        }
        if need.k_min < cov.k_min {
            let new = compute(NodeRange {
                step: self.step,
                k_min: need.k_min,
                k_max: cov.k_min - 1,
            })?;
            for node in new.into_iter().rev() {
                self.nodes.push_front(node);
            }
            self.k_min = need.k_min;
        }
        Ok(())
    }

    fn replace<E>(
        &mut self,
        need: NodeRange,
        compute: impl FnOnce(NodeRange) -> Result<Vec<T>, E>,
    ) -> Result<(), E> {
        let new = compute(need)?;
        self.nodes = new.into();
        self.k_min = need.k_min;
        Ok(())
    }

    /// Nodes `k` and `k + 1` bracketing an epoch and its fraction `s` between them.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to locate
    ///
    /// # Returns
    /// - `Option<(&T, &T, f64)>`: Node `k`, node `k + 1` and `s` in `[0, 1)`, or `None` when
    ///   either node is not held
    fn bracket(&self, epoch: Epoch) -> Option<(&T, &T, f64)> {
        let (k, s) = NodeRange::locate(self.step, epoch);
        let i = k.checked_sub(self.k_min)?;
        if i < 0 {
            return None;
        }
        let i = i as usize;
        Some((self.nodes.get(i)?, self.nodes.get(i + 1)?, s))
    }
}

/// Table of bias-precession-nutation and polar-motion matrices on the absolute
/// grid. Earth rotation angle is always evaluated exactly at the query epoch.
///
/// The matrices are interpolated elementwise with `Linear` or
/// `Lagrange { degree }`.
#[derive(Debug, Clone)]
pub(crate) struct FrameRotationTable {
    store: NodeStore<(SMatrix3, SMatrix3)>,
    stencil: Stencil,
    #[cfg(test)]
    pub(crate) nodes_computed: usize,
}

impl FrameRotationTable {
    /// Create an empty table with linear interpolation.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    pub(crate) fn new(step: f64) -> Self {
        FrameRotationTable {
            store: NodeStore::new(step),
            stencil: Stencil::new(InterpolationMethod::Linear),
            #[cfg(test)]
            nodes_computed: 0,
        }
    }

    /// Set the interpolation method of an empty table.
    ///
    /// # Arguments
    /// - `method`: `Linear` or `Lagrange { degree }` with `degree >= 1`; any other method
    ///   interpolates linearly
    ///
    /// # Returns
    /// - `FrameRotationTable`: The table using `method`
    pub(crate) fn with_interpolation(mut self, method: InterpolationMethod) -> Self {
        debug_assert!(self.store.nodes.is_empty());
        self.stencil = Stencil::new(method);
        self
    }

    /// Node range currently held, or `None` when empty.
    #[cfg(test)]
    pub(crate) fn coverage(&self) -> Option<NodeRange> {
        self.store.coverage()
    }

    /// Nodes the table reads for queries between two epochs.
    ///
    /// # Arguments
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `NodeRange`: Nodes bracketing the span, widened by the interpolation method's reach
    pub(crate) fn span(&self, start: Epoch, end: Epoch) -> NodeRange {
        self.stencil.span(self.store.step, start, end)
    }

    /// Extend the table to cover `need`, computing only missing nodes.
    ///
    /// Missing nodes are appended or prepended in place. When `need` is disjoint
    /// from the current coverage, the nodes in the gap are computed too and the
    /// table stays contiguous, provided the gap is no longer than the current
    /// coverage (both in nodes). A larger gap replaces the contents.
    ///
    /// # Arguments
    /// - `need`: Node range that must be held after the call
    pub(crate) fn ensure(&mut self, need: NodeRange) {
        let Ok(()) = self.store.ensure(need, |range| {
            #[cfg(test)]
            {
                self.nodes_computed += (range.k_max - range.k_min + 1) as usize;
            }
            Ok::<_, Infallible>(compute_parallel(range, |e| {
                (bias_precession_nutation(e), polar_motion(e))
            }))
        });
    }

    /// GCRF to ITRF rotation at an epoch.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Option<SMatrix3>`: `PM * ERA(epoch) * BPN`, with `PM` and `BPN` interpolated, or
    ///   `None` when a node the interpolation reads is not held
    pub(crate) fn rotation(&self, epoch: Epoch) -> Option<SMatrix3> {
        let InterpolationMethod::Lagrange { degree } = self.stencil.method else {
            let ((bpn0, pm0), (bpn1, pm1), s) = self.store.bracket(epoch)?;
            return Some(lerp(pm0, pm1, s) * earth_rotation(epoch) * lerp(bpn0, bpn1, s));
        };
        let (k, s) = NodeRange::locate(self.store.step, epoch);
        let first = Stencil::lagrange_first(degree, k, s);
        let i = self.store.window(first, degree + 1)?;
        let nodes = &self.store.nodes;
        if s == 0.0 {
            let (bpn, pm) = &nodes[i + (k - first) as usize];
            return Some(pm * earth_rotation(epoch) * bpn);
        }
        let u = (k - first) as f64 + s;
        let mut bpn = SMatrix3::zeros();
        let mut pm = SMatrix3::zeros();
        let mut den = 0.0;
        for (j, c) in self.stencil.lagrange_terms(u).enumerate() {
            let (bpn_j, pm_j) = &nodes[i + j];
            bpn += bpn_j * c;
            pm += pm_j * c;
            den += c;
        }
        Some((pm / den) * earth_rotation(epoch) * (bpn / den))
    }
}

/// Resolve the Sun and Moon ECI positions from an ephemeris source.
///
/// Mirrors the third-body block's resolution: [`EphemerisSource::LowPrecision`]
/// uses the analytic geocentric ephemerides; a SPICE source resolves through
/// the kernel it names (`sun_position_spice`/`moon_position_spice`, which
/// auto-load the kernel).
///
/// # Arguments
/// - `source`: Ephemeris source for both bodies
/// - `epoch`: Epoch to evaluate
///
/// # Returns
/// - `Result<(Vector3<f64>, Vector3<f64>), BraheError>`: `(r_sun_eci, r_moon_eci)` in meters
///
/// # Examples
///
/// ```ignore
/// let (r_sun, r_moon) = resolve_sun_moon_eci(EphemerisSource::DE440s, epoch)?;
/// ```
pub(crate) fn resolve_sun_moon_eci(
    source: EphemerisSource,
    epoch: Epoch,
) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
    match source {
        EphemerisSource::LowPrecision => Ok((sun_position(epoch), moon_position(epoch))),
        spice_source => {
            let kernel = SPICEKernel::try_from(spice_source)?;
            Ok((
                sun_position_spice(epoch, kernel)?,
                moon_position_spice(epoch, kernel)?,
            ))
        }
    }
}

/// Force-model site an ephemeris channel feeds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ChannelRole {
    /// Third-body entry at this index of [`ForceModelConfig::third_body`].
    ThirdBody(usize),
    /// Sun and Moon positions for the tidal corrections.
    TideSunMoon,
    /// Sun position for solar radiation pressure about a non-Earth central body.
    SrpSun,
    /// Solar-radiation-pressure occulter at this index of `occulting_bodies`.
    Occulter(usize),
    /// State of the attributed drag body relative to the central body.
    DragBody,
}

/// Ephemeris query whose values an ephemeris channel tabulates.
///
/// Each variant evaluates the function the dynamics call when no table is
/// present, so node values equal the untabulated values bit for bit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ChannelQuery {
    /// Geocentric position of the Sun, the Moon, a planet or a planetary-system
    /// barycenter, as resolved by `accel_third_body`.
    GeocentricThirdBody {
        body: ThirdBody,
        source: EphemerisSource,
    },
    /// Position of a perturber relative to a central body, as resolved by
    /// `accel_third_body_for_body`.
    ThirdBodyPosition {
        central: CentralBody,
        body: ThirdBody,
        source: EphemerisSource,
    },
    /// Geocentric Sun position followed by the geocentric Moon position.
    SunMoon { source: EphemerisSource },
    /// Position of NAIF body `target` relative to `center` across the loaded kernels.
    SpkPosition { target: i32, center: i32 },
    /// State of NAIF body `target` relative to `center` across the loaded kernels.
    SpkState { target: i32, center: i32 },
}

impl ChannelQuery {
    /// Number of values the query returns.
    ///
    /// # Returns
    /// - `usize`: 6 for `SunMoon` and `SpkState`, 3 otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let query = ChannelQuery::SunMoon { source: EphemerisSource::DE440s };
    /// assert_eq!(query.dim(), 6);
    /// ```
    pub(crate) fn dim(&self) -> usize {
        match self {
            ChannelQuery::SunMoon { .. } | ChannelQuery::SpkState { .. } => 6,
            ChannelQuery::GeocentricThirdBody { .. }
            | ChannelQuery::ThirdBodyPosition { .. }
            | ChannelQuery::SpkPosition { .. } => 3,
        }
    }

    /// Load the kernels an SPK query's bodies need, as the dynamics do before
    /// querying the registry.
    ///
    /// Does nothing for the other queries, whose ephemeris functions load
    /// their kernels themselves.
    ///
    /// # Returns
    /// - `Result<(), BraheError>`: The kernel load error, if any
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let query = ChannelQuery::SpkPosition { target: 499, center: 399 };
    /// query.load_kernels()?;
    /// ```
    pub(crate) fn load_kernels(&self) -> Result<(), BraheError> {
        match self {
            ChannelQuery::SpkPosition { target, center }
            | ChannelQuery::SpkState { target, center } => {
                ensure_bodies_loadable(&[*target, *center])
            }
            ChannelQuery::GeocentricThirdBody { .. }
            | ChannelQuery::ThirdBodyPosition { .. }
            | ChannelQuery::SunMoon { .. } => Ok(()),
        }
    }

    /// Evaluate the query at an epoch.
    ///
    /// SPK queries expect [`ChannelQuery::load_kernels`] to have run.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Result<Vec<f64>, BraheError>`: [`ChannelQuery::dim`] values; positions in meters,
    ///   velocities in meters per second
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let query = ChannelQuery::GeocentricThirdBody {
    ///     body: ThirdBody::Moon,
    ///     source: EphemerisSource::DE440s,
    /// };
    /// let r_moon = query.evaluate(epoch)?;
    /// assert_eq!(r_moon.len(), query.dim());
    /// ```
    pub(crate) fn evaluate(&self, epoch: Epoch) -> Result<Vec<f64>, BraheError> {
        Ok(match self {
            ChannelQuery::GeocentricThirdBody { body, source } => {
                let (r, _) = third_body_position_geocentric(body, *source, epoch)?;
                r.iter().copied().collect()
            }
            ChannelQuery::ThirdBodyPosition {
                central,
                body,
                source,
            } => third_body_position(central, body, *source, epoch)?
                .iter()
                .copied()
                .collect(),
            ChannelQuery::SunMoon { source } => {
                let (r_sun, r_moon) = resolve_sun_moon_eci(*source, epoch)?;
                r_sun.iter().chain(r_moon.iter()).copied().collect()
            }
            ChannelQuery::SpkPosition { target, center } => spk_position(*target, *center, epoch)?
                .iter()
                .copied()
                .collect(),
            ChannelQuery::SpkState { target, center } => spk_state(*target, *center, epoch)?
                .iter()
                .copied()
                .collect(),
        })
    }

    /// Evaluate the query and its time derivative at an epoch.
    ///
    /// The values equal those of [`ChannelQuery::evaluate`] bit for bit. The
    /// derivatives come from the same ephemeris source: velocities for
    /// position queries, and velocity followed by acceleration for
    /// `SpkState`. SPK queries expect [`ChannelQuery::load_kernels`] to have
    /// run.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Result<Vec<f64>, BraheError>`: [`ChannelQuery::dim`] values followed by their
    ///   [`ChannelQuery::dim`] time derivatives; positions in meters, velocities in meters per
    ///   second, accelerations in meters per second squared. Queries with an
    ///   [`EphemerisSource::LowPrecision`] source return an error, as they provide no velocity
    pub(crate) fn evaluate_with_derivative(&self, epoch: Epoch) -> Result<Vec<f64>, BraheError> {
        let (value, derivative): (Vec<f64>, Vec<f64>) = match self {
            ChannelQuery::GeocentricThirdBody { body, source } => {
                let (r, _) = third_body_position_geocentric(body, *source, epoch)?;
                let v = third_body_velocity_geocentric(body, *source, epoch)?;
                (r.iter().copied().collect(), v.iter().copied().collect())
            }
            ChannelQuery::ThirdBodyPosition {
                central,
                body,
                source,
            } => {
                let (r, v) = third_body_position_velocity(central, body, *source, epoch)?;
                (r.iter().copied().collect(), v.iter().copied().collect())
            }
            ChannelQuery::SunMoon { source } => {
                let (r_sun, r_moon) = resolve_sun_moon_eci(*source, epoch)?;
                let kernel = SPICEKernel::try_from(*source)?;
                let v_sun = sun_velocity_spice(epoch, kernel)?;
                let v_moon = moon_velocity_spice(epoch, kernel)?;
                (
                    r_sun.iter().chain(r_moon.iter()).copied().collect(),
                    v_sun.iter().chain(v_moon.iter()).copied().collect(),
                )
            }
            ChannelQuery::SpkPosition { target, center } => (
                spk_position(*target, *center, epoch)?
                    .iter()
                    .copied()
                    .collect(),
                spk_velocity(*target, *center, epoch)?
                    .iter()
                    .copied()
                    .collect(),
            ),
            ChannelQuery::SpkState { target, center } => {
                let x = spk_state(*target, *center, epoch)?;
                let a = spk_acceleration(*target, *center, epoch)?;
                (
                    x.iter().copied().collect(),
                    x.iter().skip(3).chain(a.iter()).copied().collect(),
                )
            }
        };
        let mut node = value;
        node.extend(derivative);
        Ok(node)
    }
}

/// SPICE-backed ephemeris queries a force model makes, in a fixed order.
///
/// Lists the third-body entries in configuration order, then the tidal
/// Sun/Moon positions, the SRP Sun position for a non-Earth central body, the
/// SRP occulters not co-located with the central body (when an eclipse model
/// is set), and the attributed drag-body state when it differs from the
/// central body. Queries with an [`EphemerisSource::LowPrecision`] source are
/// left out. Earth-central point-mass entries for the Sun, the Moon, the
/// planets and the planetary-system barycenters use
/// [`ChannelQuery::GeocentricThirdBody`]; every other third-body entry uses
/// [`ChannelQuery::ThirdBodyPosition`].
///
/// # Arguments
/// - `cfg`: Force model configuration
///
/// # Returns
/// - `Vec<(ChannelRole, ChannelQuery)>`: One entry per tabulated query
///
/// # Examples
///
/// ```ignore
/// let channels = ephemeris_channels(&ForceModelConfig::default());
/// assert_eq!(channels[0].0, ChannelRole::ThirdBody(0));
/// ```
pub(crate) fn ephemeris_channels(cfg: &ForceModelConfig) -> Vec<(ChannelRole, ChannelQuery)> {
    let central = &cfg.central_body;
    let mut channels = Vec::new();

    for (idx, entry) in cfg.third_body.iter().flatten().enumerate() {
        if entry.ephemeris_source == EphemerisSource::LowPrecision {
            continue;
        }
        let geocentric = matches!(central, CentralBody::Earth)
            && matches!(entry.gravity, GravityConfiguration::PointMass)
            && matches!(
                entry.body,
                ThirdBody::Sun
                    | ThirdBody::Moon
                    | ThirdBody::Mercury
                    | ThirdBody::Venus
                    | ThirdBody::Mars
                    | ThirdBody::MarsBarycenter
                    | ThirdBody::Jupiter
                    | ThirdBody::JupiterBarycenter
                    | ThirdBody::Saturn
                    | ThirdBody::SaturnBarycenter
                    | ThirdBody::Uranus
                    | ThirdBody::UranusBarycenter
                    | ThirdBody::Neptune
                    | ThirdBody::NeptuneBarycenter
            );
        let query = if geocentric {
            ChannelQuery::GeocentricThirdBody {
                body: entry.body.clone(),
                source: entry.ephemeris_source,
            }
        } else {
            ChannelQuery::ThirdBodyPosition {
                central: central.clone(),
                body: entry.body.clone(),
                source: entry.ephemeris_source,
            }
        };
        channels.push((ChannelRole::ThirdBody(idx), query));
    }

    if let Some(tides) = &cfg.tides
        && (tides.solid.is_some() || tides.ocean.is_some())
        && tides.ephemeris_source != EphemerisSource::LowPrecision
    {
        channels.push((
            ChannelRole::TideSunMoon,
            ChannelQuery::SunMoon {
                source: tides.ephemeris_source,
            },
        ));
    }

    if let Some(srp) = &cfg.srp {
        if !matches!(central, CentralBody::Earth) {
            channels.push((
                ChannelRole::SrpSun,
                ChannelQuery::SpkPosition {
                    target: 10,
                    center: central.naif_id(),
                },
            ));
        }
        if srp.eclipse_model != EclipseModel::None {
            for (idx, occ) in srp.occulting_bodies.iter().enumerate() {
                if occ.naif_position_id() != central.naif_id() {
                    channels.push((
                        ChannelRole::Occulter(idx),
                        ChannelQuery::SpkPosition {
                            target: occ.naif_position_id(),
                            center: central.naif_id(),
                        },
                    ));
                }
            }
        }
    }

    if let Some(drag) = &cfg.drag
        && let Some(body) = &drag.body
        && body.naif_id() != central.naif_id()
    {
        channels.push((
            ChannelRole::DragBody,
            ChannelQuery::SpkState {
                target: body.naif_id(),
                center: central.naif_id(),
            },
        ));
    }

    channels
}

/// Table of ephemeris channel values on the absolute grid.
///
/// Each node stores the values of every channel back to back; channel `i`
/// occupies `offsets[i]..offsets[i] + width`, where the width is
/// `queries[i].dim()`, or twice that with `HermiteCubic`, which stores the
/// channel's time derivatives after its values.
#[derive(Debug, Clone)]
pub(crate) struct EphemerisTable {
    store: NodeStore<Box<[f64]>>,
    queries: Vec<ChannelQuery>,
    offsets: Vec<usize>,
    stride: usize,
    stencil: Stencil,
}

impl EphemerisTable {
    /// Create an empty table with linear interpolation.
    ///
    /// # Arguments
    /// - `step`: Grid spacing in seconds
    /// - `queries`: Channel queries, in channel order
    ///
    /// # Returns
    /// - `EphemerisTable`: Table holding no nodes
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let queries = ephemeris_channels(&cfg).into_iter().map(|(_, q)| q).collect();
    /// let table = EphemerisTable::new(60.0, queries);
    /// ```
    pub(crate) fn new(step: f64, queries: Vec<ChannelQuery>) -> Self {
        let mut table = EphemerisTable {
            store: NodeStore::new(step),
            queries,
            offsets: Vec::new(),
            stride: 0,
            stencil: Stencil::new(InterpolationMethod::Linear),
        };
        table.layout();
        table
    }

    /// Set the interpolation method of an empty table.
    ///
    /// # Arguments
    /// - `method`: `Linear`, `Lagrange { degree }` with `degree >= 1`, or `HermiteCubic`; any
    ///   other method interpolates linearly
    ///
    /// # Returns
    /// - `EphemerisTable`: The table using `method`
    pub(crate) fn with_interpolation(mut self, method: InterpolationMethod) -> Self {
        debug_assert!(self.store.nodes.is_empty());
        self.stencil = Stencil::new(method);
        self.layout();
        self
    }

    /// Whether nodes store each channel's time derivatives after its values.
    fn stores_derivatives(&self) -> bool {
        self.stencil.method == InterpolationMethod::HermiteCubic
    }

    fn layout(&mut self) {
        let factor = if self.stores_derivatives() { 2 } else { 1 };
        self.offsets.clear();
        self.stride = 0;
        for query in &self.queries {
            self.offsets.push(self.stride);
            self.stride += factor * query.dim();
        }
    }

    /// Node range currently held, or `None` when empty.
    #[cfg(test)]
    pub(crate) fn coverage(&self) -> Option<NodeRange> {
        self.store.coverage()
    }

    /// Nodes the table reads for queries between two epochs.
    ///
    /// # Arguments
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `NodeRange`: Nodes bracketing the span, widened by the interpolation method's reach
    pub(crate) fn span(&self, start: Epoch, end: Epoch) -> NodeRange {
        self.stencil.span(self.store.step, start, end)
    }

    /// Extend the table to cover `need`, computing only missing nodes.
    ///
    /// Follows the extension rules of [`FrameRotationTable::ensure`]. Each
    /// batch of missing nodes first loads the kernels of every query once,
    /// then computes the nodes in parallel on the brahe thread pool.
    ///
    /// # Arguments
    /// - `need`: Node range that must be held after the call
    ///
    /// # Returns
    /// - `Result<(), BraheError>`: The first kernel load error in channel order, else the error
    ///   of the first failing query, ordered by node index and then by channel; the nodes of a
    ///   failing batch are not stored
    ///
    /// # Examples
    ///
    /// ```ignore
    /// table.ensure(table.span(epoch, epoch + 3600.0))?;
    /// ```
    pub(crate) fn ensure(&mut self, need: NodeRange) -> Result<(), BraheError> {
        let queries = &self.queries;
        let stride = self.stride;
        let derivatives = self.stores_derivatives();
        self.store.ensure(need, |range| {
            for query in queries {
                query.load_kernels()?;
            }
            compute_parallel(range, |e| {
                let mut node = Vec::with_capacity(stride);
                for query in queries {
                    if derivatives {
                        node.extend(query.evaluate_with_derivative(e)?);
                    } else {
                        node.extend(query.evaluate(e)?);
                    }
                }
                Ok(node.into_boxed_slice())
            })
            .into_iter()
            .collect()
        })
    }

    /// Interpolated value of one channel.
    ///
    /// # Arguments
    /// - `channel`: Channel index
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Option<[f64; 6]>`: The channel's interpolated values in the first
    ///   [`ChannelQuery::dim`] entries and zeros after them, or `None` when the channel does
    ///   not exist or a node the interpolation reads is not held
    ///
    /// # Examples
    ///
    /// ```ignore
    /// table.ensure(table.span(epoch, epoch + 3600.0))?;
    /// let r_moon = table.value(0, epoch + 90.0).unwrap();
    /// ```
    pub(crate) fn value(&self, channel: usize, epoch: Epoch) -> Option<[f64; 6]> {
        let dim = self.queries.get(channel)?.dim();
        let off = self.offsets[channel];
        let mut out = [0.0; 6];
        match self.stencil.method {
            InterpolationMethod::Linear | InterpolationMethod::HermiteQuintic => {
                let (a, b, s) = self.store.bracket(epoch)?;
                for (o, (a, b)) in out
                    .iter_mut()
                    .zip(a[off..off + dim].iter().zip(&b[off..off + dim]))
                {
                    *o = a + (b - a) * s;
                }
            }
            InterpolationMethod::Lagrange { degree } => {
                let (k, s) = NodeRange::locate(self.store.step, epoch);
                let first = Stencil::lagrange_first(degree, k, s);
                let i = self.store.window(first, degree + 1)?;
                let nodes = &self.store.nodes;
                if s == 0.0 {
                    let node = &nodes[i + (k - first) as usize];
                    out[..dim].copy_from_slice(&node[off..off + dim]);
                    return Some(out);
                }
                let u = (k - first) as f64 + s;
                let mut den = 0.0;
                for (j, c) in self.stencil.lagrange_terms(u).enumerate() {
                    for (o, y) in out.iter_mut().zip(&nodes[i + j][off..off + dim]) {
                        *o += c * y;
                    }
                    den += c;
                }
                for o in &mut out[..dim] {
                    *o /= den;
                }
            }
            InterpolationMethod::HermiteCubic => {
                let (a, b, s) = self.store.bracket(epoch)?;
                if s == 0.0 {
                    out[..dim].copy_from_slice(&a[off..off + dim]);
                    return Some(out);
                }
                let h = self.store.step;
                for c in (0..dim).step_by(3) {
                    let node_state = |node: &[f64]| {
                        let value = &node[off + c..off + c + 3];
                        let derivative = &node[off + dim + c..off + dim + c + 3];
                        SVector::<f64, 6>::from_iterator(value.iter().chain(derivative).copied())
                    };
                    let x = interpolate_hermite_cubic_svector6(
                        0.0,
                        h,
                        node_state(a),
                        node_state(b),
                        s * h,
                    );
                    out[c..c + 3].copy_from_slice(&x.as_slice()[..3]);
                }
            }
        }
        Some(out)
    }
}

/// Identity of a [`PrecomputedInputs`] bundle.
///
/// Two propagators whose force models produce equal keys can share one bundle.
/// The key is derived from [`ForceModelConfig::precompute`] and the parts of the
/// force model that decide which tables apply. A frame-rotation table applies
/// only to an Earth central body with
/// [`FrameTransformationModel::FullEarthRotation`]. An ephemeris table applies
/// when the force model queries SPICE ephemerides; the key holds the ordered
/// list of those queries together with the force-model site each one feeds, so
/// force models querying the same bodies for different purposes get different
/// keys. Each table's step and interpolation method are part of the key.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PrecomputeKey {
    frame_rotation: Option<(f64, InterpolationMethod)>,
    ephemeris: Option<(f64, InterpolationMethod, EphemerisChannels)>,
}

/// Ordered ephemeris channels of a force model, each with the site it feeds.
type EphemerisChannels = Vec<(ChannelRole, ChannelQuery)>;

impl PrecomputeKey {
    /// Key for the tables a force model uses.
    ///
    /// # Arguments
    /// - `cfg`: Force model configuration
    ///
    /// # Returns
    /// - `Option<PrecomputeKey>`: `None` when `cfg.precompute` is `None` or no table applies
    pub(crate) fn from_force_config(cfg: &ForceModelConfig) -> Option<PrecomputeKey> {
        let pre = cfg.precompute.as_ref()?;
        let frame_rotation = match (&cfg.central_body, &cfg.frame_transform) {
            (CentralBody::Earth, FrameTransformationModel::FullEarthRotation) => pre
                .frame_rotation_step
                .map(|step| (step, pre.frame_rotation_interpolation)),
            _ => None,
        };
        let ephemeris = pre.ephemeris_step.and_then(|step| {
            let channels = ephemeris_channels(cfg);
            (!channels.is_empty()).then_some((step, pre.ephemeris_interpolation, channels))
        });
        if frame_rotation.is_none() && ephemeris.is_none() {
            return None;
        }
        Some(PrecomputeKey {
            frame_rotation,
            ephemeris,
        })
    }
}

/// Precomputed time-only force-model inputs for a [`PrecomputeKey`].
///
/// Holds the frame-rotation and ephemeris tables on the absolute grid. A
/// propagator extends the tables over each requested span before integrating
/// and on demand when the dynamics are evaluated at an epoch outside them.
/// Bundles are shared between propagators through `Arc` and copied on write
/// when one extends them.
#[derive(Debug, Clone)]
pub(crate) struct PrecomputedInputs {
    key: PrecomputeKey,
    rotation: Option<FrameRotationTable>,
    ephemeris: Option<EphemerisTable>,
}

impl PrecomputedInputs {
    /// Create a bundle with empty tables.
    ///
    /// # Arguments
    /// - `key`: Tables to hold and their grid steps
    ///
    /// # Returns
    /// - `PrecomputedInputs`: Bundle holding no nodes
    pub(crate) fn new(key: &PrecomputeKey) -> Self {
        PrecomputedInputs {
            key: key.clone(),
            rotation: key
                .frame_rotation
                .map(|(step, method)| FrameRotationTable::new(step).with_interpolation(method)),
            ephemeris: key.ephemeris.as_ref().map(|(step, method, channels)| {
                EphemerisTable::new(*step, channels.iter().map(|(_, q)| q.clone()).collect())
                    .with_interpolation(*method)
            }),
        }
    }

    /// Key this bundle was built for.
    ///
    /// # Returns
    /// - `&PrecomputeKey`: The bundle's key
    pub(crate) fn key(&self) -> &PrecomputeKey {
        &self.key
    }

    /// Whether every table holds the nodes its interpolation method reads for
    /// queries in the span between two epochs.
    ///
    /// # Arguments
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `bool`: `true` when no table needs extending to cover the span
    pub(crate) fn covers_span(&self, start: Epoch, end: Epoch) -> bool {
        self.rotation
            .as_ref()
            .is_none_or(|t| t.store.holds(t.span(start, end)))
            && self
                .ephemeris
                .as_ref()
                .is_none_or(|t| t.store.holds(t.span(start, end)))
    }

    /// Whether every table holds the nodes its interpolation method reads at an epoch.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to check
    ///
    /// # Returns
    /// - `bool`: `true` when each table holds the nodes its interpolation method reads for
    ///   queries in the grid interval containing `epoch`
    pub(crate) fn covers(&self, epoch: Epoch) -> bool {
        self.covers_span(epoch, epoch)
    }

    /// Extend every table to hold the nodes its interpolation method reads for
    /// queries in the span between two epochs.
    ///
    /// Does nothing when the span is already covered.
    ///
    /// # Arguments
    /// - `start`: One end of the span
    /// - `end`: The other end of the span
    ///
    /// # Returns
    /// - `Result<(), BraheError>`: `Ok(())` once every table covers the span, or the first
    ///   ephemeris query error
    pub(crate) fn ensure(&mut self, start: Epoch, end: Epoch) -> Result<(), BraheError> {
        if let Some(table) = self.ephemeris.as_mut() {
            table.ensure(table.span(start, end))?;
        }
        if let Some(table) = self.rotation.as_mut() {
            table.ensure(table.span(start, end));
        }
        Ok(())
    }

    /// GCRF to ITRF rotation from the frame-rotation table.
    ///
    /// # Arguments
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Option<SMatrix3>`: Interpolated rotation, or `None` when the bundle has no
    ///   frame-rotation table or the table does not bracket `epoch`
    pub(crate) fn rotation(&self, epoch: Epoch) -> Option<SMatrix3> {
        self.rotation.as_ref()?.rotation(epoch)
    }

    /// Ephemeris channel value for a force-model site.
    ///
    /// # Arguments
    /// - `role`: Force-model site the channel feeds
    /// - `epoch`: Epoch to evaluate
    ///
    /// # Returns
    /// - `Option<[f64; 6]>`: The channel's interpolated values in the first
    ///   [`ChannelQuery::dim`] entries and zeros after them, or `None` when the key lists no
    ///   channel for `role` or the table does not bracket `epoch`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inputs.ensure(epoch, epoch + 3600.0)?;
    /// let s = inputs.ephemeris(&ChannelRole::ThirdBody(0), epoch + 90.0).unwrap();
    /// ```
    pub(crate) fn ephemeris(&self, role: &ChannelRole, epoch: Epoch) -> Option<[f64; 6]> {
        let (_, _, channels) = self.key.ephemeris.as_ref()?;
        let channel = channels.iter().position(|(r, _)| r == role)?;
        self.ephemeris.as_ref()?.value(channel, epoch)
    }

    /// Number of frame-rotation nodes computed over the bundle's lifetime.
    #[cfg(test)]
    pub(crate) fn rotation_nodes_computed(&self) -> usize {
        self.rotation.as_ref().map_or(0, |t| t.nodes_computed)
    }

    /// Node range held by the frame-rotation table.
    #[cfg(test)]
    pub(crate) fn rotation_coverage(&self) -> Option<NodeRange> {
        self.rotation.as_ref()?.coverage()
    }
}

/// Space-weather values NRLMSISE-00 reads for one UT1 day.
///
/// A field is `None` until a lookup first reads it from the global provider.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SpaceWeatherDay {
    /// Daily Ap.
    pub(crate) ap_daily: Option<f64>,
    /// The eight 3-hourly Ap values, starting at 00:00 UT1.
    pub(crate) ap: Option<[f64; 8]>,
    /// Observed F10.7 [sfu].
    pub(crate) f107_observed: Option<f64>,
    /// 81-day centered average of observed F10.7 [sfu].
    pub(crate) f107_obs_avg81: Option<f64>,
}

/// Per-propagator memo of the global space-weather provider, keyed by UT1 day.
///
/// Values are the ones the provider returns for the day; nothing is
/// interpolated. Provider errors are not stored. Read through
/// [`CachedSpaceWeather`].
#[derive(Debug, Default)]
pub(crate) struct SpaceWeatherCache {
    days: HashMap<i64, SpaceWeatherDay>,
}

impl SpaceWeatherCache {
    /// Create an empty cache.
    ///
    /// # Returns
    /// - `SpaceWeatherCache`: Cache holding no days
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Whether a force model reads space weather through the cache.
    ///
    /// # Arguments
    /// - `cfg`: Force model configuration
    ///
    /// # Returns
    /// - `bool`: `true` when `cfg.precompute` sets `cache_space_weather` and drag uses
    ///   NRLMSISE-00
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cache = SpaceWeatherCache::applies(&force_config).then(SpaceWeatherCache::new);
    /// ```
    pub(crate) fn applies(cfg: &ForceModelConfig) -> bool {
        cfg.precompute.is_some_and(|p| p.cache_space_weather)
            && cfg
                .drag
                .as_ref()
                .is_some_and(|d| matches!(d.model, AtmosphericModel::NRLMSISE00(_)))
    }

    /// Stored values for a UT1 day.
    #[cfg(test)]
    pub(crate) fn day(&self, day: i64) -> Option<&SpaceWeatherDay> {
        self.days.get(&day)
    }
}

/// [`SpaceWeatherLookup`] reading through a [`SpaceWeatherCache`].
///
/// The lookup trait takes `&self`, so the mutable cache borrow sits in a
/// `RefCell`.
pub(crate) struct CachedSpaceWeather<'a>(RefCell<&'a mut SpaceWeatherCache>);

impl<'a> CachedSpaceWeather<'a> {
    /// Lookup reading through `cache`.
    ///
    /// # Arguments
    /// - `cache`: Cache to read and fill
    ///
    /// # Returns
    /// - `CachedSpaceWeather`: Lookup borrowing the cache
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut cache = SpaceWeatherCache::new();
    /// let sw = CachedSpaceWeather::new(&mut cache);
    /// let rho = density_nrlmsise00_geod_with(&epoch, &geod, &NRLMSISE00Config::default(), &sw)?;
    /// ```
    pub(crate) fn new(cache: &'a mut SpaceWeatherCache) -> Self {
        CachedSpaceWeather(RefCell::new(cache))
    }

    /// Cached field of the UT1 day containing `mjd_ut1`, read from the
    /// provider on first use.
    ///
    /// The provider is called at `mjd_ut1` itself. Providers resolve these
    /// fields by `floor(mjd_ut1)`, so the value is the same for the whole day,
    /// and an error carries the same message as a direct call.
    ///
    /// # Arguments
    /// - `mjd_ut1`: Modified Julian Date in UT1 [days]
    /// - `slot`: Selects the field of a [`SpaceWeatherDay`] to read and fill
    /// - `fetch`: Provider call returning the field's value at an MJD in UT1
    ///
    /// # Returns
    /// - `Result<T, BraheError>`: The cached value for the UT1 day, or the error `fetch`
    ///   returns on a cache miss
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ap_daily = sw.field(mjd_ut1, |d| &mut d.ap_daily, get_global_ap_daily)?;
    /// ```
    fn field<T: Copy>(
        &self,
        mjd_ut1: f64,
        slot: impl Fn(&mut SpaceWeatherDay) -> &mut Option<T>,
        fetch: impl FnOnce(f64) -> Result<T, BraheError>,
    ) -> Result<T, BraheError> {
        let day = mjd_ut1.floor() as i64;
        let mut cache = self.0.borrow_mut();
        if let Some(value) = cache.days.get_mut(&day).and_then(|d| *slot(d)) {
            return Ok(value);
        }
        let value = fetch(mjd_ut1)?;
        *slot(cache.days.entry(day).or_default()) = Some(value);
        Ok(value)
    }
}

impl SpaceWeatherLookup for CachedSpaceWeather<'_> {
    fn ap_daily(&self, mjd_ut1: f64) -> Result<f64, BraheError> {
        self.field(mjd_ut1, |d| &mut d.ap_daily, get_global_ap_daily)
    }

    fn ap(&self, mjd_ut1: f64) -> Result<f64, BraheError> {
        let ap = self.field(mjd_ut1, |d| &mut d.ap, get_global_ap_all)?;
        Ok(ap[FileSpaceWeatherProvider::get_interval_index(mjd_ut1)])
    }

    fn f107_observed(&self, mjd_ut1: f64) -> Result<f64, BraheError> {
        self.field(mjd_ut1, |d| &mut d.f107_observed, get_global_f107_observed)
    }

    fn f107_obs_avg81(&self, mjd_ut1: f64) -> Result<f64, BraheError> {
        self.field(
            mjd_ut1,
            |d| &mut d.f107_obs_avg81,
            get_global_f107_obs_avg81,
        )
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use crate::earth_models::{
        NRLMSISE00Config, density_nrlmsise00, density_nrlmsise00_geod,
        density_nrlmsise00_geod_with, density_nrlmsise00_with,
    };
    use crate::frames::rotation_eci_to_ecef;
    use crate::orbit_dynamics::tides::SolidTideConfig;
    use crate::propagators::{
        AtmosphericModel, DragConfiguration, EclipseModel, GravityConfiguration, OccultingBody,
        ParameterSource, PrecomputeConfig, SolarRadiationPressureConfiguration,
        ThirdBodyConfiguration, TidesConfiguration, ZonalHarmonicsDegree,
    };
    use crate::space_weather::{
        FileSpaceWeatherProvider, SpaceWeatherExtrapolation, StaticSpaceWeatherProvider,
        get_global_ap, get_global_ap_daily, get_global_f107_obs_avg81, get_global_f107_observed,
        get_global_sw_mjd_last_daily_predicted, get_global_sw_mjd_max, get_global_sw_mjd_min,
        set_global_space_weather_provider,
    };
    use crate::utils::testing::{
        get_test_space_weather_filepath, setup_global_test_eop, setup_global_test_space_weather,
        setup_global_test_spice,
    };
    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    fn third_body_entry(body: ThirdBody, source: EphemerisSource) -> ThirdBodyConfiguration {
        ThirdBodyConfiguration {
            ephemeris_source: source,
            ..body.into()
        }
    }

    #[test]
    #[parallel]
    fn test_ephemeris_channels_default_config() {
        assert_eq!(
            ephemeris_channels(&ForceModelConfig::default()),
            vec![
                (
                    ChannelRole::ThirdBody(0),
                    ChannelQuery::GeocentricThirdBody {
                        body: ThirdBody::Sun,
                        source: EphemerisSource::DE440s,
                    },
                ),
                (
                    ChannelRole::ThirdBody(1),
                    ChannelQuery::GeocentricThirdBody {
                        body: ThirdBody::Moon,
                        source: EphemerisSource::DE440s,
                    },
                ),
            ]
        );
    }

    #[test]
    #[parallel]
    fn test_ephemeris_channels_lowprecision_empty() {
        let cfg = ForceModelConfig {
            third_body: Some(vec![
                third_body_entry(ThirdBody::Sun, EphemerisSource::LowPrecision),
                third_body_entry(ThirdBody::Moon, EphemerisSource::LowPrecision),
            ]),
            tides: Some(TidesConfiguration {
                solid: Some(SolidTideConfig {
                    frequency_dependent: true,
                    pole_tide: false,
                }),
                ..TidesConfiguration::default()
            }),
            ..ForceModelConfig::default()
        };
        assert!(ephemeris_channels(&cfg).is_empty());
    }

    #[test]
    #[parallel]
    fn test_precompute_key_distinguishes_channel_roles() {
        let config = |entries: Vec<ThirdBodyConfiguration>| ForceModelConfig {
            third_body: Some(entries),
            precompute: Some(PrecomputeConfig::default()),
            ..ForceModelConfig::earth_gravity()
        };
        let short = config(vec![
            third_body_entry(ThirdBody::Moon, EphemerisSource::DE440s),
            third_body_entry(ThirdBody::Venus, EphemerisSource::DE440s),
        ]);
        let long = config(vec![
            third_body_entry(ThirdBody::Sun, EphemerisSource::LowPrecision),
            third_body_entry(ThirdBody::Moon, EphemerisSource::DE440s),
            third_body_entry(ThirdBody::Venus, EphemerisSource::DE440s),
        ]);
        let short_key = PrecomputeKey::from_force_config(&short).unwrap();
        let long_key = PrecomputeKey::from_force_config(&long).unwrap();
        assert_ne!(short_key, long_key);
        assert_eq!(
            short_key,
            PrecomputeKey::from_force_config(&short.clone()).unwrap()
        );
    }

    #[test]
    #[parallel]
    fn test_precompute_key_ephemeris_only_for_non_earth_central() {
        let cfg = ForceModelConfig {
            precompute: Some(PrecomputeConfig::default()),
            ..ForceModelConfig::for_body(
                CentralBody::Moon,
                GravityConfiguration::PointMass,
                None,
                None,
                Some(vec![ThirdBody::Earth.into()]),
                false,
                None,
            )
        };
        let key = PrecomputeKey::from_force_config(&cfg).unwrap();
        let inputs = PrecomputedInputs::new(&key);
        assert!(inputs.rotation.is_none());
        assert!(inputs.ephemeris.is_some());

        let no_ephemeris = ForceModelConfig {
            precompute: Some(PrecomputeConfig {
                ephemeris_step: None,
                ..PrecomputeConfig::default()
            }),
            ..cfg
        };
        assert_eq!(PrecomputeKey::from_force_config(&no_ephemeris), None);
    }

    #[test]
    #[parallel]
    fn test_ephemeris_channels_moon_central_srp_occulter() {
        let cfg = ForceModelConfig::for_body(
            CentralBody::Moon,
            GravityConfiguration::PointMass,
            None,
            Some(SolarRadiationPressureConfiguration {
                area: ParameterSource::Value(1.0),
                cr: ParameterSource::Value(1.3),
                eclipse_model: EclipseModel::Conical,
                occulting_bodies: vec![OccultingBody::Earth, OccultingBody::Moon],
            }),
            None,
            false,
            Some(ParameterSource::Value(100.0)),
        );
        assert_eq!(
            ephemeris_channels(&cfg),
            vec![
                (
                    ChannelRole::SrpSun,
                    ChannelQuery::SpkPosition {
                        target: 10,
                        center: 301,
                    },
                ),
                (
                    ChannelRole::Occulter(0),
                    ChannelQuery::SpkPosition {
                        target: 399,
                        center: 301,
                    },
                ),
            ]
        );

        let no_eclipse = ForceModelConfig {
            srp: cfg
                .srp
                .clone()
                .map(|srp| SolarRadiationPressureConfiguration {
                    eclipse_model: EclipseModel::None,
                    ..srp
                }),
            ..cfg
        };
        assert_eq!(
            ephemeris_channels(&no_eclipse),
            vec![(
                ChannelRole::SrpSun,
                ChannelQuery::SpkPosition {
                    target: 10,
                    center: 301,
                },
            )]
        );
    }

    #[test]
    #[parallel]
    fn test_ephemeris_channels_roles_and_queries() {
        let cfg = ForceModelConfig {
            third_body: Some(vec![
                ThirdBodyConfiguration {
                    gravity: GravityConfiguration::EarthZonal {
                        degree: ZonalHarmonicsDegree::J2,
                    },
                    ..ThirdBody::Moon.into()
                },
                ThirdBody::Phobos.into(),
                ThirdBody::Venus.into(),
            ]),
            tides: Some(TidesConfiguration {
                solid: Some(SolidTideConfig {
                    frequency_dependent: true,
                    pole_tide: false,
                }),
                ephemeris_source: EphemerisSource::DE440,
                ..TidesConfiguration::default()
            }),
            ..ForceModelConfig::default()
        };
        assert_eq!(
            ephemeris_channels(&cfg),
            vec![
                (
                    ChannelRole::ThirdBody(0),
                    ChannelQuery::ThirdBodyPosition {
                        central: CentralBody::Earth,
                        body: ThirdBody::Moon,
                        source: EphemerisSource::DE440s,
                    },
                ),
                (
                    ChannelRole::ThirdBody(1),
                    ChannelQuery::ThirdBodyPosition {
                        central: CentralBody::Earth,
                        body: ThirdBody::Phobos,
                        source: EphemerisSource::DE440s,
                    },
                ),
                (
                    ChannelRole::ThirdBody(2),
                    ChannelQuery::GeocentricThirdBody {
                        body: ThirdBody::Venus,
                        source: EphemerisSource::DE440s,
                    },
                ),
                (
                    ChannelRole::TideSunMoon,
                    ChannelQuery::SunMoon {
                        source: EphemerisSource::DE440,
                    },
                ),
            ]
        );

        let drag = ForceModelConfig::for_body(
            CentralBody::EMB,
            GravityConfiguration::PointMass,
            Some(DragConfiguration {
                model: AtmosphericModel::Exponential {
                    scale_height: 60e3,
                    rho0: 1e-12,
                    h0: 400e3,
                },
                area: ParameterSource::Value(1.0),
                cd: ParameterSource::Value(2.2),
                body: Some(CentralBody::Earth),
            }),
            None,
            Some(vec![ThirdBody::Sun.into()]),
            false,
            Some(ParameterSource::Value(100.0)),
        );
        assert_eq!(
            ephemeris_channels(&drag),
            vec![
                (
                    ChannelRole::ThirdBody(0),
                    ChannelQuery::ThirdBodyPosition {
                        central: CentralBody::EMB,
                        body: ThirdBody::Sun,
                        source: EphemerisSource::DE440s,
                    },
                ),
                (
                    ChannelRole::DragBody,
                    ChannelQuery::SpkState {
                        target: 399,
                        center: 3,
                    },
                ),
            ]
        );
    }

    fn all_query_kinds() -> Vec<ChannelQuery> {
        vec![
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Moon,
                source: EphemerisSource::DE440s,
            },
            ChannelQuery::SunMoon {
                source: EphemerisSource::DE440s,
            },
            ChannelQuery::ThirdBodyPosition {
                central: CentralBody::Moon,
                body: ThirdBody::Venus,
                source: EphemerisSource::DE440s,
            },
            ChannelQuery::SpkState {
                target: 301,
                center: 399,
            },
            ChannelQuery::SpkPosition {
                target: 10,
                center: 301,
            },
        ]
    }

    #[test]
    #[serial]
    fn test_EphemerisTable_node_exact() {
        setup_global_test_eop();
        setup_global_test_spice();
        let queries = all_query_kinds();
        let mut table = EphemerisTable::new(60.0, queries.clone());
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 13.0, 0.0, TimeSystem::UTC);
        let need = NodeRange::bracketing(60.0, t0, t0 + 300.0);
        table.ensure(need).unwrap();
        for k in need.k_min..need.k_max {
            let e = need.epoch(k);
            for (i, query) in queries.iter().enumerate() {
                let exact = query.evaluate(e).unwrap();
                assert_eq!(exact.len(), query.dim());
                let value = table.value(i, e).unwrap();
                assert_eq!(&value[..query.dim()], exact.as_slice());
                assert!(value[query.dim()..].iter().all(|v| *v == 0.0));
            }
        }
        assert!(table.value(queries.len(), t0).is_none());
        assert!(table.value(0, t0 - 3600.0).is_none());
    }

    #[test]
    #[serial]
    fn test_EphemerisTable_moon_offgrid_error() {
        setup_global_test_eop();
        setup_global_test_spice();
        let queries = vec![
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Moon,
                source: EphemerisSource::DE440s,
            },
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Sun,
                source: EphemerisSource::DE440s,
            },
        ];
        let mut table = EphemerisTable::new(60.0, queries.clone());
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        table
            .ensure(NodeRange::bracketing(60.0, t0, t0 + 86400.0))
            .unwrap();
        let mut max_err = [0.0_f64; 2];
        let mut t = 0.0;
        while t < 86400.0 {
            let e = t0 + t;
            for (i, query) in queries.iter().enumerate() {
                let exact = query.evaluate(e).unwrap();
                let value = table.value(i, e).unwrap();
                let err = (0..3)
                    .map(|j| (value[j] - exact[j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                max_err[i] = max_err[i].max(err);
            }
            t += 7.3;
        }
        println!(
            "max off-grid error: Moon {:.4} m, Sun {:.4} m",
            max_err[0], max_err[1]
        );
        assert!(max_err[0] < 1.5, "Moon max error {}", max_err[0]);
        assert!(max_err[1] < 3.5, "Sun max error {}", max_err[1]);
    }

    #[test]
    #[parallel]
    fn test_EphemerisTable_ensure_returns_first_query_error() {
        let queries = vec![
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Mercury,
                source: EphemerisSource::LowPrecision,
            },
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Venus,
                source: EphemerisSource::LowPrecision,
            },
        ];
        let mut table = EphemerisTable::new(60.0, queries);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let err = table
            .ensure(NodeRange::bracketing(60.0, t0, t0 + 600.0))
            .unwrap_err();
        assert!(err.to_string().contains("Mercury"), "{err}");
        assert!(table.coverage().is_none());
    }

    #[test]
    #[serial]
    fn test_PrecomputedInputs_ephemeris_by_role() {
        setup_global_test_eop();
        setup_global_test_spice();
        let cfg = ForceModelConfig {
            third_body: Some(vec![
                third_body_entry(ThirdBody::Sun, EphemerisSource::LowPrecision),
                third_body_entry(ThirdBody::Moon, EphemerisSource::DE440s),
            ]),
            precompute: Some(PrecomputeConfig::default()),
            ..ForceModelConfig::earth_gravity()
        };
        let key = PrecomputeKey::from_force_config(&cfg).unwrap();
        let mut inputs = PrecomputedInputs::new(&key);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(!inputs.covers(t0));
        inputs.ensure(t0, t0 + 900.0).unwrap();
        assert!(inputs.covers_span(t0, t0 + 900.0));
        assert!(!inputs.covers(t0 + 86400.0));

        let (k, _) = NodeRange::locate(60.0, t0 + 300.0);
        let node = grid_anchor() + k as f64 * 60.0;
        let moon = ChannelQuery::GeocentricThirdBody {
            body: ThirdBody::Moon,
            source: EphemerisSource::DE440s,
        }
        .evaluate(node)
        .unwrap();
        let value = inputs.ephemeris(&ChannelRole::ThirdBody(1), node).unwrap();
        assert_eq!(&value[..3], moon.as_slice());
        assert!(inputs.ephemeris(&ChannelRole::ThirdBody(0), node).is_none());
        assert!(inputs.ephemeris(&ChannelRole::TideSunMoon, node).is_none());
    }

    #[test]
    #[parallel]
    fn test_NodeRange_locate_negative_index() {
        let (k, s) = NodeRange::locate(60.0, grid_anchor() - 90.0);
        assert_eq!(k, -2);
        assert_abs_diff_eq!(s, 0.5, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_NodeRange_bracketing_reversed_span() {
        let earlier = grid_anchor() + 1000.0 * 60.0 + 10.0;
        let later = earlier + 20.0;
        assert_eq!(
            NodeRange::bracketing(60.0, later, earlier),
            NodeRange::bracketing(60.0, earlier, later)
        );
        let r = NodeRange::bracketing(60.0, earlier, later);
        assert_eq!(r.k_max - r.k_min + 1, 2);
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_node_exact() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let need = NodeRange::bracketing(600.0, t0, t0 + 3000.0);
        table.ensure(need);
        for k in need.k_min..need.k_max {
            let e = need.epoch(k);
            let exact = polar_motion(e) * earth_rotation(e) * bias_precession_nutation(e);
            assert_eq!(table.rotation(e).unwrap(), exact);
        }
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_offgrid_error() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        table.ensure(NodeRange::bracketing(600.0, t0, t0 + 86400.0));
        let mut max_err: f64 = 0.0;
        let mut t = 0.0;
        while t < 86400.0 {
            let e = t0 + t;
            let exact = rotation_eci_to_ecef(e);
            let approx = table.rotation(e).unwrap();
            let err = (exact.transpose() * approx - SMatrix3::identity()).norm() / 2f64.sqrt();
            max_err = max_err.max(err);
            t += 7.3;
        }
        assert!(max_err < 1e-11, "max error {max_err}");
    }

    #[test]
    #[parallel]
    fn test_NodeRange_locate_roundtrip() {
        for step in [30.0, 60.0, 600.0, 3600.0] {
            let range = NodeRange {
                step,
                k_min: 0,
                k_max: 0,
            };
            for k in [-1_000_000, -12345, -1, 0, 1, 777, 12345, 1_000_000] {
                assert_eq!(NodeRange::locate(step, range.epoch(k)), (k, 0.0));
            }
        }
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_ensure_fills_small_gap() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let base = NodeRange {
            step: 600.0,
            k_min: 40000,
            k_max: 40002,
        };
        table.ensure(base);
        assert_eq!(table.nodes_computed, 3);
        let need = NodeRange {
            step: 600.0,
            k_min: base.k_max + 3,
            k_max: base.k_max + 4,
        };
        table.ensure(need);
        assert_eq!(table.nodes_computed, 3 + 2 + 2);
        let cov = table.coverage().unwrap();
        assert_eq!((cov.k_min, cov.k_max), (base.k_min, need.k_max));
        let e = base.epoch(base.k_max) + 700.0;
        assert!(table.rotation(e).is_some());

        let below = NodeRange {
            step: 600.0,
            k_min: base.k_min - 5,
            k_max: base.k_min - 4,
        };
        table.ensure(below);
        let cov = table.coverage().unwrap();
        assert_eq!((cov.k_min, cov.k_max), (below.k_min, need.k_max));
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_outside_range_none() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert!(table.rotation(t0).is_none());
        table.ensure(NodeRange::bracketing(600.0, t0, t0 + 100.0));
        assert!(table.rotation(t0 + 50.0).is_some());
        assert!(table.rotation(t0 - 3600.0).is_none());
        assert!(table.rotation(t0 + 3600.0).is_none());
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_ensure_appends_only_missing() {
        setup_global_test_eop();
        let mut table = FrameRotationTable::new(600.0);
        let t0 = grid_anchor() + 600.0 * 40000.0 + 300.0;
        table.ensure(NodeRange::bracketing(600.0, t0, t0 + 600.0));
        let first = table.coverage().unwrap();
        assert_eq!(first.k_max - first.k_min + 1, 3);
        assert_eq!(table.nodes_computed, 3);

        table.ensure(NodeRange::bracketing(600.0, t0 + 600.0, t0 + 1200.0));
        assert_eq!(table.nodes_computed, 4);
        let cov = table.coverage().unwrap();
        assert_eq!((cov.k_min, cov.k_max), (first.k_min, first.k_max + 1));

        table.ensure(NodeRange::bracketing(600.0, t0 - 600.0, t0));
        assert_eq!(table.nodes_computed, 5);
        let cov = table.coverage().unwrap();
        assert_eq!((cov.k_min, cov.k_max), (first.k_min - 1, first.k_max + 1));
        let e = t0 - 300.0;
        assert_abs_diff_eq!(
            (table.rotation(e).unwrap() - rotation_eci_to_ecef(e)).norm(),
            0.0,
            epsilon = 1e-9
        );

        let far = NodeRange::bracketing(600.0, t0 + 864000.0, t0 + 864100.0);
        table.ensure(far);
        assert_eq!(table.coverage().unwrap(), far);
    }

    #[test]
    #[parallel]
    fn test_PrecomputedInputs_ensure_without_tables() {
        let mut inputs = PrecomputedInputs::new(&PrecomputeKey {
            frame_rotation: None,
            ephemeris: None,
        });
        let t0 = grid_anchor() + 600.0 * 40000.0 + 300.0;
        inputs.ensure(t0, t0 + 3600.0).unwrap();
        assert!(inputs.covers_span(t0, t0 + 3600.0));
        assert!(inputs.rotation(t0).is_none());
        assert_eq!(inputs.rotation_nodes_computed(), 0);
    }

    #[test]
    #[parallel]
    fn test_PrecomputedInputs_ensure_covered_span_computes_nothing() {
        setup_global_test_eop();
        let cfg = ForceModelConfig {
            precompute: Some(PrecomputeConfig {
                frame_rotation_step: Some(600.0),
                ..PrecomputeConfig::default()
            }),
            ..ForceModelConfig::earth_gravity()
        };
        let mut inputs = PrecomputedInputs::new(&PrecomputeKey::from_force_config(&cfg).unwrap());
        let t0 = grid_anchor() + 600.0 * 40000.0 + 300.0;
        inputs.ensure(t0, t0 + 3600.0).unwrap();
        let computed = inputs.rotation_nodes_computed();
        assert!(computed > 0);

        inputs.ensure(t0 + 600.0, t0 + 1200.0).unwrap();
        assert_eq!(inputs.rotation_nodes_computed(), computed);
        assert!(inputs.covers_span(t0, t0 + 3600.0));
    }

    fn table_methods() -> Vec<InterpolationMethod> {
        vec![
            InterpolationMethod::Linear,
            InterpolationMethod::Lagrange { degree: 1 },
            InterpolationMethod::Lagrange { degree: 3 },
            InterpolationMethod::Lagrange { degree: 4 },
            InterpolationMethod::Lagrange { degree: 5 },
        ]
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_node_exact_all_methods() {
        setup_global_test_eop();
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        for method in table_methods() {
            let mut table = FrameRotationTable::new(600.0).with_interpolation(method);
            let need = table.span(t0, t0 + 3000.0);
            table.ensure(need);
            let linear = NodeRange::bracketing(600.0, t0, t0 + 3000.0);
            for k in linear.k_min..linear.k_max {
                let e = linear.epoch(k);
                let exact = polar_motion(e) * earth_rotation(e) * bias_precession_nutation(e);
                assert_eq!(table.rotation(e).unwrap(), exact, "{method:?} node {k}");
            }
        }
    }

    /// Maximum rotation error, in radians, of a frame-rotation table over one
    /// day sampled every 7.3 s.
    fn rotation_offgrid_error(step: f64, method: InterpolationMethod) -> f64 {
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut table = FrameRotationTable::new(step).with_interpolation(method);
        table.ensure(table.span(t0, t0 + 86400.0));
        let mut max_err: f64 = 0.0;
        let mut t = 0.0;
        while t < 86400.0 {
            let e = t0 + t;
            let exact = rotation_eci_to_ecef(e);
            let approx = table.rotation(e).unwrap();
            let err = (exact.transpose() * approx - SMatrix3::identity()).norm() / 2f64.sqrt();
            max_err = max_err.max(err);
            t += 7.3;
        }
        max_err
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_lagrange_offgrid_error() {
        setup_global_test_eop();
        let linear = rotation_offgrid_error(600.0, InterpolationMethod::Linear);
        let lagrange = rotation_offgrid_error(600.0, InterpolationMethod::Lagrange { degree: 3 });
        println!(
            "rotation off-grid error at 600 s: linear {linear:.3e}, Lagrange 3 {lagrange:.3e}"
        );
        assert!(
            lagrange < linear,
            "Lagrange 3 {lagrange} vs linear {linear}"
        );
    }

    #[test]
    #[parallel]
    fn test_FrameRotationTable_lagrange_coverage_padding() {
        setup_global_test_eop();
        let t0 = grid_anchor() + 600.0 * 40000.0 + 100.0;
        let t1 = t0 + 1750.0;
        let linear = NodeRange::bracketing(600.0, t0, t1);
        for (degree, below, above) in [(5, 2, 2), (4, 2, 2)] {
            let method = InterpolationMethod::Lagrange { degree };
            let mut table = FrameRotationTable::new(600.0).with_interpolation(method);
            let need = table.span(t0, t1);
            assert_eq!(
                (need.k_min, need.k_max),
                (linear.k_min - below, linear.k_max + above),
                "{method:?}"
            );
            table.ensure(need);
            assert!(table.store.holds(need));
            let mut t = 0.0;
            while t <= t1 - t0 {
                assert!(table.rotation(t0 + t).is_some(), "{method:?} at {t}");
                t += 10.0;
            }
            assert!(table.rotation(t1).is_some());
            let last = linear.epoch(linear.k_max - 1) + 599.0;
            assert!(table.rotation(last).is_some(), "{method:?}");

            let mut short = FrameRotationTable::new(600.0).with_interpolation(method);
            short.ensure(NodeRange {
                step: 600.0,
                k_min: need.k_min + 1,
                k_max: need.k_max,
            });
            assert!(short.rotation(t0).is_none(), "{method:?}");
            let mut short = FrameRotationTable::new(600.0).with_interpolation(method);
            short.ensure(NodeRange {
                step: 600.0,
                k_min: need.k_min,
                k_max: need.k_max - 1,
            });
            assert!(short.rotation(last).is_none(), "{method:?}");
        }
    }

    #[test]
    #[parallel]
    fn test_PrecomputedInputs_lagrange_covers_matches_lookup() {
        setup_global_test_eop();
        let cfg = ForceModelConfig {
            precompute: Some(PrecomputeConfig {
                ephemeris_step: None,
                frame_rotation_interpolation: InterpolationMethod::Lagrange { degree: 5 },
                ..PrecomputeConfig::default()
            }),
            ..ForceModelConfig::earth_gravity()
        };
        let key = PrecomputeKey::from_force_config(&cfg).unwrap();
        let mut inputs = PrecomputedInputs::new(&key);
        let t0 = grid_anchor() + 600.0 * 40000.0 + 100.0;
        inputs.ensure(t0, t0 + 1700.0).unwrap();
        assert!(inputs.covers_span(t0, t0 + 1700.0));
        let mut t = -1200.0;
        while t <= 3000.0 {
            let e = t0 + t;
            assert_eq!(inputs.covers(e), inputs.rotation(e).is_some(), "offset {t}");
            t += 50.0;
        }
        assert!(!inputs.covers(t0 + 3000.0));
        inputs.ensure(t0 + 3000.0, t0 + 3000.0).unwrap();
        assert!(inputs.covers(t0 + 3000.0));
        assert!(inputs.rotation(t0 + 3000.0).is_some());
    }

    #[test]
    #[parallel]
    fn test_precompute_key_distinguishes_interpolation() {
        let config = |frame_rotation_interpolation, ephemeris_interpolation| ForceModelConfig {
            precompute: Some(PrecomputeConfig {
                frame_rotation_interpolation,
                ephemeris_interpolation,
                ..PrecomputeConfig::default()
            }),
            ..ForceModelConfig::default()
        };
        let key = |a, b| PrecomputeKey::from_force_config(&config(a, b)).unwrap();
        let linear = key(InterpolationMethod::Linear, InterpolationMethod::Linear);
        assert_eq!(
            linear,
            key(InterpolationMethod::Linear, InterpolationMethod::Linear)
        );
        let rotation_lagrange = key(
            InterpolationMethod::Lagrange { degree: 3 },
            InterpolationMethod::Linear,
        );
        let ephemeris_hermite = key(
            InterpolationMethod::Linear,
            InterpolationMethod::HermiteCubic,
        );
        let ephemeris_lagrange = key(
            InterpolationMethod::Linear,
            InterpolationMethod::Lagrange { degree: 3 },
        );
        assert_ne!(linear, rotation_lagrange);
        assert_ne!(linear, ephemeris_hermite);
        assert_ne!(linear, ephemeris_lagrange);
        assert_ne!(ephemeris_hermite, ephemeris_lagrange);
        assert_ne!(
            rotation_lagrange,
            key(
                InterpolationMethod::Lagrange { degree: 5 },
                InterpolationMethod::Linear,
            )
        );
    }

    #[test]
    #[serial]
    fn test_EphemerisTable_node_exact_all_methods() {
        setup_global_test_eop();
        setup_global_test_spice();
        let queries = all_query_kinds();
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 13.0, 0.0, TimeSystem::UTC);
        let mut methods = table_methods();
        methods.push(InterpolationMethod::HermiteCubic);
        for method in methods {
            let mut table = EphemerisTable::new(60.0, queries.clone()).with_interpolation(method);
            table.ensure(table.span(t0, t0 + 300.0)).unwrap();
            let linear = NodeRange::bracketing(60.0, t0, t0 + 300.0);
            for k in linear.k_min..linear.k_max {
                let e = linear.epoch(k);
                for (i, query) in queries.iter().enumerate() {
                    let exact = query.evaluate(e).unwrap();
                    let value = table.value(i, e).unwrap();
                    assert_eq!(
                        &value[..query.dim()],
                        exact.as_slice(),
                        "{method:?} {query:?}"
                    );
                    assert!(value[query.dim()..].iter().all(|v| *v == 0.0));
                }
            }
        }
    }

    #[test]
    #[serial]
    fn test_ChannelQuery_derivative_matches_finite_difference() {
        setup_global_test_eop();
        setup_global_test_spice();
        let e = Epoch::from_datetime(2024, 3, 1, 0, 0, 13.0, 0.0, TimeSystem::UTC);
        let mut queries = all_query_kinds();
        queries.push(ChannelQuery::ThirdBodyPosition {
            central: CentralBody::EMB,
            body: ThirdBody::Earth,
            source: EphemerisSource::DE440s,
        });
        queries.push(ChannelQuery::GeocentricThirdBody {
            body: ThirdBody::JupiterBarycenter,
            source: EphemerisSource::DE440s,
        });
        // Planet centres resolve through a satellite-system kernel (mar099s)
        // leg: Mars as target and as centre on the kernel-scoped path, and
        // the geocentric planet-centre ephemeris.
        queries.push(ChannelQuery::ThirdBodyPosition {
            central: CentralBody::Earth,
            body: ThirdBody::Mars,
            source: EphemerisSource::DE440s,
        });
        queries.push(ChannelQuery::ThirdBodyPosition {
            central: CentralBody::Mars,
            body: ThirdBody::Sun,
            source: EphemerisSource::DE440s,
        });
        queries.push(ChannelQuery::GeocentricThirdBody {
            body: ThirdBody::Mars,
            source: EphemerisSource::DE440s,
        });
        for query in queries {
            query.load_kernels().unwrap();
            let dim = query.dim();
            let node = query.evaluate_with_derivative(e).unwrap();
            assert_eq!(node.len(), 2 * dim);
            assert_eq!(
                &node[..dim],
                query.evaluate(e).unwrap().as_slice(),
                "{query:?}"
            );
            let h = 1.0;
            let plus = query.evaluate(e + h).unwrap();
            let minus = query.evaluate(e - h).unwrap();
            for j in 0..dim {
                let fd = (plus[j] - minus[j]) / (2.0 * h);
                let derivative = node[dim + j];
                assert!(
                    (derivative - fd).abs() <= 1e-6 * fd.abs().max(1e-3),
                    "{query:?} component {j}: {derivative} vs {fd}"
                );
            }
        }
    }

    /// Maximum Moon and Sun position errors, in meters, of an ephemeris table
    /// over one day sampled every 7.3 s.
    fn sun_moon_offgrid_error(step: f64, method: InterpolationMethod) -> [f64; 2] {
        let queries = vec![
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Moon,
                source: EphemerisSource::DE440s,
            },
            ChannelQuery::GeocentricThirdBody {
                body: ThirdBody::Sun,
                source: EphemerisSource::DE440s,
            },
        ];
        let t0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut table = EphemerisTable::new(step, queries.clone()).with_interpolation(method);
        table.ensure(table.span(t0, t0 + 86400.0)).unwrap();
        let mut max_err = [0.0_f64; 2];
        let mut t = 0.0;
        while t < 86400.0 {
            let e = t0 + t;
            for (i, query) in queries.iter().enumerate() {
                let exact = query.evaluate(e).unwrap();
                let value = table.value(i, e).unwrap();
                let err = (0..3)
                    .map(|j| (value[j] - exact[j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                max_err[i] = max_err[i].max(err);
            }
            t += 7.3;
        }
        max_err
    }

    #[test]
    #[serial]
    fn test_EphemerisTable_higher_order_moon_offgrid_error() {
        setup_global_test_eop();
        setup_global_test_spice();
        for method in [
            InterpolationMethod::Lagrange { degree: 3 },
            InterpolationMethod::HermiteCubic,
        ] {
            let [moon, sun] = sun_moon_offgrid_error(60.0, method);
            println!("{method:?} off-grid error at 60 s: Moon {moon:.3e} m, Sun {sun:.3e} m");
            assert!(moon < 0.01, "{method:?} Moon max error {moon}");
        }
    }

    /// Reinstalls the standard test space-weather provider when dropped, so a
    /// test that installs another provider restores it even when an assertion
    /// fails.
    struct RestoreTestSpaceWeather;

    impl Drop for RestoreTestSpaceWeather {
        fn drop(&mut self) {
            setup_global_test_space_weather();
        }
    }

    /// Epochs at each 3-hour bin boundary of a UT1 day, offset by -1 ms, 0
    /// and +1 ms.
    fn space_weather_bin_epochs(day_mjd: f64) -> Vec<Epoch> {
        let midnight = Epoch::from_mjd(day_mjd, TimeSystem::UT1);
        (0..8)
            .flat_map(|bin| {
                let boundary = midnight + bin as f64 * 10800.0;
                [boundary - 1e-3, boundary, boundary + 1e-3]
            })
            .collect()
    }

    /// Asserts that every lookup and the NRLMSISE-00 density read through one
    /// shared cache equal the global-provider path bitwise at each epoch.
    fn assert_space_weather_cache_matches_global(epochs: &[Epoch]) {
        let config = NRLMSISE00Config::default();
        let geod = [-74.0, 40.7, 400e3];
        let x_ecef = Vector3::new(R_EARTH + 400e3, 0.0, 0.0);
        let mut cache = SpaceWeatherCache::new();
        for epoch in epochs {
            let sw = CachedSpaceWeather::new(&mut cache);
            let mjd = epoch.mjd_as_time_system(TimeSystem::UT1);
            for offset in [0.0, -0.125, -1.0, -57.0 / 24.0] {
                let m = mjd + offset;
                assert_eq!(
                    sw.ap_daily(m).unwrap().to_bits(),
                    get_global_ap_daily(m).unwrap().to_bits()
                );
                assert_eq!(
                    sw.ap(m).unwrap().to_bits(),
                    get_global_ap(m).unwrap().to_bits()
                );
                assert_eq!(
                    sw.f107_observed(m).unwrap().to_bits(),
                    get_global_f107_observed(m).unwrap().to_bits()
                );
                assert_eq!(
                    sw.f107_obs_avg81(m).unwrap().to_bits(),
                    get_global_f107_obs_avg81(m).unwrap().to_bits()
                );
            }
            let cached = density_nrlmsise00_geod_with(epoch, &geod, &config, &sw).unwrap();
            let global = density_nrlmsise00_geod(epoch, &geod, &config).unwrap();
            assert_eq!(cached.to_bits(), global.to_bits(), "epoch {epoch}");
            let cached = density_nrlmsise00_with(epoch, x_ecef, &config, &sw).unwrap();
            let global = density_nrlmsise00(epoch, x_ecef, &config).unwrap();
            assert_eq!(cached.to_bits(), global.to_bits(), "epoch {epoch}");
        }
    }

    /// Epochs on the first day after the daily-predicted section and five
    /// days into the monthly-predicted section.
    fn monthly_predicted_epochs() -> Vec<Epoch> {
        let last_daily = get_global_sw_mjd_last_daily_predicted().floor();
        let mut epochs = space_weather_bin_epochs(last_daily + 1.0);
        epochs.extend(space_weather_bin_epochs(last_daily + 5.0));
        epochs
    }

    #[test]
    #[parallel]
    fn test_SpaceWeatherCache_matches_provider_file() {
        setup_global_test_eop();
        setup_global_test_space_weather();
        let mut epochs = space_weather_bin_epochs(60384.0);
        epochs.extend(space_weather_bin_epochs(60385.0));
        epochs.extend(monthly_predicted_epochs());
        epochs.extend(space_weather_bin_epochs(
            get_global_sw_mjd_max().floor() + 30.0,
        ));
        assert_space_weather_cache_matches_global(&epochs);
    }

    #[test]
    #[serial]
    fn test_SpaceWeatherCache_matches_provider_zero_extrapolation() {
        setup_global_test_eop();
        let _restore = RestoreTestSpaceWeather;
        set_global_space_weather_provider(
            FileSpaceWeatherProvider::from_file(
                &get_test_space_weather_filepath(),
                SpaceWeatherExtrapolation::Zero,
            )
            .unwrap(),
        );
        let epochs = monthly_predicted_epochs();
        let last_daily = get_global_sw_mjd_last_daily_predicted();
        assert_eq!(get_global_ap(last_daily + 5.0).unwrap(), 0.0);
        assert_space_weather_cache_matches_global(&epochs);
    }

    #[test]
    #[serial]
    fn test_SpaceWeatherCache_matches_static_provider() {
        setup_global_test_eop();
        let _restore = RestoreTestSpaceWeather;
        set_global_space_weather_provider(StaticSpaceWeatherProvider::from_values(
            3.0, 15.0, 150.0, 145.0, 100,
        ));
        let mut epochs = space_weather_bin_epochs(60384.0);
        epochs.extend(space_weather_bin_epochs(70000.0));
        assert_space_weather_cache_matches_global(&epochs);
    }

    #[test]
    #[serial]
    fn test_SpaceWeatherCache_error_not_cached() {
        let _restore = RestoreTestSpaceWeather;
        set_global_space_weather_provider(
            FileSpaceWeatherProvider::from_file(
                &get_test_space_weather_filepath(),
                SpaceWeatherExtrapolation::Error,
            )
            .unwrap(),
        );
        let mjd = get_global_sw_mjd_min() - 10.25;
        let day = mjd.floor() as i64;
        let mut cache = SpaceWeatherCache::new();
        let sw = CachedSpaceWeather::new(&mut cache);
        let pairs = [
            (sw.ap_daily(mjd), get_global_ap_daily(mjd)),
            (sw.ap(mjd), get_global_ap(mjd)),
            (sw.f107_observed(mjd), get_global_f107_observed(mjd)),
            (sw.f107_obs_avg81(mjd), get_global_f107_obs_avg81(mjd)),
        ];
        for (cached, global) in pairs {
            let (cached, global) = (cached.unwrap_err(), global.unwrap_err());
            assert!(matches!(cached, BraheError::SpaceWeatherError(_)));
            assert_eq!(cached.to_string(), global.to_string());
        }
        assert!(cache.day(day).is_none());
    }
}
