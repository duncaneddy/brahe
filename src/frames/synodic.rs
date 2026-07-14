/*!
 * Synodic (two-body rotating) reference frames: the Earth-Moon Rotating
 * (EMR), Sun-Earth Rotating (SER), and Geocentric Solar Ecliptic (GSE)
 * frames defined in NASA TP-20220014814 §2.5, with the exact
 * (GTDS/STK-convention) rotation-matrix time derivative of §4.6.1 —
 * including the dẑ/dt term, evaluated from native SPK acceleration —
 * rather than the GMAT dẑ/dt ≈ 0 approximation.
 */

use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};
use crate::spice::{NAIFId, spk_acceleration, spk_position, spk_state};
use crate::time::Epoch;
use crate::utils::BraheError;

/// Computes the inertial→synodic rotation matrix `S` and its exact time
/// derivative `Ṡ` from the relative state of the two primaries
/// (NASA TP-20220014814 Eq. 66/69).
///
/// Axes: x̂ = r₁₂/‖r₁₂‖, ẑ = (r₁₂×v₁₂)/‖r₁₂×v₁₂‖, ŷ = ẑ×x̂; `S` has rows
/// x̂ᵀ, ŷᵀ, ẑᵀ. The derivative rows use dẑ/dt evaluated from `a12` (exact
/// GTDS/STK convention).
///
/// # Arguments
/// - `r12`: Position of the secondary relative to the primary, inertial axes. Units: [m]
/// - `v12`: Velocity of the secondary relative to the primary. Units: [m/s]
/// - `a12`: Acceleration of the secondary relative to the primary. Units: [m/s²]
///
/// # Returns
/// - `(S, Ṡ)`: Rotation matrix from inertial axes to synodic axes, and its
///   time derivative. Units: [-], [1/s]
///
/// # Examples
/// ```ignore
/// // Crate-internal: circular orbit in the xy-plane => S = I and Ṡ is the
/// // instantaneous rotation rate about ẑ.
/// let (s, s_dot) = synodic_axes(r12, v12, a12);
/// ```
pub(crate) fn synodic_axes(
    r12: Vector3<f64>,
    v12: Vector3<f64>,
    a12: Vector3<f64>,
) -> (SMatrix3, SMatrix3) {
    let r_norm = r12.norm();
    let x_hat = r12 / r_norm;

    let h = r12.cross(&v12);
    let h_norm = h.norm();
    let z_hat = h / h_norm;

    let y_hat = z_hat.cross(&x_hat);

    // TP Eq. 69: dx̂/dt = (v₁₂ - x̂(x̂·v₁₂))/‖r₁₂‖ — the component of the
    // relative velocity perpendicular to x̂, divided by the separation.
    let x_hat_dot = (v12 - x_hat * x_hat.dot(&v12)) / r_norm;

    // dh/dt = d/dt(r₁₂×v₁₂) = v₁₂×v₁₂ + r₁₂×a₁₂ = r₁₂×a₁₂; dẑ/dt is its
    // component perpendicular to ẑ, divided by ‖h‖ (exact GTDS/STK form).
    let h_dot = r12.cross(&a12);
    let z_hat_dot = (h_dot - z_hat * z_hat.dot(&h_dot)) / h_norm;

    let y_hat_dot = z_hat_dot.cross(&x_hat) + z_hat.cross(&x_hat_dot);

    let s = SMatrix3::from_rows(&[x_hat.transpose(), y_hat.transpose(), z_hat.transpose()]);
    let s_dot = SMatrix3::from_rows(&[
        x_hat_dot.transpose(),
        y_hat_dot.transpose(),
        z_hat_dot.transpose(),
    ]);
    (s, s_dot)
}

/// Transforms an inertial-axis state (already translated to the synodic
/// frame's origin) into synodic axes: `r_s = S r`, `v_s = S v + Ṡ r`
/// (NASA TP-20220014814 Eq. 67/70, translation handled by the caller).
///
/// # Arguments
/// - `s`: Inertial→synodic rotation matrix from [`synodic_axes`]
/// - `s_dot`: Its time derivative. Units: [1/s]
/// - `x`: Cartesian state (position, velocity), inertial axes. Units: [m; m/s]
///
/// # Returns
/// - Cartesian state in synodic axes. Units: [m; m/s]
pub(crate) fn state_inertial_to_synodic(s: &SMatrix3, s_dot: &SMatrix3, x: SVector6) -> SVector6 {
    let r = x.fixed_rows::<3>(0).into_owned();
    let v = x.fixed_rows::<3>(3).into_owned();
    let r_s: Vector3<f64> = s * r;
    let v_s: Vector3<f64> = s * v + s_dot * r;
    SVector6::new(r_s[0], r_s[1], r_s[2], v_s[0], v_s[1], v_s[2])
}

/// Inverse of [`state_inertial_to_synodic`]: `r = Sᵀ r_s`,
/// `v = Sᵀ v_s + Ṡᵀ r_s` (using Ṡᵀ = −SᵀṠSᵀ from d/dt(SᵀS) = 0, which
/// reduces TP Eq. 68/71 to this form).
///
/// # Arguments
/// - `s`: Inertial→synodic rotation matrix from [`synodic_axes`]
/// - `s_dot`: Its time derivative. Units: [1/s]
/// - `x`: Cartesian state (position, velocity), synodic axes. Units: [m; m/s]
///
/// # Returns
/// - Cartesian state in inertial axes. Units: [m; m/s]
pub(crate) fn state_synodic_to_inertial(s: &SMatrix3, s_dot: &SMatrix3, x: SVector6) -> SVector6 {
    let r_s = x.fixed_rows::<3>(0).into_owned();
    let v_s = x.fixed_rows::<3>(3).into_owned();
    let r: Vector3<f64> = s.transpose() * r_s;
    let v: Vector3<f64> = s.transpose() * v_s + s_dot.transpose() * r_s;
    SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
}

/// EMR (Earth-Moon Rotating) frame axes at `epc`: the inertial→EMR
/// rotation matrix and its exact time derivative, built from the Moon's
/// SPK state and acceleration relative to Earth (NASA TP-20220014814
/// §2.5.1/§4.6.2).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `(S, Ṡ)`: Rotation matrix from GCRF to EMR axes, and its time
///   derivative. Units: [-], [1/s]
pub(crate) fn emr_axes(epc: Epoch) -> Result<(SMatrix3, SMatrix3), BraheError> {
    let x12 = spk_state(NAIFId::Moon, NAIFId::Earth, epc)?;
    let a12 = spk_acceleration(NAIFId::Moon, NAIFId::Earth, epc)?;
    Ok(synodic_axes(
        x12.fixed_rows::<3>(0).into_owned(),
        x12.fixed_rows::<3>(3).into_owned(),
        a12,
    ))
}

/// Computes the rotation matrix from Geocentric Celestial Reference Frame
/// (GCRF) to Earth-Moon Rotating (EMR) axes.
///
/// EMR is the two-body rotating frame defined by the instantaneous
/// Earth-Moon geometry (NASA TP-20220014814 §2.5.1): x̂ points from Earth
/// to the Moon, ẑ is along the instantaneous orbit normal, and ŷ completes
/// the right-handed triad. The rotation matrix is built from the Moon's
/// SPK state and acceleration relative to Earth, using the exact
/// (GTDS/STK-convention) time derivative of §4.6.2.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming GCRF -> EMR
///
/// # Examples
/// ```
/// use brahe::frames::rotation_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_gcrf_to_emr(epc).unwrap();
/// ```
pub fn rotation_gcrf_to_emr(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(emr_axes(epc)?.0)
}

/// Computes the rotation matrix from Earth-Moon Rotating (EMR) axes to
/// Geocentric Celestial Reference Frame (GCRF). Inverse of
/// [`rotation_gcrf_to_emr`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming EMR -> GCRF
///
/// # Examples
/// ```
/// use brahe::frames::rotation_emr_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_emr_to_gcrf(epc).unwrap();
/// ```
pub fn rotation_emr_to_gcrf(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(emr_axes(epc)?.0.transpose())
}

/// Transforms a Cartesian GCRF position into the equivalent Cartesian
/// Earth-Moon Rotating (EMR) position.
///
/// The EMR origin is the Earth-Moon Barycenter (NASA TP-20220014814
/// §2.5.1); the input is re-centered from Earth to the barycenter before
/// rotating into EMR axes.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - `x_emr`: Cartesian EMR position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::position_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_emr = position_gcrf_to_emr(epc, x_gcrf).unwrap();
/// ```
pub fn position_gcrf_to_emr(epc: Epoch, x_gcrf: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    let (s, _) = emr_axes(epc)?;
    let offset = spk_position(NAIFId::EarthMoonBarycenter, NAIFId::Earth, epc)?;
    Ok(s * (x_gcrf - offset))
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) position into the
/// equivalent Cartesian GCRF position. Inverse of
/// [`position_gcrf_to_emr`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_emr`: Cartesian EMR position. Units: (*m*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::{position_emr_to_gcrf, position_gcrf_to_emr};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_emr = position_gcrf_to_emr(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = position_emr_to_gcrf(epc, x_emr).unwrap();
/// ```
pub fn position_emr_to_gcrf(epc: Epoch, x_emr: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    let (s, _) = emr_axes(epc)?;
    let offset = spk_position(NAIFId::EarthMoonBarycenter, NAIFId::Earth, epc)?;
    Ok(s.transpose() * x_emr + offset)
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Cartesian Earth-Moon Rotating (EMR) state.
///
/// The EMR origin is the Earth-Moon Barycenter (NASA TP-20220014814
/// §2.5.1); the input is re-centered from Earth to the barycenter, then
/// rotated into EMR axes using the exact (GTDS/STK-convention) rotation
/// rate of §4.6.2 (including the transport term from `Ṡ`).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_emr`: Cartesian EMR state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::state_gcrf_to_emr;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_emr = state_gcrf_to_emr(epc, x_gcrf).unwrap();
/// ```
pub fn state_gcrf_to_emr(epc: Epoch, x_gcrf: SVector6) -> Result<SVector6, BraheError> {
    let (s, s_dot) = emr_axes(epc)?;
    // EMB relative to Earth in ICRF axes: re-center Earth → EMB, then rotate.
    let offset = spk_state(NAIFId::EarthMoonBarycenter, NAIFId::Earth, epc)?;
    Ok(state_inertial_to_synodic(&s, &s_dot, x_gcrf - offset))
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) state (position and
/// velocity) into the equivalent Cartesian GCRF state. Inverse of
/// [`state_gcrf_to_emr`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_emr`: Cartesian EMR state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::{state_emr_to_gcrf, state_gcrf_to_emr};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_emr = state_gcrf_to_emr(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = state_emr_to_gcrf(epc, x_emr).unwrap();
/// ```
pub fn state_emr_to_gcrf(epc: Epoch, x_emr: SVector6) -> Result<SVector6, BraheError> {
    let (s, s_dot) = emr_axes(epc)?;
    let offset = spk_state(NAIFId::EarthMoonBarycenter, NAIFId::Earth, epc)?;
    Ok(state_synodic_to_inertial(&s, &s_dot, x_emr) + offset)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    use super::*;
    use crate::spice::{NAIFId, spk_state};
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_spice;

    /// Analytic circular-orbit relative state in the xy-plane at phase
    /// `theta`: r = R(cosθ, sinθ, 0), v = RΩ(−sinθ, cosθ, 0), a = −Ω²r.
    fn circular_rel_state(
        radius: f64,
        omega: f64,
        theta: f64,
    ) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let r = Vector3::new(radius * theta.cos(), radius * theta.sin(), 0.0);
        let v = Vector3::new(
            -radius * omega * theta.sin(),
            radius * omega * theta.cos(),
            0.0,
        );
        let a = -r * omega * omega;
        (r, v, a)
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_circular_orbit() {
        let radius = 3.844e8;
        let omega = 2.66e-6;
        let (r, v, a) = circular_rel_state(radius, omega, 0.0);
        let (s, s_dot) = synodic_axes(r, v, a);

        // At theta = 0 the synodic axes coincide with the inertial axes.
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(s[(i, j)], if i == j { 1.0 } else { 0.0 }, epsilon = 1e-14);
            }
        }
        // Ṡ is the instantaneous rotation about ẑ at rate omega.
        assert_abs_diff_eq!(s_dot[(0, 1)], omega, epsilon = 1e-18);
        assert_abs_diff_eq!(s_dot[(1, 0)], -omega, epsilon = 1e-18);
        assert_abs_diff_eq!(s_dot[(2, 0)], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(s_dot[(2, 1)], 0.0, epsilon = 1e-18);
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_orthonormal_and_skew() {
        // Generic inclined, non-circular input: orthonormality and the
        // rigid-rotation identity ṠSᵀ + SṠᵀ = 0 must hold regardless.
        let r = Vector3::new(2.5e8, -1.2e8, 0.9e8);
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let (s, s_dot) = synodic_axes(r, v, a);

        let identity = s * s.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    identity[(i, j)],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = 1e-14
                );
            }
        }
        assert_abs_diff_eq!(s.determinant(), 1.0, epsilon = 1e-14);

        let skew = s_dot * s.transpose() + s * s_dot.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(skew[(i, j)], 0.0, epsilon = 1e-18);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_derivative_matches_finite_difference() {
        let radius = 3.844e8;
        let omega = 2.66e-6;
        let theta = 0.7;
        let dt = 1.0;

        let (r, v, a) = circular_rel_state(radius, omega, theta);
        let (_, s_dot) = synodic_axes(r, v, a);

        let (rp, vp, ap) = circular_rel_state(radius, omega, theta + omega * dt);
        let (sp, _) = synodic_axes(rp, vp, ap);
        let (rm, vm, am) = circular_rel_state(radius, omega, theta - omega * dt);
        let (sm, _) = synodic_axes(rm, vm, am);

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    s_dot[(i, j)],
                    (sp[(i, j)] - sm[(i, j)]) / (2.0 * dt),
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    #[parallel]
    fn test_state_transform_roundtrip() {
        let r = Vector3::new(2.5e8, -1.2e8, 0.9e8);
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let (s, s_dot) = synodic_axes(r, v, a);

        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_syn = state_inertial_to_synodic(&s, &s_dot, x);
        let x_back = state_synodic_to_inertial(&s, &s_dot, x_syn);
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-6);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-10);
        }
    }

    #[test]
    #[serial] // global SPICE registry
    fn test_emr_moon_on_x_axis() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // The Moon's own state, expressed in EMR, must lie on +x̂ with zero
        // y/z position and zero y/z velocity (it defines the frame).
        let x_moon_gcrf = spk_state(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        let x_moon_emr = state_gcrf_to_emr(epc, x_moon_gcrf).unwrap();

        // Moon distance from EMB = d * GM_EARTH/(GM_EARTH+GM_MOON) ~ 0.9879 d.
        assert!(x_moon_emr[0] > 3.4e8 && x_moon_emr[0] < 4.1e8);
        assert_abs_diff_eq!(x_moon_emr[1], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(x_moon_emr[2], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(x_moon_emr[4], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x_moon_emr[5], 0.0, epsilon = 1e-6);

        // Earth sits on −x̂ at the EMB offset (~4.7e6 m).
        let x_earth_emr = state_gcrf_to_emr(epc, SVector6::zeros()).unwrap();
        assert!(x_earth_emr[0] < -4.0e6 && x_earth_emr[0] > -5.5e6);
        assert_abs_diff_eq!(x_earth_emr[1], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(x_earth_emr[2], 0.0, epsilon = 1e-3);
    }

    #[test]
    #[serial]
    fn test_emr_roundtrip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_back = state_emr_to_gcrf(epc, state_gcrf_to_emr(epc, x).unwrap()).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-4);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-9);
        }

        let x3 = Vector3::new(1.0e8, -2.0e8, 5.0e7);
        let x3_back = position_emr_to_gcrf(epc, position_gcrf_to_emr(epc, x3).unwrap()).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x3_back[i], x3[i], epsilon = 1e-4);
        }
    }

    #[test]
    #[serial]
    fn test_rotation_gcrf_to_emr_derivative_consistency() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let dt = 10.0;

        let (_, s_dot) = emr_axes(epc).unwrap();
        let s_p = rotation_gcrf_to_emr(epc + dt).unwrap();
        let s_m = rotation_gcrf_to_emr(epc + (-dt)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    s_dot[(i, j)],
                    (s_p[(i, j)] - s_m[(i, j)]) / (2.0 * dt),
                    epsilon = 1e-11
                );
            }
        }
        // Proper rotation
        let s = rotation_gcrf_to_emr(epc).unwrap();
        assert_abs_diff_eq!(s.determinant(), 1.0, epsilon = 1e-12);
    }
}
