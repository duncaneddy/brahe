/*!
 * Earth-Centered Inertial (ECI) to Nadir, Sun, Normal (NSW) Frame Transformations
 */

use nalgebra::Vector3;

use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame,
};
use crate::utils::BraheError;
use crate::utils::batch::{batch_zip, batch_zip3};

/// Below this norm of the Sun direction projected normal to nadir, the Sun
/// is along the nadir line and the Y axis falls back to the along-track
/// direction.
const DEGENERATE_TOLERANCE: f64 = 1e-9;

/// NSW axes and their angular velocity from the spacecraft and Sun states.
///
/// Returns the NSW-to-ECI rotation (columns X, Y, Z) and the angular velocity
/// of the NSW axes relative to ECI, expressed in NSW axes, obtained from the
/// time derivatives of the basis vectors through `ω = [ẏ·ẑ, ż·x̂, ẋ·ŷ]`.
/// The nadir derivative is `ẋ = −(v − (v·r̂) r̂)/r`; the Sun direction and
/// its derivative are taken from the spacecraft, `Δr = r_sun − r`,
/// `Δv = v_sun − v`. When the Sun lies along the nadir line the Y axis and
/// its derivative are those of the along-track direction `ĥ × r̂`.
fn nsw_axes(x_eci: SVector6, x_sun: SVector6) -> (SMatrix3, Vector3<f64>) {
    let r = x_eci.fixed_rows::<3>(0).into_owned();
    let v = x_eci.fixed_rows::<3>(3).into_owned();
    let r_norm = r.norm();
    let r_hat = r / r_norm;

    let x_hat = -r_hat;
    let x_dot = -(v - v.dot(&r_hat) * r_hat) / r_norm;

    let d_r = x_sun.fixed_rows::<3>(0).into_owned() - r;
    let d_v = x_sun.fixed_rows::<3>(3).into_owned() - v;
    let d_norm = d_r.norm();
    let s_hat = d_r / d_norm;
    let s_dot = (d_v - d_v.dot(&s_hat) * s_hat) / d_norm;

    let p = s_hat - s_hat.dot(&x_hat) * x_hat;
    let (y_hat, y_dot) = if p.norm() > DEGENERATE_TOLERANCE {
        let p_dot =
            s_dot - (s_dot.dot(&x_hat) + s_hat.dot(&x_dot)) * x_hat - s_hat.dot(&x_hat) * x_dot;
        let p_norm = p.norm();
        let y_hat = p / p_norm;
        let y_dot = (p_dot - p_dot.dot(&y_hat) * y_hat) / p_norm;
        (y_hat, y_dot)
    } else {
        let h = r.cross(&v);
        let h_hat = h / h.norm();
        (h_hat.cross(&r_hat), h_hat.cross(&(-x_dot)))
    };

    let z_hat = x_hat.cross(&y_hat);
    let z_dot = x_dot.cross(&y_hat) + x_hat.cross(&y_dot);

    let omega = Vector3::new(y_dot.dot(&z_hat), z_dot.dot(&x_hat), x_dot.dot(&y_hat));
    (SMatrix3::from_columns(&[x_hat, y_hat, z_hat]), omega)
}

/// Computes the rotation matrix transforming a vector in the Nadir, Sun, Normal (NSW) frame
/// to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The NSW frame follows the SANA definition:
/// - X: Unit vector toward nadir, opposite the position vector.
/// - Y: As close to the direction of the Sun as possible while normal to X: the unit vector
///   from the spacecraft to the Sun with its X component removed.
/// - Z: `X × Y`, completing the right-handed set.
///
/// `x_sun` is the Sun's state relative to the same center as `x_eci`. Because X lies along the
/// position vector, the frame is identical whether the Sun direction is taken from the center or
/// from the spacecraft. When the Sun lies along the nadir line (projected norm below 1e-9) the Y
/// axis is taken along the along-track direction `ĥ × r̂`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s); only the position is used here
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from NSW to ECI frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
///
/// let rotation_matrix = rotation_nsw_to_eci(x_eci, x_sun);
/// ```
pub fn rotation_nsw_to_eci(x_eci: SVector6, x_sun: SVector6) -> SMatrix3 {
    nsw_axes(x_eci, x_sun).0
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Nadir, Sun, Normal (NSW) frame.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s); only the position is used here
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECI to NSW frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
///
/// let rotation_matrix = rotation_eci_to_nsw(x_eci, x_sun);
/// ```
pub fn rotation_eci_to_nsw(x_eci: SVector6, x_sun: SVector6) -> SMatrix3 {
    rotation_nsw_to_eci(x_eci, x_sun).transpose()
}

/// Computes the angular velocity of the NSW frame with respect to the ECI frame, expressed in
/// NSW axes.
///
/// The rate follows from the time derivatives of the basis vectors,
/// `ω = [ẏ·ẑ, ż·x̂, ẋ·ŷ]`, with the nadir derivative from the spacecraft velocity and the
/// Sun-direction derivative from the Sun's velocity relative to the spacecraft. It is purely
/// kinematic and needs no gravitational parameter. Passing a Sun state with zero velocity gives
/// the fixed-Sun approximation, which omits a term of order 2e-7 rad/s for an Earth orbit.
///
/// On the nadir-aligned fallback branch the orbit plane is taken as fixed, so that branch is
/// exact only under two-body motion. As the Sun approaches the nadir line the rate grows as the
/// reciprocal of the sine of the Sun-nadir angle until the fallback engages.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the NSW frame relative to ECI, expressed in NSW axes (rad/s)
///
/// # References:
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
///
/// let omega = omega_nsw(x_eci, x_sun);
/// ```
pub fn omega_nsw(x_eci: SVector6, x_sun: SVector6) -> Vector3<f64> {
    nsw_axes(x_eci, x_sun).1
}

/// 6x6 Jacobian taking an NSW state covariance into the Earth-Centered Inertial (ECI) frame.
///
/// With `R` the NSW-to-ECI rotation and `ω` the NSW angular velocity from [`omega_nsw`], the
/// Jacobian is `[[R, 0], [R [ω]×, R]]` for the rotating variant and `[[R, 0], [0, R]]` for the
/// inertial snapshot.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_eci = J P_nsw Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
///
/// let j = jacobian_nsw_to_eci(x_eci, x_sun, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_nsw_to_eci(
    x_eci: SVector6,
    x_sun: SVector6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    let (r, omega) = nsw_axes(x_eci, x_sun);
    jacobian_to_inertial(&r, &omega, variant)
}

/// 6x6 Jacobian taking a state covariance in the Earth-Centered Inertial (ECI) frame into NSW
/// axes. Exact inverse of [`jacobian_nsw_to_eci`].
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_nsw = J P_eci Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
///
/// let j = jacobian_eci_to_nsw(x_eci, x_sun, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_eci_to_nsw(
    x_eci: SVector6,
    x_sun: SVector6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    let (r, omega) = nsw_axes(x_eci, x_sun);
    jacobian_from_inertial(&r.transpose(), &omega, variant)
}

/// Transforms a 6x6 state covariance from NSW axes into the Earth-Centered Inertial (ECI)
/// frame.
///
/// Applies the congruence `P_eci = J P_nsw Jᵀ` with `J` from [`jacobian_nsw_to_eci`], and
/// symmetrizes the result.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in NSW axes (m², m²/s, m²/s²)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_eci`: 6x6 state covariance in the ECI frame (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
/// let p_nsw = SMatrix6::identity();
///
/// let p_eci = covariance_nsw_to_eci(x_eci, x_sun, &p_nsw, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_nsw_to_eci(
    x_eci: SVector6,
    x_sun: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_nsw_to_eci(x_eci, x_sun, variant))
}

/// Transforms a 6x6 state covariance from the Earth-Centered Inertial (ECI) frame into NSW
/// axes.
///
/// Applies the congruence `P_nsw = J P_eci Jᵀ` with `J` from [`jacobian_eci_to_nsw`], and
/// symmetrizes the result.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in the ECI frame (m², m²/s, m²/s²)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_nsw`: 6x6 state covariance in NSW axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_eci = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
/// let p_eci = SMatrix6::identity();
///
/// let p_nsw = covariance_eci_to_nsw(x_eci, x_sun, &p_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_eci_to_nsw(
    x_eci: SVector6,
    x_sun: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_eci_to_nsw(x_eci, x_sun, variant))
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Nadir, Sun, Normal (NSW) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_nsw`: 6D relative state of the deputy with respect to the chief in the NSW frame [ρ_X, ρ_Y, ρ_Z, ρ̇_X, ρ̇_Y, ρ̇_Z] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_chief = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
///
/// let x_rel_nsw = state_eci_to_nsw(x_chief, x_deputy, x_sun);
/// ```
pub fn state_eci_to_nsw(x_chief: SVector6, x_deputy: SVector6, x_sun: SVector6) -> SVector6 {
    let (r, omega) = nsw_axes(x_chief, x_sun);
    relative_state_to_frame(&r.transpose(), &omega, x_chief, x_deputy)
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Nadir, Sun, Normal (NSW) frame to the absolute state of the deputy in the
/// Earth-Centered Inertial (ECI) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_nsw`: 6D relative state of the deputy with respect to the chief in the NSW frame [ρ_X, ρ_Y, ρ_Z, ρ̇_X, ρ̇_Y, ρ̇_Z] (m, m/s)
/// - `x_sun`: 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat, TimeSystem};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::ephemerides::sun_position;
/// use brahe::relative_motion::*;
/// use brahe::time::Epoch;
///
/// let x_chief = state_koe_to_eci(SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0), AngleFormat::Degrees);
/// let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
/// let r_sun = sun_position(epc);
/// let x_sun = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
/// let x_rel_nsw = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_nsw_to_eci(x_chief, x_rel_nsw, x_sun);
/// ```
pub fn state_nsw_to_eci(x_chief: SVector6, x_rel_nsw: SVector6, x_sun: SVector6) -> SVector6 {
    let (r, omega) = nsw_axes(x_chief, x_sun);
    relative_state_from_frame(&r.transpose(), &omega, x_chief, x_rel_nsw)
}

/// Computes the NSW-to-ECI rotation matrix for each pair of spacecraft and Sun states.
///
/// Batch form of [`rotation_nsw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci` and `x_sun` arguments follow the broadcast rule: each has length 1 or the
/// common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming NSW -> ECI, in input order
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_nsw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let r = rotations_nsw_to_eci(&[x, x], &[s]).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_nsw_to_eci(
    x_eci: &[SVector6],
    x_sun: &[SVector6],
) -> Result<Vec<SMatrix3>, BraheError> {
    batch_zip(|x, s| rotation_nsw_to_eci(*x, *s), x_eci, x_sun)
}

/// Computes the ECI-to-NSW rotation matrix for each pair of spacecraft and Sun states.
///
/// Batch form of [`rotation_eci_to_nsw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci` and `x_sun` arguments follow the broadcast rule: each has length 1 or the
/// common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECI -> NSW, in input order
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_eci_to_nsw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let r = rotations_eci_to_nsw(&[x, x], &[s]).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_eci_to_nsw(
    x_eci: &[SVector6],
    x_sun: &[SVector6],
) -> Result<Vec<SMatrix3>, BraheError> {
    batch_zip(|x, s| rotation_eci_to_nsw(*x, *s), x_eci, x_sun)
}

/// Computes the NSW frame angular velocity for each pair of spacecraft and Sun states.
///
/// Batch form of [`omega_nsw`]. Evaluation runs on the global thread pool for large inputs.
///
/// The `x_eci` and `x_sun` arguments follow the broadcast rule: each has length 1 or the
/// common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Angular velocities of the NSW frame relative to ECI, expressed in NSW axes, in input
///   order. Units: (*rad/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::omegas_nsw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let omega = omegas_nsw(&[x, x], &[s]).unwrap();
/// assert_eq!(omega.len(), 2);
/// ```
pub fn omegas_nsw(x_eci: &[SVector6], x_sun: &[SVector6]) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_zip(|x, s| omega_nsw(*x, *s), x_eci, x_sun)
}

/// Computes the NSW-to-ECI covariance Jacobian for each pair of spacecraft and Sun states.
///
/// Batch form of [`jacobian_nsw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci` and `x_sun` arguments follow the broadcast rule: each has length 1 or the
/// common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_eci = J P_nsw Jᵀ`, in input order
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_nsw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let j = jacobians_nsw_to_eci(&[x, x], &[s], OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_nsw_to_eci(
    x_eci: &[SVector6],
    x_sun: &[SVector6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(|x, s| jacobian_nsw_to_eci(*x, *s, variant), x_eci, x_sun)
}

/// Computes the ECI-to-NSW covariance Jacobian for each pair of spacecraft and Sun states.
///
/// Batch form of [`jacobian_eci_to_nsw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci` and `x_sun` arguments follow the broadcast rule: each has length 1 or the
/// common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_nsw = J P_eci Jᵀ`, in input order
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_eci_to_nsw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let j = jacobians_eci_to_nsw(&[x, x], &[s], OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_eci_to_nsw(
    x_eci: &[SVector6],
    x_sun: &[SVector6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(|x, s| jacobian_eci_to_nsw(*x, *s, variant), x_eci, x_sun)
}

/// Transforms each state covariance in `covariances` from NSW axes into the Earth-Centered
/// Inertial (ECI) frame.
///
/// Batch form of [`covariance_nsw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci`, `x_sun`, and `covariances` arguments follow the broadcast rule: each has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in NSW axes, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - State covariances in the ECI frame, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples
/// ```
/// use brahe::SMatrix6;
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_nsw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_eci = covariances_nsw_to_eci(&[x], &[s], &p, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_eci.len(), 2);
/// ```
pub fn covariances_nsw_to_eci(
    x_eci: &[SVector6],
    x_sun: &[SVector6],
    covariances: &[SMatrix6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip3(
        |x, s, p| covariance_nsw_to_eci(*x, *s, p, variant),
        x_eci,
        x_sun,
        covariances,
    )
}

/// Transforms each state covariance in `covariances` from the Earth-Centered Inertial (ECI)
/// frame into NSW axes.
///
/// Batch form of [`covariance_eci_to_nsw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci`, `x_sun`, and `covariances` arguments follow the broadcast rule: each has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in the ECI frame, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `variant`: Whether the NSW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - State covariances in NSW axes, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples
/// ```
/// use brahe::SMatrix6;
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_eci_to_nsw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_nsw = covariances_eci_to_nsw(&[x], &[s], &p, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_nsw.len(), 2);
/// ```
pub fn covariances_eci_to_nsw(
    x_eci: &[SVector6],
    x_sun: &[SVector6],
    covariances: &[SMatrix6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip3(
        |x, s, p| covariance_eci_to_nsw(*x, *s, p, variant),
        x_eci,
        x_sun,
        covariances,
    )
}

/// Computes the NSW relative state of each deputy with respect to its chief.
///
/// Batch form of [`state_eci_to_nsw`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// The `x_chief`, `x_deputy`, and `x_sun` arguments follow the broadcast rule: each has
/// length 1 or the common batch length, so one chief may be paired with many deputies and
/// vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_deputy`: Deputy Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy relative states in the chief NSW frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_eci_to_nsw;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let deputies = vec![
///     state_koe_to_eci(vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]), AngleFormat::Degrees),
///     state_koe_to_eci(vector6_from_array([R_EARTH + 702e3, 0.0012, 97.82, 15.02, 30.02, 45.02]), AngleFormat::Degrees),
/// ];
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let rel = states_eci_to_nsw(&[chief], &deputies, &[s]).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_eci_to_nsw(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
    x_sun: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip3(
        |c, d, s| state_eci_to_nsw(*c, *d, *s),
        x_chief,
        x_deputy,
        x_sun,
    )
}

/// Computes the ECI state of each deputy from its NSW relative state and chief.
///
/// Batch form of [`state_nsw_to_eci`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// The `x_chief`, `x_rel_nsw`, and `x_sun` arguments follow the broadcast rule: each has
/// length 1 or the common batch length, so one chief may be paired with many deputies and
/// vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_nsw`: Deputy relative states in the chief NSW frame, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_sun`: Sun states relative to the same center, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy Cartesian ECI states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NSW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_nsw_to_eci;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let s = vector6_from_array([1.4e11, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let deputies = states_nsw_to_eci(&[chief], &rel, &[s]).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_nsw_to_eci(
    x_chief: &[SVector6],
    x_rel_nsw: &[SVector6],
    x_sun: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip3(
        |c, r, s| state_nsw_to_eci(*c, *r, *s),
        x_chief,
        x_rel_nsw,
        x_sun,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::{AU, R_EARTH};
    use crate::coordinates::state_koe_to_eci;
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::{block_diagonal, skew_symmetric};
    use crate::orbits::mean_motion;
    use crate::relative_motion::rotation_rtn_to_eci;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    const SMA: f64 = R_EARTH + 700e3;

    fn sc_state(dt: f64) -> SVector6 {
        let n = mean_motion(SMA, AngleFormat::Degrees);
        state_koe_to_eci(
            SVector6::new(SMA, 0.05, 97.8, 15.0, 30.0, 45.0 + n * dt),
            AngleFormat::Degrees,
        )
    }

    /// A Sun moving linearly at a heliocentric-scale rate, so the frame rate
    /// includes the Sun-direction term.
    fn sun_state(dt: f64) -> SVector6 {
        let r = Vector3::new(0.9 * AU, 0.4 * AU, 0.17 * AU);
        let v = Vector3::new(-12.0e3, 26.0e3, 11.0e3);
        let r_t = r + v * dt;
        SVector6::new(r_t[0], r_t[1], r_t[2], v[0], v[1], v[2])
    }

    #[test]
    #[parallel]
    fn test_rotation_nsw_to_eci_axes_match_definition() {
        let x = sc_state(0.0);
        let s = sun_state(0.0);
        let r = x.fixed_rows::<3>(0).into_owned();
        let sun_dir = (s.fixed_rows::<3>(0) - r).normalize();

        let m = rotation_nsw_to_eci(x, s);
        let x_axis: Vector3<f64> = m.column(0).into();
        let y_axis: Vector3<f64> = m.column(1).into();
        let z_axis: Vector3<f64> = m.column(2).into();

        assert_abs_diff_eq!(x_axis, -r / r.norm(), epsilon = 1e-15);
        assert_abs_diff_eq!(x_axis.dot(&y_axis), 0.0, epsilon = 1e-15);
        assert!(y_axis.dot(&sun_dir) > 0.0);
        // Y is the Sun direction with its nadir component removed
        let projected = sun_dir - sun_dir.dot(&x_axis) * x_axis;
        assert_abs_diff_eq!(y_axis, projected / projected.norm(), epsilon = 1e-15);
        assert_abs_diff_eq!(z_axis, x_axis.cross(&y_axis), epsilon = 1e-15);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_nsw_is_invariant_to_sun_offset_along_nadir() {
        // X lies along the position vector, so passing the Sun state
        // relative to the spacecraft instead of relative to the center
        // changes neither the axes nor the rate.
        let x = sc_state(0.0);
        let s_center = sun_state(0.0);
        let s_shifted = s_center - x;

        assert_abs_diff_eq!(
            rotation_nsw_to_eci(x, s_center),
            rotation_nsw_to_eci(x, s_shifted),
            epsilon = 1e-12
        );
        let omega_center = omega_nsw(x, s_center);
        let omega_shifted = omega_nsw(x, s_shifted);
        assert_abs_diff_eq!(
            (omega_center - omega_shifted).norm() / omega_center.norm(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_rotation_nsw_sun_along_nadir_falls_back_to_along_track() {
        let x = sc_state(0.0);
        let r = x.fixed_rows::<3>(0).into_owned();
        let r_sun = r - r / r.norm() * AU;
        let s = SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
        let m = rotation_nsw_to_eci(x, s);
        let y: Vector3<f64> = m.column(1).into();
        let t_axis: Vector3<f64> = rotation_rtn_to_eci(x).column(1).into();
        assert_abs_diff_eq!(y, t_axis, epsilon = 1e-12);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_nsw_is_transpose() {
        let x = sc_state(0.0);
        let s = sun_state(0.0);
        assert_eq!(
            rotation_eci_to_nsw(x, s),
            rotation_nsw_to_eci(x, s).transpose()
        );
    }

    #[test]
    #[parallel]
    fn test_omega_nsw_matches_finite_difference_with_moving_sun() {
        let dt = 0.05;
        let x0 = sc_state(0.0);
        let s0 = sun_state(0.0);
        let r_dot = (rotation_eci_to_nsw(sc_state(dt), sun_state(dt))
            - rotation_eci_to_nsw(sc_state(-dt), sun_state(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_eci_to_nsw(x0, s0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_nsw(x0, s0), epsilon = 1e-9);
        // All three components are generally nonzero
        let omega = omega_nsw(x0, s0);
        assert!(omega[0].abs() > 1e-8 && omega[1].abs() > 1e-8 && omega[2].abs() > 1e-8);
    }

    #[test]
    #[parallel]
    fn test_omega_nsw_fixed_sun_matches_finite_difference() {
        let dt = 0.05;
        let x0 = sc_state(0.0);
        let s = sun_state(0.0);
        let s_fixed = SVector6::new(s[0], s[1], s[2], 0.0, 0.0, 0.0);
        let r_dot = (rotation_eci_to_nsw(sc_state(dt), s_fixed)
            - rotation_eci_to_nsw(sc_state(-dt), s_fixed))
            / (2.0 * dt);
        let omega_fd =
            angular_velocity_from_rotation_rate(&rotation_eci_to_nsw(x0, s_fixed), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_nsw(x0, s_fixed), epsilon = 1e-9);
        assert!((omega_nsw(x0, s_fixed) - omega_nsw(x0, s)).norm() > 1e-9);
    }

    #[test]
    #[parallel]
    fn test_omega_nsw_degenerate_branch_matches_finite_difference() {
        // Keep the Sun exactly along nadir at every epoch so the fallback
        // branch is exercised; its rate is then the along-track frame's rate.
        let dt = 0.05;
        let sun_on_nadir = |x: SVector6| {
            let r = x.fixed_rows::<3>(0).into_owned();
            let r_sun = r - r / r.norm() * AU;
            SVector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0)
        };
        let x0 = sc_state(0.0);
        let r_dot = (rotation_eci_to_nsw(sc_state(dt), sun_on_nadir(sc_state(dt)))
            - rotation_eci_to_nsw(sc_state(-dt), sun_on_nadir(sc_state(-dt))))
            / (2.0 * dt);
        let omega_fd =
            angular_velocity_from_rotation_rate(&rotation_eci_to_nsw(x0, sun_on_nadir(x0)), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_nsw(x0, sun_on_nadir(x0)), epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_jacobian_nsw_to_eci_inertial_is_block_diagonal() {
        let (x, s) = (sc_state(0.0), sun_state(0.0));
        let j = jacobian_nsw_to_eci(x, s, OrbitRelativeFrameVariant::Inertial);
        let r = rotation_nsw_to_eci(x, s);
        assert_abs_diff_eq!((j - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_jacobian_nsw_to_eci_rotating_coupling() {
        let (x, s) = (sc_state(0.0), sun_state(0.0));
        let j = jacobian_nsw_to_eci(x, s, OrbitRelativeFrameVariant::Rotating);
        let expected = rotation_nsw_to_eci(x, s) * skew_symmetric(&omega_nsw(x, s));
        let coupling: SMatrix3 = j.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!((coupling - expected).norm(), 0.0, epsilon = 1e-18);
    }

    #[test]
    #[parallel]
    fn test_jacobian_nsw_eci_inverse_identity() {
        let (x, s) = (sc_state(0.0), sun_state(0.0));
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let forward = jacobian_nsw_to_eci(x, s, variant);
            let inverse = jacobian_eci_to_nsw(x, s, variant);
            assert_abs_diff_eq!(
                (inverse * forward - SMatrix6::identity()).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[parallel]
    fn test_covariance_nsw_eci_round_trip() {
        let (x, s) = (sc_state(0.0), sun_state(0.0));
        let mut p = SMatrix6::zeros();
        for i in 0..3 {
            p[(i, i)] = 100.0;
            p[(3 + i, 3 + i)] = 0.01;
        }
        p[(0, 1)] = 25.0;
        p[(1, 0)] = 25.0;
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let p_eci = covariance_nsw_to_eci(x, s, &p, variant);
            let p_back = covariance_eci_to_nsw(x, s, &p_eci, variant);
            assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    #[parallel]
    fn test_state_nsw_to_eci_round_trip() {
        let (x_chief, s) = (sc_state(0.0), sun_state(0.0));
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        let x_deputy = state_nsw_to_eci(x_chief, x_rel, s);
        assert_abs_diff_eq!(
            state_eci_to_nsw(x_chief, x_deputy, s),
            x_rel,
            epsilon = 1e-8
        );
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_nsw_applies_transport_term() {
        let (x_chief, s) = (sc_state(0.0), sun_state(0.0));
        let x_deputy = x_chief + SVector6::new(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let r = rotation_eci_to_nsw(x_chief, s);
        let rho = r * Vector3::new(1000.0, 0.0, 0.0);
        let expected_v = -omega_nsw(x_chief, s).cross(&rho);
        let rel = state_eci_to_nsw(x_chief, x_deputy, s);
        assert_abs_diff_eq!(
            rel.fixed_rows::<3>(3).into_owned(),
            expected_v,
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_batch_nsw_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3).map(|i| sc_state(10.0 * i as f64)).collect();
        let suns: Vec<SVector6> = (0..3).map(|i| sun_state(10.0 * i as f64)).collect();
        let deputies: Vec<SVector6> = chiefs
            .iter()
            .map(|c| c + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3))
            .collect();
        let covs: Vec<SMatrix6> = (0..3)
            .map(|i| SMatrix6::identity() * (i as f64 + 1.0))
            .collect();

        let rot = rotations_nsw_to_eci(&chiefs, &suns).unwrap();
        let rot_inv = rotations_eci_to_nsw(&chiefs, &suns).unwrap();
        let omegas = omegas_nsw(&chiefs, &suns).unwrap();
        let rot_bsun = rotations_nsw_to_eci(&chiefs, &suns[..1]).unwrap();
        let omegas_bsun = omegas_nsw(&chiefs, &suns[..1]).unwrap();
        let rel = states_eci_to_nsw(&chiefs, &deputies, &suns).unwrap();
        let back = states_nsw_to_eci(&chiefs, &rel, &suns).unwrap();

        for i in 0..3 {
            assert_eq!(rot[i], rotation_nsw_to_eci(chiefs[i], suns[i]));
            assert_eq!(rot_inv[i], rotation_eci_to_nsw(chiefs[i], suns[i]));
            assert_eq!(omegas[i], omega_nsw(chiefs[i], suns[i]));
            assert_eq!(rot_bsun[i], rotation_nsw_to_eci(chiefs[i], suns[0]));
            assert_eq!(omegas_bsun[i], omega_nsw(chiefs[i], suns[0]));
            assert_eq!(rel[i], state_eci_to_nsw(chiefs[i], deputies[i], suns[i]));
            assert_eq!(back[i], state_nsw_to_eci(chiefs[i], rel[i], suns[i]));
        }

        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let jac = jacobians_nsw_to_eci(&chiefs, &suns, variant).unwrap();
            let jac_inv = jacobians_eci_to_nsw(&chiefs, &suns, variant).unwrap();
            let cov = covariances_nsw_to_eci(&chiefs, &suns, &covs, variant).unwrap();
            let cov_inv = covariances_eci_to_nsw(&chiefs, &suns, &covs, variant).unwrap();
            let cov_bcov = covariances_nsw_to_eci(&chiefs, &suns, &covs[..1], variant).unwrap();

            for i in 0..3 {
                assert_eq!(jac[i], jacobian_nsw_to_eci(chiefs[i], suns[i], variant));
                assert_eq!(jac_inv[i], jacobian_eci_to_nsw(chiefs[i], suns[i], variant));
                assert_eq!(
                    cov[i],
                    covariance_nsw_to_eci(chiefs[i], suns[i], &covs[i], variant)
                );
                assert_eq!(
                    cov_inv[i],
                    covariance_eci_to_nsw(chiefs[i], suns[i], &covs[i], variant)
                );
                assert_eq!(
                    cov_bcov[i],
                    covariance_nsw_to_eci(chiefs[i], suns[i], &covs[0], variant)
                );
            }
        }

        assert!(rotations_nsw_to_eci(&chiefs[..2], &suns).is_err());
        assert!(
            covariances_nsw_to_eci(
                &chiefs[..2],
                &suns,
                &covs,
                OrbitRelativeFrameVariant::Rotating
            )
            .is_err()
        );
        assert!(rotations_nsw_to_eci(&[], &[]).unwrap().is_empty());
    }
}
