/*!
 * Solar-system body position, velocity, and state queries backed by native
 * SPK ephemeris kernels.
 *
 * All outputs are geocentric (center = Earth, NAIF 399) in the kernel's
 * inertial frame. For DE4xx kernels the "J2000" frame label denotes ICRF
 * axes (NAIF Frames Required Reading), so values are GCRF-compatible
 * directly — no frame-bias rotation is applied.
 */

use nalgebra::{Vector3, Vector6};

use crate::time::Epoch;
use crate::utils::BraheError;

use super::kernels::NAIFKernel;
use super::naif_id::NAIFId;
use super::registry::{spk_position_from_kernel, spk_state_from_kernel, spk_velocity_from_kernel};

macro_rules! body_de_functions {
    ($body_name:literal, $target:expr, $pos_fn:ident, $vel_fn:ident, $state_fn:ident) => {
        #[doc = concat!("Calculate the position of ", $body_name, " relative to Earth using NAIF DE ephemerides.")]
        ///
        /// The result is expressed in the kernel's inertial frame (ICRF axes,
        /// GCRF-compatible; NAIF labels this "J2000").
        ///
        /// # Arguments
        ///
        /// * `epc` - Epoch at which to calculate the position
        /// * `kernel` - Which DE kernel to use
        ///
        /// # Returns
        ///
        #[doc = concat!("* `Ok(Vector3<f64>)` - Position of ", $body_name, " in the GCRF frame. Units: [m]")]
        /// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use brahe::spice::{NAIFKernel, ", stringify!($pos_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let r = ", stringify!($pos_fn), "(epc, NAIFKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $pos_fn(epc: Epoch, kernel: NAIFKernel) -> Result<Vector3<f64>, BraheError> {
            spk_position_from_kernel(kernel, $target, NAIFId::Earth, epc)
        }

        #[doc = concat!("Calculate the velocity of ", $body_name, " relative to Earth using NAIF DE ephemerides.")]
        ///
        /// The result is expressed in the kernel's inertial frame (ICRF axes,
        /// GCRF-compatible; NAIF labels this "J2000").
        ///
        /// # Arguments
        ///
        /// * `epc` - Epoch at which to calculate the velocity
        /// * `kernel` - Which DE kernel to use
        ///
        /// # Returns
        ///
        #[doc = concat!("* `Ok(Vector3<f64>)` - Velocity of ", $body_name, " in the GCRF frame. Units: [m/s]")]
        /// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use brahe::spice::{NAIFKernel, ", stringify!($vel_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let v = ", stringify!($vel_fn), "(epc, NAIFKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $vel_fn(epc: Epoch, kernel: NAIFKernel) -> Result<Vector3<f64>, BraheError> {
            spk_velocity_from_kernel(kernel, $target, NAIFId::Earth, epc)
        }

        #[doc = concat!("Calculate the state (position and velocity) of ", $body_name, " relative to Earth using NAIF DE ephemerides.")]
        ///
        /// The result is expressed in the kernel's inertial frame (ICRF axes,
        /// GCRF-compatible; NAIF labels this "J2000"). Computing the state
        /// shares a single record lookup between position and velocity.
        ///
        /// # Arguments
        ///
        /// * `epc` - Epoch at which to calculate the state
        /// * `kernel` - Which DE kernel to use
        ///
        /// # Returns
        ///
        #[doc = concat!("* `Ok(Vector6<f64>)` - State [x, y, z, vx, vy, vz] of ", $body_name, " in the GCRF frame. Units: [m, m/s]")]
        /// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("use brahe::spice::{NAIFKernel, ", stringify!($state_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let x = ", stringify!($state_fn), "(epc, NAIFKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $state_fn(epc: Epoch, kernel: NAIFKernel) -> Result<Vector6<f64>, BraheError> {
            spk_state_from_kernel(kernel, $target, NAIFId::Earth, epc)
        }
    };
}

/// Satellite-system SPK kernel that carries a planet body center relative to
/// its planetary-system barycenter.
///
/// The DE kernels only provide the planetary-system *barycenters* (not the
/// planet body centers) for the outer planets, so recovering a true body
/// center requires the body-rel-barycenter leg from the planet's
/// satellite-system kernel.
///
/// # Arguments
/// - `planet`: A planet body center ([`NAIFId::Mars`], [`NAIFId::Jupiter`],
///   [`NAIFId::Saturn`], [`NAIFId::Uranus`], or [`NAIFId::Neptune`])
///
/// # Returns
/// - The satellite-system [`NAIFKernel`] providing that planet's body center
///
/// # Panics
/// Panics if `planet` is not one of the five supported outer planets.
pub(crate) const fn system_kernel(planet: NAIFId) -> NAIFKernel {
    match planet.id() {
        499 => NAIFKernel::Mar099s,
        599 => NAIFKernel::Jup365,
        699 => NAIFKernel::Sat441,
        799 => NAIFKernel::Ura184,
        899 => NAIFKernel::Nep097,
        _ => panic!("no satellite-system kernel for this body"),
    }
}

macro_rules! body_center_de_functions {
    ($body_name:literal, $body:expr, $barycenter:expr, $kernel_name:literal, $kernel_size:literal,
     $pos_fn:ident, $vel_fn:ident, $state_fn:ident) => {
        #[doc = concat!("Calculate the position of ", $body_name, " (body center) relative to Earth using NAIF DE ephemerides.")]
        ///
        /// Combines the planetary-system barycenter from the DE `kernel` with
        /// the body-center offset from the planet's satellite-system kernel
        #[doc = concat!("(`", $kernel_name, "`, ~", $kernel_size, "), which is auto-downloaded and")]
        /// loaded on first use. For third-body force applications prefer the
        /// `_barycenter_` variant, which needs only the DE kernel and is
        /// numerically identical to the standard third-body formulation.
        ///
        /// The result is expressed in the kernel's inertial frame (ICRF axes,
        /// GCRF-compatible; NAIF labels this "J2000").
        ///
        /// # Arguments
        ///
        /// * `epc` - Epoch at which to calculate the position
        /// * `kernel` - Which DE kernel to use for the barycenter leg
        ///
        /// # Returns
        ///
        #[doc = concat!("* `Ok(Vector3<f64>)` - Position of ", $body_name, " in the GCRF frame. Units: [m]")]
        /// * `Err(BraheError)` - If a kernel cannot be loaded or queried
        ///
        /// # Example
        ///
        /// ```no_run
        #[doc = concat!("use brahe::spice::{NAIFKernel, ", stringify!($pos_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let r = ", stringify!($pos_fn), "(epc, NAIFKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $pos_fn(epc: Epoch, kernel: NAIFKernel) -> Result<Vector3<f64>, BraheError> {
            let r_bary = spk_position_from_kernel(kernel, $barycenter, NAIFId::Earth, epc)?;
            let r_body = spk_position_from_kernel(system_kernel($body), $body, $barycenter, epc)?;
            Ok(r_bary + r_body)
        }

        #[doc = concat!("Calculate the velocity of ", $body_name, " (body center) relative to Earth using NAIF DE ephemerides.")]
        ///
        /// Combines the planetary-system barycenter from the DE `kernel` with
        /// the body-center offset from the planet's satellite-system kernel
        #[doc = concat!("(`", $kernel_name, "`, ~", $kernel_size, "), which is auto-downloaded and")]
        /// loaded on first use. For third-body force applications prefer the
        /// `_barycenter_` variant, which needs only the DE kernel.
        ///
        /// The result is expressed in the kernel's inertial frame (ICRF axes,
        /// GCRF-compatible; NAIF labels this "J2000").
        ///
        /// # Arguments
        ///
        /// * `epc` - Epoch at which to calculate the velocity
        /// * `kernel` - Which DE kernel to use for the barycenter leg
        ///
        /// # Returns
        ///
        #[doc = concat!("* `Ok(Vector3<f64>)` - Velocity of ", $body_name, " in the GCRF frame. Units: [m/s]")]
        /// * `Err(BraheError)` - If a kernel cannot be loaded or queried
        ///
        /// # Example
        ///
        /// ```no_run
        #[doc = concat!("use brahe::spice::{NAIFKernel, ", stringify!($vel_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let v = ", stringify!($vel_fn), "(epc, NAIFKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $vel_fn(epc: Epoch, kernel: NAIFKernel) -> Result<Vector3<f64>, BraheError> {
            let v_bary = spk_velocity_from_kernel(kernel, $barycenter, NAIFId::Earth, epc)?;
            let v_body = spk_velocity_from_kernel(system_kernel($body), $body, $barycenter, epc)?;
            Ok(v_bary + v_body)
        }

        #[doc = concat!("Calculate the state (position and velocity) of ", $body_name, " (body center) relative to Earth using NAIF DE ephemerides.")]
        ///
        /// Combines the planetary-system barycenter from the DE `kernel` with
        /// the body-center offset from the planet's satellite-system kernel
        #[doc = concat!("(`", $kernel_name, "`, ~", $kernel_size, "), which is auto-downloaded and")]
        /// loaded on first use. For third-body force applications prefer the
        /// `_barycenter_` variant, which needs only the DE kernel.
        ///
        /// The result is expressed in the kernel's inertial frame (ICRF axes,
        /// GCRF-compatible; NAIF labels this "J2000").
        ///
        /// # Arguments
        ///
        /// * `epc` - Epoch at which to calculate the state
        /// * `kernel` - Which DE kernel to use for the barycenter leg
        ///
        /// # Returns
        ///
        #[doc = concat!("* `Ok(Vector6<f64>)` - State [x, y, z, vx, vy, vz] of ", $body_name, " in the GCRF frame. Units: [m, m/s]")]
        /// * `Err(BraheError)` - If a kernel cannot be loaded or queried
        ///
        /// # Example
        ///
        /// ```no_run
        #[doc = concat!("use brahe::spice::{NAIFKernel, ", stringify!($state_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let x = ", stringify!($state_fn), "(epc, NAIFKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $state_fn(epc: Epoch, kernel: NAIFKernel) -> Result<Vector6<f64>, BraheError> {
            let x_bary = spk_state_from_kernel(kernel, $barycenter, NAIFId::Earth, epc)?;
            let x_body = spk_state_from_kernel(system_kernel($body), $body, $barycenter, epc)?;
            Ok(x_bary + x_body)
        }
    };
}

body_de_functions!(
    "the Sun",
    NAIFId::Sun,
    sun_position_de,
    sun_velocity_de,
    sun_state_de
);
body_de_functions!(
    "the Moon",
    NAIFId::Moon,
    moon_position_de,
    moon_velocity_de,
    moon_state_de
);
body_de_functions!(
    "Mercury",
    NAIFId::Mercury,
    mercury_position_de,
    mercury_velocity_de,
    mercury_state_de
);
body_de_functions!(
    "Venus",
    NAIFId::Venus,
    venus_position_de,
    venus_velocity_de,
    venus_state_de
);
body_de_functions!(
    "the Mars system barycenter",
    NAIFId::MarsBarycenter,
    mars_barycenter_position_de,
    mars_barycenter_velocity_de,
    mars_barycenter_state_de
);
body_de_functions!(
    "the Jupiter system barycenter",
    NAIFId::JupiterBarycenter,
    jupiter_barycenter_position_de,
    jupiter_barycenter_velocity_de,
    jupiter_barycenter_state_de
);
body_de_functions!(
    "the Saturn system barycenter",
    NAIFId::SaturnBarycenter,
    saturn_barycenter_position_de,
    saturn_barycenter_velocity_de,
    saturn_barycenter_state_de
);
body_de_functions!(
    "the Uranus system barycenter",
    NAIFId::UranusBarycenter,
    uranus_barycenter_position_de,
    uranus_barycenter_velocity_de,
    uranus_barycenter_state_de
);
body_de_functions!(
    "the Neptune system barycenter",
    NAIFId::NeptuneBarycenter,
    neptune_barycenter_position_de,
    neptune_barycenter_velocity_de,
    neptune_barycenter_state_de
);
body_de_functions!(
    "the Solar System Barycenter",
    NAIFId::SolarSystemBarycenter,
    solar_system_barycenter_position_de,
    solar_system_barycenter_velocity_de,
    solar_system_barycenter_state_de
);

body_center_de_functions!(
    "Mars",
    NAIFId::Mars,
    NAIFId::MarsBarycenter,
    "mar099s",
    "68 MB",
    mars_position_de,
    mars_velocity_de,
    mars_state_de
);
body_center_de_functions!(
    "Jupiter",
    NAIFId::Jupiter,
    NAIFId::JupiterBarycenter,
    "jup365",
    "1.1 GB",
    jupiter_position_de,
    jupiter_velocity_de,
    jupiter_state_de
);
body_center_de_functions!(
    "Saturn",
    NAIFId::Saturn,
    NAIFId::SaturnBarycenter,
    "sat441",
    "662 MB",
    saturn_position_de,
    saturn_velocity_de,
    saturn_state_de
);
body_center_de_functions!(
    "Uranus",
    NAIFId::Uranus,
    NAIFId::UranusBarycenter,
    "ura184",
    "387 MB",
    uranus_position_de,
    uranus_velocity_de,
    uranus_state_de
);
body_center_de_functions!(
    "Neptune",
    NAIFId::Neptune,
    NAIFId::NeptuneBarycenter,
    "nep097",
    "105 MB",
    neptune_position_de,
    neptune_velocity_de,
    neptune_state_de
);

/// Calculate the position of the Solar System Barycenter in the GCRF frame using NAIF DE ephemeris.
///
/// Convenience alias for [`solar_system_barycenter_position_de`].
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the SSB position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of the SSB in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{NAIFKernel, ssb_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_ssb = ssb_position_de(epc, NAIFKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_position_de(epc: Epoch, kernel: NAIFKernel) -> Result<Vector3<f64>, BraheError> {
    solar_system_barycenter_position_de(epc, kernel)
}

/// Calculate the velocity of the Solar System Barycenter in the GCRF frame using NAIF DE ephemeris.
///
/// Convenience alias for [`solar_system_barycenter_velocity_de`].
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the SSB velocity
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Velocity of the SSB in the GCRF frame. Units: [m/s]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{NAIFKernel, ssb_velocity_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let v_ssb = ssb_velocity_de(epc, NAIFKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_velocity_de(epc: Epoch, kernel: NAIFKernel) -> Result<Vector3<f64>, BraheError> {
    solar_system_barycenter_velocity_de(epc, kernel)
}

/// Calculate the state (position and velocity) of the Solar System Barycenter in the
/// GCRF frame using NAIF DE ephemeris.
///
/// Convenience alias for [`solar_system_barycenter_state_de`].
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the SSB state
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector6<f64>)` - State [x, y, z, vx, vy, vz] of the SSB in the GCRF frame. Units: [m, m/s]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{NAIFKernel, ssb_state_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let x_ssb = ssb_state_de(epc, NAIFKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_state_de(epc: Epoch, kernel: NAIFKernel) -> Result<Vector6<f64>, BraheError> {
    solar_system_barycenter_state_de(epc, kernel)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;
    use serial_test::serial;

    use super::*;
    use crate::orbit_dynamics::ephemerides::{moon_position, sun_position};
    use crate::utils::testing::setup_global_test_spice;

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_sun_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();

        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let r_analytical = sun_position(epc);
        let r_de = sun_position_de(epc, NAIFKernel::DE440s).unwrap();

        let dot = r_analytical.dot(&r_de) / (r_analytical.norm() * r_de.norm());
        let angle = dot.acos() * (180.0 / std::f64::consts::PI);
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_moon_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();

        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let r_analytical = moon_position(epc);
        let r_de = moon_position_de(epc, NAIFKernel::DE440s).unwrap();

        let dot = r_analytical.dot(&r_de) / (r_analytical.norm() * r_de.norm());
        let angle = dot.acos() * (180.0 / std::f64::consts::PI);
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_jupiter_barycenter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = jupiter_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mars_barycenter_position_de_over_dates(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mars_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mercury_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mercury_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_neptune_barycenter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = neptune_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_saturn_barycenter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = saturn_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_uranus_barycenter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = uranus_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_venus_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = venus_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[rstest]
    #[serial]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_solar_system_barycenter_position_de(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = solar_system_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
    }

    #[test]
    #[serial]
    fn test_sun_velocity_de() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        let v = sun_velocity_de(epc, NAIFKernel::DE440s).unwrap();
        // Geocentric solar velocity magnitude ~ 29-30.3 km/s
        assert!(v.norm() > 2.8e4 && v.norm() < 3.1e4);
    }

    #[test]
    #[serial]
    fn test_moon_state_de_consistent() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        let x = moon_state_de(epc, NAIFKernel::DE440s).unwrap();
        let r = moon_position_de(epc, NAIFKernel::DE440s).unwrap();
        let v = moon_velocity_de(epc, NAIFKernel::DE440s).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_mars_barycenter_position_de() {
        // Single-leg barycenter query works with de440s alone (no network).
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        let r = mars_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
        let expected =
            spk_position_from_kernel("de440s", NAIFId::MarsBarycenter, NAIFId::Earth, epc).unwrap();
        assert_abs_diff_eq!(r, expected, epsilon = 0.0);
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)] // downloads mar099s (~68 MB)
    #[serial]
    fn test_mars_position_de_body_center() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        let r_body = mars_position_de(epc, NAIFKernel::DE440s).unwrap();
        let r_bary = mars_barycenter_position_de(epc, NAIFKernel::DE440s).unwrap();
        // Mars body center differs from the Mars-system barycenter by < 1 km
        // (Phobos/Deimos are tiny) but must be nonzero.
        let dr = (r_body - r_bary).norm();
        assert!(dr > 0.0 && dr < 1.0e3, "|body - barycenter| = {} m", dr);
        // State/velocity variants agree with position/velocity decomposition.
        let x = mars_state_de(epc, NAIFKernel::DE440s).unwrap();
        assert_abs_diff_eq!(x.fixed_rows::<3>(0).into_owned(), r_body, epsilon = 1e-6);
        let v_body = mars_velocity_de(epc, NAIFKernel::DE440s).unwrap();
        assert_abs_diff_eq!(x.fixed_rows::<3>(3).into_owned(), v_body, epsilon = 1e-9);
        // Same two-leg decomposition holds for velocity: body and barycenter
        // velocities differ by a small but nonzero amount.
        let v_bary = mars_barycenter_velocity_de(epc, NAIFKernel::DE440s).unwrap();
        let dv = (v_body - v_bary).norm();
        assert!(dv > 0.0 && dv < 1.0, "|v_body - v_bary| = {} m/s", dv);
    }

    #[test]
    #[serial]
    fn test_body_center_de_two_leg_sum_offline() {
        // Exercises every `body_center_de_functions!`-generated function (all
        // five outer planets × position/velocity/state) fully offline. A
        // synthetic DE kernel provides each planetary-system barycenter rel
        // Earth; a synthetic satellite-system kernel per planet provides the
        // body center rel its barycenter. Positions are constant on the
        // x-axis, so each result is the exact two-leg sum and velocity is
        // zero. This also covers all `system_kernel` match arms.
        use crate::spice::{clear_kernels, load_kernel};
        use crate::utils::testing::{CacheRedirect, synthetic_spk_kernel_bytes};

        setup_global_test_spice();
        // Keep the real de440s resident so any concurrent cache-reading test
        // still finds it (via load_kernel's idempotent short-circuit) while
        // BRAHE_CACHE is redirected below. The barycenter leg uses the DE440
        // (de440.bsp) slot, which no other default-suite test loads, so
        // seeding a synthetic version there cannot corrupt a concurrent read.
        load_kernel("de440s").unwrap();

        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s(); // keep de440s valid for concurrent reloads
            cache.seed(
                "de440.bsp",
                &synthetic_spk_kernel_bytes(&[
                    (4, 399, 4.0),
                    (5, 399, 5.0),
                    (6, 399, 6.0),
                    (7, 399, 7.0),
                    (8, 399, 8.0),
                ]),
            );
            cache.seed("mar099s.bsp", &synthetic_spk_kernel_bytes(&[(499, 4, 0.4)]));
            cache.seed("jup365.bsp", &synthetic_spk_kernel_bytes(&[(599, 5, 0.5)]));
            cache.seed("sat441.bsp", &synthetic_spk_kernel_bytes(&[(699, 6, 0.6)]));
            cache.seed(
                "ura184_part-3.bsp",
                &synthetic_spk_kernel_bytes(&[(799, 7, 0.7)]),
            );
            cache.seed("nep097.bsp", &synthetic_spk_kernel_bytes(&[(899, 8, 0.8)]));

            // (position fn, velocity fn, state fn, expected x sum in meters)
            type PosFn = fn(Epoch, NAIFKernel) -> Result<Vector3<f64>, BraheError>;
            type StateFn = fn(Epoch, NAIFKernel) -> Result<Vector6<f64>, BraheError>;
            let cases: [(PosFn, PosFn, StateFn, f64); 5] = [
                (mars_position_de, mars_velocity_de, mars_state_de, 4400.0),
                (
                    jupiter_position_de,
                    jupiter_velocity_de,
                    jupiter_state_de,
                    5500.0,
                ),
                (
                    saturn_position_de,
                    saturn_velocity_de,
                    saturn_state_de,
                    6600.0,
                ),
                (
                    uranus_position_de,
                    uranus_velocity_de,
                    uranus_state_de,
                    7700.0,
                ),
                (
                    neptune_position_de,
                    neptune_velocity_de,
                    neptune_state_de,
                    8800.0,
                ),
            ];
            for (pos_fn, vel_fn, state_fn, expected_x) in cases {
                let r = pos_fn(epc, NAIFKernel::DE440).unwrap();
                assert_abs_diff_eq!(r[0], expected_x, epsilon = 1e-6);
                assert_eq!(vel_fn(epc, NAIFKernel::DE440).unwrap(), Vector3::zeros());
                let x = state_fn(epc, NAIFKernel::DE440).unwrap();
                assert_abs_diff_eq!(x[0], expected_x, epsilon = 1e-6);
                assert_eq!(x.fixed_rows::<3>(3).into_owned(), Vector3::zeros());
            }
        }
        // Redirect dropped (real cache restored): drop the synthetic kernels
        // and reload the real de440s.
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    fn test_system_kernel_panics_for_unsupported_body() {
        // The `_` arm of `system_kernel` panics for any body without a
        // satellite-system kernel.
        let result = std::panic::catch_unwind(|| system_kernel(NAIFId::Earth));
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_ssb_position_de_alias_offline() {
        // `ssb_position_de` forwards to `solar_system_barycenter_position_de`.
        // A synthetic DE kernel provides SSB (0) rel Earth as a constant.
        use crate::spice::{clear_kernels, load_kernel};
        use crate::utils::testing::{CacheRedirect, synthetic_spk_kernel_bytes};

        setup_global_test_spice();
        load_kernel("de440s").unwrap(); // keep real de440s resident (see above)

        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s(); // keep de440s valid for concurrent reloads
            // de440.bsp slot: not read by any concurrent default-suite test.
            cache.seed("de440.bsp", &synthetic_spk_kernel_bytes(&[(0, 399, 3.0)]));
            let r = ssb_position_de(epc, NAIFKernel::DE440).unwrap();
            assert_abs_diff_eq!(r[0], 3000.0, epsilon = 1e-6);
        }
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_solar_system_barycenter_position_de_error_offline() {
        // Error branch: a cached-but-unparseable DE kernel makes the load
        // fail, and the error propagates out of the generated position fn.
        use crate::spice::{clear_kernels, load_kernel};
        use crate::utils::testing::CacheRedirect;

        setup_global_test_spice();
        load_kernel("de440s").unwrap(); // keep real de440s resident (see above)

        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s(); // keep de440s valid for concurrent reloads
            // de440.bsp slot: seeding invalid bytes here cannot corrupt a
            // concurrent test reading the real de440s.
            cache.seed("de440.bsp", b"not a valid DAF kernel");
            let err = solar_system_barycenter_position_de(epc, NAIFKernel::DE440).unwrap_err();
            assert!(!format!("{}", err).is_empty());
        }
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_all_bodies_have_velocity_and_state() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        for (v, x) in [
            (
                sun_velocity_de(epc, NAIFKernel::DE440s),
                sun_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                moon_velocity_de(epc, NAIFKernel::DE440s),
                moon_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                mercury_velocity_de(epc, NAIFKernel::DE440s),
                mercury_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                venus_velocity_de(epc, NAIFKernel::DE440s),
                venus_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                mars_barycenter_velocity_de(epc, NAIFKernel::DE440s),
                mars_barycenter_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                jupiter_barycenter_velocity_de(epc, NAIFKernel::DE440s),
                jupiter_barycenter_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                saturn_barycenter_velocity_de(epc, NAIFKernel::DE440s),
                saturn_barycenter_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                uranus_barycenter_velocity_de(epc, NAIFKernel::DE440s),
                uranus_barycenter_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                neptune_barycenter_velocity_de(epc, NAIFKernel::DE440s),
                neptune_barycenter_state_de(epc, NAIFKernel::DE440s),
            ),
            (
                ssb_velocity_de(epc, NAIFKernel::DE440s),
                ssb_state_de(epc, NAIFKernel::DE440s),
            ),
        ] {
            assert!(v.unwrap().iter().all(|c| c.is_finite()));
            assert!(x.unwrap().iter().all(|c| c.is_finite()));
        }
    }
}
