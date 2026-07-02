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

use super::kernels::SPKKernel;
use super::registry::{
    NAIF_EARTH, NAIF_JUPITER_BARYCENTER, NAIF_MARS_BARYCENTER, NAIF_MERCURY, NAIF_MOON,
    NAIF_NEPTUNE_BARYCENTER, NAIF_SATURN_BARYCENTER, NAIF_SSB, NAIF_SUN, NAIF_URANUS_BARYCENTER,
    NAIF_VENUS, spk_position_in_kernel, spk_state_in_kernel, spk_velocity_in_kernel,
};

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
        #[doc = concat!("use brahe::spice::{SPKKernel, ", stringify!($pos_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let r = ", stringify!($pos_fn), "(epc, SPKKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $pos_fn(epc: Epoch, kernel: SPKKernel) -> Result<Vector3<f64>, BraheError> {
            spk_position_in_kernel(kernel.name(), $target, NAIF_EARTH, epc)
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
        #[doc = concat!("use brahe::spice::{SPKKernel, ", stringify!($vel_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let v = ", stringify!($vel_fn), "(epc, SPKKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $vel_fn(epc: Epoch, kernel: SPKKernel) -> Result<Vector3<f64>, BraheError> {
            spk_velocity_in_kernel(kernel.name(), $target, NAIF_EARTH, epc)
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
        #[doc = concat!("use brahe::spice::{SPKKernel, ", stringify!($state_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let x = ", stringify!($state_fn), "(epc, SPKKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $state_fn(epc: Epoch, kernel: SPKKernel) -> Result<Vector6<f64>, BraheError> {
            spk_state_in_kernel(kernel.name(), $target, NAIF_EARTH, epc)
        }
    };
}

body_de_functions!(
    "the Sun",
    NAIF_SUN,
    sun_position_de,
    sun_velocity_de,
    sun_state_de
);
body_de_functions!(
    "the Moon",
    NAIF_MOON,
    moon_position_de,
    moon_velocity_de,
    moon_state_de
);
body_de_functions!(
    "Mercury",
    NAIF_MERCURY,
    mercury_position_de,
    mercury_velocity_de,
    mercury_state_de
);
body_de_functions!(
    "Venus",
    NAIF_VENUS,
    venus_position_de,
    venus_velocity_de,
    venus_state_de
);
body_de_functions!(
    "Mars (planetary-system barycenter)",
    NAIF_MARS_BARYCENTER,
    mars_position_de,
    mars_velocity_de,
    mars_state_de
);
body_de_functions!(
    "Jupiter (planetary-system barycenter)",
    NAIF_JUPITER_BARYCENTER,
    jupiter_position_de,
    jupiter_velocity_de,
    jupiter_state_de
);
body_de_functions!(
    "Saturn (planetary-system barycenter)",
    NAIF_SATURN_BARYCENTER,
    saturn_position_de,
    saturn_velocity_de,
    saturn_state_de
);
body_de_functions!(
    "Uranus (planetary-system barycenter)",
    NAIF_URANUS_BARYCENTER,
    uranus_position_de,
    uranus_velocity_de,
    uranus_state_de
);
body_de_functions!(
    "Neptune (planetary-system barycenter)",
    NAIF_NEPTUNE_BARYCENTER,
    neptune_position_de,
    neptune_velocity_de,
    neptune_state_de
);
body_de_functions!(
    "the Solar System Barycenter",
    NAIF_SSB,
    solar_system_barycenter_position_de,
    solar_system_barycenter_velocity_de,
    solar_system_barycenter_state_de
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
/// use brahe::spice::{SPKKernel, ssb_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_ssb = ssb_position_de(epc, SPKKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_position_de(epc: Epoch, kernel: SPKKernel) -> Result<Vector3<f64>, BraheError> {
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
/// use brahe::spice::{SPKKernel, ssb_velocity_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let v_ssb = ssb_velocity_de(epc, SPKKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_velocity_de(epc: Epoch, kernel: SPKKernel) -> Result<Vector3<f64>, BraheError> {
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
/// use brahe::spice::{SPKKernel, ssb_state_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let x_ssb = ssb_state_de(epc, SPKKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_state_de(epc: Epoch, kernel: SPKKernel) -> Result<Vector6<f64>, BraheError> {
    solar_system_barycenter_state_de(epc, kernel)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;
    use crate::orbit_dynamics::ephemerides::{moon_position, sun_position};
    use crate::utils::testing::setup_global_test_spice;

    #[rstest]
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
        let r_de = sun_position_de(epc, SPKKernel::DE440s).unwrap();

        let dot = r_analytical.dot(&r_de) / (r_analytical.norm() * r_de.norm());
        let angle = dot.acos() * (180.0 / std::f64::consts::PI);
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
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
        let r_de = moon_position_de(epc, SPKKernel::DE440s).unwrap();

        let dot = r_analytical.dot(&r_de) / (r_analytical.norm() * r_de.norm());
        let angle = dot.acos() * (180.0 / std::f64::consts::PI);
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_jupiter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = jupiter_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mars_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mars_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
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
        let _r = mercury_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_neptune_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = neptune_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_saturn_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = saturn_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_uranus_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = uranus_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
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
        let _r = venus_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[rstest]
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
        let _r = solar_system_barycenter_position_de(epc, SPKKernel::DE440s).unwrap();
    }

    #[test]
    fn test_sun_velocity_de() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        let v = sun_velocity_de(epc, SPKKernel::DE440s).unwrap();
        // Geocentric solar velocity magnitude ~ 29-30.3 km/s
        assert!(v.norm() > 2.8e4 && v.norm() < 3.1e4);
    }

    #[test]
    fn test_moon_state_de_consistent() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        let x = moon_state_de(epc, SPKKernel::DE440s).unwrap();
        let r = moon_position_de(epc, SPKKernel::DE440s).unwrap();
        let v = moon_velocity_de(epc, SPKKernel::DE440s).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_all_bodies_have_velocity_and_state() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        for (v, x) in [
            (
                sun_velocity_de(epc, SPKKernel::DE440s),
                sun_state_de(epc, SPKKernel::DE440s),
            ),
            (
                moon_velocity_de(epc, SPKKernel::DE440s),
                moon_state_de(epc, SPKKernel::DE440s),
            ),
            (
                mercury_velocity_de(epc, SPKKernel::DE440s),
                mercury_state_de(epc, SPKKernel::DE440s),
            ),
            (
                venus_velocity_de(epc, SPKKernel::DE440s),
                venus_state_de(epc, SPKKernel::DE440s),
            ),
            (
                mars_velocity_de(epc, SPKKernel::DE440s),
                mars_state_de(epc, SPKKernel::DE440s),
            ),
            (
                jupiter_velocity_de(epc, SPKKernel::DE440s),
                jupiter_state_de(epc, SPKKernel::DE440s),
            ),
            (
                saturn_velocity_de(epc, SPKKernel::DE440s),
                saturn_state_de(epc, SPKKernel::DE440s),
            ),
            (
                uranus_velocity_de(epc, SPKKernel::DE440s),
                uranus_state_de(epc, SPKKernel::DE440s),
            ),
            (
                neptune_velocity_de(epc, SPKKernel::DE440s),
                neptune_state_de(epc, SPKKernel::DE440s),
            ),
            (
                ssb_velocity_de(epc, SPKKernel::DE440s),
                ssb_state_de(epc, SPKKernel::DE440s),
            ),
        ] {
            assert!(v.unwrap().iter().all(|c| c.is_finite()));
            assert!(x.unwrap().iter().all(|c| c.is_finite()));
        }
    }
}
