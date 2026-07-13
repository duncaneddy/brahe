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

use super::kernels::SPICEKernel;
use super::naif_id::NAIFId;
use super::registry::{spk_position_from_kernel, spk_state_from_kernel, spk_velocity_from_kernel};

macro_rules! body_spice_functions {
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
        #[doc = concat!("use brahe::spice::{SPICEKernel, ", stringify!($pos_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let r = ", stringify!($pos_fn), "(epc, SPICEKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $pos_fn(epc: Epoch, kernel: SPICEKernel) -> Result<Vector3<f64>, BraheError> {
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
        #[doc = concat!("use brahe::spice::{SPICEKernel, ", stringify!($vel_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let v = ", stringify!($vel_fn), "(epc, SPICEKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $vel_fn(epc: Epoch, kernel: SPICEKernel) -> Result<Vector3<f64>, BraheError> {
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
        #[doc = concat!("use brahe::spice::{SPICEKernel, ", stringify!($state_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let x = ", stringify!($state_fn), "(epc, SPICEKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $state_fn(epc: Epoch, kernel: SPICEKernel) -> Result<Vector6<f64>, BraheError> {
            spk_state_from_kernel(kernel, $target, NAIFId::Earth, epc)
        }
    };
}

/// Satellite ephemeris kernel that carries a planet body center relative to
/// its planetary-system barycenter.
///
/// The DE kernels only provide the planetary-system *barycenters* (not the
/// planet body centers) for the outer planets, so recovering a true body
/// center requires the body-rel-barycenter leg from the planet's
/// satellite ephemeris kernel.
///
/// # Arguments
/// - `planet`: A planet body center ([`NAIFId::Mars`], [`NAIFId::Jupiter`],
///   [`NAIFId::Saturn`], [`NAIFId::Uranus`], or [`NAIFId::Neptune`])
///
/// # Returns
/// - The satellite ephemeris [`SPICEKernel`] providing that planet's body center
///
/// # Panics
/// Panics if `planet` is not in a planetary system with a supported
/// satellite ephemeris kernel.
pub(crate) const fn system_kernel(planet: NAIFId) -> SPICEKernel {
    match satellite_system_kernel(planet.id() / 100) {
        Some(kernel) => kernel,
        None => panic!("no satellite ephemeris kernel for this body"),
    }
}

/// Satellite ephemeris kernel for the planetary system whose barycenter has
/// NAIF ID `barycenter`, or `None` if the system has no supported kernel.
///
/// # Arguments
/// - `barycenter`: Planetary-system barycenter NAIF ID (`4`..=`9`)
///
/// # Returns
/// - The system's satellite ephemeris [`SPICEKernel`], or `None`
pub(crate) const fn satellite_system_kernel(barycenter: i32) -> Option<SPICEKernel> {
    match barycenter {
        4 => Some(SPICEKernel::Mar099s),
        5 => Some(SPICEKernel::Jup365),
        6 => Some(SPICEKernel::Sat441),
        7 => Some(SPICEKernel::Ura184),
        8 => Some(SPICEKernel::Nep097),
        9 => Some(SPICEKernel::Plu060),
        _ => None,
    }
}

/// True when `id` is carried directly by a planetary DE kernel: the solar
/// system barycenter (0), the planetary-system barycenters (1-9), the Sun
/// (10), and the bodies DE kernels chain to their system barycenter
/// (Mercury 199, Venus 299, Moon 301, Earth 399).
pub(crate) const fn de_kernel_covers(id: i32) -> bool {
    matches!(id, 0..=10 | 199 | 299 | 301 | 399)
}

/// True when a query for `id` can be resolved without consulting the global
/// registry's load order: either the DE kernel carries `id` directly, or
/// `id` belongs to a planetary system with a known satellite ephemeris
/// kernel (see [`satellite_system_kernel`]).
pub(crate) const fn spk_strictly_resolvable(id: i32) -> bool {
    de_kernel_covers(id) || (400 <= id && id <= 999 && satellite_system_kernel(id / 100).is_some())
}

/// Anchor `id` to a body the DE kernel carries: DE-covered IDs anchor to
/// themselves with a zero leg; satellite-system bodies anchor to their
/// system barycenter with the body-rel-barycenter leg queried from that
/// system's satellite ephemeris kernel (kernel-scoped, auto-loaded).
fn strict_anchor_and_leg(id: i32, epc: Epoch) -> Result<(i32, Vector3<f64>), BraheError> {
    if de_kernel_covers(id) {
        return Ok((id, Vector3::zeros()));
    }
    let barycenter = id / 100;
    let kernel = satellite_system_kernel(barycenter).ok_or_else(|| {
        BraheError::Error(format!(
            "NAIF ID {} is not covered by a DE kernel or a known satellite ephemeris kernel",
            id
        ))
    })?;
    Ok((
        barycenter,
        spk_position_from_kernel(kernel, id, barycenter, epc)?,
    ))
}

/// Position of `target` relative to `center` resolved strictly from
/// `de_kernel` plus (for satellite-system bodies) their system ephemeris
/// kernels — the configured kernel is honored regardless of which other
/// kernels are loaded in the global registry, unlike
/// [`spk_position`](crate::spice::spk_position)'s last-loaded-wins
/// precedence.
///
/// Both `target` and `center` must satisfy [`spk_strictly_resolvable`].
///
/// # Arguments
/// - `de_kernel`: DE kernel providing every leg between DE-covered bodies
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the position
///
/// # Returns
/// - `Ok(Vector3<f64>)`: Position of `target` relative to `center` in the
///   kernel's inertial frame (ICRF axes). Units: [m]
/// - `Err(BraheError)`: If either ID is not strictly resolvable or a kernel
///   cannot be loaded or queried
pub(crate) fn spk_pair_position_from_kernels(
    de_kernel: SPICEKernel,
    target: i32,
    center: i32,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let (target_anchor, target_leg) = strict_anchor_and_leg(target, epc)?;
    let (center_anchor, center_leg) = strict_anchor_and_leg(center, epc)?;
    let de_leg = if target_anchor == center_anchor {
        Vector3::zeros()
    } else {
        spk_position_from_kernel(de_kernel, target_anchor, center_anchor, epc)?
    };
    Ok(target_leg + de_leg - center_leg)
}

macro_rules! body_center_spice_functions {
    ($body_name:literal, $body:expr, $barycenter:expr, $kernel_name:literal, $kernel_size:literal,
     $pos_fn:ident, $vel_fn:ident, $state_fn:ident) => {
        #[doc = concat!("Calculate the position of ", $body_name, " (body center) relative to Earth using NAIF DE and satellite ephemeris kernels.")]
        ///
        /// Combines the planetary-system barycenter from the DE `kernel` with
        /// the body-center offset from the planet's satellite ephemeris kernel
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
        #[doc = concat!("use brahe::spice::{SPICEKernel, ", stringify!($pos_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let r = ", stringify!($pos_fn), "(epc, SPICEKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $pos_fn(epc: Epoch, kernel: SPICEKernel) -> Result<Vector3<f64>, BraheError> {
            let r_bary = spk_position_from_kernel(kernel, $barycenter, NAIFId::Earth, epc)?;
            let r_body = spk_position_from_kernel(system_kernel($body), $body, $barycenter, epc)?;
            Ok(r_bary + r_body)
        }

        #[doc = concat!("Calculate the velocity of ", $body_name, " (body center) relative to Earth using NAIF DE and satellite ephemeris kernels.")]
        ///
        /// Combines the planetary-system barycenter from the DE `kernel` with
        /// the body-center offset from the planet's satellite ephemeris kernel
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
        #[doc = concat!("use brahe::spice::{SPICEKernel, ", stringify!($vel_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let v = ", stringify!($vel_fn), "(epc, SPICEKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $vel_fn(epc: Epoch, kernel: SPICEKernel) -> Result<Vector3<f64>, BraheError> {
            let v_bary = spk_velocity_from_kernel(kernel, $barycenter, NAIFId::Earth, epc)?;
            let v_body = spk_velocity_from_kernel(system_kernel($body), $body, $barycenter, epc)?;
            Ok(v_bary + v_body)
        }

        #[doc = concat!("Calculate the state (position and velocity) of ", $body_name, " (body center) relative to Earth using NAIF DE and satellite ephemeris kernels.")]
        ///
        /// Combines the planetary-system barycenter from the DE `kernel` with
        /// the body-center offset from the planet's satellite ephemeris kernel
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
        #[doc = concat!("use brahe::spice::{SPICEKernel, ", stringify!($state_fn), "};")]
        /// use brahe::time::Epoch;
        /// use brahe::TimeSystem;
        ///
        /// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        #[doc = concat!("let x = ", stringify!($state_fn), "(epc, SPICEKernel::DE440s)?;")]
        /// # Ok::<(), brahe::utils::BraheError>(())
        /// ```
        pub fn $state_fn(epc: Epoch, kernel: SPICEKernel) -> Result<Vector6<f64>, BraheError> {
            let x_bary = spk_state_from_kernel(kernel, $barycenter, NAIFId::Earth, epc)?;
            let x_body = spk_state_from_kernel(system_kernel($body), $body, $barycenter, epc)?;
            Ok(x_bary + x_body)
        }
    };
}

body_spice_functions!(
    "the Sun",
    NAIFId::Sun,
    sun_position_spice,
    sun_velocity_spice,
    sun_state_spice
);
body_spice_functions!(
    "the Moon",
    NAIFId::Moon,
    moon_position_spice,
    moon_velocity_spice,
    moon_state_spice
);
body_spice_functions!(
    "Mercury",
    NAIFId::Mercury,
    mercury_position_spice,
    mercury_velocity_spice,
    mercury_state_spice
);
body_spice_functions!(
    "Venus",
    NAIFId::Venus,
    venus_position_spice,
    venus_velocity_spice,
    venus_state_spice
);
body_spice_functions!(
    "the Mars system barycenter",
    NAIFId::MarsBarycenter,
    mars_barycenter_position_spice,
    mars_barycenter_velocity_spice,
    mars_barycenter_state_spice
);
body_spice_functions!(
    "the Jupiter system barycenter",
    NAIFId::JupiterBarycenter,
    jupiter_barycenter_position_spice,
    jupiter_barycenter_velocity_spice,
    jupiter_barycenter_state_spice
);
body_spice_functions!(
    "the Saturn system barycenter",
    NAIFId::SaturnBarycenter,
    saturn_barycenter_position_spice,
    saturn_barycenter_velocity_spice,
    saturn_barycenter_state_spice
);
body_spice_functions!(
    "the Uranus system barycenter",
    NAIFId::UranusBarycenter,
    uranus_barycenter_position_spice,
    uranus_barycenter_velocity_spice,
    uranus_barycenter_state_spice
);
body_spice_functions!(
    "the Neptune system barycenter",
    NAIFId::NeptuneBarycenter,
    neptune_barycenter_position_spice,
    neptune_barycenter_velocity_spice,
    neptune_barycenter_state_spice
);
body_spice_functions!(
    "the Solar System Barycenter",
    NAIFId::SolarSystemBarycenter,
    solar_system_barycenter_position_spice,
    solar_system_barycenter_velocity_spice,
    solar_system_barycenter_state_spice
);

body_center_spice_functions!(
    "Mars",
    NAIFId::Mars,
    NAIFId::MarsBarycenter,
    "mar099s",
    "68 MB",
    mars_position_spice,
    mars_velocity_spice,
    mars_state_spice
);
body_center_spice_functions!(
    "Jupiter",
    NAIFId::Jupiter,
    NAIFId::JupiterBarycenter,
    "jup365",
    "1.1 GB",
    jupiter_position_spice,
    jupiter_velocity_spice,
    jupiter_state_spice
);
body_center_spice_functions!(
    "Saturn",
    NAIFId::Saturn,
    NAIFId::SaturnBarycenter,
    "sat441",
    "662 MB",
    saturn_position_spice,
    saturn_velocity_spice,
    saturn_state_spice
);
body_center_spice_functions!(
    "Uranus",
    NAIFId::Uranus,
    NAIFId::UranusBarycenter,
    "ura184",
    "387 MB",
    uranus_position_spice,
    uranus_velocity_spice,
    uranus_state_spice
);
body_center_spice_functions!(
    "Neptune",
    NAIFId::Neptune,
    NAIFId::NeptuneBarycenter,
    "nep097",
    "105 MB",
    neptune_position_spice,
    neptune_velocity_spice,
    neptune_state_spice
);

/// Calculate the position of the Solar System Barycenter in the GCRF frame using NAIF DE ephemeris.
///
/// Convenience alias for [`solar_system_barycenter_position_spice`].
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
/// use brahe::spice::{SPICEKernel, ssb_position_spice};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_ssb = ssb_position_spice(epc, SPICEKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_position_spice(epc: Epoch, kernel: SPICEKernel) -> Result<Vector3<f64>, BraheError> {
    solar_system_barycenter_position_spice(epc, kernel)
}

/// Calculate the velocity of the Solar System Barycenter in the GCRF frame using NAIF DE ephemeris.
///
/// Convenience alias for [`solar_system_barycenter_velocity_spice`].
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
/// use brahe::spice::{SPICEKernel, ssb_velocity_spice};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let v_ssb = ssb_velocity_spice(epc, SPICEKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_velocity_spice(epc: Epoch, kernel: SPICEKernel) -> Result<Vector3<f64>, BraheError> {
    solar_system_barycenter_velocity_spice(epc, kernel)
}

/// Calculate the state (position and velocity) of the Solar System Barycenter in the
/// GCRF frame using NAIF DE ephemeris.
///
/// Convenience alias for [`solar_system_barycenter_state_spice`].
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
/// use brahe::spice::{SPICEKernel, ssb_state_spice};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let x_ssb = ssb_state_spice(epc, SPICEKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn ssb_state_spice(epc: Epoch, kernel: SPICEKernel) -> Result<Vector6<f64>, BraheError> {
    solar_system_barycenter_state_spice(epc, kernel)
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
    fn test_sun_position_spice(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();

        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let r_analytical = sun_position(epc);
        let r_de = sun_position_spice(epc, SPICEKernel::DE440s).unwrap();

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
    fn test_moon_position_spice(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();

        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let r_analytical = moon_position(epc);
        let r_de = moon_position_spice(epc, SPICEKernel::DE440s).unwrap();

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
    fn test_jupiter_barycenter_position_spice(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = jupiter_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_mars_barycenter_position_spice_over_dates(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mars_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_mercury_position_spice(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mercury_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_neptune_barycenter_position_spice(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = neptune_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_saturn_barycenter_position_spice(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = saturn_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_uranus_barycenter_position_spice(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = uranus_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_venus_position_spice(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = venus_position_spice(epc, SPICEKernel::DE440s).unwrap();
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
    fn test_solar_system_barycenter_position_spice(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_spice();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = solar_system_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
    }

    #[test]
    #[serial]
    fn test_sun_velocity_spice() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        let v = sun_velocity_spice(epc, SPICEKernel::DE440s).unwrap();
        // Geocentric solar velocity magnitude ~ 29-30.3 km/s
        assert!(v.norm() > 2.8e4 && v.norm() < 3.1e4);
    }

    #[test]
    #[serial]
    fn test_moon_state_spice_consistent() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 6, 1, crate::time::TimeSystem::UTC);
        let x = moon_state_spice(epc, SPICEKernel::DE440s).unwrap();
        let r = moon_position_spice(epc, SPICEKernel::DE440s).unwrap();
        let v = moon_velocity_spice(epc, SPICEKernel::DE440s).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_mars_barycenter_position_spice() {
        // Single-leg barycenter query works with de440s alone (no network).
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        let r = mars_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
        let expected =
            spk_position_from_kernel("de440s", NAIFId::MarsBarycenter, NAIFId::Earth, epc).unwrap();
        assert_abs_diff_eq!(r, expected, epsilon = 0.0);
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)] // downloads mar099s (~68 MB)
    #[serial]
    fn test_mars_position_spice_body_center() {
        setup_global_test_spice();
        let epc = Epoch::from_date(2025, 1, 1, crate::time::TimeSystem::UTC);
        let r_body = mars_position_spice(epc, SPICEKernel::DE440s).unwrap();
        let r_bary = mars_barycenter_position_spice(epc, SPICEKernel::DE440s).unwrap();
        // Mars body center differs from the Mars-system barycenter by < 1 km
        // (Phobos/Deimos are tiny) but must be nonzero.
        let dr = (r_body - r_bary).norm();
        assert!(dr > 0.0 && dr < 1.0e3, "|body - barycenter| = {} m", dr);
        // State/velocity variants agree with position/velocity decomposition.
        let x = mars_state_spice(epc, SPICEKernel::DE440s).unwrap();
        assert_abs_diff_eq!(x.fixed_rows::<3>(0).into_owned(), r_body, epsilon = 1e-6);
        let v_body = mars_velocity_spice(epc, SPICEKernel::DE440s).unwrap();
        assert_abs_diff_eq!(x.fixed_rows::<3>(3).into_owned(), v_body, epsilon = 1e-9);
        // Same two-leg decomposition holds for velocity: body and barycenter
        // velocities differ by a small but nonzero amount.
        let v_bary = mars_barycenter_velocity_spice(epc, SPICEKernel::DE440s).unwrap();
        let dv = (v_body - v_bary).norm();
        assert!(dv > 0.0 && dv < 1.0, "|v_body - v_bary| = {} m/s", dv);
    }

    #[test]
    #[serial]
    fn test_body_center_spice_two_leg_sum_offline() {
        // Exercises every `body_center_spice_functions!`-generated function (all
        // five outer planets × position/velocity/state) fully offline. A
        // synthetic DE kernel provides each planetary-system barycenter rel
        // Earth; a synthetic satellite ephemeris kernel per planet provides the
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
            type PosFn = fn(Epoch, SPICEKernel) -> Result<Vector3<f64>, BraheError>;
            type StateFn = fn(Epoch, SPICEKernel) -> Result<Vector6<f64>, BraheError>;
            let cases: [(PosFn, PosFn, StateFn, f64); 5] = [
                (
                    mars_position_spice,
                    mars_velocity_spice,
                    mars_state_spice,
                    4400.0,
                ),
                (
                    jupiter_position_spice,
                    jupiter_velocity_spice,
                    jupiter_state_spice,
                    5500.0,
                ),
                (
                    saturn_position_spice,
                    saturn_velocity_spice,
                    saturn_state_spice,
                    6600.0,
                ),
                (
                    uranus_position_spice,
                    uranus_velocity_spice,
                    uranus_state_spice,
                    7700.0,
                ),
                (
                    neptune_position_spice,
                    neptune_velocity_spice,
                    neptune_state_spice,
                    8800.0,
                ),
            ];
            for (pos_fn, vel_fn, state_fn, expected_x) in cases {
                let r = pos_fn(epc, SPICEKernel::DE440).unwrap();
                assert_abs_diff_eq!(r[0], expected_x, epsilon = 1e-6);
                assert_eq!(vel_fn(epc, SPICEKernel::DE440).unwrap(), Vector3::zeros());
                let x = state_fn(epc, SPICEKernel::DE440).unwrap();
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
        // satellite ephemeris kernel.
        let result = std::panic::catch_unwind(|| system_kernel(NAIFId::Earth));
        assert!(result.is_err());
    }

    #[test]
    fn test_spk_pair_position_error_for_unmapped_id() {
        // `strict_anchor_and_leg` error branch: an ID that is neither
        // DE-covered nor in a satellite system with a known ephemeris
        // kernel errors before any kernel access (350 anchors to
        // barycenter 3, which has no satellite kernel; a self-assigned
        // negative ID has no system at all). Applies to both the target
        // and the center argument.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        for (target, center) in [(350, 399), (399, 350), (-20001, 399)] {
            let e = spk_pair_position_from_kernels(SPICEKernel::DE440s, target, center, epc);
            assert!(
                e.as_ref()
                    .unwrap_err()
                    .to_string()
                    .contains("not covered by a DE kernel"),
                "expected strict-resolution error for ({}, {}), got {:?}",
                target,
                center,
                e
            );
        }
    }

    #[test]
    #[serial]
    fn test_ssb_position_spice_alias_offline() {
        // `ssb_position_spice` forwards to `solar_system_barycenter_position_spice`.
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
            let r = ssb_position_spice(epc, SPICEKernel::DE440).unwrap();
            assert_abs_diff_eq!(r[0], 3000.0, epsilon = 1e-6);
        }
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_solar_system_barycenter_position_spice_error_offline() {
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
            let err = solar_system_barycenter_position_spice(epc, SPICEKernel::DE440).unwrap_err();
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
                sun_velocity_spice(epc, SPICEKernel::DE440s),
                sun_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                moon_velocity_spice(epc, SPICEKernel::DE440s),
                moon_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                mercury_velocity_spice(epc, SPICEKernel::DE440s),
                mercury_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                venus_velocity_spice(epc, SPICEKernel::DE440s),
                venus_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                mars_barycenter_velocity_spice(epc, SPICEKernel::DE440s),
                mars_barycenter_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                jupiter_barycenter_velocity_spice(epc, SPICEKernel::DE440s),
                jupiter_barycenter_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                saturn_barycenter_velocity_spice(epc, SPICEKernel::DE440s),
                saturn_barycenter_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                uranus_barycenter_velocity_spice(epc, SPICEKernel::DE440s),
                uranus_barycenter_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                neptune_barycenter_velocity_spice(epc, SPICEKernel::DE440s),
                neptune_barycenter_state_spice(epc, SPICEKernel::DE440s),
            ),
            (
                ssb_velocity_spice(epc, SPICEKernel::DE440s),
                ssb_state_spice(epc, SPICEKernel::DE440s),
            ),
        ] {
            assert!(v.unwrap().iter().all(|c| c.is_finite()));
            assert!(x.unwrap().iter().all(|c| c.is_finite()));
        }
    }
}
