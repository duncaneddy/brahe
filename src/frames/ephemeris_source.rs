/*!
 * Source of the Sun's state for frame-graph evaluations that need it.
 *
 * The NSW (nadir-Sun-normal) frame takes the Sun state as an explicit
 * argument to its `relative_motion` functions, and is unaffected by this
 * module. [`frame_sun_state`] is the source the frame graph consults when it
 * needs a Sun state on the caller's behalf.
 */

use std::fmt;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::frames::FrameCenter;
use crate::math::SVector6;
use crate::orbit_dynamics::ephemerides::sun_position;
use crate::spice::{NAIFId, loaded_spice_kernels, spk_state};
use crate::time::Epoch;
use crate::utils::BraheError;

/// Source of the Sun's state consulted by frame-graph evaluations that need
/// it (the NSW frame), selected by [`set_frame_ephemeris_source`].
///
/// - `Auto`: the SPICE registry when any kernel is already loaded, otherwise
///   the analytic model for an Earth-centered evaluation, otherwise the
///   registry, which auto-loads the default `de440s` ephemeris rather than
///   erroring.
/// - `Analytic`: the low-precision analytic Sun model (Montenbruck and
///   Gill), Earth-centered only; the SPICE registry is never consulted.
/// - `Kernel`: the SPICE registry, using whatever kernels are loaded
///   (auto-loading `de440s` if none are).
///
/// The `relative_motion` NSW functions take the Sun state as an argument
/// and are unaffected by this setting.
///
/// # Examples
/// ```
/// use brahe::frames::{
///     FrameEphemerisSource, get_frame_ephemeris_source, set_frame_ephemeris_source,
/// };
///
/// set_frame_ephemeris_source(FrameEphemerisSource::Analytic);
/// assert_eq!(get_frame_ephemeris_source(), FrameEphemerisSource::Analytic);
/// set_frame_ephemeris_source(FrameEphemerisSource::Auto);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameEphemerisSource {
    /// SPICE registry when a kernel is already loaded, otherwise the
    /// analytic model for Earth, otherwise the registry.
    Auto,
    /// Low-precision analytic Sun model, Earth-centered only.
    Analytic,
    /// SPICE registry, using whatever kernels are loaded.
    Kernel,
}

impl fmt::Display for FrameEphemerisSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameEphemerisSource::Auto => write!(f, "auto"),
            FrameEphemerisSource::Analytic => write!(f, "analytic"),
            FrameEphemerisSource::Kernel => write!(f, "kernel"),
        }
    }
}

static FRAME_EPHEMERIS_SOURCE: Lazy<RwLock<FrameEphemerisSource>> =
    Lazy::new(|| RwLock::new(FrameEphemerisSource::Auto));

/// Sets the global source the frame graph uses for the Sun's state.
///
/// # Arguments
/// - `source`: Frame ephemeris source to select
///
/// # Returns
/// - Nothing; the global setting is replaced
///
/// # Examples
/// ```
/// use brahe::frames::{FrameEphemerisSource, set_frame_ephemeris_source};
///
/// set_frame_ephemeris_source(FrameEphemerisSource::Kernel);
/// set_frame_ephemeris_source(FrameEphemerisSource::Auto);
/// ```
pub fn set_frame_ephemeris_source(source: FrameEphemerisSource) {
    *FRAME_EPHEMERIS_SOURCE.write().unwrap() = source;
}

/// Returns the global source the frame graph uses for the Sun's state.
///
/// # Returns
/// - `source`: Frame ephemeris source currently selected. `Auto` unless
///   [`set_frame_ephemeris_source`] has been called
///
/// # Examples
/// ```
/// use brahe::frames::{FrameEphemerisSource, get_frame_ephemeris_source};
///
/// assert_eq!(get_frame_ephemeris_source(), FrameEphemerisSource::Auto);
/// ```
pub fn get_frame_ephemeris_source() -> FrameEphemerisSource {
    *FRAME_EPHEMERIS_SOURCE.read().unwrap()
}

/// Sun state relative to `center` in ICRF axes, from the currently
/// selected [`FrameEphemerisSource`].
///
/// # Arguments
/// - `epc`: Evaluation epoch
/// - `center`: Center the Sun state is relative to
///
/// # Returns
/// - Sun position and velocity relative to `center`, in ICRF axes.
///   Units: (m, m/s)
pub(crate) fn frame_sun_state(epc: Epoch, center: FrameCenter) -> Result<SVector6, BraheError> {
    let is_earth = center == FrameCenter::Body(NAIFId::Earth);
    let use_analytic = match get_frame_ephemeris_source() {
        FrameEphemerisSource::Analytic => {
            if !is_earth {
                return Err(BraheError::Error(format!(
                    "FrameEphemerisSource::Analytic provides the Sun's state only for \
                     Earth-centered frames; center {} needs FrameEphemerisSource::Kernel \
                     or FrameEphemerisSource::Auto",
                    center.naif_id()
                )));
            }
            true
        }
        FrameEphemerisSource::Kernel => false,
        FrameEphemerisSource::Auto => loaded_spice_kernels().is_empty() && is_earth,
    };
    if use_analytic {
        let r = sun_position(epc);
        let v = (sun_position(epc + 0.5) - sun_position(epc - 0.5)) / 1.0;
        Ok(SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2]))
    } else {
        spk_state(NAIFId::Sun, center.naif_id(), epc)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::time::TimeSystem;
    use crate::utils::testing::{setup_global_test_spice, without_spice_kernels};

    #[test]
    #[serial]
    fn test_frame_ephemeris_source_default_and_setter() {
        set_frame_ephemeris_source(FrameEphemerisSource::Auto);
        assert_eq!(get_frame_ephemeris_source(), FrameEphemerisSource::Auto);
        set_frame_ephemeris_source(FrameEphemerisSource::Analytic);
        assert_eq!(get_frame_ephemeris_source(), FrameEphemerisSource::Analytic);
        assert_eq!(FrameEphemerisSource::Kernel.to_string(), "kernel");
        set_frame_ephemeris_source(FrameEphemerisSource::Auto);
    }

    #[test]
    #[serial]
    fn test_frame_sun_state_analytic_matches_sun_position() {
        set_frame_ephemeris_source(FrameEphemerisSource::Analytic);
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let x = frame_sun_state(epc, FrameCenter::Body(NAIFId::Earth)).unwrap();
        let r = sun_position(epc);
        assert_abs_diff_eq!(x.fixed_rows::<3>(0).into_owned(), r, epsilon = 1e-6);
        let v_fd = (sun_position(epc + 0.5) - sun_position(epc - 0.5)) / 1.0;
        assert_abs_diff_eq!(x.fixed_rows::<3>(3).into_owned(), v_fd, epsilon = 1e-9);
        // Speed is heliocentric-orbital in magnitude (about 30 km/s)
        assert!((x.fixed_rows::<3>(3).norm() - 29.8e3).abs() < 1.5e3);
        assert!(frame_sun_state(epc, FrameCenter::Body(NAIFId::Mars)).is_err());
        set_frame_ephemeris_source(FrameEphemerisSource::Auto);
    }

    #[test]
    #[serial]
    fn test_frame_sun_state_kernel_matches_spk_state() {
        setup_global_test_spice();
        set_frame_ephemeris_source(FrameEphemerisSource::Kernel);
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let x = frame_sun_state(epc, FrameCenter::Body(NAIFId::Earth)).unwrap();
        let expected = spk_state(NAIFId::Sun, NAIFId::Earth, epc).unwrap();
        assert_eq!(x, expected);
        set_frame_ephemeris_source(FrameEphemerisSource::Auto);
    }

    #[test]
    #[serial]
    fn test_frame_sun_state_auto_prefers_loaded_kernels() {
        set_frame_ephemeris_source(FrameEphemerisSource::Auto);
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        without_spice_kernels(|| {
            let x = frame_sun_state(epc, FrameCenter::Body(NAIFId::Earth)).unwrap();
            assert_abs_diff_eq!(
                x.fixed_rows::<3>(0).into_owned(),
                sun_position(epc),
                epsilon = 1e-6
            );
        });
        setup_global_test_spice();
        let x = frame_sun_state(epc, FrameCenter::Body(NAIFId::Earth)).unwrap();
        assert_eq!(x, spk_state(NAIFId::Sun, NAIFId::Earth, epc).unwrap());
        set_frame_ephemeris_source(FrameEphemerisSource::Auto);
    }
}
