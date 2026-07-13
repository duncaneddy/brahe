/*!
 * User-defined body-fixed frame registry.
 *
 * Lets applications plug an arbitrary orientation model into the frame
 * router as [`ReferenceFrame::BodyFixedCustom`](super::ReferenceFrame):
 * a rotation callback mapping an [`Epoch`] to the ICRF→body-fixed DCM,
 * optionally paired with an angular-velocity callback for the velocity
 * transport term. This supports orientation models the crate does not
 * ship — e.g. asteroid spin states from the DAMIT database — without any
 * change to the router API.
 *
 * When no angular-velocity callback is provided, the frame's angular
 * velocity is derived numerically from the rotation callback by central
 * differencing (`[omega]× = -Ṙ Rᵀ`), so a rotation-only model still
 * produces full state (position + velocity) transforms.
 *
 * Frames are registered process-wide under a caller-chosen `u32` key,
 * following the crate's global-provider pattern (EOP, gravity, SPICE
 * registries), which keeps [`ReferenceFrame`](super::ReferenceFrame)
 * `Copy`/serializable — the enum stores only the key.
 */

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use nalgebra::Vector3;
use once_cell::sync::Lazy;

use crate::math::SMatrix3;
use crate::time::Epoch;
use crate::utils::BraheError;

/// Rotation callback: ICRF→body-fixed DCM at an epoch.
pub type CustomFrameRotation = dyn Fn(Epoch) -> SMatrix3 + Send + Sync;

/// Angular-velocity callback: the frame's angular velocity at an epoch,
/// expressed in the body-fixed frame. Units: (rad/s)
pub type CustomFrameOmega = dyn Fn(Epoch) -> Vector3<f64> + Send + Sync;

struct CustomFrameEntry {
    rotation: Arc<CustomFrameRotation>,
    omega: Option<Arc<CustomFrameOmega>>,
}

static CUSTOM_FRAMES: Lazy<RwLock<HashMap<u32, CustomFrameEntry>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Step used by the central-difference angular-velocity fallback. One
/// second resolves every natural rotation period (the fastest-spinning
/// catalogued asteroids rotate in minutes) without losing precision to
/// f64 epoch arithmetic.
const OMEGA_DIFF_STEP: f64 = 1.0;

/// Registers (or replaces) a user-defined body-fixed frame under `key`.
///
/// The frame becomes usable as
/// [`ReferenceFrame::BodyFixedCustom`](super::ReferenceFrame) in every
/// router function. `rotation` must return the DCM rotating ICRF-axis
/// vectors into the body-fixed frame (`v_body = R * v_icrf`). If `omega`
/// is `None`, the angular velocity used for the velocity transport term
/// is derived numerically from `rotation` by central differencing.
///
/// # Arguments
/// - `key`: Caller-chosen identifier the frame is registered under; the
///   same value used in `ReferenceFrame::BodyFixedCustom { key, .. }`
/// - `rotation`: Callback returning the ICRF→body-fixed DCM at an epoch
/// - `omega`: Optional callback returning the frame's angular velocity in
///   the body-fixed frame. Units: (rad/s)
///
/// # Examples
/// ```
/// use brahe::frames::{register_custom_frame, ReferenceFrame, rotation_frame_to_frame};
/// use brahe::math::SMatrix3;
/// use brahe::time::{Epoch, TimeSystem};
///
/// // A body spinning uniformly about the ICRF z-axis (0.001 rad/s).
/// let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
/// register_custom_frame(
///     42,
///     move |epc: Epoch| {
///         let theta = 0.001 * (epc - t0);
///         let (s, c) = theta.sin_cos();
///         SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
///     },
///     None,
/// );
///
/// let frame = ReferenceFrame::BodyFixedCustom { center: -20001, key: 42 };
/// let r = rotation_frame_to_frame(ReferenceFrame::GCRF, frame, t0 + 100.0).unwrap();
/// ```
pub fn register_custom_frame<R>(key: u32, rotation: R, omega: Option<Box<CustomFrameOmega>>)
where
    R: Fn(Epoch) -> SMatrix3 + Send + Sync + 'static,
{
    CUSTOM_FRAMES.write().unwrap().insert(
        key,
        CustomFrameEntry {
            rotation: Arc::new(rotation),
            omega: omega.map(Arc::from),
        },
    );
}

/// Removes the custom frame registered under `key`.
///
/// # Arguments
/// - `key`: Identifier the frame was registered under
///
/// # Returns
/// - `true` if a frame was registered under `key` and has been removed
pub fn unregister_custom_frame(key: u32) -> bool {
    CUSTOM_FRAMES.write().unwrap().remove(&key).is_some()
}

/// Looks up the callbacks for `key`, cloning the `Arc`s out of the lock.
fn entry(
    key: u32,
) -> Result<(Arc<CustomFrameRotation>, Option<Arc<CustomFrameOmega>>), BraheError> {
    let map = CUSTOM_FRAMES.read().unwrap();
    map.get(&key)
        .map(|e| (e.rotation.clone(), e.omega.clone()))
        .ok_or_else(|| {
            BraheError::Error(format!(
                "No custom frame registered under key {} — call register_custom_frame first",
                key
            ))
        })
}

/// ICRF→body-fixed DCM of the custom frame `key` at `epc`.
pub(crate) fn custom_frame_rotation(key: u32, epc: Epoch) -> Result<SMatrix3, BraheError> {
    let (rotation, _) = entry(key)?;
    Ok(rotation(epc))
}

/// Rotation and body-frame angular velocity of the custom frame `key` at
/// `epc`. Uses the registered angular-velocity callback when present;
/// otherwise derives it from the rotation callback by central
/// differencing (`[omega]× = -Ṙ Rᵀ`, evaluated over ±[`OMEGA_DIFF_STEP`]).
pub(crate) fn custom_frame_rotation_and_omega(
    key: u32,
    epc: Epoch,
) -> Result<(SMatrix3, Vector3<f64>), BraheError> {
    let (rotation, omega) = entry(key)?;
    let r = rotation(epc);
    let w = match omega {
        Some(omega) => omega(epc),
        None => {
            let r_plus = rotation(epc + OMEGA_DIFF_STEP);
            let r_minus = rotation(epc - OMEGA_DIFF_STEP);
            let r_dot = (r_plus - r_minus) / (2.0 * OMEGA_DIFF_STEP);
            // [omega]× = -Ṙ Rᵀ for r_b = R r_i; extract the vector from the
            // skew-symmetric part.
            let s = -r_dot * r.transpose();
            Vector3::new(
                0.5 * (s[(2, 1)] - s[(1, 2)]),
                0.5 * (s[(0, 2)] - s[(2, 0)]),
                0.5 * (s[(1, 0)] - s[(0, 1)]),
            )
        }
    };
    Ok((r, w))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::time::TimeSystem;

    /// Uniform rotation about z at `rate` rad/s from `t0`.
    fn spin_z(t0: Epoch, rate: f64) -> impl Fn(Epoch) -> SMatrix3 + Send + Sync + Clone {
        move |epc: Epoch| {
            let theta = rate * (epc - t0);
            let (s, c) = theta.sin_cos();
            SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
        }
    }

    #[test]
    fn test_custom_frame_rotation_and_numeric_omega() {
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let rate = 1.0e-3;
        register_custom_frame(9001, spin_z(t0, rate), None);

        let epc = t0 + 250.0;
        let (r, w) = custom_frame_rotation_and_omega(9001, epc).unwrap();

        // Rotation matches the callback directly.
        let expected_r = spin_z(t0, rate)(epc);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r[(i, j)], expected_r[(i, j)], epsilon = 1e-15);
            }
        }

        // Numeric omega recovers the spin vector (z-axis, `rate` rad/s).
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w[2], rate, epsilon = 1e-9);

        assert!(unregister_custom_frame(9001));
    }

    #[test]
    fn test_custom_frame_explicit_omega_used() {
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let rate = 2.0e-4;
        register_custom_frame(
            9002,
            spin_z(t0, rate),
            Some(Box::new(move |_| Vector3::new(0.0, 0.0, rate))),
        );

        let (_, w) = custom_frame_rotation_and_omega(9002, t0 + 10.0).unwrap();
        assert_eq!(w, Vector3::new(0.0, 0.0, rate));

        assert!(unregister_custom_frame(9002));
    }

    #[test]
    fn test_custom_frame_unregistered_key_errors() {
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let err = custom_frame_rotation(4_000_000_000, epc).unwrap_err();
        assert!(format!("{}", err).contains("No custom frame registered"));
        assert!(!unregister_custom_frame(4_000_000_000));
    }
}
