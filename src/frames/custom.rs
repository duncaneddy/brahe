/*!
 * User-defined body-fixed frame registry.
 *
 * Lets applications plug an arbitrary orientation model into the frame
 * router as [`CelestialFrame::BodyFixedCustom`](super::CelestialFrame):
 * a rotation callback mapping an [`Epoch`] to the ICRF→body-fixed DCM,
 * optionally paired with an angular-velocity callback for the velocity
 * transport term. This supports orientation models the crate does not
 * ship — e.g. asteroid spin states from the DAMIT database — without any
 * change to the router API.
 *
 * When no angular-velocity callback is provided, the frame's angular
 * velocity is derived numerically from the rotation callback by central
 * differencing, via [`OrientationProviderExt::with_numerical_rates`], so a
 * rotation-only model still produces full state (position + velocity)
 * transforms.
 *
 * Frames are registered process-wide under a caller-chosen `u32` key,
 * following the crate's global-provider pattern (EOP, gravity, SPICE
 * registries), which keeps [`CelestialFrame`](super::CelestialFrame)
 * `Copy`/serializable — the enum stores only the key. Entries live in the
 * same global table as [`register_frame`](super::register_frame), under
 * the registry-internal `FrameKey::Custom` key variant.
 */

use std::sync::Arc;

use nalgebra::Vector3;

use crate::frames::orientation::{
    CallbackOrientation, OrientationProvider, OrientationProviderExt,
};
use crate::frames::registry::{FRAME_REGISTRY, FrameEntry, FrameKey};
use crate::math::SMatrix3;
use crate::time::Epoch;
use crate::utils::BraheError;

/// Rotation callback: ICRF→body-fixed DCM at an epoch.
pub type CustomFrameRotation = dyn Fn(Epoch) -> Result<SMatrix3, BraheError> + Send + Sync;

/// Angular-velocity callback: the frame's angular velocity at an epoch,
/// expressed in the body-fixed frame. Units: (rad/s)
pub type CustomFrameOmega = dyn Fn(Epoch) -> Result<Vector3<f64>, BraheError> + Send + Sync;

/// Registers (or replaces) a user-defined body-fixed frame under `key`.
///
/// The frame becomes usable as
/// [`CelestialFrame::BodyFixedCustom`](super::CelestialFrame) in every
/// router function. `rotation` must return the DCM rotating ICRF-axis
/// vectors into the body-fixed frame (`v_body = R * v_icrf`). If `omega`
/// is `None`, the angular velocity used for the velocity transport term
/// is derived numerically from `rotation` by central differencing.
///
/// # Arguments
/// - `key`: Caller-chosen identifier the frame is registered under; the
///   same value used in `CelestialFrame::BodyFixedCustom { key, .. }`
/// - `rotation`: Callback returning the ICRF→body-fixed DCM at an epoch
/// - `omega`: Optional callback returning the frame's angular velocity in
///   the body-fixed frame. Units: (rad/s)
///
/// # Examples
/// ```
/// use brahe::frames::{register_custom_frame, CelestialFrame, rotation_frame_to_frame};
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
///         Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
///     },
///     None,
/// );
///
/// let frame = CelestialFrame::BodyFixedCustom { center: -20001, key: 42 };
/// let r = rotation_frame_to_frame(CelestialFrame::GCRF, frame, t0 + 100.0).unwrap();
/// ```
pub fn register_custom_frame<R>(key: u32, rotation: R, omega: Option<Box<CustomFrameOmega>>)
where
    R: Fn(Epoch) -> Result<SMatrix3, BraheError> + Send + Sync + 'static,
{
    let provider: Arc<dyn OrientationProvider> = match omega {
        Some(omega) => Arc::new(CallbackOrientation::new(rotation, Some(omega))),
        None => Arc::new(CallbackOrientation::new(rotation, None).with_numerical_rates(1.0)),
    };
    FRAME_REGISTRY.write().unwrap().insert(
        FrameKey::Custom(key),
        FrameEntry {
            parent: None,
            provider,
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
    FRAME_REGISTRY
        .write()
        .unwrap()
        .remove(&FrameKey::Custom(key))
        .is_some()
}

/// Looks up the orientation provider for `key`, cloning the `Arc` out of
/// the lock.
fn provider(key: u32) -> Result<Arc<dyn OrientationProvider>, BraheError> {
    FRAME_REGISTRY
        .read()
        .unwrap()
        .get(&FrameKey::Custom(key))
        .map(|e| e.provider.clone())
        .ok_or_else(|| {
            BraheError::Error(format!(
                "No custom frame registered under key {} — call register_custom_frame first",
                key
            ))
        })
}

/// ICRF→body-fixed DCM of the custom frame `key` at `epc`.
pub(crate) fn custom_frame_rotation(key: u32, epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(provider(key)?.rotation_matrix(epc)?.to_matrix())
}

/// Rotation and body-frame angular velocity of the custom frame `key` at
/// `epc`. Uses the registered angular-velocity callback when present;
/// otherwise derives it from the rotation callback by central
/// differencing (`[omega]× = -Ṙ Rᵀ`, evaluated over ±0.5 s).
pub(crate) fn custom_frame_rotation_and_omega(
    key: u32,
    epc: Epoch,
) -> Result<(SMatrix3, Vector3<f64>), BraheError> {
    let provider = provider(key)?;
    let r = provider.rotation_matrix(epc)?.to_matrix();
    let w = provider.angular_velocity(epc)?.ok_or_else(|| {
        BraheError::Error(format!(
            "custom frame {} has no angular velocity data available",
            key
        ))
    })?;
    Ok((r, w))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::time::TimeSystem;

    /// Uniform rotation about z at `rate` rad/s from `t0`.
    fn spin_z(
        t0: Epoch,
        rate: f64,
    ) -> impl Fn(Epoch) -> Result<SMatrix3, BraheError> + Send + Sync + Clone {
        move |epc: Epoch| {
            let theta = rate * (epc - t0);
            let (s, c) = theta.sin_cos();
            Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
        }
    }

    #[test]
    #[serial]
    fn test_custom_frame_rotation_and_numeric_omega() {
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let rate = 1.0e-3;
        register_custom_frame(9001, spin_z(t0, rate), None);

        let epc = t0 + 250.0;
        let (r, w) = custom_frame_rotation_and_omega(9001, epc).unwrap();

        // Rotation matches the callback directly.
        let expected_r = spin_z(t0, rate)(epc).unwrap();
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
    #[serial]
    fn test_custom_frame_explicit_omega_used() {
        let t0 = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let rate = 2.0e-4;
        register_custom_frame(
            9002,
            spin_z(t0, rate),
            Some(Box::new(move |_| Ok(Vector3::new(0.0, 0.0, rate)))),
        );

        let (_, w) = custom_frame_rotation_and_omega(9002, t0 + 10.0).unwrap();
        assert_eq!(w, Vector3::new(0.0, 0.0, rate));

        assert!(unregister_custom_frame(9002));
    }

    #[test]
    #[serial]
    fn test_custom_frame_unregistered_key_errors() {
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let err = custom_frame_rotation(4_000_000_000, epc).unwrap_err();
        assert!(format!("{}", err).contains("No custom frame registered"));
        assert!(!unregister_custom_frame(4_000_000_000));
    }

    #[test]
    #[serial]
    fn test_custom_frame_rotation_and_omega_errors_without_rate_data() {
        // register_custom_frame always wraps a rate-less callback with
        // with_numerical_rates, so angular_velocity never actually returns
        // None through the public API. Insert a raw CallbackOrientation
        // (no rates, no numerical fallback) directly to exercise that
        // defensive error path in custom_frame_rotation_and_omega.
        use crate::frames::orientation::CallbackOrientation;

        let key = 9003;
        let provider: Arc<dyn OrientationProvider> = Arc::new(CallbackOrientation::new(
            |_epc: Epoch| Ok(SMatrix3::identity()),
            None,
        ));
        FRAME_REGISTRY.write().unwrap().insert(
            FrameKey::Custom(key),
            FrameEntry {
                parent: None,
                provider,
            },
        );

        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TDB);
        let err = custom_frame_rotation_and_omega(key, epc).unwrap_err();
        assert!(
            err.to_string()
                .contains("no angular velocity data available")
        );

        assert!(unregister_custom_frame(key));
    }
}
