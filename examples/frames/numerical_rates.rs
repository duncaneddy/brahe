//! Derive a body frame's angular velocity numerically from a rotation-only
//! callback with `with_numerical_rates`, and show the state transform that
//! requires it.

#[allow(unused_imports)]
use brahe as bh;
use brahe::frames::{CallbackOrientation, CelestialFrame, OrientationProvider, ReferenceFrame};
use brahe::math::SMatrix3;
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::Vector6;

/// A state provider returning a fixed state at every epoch, for registering
/// a spacecraft as an object.
struct ConstantProvider(Vector6<f64>);
impl SStateProvider for ConstantProvider {
    fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.0)
    }
}

fn main() {
    bh::clear_frame_registry();
    bh::clear_object_registry();

    let t0 = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
    let rate = 1.0e-3; // spin rate (rad/s)
    let rotation = move |epc: Epoch| {
        let dt = epc - t0;
        let (s, c) = (rate * dt).sin_cos();
        Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
    };

    // A rotation-only callback carries no angular velocity, so a state
    // transform through it fails: the velocity transport term is otherwise
    // undefined.
    bh::register_frame(
        ReferenceFrame::SC_BODY("SC"),
        CelestialFrame::GCRF.into(),
        CallbackOrientation::new(rotation, None),
    )
    .unwrap();
    bh::register_object(
        "SC",
        ConstantProvider(Vector6::zeros()),
        CelestialFrame::GCRF,
    )
    .unwrap();

    let epc = t0 + 100.0;
    let x_gcrf = Vector6::new(1.0e3, 2.0e3, 3.0e3, 0.0, 0.0, 0.0);
    let err = bh::state_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::SC_BODY("SC"), epc, x_gcrf)
        .unwrap_err();
    println!("Rates rule error: {}", err);

    // Re-registering with `with_numerical_rates` wraps the same callback so
    // a missing angular velocity is derived by central differencing the
    // rotation over +/- step/2 seconds; a provider that already returns
    // rates is used unchanged. The state transform then succeeds.
    bh::unregister_frame(&ReferenceFrame::SC_BODY("SC"));
    bh::register_frame(
        ReferenceFrame::SC_BODY("SC"),
        CelestialFrame::GCRF.into(),
        CallbackOrientation::new(rotation, None)
            .with_numerical_rates(1.0)
            .unwrap(),
    )
    .unwrap();
    let x_body =
        bh::state_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::SC_BODY("SC"), epc, x_gcrf).unwrap();
    println!(
        "\nBody-frame state with numerical rates: {:.6?}",
        x_body.as_slice()
    );

    // Compare against a hand-differenced velocity: with_numerical_rates
    // recovers the transport term from the same central-difference recipe.
    let delta = 0.5;
    let p = x_gcrf.fixed_rows::<3>(0).into_owned();
    let r_plus = rotation(epc + delta).unwrap() * p;
    let r_minus = rotation(epc - delta).unwrap() * p;
    let v_numerical = (r_plus - r_minus) / (2.0 * delta);
    let v_body = x_body.fixed_rows::<3>(3).into_owned();
    assert!((v_body - v_numerical).norm() < 1e-6);

    bh::clear_frame_registry();
    bh::clear_object_registry();
    println!("\nExample validated successfully!");
}
