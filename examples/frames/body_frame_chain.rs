//! Register a spacecraft body frame and a sensor frame mounted on it, then
//! route the Sun direction through the chain into the sensor frame.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::{EulerAxis, Quaternion, ToAttitude};
use brahe::constants::AngleFormat;
use brahe::frames::{BodyFrame, CelestialFrame, Frame};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::{SVector, Vector6};

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

    // A Frame::Body or Frame::OrbitRelative frame is a pure label until it
    // is bound to an object; a Frame::Celestial frame (CelestialFrame::GCRF,
    // ...) is always bound. The family constructors (Frame::RTN,
    // Frame::SC_BODY, Frame::CSS, ...) construct a bound frame directly; an
    // unbound label converts from a bare BodyFrame or OrbitRelativeFrame.
    let rtn = Frame::RTN("SC");
    let unbound: Frame = BodyFrame::SCBody(None).into();
    println!(
        "{}: bound={}, object={:?}",
        rtn,
        rtn.is_bound(),
        rtn.object()
    );
    println!(
        "{}: bound={}, object={:?}",
        unbound,
        unbound.is_bound(),
        unbound.object()
    );

    // Register "SC" as an object: any SStateProvider, in GCRF. An
    // OrbitTrajectory (via DStateAdapter), or an OEM's `register_for`
    // one-liner (see the CCSDS OEM docs), registers the same way.
    let oe = SVector::<f64, 6>::new(
        bh::constants::R_EARTH + 500e3,
        0.001,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let x_sc = bh::state_koe_to_eci(oe, AngleFormat::Degrees);
    bh::register_object("SC", ConstantProvider(x_sc), CelestialFrame::GCRF).unwrap();

    // Register SC's body frame and a coarse sun sensor mounted on it. A
    // constant attitude (Quaternion, RotationMatrix, EulerAngle, EulerAxis)
    // registers directly; orientation chains driven by an attitude ephemeris
    // ship in a later release.
    let q_body = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let q_css = EulerAxis::new(
        nalgebra::Vector3::new(0.0, 1.0, 0.0),
        0.7,
        AngleFormat::Radians,
    )
    .to_quaternion();
    bh::register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), q_body).unwrap();
    bh::register_frame(Frame::CSS("SC", "1"), Frame::SC_BODY("SC"), q_css).unwrap();

    // Route the Sun's GCRF position through GCRF -> SC_BODY -> CSS_1.
    let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
    let sun_gcrf = bh::sun_position(epc);
    let sun_css =
        bh::position_frame_to_frame(CelestialFrame::GCRF, Frame::CSS("SC", "1"), epc, sun_gcrf)
            .unwrap();
    println!("\nSun direction in CSS_1: {:.3?}", sun_css.as_slice());

    // Body frames share their object's origin exactly: routing SC's own
    // position into CSS_1 lands at the origin, with no lever arm applied.
    let sc_pos = x_sc.fixed_rows::<3>(0).into_owned();
    let sc_in_css =
        bh::position_frame_to_frame(CelestialFrame::GCRF, Frame::CSS("SC", "1"), epc, sc_pos)
            .unwrap();
    println!(
        "SC origin in CSS_1 (zero lever arm): {:.3?}",
        sc_in_css.as_slice()
    );
    assert!(sc_in_css.norm() < 1e-6);

    // Querying an unregistered link errors with a fix: which frame is
    // missing and the register_frame call that would supply it.
    let err =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, Frame::CSS("SC", "2"), epc).unwrap_err();
    println!("\nMissing-link error: {}", err);

    bh::clear_frame_registry();
    bh::clear_object_registry();
    println!("\nExample validated successfully!");
}
