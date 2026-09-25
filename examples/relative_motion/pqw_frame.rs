//! Build the PQW frame on an eccentric orbit, check it against the true anomaly and the circular-orbit node-line fallback, verify the Mars `_for_body` periapsis direction, and evaluate it through the frame graph.

#[allow(unused_imports)]
use brahe as bh;
use brahe::frames::{CelestialFrame, ReferenceFrame};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::Vector6;

/// A state provider returning a fixed state at every epoch, for registering
/// the spacecraft as an object.
struct ConstantProvider(Vector6<f64>);
impl SStateProvider for ConstantProvider {
    fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.0)
    }
}

fn main() {
    bh::initialize_eop().unwrap();

    let oe = Vector6::new(bh::R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0);
    let x_ecc = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let r_pqw = bh::rotation_eci_to_pqw(x_ecc) * x_ecc.fixed_rows::<3>(0).into_owned();
    let r = x_ecc.fixed_rows::<3>(0).norm();
    let f = bh::anomaly_mean_to_true(45.0, 0.1, bh::AngleFormat::Degrees)
        .unwrap()
        .to_radians();
    let r_true_anomaly = r * nalgebra::Vector3::new(f.cos(), f.sin(), 0.0);
    println!(
        "Own position in PQW equals r [cos f, sin f, 0]: {}",
        (r_pqw - r_true_anomaly).norm() < 1e-6
    );

    // Circular orbit: periapsis is undefined, P falls back to the ascending node
    let oe_circ = Vector6::new(bh::R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0);
    let x_circ = bh::state_koe_to_eci(oe_circ, bh::AngleFormat::Degrees);
    let p_circ = bh::rotation_pqw_to_eci(x_circ).column(0).into_owned();
    let raan = 15.0_f64.to_radians();
    println!(
        "Circular fallback: P equals [cos Omega, sin Omega, 0]: {}",
        (p_circ - nalgebra::Vector3::new(raan.cos(), raan.sin(), 0.0)).norm() < 1e-6
    );

    // About Mars, the periapsis direction is the position at mean anomaly zero
    let oe_mars = Vector6::new(bh::R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0);
    let x_mars =
        bh::state_koe_to_inertial_for_body(oe_mars, &bh::CentralBody::Mars, bh::AngleFormat::Degrees)
            .unwrap();
    let mut oe_mars_periapsis = oe_mars;
    oe_mars_periapsis[5] = 0.0;
    let x_mars_periapsis = bh::state_koe_to_inertial_for_body(
        oe_mars_periapsis,
        &bh::CentralBody::Mars,
        bh::AngleFormat::Degrees,
    )
    .unwrap();
    let p_mars = bh::rotation_pqw_to_inertial_for_body(x_mars, bh::GM_MARS)
        .column(0)
        .into_owned();
    let r_periapsis = x_mars_periapsis.fixed_rows::<3>(0);
    let r_periapsis_hat = r_periapsis / r_periapsis.norm();
    println!(
        "Mars periapsis direction matches the position at M = 0: {}",
        (p_mars - r_periapsis_hat).norm() < 1e-6
    );

    // Frame graph
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    bh::register_object("SC", ConstantProvider(x_ecc), CelestialFrame::GCRF).unwrap();
    let r_graph =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::PQW("SC"), epc)
            .unwrap();
    println!(
        "Frame graph matches rotation_eci_to_pqw: {}",
        (r_graph - bh::rotation_eci_to_pqw(x_ecc)).norm() < 1e-12
    );
    bh::clear_object_registry();
}
