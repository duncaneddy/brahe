//! Build the NTW frame on circular and eccentric orbits, compare it with RTN, and evaluate its rate about Earth and Mars.

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

    // Circular orbit: NTW and RTN coincide
    let x_circ = bh::state_koe_to_eci(
        Vector6::new(bh::R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0),
        bh::AngleFormat::Degrees,
    );
    println!(
        "Circular: NTW equals RTN: {}",
        (bh::rotation_ntw_to_eci(x_circ) - bh::rotation_rtn_to_eci(x_circ)).norm() < 1e-12
    );

    // Eccentric orbit: the NTW T axis is tilted from the RTN along-track axis
    // toward R by the flight-path angle
    let x_ecc = bh::state_koe_to_eci(
        Vector6::new(bh::R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0),
        bh::AngleFormat::Degrees,
    );
    let r_ntw = bh::rotation_ntw_to_eci(x_ecc);
    let r_rtn = bh::rotation_rtn_to_eci(x_ecc);
    let gamma = r_ntw.column(1).dot(&r_rtn.column(0)).asin().to_degrees();
    println!(
        "Eccentric: flight-path angle from the RTN along-track axis to the NTW T axis: {:.3} deg",
        gamma
    );

    // Rates: the velocity direction turns slower than the position direction near periapsis
    let omega_ntw = bh::omega_ntw(x_ecc);
    let omega_rtn = bh::omega_rtn(x_ecc);
    println!(
        "omega_ntw z: {:.6e} rad/s, omega_rtn z: {:.6e} rad/s",
        omega_ntw[2], omega_rtn[2]
    );

    // About Mars, pass the gravitational parameter explicitly
    let oe_mars = Vector6::new(bh::R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0);
    let x_mars = bh::state_koe_to_inertial_for_body(
        oe_mars,
        &bh::CentralBody::Mars,
        bh::AngleFormat::Degrees,
    )
    .unwrap();
    println!(
        "omega_ntw_for_body about Mars: {:.6e} rad/s",
        bh::omega_ntw_for_body(x_mars, bh::GM_MARS)[2]
    );

    // Relative state round trip
    let x_rel = Vector6::new(200.0, 1000.0, 0.0, 0.0, 0.0, 0.0);
    let x_deputy = bh::state_ntw_to_eci(x_ecc, x_rel);
    println!(
        "Round trip error: {:.2e}",
        (bh::state_eci_to_ntw(x_ecc, x_deputy) - x_rel).norm()
    );

    // Frame graph
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    bh::register_object("SC", ConstantProvider(x_ecc), CelestialFrame::GCRF).unwrap();
    let r_graph =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::NTW("SC"), epc)
            .unwrap();
    println!(
        "Frame graph matches rotation_eci_to_ntw: {}",
        (r_graph - bh::rotation_eci_to_ntw(x_ecc)).norm() < 1e-12
    );
    bh::clear_object_registry();
}
