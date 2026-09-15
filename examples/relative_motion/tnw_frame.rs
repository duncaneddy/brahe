//! Build the TNW frame on a circular orbit, compare it with RTN and NTW, and evaluate its rate about Earth and Mars.

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

    // Circular orbit: X = T, Z = N, and the in-plane axes are reversed from RTN
    let x_circ = bh::state_koe_to_eci(
        Vector6::new(bh::R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0),
        bh::AngleFormat::Degrees,
    );
    let r_tnw = bh::rotation_tnw_to_eci(x_circ);
    let r_rtn = bh::rotation_rtn_to_eci(x_circ);
    println!(
        "Circular: TNW Y equals -RTN X, TNW X equals RTN Y, TNW Z equals RTN Z: {}",
        (r_tnw.column(1) + r_rtn.column(0)).norm() < 1e-12
            && (r_tnw.column(0) - r_rtn.column(1)).norm() < 1e-12
            && (r_tnw.column(2) - r_rtn.column(2)).norm() < 1e-12
    );

    // TNW is NTW with its in-plane axes reordered
    let r_ntw = bh::rotation_ntw_to_eci(x_circ);
    println!(
        "TNW is NTW reordered: X_TNW equals Y_NTW, Y_TNW equals -X_NTW: {}",
        (r_tnw.column(0) - r_ntw.column(1)).norm() < 1e-12
            && (r_tnw.column(1) + r_ntw.column(0)).norm() < 1e-12
    );

    // Rates: TNW shares its rate with NTW
    let omega_tnw = bh::omega_tnw(x_circ);
    let omega_ntw = bh::omega_ntw(x_circ);
    println!(
        "omega_tnw z: {:.6e} rad/s, omega_ntw z: {:.6e} rad/s",
        omega_tnw[2], omega_ntw[2]
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
        "omega_tnw_for_body about Mars: {:.6e} rad/s",
        bh::omega_tnw_for_body(x_mars, bh::GM_MARS)[2]
    );

    // Relative state round trip: 1 km along-track, 200 m above along -N
    let x_rel = Vector6::new(1000.0, -200.0, 0.0, 0.0, 0.0, 0.0);
    let x_deputy = bh::state_tnw_to_eci(x_circ, x_rel);
    println!(
        "Round trip error: {:.2e}",
        (bh::state_eci_to_tnw(x_circ, x_deputy) - x_rel).norm()
    );

    // Frame graph
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    bh::register_object("SC", ConstantProvider(x_circ), CelestialFrame::GCRF).unwrap();
    let r_graph =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::TNW("SC"), epc)
            .unwrap();
    println!(
        "Frame graph matches rotation_eci_to_tnw: {}",
        (r_graph - bh::rotation_eci_to_tnw(x_circ)).norm() < 1e-12
    );
    bh::clear_object_registry();
}
