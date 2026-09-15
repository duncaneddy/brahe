//! Build the EQW frame on an inclined orbit, check it against the ascending node and the argument of latitude, verify the equatorial-orbit fallback, and evaluate it through the frame graph.

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
    let e_eqw = bh::rotation_eqw_to_eci(x_ecc).column(0).into_owned();
    let raan = 15.0_f64.to_radians();
    println!(
        "E equals [cos Omega, sin Omega, 0]: {}",
        (e_eqw - nalgebra::Vector3::new(raan.cos(), raan.sin(), 0.0)).norm() < 1e-6
    );

    let r_eqw = bh::rotation_eci_to_eqw(x_ecc) * x_ecc.fixed_rows::<3>(0).into_owned();
    let r = x_ecc.fixed_rows::<3>(0).norm();
    let f = bh::anomaly_mean_to_true(45.0, 0.1, bh::AngleFormat::Degrees)
        .unwrap()
        .to_radians();
    let u = 30.0_f64.to_radians() + f;
    let r_argument_of_latitude = r * nalgebra::Vector3::new(u.cos(), u.sin(), 0.0);
    println!(
        "Own position in EQW equals r [cos u, sin u, 0]: {}",
        (r_eqw - r_argument_of_latitude).norm() < 1e-6
    );

    // Equatorial orbit: the node is undefined, E falls back to the inertial x axis
    let oe_equatorial = Vector6::new(bh::R_EARTH + 700e3, 0.1, 0.0, 15.0, 30.0, 45.0);
    let x_equatorial = bh::state_koe_to_eci(oe_equatorial, bh::AngleFormat::Degrees);
    let e_equatorial = bh::rotation_eqw_to_eci(x_equatorial)
        .column(0)
        .into_owned();
    println!(
        "Equatorial fallback: E equals [1, 0, 0]: {}",
        (e_equatorial - nalgebra::Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-6
    );

    // Frame graph
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    bh::register_object("SC", ConstantProvider(x_ecc), CelestialFrame::GCRF).unwrap();
    let r_graph =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::EQW("SC"), epc)
            .unwrap();
    println!(
        "Frame graph matches rotation_eci_to_eqw: {}",
        (r_graph - bh::rotation_eci_to_eqw(x_ecc)).norm() < 1e-12
    );
    bh::clear_object_registry();
}
