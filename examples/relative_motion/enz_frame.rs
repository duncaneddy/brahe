//! Build the ENZ frame of a ground station, find a satellite's azimuth and elevation, show it as the SEZ permutation, compare the rate of a moving site to the permuted SEZ rate, and evaluate it through the frame graph.

#[allow(unused_imports)]
use brahe as bh;
use brahe::frames::{CelestialFrame, ReferenceFrame};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::{Vector3, Vector6};

/// A state provider returning a fixed state at every epoch, for registering
/// the ground station as an object.
struct ConstantProvider(Vector6<f64>);
impl SStateProvider for ConstantProvider {
    fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.0)
    }
}

fn main() {
    bh::initialize_eop().unwrap();

    let r_gs =
        bh::position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), bh::AngleFormat::Degrees)
            .unwrap();
    let x_gs = Vector6::new(r_gs[0], r_gs[1], r_gs[2], 0.0, 0.0, 0.0);

    let oe = Vector6::new(bh::R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 105.0);
    let x_eci = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let x_sat = bh::state_gcrf_to_itrf(epc, x_eci);

    let x_rel_enz = bh::state_ecef_to_enz(x_gs, x_sat);
    let azel = bh::position_enz_to_azel(
        x_rel_enz.fixed_rows::<3>(0).into_owned(),
        bh::AngleFormat::Degrees,
    );
    println!("Azimuth: {:.3} deg, Elevation: {:.3} deg", azel[0], azel[1]);

    let x_rel_sez = bh::state_ecef_to_sez(x_gs, x_sat);
    let is_sez_permutation = x_rel_enz[0] == x_rel_sez[1]
        && x_rel_enz[1] == -x_rel_sez[0]
        && x_rel_enz[2] == x_rel_sez[2];
    println!(
        "ENZ columns are the SEZ permutation (E=E, N=-S, Z=Z): {}",
        is_sez_permutation
    );

    let x_aircraft = Vector6::new(r_gs[0], r_gs[1], r_gs[2], -120.0, 180.0, 90.0);
    let omega_enz = bh::omega_enz(x_aircraft);
    let omega_sez = bh::omega_sez(x_aircraft);
    let omega_permuted = Vector3::new(omega_sez[1], -omega_sez[0], omega_sez[2]);
    println!(
        "Moving site omega_enz matches the permuted SEZ rate: {}",
        (omega_enz - omega_permuted).norm() < 1e-15
    );

    bh::register_object("GS", ConstantProvider(x_gs), CelestialFrame::ITRF).unwrap();
    let rel_graph =
        bh::state_frame_to_frame(CelestialFrame::ITRF, ReferenceFrame::ENZ("GS"), epc, x_sat)
            .unwrap();
    println!(
        "Frame graph matches state_ecef_to_enz: {}",
        (rel_graph - x_rel_enz).norm() < 1e-6
    );
    bh::clear_object_registry();
}
