//! Build the SEZ frame of a ground station, find a satellite's azimuth and elevation, compare the rate of a static and a moving site, and evaluate it through the frame graph.

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

    let oe = Vector6::new(bh::R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0);
    let x_eci = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let x_sat = bh::state_gcrf_to_itrf(epc, x_eci);

    let x_rel_sez = bh::state_ecef_to_sez(x_gs, x_sat);
    let azel = bh::position_sez_to_azel(
        x_rel_sez.fixed_rows::<3>(0).into_owned(),
        bh::AngleFormat::Degrees,
    );
    println!("Azimuth: {:.3} deg, Elevation: {:.3} deg", azel[0], azel[1]);

    let omega_static = bh::omega_sez(x_gs);
    println!(
        "Static station omega norm is zero: {}",
        omega_static.norm() == 0.0
    );

    let x_aircraft = Vector6::new(r_gs[0], r_gs[1], r_gs[2], -120.0, 180.0, 90.0);
    let omega_moving = bh::omega_sez(x_aircraft);
    println!("Moving site omega norm: {:.3e} rad/s", omega_moving.norm());

    bh::register_object("GS", ConstantProvider(x_gs), CelestialFrame::ITRF).unwrap();
    let rel_graph =
        bh::state_frame_to_frame(CelestialFrame::ITRF, ReferenceFrame::SEZ("GS"), epc, x_sat)
            .unwrap();
    println!(
        "Frame graph matches state_ecef_to_sez: {}",
        (rel_graph - x_rel_sez).norm() < 1e-6
    );
    bh::clear_object_registry();
}
