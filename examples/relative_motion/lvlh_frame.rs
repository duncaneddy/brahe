//! Build the LVLH frame of a satellite, check it against RTN, and express a
//! deputy's relative state in it, both directly and through the frame graph.

#[allow(unused_imports)]
use brahe as bh;
use brahe::frames::{CelestialFrame, ReferenceFrame};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::Vector6;

/// A state provider returning a fixed state at every epoch, for registering
/// the chief as an object.
struct ConstantProvider(Vector6<f64>);
impl SStateProvider for ConstantProvider {
    fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.0)
    }
}

fn main() {
    bh::initialize_eop().unwrap();

    // Chief in a 700 km, e = 0.01, sun-synchronous orbit
    let oe_chief = Vector6::new(bh::R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0);
    let x_chief = bh::state_koe_to_eci(oe_chief, bh::AngleFormat::Degrees);

    // LVLH axes (CCSDS/SANA): Z nadir, Y anti-normal, X = Y x Z
    let r_lvlh_to_eci = bh::rotation_lvlh_to_eci(x_chief);
    let r_rtn_to_eci = bh::rotation_rtn_to_eci(x_chief);
    println!(
        "LVLH X equals RTN T: {}",
        (r_lvlh_to_eci.column(0) - r_rtn_to_eci.column(1)).norm() < 1e-12
    );
    println!(
        "LVLH Z equals -RTN R: {}",
        (r_lvlh_to_eci.column(2) + r_rtn_to_eci.column(0)).norm() < 1e-12
    );

    // Frame rate: the orbit rate about the -Y axis
    let omega = bh::omega_lvlh(x_chief);
    println!("omega_lvlh (rad/s): [{:.3e}, {:.3e}, {:.3e}]", omega[0], omega[1], omega[2]);

    // Deputy 1 km ahead and 200 m above the chief in the rotating LVLH frame
    let x_rel = Vector6::new(1000.0, 0.0, -200.0, 0.0, 0.0, 0.0);
    let x_deputy = bh::state_lvlh_to_eci(x_chief, x_rel);
    let x_rel_back = bh::state_eci_to_lvlh(x_chief, x_deputy);
    println!("Relative state round trip error: {:.2e}", (x_rel_back - x_rel).norm());

    // Frame graph: the same rotation from a registered object
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    bh::register_object("CHIEF", ConstantProvider(x_chief), CelestialFrame::GCRF).unwrap();
    let r_graph =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::LVLH("CHIEF"), epc)
            .unwrap();
    println!(
        "Frame graph matches rotation_eci_to_lvlh: {}",
        (r_graph - bh::rotation_eci_to_lvlh(x_chief)).norm() < 1e-12
    );
    bh::clear_object_registry();
}
