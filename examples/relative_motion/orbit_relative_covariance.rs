//! Transform a covariance into and out of an orbit-relative frame from the frame's rotation and rate, and through the frame graph.

#[allow(unused_imports)]
use brahe as bh;
use brahe::frames::{CelestialFrame, ReferenceFrame};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::{DMatrix, SMatrix, Vector3, Vector6};

/// A state provider returning a fixed state at every epoch, for registering
/// the satellite as an object.
struct ConstantProvider(Vector6<f64>);
impl SStateProvider for ConstantProvider {
    fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.0)
    }
}

fn main() {
    bh::initialize_eop().unwrap();

    // Satellite in a 700 km, e = 0.01, sun-synchronous orbit
    let oe = Vector6::new(bh::R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0);
    let x_eci = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // ECI covariance: 100 m position and 0.1 m/s velocity standard deviations
    let p_eci = SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(
        1.0e4, 1.0e4, 1.0e4, 1.0e-2, 1.0e-2, 1.0e-2,
    ));

    // Rotating LVLH: Jacobian from the frame's rotation and angular velocity
    let r = bh::rotation_eci_to_lvlh(x_eci);
    let omega = bh::omega_lvlh(x_eci);
    let p_lvlh = bh::rotate_covariance_6(&p_eci, &bh::jacobian_inertial_to_rotating(&r, &omega));
    println!(
        "Rotating LVLH velocity sigmas (m/s): [{:.4}, {:.4}, {:.4}]",
        p_lvlh[(3, 3)].sqrt(),
        p_lvlh[(4, 4)].sqrt(),
        p_lvlh[(5, 5)].sqrt()
    );

    // Inertial snapshot: zero rate gives the block-diagonal Jacobian
    let p_snapshot =
        bh::rotate_covariance_6(&p_eci, &bh::jacobian_inertial_to_rotating(&r, &Vector3::zeros()));
    println!(
        "Snapshot LVLH velocity sigmas (m/s): [{:.4}, {:.4}, {:.4}]",
        p_snapshot[(3, 3)].sqrt(),
        p_snapshot[(4, 4)].sqrt(),
        p_snapshot[(5, 5)].sqrt()
    );

    // Back to ECI with the inverse Jacobian
    let p_back = bh::rotate_covariance_6(&p_lvlh, &bh::jacobian_rotating_to_inertial(&r, &omega));
    println!(
        "Round trip recovers the ECI covariance: {}",
        (p_back - p_eci).norm() / p_eci.norm() < 1e-12
    );

    // Frame graph: the same transform for a registered object
    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    bh::register_object("SC", ConstantProvider(x_eci), CelestialFrame::GCRF).unwrap();
    let p_graph = bh::covariance_frame_to_frame(
        CelestialFrame::GCRF,
        ReferenceFrame::LVLH("SC"),
        epc,
        &DMatrix::from_column_slice(6, 6, p_eci.as_slice()),
    )
    .unwrap();
    let p_graph = SMatrix::<f64, 6, 6>::from_column_slice(p_graph.as_slice());
    println!(
        "Frame graph matches the Jacobian route: {}",
        (p_graph - p_lvlh).norm() / p_lvlh.norm() < 1e-12
    );
    bh::clear_object_registry();
}
