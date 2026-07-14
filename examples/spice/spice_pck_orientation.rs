//! Query lunar orientation from a binary PCK kernel.
//!
//! This example demonstrates pck_euler_angles and pck_rotation_matrix using the
//! moon_pa_de440 kernel, whose principal-axis lunar frame is registered under
//! NAIF frame class ID 31008 (MOON_PA_DE440). Downloads the PCK kernel on first
//! run.

#[allow(unused_imports)]
use brahe as bh;
use brahe::spice::FrameId;

fn main() {
    bh::initialize_eop().unwrap();

    // PCKs are never auto-initialized; they must be loaded explicitly.
    bh::spice::load_spice_kernel("moon_pa_de440").unwrap();

    let epc = bh::Epoch::from_date(2025, 1, 1, bh::TimeSystem::UTC);

    let (angles, rates) = bh::spice::pck_euler_angles(FrameId::MoonPaDe440, epc).unwrap();
    println!(
        "Euler angles [phi, delta, w] (rad): [{:.6}, {:.6}, {:.6}]",
        angles[0], angles[1], angles[2]
    );
    println!(
        "Euler angle rates (rad/s): [{:.3e}, {:.3e}, {:.3e}]",
        rates[0], rates[1], rates[2]
    );

    let r = bh::spice::pck_rotation_matrix(FrameId::MoonPaDe440, epc)
        .unwrap()
        .to_matrix();
    println!("\nICRF -> Moon principal-axis rotation matrix:");
    for row in 0..3 {
        println!(
            "  [{:.6}, {:.6}, {:.6}]",
            r[(row, 0)],
            r[(row, 1)],
            r[(row, 2)]
        );
    }
    let identity_error = (r * r.transpose() - nalgebra::Matrix3::identity()).norm();
    println!("Orthogonality check |R * R^T - I|: {:.3e}", identity_error);
}
