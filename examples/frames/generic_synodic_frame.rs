//! Construct a generic two-body Synodic frame and transform a state into it.
//!
//! FLAGS = ["NETWORK"]

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();
    bh::load_common_spice_kernels().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // A generic Synodic frame isn't limited to the named EMR/SER/GSE
    // instances: any two SPK-covered bodies work as the primary/secondary
    // pair. Here the Sun-Mars barycenter defines a Sun-Mars rotating frame.
    let frame = bh::ReferenceFrame::Synodic {
        origin: bh::SynodicOrigin::Barycenter,
        primary: 10,
        secondary: 4,
    };

    // A LEO state in GCRF, transformed into the Sun-Mars rotating frame
    // through the frame router.
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let x_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let x_syn = bh::state_frame_to_frame(bh::ReferenceFrame::GCRF, frame, epc, x_gcrf).unwrap();

    println!("Epoch: {}", epc);
    println!(
        "GCRF state (km): [{:.3}, {:.3}, {:.3}]",
        x_gcrf[0] / 1e3,
        x_gcrf[1] / 1e3,
        x_gcrf[2] / 1e3
    );
    println!(
        "Sun-Mars synodic state (km): [{:.3}, {:.3}, {:.3}]",
        x_syn[0] / 1e3,
        x_syn[1] / 1e3,
        x_syn[2] / 1e3
    );

    // Round-tripping back to GCRF recovers the original state. The synodic
    // frame's origin is ~1e11 m from Earth, so the absolute tolerance is
    // scaled to the position magnitude rather than held at GCRF-local
    // precision.
    let x_back = bh::state_frame_to_frame(frame, bh::ReferenceFrame::GCRF, epc, x_syn).unwrap();
    assert!((x_back - x_gcrf).norm() < 1e-6 * x_gcrf.norm());

    println!("\nExample validated successfully!");
}
