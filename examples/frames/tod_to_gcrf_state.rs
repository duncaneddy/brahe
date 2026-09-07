//! Transform a state vector from the true equator and equinox of date (TOD) to GCRF

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    // Hard-coded TOD state vector
    let state_tod = na::SVector::<f64, 6>::new(6878137.0, 0.0, 0.0, 0.0, 7612.0, 0.0);

    println!("\nTOD state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_tod[0], state_tod[1], state_tod[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_tod[3], state_tod[4], state_tod[5]);

    // Transform to GCRF at the given epoch
    let state_gcrf = bh::state_tod_to_gcrf(epc, state_tod);

    println!("GCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);

    // Round trip back to TOD
    let state_tod_roundtrip = bh::state_gcrf_to_tod(epc, state_gcrf);

    let pos_tod = na::Vector3::new(state_tod[0], state_tod[1], state_tod[2]);
    let pos_roundtrip = na::Vector3::new(state_tod_roundtrip[0], state_tod_roundtrip[1], state_tod_roundtrip[2]);
    let vel_tod = na::Vector3::new(state_tod[3], state_tod[4], state_tod[5]);
    let vel_roundtrip = na::Vector3::new(state_tod_roundtrip[3], state_tod_roundtrip[4], state_tod_roundtrip[5]);

    let pos_err = (pos_tod - pos_roundtrip).norm();
    let vel_err = (vel_tod - vel_roundtrip).norm();
    println!("Round-trip error (TOD -> GCRF -> TOD):");
    println!("  Position: {:.6e} m", pos_err);
    println!("  Velocity: {:.6e} m/s", vel_err);
}
