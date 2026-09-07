//! Transform a state vector from TOD to ITRF and compare with the direct GCRF to ITRF path

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

    // Transform directly from TOD to ITRF
    let state_itrf_direct = bh::state_tod_to_itrf(epc, state_tod);

    println!("ITRF state vector (direct TOD -> ITRF):");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_itrf_direct[0], state_itrf_direct[1], state_itrf_direct[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_itrf_direct[3], state_itrf_direct[4], state_itrf_direct[5]);

    // Transform via GCRF: TOD -> GCRF -> ITRF
    let state_gcrf = bh::state_tod_to_gcrf(epc, state_tod);
    let state_itrf_via_gcrf = bh::state_gcrf_to_itrf(epc, state_gcrf);

    println!("ITRF state vector (via GCRF: TOD -> GCRF -> ITRF):");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_itrf_via_gcrf[0], state_itrf_via_gcrf[1], state_itrf_via_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_itrf_via_gcrf[3], state_itrf_via_gcrf[4], state_itrf_via_gcrf[5]);

    let pos_direct = na::Vector3::new(state_itrf_direct[0], state_itrf_direct[1], state_itrf_direct[2]);
    let pos_via_gcrf = na::Vector3::new(state_itrf_via_gcrf[0], state_itrf_via_gcrf[1], state_itrf_via_gcrf[2]);
    let vel_direct = na::Vector3::new(state_itrf_direct[3], state_itrf_direct[4], state_itrf_direct[5]);
    let vel_via_gcrf = na::Vector3::new(state_itrf_via_gcrf[3], state_itrf_via_gcrf[4], state_itrf_via_gcrf[5]);

    let pos_diff = (pos_direct - pos_via_gcrf).norm();
    let vel_diff = (vel_direct - vel_via_gcrf).norm();
    println!("Difference between the direct and GCRF-mediated paths:");
    println!("  Position: {:.6e} m", pos_diff);
    println!("  Velocity: {:.6e} m/s", vel_diff);
}
