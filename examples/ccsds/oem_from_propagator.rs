//! Generate an OEM from NumericalOrbitPropagator output — propagate a LEO orbit,
//! extract the trajectory, and build an OEM message.

use brahe as bh;
use bh::ccsds::{
    CCSDSFormat, CCSDSRefFrame, CCSDSTimeSystem, OEM, OEMMetadata, OEMSegment, OEMStateVector,
};
use bh::traits::{DStatePropagator, Trajectory};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Define initial state
    let epoch = bh::Epoch::from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 45.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create propagator with default force model
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Propagate for 90 minutes
    let target_epoch = epoch + 5400.0;
    prop.propagate_to(target_epoch);
    println!("Propagated from {} to {}", epoch, prop.current_epoch());

    // Get the accumulated trajectory
    let traj = prop.trajectory();
    println!(
        "Trajectory: {} states, span={:.0}s",
        traj.len(),
        traj.timespan().unwrap()
    );

    // Build an OEM from the trajectory states
    let mut oem = OEM::new("BRAHE_PROP".to_string());
    let stop_epoch = prop.current_epoch();

    let metadata = OEMMetadata::new(
        "LEO SAT".to_string(),
        "2024-100A".to_string(),
        "EARTH".to_string(),
        CCSDSRefFrame::EME2000,
        CCSDSTimeSystem::UTC,
        epoch,
        stop_epoch,
    )
    .with_interpolation("LAGRANGE".to_string(), Some(7));

    let mut seg = OEMSegment::new(metadata);

    // Extract states from trajectory and add to OEM
    for i in 0..traj.len() {
        let (epc, s) = traj.get(i).unwrap();
        seg.push_state(OEMStateVector::new(
            epc,
            [s[0], s[1], s[2]],
            [s[3], s[4], s[5]],
        ));
    }

    let num_states = seg.states.len();
    oem.push_segment(seg);
    println!("\nOEM: {} segment, {} states", oem.segments.len(), num_states);

    // Write to KVN
    let kvn = oem.to_string(CCSDSFormat::KVN).unwrap();
    println!("KVN output: {} characters", kvn.len());

    // Verify by re-parsing
    let oem2 = OEM::from_str(&kvn).unwrap();
    println!("Round-trip: {} states", oem2.segments[0].states.len());
}
