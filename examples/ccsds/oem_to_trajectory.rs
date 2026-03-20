//! Parse an OEM and convert segments to OrbitTrajectory objects for interpolation.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OEM;
use brahe::traits::Trajectory;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse an OEM file
    let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
    let seg = &oem.segments[0];
    println!(
        "Segment: {}, {} states, frame={}",
        seg.metadata.object_name,
        seg.states.len(),
        seg.metadata.ref_frame
    );
    // Expected output:
    // Segment: ISS, 49 states, frame=GCRF

    // Convert segment 0 to an SOrbitTrajectory
    let traj = oem.segment_to_orbit_trajectory(0).unwrap();
    println!("\nTrajectory: {} states", traj.len());
    println!("  Frame: {:?}", traj.frame);
    println!("  Start: {}", traj.start_epoch().unwrap());
    println!("  End:   {}", traj.end_epoch().unwrap());
    println!("  Span:  {:.0} seconds", traj.timespan().unwrap());

    // Access states by index
    let (epoch, state) = traj.first().unwrap();
    println!("\nFirst state:");
    println!("  Epoch: {}", epoch);
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] km",
        state[0] / 1e3,
        state[1] / 1e3,
        state[2] / 1e3
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        state[3], state[4], state[5]
    );

    // Convert all segments from a multi-segment OEM
    let oem_multi = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    let trajs = oem_multi.to_orbit_trajectories().unwrap();
    println!("\nMulti-segment OEM: {} trajectories", trajs.len());
    for (i, t) in trajs.iter().enumerate() {
        println!("  [{}] {} states, span={:.0}s", i, t.len(), t.timespan().unwrap());
    }
    // Expected output:
    // Multi-segment OEM: 3 trajectories
}
