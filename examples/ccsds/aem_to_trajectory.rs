//! Parse an AEM and convert a segment to an AttitudeTrajectory for interpolation.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::AEM;
use brahe::traits::{AttitudeProvider, Trajectory};

fn main() {
    bh::initialize_eop().unwrap();

    // Parse an AEM with two quaternion segments
    let aem = AEM::from_file("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();

    // Segment 1 carries no INTERPOLATION_METHOD, so it converts cleanly to
    // the default slerp trajectory. Segment 0 sets INTERPOLATION_METHOD =
    // HERMITE, which has no AttitudeTrajectory equivalent and would error.
    let traj = aem.segment_to_attitude_trajectory(1).unwrap();
    println!("Trajectory: {} states", traj.len());
    println!("  Frame A:       {}", traj.frame_a);
    println!("  Frame B:       {}", traj.frame_b);
    println!("  Interpolation: {:?}", traj.interpolation_method);
    println!("  Has rates:     {}", traj.has_rates());
    println!("  Start:         {}", traj.start_epoch().unwrap());
    println!("  End:           {}", traj.end_epoch().unwrap());

    // Slerp-query the attitude at the midpoint of the trajectory's span
    let t0 = traj.start_epoch().unwrap();
    let t1 = traj.end_epoch().unwrap();
    let mid = t0 + (t1 - t0) / 2.0;
    let quaternion = traj.quaternion(mid).unwrap();
    let wire = quaternion.to_vector(false);
    println!(
        "\nInterpolated quaternion [Q1, Q2, Q3, QC] at {}: [{:.5}, {:.5}, {:.5}, {:.5}]",
        mid, wire[0], wire[1], wire[2], wire[3]
    );
}
