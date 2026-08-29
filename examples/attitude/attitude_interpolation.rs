//! Build an AttitudeTrajectory in code and compare Slerp and Linear interpolation.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::{AttitudeFrame, Quaternion, SpacecraftFrame, ToAttitude};
use brahe::frames::ReferenceFrame;
use brahe::time::{Epoch, TimeSystem};
use brahe::traits::{AttitudeProvider, Trajectory};
use brahe::trajectories::{AttitudeInterpolationMethod, AttitudeState, AttitudeTrajectory};

fn main() {
    bh::initialize_eop().unwrap();

    // Two attitude samples 60 seconds apart: a constant-rate rotation of
    // 2 deg/s about the spacecraft Z axis, from 0 to 120 degrees.
    let mut traj = AttitudeTrajectory::new(
        AttitudeFrame::Reference(ReferenceFrame::GCRF),
        AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(None)),
    );

    let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let t1 = t0 + 60.0;
    let q0 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let half_angle = 60.0_f64.to_radians();
    let q1 = Quaternion::new(half_angle.cos(), 0.0, 0.0, half_angle.sin());
    traj.add(t0, AttitudeState::new(q0)).unwrap();
    traj.add(t1, AttitudeState::new(q1)).unwrap();

    // Query one third of the way between the two nodes. At the exact
    // midpoint, linear interpolation of a single-axis rotation happens to
    // coincide with slerp, so an off-center query is needed to see them
    // diverge.
    let query = t0 + 20.0;

    traj.set_interpolation_method(AttitudeInterpolationMethod::Slerp);
    let slerp_q = traj.quaternion(query).unwrap();

    traj.set_interpolation_method(AttitudeInterpolationMethod::Linear);
    let linear_q = traj.quaternion(query).unwrap();

    let slerp_angle_deg = slerp_q.to_euler_axis().angle.to_degrees();
    let linear_angle_deg = linear_q.to_euler_axis().angle.to_degrees();

    println!("Query epoch: {} (1/3 of the way from t0 to t1)", query);
    println!("Slerp  rotation angle:  {:.4} deg (exact)", slerp_angle_deg);
    println!(
        "Linear rotation angle:  {:.4} deg (approximate)",
        linear_angle_deg
    );
}
