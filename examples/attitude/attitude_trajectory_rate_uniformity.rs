//! Build an AttitudeTrajectory and show the uniform rate-presence rule that add enforces.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::Quaternion;
use brahe::frames::{BodyFrame, CelestialFrame, ReferenceFrame};
use brahe::time::{Epoch, TimeSystem};
use brahe::traits::Trajectory;
use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
use nalgebra::Vector3;

fn main() {
    bh::initialize_eop().unwrap();

    // A trajectory relates two frame endpoints: every stored quaternion
    // rotates from frame_a into frame_b.
    let mut traj = AttitudeTrajectory::new(
        ReferenceFrame::from(CelestialFrame::GCRF),
        ReferenceFrame::from(BodyFrame::SCBody(Some("1".to_string()))),
    );

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    traj.add(epoch, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)))
        .unwrap();

    println!("frame_a:   {}", traj.frame_a);
    println!("frame_b:   {}", traj.frame_b);
    println!("States:    {}", traj.len());
    println!("has_rates: {}", traj.has_rates());

    // The first state carries no angular velocity, so every later state must
    // omit it as well. Adding a rate-carrying state to a rate-free trajectory
    // is rejected rather than producing a mixed trajectory.
    let with_rate = AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))
        .with_angular_velocity(Vector3::new(0.0, 0.0, 0.01));
    match traj.add(epoch + 60.0, with_rate) {
        Ok(()) => println!("\nMixed rate accepted"),
        Err(e) => println!("\nMixed rate rejected: {}", e),
    }
}
