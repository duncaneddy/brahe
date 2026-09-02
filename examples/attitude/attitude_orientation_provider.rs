//! Query an AttitudeTrajectory through the OrientationProvider accessors.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::{EulerAngleOrder, Quaternion};
use brahe::frames::{BodyFrame, CelestialFrame, OrientationProvider, ReferenceFrame};
use brahe::time::{Epoch, TimeSystem};
use brahe::traits::Trajectory;
use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
use nalgebra::Vector3;

fn main() {
    bh::initialize_eop().unwrap();

    // Two samples 60 seconds apart, rotating about the spacecraft Z axis at
    // 0.01 rad/s. The rate is carried on both states, so angular_velocity is
    // available alongside the attitude.
    let mut traj = AttitudeTrajectory::new(
        ReferenceFrame::from(CelestialFrame::GCRF),
        ReferenceFrame::from(BodyFrame::SCBody(Some("1".to_string()))),
    );

    let omega = Vector3::new(0.0, 0.0, 0.01);
    let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let t1 = t0 + 60.0;
    let half_angle = 0.5 * omega[2] * 60.0;
    traj.add(
        t0,
        AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)).with_angular_velocity(omega),
    )
    .unwrap();
    traj.add(
        t1,
        AttitudeState::new(Quaternion::new(half_angle.cos(), 0.0, 0.0, half_angle.sin()))
            .with_angular_velocity(omega),
    )
    .unwrap();

    // Every accessor evaluates the same interpolated attitude and returns it
    // in a different representation.
    let epoch = t0 + 30.0;
    let quaternion = traj.quaternion(epoch).unwrap();
    let angles = traj.euler_angle(epoch, EulerAngleOrder::ZYX).unwrap();
    let axis = traj.euler_axis(epoch).unwrap();
    let matrix = traj.rotation_matrix(epoch).unwrap().to_matrix();
    let rate = traj.angular_velocity(epoch).unwrap().unwrap();

    let q = quaternion.to_vector(true);
    let (start, end) = traj.coverage().unwrap();
    println!("Coverage:    {} to {}", start, end);
    println!("Query epoch: {}", epoch);
    println!(
        "Quaternion:  s {:.6}, v {:.6} {:.6} {:.6}",
        q[0], q[1], q[2], q[3]
    );
    println!(
        "Euler ZYX:   {:.4}, {:.4}, {:.4} deg",
        angles.phi.to_degrees(),
        angles.theta.to_degrees(),
        angles.psi.to_degrees()
    );
    println!(
        "Euler axis:  {:.4} {:.4} {:.4}, angle {:.4} deg",
        axis.axis[0],
        axis.axis[1],
        axis.axis[2],
        axis.angle.to_degrees()
    );
    println!("Rotation matrix:");
    for row in matrix.row_iter() {
        println!("  {:9.6} {:9.6} {:9.6}", row[0], row[1], row[2]);
    }
    println!("Rate:        {:.4} {:.4} {:.4} rad/s", rate[0], rate[1], rate[2]);
}
