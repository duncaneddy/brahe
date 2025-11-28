//! Complete workflow: propagation, frame conversion, and analysis

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::traits::{Trajectory, SStatePropagator};
use bh::{KeplerianPropagator, orbital_period, position_ecef_to_geodetic, R_EARTH, AngleFormat};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // 1. Define orbit and create propagator
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 97.8_f64.to_radians(),
        15.0_f64.to_radians(), 30.0_f64.to_radians(), 0.0
    );

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let mut propagator = KeplerianPropagator::from_keplerian(
        epoch, oe, AngleFormat::Radians, 60.0
    );

    // 2. Propagate for one orbit period
    let period = orbital_period(R_EARTH + 500e3);
    let end_epoch = epoch + period;
    propagator.propagate_to(end_epoch);

    // 3. Get trajectory in ECI Cartesian
    let traj_eci = &propagator.trajectory;
    println!("Propagated {} states over {:.1} minutes",
        traj_eci.len(), traj_eci.timespan().unwrap() / 60.0);

    // 4. Convert to ECEF
    let traj_ecef = traj_eci.to_ecef();
    println!("\nGround track in ECEF frame:");
    for (i, (epoch, state_ecef)) in traj_ecef.into_iter().enumerate() {
        if i % 10 == 0 {
            let pos_ecef: na::Vector3<f64> = state_ecef.fixed_rows::<3>(0).into();
            let lla = position_ecef_to_geodetic(pos_ecef, AngleFormat::Degrees);
            println!("  {}: Lat={:6.2}°, Lon={:7.2}°, Alt={:6.2} km",
                epoch, lla[0], lla[1], lla[2] / 1e3);
        }
    }

    // 5. Convert to Keplerian
    let traj_kep = traj_eci.to_keplerian(AngleFormat::Radians);
    let first_oe = traj_kep.state_at_idx(0).unwrap();
    let last_oe = traj_kep.state_at_idx(traj_kep.len() - 1).unwrap();

    println!("\nOrbital element evolution:");
    println!("  Semi-major axis: {:.2} km → {:.2} km",
        first_oe[0] / 1e3, last_oe[0] / 1e3);
    println!("  Eccentricity: {:.6} → {:.6}",
        first_oe[1], last_oe[1]);
    println!("  Inclination: {:.2}° → {:.2}°",
        first_oe[2].to_degrees(), last_oe[2].to_degrees());
}
