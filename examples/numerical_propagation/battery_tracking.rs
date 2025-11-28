//! Battery state tracking using NumericalOrbitPropagator.
//! Demonstrates using additional_dynamics for battery charge with solar illumination.

use brahe as bh;
use bh::integrators::traits::DStateDynamics;
use bh::propagators::{DNumericalOrbitPropagator, ForceModelConfig, NumericalPropagationConfig};
use bh::time::Epoch;
use bh::traits::{DStatePropagator, InterpolatableTrajectory};
use nalgebra as na;
use std::sync::Arc;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = Epoch::from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Initial orbital elements and state - LEO orbit
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0);
    let orbital_state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Extended state: [x, y, z, vx, vy, vz, battery_charge]
    let battery_capacity = 100.0; // Wh
    let initial_charge = 80.0; // Wh (80% SOC)
    let mut initial_state = na::DVector::zeros(7);
    for i in 0..6 {
        initial_state[i] = orbital_state[i];
    }
    initial_state[6] = initial_charge;

    // Power system parameters
    let solar_panel_power = 50.0; // W (when fully illuminated)
    let load_power = 30.0; // W (continuous consumption)

    println!("Power system parameters:");
    println!("  Battery capacity: {} Wh", battery_capacity);
    println!(
        "  Initial charge: {} Wh ({:.0}% SOC)",
        initial_charge,
        100.0 * initial_charge / battery_capacity
    );
    println!("  Solar panel power: {} W", solar_panel_power);
    println!("  Load power: {} W", load_power);
    println!(
        "  Net charging rate (sunlit): {} W",
        solar_panel_power - load_power
    );
    println!("  Net discharge rate (eclipse): {} W", load_power);

    // Store epoch in Arc for closure
    let epoch_ref = Arc::new(epoch);
    let epoch_clone = Arc::clone(&epoch_ref);

    // Additional dynamics for battery tracking
    let additional_dynamics: DStateDynamics = Box::new(move |t, state, _params| {
        let mut dx = na::DVector::zeros(state.len());
        let r_eci = na::Vector3::new(state[0], state[1], state[2]);

        // Get sun position at current epoch
        let current_epoch = *epoch_clone + t;
        let r_sun = bh::sun_position(current_epoch);

        // Get illumination fraction (0 = umbra, 0-1 = penumbra, 1 = sunlit)
        let illumination = bh::eclipse_conical(r_eci, r_sun);

        // Battery dynamics (Wh/s = W / 3600)
        let power_in = illumination * solar_panel_power; // W
        let power_out = load_power; // W
        let mut charge_rate = (power_in - power_out) / 3600.0; // Wh/s

        // Apply battery limits (0 to capacity)
        let charge = state[6];
        if charge >= battery_capacity && charge_rate > 0.0 {
            charge_rate = 0.0; // Battery full
        } else if charge <= 0.0 && charge_rate < 0.0 {
            charge_rate = 0.0; // Battery empty
        }

        dx[6] = charge_rate;
        dx
    });

    // Create propagator with two-body dynamics
    let mut prop = DNumericalOrbitPropagator::new(
        epoch,
        initial_state.clone(),
        NumericalPropagationConfig::default(),
        ForceModelConfig::two_body_gravity(),
        None, // params
        Some(additional_dynamics),
        None, // control_input
        None, // initial_covariance
    )
    .unwrap();

    // Calculate orbital period and propagate for 3 orbits
    let orbital_period = bh::orbital_period(oe[0]);
    let num_orbits = 3;
    let total_time = num_orbits as f64 * orbital_period;

    println!(
        "\nOrbital period: {:.1} s ({:.1} min)",
        orbital_period,
        orbital_period / 60.0
    );
    println!(
        "Propagating for {} orbits ({:.1} min)",
        num_orbits,
        total_time / 60.0
    );

    // Propagate
    prop.propagate_to(epoch + total_time);

    // Check final state
    let final_state = prop.current_state();
    let final_charge = final_state[6];
    let charge_change = final_charge - initial_charge;

    println!("\nFinal battery state:");
    println!(
        "  Final charge: {:.2} Wh ({:.1}% SOC)",
        final_charge,
        100.0 * final_charge / battery_capacity
    );
    println!("  Charge change: {:+.2} Wh", charge_change);

    // Sample trajectory to find eclipse statistics
    let traj = prop.trajectory();
    let dt = 30.0; // 30 second samples
    let mut t = 0.0;
    let mut eclipse_time = 0.0;
    let mut sunlit_time = 0.0;

    while t <= total_time {
        let current_epoch = epoch + t;
        if let Ok(state) = traj.interpolate(&current_epoch) {
            let r_eci = na::Vector3::new(state[0], state[1], state[2]);
            let r_sun = bh::sun_position(current_epoch);
            let illumination = bh::eclipse_conical(r_eci, r_sun);

            if illumination < 0.01 {
                // In eclipse (< 1% illumination)
                eclipse_time += dt;
            } else {
                sunlit_time += dt;
            }
        }
        t += dt;
    }

    let eclipse_fraction = eclipse_time / (eclipse_time + sunlit_time);
    println!("\nEclipse statistics:");
    println!(
        "  Sunlit time: {:.1} min ({:.1}%)",
        sunlit_time / 60.0,
        100.0 * (1.0 - eclipse_fraction)
    );
    println!(
        "  Eclipse time: {:.1} min ({:.1}%)",
        eclipse_time / 60.0,
        100.0 * eclipse_fraction
    );

    // Validate
    assert!(final_charge > 0.0, "Battery should not be depleted");
    assert!(
        final_charge <= battery_capacity,
        "Battery should not exceed capacity"
    );
    assert!(eclipse_time > 0.0, "Should have some eclipse periods");
    assert!(sunlit_time > 0.0, "Should have some sunlit periods");

    println!("\nExample validated successfully!");
}
