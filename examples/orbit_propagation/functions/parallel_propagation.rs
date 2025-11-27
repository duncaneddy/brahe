//! Parallel propagation of multiple satellites
//!
//! This example demonstrates using par_propagate_to() to efficiently propagate
//! multiple satellites to a target epoch in parallel, useful for constellation
//! analysis and Monte Carlo simulations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::SStatePropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Create multiple propagators for a constellation
    let num_sats = 10;
    let mut propagators = Vec::new();

    for i in 0..num_sats {
        // Vary semi-major axis slightly for each satellite
        let a = bh::R_EARTH + 500e3 + (i as f64) * 10e3;
        let oe = na::SVector::<f64, 6>::new(
            a,
            0.001,
            98.0,
            (i as f64) * 36.0,
            0.0,
            (i as f64) * 36.0,
        );
        let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
        let prop = bh::KeplerianPropagator::from_eci(epoch, state, 60.0);
        propagators.push(prop);
    }

    // Target epoch: 24 hours later
    let target = epoch + 86400.0;

    // Propagate all satellites in parallel
    let start = std::time::Instant::now();
    bh::par_propagate_to(&mut propagators, target);
    let parallel_time = start.elapsed();

    println!(
        "Propagated {} satellites in parallel: {:.4} seconds",
        num_sats,
        parallel_time.as_secs_f64()
    );
    println!("\nFinal states:");
    for (i, prop) in propagators.iter().enumerate() {
        let state = prop.current_state();
        let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
        println!("  Satellite {}: r = {:.1} km", i, r / 1e3);
    }
}

// Output:
// Propagated 10 satellites in parallel: 0.0026 seconds

// Final states:
//   Satellite 0: r = 6876.8 km
//   Satellite 1: r = 6889.7 km
//   Satellite 2: r = 6902.3 km
//   Satellite 3: r = 6914.2 km
//   Satellite 4: r = 6925.0 km
//   Satellite 5: r = 6934.7 km
//   Satellite 6: r = 6943.1 km
//   Satellite 7: r = 6950.7 km
//   Satellite 8: r = 6957.8 km
//   Satellite 9: r = 6965.0 km