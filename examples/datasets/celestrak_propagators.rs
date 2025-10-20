//! FLAGS = [IGNORE]
//! Get CelesTrak ephemeris as propagators.
//!
//! Demonstrates:
//! - Getting active satellites as propagators

use brahe::datasets::celestrak;
use brahe::eop::*;
use brahe::orbits::traits::StateProvider;
use brahe::time::*;

fn main() {
    println!("Get Propagators from CelesTrak");
    println!("{}", "=".repeat(60));

    // Initialize EOP for propagators
    let eop = StaticEOPProvider::from_zero();
    set_global_eop_provider(eop);

    // Get ephemeris as propagators for active satellites
    let propagators = celestrak::get_ephemeris_as_propagators("active", 60.0).unwrap();
    println!("\nCreated {} propagators for active satellites", propagators.len());

    // Propagate first satellite
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    if !propagators.is_empty() {
        let state = propagators[0].state(epoch);
        println!("\nFirst satellite state at {}:", epoch);
        println!("  Position: [{:.3}, {:.3}, {:.3}] km",
            state[0]/1000.0, state[1]/1000.0, state[2]/1000.0);
        println!("  Velocity: [{:.3}, {:.3}, {:.3}] km/s",
            state[3]/1000.0, state[4]/1000.0, state[5]/1000.0);
    }

    println!("\n{}", "=".repeat(60));
}
