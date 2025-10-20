//! Batch propagation of multiple satellites.
//!
//! Demonstrates:
//! - Getting ephemeris as propagators directly
//! - Propagating multiple satellites to same epoch
//! - Working with propagator collections
//!
//! FLAGS = [IGNORE]

use brahe::datasets::celestrak;
use brahe::eop::*;
use brahe::orbits::traits::StateProvider;
use brahe::time::*;

fn main() {
    // Initialize EOP (required for propagators)
    let eop = StaticEOPProvider::from_zero();
    set_global_eop_provider(eop);

    // Get ephemeris and initialize propagators directly
    let propagators = celestrak::get_ephemeris_as_propagators("gnss", 60.0).unwrap();

    println!("Loaded {} GNSS propagators", propagators.len());

    // Propagate all satellites to same epoch
    let epoch = Epoch::from_datetime(2021, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    let states: Vec<_> = propagators.iter().map(|prop| prop.state(epoch)).collect();

    println!("Propagated {} satellites to {}", states.len(), epoch);
    println!("First satellite position: [{:.3}, {:.3}, {:.3}] km",
        states[0][0] / 1000.0, states[0][1] / 1000.0, states[0][2] / 1000.0);
}
