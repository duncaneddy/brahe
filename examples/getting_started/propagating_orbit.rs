//! Propagate an orbit using SGP4 from TLE data.
//!
//! Demonstrates:
//! - Initializing EOP provider (required for frame transformations)
//! - Creating an SGP4 propagator from TLE
//! - Propagating to a specific epoch
//! - Working with state vectors

use brahe::eop::*;
use brahe::orbits::sgp_propagator::*;
use brahe::orbits::traits::StateProvider;
use brahe::time::*;

fn main() {
    // Initialize EOP provider (required for frame transformations)
    let eop = StaticEOPProvider::from_zero();
    set_global_eop_provider(eop);

    // Create an SGP4 propagator from Two-Line Element (TLE) data
    // ISS TLE from January 1, 2021
    let line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997";
    let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Propagate to a specific epoch
    let epc = Epoch::from_datetime(2021, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let state = prop.state(epc);

    println!("Position: [{:.3}, {:.3}, {:.3}] km",
        state[0] / 1000.0, state[1] / 1000.0, state[2] / 1000.0);
    println!("Velocity: [{:.3}, {:.3}, {:.3}] km/s",
        state[3] / 1000.0, state[4] / 1000.0, state[5] / 1000.0);
}
