//! Convert CelesTrak GP data directly to SGP propagator.
//!
//! This example shows how to query a satellite from CelesTrak and convert it
//! to a propagator in a few steps, which is the most common use case.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::celestrak::{CelestrakClient, CelestrakQuery};
use bh::propagators::SGPPropagator;
use bh::traits::SStatePropagator;
use bh::utils::Identifiable;

fn main() {
    bh::initialize_eop().unwrap();

    // Query ISS GP data from CelesTrak and create a propagator with 60-second step size
    let client = CelestrakClient::new();
    let query = CelestrakQuery::gp().catnr(25544);
    let records = client.query_gp(&query).unwrap();
    let mut iss_prop = SGPPropagator::from_gp_record(&records[0], 60.0).unwrap();

    println!("Created propagator: {}", iss_prop.get_name().unwrap_or("Unknown"));
    println!("Epoch: {}", iss_prop.epoch);

    // Propagate forward 1 orbit period (~93 minutes for ISS)
    iss_prop.propagate_to(iss_prop.epoch + bh::orbital_period(iss_prop.semi_major_axis()));
    let state = iss_prop.current_state();

    println!("\nState after 1 orbit:");
    println!(
        "  Position: [{:.1}, {:.1}, {:.1}] m",
        state[0], state[1], state[2]
    );
    println!(
        "  Velocity: [{:.1}, {:.1}, {:.1}] m/s",
        state[3], state[4], state[5]
    );

    // Expected output:
    // Created propagator: ISS (ZARYA)
    // Epoch: 2025-11-02 10:09:34.283 UTC

    // State after 1 orbit:
    //   Position: [6451630.2, -2126316.1, 34427.2] m
    //   Velocity: [2019.6, 5281.4, 6006.2] m/s
}
