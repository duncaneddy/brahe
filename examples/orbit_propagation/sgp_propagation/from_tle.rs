//! Initialize SGPPropagator from 2-line TLE data

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::OrbitPropagator;

fn main() {
    bh::initialize_eop().unwrap();  // Required for accurate frame transformations

    // ISS TLE data (example)
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator with 60-second step size
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    println!("NORAD ID: {}", prop.norad_id);
    println!("TLE epoch: {}", prop.epoch);
    println!("Initial position magnitude: {:.1} km",
             prop.initial_state().fixed_rows::<3>(0).norm() / 1e3);
    // Expected output:
    // NORAD ID: 25544
    // TLE epoch: 2008-09-20 12:25:40.104 UTC
    // Initial position magnitude: 6720.2 km
}
