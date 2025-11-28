//! Query satellite states for multiple epochs at once

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::DOrbitStateProvider;

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Generate states for multiple orbits
    let orbital_period = 5400.0;  // Approximate ISS period (seconds)
    let query_epochs: Vec<bh::Epoch> = (0..5)
        .map(|i| prop.epoch + i as f64 * orbital_period)
        .collect();
    let states_eci = prop.states_eci(&query_epochs).unwrap();

    println!("Generated {} states over {} orbits", states_eci.len(), query_epochs.len());
    for (i, state) in states_eci.iter().enumerate() {
        let altitude = (state.fixed_rows::<3>(0).norm() - bh::R_EARTH) / 1e3;
        println!("  Orbit {}: altitude = {:.1} km", i, altitude);
    }
    // Expected output:
    // Generated 5 states over 5 orbits
    //   Orbit 0: altitude = 342.1 km
    //   Orbit 1: altitude = 342.3 km
    //   Orbit 2: altitude = 342.7 km
    //   Orbit 3: altitude = 343.3 km
    //   Orbit 4: altitude = 344.0 km
}
