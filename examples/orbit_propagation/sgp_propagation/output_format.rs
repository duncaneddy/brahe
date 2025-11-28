//! Configure SGPPropagator output format (frame and representation)

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::{OrbitFrame, SStatePropagator, OrbitRepresentation};

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create with ECEF Cartesian output
    let mut prop_ecef = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap()
        .with_output_format(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

    // Or with Keplerian output (ECI only)
    let mut prop_kep = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap()
        .with_output_format(OrbitFrame::ECI, OrbitRepresentation::Keplerian, Some(bh::AngleFormat::Degrees));

    // Propagate to 1 hour after epoch
    let dt = 3600.0;
    prop_ecef.propagate_to(prop_ecef.epoch + dt);
    prop_kep.propagate_to(prop_kep.epoch + dt);

    let state_ecef = prop_ecef.current_state();
    println!("ECEF position (km): [{:.3}, {:.3}, {:.3}]",
             state_ecef[0] / 1e3, state_ecef[1] / 1e3, state_ecef[2] / 1e3);

    let state_kep = prop_kep.current_state();
    println!("Keplerian elements: [{:.1} km, {:.4}, {:.4}, {:.4} deg, {:.4} deg, {:.4} deg]",
             state_kep[0] / 1e3, state_kep[1], state_kep[2],
             state_kep[3], state_kep[4], state_kep[5]);
}

// Output:
// ECEF position (km): [5548.632, 2869.310, -2526.643]
// Keplerian elements: [8198.2 km, 0.1789, 47.9402, 249.8056 deg, 323.0545 deg, 4.5675 deg]
