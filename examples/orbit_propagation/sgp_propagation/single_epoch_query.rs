//! Query satellite state at arbitrary epochs without stepping

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::StateProvider;

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Query state 1 orbit later (doesn't add to trajectory)
    let query_epoch = prop.epoch + 5400.0;  // ~90 minutes

    let state_eci = prop.state_eci(query_epoch);          // ECI Cartesian
    let state_ecef = prop.state_ecef(query_epoch);        // ECEF Cartesian
    let state_kep = prop.state_as_osculating_elements(query_epoch, bh::AngleFormat::Degrees);    // Osculating Keplerian

    println!("ECI position: [{:.1}, {:.1}, {:.1}] km",
             state_eci[0]/1e3, state_eci[1]/1e3, state_eci[2]/1e3);
    println!("Osculating semi-major axis: {:.1} km", state_kep[0]/1e3);

    // Expected output:
    // ECI position: [3822.2, -1684.2, 5264.9] km
    // Osculating semi-major axis: 6725.4 km
}
