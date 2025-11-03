//! Initialize SGPPropagator from 3-line TLE with satellite name

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // 3-line TLE with satellite name
    let name = "ISS (ZARYA)";
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator with satellite name
    let prop = bh::SGPPropagator::from_3le(Some(name), line1, line2, 60.0).unwrap();

    println!("Satellite name: {:?}", prop.satellite_name);
    println!("NORAD ID: {}", prop.norad_id);
    // Expected output:
    // Satellite name: Some("ISS (ZARYA)")
    // NORAD ID: 25544
}
