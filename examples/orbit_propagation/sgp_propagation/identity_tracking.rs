//! Assign custom names and IDs to propagators

#[allow(unused_imports)]
use brahe as bh;
use brahe::utils::Identifiable;

fn main() {
    bh::initialize_eop().unwrap();

    let line0 = "ISS (ZARYA)";
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator and set identity
    let mut prop = bh::SGPPropagator::from_3le(Some(line0), line1, line2, 60.0).unwrap();

    println!("Name: {:?}", prop.get_name());
    println!("ID: {:?}", prop.get_id());
    println!("NORAD ID from TLE: {}", prop.norad_id);
    // Expected output:
    // Name:  Some("ISS (ZARYA)")
    // ID: Some(25544)
    // NORAD ID from TLE: 25544
}
