//! Extract Keplerian orbital elements from TLE data

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Extract Keplerian elements from TLE
    let elements_deg = prop.get_elements(bh::AngleFormat::Degrees).unwrap();
    let elements_rad = prop.get_elements(bh::AngleFormat::Radians).unwrap();

    println!("Semi-major axis: {:.1} km", elements_deg[0]/1e3);
    println!("Eccentricity: {:.6}", elements_deg[1]);
    println!("Inclination: {:.4} degrees", elements_deg[2]);
    println!("RAAN: {:.4} degrees", elements_deg[3]);
    println!("Argument of perigee: {:.4} degrees", elements_deg[4]);
    println!("Mean anomaly: {:.4} degrees", elements_deg[5]);
    // Expected output:
    // Semi-major axis: 6758.7 km
    // Eccentricity: 0.000670
    // Inclination: 51.6416 degrees
    // RAAN: 247.4627 degrees
    // Argument of perigee: 130.5360 degrees
    // Mean anomaly: 325.0288 degrees
}
