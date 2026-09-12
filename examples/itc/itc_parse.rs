//! Read a Modified ITC ephemeris file and inspect its contents.
//!
//! Parses a Starlink public ephemeris, prints the header, the record count,
//! the first state vector in SI units, and the diagonal of the first
//! covariance matrix.

use brahe as bh;

const PATH: &str = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt";

fn main() {
    let itc = bh::itc::ITC::from_file(PATH).unwrap();
    let header = &itc.header;

    println!("Source file:      {}", itc.source_name.as_ref().map(|n| n.to_string()).unwrap_or_default());
    println!("Created:          {}", header.created.map(|e| e.to_string()).unwrap_or_default());
    println!("Ephemeris start:  {}", header.ephemeris_start.map(|e| e.to_string()).unwrap_or_default());
    println!("Ephemeris stop:   {}", header.ephemeris_stop.map(|e| e.to_string()).unwrap_or_default());
    println!("Step size:        {} s", header.step_size.unwrap_or(0.0));
    println!("State frame:      {}", header.state_frame);
    println!("Covariance frame: {}", header.covariance_frame);
    println!("Records:          {}", itc.len());
    println!("Has covariance:   {}", itc.has_covariance());

    let first = &itc.states[0];
    println!("First epoch:      {}", first.epoch);
    println!("Position [m]:     [{:.3}, {:.3}, {:.3}]", first.position[0], first.position[1], first.position[2]);
    println!("Velocity [m/s]:   [{:.6}, {:.6}, {:.6}]", first.velocity[0], first.velocity[1], first.velocity[2]);

    let cov = &itc.covariances[0];
    let sigma: Vec<f64> = (0..6).map(|i| cov[(i, i)].sqrt()).collect();
    println!("1-sigma RTN position [m]:   [{:.3}, {:.3}, {:.3}]", sigma[0], sigma[1], sigma[2]);
    println!("1-sigma RTN velocity [m/s]: [{:.6}, {:.6}, {:.6}]", sigma[3], sigma[4], sigma[5]);
}
