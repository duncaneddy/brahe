//! Compute a star's azimuth/elevation as seen from a ground site at a given
//! epoch, and convert back from azimuth/elevation to right
//! ascension/declination.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let site = na::SVector::<f64, 3>::new(-122.17, 37.43, 100.0); // Stanford, deg/deg/m
    let x_radec = na::SVector::<f64, 3>::new(101.28, -16.72, 1.0); // Sirius, deg/deg/(unit range)

    let x_azel = bh::position_radec_to_azel(x_radec, site, epc, bh::AngleFormat::Degrees);
    println!(
        "Azimuth: {:.4} deg, Elevation: {:.4} deg",
        x_azel[0], x_azel[1]
    );

    let x_radec_back = bh::position_azel_to_radec(x_azel, site, epc, bh::AngleFormat::Degrees);
    println!(
        "RA: {:.6} deg, Dec: {:.6} deg",
        x_radec_back[0], x_radec_back[1]
    );

    assert!((x_radec_back[0] - x_radec[0]).abs() < 1e-6);
    assert!((x_radec_back[1] - x_radec[1]).abs() < 1e-6);
}
