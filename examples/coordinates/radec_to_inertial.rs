//! Convert a star's right ascension/declination to an inertial unit vector and
//! back, and propagate its position forward in time using proper motion.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Barnard's Star (HIP 87937), J1991.25 Hipparcos catalog values.
    let ra = 269.454_023_05; // deg
    let dec = 4.668_288_15; // deg

    // Convert RA/Dec to an inertial unit vector (range = 1.0)
    let x_radec = na::SVector::<f64, 3>::new(ra, dec, 1.0);
    let x_inertial = bh::position_radec_to_inertial(x_radec, bh::AngleFormat::Degrees);
    println!(
        "Unit vector: [{:.6}, {:.6}, {:.6}]",
        x_inertial[0], x_inertial[1], x_inertial[2]
    );

    // Convert back to RA/Dec
    let x_radec_back = bh::position_inertial_to_radec(x_inertial, bh::AngleFormat::Degrees);
    println!(
        "RA: {:.8} deg, Dec: {:.8} deg",
        x_radec_back[0], x_radec_back[1]
    );

    assert!((x_radec_back[0] - ra).abs() < 1e-9);
    assert!((x_radec_back[1] - dec).abs() < 1e-9);

    // Propagate the star's position forward 10 years using its proper motion,
    // parallax, and radial velocity (ESA SP-1200 Vol. 1, §1.5.5).
    let epoch_from = bh::Epoch::from_mjd(48348.5625, bh::TimeSystem::TT);
    let epoch_to = bh::Epoch::from_mjd(48348.5625 + 10.0 * 365.25, bh::TimeSystem::TT);

    let (ra_new, dec_new) = bh::apply_proper_motion(
        ra,
        dec,
        -797.84,      // pm_ra* (mu_alpha* = mu_alpha * cos(dec)), mas/yr
        10326.93,     // pm_dec, mas/yr
        Some(549.30), // parallax, mas
        Some(-106.8), // radial_velocity, km/s
        epoch_from,
        epoch_to,
        bh::AngleFormat::Degrees,
    );
    println!("After 10 yr: RA: {:.6} deg, Dec: {:.6} deg", ra_new, dec_new);

    assert!((ra_new - ra).abs() > 1e-6);
}
