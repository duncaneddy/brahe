//! Convert a single Keplerian state between mean and osculating elements
//! using the analytical Brouwer-Lyddane method.
//!
//! Demonstrates mean -> osculating -> mean: the osculating state differs
//! from the mean state by short-period J2 oscillations, and round-tripping
//! recovers the original mean state only approximately (first-order
//! truncation of the underlying series).

use brahe as bh;
use nalgebra as na;

fn print_koe(label: &str, koe: &na::SVector<f64, 6>) {
    println!(
        "{}: a={:.3} m, e={:.6}, i={:.3} deg, raan={:.3} deg, argp={:.3} deg, M={:.3} deg",
        label, koe[0], koe[1], koe[2], koe[3], koe[4], koe[5]
    );
}

fn main() {
    bh::initialize_eop().unwrap();

    // Mean Keplerian elements for a LEO satellite (angles in degrees)
    let mean = na::SVector::<f64, 6>::new(bh::constants::R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0);
    let method = bh::orbits::MeanElementMethod::BrouwerLyddane;

    // Mean -> osculating
    let osc = bh::orbits::state_koe_mean_to_osc(
        &mean,
        method.clone(),
        bh::constants::AngleFormat::Degrees,
    )
    .unwrap();

    // Osculating -> mean (round trip back toward the original mean state)
    let mean_recovered =
        bh::orbits::state_koe_osc_to_mean(&osc, method, bh::constants::AngleFormat::Degrees)
            .unwrap();

    print_koe("Mean elements       ", &mean);
    print_koe("Osculating elements ", &osc);
    print_koe("Recovered mean      ", &mean_recovered);

    println!(
        "\nSemi-major axis, osculating - mean:       {:+.3} m",
        osc[0] - mean[0]
    );
    println!(
        "Semi-major axis, round-trip residual:     {:+.6} m",
        mean_recovered[0] - mean[0]
    );
    println!(
        "Argument of perigee, round-trip residual: {:+.6e} deg",
        mean_recovered[4] - mean[4]
    );
}
