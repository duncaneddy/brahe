//! Recover an osculating Keplerian state from a mean state using the
//! numerical method's iterative mean -> osculating inverse.
//!
//! Unlike osculating -> mean (a direct average), mean -> osculating with
//! the numerical method has no closed form: it differentially corrects a
//! trial osculating state, numerically propagating it across the averaging
//! window and comparing the forward-averaged mean against the target,
//! until the residual converges.

use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Mean Keplerian elements for a LEO satellite (angles in degrees)
    let mean = na::SVector::<f64, 6>::new(bh::constants::R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0);
    let period = bh::orbits::orbital_period(mean[0]);

    // Dynamics used to test each trial osculating state during
    // differential correction: gravity-only (no drag/SRP parameters
    // required).
    let inverse = bh::orbits::MeanElementInverseConfig {
        force_model: bh::propagators::ForceModelConfig::earth_gravity(),
        propagation: bh::propagators::NumericalPropagationConfig::default(),
        tolerance: 1.0,
        max_iterations: 50,
    };
    let config = bh::orbits::MeanElementNumericalMethodConfig {
        window_seconds: period,
        alignment: bh::orbits::WindowAlignment::Centered,
        edge: bh::orbits::WindowEdgeHandling::PreserveWindow,
        inverse: Some(inverse),
    };
    let method = bh::orbits::MeanElementMethod::Numerical(config);

    let epoch = bh::time::Epoch::from_gps_seconds(0.0);
    let out = bh::orbits::batch_state_koe_mean_to_osc(
        &[epoch],
        &[mean],
        method,
        bh::constants::AngleFormat::Degrees,
    )
    .unwrap();
    let osc = out[0].1;

    println!(
        "Mean state:            a={:.3} m, e={:.6}, i={:.3} deg, raan={:.3} deg, argp={:.3} deg, M={:.3} deg",
        mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]
    );
    println!(
        "Recovered osculating:  a={:.3} m, e={:.6}, i={:.3} deg, raan={:.3} deg, argp={:.3} deg, M={:.3} deg",
        osc[0], osc[1], osc[2], osc[3], osc[4], osc[5]
    );
    println!("\nSemi-major axis, osculating - mean: {:+.3} m", osc[0] - mean[0]);
}
