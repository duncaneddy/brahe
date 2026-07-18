//! Recover mean Keplerian elements from an osculating trajectory using
//! windowed numerical averaging.
//!
//! Synthesizes one period of osculating states by evaluating the analytical
//! Brouwer-Lyddane mean -> osculating mapping at a sweep of mean anomalies
//! (holding the other mean elements fixed; no numerical propagation is
//! used). The synthesized trajectory is then averaged back down to mean
//! elements with a centered window spanning the full period.

use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Mean Keplerian elements for a LEO satellite (angles in degrees); only
    // the mean anomaly varies across the synthesized trajectory.
    let a = bh::constants::R_EARTH + 500e3;
    let (e, i, raan, argp) = (0.01, 45.0, 30.0, 60.0);
    let period = bh::orbits::orbital_period(a);

    // Sample one full period at 121 points so the exact midpoint
    // (M = 180 deg) falls on the epoch grid.
    let n_samples = 121usize;
    let epoch0 = bh::time::Epoch::from_gps_seconds(0.0);
    let method_analytical = bh::orbits::MeanElementMethod::BrouwerLyddane;

    let mut epochs = Vec::with_capacity(n_samples);
    let mut osc_states = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let frac = (k as f64) / ((n_samples - 1) as f64);
        let m = 360.0 * frac;
        let t = epoch0 + frac * period;
        let mean_state = na::SVector::<f64, 6>::new(a, e, i, raan, argp, m);
        let osc = bh::orbits::state_koe_mean_to_osc(
            &mean_state,
            method_analytical.clone(),
            bh::constants::AngleFormat::Degrees,
        )
        .unwrap();
        epochs.push(t);
        osc_states.push(osc);
    }

    // Average the synthesized osculating trajectory over a centered window
    // spanning one full period. With Truncate edge handling, only output
    // epochs whose window is fully covered by the input data survive; since
    // the window equals the full data span, that is exactly the midpoint
    // epoch.
    let config = bh::orbits::MeanElementNumericalMethodConfig {
        window_seconds: period,
        alignment: bh::orbits::WindowAlignment::Centered,
        edge: bh::orbits::WindowEdgeHandling::Truncate,
        inverse: None,
    };
    let method_numerical = bh::orbits::MeanElementMethod::Numerical(config);

    let out = bh::orbits::batch_state_koe_osc_to_mean(
        &epochs,
        &osc_states,
        method_numerical,
        bh::constants::AngleFormat::Degrees,
    )
    .unwrap();

    println!("Input osculating samples:                      {}", n_samples);
    println!("Output mean samples after windowed averaging:  {}", out.len());

    // Only the exact window-center epoch survives Truncate edge handling
    // here, since the averaging window spans the full data range.
    let recovered = out[0].1;
    let mid_m = 360.0 * ((n_samples / 2) as f64) / ((n_samples - 1) as f64);
    let original = na::SVector::<f64, 6>::new(a, e, i, raan, argp, mid_m);

    println!(
        "\nRecovered mean state: a={:.3} m, e={:.6}, i={:.3} deg, raan={:.3} deg, argp={:.3} deg, M={:.3} deg",
        recovered[0], recovered[1], recovered[2], recovered[3], recovered[4], recovered[5]
    );
    let residual = recovered - original;
    println!(
        "Residual vs. synthesized mean state (a in m, angles in deg): [{:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}]",
        residual[0], residual[1], residual[2], residual[3], residual[4], residual[5]
    );
}
