//! High-precision orbit propagation using RKN1210 integrator.

#[allow(unused_imports)]
use brahe as bh;
use brahe::{constants::*, integrators::*};
use nalgebra::DVector;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Define HEO orbit (Molniya-type)
    let a = 26554e3;  // Semi-major axis (m)
    let e = 0.74;     // Eccentricity
    let i = 63.4;  // Inclination

    // Convert to Cartesian state
    let oe = [a, e, i, 0.0, 0.0, 0.0];
    let state0 = bh::state_osculating_to_cartesian(oe.into(), bh::AngleFormat::Degrees);
    let state0_dv = DVector::from_vec(state0.as_slice().to_vec());

    // Orbital period
    let period = bh::orbital_period(a);

    // Two-body dynamics
    let dynamics = |_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        let r = nalgebra::Vector3::new(state[0], state[1], state[2]);
        let v = nalgebra::Vector3::new(state[3], state[4], state[5]);
        let r_norm = r.norm();
        let a = -GM_EARTH / r_norm.powi(3) * r;
        DVector::from_vec(vec![v[0], v[1], v[2], a[0], a[1], a[2]])
    };

    println!("High-Precision HEO Orbit Propagation");
    println!("Semi-major axis: {:.1} km", a / 1e3);
    println!("Eccentricity: {}", e);
    println!("Period: {:.2} hours\n", period / 3600.0);

    // Create RKN1210 integrator with very tight tolerances
    let abs_tol = 1e-14;
    let rel_tol = 1e-13;
    let config = IntegratorConfig::adaptive(abs_tol, rel_tol);
    let integrator = RKN1210DIntegrator::with_config(6, Box::new(dynamics), None, None, None, config);

    println!("Using RKN1210 with tol={:.0e}", abs_tol);
    println!("Propagating for one orbital period...\n");

    // Propagate for one orbit
    let mut t = 0.0;
    let mut state = state0_dv.clone();
    let mut dt: f64 = 60.0;
    let mut steps = 0;
    let mut total_error = 0.0;

    while t < period {
        let result = integrator.step(t, state, Some(dt.min(period - t)));

        let dt_used = result.dt_used;
        t += dt_used;
        state = result.state;
        dt = result.dt_next;
        steps += 1;
        total_error += result.error_estimate.unwrap();

        // Print at intervals
        if steps % 10 == 1 {
            let r = nalgebra::Vector3::new(state[0], state[1], state[2]);
            let r_norm = r.norm();
            println!("t={:6.2}h  r={:8.1}km  dt={:6.1}s  err={:.2e}",
                     t / 3600.0, r_norm / 1e3, dt_used, result.error_estimate.unwrap());
        }
    }

    println!("\nPropagation complete!");
    println!("Total steps: {}", steps);
    println!("Average step: {:.1} s", period / steps as f64);
    println!("Cumulative error estimate: {:.2e}", total_error);

    // Verify orbit closure
    let final_state = [state[0], state[1], state[2], state[3], state[4], state[5]];
    let final_oe = bh::state_cartesian_to_osculating(final_state.into(), bh::AngleFormat::Degrees);
    let initial_oe = bh::state_cartesian_to_osculating(state0, bh::AngleFormat::Degrees);

    println!("\nOrbit element errors after one period:");
    println!("  Semi-major axis: {:.3e} m", (final_oe[0] - initial_oe[0]).abs());
    println!("  Eccentricity:    {:.3e}", (final_oe[1] - initial_oe[1]).abs());
}

// Expected output:
// High-Precision HEO Orbit Propagation
// Semi-major axis: 26554.0 km
// Eccentricity: 0.74
// Period: 11.96 hours

// Using RKN1210 with tol=1e-14
// Propagating for one orbital period...

// t=  0.02h  r=  6915.2km  dt=  60.0s  err=1.14e-03
// t=  0.61h  r= 14336.8km  dt= 308.4s  err=2.37e-02
// t=  2.54h  r= 34813.8km  dt= 900.0s  err=1.52e-03
// t=  5.04h  r= 45404.2km  dt= 900.0s  err=1.65e-03
// t=  7.54h  r= 44004.3km  dt= 900.0s  err=0.00e+00
// t= 10.04h  r= 29862.5km  dt= 900.0s  err=6.33e-03
// t= 11.66h  r=  9739.4km  dt= 291.4s  err=2.59e-01

// Propagation complete!
// Total steps: 67
// Average step: 642.7 s
// Cumulative error estimate: 5.85e+00

// Orbit element errors after one period:
//   Semi-major axis: 3.725e-08 m
//   Eccentricity:    4.441e-16