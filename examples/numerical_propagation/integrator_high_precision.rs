//! High-precision numerical propagation using RKN1210 integrator.
//! Demonstrates the highest-accuracy integrator for precision requirements.

use brahe as bh;
use bh::traits::{DStatePropagator, SStatePropagator};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.001,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // High-precision configuration: RKN1210 with tight tolerances
    let config_hp = bh::NumericalPropagationConfig::high_precision();
    println!("High-precision config:");
    println!("  Method: RKN1210 (Runge-Kutta-Nystrom 12(10))");
    println!("  Tolerances: Very tight (1e-10 rel, 1e-8 abs)");

    // Standard precision for comparison
    let config_std = bh::NumericalPropagationConfig::default();

    // Create propagators (use two-body for analytical comparison)
    let mut prop_hp = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        config_hp,
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let mut prop_std = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        config_std,
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // Keplerian propagator as analytical reference (use Cartesian for direct comparison)
    let mut kep_prop = bh::KeplerianPropagator::from_eci(
        epoch,
        state,
        60.0,
    );

    // Propagate for 10 orbits
    let orbital_period = 2.0 * PI * (oe[0].powi(3) / bh::GM_EARTH).sqrt();
    let end_time = epoch + 10.0 * orbital_period;

    prop_hp.propagate_to(end_time);
    prop_std.propagate_to(end_time);
    kep_prop.propagate_to(end_time);

    // Compare errors vs analytical
    let kep_state = kep_prop.current_state();
    let kep_dvec = na::DVector::from_column_slice(kep_state.as_slice());
    let hp_error =
        (prop_hp.current_state().fixed_rows::<3>(0) - kep_dvec.fixed_rows::<3>(0)).norm();
    let std_error =
        (prop_std.current_state().fixed_rows::<3>(0) - kep_dvec.fixed_rows::<3>(0)).norm();

    println!(
        "\nAfter 10 orbits ({:.1} hours):",
        10.0 * orbital_period / 3600.0
    );
    println!("  High-precision error: {:.9} m", hp_error);
    println!("  Standard error:       {:.9} m", std_error);
    if hp_error > 0.0 {
        println!("  Improvement factor:   {:.1}x", std_error / hp_error);
    }

    // Validate - high-precision should be excellent, standard is reasonable
    assert!(hp_error < 0.001); // Sub-millimeter for high precision
    assert!(std_error < 50.0); // Standard accumulates more error over 10 orbits

    println!("\nExample validated successfully!");
}
