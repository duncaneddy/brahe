//! Using the default force model configuration.
//! Includes 20x20 gravity, Harris-Priester drag, SRP, and Sun/Moon third-body.

use brahe as bh;
use bh::traits::{DStatePropagator, DOrbitStateProvider};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    // Orbital elements in degrees for AngleFormat::Degrees
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // semi-major axis (m)
        0.001,                 // eccentricity
        97.8,                  // inclination (deg)
        15.0,                  // RAAN (deg)
        30.0,                  // argument of perigee (deg)
        45.0,                  // mean anomaly (deg)
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Default force model configuration
    // Includes: 20x20 EGM2008 gravity, Harris-Priester drag, SRP with conical eclipse,
    // Sun and Moon third-body perturbations
    let force_config = bh::ForceModelConfig::default();

    // Check what's enabled
    println!("Default ForceModelConfig:");
    println!("  Requires params: {}", force_config.requires_params());

    // Parameters required for default config: [mass, drag_area, Cd, srp_area, Cr]
    let params = na::DVector::from_vec(vec![
        500.0, // mass (kg)
        2.0,   // drag_area (m^2)
        2.2,   // Cd (drag coefficient)
        2.0,   // srp_area (m^2)
        1.3,   // Cr (reflectivity coefficient)
    ]);

    // Create propagator
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        force_config.clone(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Propagate for 1 orbital period (~94 minutes for LEO)
    let sma = bh::R_EARTH + 500e3;
    let orbital_period = 2.0 * PI * (sma.powi(3) / bh::GM_EARTH).sqrt();
    prop.propagate_to(epoch + orbital_period);

    // Check orbit evolution
    let final_koe = prop
        .state_koe(prop.current_epoch(), bh::AngleFormat::Degrees)
        .unwrap();
    println!("\nAfter 1 orbit ({:.1} min):", orbital_period / 60.0);
    println!("  Semi-major axis change: {:.3} m", final_koe[0] - oe[0]);
    println!("  Eccentricity change: {:.9}", final_koe[1] - oe[1]);

    // Validate - default config requires mass/area parameters
    assert!(force_config.requires_params());
    // Orbit should be affected by perturbations (change > 1m indicates non-Keplerian)
    assert!((final_koe[0] - oe[0]).abs() > 1.0);

    println!("\nExample validated successfully!");
}
