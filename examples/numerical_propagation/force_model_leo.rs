//! Using LEO-optimized force model configuration.
//! Appropriate for low Earth orbit satellites where drag is significant.

use brahe as bh;
use bh::traits::{DStatePropagator, DOrbitStateProvider};
use nalgebra as na;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // LEO satellite at 400 km altitude (ISS-like)
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 400e3,
        0.001,
        51.6_f64.to_radians(),
        0.0,
        0.0,
        0.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // LEO-optimized force model
    let force_config = bh::ForceModelConfig::leo_default();

    // Parameters for LEO config: [mass, drag_area, Cd, srp_area, Cr]
    // ISS-like parameters
    let params = na::DVector::from_vec(vec![
        420000.0, // mass (kg) - ISS is ~420,000 kg
        1600.0,   // drag_area (m^2) - ISS cross-section
        2.2,      // Cd (drag coefficient)
        1600.0,   // srp_area (m^2)
        1.2,      // Cr (reflectivity coefficient)
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

    // Propagate for 1 day
    prop.propagate_to(epoch + 86400.0);

    // Check orbit decay due to drag
    let final_koe = prop
        .state_koe(prop.current_epoch(), bh::AngleFormat::Degrees)
        .unwrap();
    println!("LEO Force Model (ISS-like orbit after 1 day):");
    println!(
        "  Initial altitude: {:.3} km",
        (oe[0] - bh::R_EARTH) / 1e3
    );
    println!(
        "  Final altitude:   {:.3} km",
        (final_koe[0] - bh::R_EARTH) / 1e3
    );
    println!("  Altitude decay:   {:.3} m", oe[0] - final_koe[0]);

    // Validate - LEO config requires mass/area parameters
    assert!(force_config.requires_params());
    // Orbit should be affected by perturbations (change > 1m indicates non-Keplerian)
    assert!((final_koe[0] - oe[0]).abs() > 1.0);

    println!("\nExample validated successfully!");
}
