//! Using GEO-optimized force model configuration.
//! Appropriate for geostationary orbit where SRP is dominant perturbation.

use brahe as bh;
use bh::traits::{DStatePropagator, DOrbitStateProvider};
use nalgebra as na;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch and state
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // GEO satellite
    let geo_radius = bh::R_EARTH + 35786e3; // GEO altitude
    let oe = na::SVector::<f64, 6>::new(geo_radius, 0.0001, 0.1_f64.to_radians(), 0.0, 0.0, 0.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // GEO-optimized force model (SRP dominant, no drag)
    let force_config = bh::ForceModelConfiguration::geo_default();

    // Parameters for GEO config: [mass, _, _, srp_area, Cr]
    // Note: drag_area and Cd are ignored at GEO (no atmosphere)
    let params = na::DVector::from_vec(vec![
        3000.0, // mass (kg)
        0.0,    // drag_area - not used at GEO
        0.0,    // Cd - not used at GEO
        50.0,   // srp_area (m^2) - large solar panels
        1.5,    // Cr (reflectivity coefficient)
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

    // Propagate for 7 days
    prop.propagate_to(epoch + 7.0 * 86400.0);

    // Check orbit evolution (mainly from SRP and third-body)
    let final_koe = prop
        .state_koe(prop.current_epoch(), bh::AngleFormat::Degrees)
        .unwrap();
    println!("GEO Force Model (after 7 days):");
    println!("  Initial eccentricity: {:.6}", oe[1]);
    println!("  Final eccentricity:   {:.6}", final_koe[1]);
    println!("  Eccentricity change:  {:.6}", (final_koe[1] - oe[1]).abs());
    println!(
        "  Inclination change:   {:.6} deg",
        (final_koe[2] - oe[2].to_degrees()).abs()
    );

    // Validate - SRP and third-body cause eccentricity/inclination changes
    assert!(force_config.requires_params());
    // GEO should remain near GEO (SRP causes small eccentricity growth)
    assert!((final_koe[0] - geo_radius).abs() < 10000.0); // Within 10 km

    println!("\nExample validated successfully!");
}
