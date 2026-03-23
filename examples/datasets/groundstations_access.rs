//! Compute satellite access windows using groundstation datasets.
//!
//! This example demonstrates using groundstation data with brahe's
//! access computation to find contact opportunities.

#[allow(unused_imports)]
use brahe as bh;
use bh::access::location_accesses;
use bh::utils::Identifiable;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Load groundstations from a provider
    let stations = bh::datasets::groundstations::load_groundstations("ksat").unwrap();
    println!("Computing access for {} KSAT stations", stations.len());

    // Create a sun-synchronous orbit satellite
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 600e3,
        0.001,
        97.8_f64.to_radians(),
        0.0,
        0.0,
        0.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Radians);
    let propagator =
        bh::KeplerianPropagator::from_eci(epoch, state, 60.0).with_name("EO-Sat");

    // Define access constraint (minimum 5° elevation)
    let constraint = bh::ElevationConstraint::new(Some(5.0), None).unwrap();

    // Compute access windows for 24 hours
    let duration = 24.0 * 3600.0; // seconds
    let windows = location_accesses(
        &stations,
        &vec![propagator],
        epoch,
        epoch + duration,
        &constraint,
        None,
        None,
    ).unwrap();

    // Display results
    println!("\nTotal access windows: {}", windows.len());
    println!("\nFirst 5 windows:");
    for (i, window) in windows.iter().take(5).enumerate() {
        let duration_min = (window.end() - window.start()) / 60.0;
        let loc_name = window.location_name.as_deref().unwrap_or("Unknown");
        let sat_name = window.satellite_name.as_deref().unwrap_or("Unknown");
        println!(
            "{}. {:20} -> {:10}",
            i + 1,
            loc_name,
            sat_name
        );
        println!("   Start: {}", window.start());
        println!("   Duration: {:.1} minutes", duration_min);
    }

}

