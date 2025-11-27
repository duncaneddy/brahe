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

    // Expected output:
    // Computing access for 36 KSAT stations

    // Total access windows: 213

    // First 5 windows:
    // 1. Long Beach           -> EO-Sat    
    //    Start: 2024-01-01 00:05:08.313 UTC
    //    Duration: 8.9 minutes
    // 2. Thomaston            -> EO-Sat    
    //    Start: 2024-01-01 00:07:15.029 UTC
    //    Duration: 1.7 minutes
    // 3. Inuvik               -> EO-Sat    
    //    Start: 2024-01-01 00:13:53.159 UTC
    //    Duration: 10.1 minutes
    // 4. Fairbanks            -> EO-Sat    
    //    Start: 2024-01-01 00:14:39.836 UTC
    //    Duration: 8.3 minutes
    // 5. Prudhoe Bay          -> EO-Sat    
    //    Start: 2024-01-01 00:15:18.853 UTC
    //    Duration: 9.7 minutes
}
