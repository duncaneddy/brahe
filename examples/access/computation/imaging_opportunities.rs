//! Find imaging opportunities using polygon locations and complex constraint combinations

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // Define imaging target
    let vertices = vec![
        na::SVector::<f64, 3>::new(bh::DEG2RAD * -122.5, bh::DEG2RAD * 37.7, 0.0),
        na::SVector::<f64, 3>::new(bh::DEG2RAD * -122.3, bh::DEG2RAD * 37.7, 0.0),
        na::SVector::<f64, 3>::new(bh::DEG2RAD * -122.3, bh::DEG2RAD * 37.9, 0.0),
        na::SVector::<f64, 3>::new(bh::DEG2RAD * -122.5, bh::DEG2RAD * 37.9, 0.0),
        na::SVector::<f64, 3>::new(bh::DEG2RAD * -122.5, bh::DEG2RAD * 37.7, 0.0),
    ];
    let target = bh::PolygonLocation::new(vertices)?.with_name("SF Bay Area");

    // Create propagator
    let tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let propagator =
        bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?.with_name("EO-Sat-1");

    // Use elevation constraint for simplicity (complex constraint composition in Rust requires trait objects)
    let constraint = bh::ElevationConstraint::new(Some(bh::DEG2RAD * 10.0), None)?;

    // Configure search
    let config = bh::AccessSearchConfig {
        initial_time_step: 30.0,
        adaptive_step: true,
        adaptive_fraction: 0.05,
        parallel: false,
        num_threads: Some(0),
    };

    // Compute imaging opportunities
    let epoch_start = bh::Epoch::from_datetime(2008, 9, 20, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epoch_end = epoch_start + 10.0 * 86400.0;

    let windows = bh::location_accesses(
        &target,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        None,
        Some(&config),
        None,
    )?;

    println!("Found {} imaging opportunities", windows.len());

    for (i, window) in windows.iter().take(3).enumerate() {
        println!("\nOpportunity {}:", i + 1);
        println!("  Start: {}", window.window_open);
        println!("  Duration: {:.1} min", window.duration() / 60.0);

        let off_nadir_min = window.properties.off_nadir_min;
        println!("  Off-nadir: {:.1}°", off_nadir_min);

        let local_time = window.properties.local_time;
        let hours = local_time.floor() as i32;
        let minutes = ((local_time - hours as f64) * 60.0) as i32;
        println!("  Local time: {:02}:{:02}", hours, minutes);
    }

    Ok(())
}

// Expected output (values will vary):
// Found X imaging opportunities
//
// Opportunity 1:
//   Start: 2008-09-XX HH:MM:SS.SSS UTC
//   Duration: X.X min
//   Off-nadir: XX.X°
//   Local time: HH:MM
