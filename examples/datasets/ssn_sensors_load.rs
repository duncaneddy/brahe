//! Load and filter the SSN sensor dataset.
//!
//! This example demonstrates loading the Vallado SSN sensor sites,
//! filtering by sensor type, and inspecting a site's properties.

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::ssn_sensors::load_ssn_sensors;
use bh::utils::Identifiable;
use bh::AccessibleLocation;

fn main() {
    bh::initialize_eop().unwrap();

    // Load all SSN sensor sites
    let sites = load_ssn_sensors().unwrap();
    println!("Total SSN sites: {}", sites.len());

    // Filter by sensor type: radar/phased-array/mechanical trackers report
    // az/el/range, optical trackers report angles-only az/el
    let radars: Vec<_> = sites
        .iter()
        .filter(|s| s.properties()["sensor_type"] == "azel_range")
        .collect();
    let optical: Vec<_> = sites
        .iter()
        .filter(|s| s.properties()["sensor_type"] == "optical")
        .collect();
    println!("Radar/phased-array/mechanical sites: {}", radars.len());
    println!("Optical (angles-only) sites: {}", optical.len());

    // Inspect one site's properties
    let eglin = sites.iter().find(|s| s.get_name() == Some("Eglin")).unwrap();
    let props = eglin.properties();
    println!("\n{}", eglin.get_name().unwrap());
    println!("Location: ({:.2}, {:.2})", eglin.lat(), eglin.lon());
    println!("System: {}", props["system"]);
    println!("Category: {}", props["category"]);
    println!(
        "Elevation limits: {} - {} deg",
        props["el_min_deg"], props["el_max_deg"]
    );
    println!(
        "Range max: {:.0} km",
        props["range_max_m"].as_f64().unwrap() / 1e3
    );
    println!("Azimuth noise: {} deg", props["az_noise_deg"]);

    assert_eq!(sites.len(), 21);
    assert_eq!(radars.len() + optical.len(), sites.len());
    assert_eq!(props["sensor_type"], "azel_range");
    println!("\nExample validated successfully!");
}
