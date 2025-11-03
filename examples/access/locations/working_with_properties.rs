//! Demonstrate adding and retrieving custom properties on locations.
//! Shows scalar, string, and boolean property types.

#[allow(unused_imports)]
use brahe as bh;
use bh::AccessibleLocation;
use serde_json::json;

fn main() {
    bh::initialize_eop().unwrap();

    let location = bh::PointLocation::new(-122.4194, 37.7749, 0.0)
        .add_property("antenna_gain_db", json!(42.5))
        .add_property("frequency_mhz", json!(8450.0))
        .add_property("operator", json!("NOAA"))
        .add_property("uplink_enabled", json!(true));

    // Access properties
    let props = location.properties();
    if let Some(gain) = props.get("antenna_gain_db") {
        println!("Antenna Gain: {}", gain);
    }
    if let Some(operator) = props.get("operator") {
        println!("Operator: {}", operator);
    }
    if let Some(uplink) = props.get("uplink_enabled") {
        println!("Uplink Enabled: {}", uplink);
    }

    // Expected output:
    // Antenna Gain: 42.5
    // Operator: "NOAA"
    // Uplink Enabled: true
}
