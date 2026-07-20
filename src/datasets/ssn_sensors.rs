/*!
 * SSN (Space Surveillance Network) sensor dataset loading.
 *
 * Provides representative SSN sensor sites — locations, field-of-view limits,
 * and calibration (bias/noise) values — from Vallado, *Fundamentals of
 * Astrodynamics and Applications*, 4th Ed., Tables 4-2, 4-3, and 4-4.
 * Data is embedded in the binary for offline use.
 */

use crate::access::location::PointLocation;
use crate::datasets::loaders::parse_point_locations_from_geojson;
use crate::utils::errors::BraheError;

/// Load the Vallado SSN sensor sites.
///
/// Returns representative US Space Surveillance Network sensor sites with
/// locations from Vallado Table 4-2, az/el/range limits from Table 4-3, and
/// bias/noise calibration values from Table 4-4. All values are embedded in
/// the binary; no network access is required.
///
/// Each returned [`PointLocation`] carries properties:
/// - `sensor_type`: `"azel_range"` (radar/phased-array/mechanical trackers)
///   or `"azel"` (angles-only optical trackers)
/// - `system`, `category`, `sensor_numbers`: descriptive metadata
/// - Optional limits (degrees / meters): `az_min_deg`, `az_max_deg`
///   (`az_min_deg > az_max_deg` means the window crosses north),
///   `el_min_deg`, `el_max_deg`, `range_max_m`
/// - Optional calibration: `range_bias_m`, `az_bias_deg`, `el_bias_deg`,
///   `range_noise_m`, `az_noise_deg`, `el_noise_deg`
///
/// # Returns
/// * `Result<Vec<PointLocation>, BraheError>` - 21 sensor site locations
///
/// # Example
/// ```
/// use brahe::datasets::ssn_sensors::load_ssn_sensors;
///
/// let sensors = load_ssn_sensors().unwrap();
/// assert_eq!(sensors.len(), 21);
/// ```
pub fn load_ssn_sensors() -> Result<Vec<PointLocation>, BraheError> {
    let json_str = include_str!("../../data/ssn_sensors/vallado_ssn.json");
    let geojson: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| BraheError::ParseError(format!("Failed to parse embedded JSON: {}", e)))?;
    parse_point_locations_from_geojson(&geojson)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::access::location::AccessibleLocation;
    use crate::utils::identifiable::Identifiable;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_load_ssn_sensors() {
        let sensors = load_ssn_sensors().unwrap();
        assert_eq!(sensors.len(), 21);
        for s in &sensors {
            assert!(s.get_name().is_some());
            assert!(s.lon() >= -180.0 && s.lon() <= 180.0);
            assert!(s.lat() >= -90.0 && s.lat() <= 90.0);
            let props = s.properties();
            assert!(props.contains_key("sensor_type"));
            assert!(props.contains_key("system"));
            assert!(props.contains_key("category"));
            assert!(props["sensor_numbers"].is_array());
        }
    }

    #[test]
    #[parallel]
    fn test_ssn_sensor_values_against_vallado_tables() {
        let sensors = load_ssn_sensors().unwrap();
        let eglin = sensors
            .iter()
            .find(|s| s.get_name() == Some("Eglin"))
            .unwrap();
        assert_eq!(eglin.lat(), 30.57);
        assert_eq!(eglin.lon(), -86.21);
        let p = eglin.properties();
        assert_eq!(p["sensor_type"], "azel_range");
        assert_eq!(p["range_max_m"].as_f64().unwrap(), 13_210_000.0);
        assert_eq!(p["az_min_deg"].as_f64().unwrap(), 145.0);
        assert_eq!(p["az_max_deg"].as_f64().unwrap(), 215.0);
        assert_eq!(p["el_min_deg"].as_f64().unwrap(), 1.0);
        assert_eq!(p["range_bias_m"].as_f64().unwrap(), 4.3);
        assert_eq!(p["az_noise_deg"].as_f64().unwrap(), 0.0154);

        // Wrap-around azimuth window site
        let cape_cod = sensors
            .iter()
            .find(|s| s.get_name() == Some("Cape Cod"))
            .unwrap();
        let p = cape_cod.properties();
        assert!(p["az_min_deg"].as_f64().unwrap() > p["az_max_deg"].as_f64().unwrap());

        // Optical site has no range fields
        let socorro = sensors
            .iter()
            .find(|s| s.get_name() == Some("Socorro"))
            .unwrap();
        let p = socorro.properties();
        assert_eq!(p["sensor_type"], "azel");
        assert!(!p.contains_key("range_max_m"));
        assert!(!p.contains_key("range_noise_m"));

        // Sites present only in Table 4-2 carry no calibration fields
        let haystack = sensors
            .iter()
            .find(|s| s.get_name() == Some("Haystack"))
            .unwrap();
        assert!(!haystack.properties().contains_key("az_noise_deg"));
    }

    #[test]
    #[parallel]
    fn test_ssn_sensor_type_counts() {
        let sensors = load_ssn_sensors().unwrap();
        let azel = sensors
            .iter()
            .filter(|s| s.properties()["sensor_type"] == "azel_range")
            .count();
        let optical = sensors
            .iter()
            .filter(|s| s.properties()["sensor_type"] == "azel")
            .count();
        assert_eq!(azel, 15);
        assert_eq!(optical, 6);
    }
}
