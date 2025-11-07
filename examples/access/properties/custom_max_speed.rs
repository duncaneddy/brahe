//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! nalgebra = "0.34"
//! ```

#[allow(unused_imports)]
use brahe as bh;
use bh::access::{AccessPropertyComputer, AccessWindow, PropertyValue, SamplingConfig};
use bh::utils::{BraheError, Identifiable};
use std::collections::HashMap;

struct MaxSpeedComputer {
    sampling_config: SamplingConfig,
}

impl AccessPropertyComputer for MaxSpeedComputer {
    fn sampling_config(&self) -> SamplingConfig {
        self.sampling_config.clone()
    }

    fn compute(
        &self,
        _window: &AccessWindow,
        _sample_epochs: &[f64],
        sample_states_ecef: &[nalgebra::SVector<f64, 6>],
        _location_ecef: &nalgebra::Vector3<f64>,
        _location_geodetic: &nalgebra::Vector3<f64>,
    ) -> Result<HashMap<String, PropertyValue>, BraheError> {
        let mut max_speed = 0.0;

        for state in sample_states_ecef {
            let velocity = state.fixed_rows::<3>(3);
            let speed = velocity.norm();
            if speed > max_speed {
                max_speed = speed;
            }
        }

        let mut props = HashMap::new();
        props.insert("max_ground_speed".to_string(), PropertyValue::Scalar(max_speed));
        Ok(props)
    }

    fn property_names(&self) -> Vec<String> {
        vec!["max_ground_speed".to_string()]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    // ISS orbit
    let tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999";
    let tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601";
    let propagator = bh::SGPPropagator::from_tle(tle_line1, tle_line2, 60.0)?
        .with_name("ISS");

    let epoch_start = propagator.epoch;
    let epoch_end = epoch_start + 24.0 * 3600.0;

    // Ground station
    let location = bh::PointLocation::new(-74.0060, 40.7128, 0.0);

    // Compute with custom property
    let max_speed = MaxSpeedComputer {
        sampling_config: SamplingConfig::FixedInterval {
            interval: 0.5,  // 0.5 seconds
            offset: 0.0,
        },
    };

    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;
    let windows = bh::location_accesses(
        &location,
        &propagator,
        epoch_start,
        epoch_end,
        &constraint,
        Some(&[&max_speed]),  // Property computers
        None,  // Use default config
        None,  // No time tolerance
    )?;

    for window in &windows {
        let speed = match window.properties.additional.get("max_ground_speed").unwrap() {
            PropertyValue::Scalar(s) => s,
            _ => panic!("Expected Scalar"),
        };
        println!("Max speed: {:.1} m/s", speed);
    }

    // Output example:
    // Max speed: 7360.1 m/s
    // Max speed: 7365.5 m/s
    // Max speed: 7361.2 m/s
    // Max speed: 7357.5 m/s
    // Max speed: 7357.8 m/s
    // Max speed: 7360.0 m/s

    Ok(())
}
