//! Build a SimpleSSNSensor and generate an az/el/range measurement.
//!
//! This example demonstrates constructing a calibrated SimpleSSNSensor from
//! a Vallado SSN site, building its matching measurement model, and
//! generating a single measurement of a target inside the sensor's field of
//! view.

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::ssn_sensors::load_ssn_sensors;
use bh::estimation::SimpleSSNSensor;
use bh::utils::Identifiable;
use bh::AccessibleLocation;
use nalgebra::{SVector, Vector3};

fn main() {
    bh::initialize_eop().unwrap();

    // Load a fully-calibrated radar site and build a sensor from it
    let sites = load_ssn_sensors().unwrap();
    let eglin_site = sites
        .iter()
        .find(|s| s.get_name() == Some("Eglin"))
        .unwrap();
    let mut sensor = SimpleSSNSensor::from_location(eglin_site)
        .unwrap()
        .with_seed(42);
    println!("Sensor: {}", sensor.name());
    println!(
        "Azimuth window: {:.1} - {:.1} deg",
        sensor.az_min(),
        sensor.az_max()
    );
    println!(
        "Elevation limits: {:.1} - {:.1} deg",
        sensor.el_min(),
        sensor.el_max()
    );
    println!("Range max: {:.0} km", sensor.range_max().unwrap() / 1e3);
    println!("Calibrated: {}", sensor.calibrated());

    // Build the matching measurement model: same bias/noise as the sensor,
    // so a filter built from it stays consistent with measurements the
    // sensor produces.
    let model = sensor.measurement_model();
    println!("Measurement model: {}", model.name());

    // A target 500 km away, due south (within Eglin's southwest-facing
    // azimuth window) and 45 deg above the horizon, built by offsetting the
    // site in the local East-North-Zenith frame and converting to ECI.
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let (az, el, rng) = (180.0f64.to_radians(), 45.0f64.to_radians(), 500e3);
    let horizontal = rng * el.cos();
    let enz_offset = Vector3::new(horizontal * az.sin(), horizontal * az.cos(), rng * el.sin());
    let target_ecef = bh::relative_position_enz_to_ecef(
        eglin_site.center_ecef(),
        enz_offset,
        bh::EllipsoidalConversionType::Geodetic,
    );
    let state_ecef = SVector::<f64, 6>::new(
        target_ecef[0],
        target_ecef[1],
        target_ecef[2],
        0.0,
        0.0,
        0.0,
    );
    let state_eci = bh::state_ecef_to_eci(epoch, state_ecef);
    let state_eci_dvec = nalgebra::DVector::from_column_slice(state_eci.as_slice());

    // True (noise-free, bias-free) geometry vs. a simulated measurement
    let truth = sensor.azelrange(&epoch, &state_eci_dvec);
    println!(
        "\nTrue az/el/range: [{:.2} deg, {:.2} deg, {:.1} km]",
        truth[0],
        truth[1],
        truth[2] / 1e3
    );

    let measurement = sensor.measure(&epoch, &state_eci_dvec).unwrap();
    println!(
        "Measured az/el/range: [{:.2} deg, {:.2} deg, {:.1} km]",
        measurement[0],
        measurement[1],
        measurement[2] / 1e3
    );

    assert!(
        (measurement[0] - truth[0]).abs() < 1.0,
        "azimuth should stay close to truth"
    );
    assert!(
        (measurement[1] - truth[1]).abs() < 1.0,
        "elevation should stay close to truth"
    );
    assert!(
        (measurement[2] - truth[2]).abs() < 5000.0,
        "range should stay close to truth"
    );
    println!("\nExample validated successfully!");
}
