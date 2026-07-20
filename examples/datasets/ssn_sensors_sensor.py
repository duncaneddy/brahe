# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build a SimpleSSNSensor and generate an az/el/range measurement.

This example demonstrates constructing a calibrated SimpleSSNSensor from a
Vallado SSN site, building its matching measurement model, and generating a
single measurement of a target inside the sensor's field of view.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Load a fully-calibrated radar site and build a sensor from it
sites = bh.datasets.ssn_sensors.load()
eglin_site = next(s for s in sites if s.get_name() == "Eglin")
sensor = bh.SimpleSSNSensor.from_location(eglin_site, seed=42)
print(f"Sensor: {sensor.name}")
print(f"Azimuth window: {sensor.az_min:.1f} - {sensor.az_max:.1f} deg")
print(f"Elevation limits: {sensor.el_min:.1f} - {sensor.el_max:.1f} deg")
print(f"Range max: {sensor.range_max / 1e3:.0f} km")
print(f"Calibrated: {sensor.calibrated}")

# Build the matching measurement model: same bias/noise as the sensor, so a
# filter built from it stays consistent with measurements the sensor produces
model = sensor.measurement_model()
print(f"Measurement model: {model.name()}")

# A target 500 km away, due south (within Eglin's southwest-facing azimuth
# window) and 45 deg above the horizon, built by offsetting the site in the
# local East-North-Zenith frame and converting to ECI.
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
az, el, rng = np.radians(180.0), np.radians(45.0), 500e3
horizontal = rng * np.cos(el)
enz_offset = np.array(
    [horizontal * np.sin(az), horizontal * np.cos(az), rng * np.sin(el)]
)
target_ecef = bh.relative_position_enz_to_ecef(
    eglin_site.center_ecef(), enz_offset, bh.EllipsoidalConversionType.GEODETIC
)
state_eci = bh.state_ecef_to_eci(epoch, np.concatenate([target_ecef, np.zeros(3)]))

# True (noise-free, bias-free) geometry vs. a simulated measurement
truth = sensor.azelrange(epoch, state_eci)
print(
    f"\nTrue az/el/range: [{truth[0]:.2f} deg, {truth[1]:.2f} deg, {truth[2] / 1e3:.1f} km]"
)

measurement = sensor.measure(epoch, state_eci)
print(
    f"Measured az/el/range: [{measurement[0]:.2f} deg, {measurement[1]:.2f} deg, "
    f"{measurement[2] / 1e3:.1f} km]"
)

assert measurement is not None, "Target inside the field of view should be visible"
assert abs(measurement[0] - truth[0]) < 1.0, "azimuth should stay close to truth"
assert abs(measurement[1] - truth[1]) < 1.0, "elevation should stay close to truth"
assert abs(measurement[2] - truth[2]) < 5000.0, "range should stay close to truth"
print("\nExample validated successfully!")
