# SSN Sensor Datasets

## Overview

The SSN sensor dataset provides representative locations, field-of-view limits, and
calibration (bias/noise) values for U.S. Space Surveillance Network sites. This data is
essential for:

- **Simulating radar/optical tracking**: Build `SimpleSSNSensor` instances that generate
  az/el/range measurements consistent with a matching `AzElRangeMeasurementModel`
- **Access analysis**: Determine when a sensor's field of view covers a target orbit
- **Orbit determination testing**: Exercise EKF/UKF/BLS estimators against a realistic,
  multi-site sensor network

Brahe includes embedded GeoJSON data for 21 SSN sites. The data is:

- **Offline-capable**: No network requests required
- **Calibrated**: Includes bias and noise values for sensors with published Table 4-4 entries
- **Wrap-aware**: Azimuth field-of-view windows that cross north are represented correctly

## Loading

Load all SSN sensor sites, filter by sensor type, and inspect a site's properties:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/ssn_sensors_load.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/ssn_sensors_load.rs:7"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/ssn_sensors_load.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/ssn_sensors_load.rs.txt"
        ```

`bh.datasets.ssn_sensors.load()` returns every site as a `PointLocation`.
`SimpleSSNSensor.from_locations()` builds a sensor for every site -- radar (`azel_range`,
measuring az/el/range) and optical (`optical`, angles-only az/el) alike -- defaulting the
sites that lack full Table 4-4 calibration to zero noise and bias (flagged
`calibrated == False`, overridable with `with_noise()`/`with_bias()`).
`from_locations_calibrated()` restricts the result to the fully-calibrated sites -- this is
the set used throughout the [SSN Radar Tracking example](../../examples/ssn_tracking.md).

## Properties

Each site is a `PointLocation` with geodetic coordinates (`lon()`, `lat()`, `alt()`) and a
`properties` dictionary:

| Field | Type | Units | Description |
|-------|------|-------|--------------|
| `sensor_type` | `str` | -- | `"azel_range"` (radar/phased-array/mechanical trackers, az/el/range) or `"optical"` (angles-only optical trackers, az/el) |
| `system` | `str` | -- | Sensor system description, e.g. `"Phased Array"`, `"Radar"`, `"GEODSS"` |
| `category` | `str` | -- | Vallado network category, e.g. `"dedicated"`, `"collateral"`, `"contributing"` |
| `sensor_numbers` | `list[int]` | -- | SSN sensor ID number(s) at the site |
| `az_min_deg` | `float`, optional | degrees | Azimuth field-of-view start. Wrap-aware: when `az_min_deg > az_max_deg`, the window crosses north |
| `az_max_deg` | `float`, optional | degrees | Azimuth field-of-view end |
| `el_min_deg` | `float`, optional | degrees | Minimum elevation angle |
| `el_max_deg` | `float`, optional | degrees | Maximum elevation angle |
| `range_max_m` | `float`, optional | meters | Maximum range; absent means effectively unlimited |
| `az_bias_deg` | `float`, optional | degrees | Constant azimuth measurement bias (Table 4-4) |
| `el_bias_deg` | `float`, optional | degrees | Constant elevation measurement bias |
| `range_bias_m` | `float`, optional | meters | Constant range measurement bias |
| `az_noise_deg` | `float`, optional | degrees | Azimuth measurement noise standard deviation |
| `el_noise_deg` | `float`, optional | degrees | Elevation measurement noise standard deviation |
| `range_noise_m` | `float`, optional | meters | Range measurement noise standard deviation |

`sensor_type` determines which fields are present: `optical` sites carry no range fields at
all, and sites appearing only in Table 4-2 (location and field-of-view, no calibration)
carry no bias/noise fields. `SimpleSSNSensor.from_location()` accepts both `azel_range` and
`optical` sites; a site missing one or more noise fields still constructs, defaulted to zero
noise and flagged uncalibrated, rather than raising an error.

## Building Sensors and Measurement Models

Build a sensor from a single site, generate a measurement, and inspect the matching
`AzElRangeMeasurementModel` that `measurement_model()` builds from the sensor's own bias
and noise -- the model and the sensor stay consistent because both read the same
calibration:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/ssn_sensors_sensor.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/ssn_sensors_sensor.rs:9"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/ssn_sensors_sensor.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/ssn_sensors_sensor.rs.txt"
        ```

See [Azimuth/Elevation/Range Measurements](../estimation/measurement_models.md#azimuthelevationrange)
for how the resulting model is used in a filter.

## Source

Values are from Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed.,
Tables 4-2 (site locations and systems), 4-3 (field-of-view limits), and 4-4 (bias/noise
calibration). NAVSPASUR is excluded from the embedded dataset. These values are
representative and dated -- they reflect the published tables, not current SSN
configuration or performance, and should not be used for operational sensor modeling.

---

## See Also

- [Datasets Overview](index.md) - Understanding datasets in Brahe
- [Measurement Models](../estimation/measurement_models.md) - Azimuth/Elevation/Range measurement model
- [SSN Radar Tracking Example](../../examples/ssn_tracking.md) - Full EKF/UKF/BLS walkthrough
- [SSN Sensor Datasets API Reference](../../library_api/datasets/ssn_sensors.md) - Complete function documentation
