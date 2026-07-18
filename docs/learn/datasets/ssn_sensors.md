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

Load all SSN sensor sites and build `SimpleSSNSensor` instances from the ones that support
az/el/range simulation:

``` python
--8<-- "./examples/examples/ssn_tracking.py:load_sensors"
```

This prints:

```
Loaded 21 SSN sites, 13 az/el/range sensors

====================================================================================================
SSN Sensor Network
====================================================================================================
Name                        System        El Min   Range Max  Az Noise  El Noise  Range Noise
----------------------------------------------------------------------------------------------------
Eglin                       Phased Array    1.0°    13210 km   0.0154°   0.0147°       32.1 m
Fylingdales                 Phased Array    4.0°     4820 km   0.0220°   0.0200°       50.0 m
Ascension                   Radar           1.0°     1900 km   0.0283°   0.0248°      101.7 m
Clear                       Radar           1.0°     4910 km   0.0791°   0.0240°       62.5 m
Antigua                     Radar           0.0°     2550 km   0.0224°   0.0139°       92.5 m
Cape Cod                    Phased Array    3.0°     5555 km   0.0260°   0.0220°       26.0 m
Beale                       Phased Array    3.0°     5555 km   0.0320°   0.0330°       35.0 m
Shemya                      Phased Array    0.0°   unlimited   0.0540°   0.0530°        2.9 m
Thule                       Phased Array    3.0°     5555 km   0.0260°   0.0220°       26.0 m
Cavalier                    Phased Array    2.0°     3300 km   0.0125°   0.0086°       28.0 m
Kaena Point                 Radar           0.0°     6380 km   0.0224°   0.0139°       92.5 m
Kwajalein                   Radar (ALCOR/ALTAIR/TRADEX)    1.0°     4500 km   0.0318°   0.0129°      162.9 m
Millstone                   Radar           0.0°    40744 km   0.0100°   0.0100°      150.0 m
====================================================================================================
```

`bh.datasets.ssn_sensors.load()` returns all 21 sites as `PointLocation` objects.
`SimpleSSNSensor.from_locations()` builds a sensor for every site that has both
`sensor_type == "azel_range"` and complete noise calibration -- 13 of the 21 sites. The
remaining 8 are skipped: 6 optical (`radec`) sites, and 2 sites (HAX, a mechanical tracker,
and Haystack, a radar) that Table 4-2 lists without the Table 4-4 calibration values
`SimpleSSNSensor` requires.

## Properties

Each site is a `PointLocation` with geodetic coordinates (`lon()`, `lat()`, `alt()`) and a
`properties` dictionary:

| Field | Type | Units | Description |
|-------|------|-------|--------------|
| `sensor_type` | `str` | -- | `"azel_range"` (radar/phased-array/mechanical trackers) or `"radec"` (optical sites; not yet supported by a measurement model) |
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

`sensor_type` determines which fields are present: `radec` sites carry no range fields at
all, and sites appearing only in Table 4-2 (location and field-of-view, no calibration)
carry no bias/noise fields. `SimpleSSNSensor.from_location()` requires `sensor_type ==
"azel_range"` plus all three noise fields; it raises `ValueError` otherwise, and
`from_locations()` silently skips sites that don't qualify.

## Building Sensors and Measurement Models

Filter and simulate on the same sensor with `measurement_model()`, which builds an
`AzElRangeMeasurementModel` using the sensor's own bias and noise so the two stay
consistent:

```python
import brahe as bh

sites = bh.datasets.ssn_sensors.load()
radars = [s for s in sites if s.properties["sensor_type"] == "azel_range"]
sensors = bh.SimpleSSNSensor.from_locations(sites, seed=42)

sensor = sensors[0]
model = sensor.measurement_model()  # AzElRangeMeasurementModel matching this sensor
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
