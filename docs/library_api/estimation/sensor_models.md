# Sensor Models

Simulated SSN ground sensor producing az/el/range (radar) or angles-only az/el
(optical) measurements.

---

::: brahe.SimpleSSNSensor
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [SSN Sensor Datasets Guide](../../learn/datasets/ssn_sensors.md) - Loading sensor sites
- [Measurement Models Guide](../../learn/estimation/measurement_models.md#azimuthelevationrange) - The matching AzElRangeMeasurementModel / AzElMeasurementModel
- [SSN Radar Tracking Example](../../examples/ssn_tracking.md) - Full EKF/UKF/BLS walkthrough
- [Measurement Models](measurement_models.md) - Filter-side measurement models
