# Measurement Models

Built-in measurement models for GPS-like observations in inertial and ECEF
frames, and topocentric ground-sensor observations.

## Inertial Frame Models

---

::: brahe.InertialPositionMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.InertialVelocityMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.InertialStateMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

## ECEF Frame Models

---

::: brahe.ECEFPositionMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ECEFVelocityMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ECEFStateMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

## Topocentric Ground-Sensor Models

Topocentric az/el/range (radar) and angles-only az/el (optical) models for
fixed ground stations. Both convert the ECI state to the station's ENZ frame
and wrap the azimuth residual across 0/360.

---

::: brahe.AzElRangeMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.AzElMeasurementModel
    options:
      show_root_heading: true
      show_root_full_path: false

## Covariance Matrix Helpers

Utility functions for constructing noise covariance matrices, usable with both built-in and custom measurement models.

---

::: brahe.isotropic_covariance
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.diagonal_covariance
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [Measurement Models Guide](../../learn/estimation/measurement_models.md) - Usage guidance and model selection
- [Custom Models Guide](../../learn/estimation/custom_models.md) - Defining custom models in Python
- [Covariance Matrix Utilities](../mathematics/covariance.md) - Full covariance utility reference
