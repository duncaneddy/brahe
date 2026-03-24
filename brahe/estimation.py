"""
Estimation Module

State estimation filters and measurement models for orbit determination.

**Filters:**
- ExtendedKalmanFilter: Sequential filter using linearized dynamics and measurements

**Measurement Models (Inertial):**
- InertialPositionMeasurementModel: 3D ECI position observations
- InertialVelocityMeasurementModel: 3D ECI velocity observations
- InertialStateMeasurementModel: 6D ECI state observations

**Measurement Models (ECEF):**
- ECEFPositionMeasurementModel: 3D ECEF position observations
- ECEFVelocityMeasurementModel: 3D ECEF velocity observations
- ECEFStateMeasurementModel: 6D ECEF state observations

**Custom Measurement Models:**
- MeasurementModel: Base class for Python-defined measurement models

**Configuration:**
- EKFConfig: EKF configuration
- ProcessNoiseConfig: Process noise specification

**Data Types:**
- Observation: Single measurement at an epoch
- FilterRecord: Record of a filter update step
- EstimationResult: Complete estimation run results
"""

from brahe._brahe import (
    # Filters
    ExtendedKalmanFilter,
    # Base class for custom models
    MeasurementModel,
    # Built-in measurement models
    InertialPositionMeasurementModel,
    InertialVelocityMeasurementModel,
    InertialStateMeasurementModel,
    ECEFPositionMeasurementModel,
    ECEFVelocityMeasurementModel,
    ECEFStateMeasurementModel,
    # Configuration
    EKFConfig,
    ProcessNoiseConfig,
    # Data types
    Observation,
    FilterRecord,
)

__all__ = [
    "ExtendedKalmanFilter",
    "MeasurementModel",
    "InertialPositionMeasurementModel",
    "InertialVelocityMeasurementModel",
    "InertialStateMeasurementModel",
    "ECEFPositionMeasurementModel",
    "ECEFVelocityMeasurementModel",
    "ECEFStateMeasurementModel",
    "EKFConfig",
    "ProcessNoiseConfig",
    "Observation",
    "FilterRecord",
]
