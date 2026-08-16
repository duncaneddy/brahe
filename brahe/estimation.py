"""
Estimation Module

State estimation filters and measurement models for orbit determination.

**Filters:**
- ExtendedKalmanFilter: Sequential filter using linearized dynamics and measurements
- ExtendedKalmanFilterBuilder: Builder for ExtendedKalmanFilter
- UnscentedKalmanFilter: Sequential filter using sigma points (no linearization)
- UnscentedKalmanFilterBuilder: Builder for UnscentedKalmanFilter
- BatchLeastSquares: Batch estimator using iterative Gauss-Newton
- BatchLeastSquaresBuilder: Builder for BatchLeastSquares

**Measurement Models (Inertial):**
- InertialPositionMeasurementModel: 3D ECI position observations
- InertialVelocityMeasurementModel: 3D ECI velocity observations
- InertialStateMeasurementModel: 6D ECI state observations

**Measurement Models (ECEF):**
- ECEFPositionMeasurementModel: 3D ECEF position observations
- ECEFVelocityMeasurementModel: 3D ECEF velocity observations
- ECEFStateMeasurementModel: 6D ECEF state observations

**Measurement Models (Ground Sensors):**
- AzElRangeMeasurementModel: Topocentric azimuth/elevation/range observations (radar)
- AzElMeasurementModel: Topocentric angles-only azimuth/elevation observations (optical)

**Sensors:**
- SensorType: Sensor measurement type (AZEL_RANGE radar or AZEL optical)
- SimpleSSNSensor: Simulated SSN ground sensor producing az/el/range or az/el measurements

**Custom Measurement Models:**
- MeasurementModel: Base class for Python-defined measurement models

**Configuration:**
- EKFConfig: EKF configuration
- UKFConfig: UKF configuration
- BLSConfig: Batch Least Squares configuration
- ProcessNoiseConfig: Process noise specification

**Data Types:**
- Observation: Single measurement at an epoch
- FilterRecord: Record of a filter update step
- BLSIterationRecord: Record of a BLS iteration
- BLSObservationResidual: Per-observation residual from BLS
"""

from brahe._brahe import (
    AzElMeasurementModel,
    AzElRangeMeasurementModel,
    BatchLeastSquares,
    BatchLeastSquaresBuilder,
    BLSConfig,
    BLSIterationRecord,
    BLSObservationResidual,
    BLSSolverMethod,
    ConsiderParameterConfig,
    ECEFPositionMeasurementModel,
    ECEFStateMeasurementModel,
    ECEFVelocityMeasurementModel,
    # Configuration
    EKFConfig,
    # Filters
    ExtendedKalmanFilter,
    ExtendedKalmanFilterBuilder,
    FilterRecord,
    # Built-in measurement models
    InertialPositionMeasurementModel,
    InertialStateMeasurementModel,
    InertialVelocityMeasurementModel,
    # Base class for custom models
    MeasurementModel,
    # Data types
    Observation,
    ProcessNoiseConfig,
    # Sensors
    SensorType,
    SimpleSSNSensor,
    UKFConfig,
    UnscentedKalmanFilter,
    UnscentedKalmanFilterBuilder,
    diagonal_covariance,
    # Covariance matrix helpers
    isotropic_covariance,
)

__all__ = [
    "AzElMeasurementModel",
    "AzElRangeMeasurementModel",
    "BLSConfig",
    "BLSIterationRecord",
    "BLSObservationResidual",
    "BLSSolverMethod",
    "BatchLeastSquares",
    "BatchLeastSquaresBuilder",
    "ConsiderParameterConfig",
    "ECEFPositionMeasurementModel",
    "ECEFStateMeasurementModel",
    "ECEFVelocityMeasurementModel",
    "EKFConfig",
    "ExtendedKalmanFilter",
    "ExtendedKalmanFilterBuilder",
    "FilterRecord",
    "InertialPositionMeasurementModel",
    "InertialStateMeasurementModel",
    "InertialVelocityMeasurementModel",
    "MeasurementModel",
    "Observation",
    "ProcessNoiseConfig",
    "SensorType",
    "SimpleSSNSensor",
    "UKFConfig",
    "UnscentedKalmanFilter",
    "UnscentedKalmanFilterBuilder",
    "diagonal_covariance",
    "isotropic_covariance",
]
