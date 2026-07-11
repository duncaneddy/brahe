"""
Estimation Module

State estimation filters and measurement models for orbit determination.

**Filters:**
- ExtendedKalmanFilter: Sequential filter using linearized dynamics and measurements
- UnscentedKalmanFilter: Sequential filter using sigma points (no linearization)
- BatchLeastSquares: Batch estimator using iterative Gauss-Newton

**Measurement Models (Inertial):**
- InertialPositionMeasurementModel: 3D ECI position observations
- InertialVelocityMeasurementModel: 3D ECI velocity observations
- InertialStateMeasurementModel: 6D ECI state observations

**Measurement Models (ECEF):**
- EcefPositionMeasurementModel: 3D ECEF position observations
- EcefVelocityMeasurementModel: 3D ECEF velocity observations
- EcefStateMeasurementModel: 6D ECEF state observations

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
    # Filters
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    BatchLeastSquares,
    # Base class for custom models
    MeasurementModel,
    # Built-in measurement models
    InertialPositionMeasurementModel,
    InertialVelocityMeasurementModel,
    InertialStateMeasurementModel,
    EcefPositionMeasurementModel,
    EcefVelocityMeasurementModel,
    EcefStateMeasurementModel,
    # Covariance matrix helpers
    isotropic_covariance,
    diagonal_covariance,
    # Configuration
    EKFConfig,
    UKFConfig,
    BLSConfig,
    BLSSolverMethod,
    ConsiderParameterConfig,
    ProcessNoiseConfig,
    # Data types
    Observation,
    FilterRecord,
    BLSIterationRecord,
    BLSObservationResidual,
)

__all__ = [
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "BatchLeastSquares",
    "MeasurementModel",
    "InertialPositionMeasurementModel",
    "InertialVelocityMeasurementModel",
    "InertialStateMeasurementModel",
    "EcefPositionMeasurementModel",
    "EcefVelocityMeasurementModel",
    "EcefStateMeasurementModel",
    "isotropic_covariance",
    "diagonal_covariance",
    "EKFConfig",
    "UKFConfig",
    "BLSConfig",
    "BLSSolverMethod",
    "ConsiderParameterConfig",
    "ProcessNoiseConfig",
    "Observation",
    "FilterRecord",
    "BLSIterationRecord",
    "BLSObservationResidual",
]
