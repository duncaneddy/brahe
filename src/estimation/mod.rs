/*!
 * Estimation module for orbit determination and state estimation.
 *
 * Provides extensible estimation filters for processing measurements and
 * determining spacecraft state. Supports multiple estimator types and
 * user-defined measurement models.
 *
 * # Estimator Types
 *
 * - **Extended Kalman Filter (EKF)**: Sequential filter using linearized
 *   dynamics and measurement models. Leverages the propagator's built-in
 *   STM for the prediction step.
 *
 * - **Unscented Kalman Filter (UKF)**: Sequential filter using sigma points
 *   to capture nonlinear behavior without linearization.
 *
 * - **Batch Least Squares (BLS)**: Accumulates all measurements and iteratively
 *   solves for the state correction. Supports weighted and constrained modes.
 *
 * # Measurement Models
 *
 * Built-in models are organized by reference frame:
 *
 * **Inertial (ECI) frame** — direct state observations:
 * - [`InertialPositionMeasurementModel`]: 3D inertial position (meters)
 * - [`InertialVelocityMeasurementModel`]: 3D inertial velocity (m/s)
 * - [`InertialStateMeasurementModel`]: 6D inertial state
 *
 * **GNSS (ECEF) frame** — receiver outputs with ECI→ECEF conversion:
 * - [`ECEFPositionMeasurementModel`]: 3D ECEF position (meters)
 * - [`ECEFVelocityMeasurementModel`]: 3D ECEF velocity (m/s)
 * - [`ECEFStateMeasurementModel`]: 6D ECEF state
 *
 * **Topocentric (ENZ) frame** — ground-sensor observations:
 * - [`AzElRangeMeasurementModel`]: Azimuth/elevation/range for radar sensors
 * - [`AzElMeasurementModel`]: Angles-only azimuth/elevation for optical sensors
 *
 * Custom models can be defined by implementing the [`MeasurementModel`] trait
 * in Rust or by subclassing `MeasurementModel` in Python.
 */

mod bls;
mod config;
mod dynamics_source;
mod ekf;
mod measurement;
mod sensors;
mod traits;
mod types;
mod ukf;

pub use bls::BatchLeastSquares;
pub use config::{
    BLSConfig, BLSSolverMethod, ConsiderParameterConfig, EKFConfig, ProcessNoiseConfig, UKFConfig,
};
pub use dynamics_source::DynamicsSource;
pub use ekf::ExtendedKalmanFilter;
pub use measurement::{
    AzElMeasurementModel, AzElRangeMeasurementModel, ECEFPositionMeasurementModel,
    ECEFStateMeasurementModel, ECEFVelocityMeasurementModel, InertialPositionMeasurementModel,
    InertialStateMeasurementModel, InertialVelocityMeasurementModel,
};
pub use sensors::{SensorType, SimpleSSNSensor};
pub use traits::MeasurementModel;
pub use traits::measurement_jacobian_numerical;
pub use types::{BLSIterationRecord, BLSObservationResidual, FilterRecord, Observation};
pub use ukf::UnscentedKalmanFilter;
