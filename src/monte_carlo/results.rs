/*!
 * Result types and statistical analysis for Monte Carlo simulations.
 *
 * Provides types for collecting simulation outputs and computing statistics
 * (mean, standard deviation, percentiles, covariance) across Monte Carlo runs.
 */

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use crate::time::Epoch;
use crate::utils::BraheError;

use super::config::MonteCarloConfig;
use super::variables::{MonteCarloSampledParameters, MonteCarloSampledValue, MonteCarloVariableId};

/// Output value from a single simulation run.
///
/// Represents a named output quantity that can be a scalar, vector, or an
/// optional scalar (for quantities that may not be computed in every run).
#[derive(Clone, Debug)]
pub enum MonteCarloOutputValue {
    /// A single floating-point output value.
    Scalar(f64),
    /// A vector output value (e.g., position, velocity).
    Vector(DVector<f64>),
    /// An optional scalar that may or may not have been computed.
    /// Runs where this is `None` are excluded from statistics.
    OptionalScalar(Option<f64>),
}

/// Outputs from a single Monte Carlo simulation run.
///
/// A named collection of output values produced by a simulation function.
/// Each output is identified by a string name (e.g., "final_altitude",
/// "total_delta_v", "final_position").
#[derive(Clone, Debug)]
pub struct MonteCarloOutputs {
    /// Map from output name to its value.
    pub values: HashMap<String, MonteCarloOutputValue>,
}

impl MonteCarloOutputs {
    /// Creates an empty set of outputs.
    ///
    /// # Returns
    ///
    /// A new `MonteCarloOutputs` with no values
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Inserts a scalar output value.
    ///
    /// # Arguments
    ///
    /// * `name` - Name identifying this output
    /// * `value` - The scalar value in SI units
    pub fn insert_scalar(&mut self, name: &str, value: f64) {
        self.values
            .insert(name.to_string(), MonteCarloOutputValue::Scalar(value));
    }

    /// Inserts a vector output value.
    ///
    /// # Arguments
    ///
    /// * `name` - Name identifying this output
    /// * `value` - The vector value in SI units
    pub fn insert_vector(&mut self, name: &str, value: DVector<f64>) {
        self.values
            .insert(name.to_string(), MonteCarloOutputValue::Vector(value));
    }

    /// Inserts an optional scalar output value.
    ///
    /// # Arguments
    ///
    /// * `name` - Name identifying this output
    /// * `value` - The optional scalar value in SI units, or `None` if not computed
    pub fn insert_optional_scalar(&mut self, name: &str, value: Option<f64>) {
        self.values.insert(
            name.to_string(),
            MonteCarloOutputValue::OptionalScalar(value),
        );
    }

    /// Returns a reference to the output value with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the output to retrieve
    ///
    /// # Returns
    ///
    /// Reference to the output value, or `None` if not found
    pub fn get(&self, name: &str) -> Option<&MonteCarloOutputValue> {
        self.values.get(name)
    }
}

impl Default for MonteCarloOutputs {
    fn default() -> Self {
        Self::new()
    }
}

/// Reason why a single Monte Carlo simulation run terminated.
#[derive(Clone, Debug)]
pub enum MonteCarloTerminationReason {
    /// The simulation reached the configured target epoch.
    ReachedTargetEpoch,
    /// A terminal event triggered early termination.
    TerminalEvent {
        /// Name of the event that triggered termination.
        event_name: String,
        /// Epoch at which the event occurred.
        epoch: Epoch,
    },
    /// The simulation encountered an error during execution.
    Error,
    /// The simulation was terminated by user logic.
    UserTerminated {
        /// Description of why the user terminated the run.
        reason: String,
    },
}

/// Data from a single Monte Carlo simulation run.
///
/// Contains the sampled input parameters, the simulation result (success or
/// failure), and the reason the run terminated.
#[derive(Debug)]
pub struct MonteCarloRun {
    /// Zero-based index of this run in the simulation.
    pub run_index: usize,
    /// The sampled input parameters used for this run.
    pub sampled_parameters: MonteCarloSampledParameters,
    /// The simulation result: outputs on success, error on failure.
    pub result: Result<MonteCarloOutputs, BraheError>,
    /// Why this run terminated.
    pub termination: MonteCarloTerminationReason,
}

impl MonteCarloRun {
    /// Returns whether this run completed successfully.
    ///
    /// # Returns
    ///
    /// `true` if the run produced outputs without error
    pub fn succeeded(&self) -> bool {
        self.result.is_ok()
    }

    /// Returns the outputs from this run, if it succeeded.
    ///
    /// # Returns
    ///
    /// Reference to the outputs, or `None` if the run failed
    pub fn outputs(&self) -> Option<&MonteCarloOutputs> {
        self.result.as_ref().ok()
    }
}

/// Results from a complete Monte Carlo simulation.
///
/// Provides access to individual run data and statistical analysis methods
/// for computing means, standard deviations, percentiles, and covariances
/// across all successful runs.
#[derive(Debug)]
pub struct MonteCarloResults {
    /// Configuration used for the simulation.
    pub config: MonteCarloConfig,
    /// All simulation runs (both successful and failed).
    pub runs: Vec<MonteCarloRun>,
    /// Master seed used for reproducibility.
    pub master_seed: u64,
    /// Whether the simulation converged (if convergence checking was enabled).
    pub converged: Option<bool>,
    /// Final standard errors for monitored outputs at simulation completion.
    pub final_standard_errors: Option<HashMap<String, f64>>,
}

impl MonteCarloResults {
    /// Creates a new results collection from completed simulation runs.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration used for the simulation
    /// * `runs` - All completed simulation runs
    /// * `master_seed` - The master seed used for reproducibility
    ///
    /// # Returns
    ///
    /// A new `MonteCarloResults`
    pub fn new(config: MonteCarloConfig, runs: Vec<MonteCarloRun>, master_seed: u64) -> Self {
        Self {
            config,
            runs,
            master_seed,
            converged: None,
            final_standard_errors: None,
        }
    }

    /// Collects scalar values for a named output from all successful runs.
    ///
    /// Extracts `f64` values from `Scalar` variants and `Some` values from
    /// `OptionalScalar` variants. `None` optional scalars and non-scalar
    /// outputs are excluded.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the output to collect
    ///
    /// # Returns
    ///
    /// Vector of scalar values from successful runs
    pub fn scalar_values(&self, name: &str) -> Vec<f64> {
        self.runs
            .iter()
            .filter_map(|run| {
                run.result.as_ref().ok().and_then(|outputs| {
                    outputs.get(name).and_then(|val| match val {
                        MonteCarloOutputValue::Scalar(v) => Some(*v),
                        MonteCarloOutputValue::OptionalScalar(Some(v)) => Some(*v),
                        _ => None,
                    })
                })
            })
            .collect()
    }

    /// Computes the arithmetic mean of a scalar output across successful runs.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar output
    ///
    /// # Returns
    ///
    /// The mean value, or an error if no values are available
    pub fn mean(&self, name: &str) -> Result<f64, BraheError> {
        let values = self.scalar_values(name);
        if values.is_empty() {
            return Err(BraheError::NumericalError(format!(
                "No scalar values found for output '{}'",
                name
            )));
        }
        let sum: f64 = values.iter().sum();
        Ok(sum / values.len() as f64)
    }

    /// Computes the sample standard deviation (ddof=1) of a scalar output.
    ///
    /// Uses the unbiased estimator: sqrt(sum((x - mean)^2) / (n - 1)).
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar output
    ///
    /// # Returns
    ///
    /// The sample standard deviation, or an error if fewer than 2 values
    pub fn std(&self, name: &str) -> Result<f64, BraheError> {
        let values = self.scalar_values(name);
        if values.len() < 2 {
            return Err(BraheError::NumericalError(format!(
                "Need at least 2 values for standard deviation of '{}', found {}",
                name,
                values.len()
            )));
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        Ok(variance.sqrt())
    }

    /// Computes the standard error of the mean for a scalar output.
    ///
    /// Standard error = sample std / sqrt(n), which estimates how precisely
    /// the population mean is known from the sample.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar output
    ///
    /// # Returns
    ///
    /// The standard error, or an error if fewer than 2 values
    pub fn standard_error(&self, name: &str) -> Result<f64, BraheError> {
        let values = self.scalar_values(name);
        let n = values.len();
        if n < 2 {
            return Err(BraheError::NumericalError(format!(
                "Need at least 2 values for standard error of '{}', found {}",
                name, n
            )));
        }
        let std_dev = self.std(name)?;
        Ok(std_dev / (n as f64).sqrt())
    }

    /// Computes a percentile of a scalar output using linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar output
    /// * `p` - Percentile in the range [0.0, 1.0] (e.g., 0.5 for median)
    ///
    /// # Returns
    ///
    /// The interpolated percentile value, or an error if no values or invalid percentile
    pub fn percentile(&self, name: &str, p: f64) -> Result<f64, BraheError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(BraheError::OutOfBoundsError(format!(
                "Percentile must be in [0.0, 1.0], got {}",
                p
            )));
        }
        let mut values = self.scalar_values(name);
        if values.is_empty() {
            return Err(BraheError::NumericalError(format!(
                "No scalar values found for output '{}'",
                name
            )));
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len();
        if n == 1 {
            return Ok(values[0]);
        }

        let index = p * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let frac = index - lower as f64;

        if lower == upper {
            Ok(values[lower])
        } else {
            Ok(values[lower] * (1.0 - frac) + values[upper] * frac)
        }
    }

    /// Returns the minimum value of a scalar output across successful runs.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar output
    ///
    /// # Returns
    ///
    /// The minimum value, or an error if no values are available
    pub fn min(&self, name: &str) -> Result<f64, BraheError> {
        let values = self.scalar_values(name);
        values
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| {
                BraheError::NumericalError(format!("No scalar values found for output '{}'", name))
            })
    }

    /// Returns the maximum value of a scalar output across successful runs.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the scalar output
    ///
    /// # Returns
    ///
    /// The maximum value, or an error if no values are available
    pub fn max(&self, name: &str) -> Result<f64, BraheError> {
        let values = self.scalar_values(name);
        values
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| {
                BraheError::NumericalError(format!("No scalar values found for output '{}'", name))
            })
    }

    /// Computes the element-wise mean vector of a vector output across successful runs.
    ///
    /// All vector outputs with the given name must have the same dimension.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the vector output
    ///
    /// # Returns
    ///
    /// The mean vector, or an error if no vectors are available or dimensions mismatch
    pub fn mean_vector(&self, name: &str) -> Result<DVector<f64>, BraheError> {
        let vectors: Vec<&DVector<f64>> = self
            .runs
            .iter()
            .filter_map(|run| {
                run.result.as_ref().ok().and_then(|outputs| {
                    outputs.get(name).and_then(|val| match val {
                        MonteCarloOutputValue::Vector(v) => Some(v),
                        _ => None,
                    })
                })
            })
            .collect();

        if vectors.is_empty() {
            return Err(BraheError::NumericalError(format!(
                "No vector values found for output '{}'",
                name
            )));
        }

        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(BraheError::NumericalError(format!(
                    "Vector dimension mismatch for output '{}': run 0 has dimension {}, run {} has dimension {}",
                    name,
                    dim,
                    i,
                    v.len()
                )));
            }
        }

        let mut sum = DVector::zeros(dim);
        for v in &vectors {
            sum += *v;
        }
        Ok(sum / vectors.len() as f64)
    }

    /// Computes the sample covariance matrix of a vector output across successful runs.
    ///
    /// Uses the unbiased estimator: (1/(n-1)) * sum((x - mean)(x - mean)^T).
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the vector output
    ///
    /// # Returns
    ///
    /// The covariance matrix (dim x dim), or an error if fewer than 2 vectors
    pub fn covariance(&self, name: &str) -> Result<DMatrix<f64>, BraheError> {
        let vectors: Vec<&DVector<f64>> = self
            .runs
            .iter()
            .filter_map(|run| {
                run.result.as_ref().ok().and_then(|outputs| {
                    outputs.get(name).and_then(|val| match val {
                        MonteCarloOutputValue::Vector(v) => Some(v),
                        _ => None,
                    })
                })
            })
            .collect();

        if vectors.len() < 2 {
            return Err(BraheError::NumericalError(format!(
                "Need at least 2 vectors for covariance of '{}', found {}",
                name,
                vectors.len()
            )));
        }

        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(BraheError::NumericalError(format!(
                    "Vector dimension mismatch for output '{}': run 0 has dimension {}, run {} has dimension {}",
                    name,
                    dim,
                    i,
                    v.len()
                )));
            }
        }

        let mean = self.mean_vector(name)?;
        let n = vectors.len();
        let mut cov = DMatrix::zeros(dim, dim);
        for v in &vectors {
            let diff = *v - &mean;
            cov += &diff * diff.transpose();
        }
        Ok(cov / (n - 1) as f64)
    }

    /// Computes the mean duration for a named output.
    ///
    /// This is an alias for `mean()` since durations are stored as `f64` seconds.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the duration output (stored as seconds)
    ///
    /// # Returns
    ///
    /// The mean duration in seconds, or an error if no values are available
    pub fn mean_duration(&self, name: &str) -> Result<f64, BraheError> {
        self.mean(name)
    }

    /// Returns the number of runs that completed successfully.
    ///
    /// # Returns
    ///
    /// Count of successful runs
    pub fn num_successful(&self) -> usize {
        self.runs.iter().filter(|r| r.result.is_ok()).count()
    }

    /// Returns the number of runs that failed.
    ///
    /// # Returns
    ///
    /// Count of failed runs
    pub fn num_failed(&self) -> usize {
        self.runs.iter().filter(|r| r.result.is_err()).count()
    }

    /// Returns references to all successful runs.
    ///
    /// # Returns
    ///
    /// Vector of references to runs that completed without error
    pub fn successful_runs(&self) -> Vec<&MonteCarloRun> {
        self.runs.iter().filter(|r| r.result.is_ok()).collect()
    }

    /// Collects sampled input values for a specific variable across all runs.
    ///
    /// # Arguments
    ///
    /// * `id` - The variable identifier to look up
    ///
    /// # Returns
    ///
    /// Vector of references to the sampled values from each run that has this variable
    pub fn input_values(&self, id: &MonteCarloVariableId) -> Vec<&MonteCarloSampledValue> {
        self.runs
            .iter()
            .filter_map(|run| run.sampled_parameters.get(id))
            .collect()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Helper to create a successful MonteCarloRun with scalar outputs.
    fn make_successful_run(run_index: usize, outputs: Vec<(&str, f64)>) -> MonteCarloRun {
        let mut mc_outputs = MonteCarloOutputs::new();
        for (name, value) in outputs {
            mc_outputs.insert_scalar(name, value);
        }
        MonteCarloRun {
            run_index,
            sampled_parameters: MonteCarloSampledParameters::new(run_index, run_index as u64),
            result: Ok(mc_outputs),
            termination: MonteCarloTerminationReason::ReachedTargetEpoch,
        }
    }

    /// Helper to create a failed MonteCarloRun.
    fn make_failed_run(run_index: usize) -> MonteCarloRun {
        MonteCarloRun {
            run_index,
            sampled_parameters: MonteCarloSampledParameters::new(run_index, run_index as u64),
            result: Err(BraheError::PropagatorError("diverged".to_string())),
            termination: MonteCarloTerminationReason::Error,
        }
    }

    /// Helper to create a successful MonteCarloRun with vector outputs.
    fn make_vector_run(run_index: usize, name: &str, values: Vec<f64>) -> MonteCarloRun {
        let mut mc_outputs = MonteCarloOutputs::new();
        mc_outputs.insert_vector(name, DVector::from_vec(values));
        MonteCarloRun {
            run_index,
            sampled_parameters: MonteCarloSampledParameters::new(run_index, run_index as u64),
            result: Ok(mc_outputs),
            termination: MonteCarloTerminationReason::ReachedTargetEpoch,
        }
    }

    /// Helper to create MonteCarloResults from a set of runs.
    fn make_results(runs: Vec<MonteCarloRun>) -> MonteCarloResults {
        MonteCarloResults::new(MonteCarloConfig::default(), runs, 42)
    }

    // ---- MonteCarloOutputValue tests ----

    #[test]
    fn test_output_value_scalar() {
        let val = MonteCarloOutputValue::Scalar(3.125);
        match val {
            MonteCarloOutputValue::Scalar(v) => assert!((v - 3.125).abs() < 1e-15),
            _ => panic!("Expected Scalar variant"),
        }
    }

    #[test]
    fn test_output_value_vector() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let val = MonteCarloOutputValue::Vector(vec.clone());
        match val {
            MonteCarloOutputValue::Vector(v) => assert_eq!(v.len(), 3),
            _ => panic!("Expected Vector variant"),
        }
    }

    #[test]
    fn test_output_value_optional_scalar() {
        let some_val = MonteCarloOutputValue::OptionalScalar(Some(5.0));
        let none_val = MonteCarloOutputValue::OptionalScalar(None);
        match some_val {
            MonteCarloOutputValue::OptionalScalar(Some(v)) => assert!((v - 5.0).abs() < 1e-15),
            _ => panic!("Expected OptionalScalar(Some)"),
        }
        match none_val {
            MonteCarloOutputValue::OptionalScalar(None) => {}
            _ => panic!("Expected OptionalScalar(None)"),
        }
    }

    #[test]
    fn test_output_value_clone_debug() {
        let val = MonteCarloOutputValue::Scalar(1.0);
        let cloned = val.clone();
        let _ = format!("{:?}", cloned);
    }

    // ---- MonteCarloOutputs tests ----

    #[test]
    fn test_outputs_new() {
        let outputs = MonteCarloOutputs::new();
        assert!(outputs.values.is_empty());
    }

    #[test]
    fn test_outputs_default() {
        let outputs = MonteCarloOutputs::default();
        assert!(outputs.values.is_empty());
    }

    #[test]
    fn test_outputs_insert_and_get_scalar() {
        let mut outputs = MonteCarloOutputs::new();
        outputs.insert_scalar("altitude", 500e3);
        let val = outputs.get("altitude").unwrap();
        match val {
            MonteCarloOutputValue::Scalar(v) => assert!((v - 500e3).abs() < 1e-10),
            _ => panic!("Expected Scalar"),
        }
    }

    #[test]
    fn test_outputs_insert_and_get_vector() {
        let mut outputs = MonteCarloOutputs::new();
        let pos = DVector::from_vec(vec![7000e3, 0.0, 0.0]);
        outputs.insert_vector("position", pos);
        let val = outputs.get("position").unwrap();
        match val {
            MonteCarloOutputValue::Vector(v) => assert_eq!(v.len(), 3),
            _ => panic!("Expected Vector"),
        }
    }

    #[test]
    fn test_outputs_insert_and_get_optional_scalar() {
        let mut outputs = MonteCarloOutputs::new();
        outputs.insert_optional_scalar("contact_duration", Some(300.0));
        outputs.insert_optional_scalar("no_contact", None);

        match outputs.get("contact_duration").unwrap() {
            MonteCarloOutputValue::OptionalScalar(Some(v)) => assert!((v - 300.0).abs() < 1e-10),
            _ => panic!("Expected OptionalScalar(Some)"),
        }
        match outputs.get("no_contact").unwrap() {
            MonteCarloOutputValue::OptionalScalar(None) => {}
            _ => panic!("Expected OptionalScalar(None)"),
        }
    }

    #[test]
    fn test_outputs_get_missing() {
        let outputs = MonteCarloOutputs::new();
        assert!(outputs.get("nonexistent").is_none());
    }

    // ---- MonteCarloTerminationReason tests ----

    #[test]
    fn test_termination_reached_target() {
        let reason = MonteCarloTerminationReason::ReachedTargetEpoch;
        let _ = format!("{:?}", reason);
    }

    #[test]
    fn test_termination_terminal_event() {
        let epoch = Epoch::from_mjd(59000.0, crate::time::TimeSystem::UTC);
        let reason = MonteCarloTerminationReason::TerminalEvent {
            event_name: "reentry".to_string(),
            epoch,
        };
        match &reason {
            MonteCarloTerminationReason::TerminalEvent { event_name, .. } => {
                assert_eq!(event_name, "reentry");
            }
            _ => panic!("Expected TerminalEvent"),
        }
        let _ = format!("{:?}", reason);
    }

    #[test]
    fn test_termination_error() {
        let reason = MonteCarloTerminationReason::Error;
        let _ = format!("{:?}", reason);
    }

    #[test]
    fn test_termination_user_terminated() {
        let reason = MonteCarloTerminationReason::UserTerminated {
            reason: "constraint violated".to_string(),
        };
        match &reason {
            MonteCarloTerminationReason::UserTerminated { reason } => {
                assert_eq!(reason, "constraint violated");
            }
            _ => panic!("Expected UserTerminated"),
        }
    }

    #[test]
    fn test_termination_clone() {
        let reason = MonteCarloTerminationReason::ReachedTargetEpoch;
        let _cloned = reason.clone();
    }

    // ---- MonteCarloRun tests ----

    #[test]
    fn test_run_succeeded() {
        let run = make_successful_run(0, vec![("alt", 500e3)]);
        assert!(run.succeeded());
    }

    #[test]
    fn test_run_failed() {
        let run = make_failed_run(0);
        assert!(!run.succeeded());
    }

    #[test]
    fn test_run_outputs_success() {
        let run = make_successful_run(0, vec![("alt", 500e3)]);
        let outputs = run.outputs().unwrap();
        assert!(outputs.get("alt").is_some());
    }

    #[test]
    fn test_run_outputs_failure() {
        let run = make_failed_run(0);
        assert!(run.outputs().is_none());
    }

    #[test]
    fn test_run_debug() {
        let run = make_successful_run(0, vec![("alt", 500e3)]);
        let _ = format!("{:?}", run);
    }

    // ---- MonteCarloResults constructor tests ----

    #[test]
    fn test_results_new() {
        let results = make_results(vec![]);
        assert_eq!(results.master_seed, 42);
        assert!(results.runs.is_empty());
        assert!(results.converged.is_none());
        assert!(results.final_standard_errors.is_none());
    }

    // ---- scalar_values tests ----

    #[test]
    fn test_scalar_values_basic() {
        let runs = vec![
            make_successful_run(0, vec![("alt", 100.0)]),
            make_successful_run(1, vec![("alt", 200.0)]),
            make_successful_run(2, vec![("alt", 300.0)]),
        ];
        let results = make_results(runs);
        let values = results.scalar_values("alt");
        assert_eq!(values.len(), 3);
        assert!(values.contains(&100.0));
        assert!(values.contains(&200.0));
        assert!(values.contains(&300.0));
    }

    #[test]
    fn test_scalar_values_skips_failed_runs() {
        let runs = vec![
            make_successful_run(0, vec![("alt", 100.0)]),
            make_failed_run(1),
            make_successful_run(2, vec![("alt", 300.0)]),
        ];
        let results = make_results(runs);
        let values = results.scalar_values("alt");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_scalar_values_optional_scalar_some_and_none() {
        let mut outputs1 = MonteCarloOutputs::new();
        outputs1.insert_optional_scalar("duration", Some(100.0));
        let mut outputs2 = MonteCarloOutputs::new();
        outputs2.insert_optional_scalar("duration", None);
        let mut outputs3 = MonteCarloOutputs::new();
        outputs3.insert_optional_scalar("duration", Some(200.0));

        let runs = vec![
            MonteCarloRun {
                run_index: 0,
                sampled_parameters: MonteCarloSampledParameters::new(0, 0),
                result: Ok(outputs1),
                termination: MonteCarloTerminationReason::ReachedTargetEpoch,
            },
            MonteCarloRun {
                run_index: 1,
                sampled_parameters: MonteCarloSampledParameters::new(1, 1),
                result: Ok(outputs2),
                termination: MonteCarloTerminationReason::ReachedTargetEpoch,
            },
            MonteCarloRun {
                run_index: 2,
                sampled_parameters: MonteCarloSampledParameters::new(2, 2),
                result: Ok(outputs3),
                termination: MonteCarloTerminationReason::ReachedTargetEpoch,
            },
        ];

        let results = make_results(runs);
        let values = results.scalar_values("duration");
        // Only the two Some values should be collected, None is skipped
        assert_eq!(values.len(), 2);
        assert!(values.contains(&100.0));
        assert!(values.contains(&200.0));
    }

    #[test]
    fn test_scalar_values_missing_output_name() {
        let runs = vec![make_successful_run(0, vec![("alt", 100.0)])];
        let results = make_results(runs);
        let values = results.scalar_values("nonexistent");
        assert!(values.is_empty());
    }

    #[test]
    fn test_scalar_values_skips_vector_outputs() {
        let runs = vec![make_vector_run(0, "position", vec![1.0, 2.0, 3.0])];
        let results = make_results(runs);
        let values = results.scalar_values("position");
        assert!(values.is_empty());
    }

    // ---- mean tests ----

    #[test]
    fn test_mean_known_values() {
        // values: 2, 4, 6, 8, 10 -> mean = 6
        let runs: Vec<MonteCarloRun> = vec![2.0, 4.0, 6.0, 8.0, 10.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let mean = results.mean("x").unwrap();
        assert!((mean - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_single_value() {
        let runs = vec![make_successful_run(0, vec![("x", 42.0)])];
        let results = make_results(runs);
        let mean = results.mean("x").unwrap();
        assert!((mean - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_no_values() {
        let results = make_results(vec![]);
        let result = results.mean("x");
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_all_failed_runs() {
        let runs = vec![make_failed_run(0), make_failed_run(1)];
        let results = make_results(runs);
        let result = results.mean("x");
        assert!(result.is_err());
    }

    // ---- std tests ----

    #[test]
    fn test_std_known_values() {
        // values: 2, 4, 4, 4, 5, 5, 7, 9
        // mean = 5.0
        // variance = sum((x-5)^2) / 7 = (9+1+1+1+0+0+4+16)/7 = 32/7
        // std = sqrt(32/7) = sqrt(4.571...) ~= 2.13809
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let runs: Vec<MonteCarloRun> = values
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let std_dev = results.std("x").unwrap();
        let expected = (32.0_f64 / 7.0).sqrt();
        assert!((std_dev - expected).abs() < 1e-10);
    }

    #[test]
    fn test_std_identical_values() {
        let runs: Vec<MonteCarloRun> = (0..5)
            .map(|i| make_successful_run(i, vec![("x", 3.0)]))
            .collect();
        let results = make_results(runs);
        let std_dev = results.std("x").unwrap();
        assert!(std_dev.abs() < 1e-15);
    }

    #[test]
    fn test_std_single_value() {
        let runs = vec![make_successful_run(0, vec![("x", 42.0)])];
        let results = make_results(runs);
        let result = results.std("x");
        assert!(result.is_err());
    }

    #[test]
    fn test_std_no_values() {
        let results = make_results(vec![]);
        assert!(results.std("x").is_err());
    }

    // ---- standard_error tests ----

    #[test]
    fn test_standard_error() {
        // SE = std / sqrt(n)
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let n = values.len();
        let runs: Vec<MonteCarloRun> = values
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let se = results.standard_error("x").unwrap();
        let expected_std = (32.0_f64 / 7.0).sqrt();
        let expected_se = expected_std / (n as f64).sqrt();
        assert!((se - expected_se).abs() < 1e-10);
    }

    #[test]
    fn test_standard_error_single_value() {
        let runs = vec![make_successful_run(0, vec![("x", 1.0)])];
        let results = make_results(runs);
        assert!(results.standard_error("x").is_err());
    }

    // ---- percentile tests ----

    #[test]
    fn test_percentile_median() {
        // sorted: 1, 2, 3, 4, 5 -> median (p=0.5) = 3.0
        let runs: Vec<MonteCarloRun> = vec![3.0, 1.0, 5.0, 2.0, 4.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let median = results.percentile("x", 0.5).unwrap();
        assert!((median - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_min_max() {
        // sorted: 1, 2, 3, 4, 5
        let runs: Vec<MonteCarloRun> = vec![3.0, 1.0, 5.0, 2.0, 4.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let p0 = results.percentile("x", 0.0).unwrap();
        let p100 = results.percentile("x", 1.0).unwrap();
        assert!((p0 - 1.0).abs() < 1e-12);
        assert!((p100 - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_interpolation() {
        // sorted: 10, 20, 30, 40
        // p=0.25 -> index = 0.25 * 3 = 0.75 -> 10 * 0.25 + 20 * 0.75 = 17.5
        let runs: Vec<MonteCarloRun> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let p25 = results.percentile("x", 0.25).unwrap();
        assert!((p25 - 17.5).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_single_value() {
        let runs = vec![make_successful_run(0, vec![("x", 7.0)])];
        let results = make_results(runs);
        let median = results.percentile("x", 0.5).unwrap();
        assert!((median - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_invalid_low() {
        let runs = vec![make_successful_run(0, vec![("x", 1.0)])];
        let results = make_results(runs);
        assert!(results.percentile("x", -0.1).is_err());
    }

    #[test]
    fn test_percentile_invalid_high() {
        let runs = vec![make_successful_run(0, vec![("x", 1.0)])];
        let results = make_results(runs);
        assert!(results.percentile("x", 1.1).is_err());
    }

    #[test]
    fn test_percentile_no_values() {
        let results = make_results(vec![]);
        assert!(results.percentile("x", 0.5).is_err());
    }

    // ---- min/max tests ----

    #[test]
    fn test_min() {
        let runs: Vec<MonteCarloRun> = vec![5.0, 2.0, 8.0, 1.0, 9.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let min_val = results.min("x").unwrap();
        assert!((min_val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_max() {
        let runs: Vec<MonteCarloRun> = vec![5.0, 2.0, 8.0, 1.0, 9.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        let max_val = results.max("x").unwrap();
        assert!((max_val - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_min_no_values() {
        let results = make_results(vec![]);
        assert!(results.min("x").is_err());
    }

    #[test]
    fn test_max_no_values() {
        let results = make_results(vec![]);
        assert!(results.max("x").is_err());
    }

    #[test]
    fn test_min_negative_values() {
        let runs: Vec<MonteCarloRun> = vec![-3.0, -1.0, -5.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("x", v)]))
            .collect();
        let results = make_results(runs);
        assert!((results.min("x").unwrap() - (-5.0)).abs() < 1e-12);
        assert!((results.max("x").unwrap() - (-1.0)).abs() < 1e-12);
    }

    // ---- mean_vector tests ----

    #[test]
    fn test_mean_vector() {
        // Vectors: [1, 2], [3, 4], [5, 6] -> mean = [3, 4]
        let runs = vec![
            make_vector_run(0, "pos", vec![1.0, 2.0]),
            make_vector_run(1, "pos", vec![3.0, 4.0]),
            make_vector_run(2, "pos", vec![5.0, 6.0]),
        ];
        let results = make_results(runs);
        let mean = results.mean_vector("pos").unwrap();
        assert_eq!(mean.len(), 2);
        assert!((mean[0] - 3.0).abs() < 1e-12);
        assert!((mean[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_vector_single() {
        let runs = vec![make_vector_run(0, "pos", vec![7.0, 8.0, 9.0])];
        let results = make_results(runs);
        let mean = results.mean_vector("pos").unwrap();
        assert!((mean[0] - 7.0).abs() < 1e-12);
        assert!((mean[1] - 8.0).abs() < 1e-12);
        assert!((mean[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_vector_no_values() {
        let results = make_results(vec![]);
        assert!(results.mean_vector("pos").is_err());
    }

    #[test]
    fn test_mean_vector_skips_failed() {
        let runs = vec![
            make_vector_run(0, "pos", vec![2.0, 4.0]),
            make_failed_run(1),
            make_vector_run(2, "pos", vec![4.0, 6.0]),
        ];
        let results = make_results(runs);
        let mean = results.mean_vector("pos").unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-12);
        assert!((mean[1] - 5.0).abs() < 1e-12);
    }

    // ---- covariance tests ----

    #[test]
    fn test_covariance_known_values() {
        // Vectors: [1, 2], [3, 4], [5, 6]
        // mean = [3, 4]
        // diffs: [-2, -2], [0, 0], [2, 2]
        // cov = (1/2) * ([4,4;4,4] + [0,0;0,0] + [4,4;4,4]) = [4,4;4,4]
        let runs = vec![
            make_vector_run(0, "pos", vec![1.0, 2.0]),
            make_vector_run(1, "pos", vec![3.0, 4.0]),
            make_vector_run(2, "pos", vec![5.0, 6.0]),
        ];
        let results = make_results(runs);
        let cov = results.covariance("pos").unwrap();
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
        assert!((cov[(0, 0)] - 4.0).abs() < 1e-12);
        assert!((cov[(0, 1)] - 4.0).abs() < 1e-12);
        assert!((cov[(1, 0)] - 4.0).abs() < 1e-12);
        assert!((cov[(1, 1)] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_covariance_uncorrelated() {
        // Vectors where components are uncorrelated
        // [1, 0], [0, 1], [-1, 0], [0, -1]
        // mean = [0, 0]
        // cov = (1/3) * ([1,0;0,0] + [0,0;0,1] + [1,0;0,0] + [0,0;0,1])
        //     = (1/3) * [2,0;0,2] = [2/3, 0; 0, 2/3]
        let runs = vec![
            make_vector_run(0, "v", vec![1.0, 0.0]),
            make_vector_run(1, "v", vec![0.0, 1.0]),
            make_vector_run(2, "v", vec![-1.0, 0.0]),
            make_vector_run(3, "v", vec![0.0, -1.0]),
        ];
        let results = make_results(runs);
        let cov = results.covariance("v").unwrap();
        assert!((cov[(0, 0)] - 2.0 / 3.0).abs() < 1e-12);
        assert!((cov[(0, 1)]).abs() < 1e-12);
        assert!((cov[(1, 0)]).abs() < 1e-12);
        assert!((cov[(1, 1)] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_covariance_single_vector() {
        let runs = vec![make_vector_run(0, "pos", vec![1.0, 2.0])];
        let results = make_results(runs);
        assert!(results.covariance("pos").is_err());
    }

    #[test]
    fn test_covariance_no_values() {
        let results = make_results(vec![]);
        assert!(results.covariance("pos").is_err());
    }

    // ---- mean_duration tests ----

    #[test]
    fn test_mean_duration() {
        let runs: Vec<MonteCarloRun> = vec![100.0, 200.0, 300.0]
            .into_iter()
            .enumerate()
            .map(|(i, v)| make_successful_run(i, vec![("contact_time", v)]))
            .collect();
        let results = make_results(runs);
        let mean_dur = results.mean_duration("contact_time").unwrap();
        assert!((mean_dur - 200.0).abs() < 1e-12);
    }

    // ---- num_successful / num_failed tests ----

    #[test]
    fn test_num_successful() {
        let runs = vec![
            make_successful_run(0, vec![("x", 1.0)]),
            make_failed_run(1),
            make_successful_run(2, vec![("x", 3.0)]),
            make_failed_run(3),
            make_successful_run(4, vec![("x", 5.0)]),
        ];
        let results = make_results(runs);
        assert_eq!(results.num_successful(), 3);
        assert_eq!(results.num_failed(), 2);
    }

    #[test]
    fn test_num_successful_all_success() {
        let runs: Vec<MonteCarloRun> = (0..4)
            .map(|i| make_successful_run(i, vec![("x", i as f64)]))
            .collect();
        let results = make_results(runs);
        assert_eq!(results.num_successful(), 4);
        assert_eq!(results.num_failed(), 0);
    }

    #[test]
    fn test_num_successful_all_failed() {
        let runs = vec![make_failed_run(0), make_failed_run(1)];
        let results = make_results(runs);
        assert_eq!(results.num_successful(), 0);
        assert_eq!(results.num_failed(), 2);
    }

    #[test]
    fn test_num_successful_empty() {
        let results = make_results(vec![]);
        assert_eq!(results.num_successful(), 0);
        assert_eq!(results.num_failed(), 0);
    }

    // ---- successful_runs tests ----

    #[test]
    fn test_successful_runs() {
        let runs = vec![
            make_successful_run(0, vec![("x", 1.0)]),
            make_failed_run(1),
            make_successful_run(2, vec![("x", 3.0)]),
        ];
        let results = make_results(runs);
        let successful = results.successful_runs();
        assert_eq!(successful.len(), 2);
        assert_eq!(successful[0].run_index, 0);
        assert_eq!(successful[1].run_index, 2);
    }

    // ---- input_values tests ----

    #[test]
    fn test_input_values() {
        let id = MonteCarloVariableId::Mass;
        let mut params0 = MonteCarloSampledParameters::new(0, 0);
        params0.insert(id.clone(), MonteCarloSampledValue::Scalar(7000e3));
        let mut params1 = MonteCarloSampledParameters::new(1, 1);
        params1.insert(id.clone(), MonteCarloSampledValue::Scalar(7100e3));

        let mut outputs0 = MonteCarloOutputs::new();
        outputs0.insert_scalar("alt", 500e3);
        let mut outputs1 = MonteCarloOutputs::new();
        outputs1.insert_scalar("alt", 600e3);

        let runs = vec![
            MonteCarloRun {
                run_index: 0,
                sampled_parameters: params0,
                result: Ok(outputs0),
                termination: MonteCarloTerminationReason::ReachedTargetEpoch,
            },
            MonteCarloRun {
                run_index: 1,
                sampled_parameters: params1,
                result: Ok(outputs1),
                termination: MonteCarloTerminationReason::ReachedTargetEpoch,
            },
        ];
        let results = make_results(runs);
        let values = results.input_values(&id);
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_input_values_missing_variable() {
        let runs = vec![make_successful_run(0, vec![("x", 1.0)])];
        let results = make_results(runs);
        let id = MonteCarloVariableId::Custom("nonexistent".to_string());
        let values = results.input_values(&id);
        assert!(values.is_empty());
    }

    #[test]
    fn test_input_values_includes_failed_runs() {
        let id = MonteCarloVariableId::DragCoefficient;
        let mut params = MonteCarloSampledParameters::new(0, 0);
        params.insert(id.clone(), MonteCarloSampledValue::Scalar(2.2));

        let run = MonteCarloRun {
            run_index: 0,
            sampled_parameters: params,
            result: Err(BraheError::PropagatorError("failed".to_string())),
            termination: MonteCarloTerminationReason::Error,
        };
        let results = make_results(vec![run]);
        let values = results.input_values(&id);
        // input_values returns values from all runs, including failed ones
        assert_eq!(values.len(), 1);
    }

    // ---- Results debug test ----

    #[test]
    fn test_results_debug() {
        let runs = vec![make_successful_run(0, vec![("x", 1.0)])];
        let results = make_results(runs);
        let _ = format!("{:?}", results);
    }

    // ---- Integration-style tests ----

    #[test]
    fn test_full_statistics_workflow() {
        // Simulate a simple Monte Carlo with known statistics
        // Values: 1, 2, 3, 4, 5
        // mean = 3, std = sqrt(2.5) ~= 1.5811, SE = std/sqrt(5) ~= 0.7071
        let runs: Vec<MonteCarloRun> = (1..=5)
            .map(|i| make_successful_run(i - 1, vec![("x", i as f64)]))
            .collect();
        let results = make_results(runs);

        assert_eq!(results.num_successful(), 5);
        assert_eq!(results.num_failed(), 0);

        let mean = results.mean("x").unwrap();
        assert!((mean - 3.0).abs() < 1e-12);

        let std_dev = results.std("x").unwrap();
        let expected_std = (2.5_f64).sqrt();
        assert!((std_dev - expected_std).abs() < 1e-10);

        let se = results.standard_error("x").unwrap();
        let expected_se = expected_std / (5.0_f64).sqrt();
        assert!((se - expected_se).abs() < 1e-10);

        assert!((results.min("x").unwrap() - 1.0).abs() < 1e-12);
        assert!((results.max("x").unwrap() - 5.0).abs() < 1e-12);

        let median = results.percentile("x", 0.5).unwrap();
        assert!((median - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mixed_success_and_failure() {
        // 3 successful, 2 failed
        let runs = vec![
            make_successful_run(0, vec![("alt", 500e3)]),
            make_failed_run(1),
            make_successful_run(2, vec![("alt", 600e3)]),
            make_failed_run(3),
            make_successful_run(4, vec![("alt", 700e3)]),
        ];
        let results = make_results(runs);

        assert_eq!(results.num_successful(), 3);
        assert_eq!(results.num_failed(), 2);

        // Statistics should only use successful runs
        let mean = results.mean("alt").unwrap();
        assert!((mean - 600e3).abs() < 1e-6);

        let values = results.scalar_values("alt");
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn test_multiple_output_names() {
        let runs: Vec<MonteCarloRun> = (0..3)
            .map(|i| {
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("altitude", (500.0 + i as f64 * 100.0) * 1e3);
                outputs.insert_scalar("velocity", 7000.0 + i as f64 * 100.0);
                MonteCarloRun {
                    run_index: i,
                    sampled_parameters: MonteCarloSampledParameters::new(i, i as u64),
                    result: Ok(outputs),
                    termination: MonteCarloTerminationReason::ReachedTargetEpoch,
                }
            })
            .collect();
        let results = make_results(runs);

        let alt_mean = results.mean("altitude").unwrap();
        let vel_mean = results.mean("velocity").unwrap();
        assert!((alt_mean - 600e3).abs() < 1e-6);
        assert!((vel_mean - 7100.0).abs() < 1e-6);
    }
}
