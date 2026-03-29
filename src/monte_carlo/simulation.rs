/*!
 * Monte Carlo simulation engine.
 *
 * Provides [`MonteCarloSimulation`], the core orchestrator that ties together
 * variable sampling, parallel execution via [rayon], and result collection
 * into [`MonteCarloResults`].
 *
 * # Sampling sources
 *
 * Each variable can be sampled from one of three sources:
 * - [`MonteCarloSamplingSource::Distribution`]: a built-in distribution
 * - [`MonteCarloSamplingSource::PreSampled`]: a pre-computed vector of values
 * - [`MonteCarloSamplingSource::Callback`]: a user-provided closure
 *
 * # Deterministic seeding
 *
 * Per-run seeds are derived deterministically from the master seed and the
 * run index via [`derive_run_seed`], guaranteeing reproducible results
 * regardless of thread scheduling.
 */

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::utils::BraheError;

use super::config::{MonteCarloConfig, MonteCarloStoppingCondition};
use super::distributions::MonteCarloDistribution;
use super::results::{
    MonteCarloOutputs, MonteCarloResults, MonteCarloRun, MonteCarloTerminationReason,
};
use super::variables::{MonteCarloSampledParameters, MonteCarloSampledValue, MonteCarloVariableId};

/// Type alias for the callback sampling function.
pub type MonteCarloSamplerFn =
    Box<dyn Fn(usize, &mut dyn rand::RngCore) -> MonteCarloSampledValue + Send + Sync>;

/// Describes how samples are generated for a single variable.
pub enum MonteCarloSamplingSource {
    /// Sample from a built-in probability distribution.
    Distribution(Box<dyn MonteCarloDistribution>),
    /// Use pre-computed values, one per run (indexed by run index).
    PreSampled(Vec<MonteCarloSampledValue>),
    /// Call a user-provided function with (run_index, rng) to produce a value.
    Callback(MonteCarloSamplerFn),
}

/// A variable registered with the simulation, pairing an identifier with its
/// sampling source.
pub struct MonteCarloVariable {
    /// Typed identifier for this variable.
    pub id: MonteCarloVariableId,
    /// How values are generated for this variable.
    pub source: MonteCarloSamplingSource,
}

/// The Monte Carlo simulation engine.
///
/// Orchestrates variable sampling, parallel execution of a user-provided
/// simulation function, and collection of results with statistical analysis.
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::{
///     MonteCarloConfig, MonteCarloOutputs, MonteCarloVariableId,
/// };
/// use brahe::monte_carlo::simulation::MonteCarloSimulation;
/// use brahe::monte_carlo::distributions::Gaussian;
///
/// let config = MonteCarloConfig::fixed_runs(100, 42);
/// let mut sim = MonteCarloSimulation::new(config);
/// sim.add_distribution(
///     MonteCarloVariableId::Custom("x".to_string()),
///     Gaussian { mean: 0.0, std: 1.0 },
/// );
///
/// let results = sim.run(|_run_index, params| {
///     let x = params.get_scalar(&MonteCarloVariableId::Custom("x".to_string()))?;
///     let mut outputs = MonteCarloOutputs::new();
///     outputs.insert_scalar("x_squared", x * x);
///     Ok(outputs)
/// }).unwrap();
///
/// assert_eq!(results.runs.len(), 100);
/// ```
pub struct MonteCarloSimulation {
    config: MonteCarloConfig,
    variables: Vec<MonteCarloVariable>,
}

impl MonteCarloSimulation {
    /// Create a new simulation engine with the given configuration.
    ///
    /// # Arguments
    ///
    /// - `config` - Configuration controlling run count, seed, and parallelism
    ///
    /// # Returns
    ///
    /// `MonteCarloSimulation`: A new engine with no variables registered
    pub fn new(config: MonteCarloConfig) -> Self {
        Self {
            config,
            variables: Vec::new(),
        }
    }

    /// Add a variable with an arbitrary sampling source.
    ///
    /// # Arguments
    ///
    /// - `id` - Typed identifier for this variable
    /// - `source` - How samples will be generated
    pub fn add_variable(&mut self, id: MonteCarloVariableId, source: MonteCarloSamplingSource) {
        self.variables.push(MonteCarloVariable { id, source });
    }

    /// Add a variable sampled from a distribution.
    ///
    /// Convenience wrapper around [`add_variable`](Self::add_variable) with
    /// [`MonteCarloSamplingSource::Distribution`].
    ///
    /// # Arguments
    ///
    /// - `id` - Typed identifier for this variable
    /// - `dist` - Distribution to sample from
    pub fn add_distribution(
        &mut self,
        id: MonteCarloVariableId,
        dist: impl MonteCarloDistribution + 'static,
    ) {
        self.add_variable(id, MonteCarloSamplingSource::Distribution(Box::new(dist)));
    }

    /// Add a variable with pre-computed sample values.
    ///
    /// The vector must contain at least as many values as the number of runs
    /// that will be executed. Each run uses `samples[run_index]`.
    ///
    /// # Arguments
    ///
    /// - `id` - Typed identifier for this variable
    /// - `samples` - Pre-computed values, one per run
    pub fn add_presampled(
        &mut self,
        id: MonteCarloVariableId,
        samples: Vec<MonteCarloSampledValue>,
    ) {
        self.add_variable(id, MonteCarloSamplingSource::PreSampled(samples));
    }

    /// Sample a batch of parameters for runs `[start..start+count)`.
    ///
    /// Uses deterministic seeding: each run's seed is derived from the master
    /// seed and the run index via [`derive_run_seed`].
    ///
    /// # Arguments
    ///
    /// - `start` - First run index in the batch
    /// - `count` - Number of runs to sample
    /// - `master_seed` - Master seed for deterministic derivation
    ///
    /// # Returns
    ///
    /// `Vec<MonteCarloSampledParameters>`: One parameter set per run
    ///
    /// # Errors
    ///
    /// Returns [`BraheError::Error`] if a pre-sampled variable has fewer values
    /// than needed for the requested run indices.
    fn sample_batch(
        &self,
        start: usize,
        count: usize,
        master_seed: u64,
    ) -> Result<Vec<MonteCarloSampledParameters>, BraheError> {
        let mut params_vec = Vec::with_capacity(count);
        for i in start..start + count {
            let run_seed = derive_run_seed(master_seed, i);
            let mut rng = StdRng::seed_from_u64(run_seed);
            let mut params = MonteCarloSampledParameters::new(i, run_seed);

            for var in &self.variables {
                let value = match &var.source {
                    MonteCarloSamplingSource::Distribution(dist) => dist.sample(&mut rng),
                    MonteCarloSamplingSource::PreSampled(samples) => {
                        if i >= samples.len() {
                            return Err(BraheError::Error(format!(
                                "Pre-sampled variable '{}' has {} values but run index {} was requested",
                                var.id,
                                samples.len(),
                                i
                            )));
                        }
                        samples[i].clone()
                    }
                    MonteCarloSamplingSource::Callback(f) => f(i, &mut rng),
                };
                params.insert(var.id.clone(), value);
            }
            params_vec.push(params);
        }
        Ok(params_vec)
    }

    /// Run the simulation with a user-provided closure.
    ///
    /// The closure receives the run index and sampled parameters, and must
    /// return either [`MonteCarloOutputs`] on success or [`BraheError`] on
    /// failure. Successful runs are tagged with
    /// [`MonteCarloTerminationReason::ReachedTargetEpoch`]; failed runs are
    /// tagged with [`MonteCarloTerminationReason::Error`].
    ///
    /// For [`MonteCarloStoppingCondition::FixedRuns`], all parameters are
    /// sampled up front and runs execute in parallel via [rayon].
    ///
    /// For [`MonteCarloStoppingCondition::Convergence`], runs execute in
    /// batches. After each batch the standard error of monitored outputs is
    /// checked against the threshold. The simulation stops when all targets
    /// converge or `max_runs` is reached.
    ///
    /// # Arguments
    ///
    /// - `simulation_fn` - Closure that executes one simulation run
    ///
    /// # Returns
    ///
    /// `MonteCarloResults`: Collected results with statistical accessors
    ///
    /// # Errors
    ///
    /// Returns [`BraheError`] if parameter sampling fails (e.g., pre-sampled
    /// variable has insufficient values).
    pub fn run<F>(&self, simulation_fn: F) -> Result<MonteCarloResults, BraheError>
    where
        F: Fn(usize, &MonteCarloSampledParameters) -> Result<MonteCarloOutputs, BraheError>
            + Send
            + Sync,
    {
        match &self.config.stopping_condition {
            MonteCarloStoppingCondition::FixedRuns(n) => self.run_fixed(*n, &simulation_fn),
            MonteCarloStoppingCondition::Convergence {
                targets,
                threshold,
                min_runs,
                max_runs,
                check_interval,
            } => self.run_convergence(
                targets,
                *threshold,
                *min_runs,
                *max_runs,
                *check_interval,
                &simulation_fn,
            ),
        }
    }

    /// Execute a fixed number of runs in parallel.
    fn run_fixed<F>(&self, n: usize, simulation_fn: &F) -> Result<MonteCarloResults, BraheError>
    where
        F: Fn(usize, &MonteCarloSampledParameters) -> Result<MonteCarloOutputs, BraheError>
            + Send
            + Sync,
    {
        let all_params = self.sample_batch(0, n, self.config.seed)?;

        let runs: Vec<MonteCarloRun> = all_params
            .into_par_iter()
            .map(|params| execute_run(&params, simulation_fn))
            .collect();

        Ok(MonteCarloResults::new(
            self.config.clone(),
            runs,
            self.config.seed,
        ))
    }

    /// Execute runs in batches until convergence or max_runs is reached.
    fn run_convergence<F>(
        &self,
        targets: &[String],
        threshold: f64,
        min_runs: usize,
        max_runs: usize,
        check_interval: usize,
        simulation_fn: &F,
    ) -> Result<MonteCarloResults, BraheError>
    where
        F: Fn(usize, &MonteCarloSampledParameters) -> Result<MonteCarloOutputs, BraheError>
            + Send
            + Sync,
    {
        let mut all_runs: Vec<MonteCarloRun> = Vec::new();
        let mut total = 0;

        // First batch: min_runs
        let first_params = self.sample_batch(0, min_runs, self.config.seed)?;
        let first_runs: Vec<MonteCarloRun> = first_params
            .into_par_iter()
            .map(|params| execute_run(&params, simulation_fn))
            .collect();
        all_runs.extend(first_runs);
        total += min_runs;

        // Check convergence after the initial batch and after each subsequent batch
        let mut converged = check_convergence(&all_runs, targets, threshold);

        // Subsequent batches
        while total < max_runs && !converged {
            let batch_size = check_interval.min(max_runs - total);
            let batch_params = self.sample_batch(total, batch_size, self.config.seed)?;
            let batch_runs: Vec<MonteCarloRun> = batch_params
                .into_par_iter()
                .map(|params| execute_run(&params, simulation_fn))
                .collect();
            all_runs.extend(batch_runs);
            total += batch_size;

            converged = check_convergence(&all_runs, targets, threshold);
        }

        let mut results = MonteCarloResults::new(self.config.clone(), all_runs, self.config.seed);
        results.converged = Some(converged);

        // Compute final standard errors for monitored targets
        let mut final_ses = HashMap::new();
        for target in targets {
            if let Ok(se) = results.standard_error(target) {
                final_ses.insert(target.clone(), se);
            }
        }
        results.final_standard_errors = Some(final_ses);

        Ok(results)
    }
}

/// Execute a single simulation run and wrap the result in a [`MonteCarloRun`].
fn execute_run<F>(params: &MonteCarloSampledParameters, simulation_fn: &F) -> MonteCarloRun
where
    F: Fn(usize, &MonteCarloSampledParameters) -> Result<MonteCarloOutputs, BraheError>,
{
    let run_index = params.run_index;
    let result = simulation_fn(run_index, params);
    let termination = match &result {
        Ok(_) => MonteCarloTerminationReason::ReachedTargetEpoch,
        Err(_) => MonteCarloTerminationReason::Error,
    };
    MonteCarloRun {
        run_index,
        sampled_parameters: params.clone(),
        result,
        termination,
    }
}

/// Check whether all target outputs have converged below the threshold.
///
/// Convergence is declared when the standard error of the mean for every
/// target output drops below `threshold`. Targets that have no successful
/// runs or fewer than 2 values are considered not converged.
fn check_convergence(runs: &[MonteCarloRun], targets: &[String], threshold: f64) -> bool {
    targets.iter().all(|target| {
        // Collect scalar values from successful runs for this target
        let values: Vec<f64> = runs
            .iter()
            .filter_map(|run| {
                run.result.as_ref().ok().and_then(|outputs| {
                    outputs.get(target).and_then(|val| match val {
                        super::results::MonteCarloOutputValue::Scalar(v) => Some(*v),
                        super::results::MonteCarloOutputValue::OptionalScalar(Some(v)) => Some(*v),
                        _ => None,
                    })
                })
            })
            .collect();

        let n = values.len();
        if n < 2 {
            return false;
        }

        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = variance.sqrt() / (n as f64).sqrt();
        se < threshold
    })
}

/// Derive a deterministic per-run seed from a master seed and run index.
///
/// Uses wrapping arithmetic with constants from the PCG family to produce
/// well-distributed seeds. The same (master_seed, run_index) pair always
/// produces the same output, regardless of execution order.
///
/// # Arguments
///
/// - `master_seed` - The simulation's master seed
/// - `run_index` - Zero-based index of the run
///
/// # Returns
///
/// `u64`: A deterministic seed for this specific run
pub fn derive_run_seed(master_seed: u64, run_index: usize) -> u64 {
    let mut h = master_seed;
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(run_index as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    h
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::monte_carlo::distributions::{Gaussian, UniformDist};

    // -----------------------------------------------------------------------
    // Test 1: Simple run — sample a scalar, square it, check results
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_fixed_run() {
        let config = MonteCarloConfig::fixed_runs(50, 42);
        let mut sim = MonteCarloSimulation::new(config);
        sim.add_distribution(
            MonteCarloVariableId::Custom("x".to_string()),
            Gaussian {
                mean: 0.0,
                std: 1.0,
            },
        );

        let results = sim
            .run(|_run_index, params| {
                let x = params.get_scalar(&MonteCarloVariableId::Custom("x".to_string()))?;
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("x_squared", x * x);
                Ok(outputs)
            })
            .unwrap();

        assert_eq!(results.runs.len(), 50);
        assert_eq!(results.num_successful(), 50);
        assert_eq!(results.num_failed(), 0);

        // All runs should have the x_squared output
        let values = results.scalar_values("x_squared");
        assert_eq!(values.len(), 50);

        // All x_squared values should be non-negative
        for v in &values {
            assert!(*v >= 0.0, "x^2 should be non-negative, got {}", v);
        }

        // Mean of x^2 where x ~ N(0,1) should be close to 1.0 (the variance)
        let mean = results.mean("x_squared").unwrap();
        assert!(
            (mean - 1.0).abs() < 1.0,
            "Mean of x^2 for N(0,1) should be near 1.0, got {}",
            mean
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Reproducibility — same seed produces identical results
    // -----------------------------------------------------------------------

    #[test]
    fn test_reproducibility_same_seed() {
        let run_sim = |seed: u64| -> Vec<f64> {
            let config = MonteCarloConfig::fixed_runs(20, seed);
            let mut sim = MonteCarloSimulation::new(config);
            sim.add_distribution(
                MonteCarloVariableId::Custom("val".to_string()),
                Gaussian {
                    mean: 5.0,
                    std: 2.0,
                },
            );

            let results = sim
                .run(|_run_index, params| {
                    let v = params.get_scalar(&MonteCarloVariableId::Custom("val".to_string()))?;
                    let mut outputs = MonteCarloOutputs::new();
                    outputs.insert_scalar("result", v * 2.0);
                    Ok(outputs)
                })
                .unwrap();

            // Collect results ordered by run_index for deterministic comparison
            let mut indexed: Vec<(usize, f64)> = results
                .runs
                .iter()
                .filter_map(|r| {
                    r.result.as_ref().ok().and_then(|o| match o.get("result") {
                        Some(super::super::results::MonteCarloOutputValue::Scalar(v)) => {
                            Some((r.run_index, *v))
                        }
                        _ => None,
                    })
                })
                .collect();
            indexed.sort_by_key(|(idx, _)| *idx);
            indexed.into_iter().map(|(_, v)| v).collect()
        };

        let results_a = run_sim(99);
        let results_b = run_sim(99);

        assert_eq!(results_a.len(), results_b.len());
        for (a, b) in results_a.iter().zip(results_b.iter()) {
            assert_eq!(*a, *b, "Results should be bit-identical with same seed");
        }

        // Different seed should produce different results
        let results_c = run_sim(100);
        let any_different = results_a
            .iter()
            .zip(results_c.iter())
            .any(|(a, c)| *a != *c);
        assert!(
            any_different,
            "Different seeds should produce different results"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Convergence mode — verify early stopping
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_early_stopping() {
        // Use a tight distribution so SE converges quickly
        let config = MonteCarloConfig::convergence(
            vec!["output".to_string()],
            0.1,  // threshold: generous SE target
            20,   // min_runs
            1000, // max_runs: high limit we should NOT reach
            10,   // check_interval
            42,
        );
        let mut sim = MonteCarloSimulation::new(config);
        sim.add_distribution(
            MonteCarloVariableId::Custom("x".to_string()),
            Gaussian {
                mean: 100.0,
                std: 0.5, // tight distribution
            },
        );

        let results = sim
            .run(|_run_index, params| {
                let x = params.get_scalar(&MonteCarloVariableId::Custom("x".to_string()))?;
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("output", x);
                Ok(outputs)
            })
            .unwrap();

        // Should have converged before max_runs
        assert!(
            results.converged == Some(true),
            "Should have converged with tight distribution"
        );
        assert!(
            results.runs.len() < 1000,
            "Should stop early, but ran {} times",
            results.runs.len()
        );

        // Final standard errors should be populated
        let final_ses = results.final_standard_errors.as_ref().unwrap();
        assert!(final_ses.contains_key("output"));
        assert!(
            final_ses["output"] < 0.1,
            "Final SE {} should be below threshold 0.1",
            final_ses["output"]
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Failed runs — closure returns Err for some runs
    // -----------------------------------------------------------------------

    #[test]
    fn test_failed_runs_excluded_from_statistics() {
        let config = MonteCarloConfig::fixed_runs(100, 42);
        let mut sim = MonteCarloSimulation::new(config);
        sim.add_distribution(
            MonteCarloVariableId::Custom("x".to_string()),
            UniformDist {
                low: -10.0,
                high: 10.0,
            },
        );

        let results = sim
            .run(|_run_index, params| {
                let x = params.get_scalar(&MonteCarloVariableId::Custom("x".to_string()))?;
                // Fail for negative x values
                if x < 0.0 {
                    return Err(BraheError::Error(format!(
                        "Negative value not allowed: {}",
                        x
                    )));
                }
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("result", x);
                Ok(outputs)
            })
            .unwrap();

        assert_eq!(results.runs.len(), 100);

        let n_success = results.num_successful();
        let n_failed = results.num_failed();
        assert_eq!(n_success + n_failed, 100);
        assert!(n_failed > 0, "Some runs should have failed");
        assert!(n_success > 0, "Some runs should have succeeded");

        // scalar_values should only contain successful runs
        let values = results.scalar_values("result");
        assert_eq!(values.len(), n_success);

        // All collected values should be non-negative (failed runs excluded)
        for v in &values {
            assert!(*v >= 0.0, "Only non-negative values should survive: {}", v);
        }

        // Check termination reasons
        for run in &results.runs {
            match &run.result {
                Ok(_) => assert!(matches!(
                    run.termination,
                    MonteCarloTerminationReason::ReachedTargetEpoch
                )),
                Err(_) => assert!(matches!(
                    run.termination,
                    MonteCarloTerminationReason::Error
                )),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Pre-sampled source validation — error if not enough samples
    // -----------------------------------------------------------------------

    #[test]
    fn test_presampled_insufficient_values() {
        let config = MonteCarloConfig::fixed_runs(5, 42);
        let mut sim = MonteCarloSimulation::new(config);
        // Only provide 3 samples but request 5 runs
        sim.add_presampled(
            MonteCarloVariableId::Custom("v".to_string()),
            vec![
                MonteCarloSampledValue::Scalar(1.0),
                MonteCarloSampledValue::Scalar(2.0),
                MonteCarloSampledValue::Scalar(3.0),
            ],
        );

        let result = sim.run(|_run_index, params| {
            let v = params.get_scalar(&MonteCarloVariableId::Custom("v".to_string()))?;
            let mut outputs = MonteCarloOutputs::new();
            outputs.insert_scalar("result", v);
            Ok(outputs)
        });

        assert!(
            result.is_err(),
            "Should fail with insufficient pre-sampled values"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Pre-sampled"),
            "Error should mention pre-sampled: {}",
            err_msg
        );
        assert!(
            err_msg.contains("3"),
            "Error should mention the count of available values: {}",
            err_msg
        );
    }

    #[test]
    fn test_presampled_sufficient_values() {
        let config = MonteCarloConfig::fixed_runs(3, 42);
        let mut sim = MonteCarloSimulation::new(config);
        sim.add_presampled(
            MonteCarloVariableId::Custom("v".to_string()),
            vec![
                MonteCarloSampledValue::Scalar(10.0),
                MonteCarloSampledValue::Scalar(20.0),
                MonteCarloSampledValue::Scalar(30.0),
            ],
        );

        let results = sim
            .run(|_run_index, params| {
                let v = params.get_scalar(&MonteCarloVariableId::Custom("v".to_string()))?;
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("result", v);
                Ok(outputs)
            })
            .unwrap();

        assert_eq!(results.runs.len(), 3);
        assert_eq!(results.num_successful(), 3);

        // Verify exact pre-sampled values were used (ordered by run_index)
        let mut indexed: Vec<(usize, f64)> = results
            .runs
            .iter()
            .map(|r| {
                let v = r.result.as_ref().unwrap().get("result").unwrap();
                let scalar = match v {
                    super::super::results::MonteCarloOutputValue::Scalar(s) => *s,
                    _ => panic!("Expected scalar"),
                };
                (r.run_index, scalar)
            })
            .collect();
        indexed.sort_by_key(|(idx, _)| *idx);

        assert_eq!(indexed[0].1, 10.0);
        assert_eq!(indexed[1].1, 20.0);
        assert_eq!(indexed[2].1, 30.0);
    }

    // -----------------------------------------------------------------------
    // Test 6: Callback source — verify correct run indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_callback_receives_correct_run_indices() {
        use std::sync::{Arc, Mutex};

        let config = MonteCarloConfig::fixed_runs(10, 42);
        let mut sim = MonteCarloSimulation::new(config);

        let received_indices = Arc::new(Mutex::new(Vec::new()));
        let indices_clone = Arc::clone(&received_indices);

        sim.add_variable(
            MonteCarloVariableId::Custom("cb".to_string()),
            MonteCarloSamplingSource::Callback(Box::new(move |run_index, _rng| {
                indices_clone.lock().unwrap().push(run_index);
                // Return the run index as a scalar so we can verify in outputs
                MonteCarloSampledValue::Scalar(run_index as f64)
            })),
        );

        let results = sim
            .run(|_run_index, params| {
                let v = params.get_scalar(&MonteCarloVariableId::Custom("cb".to_string()))?;
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("index", v);
                Ok(outputs)
            })
            .unwrap();

        assert_eq!(results.runs.len(), 10);

        // The callback should have been called with indices 0..10
        let mut indices = received_indices.lock().unwrap().clone();
        indices.sort();
        assert_eq!(indices, (0..10).collect::<Vec<_>>());

        // Verify outputs match the run indices
        for run in &results.runs {
            let output_val = run.result.as_ref().unwrap().get("index").unwrap();
            match output_val {
                super::super::results::MonteCarloOutputValue::Scalar(v) => {
                    assert_eq!(*v as usize, run.run_index);
                }
                _ => panic!("Expected scalar output"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Supplementary: derive_run_seed determinism
    // -----------------------------------------------------------------------

    #[test]
    fn test_derive_run_seed_deterministic() {
        let seed_a = derive_run_seed(42, 0);
        let seed_b = derive_run_seed(42, 0);
        assert_eq!(seed_a, seed_b);

        // Different indices produce different seeds
        let seed_c = derive_run_seed(42, 1);
        assert_ne!(seed_a, seed_c);

        // Different master seeds produce different seeds
        let seed_d = derive_run_seed(43, 0);
        assert_ne!(seed_a, seed_d);
    }

    // -----------------------------------------------------------------------
    // Supplementary: convergence reaches max_runs when not converging
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_reaches_max_runs() {
        // Use a wide distribution with a very tight threshold so it cannot converge
        let config = MonteCarloConfig::convergence(
            vec!["output".to_string()],
            1e-10, // impossibly tight threshold
            10,    // min_runs
            50,    // max_runs
            10,    // check_interval
            42,
        );
        let mut sim = MonteCarloSimulation::new(config);
        sim.add_distribution(
            MonteCarloVariableId::Custom("x".to_string()),
            Gaussian {
                mean: 0.0,
                std: 100.0,
            },
        );

        let results = sim
            .run(|_run_index, params| {
                let x = params.get_scalar(&MonteCarloVariableId::Custom("x".to_string()))?;
                let mut outputs = MonteCarloOutputs::new();
                outputs.insert_scalar("output", x);
                Ok(outputs)
            })
            .unwrap();

        assert_eq!(results.converged, Some(false));
        assert_eq!(results.runs.len(), 50);
    }
}
