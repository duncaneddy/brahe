/*!
 * Configuration types for Monte Carlo simulations.
 *
 * Provides [`MonteCarloConfig`] to control the execution parameters of a
 * Monte Carlo simulation, including stopping conditions, random seeding,
 * and parallelism.
 */

use std::fmt;

/// Configuration for a Monte Carlo simulation.
///
/// Controls how many simulation runs to execute, the random seed for
/// reproducibility, and the number of parallel workers.
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::MonteCarloConfig;
///
/// // Fixed number of runs
/// let config = MonteCarloConfig::fixed_runs(1000, 42);
/// assert_eq!(config.max_runs(), 1000);
///
/// // Convergence-based stopping
/// let config = MonteCarloConfig::convergence(
///     vec!["position_error".to_string()],
///     0.01,   // threshold
///     100,    // min_runs
///     10000,  // max_runs
///     50,     // check_interval
///     42,     // seed
/// );
/// assert_eq!(config.max_runs(), 10000);
/// ```
#[derive(Clone, Debug)]
pub struct MonteCarloConfig {
    /// When to stop running simulations.
    pub stopping_condition: MonteCarloStoppingCondition,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Number of parallel workers (0 = auto-detect via num_cpus).
    pub num_workers: usize,
}

/// Controls when the Monte Carlo simulation stops generating new runs.
///
/// Two strategies are supported:
/// - [`FixedRuns`](MonteCarloStoppingCondition::FixedRuns): run exactly N simulations
/// - [`Convergence`](MonteCarloStoppingCondition::Convergence): run until the standard
///   error of monitored outputs drops below a threshold
#[derive(Clone, Debug)]
pub enum MonteCarloStoppingCondition {
    /// Run exactly N simulations.
    FixedRuns(usize),
    /// Run until standard error of monitored outputs drops below threshold.
    Convergence {
        /// Output variable names to monitor for convergence.
        targets: Vec<String>,
        /// Standard error threshold below which convergence is declared.
        threshold: f64,
        /// Minimum number of runs before checking convergence.
        min_runs: usize,
        /// Maximum number of runs (safety limit).
        max_runs: usize,
        /// Check convergence every N runs.
        check_interval: usize,
    },
}

impl MonteCarloConfig {
    /// Create a configuration with a fixed number of runs.
    ///
    /// # Arguments
    ///
    /// - `num_runs` - Number of simulation runs to execute
    /// - `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// `MonteCarloConfig`: Configuration with [`MonteCarloStoppingCondition::FixedRuns`]
    pub fn fixed_runs(num_runs: usize, seed: u64) -> Self {
        Self {
            stopping_condition: MonteCarloStoppingCondition::FixedRuns(num_runs),
            seed,
            num_workers: 0,
        }
    }

    /// Create a configuration with convergence-based stopping.
    ///
    /// The simulation will run until the standard error of the monitored
    /// output variables drops below the specified threshold, or until
    /// `max_runs` is reached.
    ///
    /// # Arguments
    ///
    /// - `targets` - Output variable names to monitor for convergence
    /// - `threshold` - Standard error threshold (convergence criterion)
    /// - `min_runs` - Minimum number of runs before checking convergence
    /// - `max_runs` - Maximum number of runs (safety limit)
    /// - `check_interval` - Check convergence every N runs
    /// - `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// `MonteCarloConfig`: Configuration with [`MonteCarloStoppingCondition::Convergence`]
    pub fn convergence(
        targets: Vec<String>,
        threshold: f64,
        min_runs: usize,
        max_runs: usize,
        check_interval: usize,
        seed: u64,
    ) -> Self {
        Self {
            stopping_condition: MonteCarloStoppingCondition::Convergence {
                targets,
                threshold,
                min_runs,
                max_runs,
                check_interval,
            },
            seed,
            num_workers: 0,
        }
    }

    /// Get the maximum possible number of runs for this configuration.
    ///
    /// For [`MonteCarloStoppingCondition::FixedRuns`], returns the fixed count.
    /// For [`MonteCarloStoppingCondition::Convergence`], returns `max_runs`.
    ///
    /// # Returns
    ///
    /// `usize`: Maximum number of runs
    pub fn max_runs(&self) -> usize {
        match &self.stopping_condition {
            MonteCarloStoppingCondition::FixedRuns(n) => *n,
            MonteCarloStoppingCondition::Convergence { max_runs, .. } => *max_runs,
        }
    }
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self::fixed_runs(100, 42)
    }
}

impl fmt::Display for MonteCarloConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MonteCarloConfig {{ stopping: {}, seed: {}, workers: {} }}",
            self.stopping_condition,
            self.seed,
            if self.num_workers == 0 {
                "auto".to_string()
            } else {
                self.num_workers.to_string()
            }
        )
    }
}

impl fmt::Display for MonteCarloStoppingCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MonteCarloStoppingCondition::FixedRuns(n) => {
                write!(f, "FixedRuns({})", n)
            }
            MonteCarloStoppingCondition::Convergence {
                targets,
                threshold,
                min_runs,
                max_runs,
                check_interval,
            } => {
                write!(
                    f,
                    "Convergence {{ targets: [{}], threshold: {}, min_runs: {}, max_runs: {}, check_interval: {} }}",
                    targets.join(", "),
                    threshold,
                    min_runs,
                    max_runs,
                    check_interval
                )
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_runs_creation() {
        let config = MonteCarloConfig::fixed_runs(500, 123);
        assert_eq!(config.max_runs(), 500);
        assert_eq!(config.seed, 123);
        assert_eq!(config.num_workers, 0);
        assert!(matches!(
            config.stopping_condition,
            MonteCarloStoppingCondition::FixedRuns(500)
        ));
    }

    #[test]
    fn test_convergence_creation() {
        let targets = vec!["pos_err".to_string(), "vel_err".to_string()];
        let config = MonteCarloConfig::convergence(targets.clone(), 0.01, 50, 5000, 25, 99);
        assert_eq!(config.max_runs(), 5000);
        assert_eq!(config.seed, 99);
        assert_eq!(config.num_workers, 0);
        match &config.stopping_condition {
            MonteCarloStoppingCondition::Convergence {
                targets: t,
                threshold,
                min_runs,
                max_runs,
                check_interval,
            } => {
                assert_eq!(t, &targets);
                assert_eq!(*threshold, 0.01);
                assert_eq!(*min_runs, 50);
                assert_eq!(*max_runs, 5000);
                assert_eq!(*check_interval, 25);
            }
            _ => panic!("Expected Convergence stopping condition"),
        }
    }

    #[test]
    fn test_default() {
        let config = MonteCarloConfig::default();
        assert_eq!(config.max_runs(), 100);
        assert_eq!(config.seed, 42);
        assert_eq!(config.num_workers, 0);
        assert!(matches!(
            config.stopping_condition,
            MonteCarloStoppingCondition::FixedRuns(100)
        ));
    }

    #[test]
    fn test_max_runs_fixed() {
        let config = MonteCarloConfig::fixed_runs(200, 0);
        assert_eq!(config.max_runs(), 200);
    }

    #[test]
    fn test_max_runs_convergence() {
        let config = MonteCarloConfig::convergence(vec!["x".to_string()], 0.1, 10, 999, 5, 0);
        assert_eq!(config.max_runs(), 999);
    }

    #[test]
    fn test_num_workers_can_be_set() {
        let mut config = MonteCarloConfig::fixed_runs(100, 42);
        config.num_workers = 4;
        assert_eq!(config.num_workers, 4);
    }

    #[test]
    fn test_clone() {
        let config = MonteCarloConfig::fixed_runs(100, 42);
        let cloned = config.clone();
        assert_eq!(cloned.max_runs(), config.max_runs());
        assert_eq!(cloned.seed, config.seed);
        assert_eq!(cloned.num_workers, config.num_workers);
    }

    #[test]
    fn test_debug() {
        let config = MonteCarloConfig::fixed_runs(100, 42);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("MonteCarloConfig"));
        assert!(debug_str.contains("FixedRuns"));
    }

    #[test]
    fn test_display_fixed_runs() {
        let config = MonteCarloConfig::fixed_runs(100, 42);
        let display = format!("{}", config);
        assert!(display.contains("FixedRuns(100)"));
        assert!(display.contains("seed: 42"));
        assert!(display.contains("workers: auto"));
    }

    #[test]
    fn test_display_fixed_runs_with_workers() {
        let mut config = MonteCarloConfig::fixed_runs(100, 42);
        config.num_workers = 8;
        let display = format!("{}", config);
        assert!(display.contains("workers: 8"));
    }

    #[test]
    fn test_display_convergence() {
        let config = MonteCarloConfig::convergence(
            vec!["pos".to_string(), "vel".to_string()],
            0.05,
            100,
            10000,
            50,
            7,
        );
        let display = format!("{}", config);
        assert!(display.contains("Convergence"));
        assert!(display.contains("pos, vel"));
        assert!(display.contains("threshold: 0.05"));
        assert!(display.contains("min_runs: 100"));
        assert!(display.contains("max_runs: 10000"));
        assert!(display.contains("check_interval: 50"));
    }

    #[test]
    fn test_display_stopping_condition_fixed() {
        let cond = MonteCarloStoppingCondition::FixedRuns(42);
        assert_eq!(format!("{}", cond), "FixedRuns(42)");
    }

    #[test]
    fn test_display_stopping_condition_convergence() {
        let cond = MonteCarloStoppingCondition::Convergence {
            targets: vec!["a".to_string()],
            threshold: 1e-3,
            min_runs: 10,
            max_runs: 500,
            check_interval: 10,
        };
        let display = format!("{}", cond);
        assert!(display.contains("a"));
        assert!(display.contains("0.001"));
    }
}
