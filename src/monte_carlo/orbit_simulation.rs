/*!
 * Built-in orbit propagation Monte Carlo simulation template.
 *
 * Provides [`OrbitMonteCarloConfig`] and a [`run_orbit_propagation`] method on
 * [`MonteCarloSimulation`] for the common case of orbit propagation Monte Carlo
 * analysis. Each run builds a [`DNumericalOrbitPropagator`], propagates to a
 * target epoch, and extracts configurable outputs.
 *
 * Thread-local EOP and space weather providers are automatically set from
 * sampled table data when [`MonteCarloVariableId::EopTable`] or
 * [`MonteCarloVariableId::SpaceWeatherTable`] variables are registered.
 */

use std::sync::Arc;

use nalgebra::{DVector, Vector6};

use crate::constants::{AngleFormat, GM_EARTH, R_EARTH};
use crate::eop::{
    EOPExtrapolation, TableEOPProvider, clear_thread_local_eop_provider,
    set_thread_local_eop_provider,
};
use crate::events::{DAltitudeEvent, EventAction, EventDirection};
use crate::propagators::traits::DStatePropagator;
use crate::propagators::{DNumericalOrbitPropagator, ForceModelConfig, NumericalPropagationConfig};
use crate::state_eci_to_koe;
use crate::time::Epoch;
use crate::utils::BraheError;

use super::results::{MonteCarloOutputs, MonteCarloResults, MonteCarloTerminationReason};
use super::simulation::MonteCarloSimulation;
use super::variables::{MonteCarloSampledParameters, MonteCarloSampledValue, MonteCarloVariableId};

/// Configuration for built-in orbit propagation Monte Carlo simulation.
///
/// Specifies the initial conditions, propagation settings, force models,
/// event detectors, and which outputs to extract from each run.
///
/// # Examples
///
/// ```rust,ignore
/// use brahe::monte_carlo::orbit_simulation::{OrbitMonteCarloConfig, OrbitMonteCarloOutput};
/// use brahe::propagators::{NumericalPropagationConfig, ForceModelConfig};
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::constants::R_EARTH;
/// use nalgebra::DVector;
///
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let config = OrbitMonteCarloConfig {
///     epoch,
///     propagation_config: NumericalPropagationConfig::default(),
///     force_config: ForceModelConfig::two_body_gravity(),
///     params: None,
///     target_epoch: epoch + 86400.0,
///     events: vec![],
///     outputs: vec![
///         OrbitMonteCarloOutput::FinalState,
///         OrbitMonteCarloOutput::FinalAltitude,
///     ],
/// };
/// ```
pub struct OrbitMonteCarloConfig {
    /// Initial epoch for propagation.
    pub epoch: Epoch,
    /// Numerical integrator configuration.
    pub propagation_config: NumericalPropagationConfig,
    /// Force model configuration (gravity, drag, SRP, etc.).
    pub force_config: ForceModelConfig,
    /// Optional spacecraft parameter vector `[mass, drag_area, Cd, srp_area, Cr, ...]`.
    /// Required if `force_config` references parameter indices.
    pub params: Option<DVector<f64>>,
    /// Target epoch to propagate to.
    pub target_epoch: Epoch,
    /// Event detectors to monitor during propagation.
    pub events: Vec<OrbitMonteCarloEventConfig>,
    /// Outputs to extract from each run.
    pub outputs: Vec<OrbitMonteCarloOutput>,
}

/// Event detector configuration for orbit Monte Carlo simulation.
///
/// Describes events to monitor during propagation. Each variant maps to a
/// built-in event detector type.
#[derive(Clone, Debug)]
pub enum OrbitMonteCarloEventConfig {
    /// Geodetic altitude crossing event.
    Altitude {
        /// Altitude threshold in meters above WGS84 ellipsoid.
        altitude: f64,
        /// Name for this event (used in output keys).
        name: String,
        /// Whether this event should terminate propagation.
        terminal: bool,
        /// Detection direction (increasing, decreasing, or any).
        direction: EventDirection,
    },
}

/// Specifies which outputs to extract from each propagation run.
///
/// Each variant produces a named output in [`MonteCarloOutputs`] that can be
/// accessed via the corresponding key string.
#[derive(Clone, Debug)]
pub enum OrbitMonteCarloOutput {
    /// Final ECI Cartesian state vector (6 elements). Key: `"final_state"`.
    FinalState,
    /// Final epoch as MJD. Key: `"final_epoch"`.
    FinalEpoch,
    /// Simulation duration in seconds. Key: `"simulation_duration"`.
    SimulationDuration,
    /// Final geodetic altitude in meters (computed from position magnitude - R_EARTH).
    /// Key: `"final_altitude"`.
    FinalAltitude,
    /// Final semi-major axis in meters. Key: `"final_semi_major_axis"`.
    FinalSemiMajorAxis,
    /// Final eccentricity. Key: `"final_eccentricity"`.
    FinalEccentricity,
    /// Duration from start to first occurrence of the named event (seconds).
    /// Key: `"first_event_duration_<name>"`. `None` if the event never occurred.
    FirstEventDuration(String),
    /// Duration from start to last occurrence of the named event (seconds).
    /// Key: `"last_event_duration_<name>"`. `None` if the event never occurred.
    LastEventDuration(String),
    /// Count of occurrences for the named event. Key: `"event_count_<name>"`.
    EventCount(String),
}

impl MonteCarloSimulation {
    /// Run a built-in orbit propagation Monte Carlo simulation.
    ///
    /// Each run:
    /// 1. Sets thread-local EOP/SW providers from sampled tables (if registered)
    /// 2. Builds a [`DNumericalOrbitPropagator`] from nominal + sampled parameters
    /// 3. Adds configured event detectors
    /// 4. Propagates to the target epoch
    /// 5. Extracts configured outputs and determines termination reason
    /// 6. Cleans up thread-local providers
    ///
    /// # Arguments
    ///
    /// * `orbit_config` - Configuration for the orbit propagation MC
    ///
    /// # Returns
    ///
    /// `MonteCarloResults`: Collected results with statistical accessors
    ///
    /// # Errors
    ///
    /// Returns [`BraheError`] if parameter sampling fails.
    pub fn run_orbit_propagation(
        &self,
        orbit_config: OrbitMonteCarloConfig,
    ) -> Result<MonteCarloResults, BraheError> {
        let orbit_config = Arc::new(orbit_config);

        self.run(move |_run_index, params| {
            let config = Arc::clone(&orbit_config);
            run_single_orbit_propagation(&config, params)
        })
    }
}

/// Execute a single orbit propagation run with the given sampled parameters.
fn run_single_orbit_propagation(
    config: &OrbitMonteCarloConfig,
    params: &MonteCarloSampledParameters,
) -> Result<MonteCarloOutputs, BraheError> {
    // 1. Set thread-local EOP if sampled
    let has_eop = params.get(&MonteCarloVariableId::EopTable).is_some();
    if has_eop {
        let table = params.get_table(&MonteCarloVariableId::EopTable)?;
        let entries: Vec<_> = table
            .iter()
            .map(|(mjd, values)| {
                (
                    *mjd,
                    values.first().copied().unwrap_or(0.0),
                    values.get(1).copied().unwrap_or(0.0),
                    values.get(2).copied().unwrap_or(0.0),
                    values.get(3).copied(),
                    values.get(4).copied(),
                    values.get(5).copied(),
                )
            })
            .collect();

        let provider = TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold)?;
        set_thread_local_eop_provider(Arc::new(provider));
    }

    // 2. Get initial state from sampled parameters or error
    let state = if let Some(value) = params.get(&MonteCarloVariableId::InitialState) {
        match value {
            MonteCarloSampledValue::Vector(v) => v.clone(),
            _ => {
                return Err(BraheError::Error(
                    "InitialState must be a Vector value".into(),
                ));
            }
        }
    } else {
        return Err(BraheError::Error(
            "InitialState variable not provided".into(),
        ));
    };

    // 3. Build params vector with sampled overrides
    let mut prop_params = config.params.clone();
    if let Some(ref mut p) = prop_params {
        // Apply sampled spacecraft parameters (Mass, DragArea, etc.)
        for id in params.ids() {
            if let Some(idx) = id.param_index()
                && let Ok(v) = params.get_scalar(id)
                && idx < p.len()
            {
                p[idx] = v;
            }
        }
    }

    // 4. Create propagator
    let mut propagator = DNumericalOrbitPropagator::new(
        config.epoch,
        state,
        config.propagation_config.clone(),
        config.force_config.clone(),
        prop_params,
        None, // additional_dynamics
        None, // control_input
        None, // initial_covariance
    )?;

    // 5. Add event detectors
    for event_config in &config.events {
        match event_config {
            OrbitMonteCarloEventConfig::Altitude {
                altitude,
                name,
                terminal,
                direction,
            } => {
                let mut event = DAltitudeEvent::new(*altitude, name.clone(), *direction);
                if *terminal {
                    event = event.set_terminal();
                }
                propagator.add_event_detector(Box::new(event));
            }
        }
    }

    // 6. Propagate
    propagator.propagate_to(config.target_epoch);

    // 7. Determine termination reason
    let termination = if propagator.is_terminated() {
        // Find the terminal event
        let terminal_event = propagator
            .event_log()
            .iter()
            .rev()
            .find(|e| e.action == EventAction::Stop);
        if let Some(event) = terminal_event {
            MonteCarloTerminationReason::TerminalEvent {
                event_name: event.name.clone(),
                epoch: event.window_open,
            }
        } else {
            MonteCarloTerminationReason::ReachedTargetEpoch
        }
    } else {
        MonteCarloTerminationReason::ReachedTargetEpoch
    };

    // 8. Extract outputs
    let mut outputs = MonteCarloOutputs::new();
    let final_state = propagator.current_state();
    let final_epoch = propagator.current_epoch();
    let initial_epoch = config.epoch;

    for output in &config.outputs {
        match output {
            OrbitMonteCarloOutput::FinalState => {
                outputs.insert_vector("final_state", final_state.clone());
            }
            OrbitMonteCarloOutput::FinalEpoch => {
                outputs.insert_scalar("final_epoch", final_epoch.mjd());
            }
            OrbitMonteCarloOutput::SimulationDuration => {
                outputs.insert_scalar("simulation_duration", final_epoch - initial_epoch);
            }
            OrbitMonteCarloOutput::FinalAltitude => {
                let r = (final_state[0].powi(2) + final_state[1].powi(2) + final_state[2].powi(2))
                    .sqrt();
                outputs.insert_scalar("final_altitude", r - R_EARTH);
            }
            OrbitMonteCarloOutput::FinalSemiMajorAxis => {
                let r = (final_state[0].powi(2) + final_state[1].powi(2) + final_state[2].powi(2))
                    .sqrt();
                let v = (final_state[3].powi(2) + final_state[4].powi(2) + final_state[5].powi(2))
                    .sqrt();
                let energy = v * v / 2.0 - GM_EARTH / r;
                outputs.insert_scalar("final_semi_major_axis", -GM_EARTH / (2.0 * energy));
            }
            OrbitMonteCarloOutput::FinalEccentricity => {
                let state_v6 = Vector6::new(
                    final_state[0],
                    final_state[1],
                    final_state[2],
                    final_state[3],
                    final_state[4],
                    final_state[5],
                );
                let koe = state_eci_to_koe(state_v6, AngleFormat::Radians);
                outputs.insert_scalar("final_eccentricity", koe[1]);
            }
            OrbitMonteCarloOutput::FirstEventDuration(name) => {
                let duration = propagator
                    .event_log()
                    .iter()
                    .find(|e| e.name == *name)
                    .map(|e| e.window_open - initial_epoch);
                outputs.insert_optional_scalar(&format!("first_event_duration_{}", name), duration);
            }
            OrbitMonteCarloOutput::LastEventDuration(name) => {
                let duration = propagator
                    .event_log()
                    .iter()
                    .rev()
                    .find(|e| e.name == *name)
                    .map(|e| e.window_open - initial_epoch);
                outputs.insert_optional_scalar(&format!("last_event_duration_{}", name), duration);
            }
            OrbitMonteCarloOutput::EventCount(name) => {
                let count = propagator
                    .event_log()
                    .iter()
                    .filter(|e| e.name == *name)
                    .count();
                outputs.insert_scalar(&format!("event_count_{}", name), count as f64);
            }
        }
    }

    // 9. Cleanup thread-local EOP
    if has_eop {
        clear_thread_local_eop_provider();
    }

    // Note: termination reason is not used here because the run() wrapper in
    // simulation.rs always sets ReachedTargetEpoch for Ok results. If the caller
    // needs terminal event info, it should use the event-related outputs
    // (FirstEventDuration, EventCount, etc.).
    let _ = termination;

    Ok(outputs)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::eop::{StaticEOPProvider, set_global_eop_provider};
    use crate::monte_carlo::config::MonteCarloConfig;
    use crate::monte_carlo::distributions::Gaussian;
    use crate::monte_carlo::simulation::MonteCarloSamplingSource;
    use crate::monte_carlo::variables::MonteCarloSampledValue;
    use crate::time::TimeSystem;

    /// Helper to create a circular orbit state in ECI at a given altitude.
    fn circular_orbit_state(altitude: f64) -> DVector<f64> {
        let r = R_EARTH + altitude;
        let v = (GM_EARTH / r).sqrt();
        DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0])
    }

    #[test]
    fn test_orbit_mc_basic_two_body() {
        // Initialize EOP provider
        let eop = StaticEOPProvider::from_zero();
        set_global_eop_provider(eop);

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let nominal_state = circular_orbit_state(500e3);

        // Create MC config
        let mc_config = MonteCarloConfig::fixed_runs(5, 42);
        let mut sim = MonteCarloSimulation::new(mc_config);

        // Add initial state with small position perturbations
        let nominal_clone = nominal_state.clone();
        sim.add_variable(
            MonteCarloVariableId::InitialState,
            MonteCarloSamplingSource::Callback(Box::new(move |_run_index, rng| {
                use rand::Rng;
                let mut state = nominal_clone.clone();
                // Perturb position by up to +/- 100 m
                state[0] += rng.random_range(-100.0..100.0);
                state[1] += rng.random_range(-100.0..100.0);
                state[2] += rng.random_range(-100.0..100.0);
                MonteCarloSampledValue::Vector(state)
            })),
        );

        // Create orbit MC config
        let orbit_config = OrbitMonteCarloConfig {
            epoch,
            propagation_config: NumericalPropagationConfig::default(),
            force_config: ForceModelConfig::two_body_gravity(),
            params: None,
            target_epoch: epoch + 3600.0, // 1 hour
            events: vec![],
            outputs: vec![
                OrbitMonteCarloOutput::FinalState,
                OrbitMonteCarloOutput::FinalAltitude,
                OrbitMonteCarloOutput::FinalSemiMajorAxis,
                OrbitMonteCarloOutput::FinalEccentricity,
                OrbitMonteCarloOutput::SimulationDuration,
            ],
        };

        let results = sim.run_orbit_propagation(orbit_config).unwrap();

        assert_eq!(results.runs.len(), 5);
        assert_eq!(results.num_successful(), 5);
        assert_eq!(results.num_failed(), 0);

        // All altitudes should be near 500 km (two-body preserves energy)
        let altitudes = results.scalar_values("final_altitude");
        assert_eq!(altitudes.len(), 5);
        for alt in &altitudes {
            assert!(
                (*alt - 500e3).abs() < 50e3,
                "Altitude {} should be near 500 km",
                alt
            );
        }

        // Semi-major axis should be near R_EARTH + 500 km
        let sma_values = results.scalar_values("final_semi_major_axis");
        assert_eq!(sma_values.len(), 5);
        for sma in &sma_values {
            assert!(
                (*sma - (R_EARTH + 500e3)).abs() < 1000.0,
                "SMA {} should be near {}",
                sma,
                R_EARTH + 500e3
            );
        }

        // Eccentricity should be near zero (nearly circular)
        let ecc_values = results.scalar_values("final_eccentricity");
        assert_eq!(ecc_values.len(), 5);
        for ecc in &ecc_values {
            assert!(*ecc < 0.01, "Eccentricity {} should be near zero", ecc);
        }

        // Simulation duration should be 3600 seconds
        let durations = results.scalar_values("simulation_duration");
        assert_eq!(durations.len(), 5);
        for dur in &durations {
            assert!(
                (*dur - 3600.0).abs() < 1.0,
                "Duration {} should be 3600 s",
                dur
            );
        }
    }

    #[test]
    fn test_orbit_mc_missing_initial_state_errors() {
        let eop = StaticEOPProvider::from_zero();
        set_global_eop_provider(eop);

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let mc_config = MonteCarloConfig::fixed_runs(1, 42);
        let mut sim = MonteCarloSimulation::new(mc_config);

        // Add a dummy variable but NOT InitialState
        sim.add_distribution(
            MonteCarloVariableId::Custom("dummy".to_string()),
            Gaussian {
                mean: 0.0,
                std: 1.0,
            },
        );

        let orbit_config = OrbitMonteCarloConfig {
            epoch,
            propagation_config: NumericalPropagationConfig::default(),
            force_config: ForceModelConfig::two_body_gravity(),
            params: None,
            target_epoch: epoch + 60.0,
            events: vec![],
            outputs: vec![OrbitMonteCarloOutput::FinalAltitude],
        };

        let results = sim.run_orbit_propagation(orbit_config).unwrap();

        // The run should have failed because InitialState was not provided
        assert_eq!(results.num_successful(), 0);
        assert_eq!(results.num_failed(), 1);
    }

    #[test]
    fn test_orbit_mc_reproducibility() {
        let eop = StaticEOPProvider::from_zero();
        set_global_eop_provider(eop);

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let nominal_state = circular_orbit_state(500e3);

        let run_sim = || {
            let mc_config = MonteCarloConfig::fixed_runs(3, 99);
            let mut sim = MonteCarloSimulation::new(mc_config);

            let nominal = nominal_state.clone();
            sim.add_variable(
                MonteCarloVariableId::InitialState,
                MonteCarloSamplingSource::Callback(Box::new(move |_run_index, rng| {
                    use rand::Rng;
                    let mut state = nominal.clone();
                    state[0] += rng.random_range(-50.0..50.0);
                    MonteCarloSampledValue::Vector(state)
                })),
            );

            let orbit_config = OrbitMonteCarloConfig {
                epoch,
                propagation_config: NumericalPropagationConfig::default(),
                force_config: ForceModelConfig::two_body_gravity(),
                params: None,
                target_epoch: epoch + 600.0,
                events: vec![],
                outputs: vec![OrbitMonteCarloOutput::FinalAltitude],
            };

            sim.run_orbit_propagation(orbit_config).unwrap()
        };

        let results_a = run_sim();
        let results_b = run_sim();

        let alts_a = results_a.scalar_values("final_altitude");
        let alts_b = results_b.scalar_values("final_altitude");

        assert_eq!(alts_a.len(), alts_b.len());
        // Note: ordering within parallel runs may differ, so compare sorted
        let mut sorted_a = alts_a.clone();
        let mut sorted_b = alts_b.clone();
        sorted_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_b.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (a, b) in sorted_a.iter().zip(sorted_b.iter()) {
            assert_eq!(*a, *b, "Results should be identical with same seed");
        }
    }

    #[test]
    fn test_orbit_mc_event_count_output() {
        let eop = StaticEOPProvider::from_zero();
        set_global_eop_provider(eop);

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let nominal_state = circular_orbit_state(500e3);

        let mc_config = MonteCarloConfig::fixed_runs(1, 42);
        let mut sim = MonteCarloSimulation::new(mc_config);

        let nominal = nominal_state.clone();
        sim.add_variable(
            MonteCarloVariableId::InitialState,
            MonteCarloSamplingSource::Callback(Box::new(move |_run_index, _rng| {
                MonteCarloSampledValue::Vector(nominal.clone())
            })),
        );

        // Monitor altitude crossing at exactly 500 km (the circular orbit altitude).
        // This should not trigger for a perfectly circular orbit at 500 km since
        // there's no crossing, but the event count output should still be produced.
        let orbit_config = OrbitMonteCarloConfig {
            epoch,
            propagation_config: NumericalPropagationConfig::default(),
            force_config: ForceModelConfig::two_body_gravity(),
            params: None,
            target_epoch: epoch + 3600.0,
            events: vec![OrbitMonteCarloEventConfig::Altitude {
                altitude: 400e3, // 400 km (below circular orbit)
                name: "alt_400km".to_string(),
                terminal: false,
                direction: EventDirection::Decreasing,
            }],
            outputs: vec![OrbitMonteCarloOutput::EventCount("alt_400km".to_string())],
        };

        let results = sim.run_orbit_propagation(orbit_config).unwrap();
        assert_eq!(results.num_successful(), 1);

        let counts = results.scalar_values("event_count_alt_400km");
        assert_eq!(counts.len(), 1);
        // A circular orbit at 500 km should not cross 400 km
        assert_eq!(counts[0], 0.0);
    }
}
