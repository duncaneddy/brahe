/*!
 * Utility functions for orbit propagation operations
 */

use rayon::prelude::*;

use crate::propagators::traits::SStatePropagator;
use crate::time::Epoch;
use crate::utils::threading::get_thread_pool;

/// Propagate multiple propagators to a target epoch in parallel.
///
/// This function takes a slice of propagators and calls `propagate_to` on each one
/// in parallel using the global thread pool. Each propagator's internal state is updated
/// to reflect the new epoch.
///
/// This is useful for:
/// - Monte Carlo simulations with multiple orbital scenarios
/// - Constellation analysis with many satellites
/// - Batch processing of orbital predictions
///
/// The function uses Rayon's parallel iteration with the global thread pool configured
/// via `brahe::set_num_threads()`. Threading overhead is minimal, so this function
/// usually provides speedups even for small numbers of propagators on multi-core systems.
///
/// # Arguments
///
/// * `propagators` - Mutable slice of propagators to update
/// * `target_epoch` - The epoch to propagate all propagators to
///
/// # Examples
///
/// ```
/// use brahe::propagators::{KeplerianPropagator, par_propagate_to_s};
/// use brahe::traits::SStatePropagator;
/// use brahe::constants::AngleFormat;
/// use brahe::Epoch;
/// use nalgebra as na;
///
/// brahe::initialize_eop().unwrap();
///
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, brahe::TimeSystem::UTC);
///
/// // Create multiple propagators
/// let mut propagators = vec![
///     KeplerianPropagator::from_keplerian(
///         epoch,
///         na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
///         AngleFormat::Degrees,
///         60.0,
///     ),
///     KeplerianPropagator::from_keplerian(
///         epoch,
///         na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
///         AngleFormat::Degrees,
///         60.0,
///     ),
/// ];
///
/// // Propagate all to target epoch in parallel
/// let target = epoch + 3600.0; // 1 hour later
/// par_propagate_to_s(&mut propagators, target);
///
/// // All propagators are now at target epoch
/// assert_eq!(propagators[0].current_epoch(), target);
/// assert_eq!(propagators[1].current_epoch(), target);
/// ```
pub fn par_propagate_to_s<P: SStatePropagator + Send>(propagators: &mut [P], target_epoch: Epoch) {
    get_thread_pool().install(|| {
        propagators
            .par_iter_mut()
            .for_each(|prop| prop.propagate_to(target_epoch));
    });
}

/// Propagate multiple dynamic-state propagators to a target epoch in parallel.
///
/// This function is similar to `par_propagate_to_s` but works with propagators
/// that implement `DStatePropagator` (dynamic state vectors) instead of
/// `SStatePropagator` (static 6D vectors).
///
/// This is useful for:
/// - Numerical orbit propagators with STM/sensitivity matrices
/// - Monte Carlo simulations with numerical propagation
/// - Batch processing of high-fidelity orbital predictions
///
/// # Arguments
///
/// * `propagators` - Mutable slice of propagators to update
/// * `target_epoch` - The epoch to propagate all propagators to
pub fn par_propagate_to_d<P: super::traits::DStatePropagator + Send>(
    propagators: &mut [P],
    target_epoch: Epoch,
) {
    get_thread_pool().install(|| {
        propagators
            .par_iter_mut()
            .for_each(|prop| prop.propagate_to(target_epoch));
    });
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::AngleFormat;
    use crate::propagators::{KeplerianPropagator, SGPPropagator};
    use crate::time::Epoch;
    use crate::traits::SStatePropagator;
    use crate::utils::testing::setup_global_test_eop;
    use nalgebra as na;

    #[test]
    fn test_par_propagate_to_s_keplerian() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0; // 1 hour later

        // Create multiple Keplerian propagators with different initial conditions
        let mut propagators = vec![
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
                AngleFormat::Degrees,
                60.0,
            ),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
                AngleFormat::Degrees,
                60.0,
            ),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(6800e3, 0.0005, 51.6, 45.0, 90.0, 120.0),
                AngleFormat::Degrees,
                60.0,
            ),
        ];

        // Propagate in parallel
        par_propagate_to_s(&mut propagators, target);

        // Verify all propagators reached target epoch
        for prop in &propagators {
            assert_eq!(prop.current_epoch(), target);
        }

        // Verify states are different (they had different initial conditions)
        let state0 = propagators[0].current_state();
        let state1 = propagators[1].current_state();
        let state2 = propagators[2].current_state();

        assert_ne!(state0[0], state1[0]);
        assert_ne!(state0[0], state2[0]);
        assert_ne!(state1[0], state2[0]);
    }

    #[test]
    fn test_par_propagate_to_s_sgp() {
        setup_global_test_eop();

        // ISS TLE data (using same TLE multiple times to test parallel execution)
        let line1_iss = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2_iss = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let epoch_iss = SGPPropagator::from_tle(line1_iss, line2_iss, 60.0)
            .unwrap()
            .initial_epoch();

        // Create multiple propagators from same TLE
        let mut propagators = vec![
            SGPPropagator::from_tle(line1_iss, line2_iss, 60.0).unwrap(),
            SGPPropagator::from_tle(line1_iss, line2_iss, 60.0).unwrap(),
            SGPPropagator::from_tle(line1_iss, line2_iss, 60.0).unwrap(),
        ];

        // Propagate all forward 1 hour from TLE epoch
        let target = epoch_iss + 3600.0;
        par_propagate_to_s(&mut propagators, target);

        // Verify all reached target epoch
        for prop in &propagators {
            assert_eq!(prop.current_epoch(), target);
        }

        // Verify states are the same (same TLE, same propagation)
        let state0 = propagators[0].current_state();
        let state1 = propagators[1].current_state();
        let state2 = propagators[2].current_state();

        for i in 0..6 {
            assert!((state0[i] - state1[i]).abs() < 1e-9);
            assert!((state0[i] - state2[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_par_propagate_to_s_matches_sequential() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 7200.0; // 2 hours

        // Create identical propagators for parallel test
        let mut parallel_props = vec![
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
                AngleFormat::Degrees,
                60.0,
            ),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
                AngleFormat::Degrees,
                60.0,
            ),
        ];

        // Create identical propagators for sequential test
        let mut sequential_props = vec![
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
                AngleFormat::Degrees,
                60.0,
            ),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
                AngleFormat::Degrees,
                60.0,
            ),
        ];

        // Propagate in parallel
        par_propagate_to_s(&mut parallel_props, target);

        // Propagate sequentially
        for prop in &mut sequential_props {
            prop.propagate_to(target);
        }

        // Results should be identical
        for i in 0..parallel_props.len() {
            assert_eq!(
                parallel_props[i].current_epoch(),
                sequential_props[i].current_epoch()
            );

            let parallel_state = parallel_props[i].current_state();
            let sequential_state = sequential_props[i].current_state();

            for j in 0..6 {
                assert!((parallel_state[j] - sequential_state[j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_par_propagate_to_s_empty_slice() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let mut propagators: Vec<KeplerianPropagator> = vec![];

        // Should not panic with empty slice
        par_propagate_to_s(&mut propagators, target);
    }

    #[test]
    fn test_par_propagate_to_s_single_propagator() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let mut propagators = vec![KeplerianPropagator::from_keplerian(
            epoch,
            na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
            AngleFormat::Degrees,
            60.0,
        )];

        par_propagate_to_s(&mut propagators, target);

        assert_eq!(propagators[0].current_epoch(), target);
    }

    #[test]
    fn test_par_propagate_to_s_sgp_with_events() {
        use crate::events::DTimeEvent;

        setup_global_test_eop();

        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let mut propagators: Vec<SGPPropagator> = (0..3)
            .map(|_| SGPPropagator::from_tle(line1, line2, 60.0).unwrap())
            .collect();

        let epoch = propagators[0].epoch;

        // Add event detector to each propagator
        for (i, prop) in propagators.iter_mut().enumerate() {
            let event = DTimeEvent::new(epoch + 100.0 * (i + 1) as f64, format!("Event_{}", i));
            prop.add_event_detector(Box::new(event));
        }

        // Propagate in parallel
        let target = epoch + 400.0;
        par_propagate_to_s(&mut propagators, target);

        // Verify events were detected
        for (i, prop) in propagators.iter().enumerate() {
            assert!(
                !prop.event_log().is_empty(),
                "Propagator {} should have detected events",
                i
            );
            assert_eq!(
                prop.event_log().len(),
                1,
                "Propagator {} should have exactly 1 event",
                i
            );
            assert!(
                prop.event_log()[0].name.contains(&format!("Event_{}", i)),
                "Event name should contain Event_{}",
                i
            );
        }
    }

    // =========================================================================
    // par_propagate_to_d Tests (Dynamic State Propagators)
    // =========================================================================

    use crate::propagators::force_model_config::ForceModelConfig;
    use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig};
    use crate::traits::DStatePropagator;
    use nalgebra::DVector;

    #[test]
    fn test_par_propagate_to_d_numerical_orbit() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0; // 1 hour later

        // Create multiple DNumericalOrbitPropagators with different initial conditions
        let states = vec![
            DVector::from_vec(vec![crate::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0]),
            DVector::from_vec(vec![crate::R_EARTH + 600e3, 0.0, 0.0, 0.0, 7400.0, 0.0]),
            DVector::from_vec(vec![crate::R_EARTH + 400e3, 0.0, 0.0, 0.0, 7800.0, 0.0]),
        ];

        let mut propagators: Vec<DNumericalOrbitPropagator> = states
            .into_iter()
            .map(|state| {
                DNumericalOrbitPropagator::new(
                    epoch,
                    state,
                    NumericalPropagationConfig::default(),
                    ForceModelConfig::earth_gravity(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect();

        // Propagate in parallel
        par_propagate_to_d(&mut propagators, target);

        // Verify all propagators reached target epoch
        for prop in &propagators {
            assert_eq!(prop.current_epoch(), target);
        }

        // Verify states are different (they had different initial conditions)
        let state0 = propagators[0].current_state();
        let state1 = propagators[1].current_state();
        let state2 = propagators[2].current_state();

        assert!((state0[0] - state1[0]).abs() > 1e-3);
        assert!((state0[0] - state2[0]).abs() > 1e-3);
        assert!((state1[0] - state2[0]).abs() > 1e-3);
    }

    #[test]
    fn test_par_propagate_to_d_matches_sequential() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 1800.0; // 30 minutes

        let states = [
            DVector::from_vec(vec![crate::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0]),
            DVector::from_vec(vec![crate::R_EARTH + 600e3, 0.0, 0.0, 0.0, 7400.0, 0.0]),
        ];

        // Create identical propagators for parallel test
        let mut parallel_props: Vec<DNumericalOrbitPropagator> = states
            .iter()
            .map(|state| {
                DNumericalOrbitPropagator::new(
                    epoch,
                    state.clone(),
                    NumericalPropagationConfig::default(),
                    ForceModelConfig::earth_gravity(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect();

        // Create identical propagators for sequential test
        let mut sequential_props: Vec<DNumericalOrbitPropagator> = states
            .iter()
            .map(|state| {
                DNumericalOrbitPropagator::new(
                    epoch,
                    state.clone(),
                    NumericalPropagationConfig::default(),
                    ForceModelConfig::earth_gravity(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect();

        // Propagate in parallel
        par_propagate_to_d(&mut parallel_props, target);

        // Propagate sequentially
        for prop in &mut sequential_props {
            prop.propagate_to(target);
        }

        // Results should be identical
        for i in 0..parallel_props.len() {
            assert_eq!(
                parallel_props[i].current_epoch(),
                sequential_props[i].current_epoch()
            );

            let parallel_state = parallel_props[i].current_state();
            let sequential_state = sequential_props[i].current_state();

            for j in 0..6 {
                assert!(
                    (parallel_state[j] - sequential_state[j]).abs() < 1e-6,
                    "State element {} differs: parallel={}, sequential={}",
                    j,
                    parallel_state[j],
                    sequential_state[j]
                );
            }
        }
    }

    #[test]
    fn test_par_propagate_to_d_empty_slice() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let mut propagators: Vec<DNumericalOrbitPropagator> = vec![];

        // Should not panic with empty slice
        par_propagate_to_d(&mut propagators, target);
    }

    #[test]
    fn test_par_propagate_to_d_single_propagator() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let state = DVector::from_vec(vec![crate::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0]);

        let mut propagators = vec![
            DNumericalOrbitPropagator::new(
                epoch,
                state,
                NumericalPropagationConfig::default(),
                ForceModelConfig::earth_gravity(),
                None,
                None,
                None,
                None,
            )
            .unwrap(),
        ];

        par_propagate_to_d(&mut propagators, target);

        assert_eq!(propagators[0].current_epoch(), target);
    }

    #[test]
    fn test_par_propagate_to_d_with_events() {
        use crate::events::DTimeEvent;

        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let state = DVector::from_vec(vec![crate::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0]);

        let mut propagators: Vec<DNumericalOrbitPropagator> = (0..3)
            .map(|_| {
                DNumericalOrbitPropagator::new(
                    epoch,
                    state.clone(),
                    NumericalPropagationConfig::default(),
                    ForceModelConfig::earth_gravity(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect();

        // Add event detector to each propagator at different times
        for (i, prop) in propagators.iter_mut().enumerate() {
            let event =
                DTimeEvent::new(epoch + 600.0 * (i + 1) as f64, format!("OrbitEvent_{}", i));
            prop.add_event_detector(Box::new(event));
        }

        // Propagate in parallel
        let target = epoch + 2400.0;
        par_propagate_to_d(&mut propagators, target);

        // Verify events were detected
        for (i, prop) in propagators.iter().enumerate() {
            assert!(
                !prop.event_log().is_empty(),
                "Propagator {} should have detected events",
                i
            );
            assert_eq!(
                prop.event_log().len(),
                1,
                "Propagator {} should have exactly 1 event",
                i
            );
            assert!(
                prop.event_log()[0]
                    .name
                    .contains(&format!("OrbitEvent_{}", i)),
                "Event name should contain OrbitEvent_{}",
                i
            );
        }
    }
}
