/*!
 * Utility functions for orbit propagation operations
 */

use std::any::Any;
use std::sync::Arc;

use rayon::prelude::*;

use crate::propagators::DNumericalOrbitPropagator;
use crate::propagators::precomputed_inputs::{PrecomputeKey, PrecomputedInputs};
use crate::propagators::traits::{DStatePropagator, SStatePropagator};
use crate::time::Epoch;
use crate::utils::BraheError;
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
///     ).unwrap(),
///     KeplerianPropagator::from_keplerian(
///         epoch,
///         na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
///         AngleFormat::Degrees,
///         60.0,
///     ).unwrap(),
/// ];
///
/// // Propagate all to target epoch in parallel
/// let target = epoch + 3600.0; // 1 hour later
/// par_propagate_to_s(&mut propagators, target).unwrap();
///
/// // All propagators are now at target epoch
/// assert_eq!(propagators[0].current_epoch(), target);
/// assert_eq!(propagators[1].current_epoch(), target);
/// ```
pub fn par_propagate_to_s<P: SStatePropagator + Send>(
    propagators: &mut [P],
    target_epoch: Epoch,
) -> Result<(), BraheError> {
    get_thread_pool().install(|| {
        propagators
            .par_iter_mut()
            .try_for_each(|prop| prop.propagate_to(target_epoch))
    })
}

/// Propagate multiple dynamic-state propagators to a target epoch in parallel.
///
/// This function is similar to `par_propagate_to_s` but works with propagators
/// that implement `DStatePropagator` (dynamic state vectors) instead of
/// `SStatePropagator` (static 6D vectors).
///
/// Before propagating, [`DNumericalOrbitPropagator`] members whose
/// [`PrecomputeConfig`](crate::propagators::PrecomputeConfig) settings produce
/// tables holding the same values are grouped, and each group shares one set of
/// precomputed tables covering its members' spans, so the group evaluates each
/// table node once. Tables are built over the requested span of every
/// propagator not already terminated when the call starts, before any
/// propagator processes events, so a propagator that an event stops at or
/// after its start epoch still contributes its span, and an EOP or SPICE
/// coverage error for that span can be returned where serial `propagate_to`
/// would not reach it. Other propagator types build no shared tables.
///
/// This is useful for:
/// - Numerical orbit propagators with STM/sensitivity matrices
/// - Monte Carlo simulations with numerical propagation
/// - Batch processing of high-fidelity orbital predictions
///
/// # Arguments
///
/// * `propagators` - Mutable references to the propagators to update, such as
///   `&mut Vec<P>`, `&mut [P]`, or an iterator of `&mut P`
/// * `target_epoch` - The epoch to propagate all propagators to
///
/// # Returns
///
/// `Ok(())` once every propagator has been propagated toward the target epoch,
/// or the first error raised while building the shared precomputed inputs or
/// propagating. A propagator stopped by a terminal event remains at the epoch
/// where it stopped; every other propagator reaches the target epoch.
///
/// # Examples
///
/// ```rust
/// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
/// use brahe::propagators::{
///     DNumericalOrbitPropagator, ForceModelConfig, PrecomputeConfig, par_propagate_to_d,
/// };
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::traits::DStatePropagator;
/// use brahe::constants::R_EARTH;
/// use nalgebra::DVector;
///
/// set_global_eop_provider(StaticEOPProvider::from_zero());
///
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let force_config = ForceModelConfig {
///     precompute: Some(PrecomputeConfig::default()),
///     ..ForceModelConfig::earth_gravity()
/// };
/// let mut propagators: Vec<DNumericalOrbitPropagator> = [500e3, 600e3, 700e3]
///     .iter()
///     .map(|altitude| {
///         let state = DVector::from_vec(vec![R_EARTH + altitude, 0.0, 0.0, 0.0, 7500.0, 0.0]);
///         DNumericalOrbitPropagator::builder(epoch, state, force_config.clone())
///             .build()
///             .unwrap()
///     })
///     .collect();
///
/// // The three propagators share one set of precomputed tables.
/// let target = epoch + 3600.0;
/// par_propagate_to_d(&mut propagators, target).unwrap();
/// for prop in &propagators {
///     assert_eq!(prop.current_epoch(), target);
/// }
///
/// // Any iterator of mutable references is accepted.
/// let later = target + 600.0;
/// par_propagate_to_d(propagators.iter_mut().take(2), later).unwrap();
/// assert_eq!(propagators[1].current_epoch(), later);
/// assert_eq!(propagators[2].current_epoch(), target);
/// ```
pub fn par_propagate_to_d<'a, P: DStatePropagator + Send + 'static>(
    propagators: impl IntoIterator<Item = &'a mut P>,
    target_epoch: Epoch,
) -> Result<(), BraheError> {
    let mut propagators: Vec<&'a mut P> = propagators.into_iter().collect();
    share_precomputed_inputs(
        propagators.iter_mut().filter_map(|prop| {
            (&mut **prop as &mut dyn Any).downcast_mut::<DNumericalOrbitPropagator>()
        }),
        target_epoch,
    )?;
    get_thread_pool().install(|| {
        propagators
            .par_iter_mut()
            .try_for_each(|prop| prop.propagate_to(target_epoch))
    })
}

/// Propagators sharing one precompute key, and the span their bundle covers.
struct PrecomputeGroup<'a> {
    key: PrecomputeKey,
    start: Epoch,
    end: Epoch,
    members: Vec<&'a mut DNumericalOrbitPropagator>,
}

/// Build one precomputed-input bundle per group of compatible propagators and attach it.
///
/// Propagators are grouped by equal
/// [`DNumericalOrbitPropagator::precompute_key`]; propagators without a key,
/// including those already terminated, are left unchanged. No events are
/// processed here, so a propagator that an event stops at or after its current
/// epoch still contributes its span, and an EOP or SPICE coverage error for that
/// span can be returned where serial `propagate_to` would not reach it. Each
/// group's bundle covers the span from the earliest to the latest of the
/// members' current epochs and `target_epoch`. Its table nodes are computed in
/// parallel on the brahe thread pool before the bundle is attached to every
/// member with [`DNumericalOrbitPropagator::attach_precomputed`]. Propagating a
/// member within that span then reads the shared bundle without copying it.
///
/// # Arguments
///
/// * `propagators` - Propagators to prepare
/// * `target_epoch` - Epoch the propagators will be propagated to
///
/// # Returns
///
/// `Ok(())` once every group has its bundle, or the first error raised while
/// evaluating a table node or attaching a bundle.
pub(crate) fn share_precomputed_inputs<'a>(
    propagators: impl IntoIterator<Item = &'a mut DNumericalOrbitPropagator>,
    target_epoch: Epoch,
) -> Result<(), BraheError> {
    let mut groups: Vec<PrecomputeGroup<'a>> = Vec::new();
    for prop in propagators {
        let Some(key) = prop.precompute_key() else {
            continue;
        };
        let epoch = prop.current_epoch();
        let (start, end) = if epoch < target_epoch {
            (epoch, target_epoch)
        } else {
            (target_epoch, epoch)
        };
        match groups.iter_mut().find(|group| group.key == key) {
            Some(group) => {
                if start < group.start {
                    group.start = start;
                }
                if end > group.end {
                    group.end = end;
                }
                group.members.push(prop);
            }
            None => groups.push(PrecomputeGroup {
                key,
                start,
                end,
                members: vec![prop],
            }),
        }
    }
    for group in groups {
        let mut inputs = PrecomputedInputs::new(&group.key);
        inputs.ensure(group.start, group.end)?;
        let inputs = Arc::new(inputs);
        for prop in group.members {
            prop.attach_precomputed(Arc::clone(&inputs))?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::AngleFormat;
    use crate::events::DTimeEvent;
    use crate::integrators::IntegratorConfig;
    use crate::propagators::{IntegratorMethod, PrecomputeConfig};
    use crate::propagators::{KeplerianPropagator, SGPPropagator};
    use crate::time::Epoch;
    use crate::time::TimeSystem;
    use crate::traits::SStatePropagator;
    use crate::utils::testing::setup_global_test_eop;
    use crate::{R_EARTH, state_koe_to_eci};
    use nalgebra as na;
    use serial_test::{parallel, serial};

    #[test]
    #[parallel]
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
            )
            .unwrap(),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
                AngleFormat::Degrees,
                60.0,
            )
            .unwrap(),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(6800e3, 0.0005, 51.6, 45.0, 90.0, 120.0),
                AngleFormat::Degrees,
                60.0,
            )
            .unwrap(),
        ];

        // Propagate in parallel
        par_propagate_to_s(&mut propagators, target).unwrap();

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
    #[parallel]
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
        par_propagate_to_s(&mut propagators, target).unwrap();

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
    #[parallel]
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
            )
            .unwrap(),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
                AngleFormat::Degrees,
                60.0,
            )
            .unwrap(),
        ];

        // Create identical propagators for sequential test
        let mut sequential_props = vec![
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
                AngleFormat::Degrees,
                60.0,
            )
            .unwrap(),
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7200e3, 0.002, 97.0, 10.0, 20.0, 30.0),
                AngleFormat::Degrees,
                60.0,
            )
            .unwrap(),
        ];

        // Propagate in parallel
        par_propagate_to_s(&mut parallel_props, target).unwrap();

        // Propagate sequentially
        for prop in &mut sequential_props {
            prop.propagate_to(target).unwrap();
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
    #[parallel]
    fn test_par_propagate_to_s_empty_slice() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let mut propagators: Vec<KeplerianPropagator> = vec![];

        // Should not panic with empty slice
        par_propagate_to_s(&mut propagators, target).unwrap();
    }

    #[test]
    #[parallel]
    fn test_par_propagate_to_s_single_propagator() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let mut propagators = vec![
            KeplerianPropagator::from_keplerian(
                epoch,
                na::SVector::<f64, 6>::new(7000e3, 0.001, 98.0, 0.0, 0.0, 0.0),
                AngleFormat::Degrees,
                60.0,
            )
            .unwrap(),
        ];

        par_propagate_to_s(&mut propagators, target).unwrap();

        assert_eq!(propagators[0].current_epoch(), target);
    }

    #[test]
    #[parallel]
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
        par_propagate_to_s(&mut propagators, target).unwrap();

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

    /// Create an SGPPropagator for a decaying/reentering satellite (STARLINK-31304)
    /// that triggers SGP4 eccentricity divergence when propagated ~24 hours.
    fn make_decaying_propagator(step_size: f64) -> SGPPropagator {
        SGPPropagator::from_omm_elements(
            Epoch::from_datetime(2026, 4, 6, 0, 15, 48.265056, 0.0, TimeSystem::UTC),
            16.25673795,
            0.00191239,
            42.9658,
            302.8224,
            303.9951,
            55.9119,
            59231,
            step_size,
            Some("STARLINK-31304"),
            Some("2024-049A"),
            Some('U'),
            Some(0.0038650148),
            Some(0.13132014),
            Some(9.155391e-06),
            Some(0),
            Some(999),
            Some(0),
        )
        .unwrap()
    }

    #[test]
    #[parallel]
    fn test_par_propagate_with_divergent_propagator() {
        setup_global_test_eop();

        // ISS TLE - well-behaved orbit
        let iss_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let iss_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        // Use large step size since each SGP4 evaluation is independent
        let step = 3600.0;

        let mut propagators = vec![
            make_decaying_propagator(step),
            make_decaying_propagator(step),
        ];
        let epoch = propagators[0].initial_epoch();

        // Target 2 days — this satellite diverges within ~24 hours
        let target = epoch + 2.0 * 86400.0;
        par_propagate_to_s(&mut propagators, target).unwrap();

        // Both propagators should have terminated due to divergence
        // The key test is that NO PANIC occurred - the batch completed
        for prop in &propagators {
            assert!(prop.is_terminated());
            assert!(prop.termination_error().is_some());
        }

        // Test mixed case: good orbit (ISS) propagated short, decaying orbit propagated past divergence
        let mut prop_good = SGPPropagator::from_tle(iss_line1, iss_line2, step).unwrap();
        let mut prop_decay = make_decaying_propagator(step);

        let short_target = prop_good.initial_epoch() + 3600.0; // 1 hour
        prop_good.propagate_to(short_target).unwrap();
        prop_decay.propagate_to(target).unwrap();

        // Short propagation of good orbit should succeed
        assert!(!prop_good.is_terminated());
        assert!(prop_good.termination_error().is_none());
        assert_eq!(prop_good.current_epoch(), short_target);

        // Decaying orbit should terminate with error
        assert!(prop_decay.is_terminated());
        assert!(prop_decay.termination_error().is_some());
        assert!(prop_decay.current_epoch() < target);
    }

    // =========================================================================
    // par_propagate_to_d Tests (Dynamic State Propagators)
    // =========================================================================

    use crate::propagators::force_model_config::ForceModelConfig;
    use crate::propagators::{DNumericalOrbitPropagator, NumericalPropagationConfig};
    use crate::traits::DStatePropagator;
    use nalgebra::DVector;

    #[test]
    #[parallel]
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
        par_propagate_to_d(&mut propagators, target).unwrap();

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
    #[serial]
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
        par_propagate_to_d(&mut parallel_props, target).unwrap();

        // Propagate sequentially
        for prop in &mut sequential_props {
            prop.propagate_to(target).unwrap();
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
    #[parallel]
    fn test_par_propagate_to_d_empty_slice() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::TimeSystem::UTC);
        let target = epoch + 3600.0;

        let mut propagators: Vec<DNumericalOrbitPropagator> = vec![];

        // Should not panic with empty slice
        par_propagate_to_d(&mut propagators, target).unwrap();
    }

    #[test]
    #[parallel]
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

        par_propagate_to_d(&mut propagators, target).unwrap();

        assert_eq!(propagators[0].current_epoch(), target);
    }

    #[test]
    #[parallel]
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
        par_propagate_to_d(&mut propagators, target).unwrap();

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

    // =========================================================================
    // share_precomputed_inputs Tests
    // =========================================================================

    fn precompute_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    fn precompute_rk4_30s() -> NumericalPropagationConfig {
        NumericalPropagationConfig {
            method: IntegratorMethod::RK4,
            integrator: IntegratorConfig {
                initial_step: Some(30.0),
                ..IntegratorConfig::fixed_step(30.0)
            },
            ..Default::default()
        }
    }

    fn precompute_prop(
        epoch: Epoch,
        precompute: Option<PrecomputeConfig>,
    ) -> DNumericalOrbitPropagator {
        let oe = na::Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let state = state_koe_to_eci(oe, AngleFormat::Degrees);
        DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_column_slice(state.as_slice()),
            precompute_rk4_30s(),
            ForceModelConfig {
                precompute,
                ..ForceModelConfig::earth_gravity()
            },
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn rotation_step(step: f64) -> Option<PrecomputeConfig> {
        Some(PrecomputeConfig {
            frame_rotation_step: Some(step),
            ..PrecomputeConfig::default()
        })
    }

    #[test]
    #[parallel]
    fn test_share_precomputed_inputs_same_key_shares_arc() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let mut props: Vec<_> = (0..3)
            .map(|_| precompute_prop(t0, rotation_step(600.0)))
            .collect();

        share_precomputed_inputs(props.iter_mut(), t0 + 3600.0).unwrap();

        let shared = props[0].precomputed_inputs().unwrap();
        assert!(shared.covers_span(t0, t0 + 3600.0));
        for prop in &props[1..] {
            assert!(Arc::ptr_eq(&prop.precomputed_inputs().unwrap(), &shared));
        }
    }

    #[test]
    #[parallel]
    fn test_share_precomputed_inputs_distinct_keys() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let mut props = [
            precompute_prop(t0, rotation_step(600.0)),
            precompute_prop(t0, rotation_step(300.0)),
            precompute_prop(t0, rotation_step(600.0)),
            precompute_prop(t0, rotation_step(300.0)),
            precompute_prop(t0, None),
        ];

        share_precomputed_inputs(props.iter_mut(), t0 + 3600.0).unwrap();

        let coarse = props[0].precomputed_inputs().unwrap();
        let fine = props[1].precomputed_inputs().unwrap();
        assert!(!Arc::ptr_eq(&coarse, &fine));
        assert!(Arc::ptr_eq(
            &props[2].precomputed_inputs().unwrap(),
            &coarse
        ));
        assert!(Arc::ptr_eq(&props[3].precomputed_inputs().unwrap(), &fine));
        assert!(coarse.covers_span(t0, t0 + 3600.0));
        assert!(fine.covers_span(t0, t0 + 3600.0));
        assert!(props[4].precomputed_inputs().is_none());
    }

    #[test]
    #[parallel]
    fn test_attach_precomputed_mid_propagation_matches_unattached() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let mut attached = precompute_prop(t0, rotation_step(600.0));
        let mut own = precompute_prop(t0, rotation_step(600.0));
        attached.propagate_to(t0 + 3600.0).unwrap();
        own.propagate_to(t0 + 3600.0).unwrap();

        let mut inputs = PrecomputedInputs::new(&attached.precompute_key().unwrap());
        inputs.ensure(t0 + 3600.0, t0 + 86400.0).unwrap();
        let inputs = Arc::new(inputs);
        attached.attach_precomputed(Arc::clone(&inputs)).unwrap();

        attached.propagate_to(t0 + 86400.0).unwrap();
        own.propagate_to(t0 + 86400.0).unwrap();

        assert!(Arc::ptr_eq(
            &attached.precomputed_inputs().unwrap(),
            &inputs
        ));
        assert_eq!(attached.current_epoch(), own.current_epoch());
        assert_eq!(attached.current_state(), own.current_state());
    }

    #[test]
    #[parallel]
    fn test_share_precomputed_inputs_survives_propagation() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let mut props: Vec<_> = (0..3)
            .map(|_| precompute_prop(t0, rotation_step(600.0)))
            .collect();
        share_precomputed_inputs(props.iter_mut(), t0 + 86400.0).unwrap();
        let shared = props[0].precomputed_inputs().unwrap();

        for prop in &mut props {
            prop.propagate_to(t0 + 43200.0).unwrap();
        }

        for prop in &props {
            assert_eq!(prop.current_epoch(), t0 + 43200.0);
            assert!(Arc::ptr_eq(&prop.precomputed_inputs().unwrap(), &shared));
        }
    }

    #[test]
    #[parallel]
    fn test_par_propagate_to_d_precompute_matches_serial() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let target = t0 + 43200.0;
        let build = || -> Vec<DNumericalOrbitPropagator> {
            (0..8)
                .map(|i| precompute_prop(t0 + 3600.0 * i as f64, rotation_step(600.0)))
                .collect()
        };
        let mut parallel_props = build();
        let mut serial_props = build();

        par_propagate_to_d(&mut parallel_props, target).unwrap();
        for prop in &mut serial_props {
            prop.propagate_to(target).unwrap();
        }

        let shared = parallel_props[0].precomputed_inputs().unwrap();
        for (par, ser) in parallel_props.iter().zip(&serial_props) {
            assert!(Arc::ptr_eq(&par.precomputed_inputs().unwrap(), &shared));
            assert_eq!(par.current_epoch(), target);
            assert_eq!(ser.current_epoch(), target);
            assert_eq!(par.current_state(), ser.current_state());
        }
    }

    #[test]
    #[parallel]
    fn test_par_propagate_to_d_terminated_propagator_not_shared() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let target = t0 + 7200.0;

        let mut stopped = precompute_prop(t0, rotation_step(600.0));
        stopped.add_event_detector(Box::new(DTimeEvent::new(t0 + 600.0, "Stop").set_terminal()));
        stopped.propagate_to(t0 + 1200.0).unwrap();
        assert!(stopped.is_terminated());
        assert!(stopped.precompute_key().is_none());
        let stopped_epoch = stopped.current_epoch();
        let stopped_inputs = stopped.precomputed_inputs().unwrap();
        let stopped_nodes = stopped.precomputed_nodes_computed();

        let mut props = vec![
            stopped,
            precompute_prop(t0 + 3600.0, rotation_step(600.0)),
            precompute_prop(t0 + 3600.0, rotation_step(600.0)),
        ];
        par_propagate_to_d(&mut props, target).unwrap();

        assert_eq!(props[0].current_epoch(), stopped_epoch);
        assert!(Arc::ptr_eq(
            &props[0].precomputed_inputs().unwrap(),
            &stopped_inputs
        ));
        assert_eq!(props[0].precomputed_nodes_computed(), stopped_nodes);

        let shared = props[1].precomputed_inputs().unwrap();
        assert!(Arc::ptr_eq(
            &props[2].precomputed_inputs().unwrap(),
            &shared
        ));
        assert!(shared.covers_span(t0 + 3600.0, target));
        assert!(!shared.covers(stopped_epoch));
        for prop in &props[1..] {
            assert_eq!(prop.current_epoch(), target);
        }
    }

    #[test]
    #[parallel]
    fn test_par_propagate_to_d_precompute_mixed_directions_matches_serial() {
        setup_global_test_eop();
        let t0 = precompute_epoch();
        let target = t0 + 3600.0;
        // The second member starts before the first and the third propagates
        // backward from after the target, so the group span widens at both
        // ends to [t0, t0 + 7200 s].
        let build = || -> Vec<DNumericalOrbitPropagator> {
            [1800.0, 0.0, 7200.0]
                .iter()
                .map(|dt| precompute_prop(t0 + *dt, rotation_step(600.0)))
                .collect()
        };
        let mut parallel_props = build();
        let mut serial_props = build();

        par_propagate_to_d(&mut parallel_props, target).unwrap();
        for prop in &mut serial_props {
            prop.propagate_to(target).unwrap();
        }

        let shared = parallel_props[0].precomputed_inputs().unwrap();
        assert!(shared.covers_span(t0, t0 + 7200.0));
        for (par, ser) in parallel_props.iter().zip(&serial_props) {
            assert!(Arc::ptr_eq(&par.precomputed_inputs().unwrap(), &shared));
            assert_eq!(par.current_epoch(), target);
            assert_eq!(par.current_state(), ser.current_state());
        }
    }

    #[test]
    #[serial]
    fn test_par_propagate_to_d_precompute_table_error_propagates() {
        // de440s ends on 2150-01-22, so the shared Sun ephemeris table cannot
        // be built over this span and no propagator advances.
        setup_global_test_eop();
        crate::utils::testing::setup_global_test_spice();
        let t0 = Epoch::from_datetime(2150, 1, 21, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let oe = na::Vector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let state = state_koe_to_eci(oe, AngleFormat::Degrees);
        let mut props: Vec<DNumericalOrbitPropagator> = (0..2)
            .map(|_| {
                DNumericalOrbitPropagator::new(
                    t0,
                    DVector::from_column_slice(state.as_slice()),
                    precompute_rk4_30s(),
                    ForceModelConfig {
                        third_body: Some(vec![
                            crate::propagators::force_model_config::ThirdBody::Sun.into(),
                        ]),
                        precompute: Some(PrecomputeConfig::default()),
                        ..ForceModelConfig::earth_gravity()
                    },
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect();

        assert!(par_propagate_to_d(&mut props, t0 + 2.0 * 86400.0).is_err());
        for prop in &props {
            assert_eq!(prop.current_epoch(), t0);
        }
    }
}
