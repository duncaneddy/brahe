/*!
 * Access computation API
 *
 * This module provides a unified function for computing access windows
 * between satellites and ground locations. It integrates window finding,
 * constraint evaluation, and property computation with an ergonomic API
 * that accepts both single items and slices.
 */

use crate::access::constraints::AccessConstraint;
use crate::access::location::AccessibleLocation;
use crate::access::properties::AccessPropertyComputer;
use crate::access::windows::{AccessSearchConfig, AccessWindow, find_access_windows};
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::utils::state_providers::{DIdentifiableStateProvider, ToPropagatorRefs};
use crate::utils::threading::get_thread_pool;
use rayon::prelude::*;

// ================================
// Conversion Traits for Ergonomic API
// ================================

/// Trait to convert various location inputs into a slice of references.
///
/// This trait enables the unified `location_accesses` function to accept
/// either single locations or slices/vectors of locations.
pub(crate) trait ToLocationRefs<L: AccessibleLocation> {
    fn to_refs(&self) -> Vec<&L>;
}

// Single location reference
impl<L: AccessibleLocation> ToLocationRefs<L> for L {
    fn to_refs(&self) -> Vec<&L> {
        vec![self]
    }
}

// Slice of locations
impl<L: AccessibleLocation> ToLocationRefs<L> for [L] {
    fn to_refs(&self) -> Vec<&L> {
        self.iter().collect()
    }
}

// Vec of locations
impl<L: AccessibleLocation> ToLocationRefs<L> for Vec<L> {
    fn to_refs(&self) -> Vec<&L> {
        self.iter().collect()
    }
}

// ================================
// Internal Computation Functions
// ================================

/// Sequential access computation (for debugging or single-threaded operation)
#[allow(clippy::too_many_arguments)]
fn compute_accesses_sequential<L, P>(
    locations: &[&L],
    propagators: &[&P],
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    search_config: &AccessSearchConfig,
    time_tolerance: Option<f64>,
) -> Result<Vec<AccessWindow>, BraheError>
where
    L: AccessibleLocation,
    P: DIdentifiableStateProvider,
{
    let mut all_windows = Vec::new();

    for location in locations {
        for propagator in propagators {
            let mut windows = find_access_windows(
                *location,
                *propagator,
                search_start,
                search_end,
                constraint,
                property_computers,
                Some(search_config.initial_time_step),
                time_tolerance,
            )?;
            all_windows.append(&mut windows);
        }
    }

    // Sort by window start time
    all_windows.sort_by(|a, b| {
        a.window_open
            .partial_cmp(&b.window_open)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(all_windows)
}

/// Parallel access computation using rayon
#[allow(clippy::too_many_arguments)]
fn compute_accesses_parallel<L, P>(
    locations: &[&L],
    propagators: &[&P],
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    search_config: &AccessSearchConfig,
    time_tolerance: Option<f64>,
) -> Result<Vec<AccessWindow>, BraheError>
where
    L: AccessibleLocation + Sync,
    P: DIdentifiableStateProvider + Sync,
{
    // Create all location-propagator pairs
    let pairs: Vec<(&L, &P)> = locations
        .iter()
        .flat_map(|loc| propagators.iter().map(move |prop| (*loc, *prop)))
        .collect();

    // Compute windows in parallel
    let results: Result<Vec<Vec<AccessWindow>>, BraheError> = pairs
        .par_iter()
        .map(|(location, propagator)| {
            find_access_windows(
                *location,
                *propagator,
                search_start,
                search_end,
                constraint,
                property_computers,
                Some(search_config.initial_time_step),
                time_tolerance,
            )
        })
        .collect();

    let mut all_windows: Vec<AccessWindow> = results?.into_iter().flatten().collect();

    // Sort by window start time
    all_windows.sort_by(|a, b| {
        a.window_open
            .partial_cmp(&b.window_open)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(all_windows)
}

// ================================
// Unified Access Computation API
// ================================

/// Compute access windows for locations and propagators.
///
/// This function accepts either single items or slices/vectors for both
/// locations and propagators, automatically handling the conversion.
/// All location-propagator pairs are computed and results are returned
/// sorted by window start time.
///
/// # Arguments
/// * `locations` - Single location or slice/vec of locations
/// * `propagators` - Single propagator or slice/vec of propagators
/// * `search_start` - Start of search window
/// * `search_end` - End of search window
/// * `constraint` - Access constraints to evaluate
/// * `property_computers` - Optional custom property computers
/// * `config` - Optional search configuration (time step, adaptive parameters)
/// * `time_tolerance` - Optional boundary refinement tolerance (default: 0.01s)
///
/// # Returns
/// Result containing vector of `AccessWindow` objects sorted by start time,
/// or error if property computation fails
///
/// # Examples
/// ```
/// use brahe::access::*;
/// use brahe::eop::*;
/// use brahe::propagators::KeplerianPropagator;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector6;
///
/// // Set up EOP
/// let eop = StaticEOPProvider::from_values((0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
/// set_global_eop_provider(eop);
///
/// // Create a location (latitude, longitude in degrees, altitude in meters)
/// let location = PointLocation::new(40.7128, -74.0060, 0.0);
///
/// // Create epoch and orbital state
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = Vector6::new(R_EARTH + 600e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
/// let propagator = KeplerianPropagator::from_eci(epoch, state, 60.0);
///
/// // Create constraint
/// let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();
///
/// // Compute access windows for 24 hours
/// let windows = location_accesses(
///     &location,
///     &propagator,
///     epoch,
///     epoch + 86400.0,
///     &constraint,
///     None,
///     None,
///     None,
/// ).unwrap();
///
/// // Windows contains all periods when satellite is above 10 degrees elevation
/// for window in windows {
///     println!("Access from {} to {}", window.window_open, window.window_close);
/// }
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(private_bounds)]
pub fn location_accesses<L, P, Locs, Props>(
    locations: &Locs,
    propagators: &Props,
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    config: Option<&AccessSearchConfig>,
    time_tolerance: Option<f64>,
) -> Result<Vec<AccessWindow>, BraheError>
where
    L: AccessibleLocation + Sync,
    P: DIdentifiableStateProvider + Sync,
    Locs: ToLocationRefs<L> + ?Sized,
    Props: ToPropagatorRefs<P> + ?Sized,
{
    let search_config = config.copied().unwrap_or_default();

    let loc_refs = locations.to_refs();
    let prop_refs = propagators.to_refs();

    // Process all location-propagator combinations

    if search_config.parallel {
        // Parallel computation using rayon
        if let Some(n_threads) = search_config.num_threads {
            // Use custom thread pool with specific thread count
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("Failed to build thread pool");

            pool.install(|| {
                compute_accesses_parallel(
                    &loc_refs,
                    &prop_refs,
                    search_start,
                    search_end,
                    constraint,
                    property_computers,
                    &search_config,
                    time_tolerance,
                )
            })
        } else {
            // Use global thread pool (default: 90% of cores)
            get_thread_pool().install(|| {
                compute_accesses_parallel(
                    &loc_refs,
                    &prop_refs,
                    search_start,
                    search_end,
                    constraint,
                    property_computers,
                    &search_config,
                    time_tolerance,
                )
            })
        }
    } else {
        // Sequential computation
        compute_accesses_sequential(
            &loc_refs,
            &prop_refs,
            search_start,
            search_end,
            constraint,
            property_computers,
            &search_config,
            time_tolerance,
        )
    }
}

// ================================
// Tests
// ================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::access::constraints::ElevationConstraint;
    use crate::access::location::PointLocation;
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::propagators::keplerian_propagator::KeplerianPropagator;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;
    use nalgebra::Vector6;

    fn create_test_propagator(epoch: Epoch) -> KeplerianPropagator {
        let oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0_f64.to_radians(), 0.0, 0.0, 0.0);
        KeplerianPropagator::new(
            epoch,
            oe,
            crate::trajectories::traits::OrbitFrame::ECI,
            crate::trajectories::traits::OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        )
    }

    #[test]
    fn test_location_accesses_single() {
        setup_global_test_eop();

        let location = PointLocation::new(45.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = create_test_propagator(epoch);

        let period = 5674.0;
        let search_end = epoch + (period * 2.0);

        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        let config = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
            parallel: true,
            num_threads: None,
        };

        let windows = location_accesses(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
            Some(0.1),
        )
        .unwrap();

        // Should find at least one window
        assert!(
            !windows.is_empty(),
            "Expected at least 1 window, found {}",
            windows.len()
        );

        // Verify windows are sorted
        for i in 1..windows.len() {
            assert!(windows[i - 1].window_open <= windows[i].window_open);
        }
    }

    #[test]
    fn test_location_accesses_multiple_sats() {
        setup_global_test_eop();

        let location = PointLocation::new(45.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Create 3 propagators with different RAANs
        let propagators = vec![
            create_test_propagator(epoch),
            {
                let oe = Vector6::new(
                    R_EARTH + 500e3,
                    0.0,
                    45.0_f64.to_radians(),
                    60.0_f64.to_radians(), // Different RAAN
                    0.0,
                    0.0,
                );
                KeplerianPropagator::new(
                    epoch,
                    oe,
                    crate::trajectories::traits::OrbitFrame::ECI,
                    crate::trajectories::traits::OrbitRepresentation::Keplerian,
                    Some(AngleFormat::Radians),
                    60.0,
                )
            },
            {
                let oe = Vector6::new(
                    R_EARTH + 500e3,
                    0.0,
                    45.0_f64.to_radians(),
                    120.0_f64.to_radians(), // Different RAAN
                    0.0,
                    0.0,
                );
                KeplerianPropagator::new(
                    epoch,
                    oe,
                    crate::trajectories::traits::OrbitFrame::ECI,
                    crate::trajectories::traits::OrbitRepresentation::Keplerian,
                    Some(AngleFormat::Radians),
                    60.0,
                )
            },
        ];

        let period = 5674.0;
        let search_end = epoch + (period * 2.0);

        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        let config = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
            parallel: true,
            num_threads: None,
        };

        let windows = location_accesses(
            &location,
            &propagators,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
            Some(0.1),
        )
        .unwrap();

        // Should find windows from multiple satellites
        assert!(
            !windows.is_empty(),
            "Expected at least 1 window, found {}",
            windows.len()
        );

        // Verify windows are sorted
        for i in 1..windows.len() {
            assert!(windows[i - 1].window_open <= windows[i].window_open);
        }
    }

    #[test]
    fn test_location_accesses_multiple_locations() {
        setup_global_test_eop();

        let locations = vec![
            PointLocation::new(0.0, 45.0, 0.0),    // 45°N, 0°E (lon, lat, alt)
            PointLocation::new(-120.0, 30.0, 0.0), // 30°N, 120°W
        ];

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = create_test_propagator(epoch);

        let period = 5674.0;
        let search_end = epoch + (period * 3.0); // More time to ensure access to both locations

        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        let config = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
            parallel: true,
            num_threads: None,
        };

        let windows = location_accesses(
            &locations,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
            Some(0.1),
        )
        .unwrap();

        // Should find windows for multiple locations
        assert!(
            !windows.is_empty(),
            "Expected at least 1 window, found {}",
            windows.len()
        );

        // Verify windows are sorted
        for i in 1..windows.len() {
            assert!(windows[i - 1].window_open <= windows[i].window_open);
        }
    }

    #[test]
    fn test_location_accesses_multiple() {
        setup_global_test_eop();

        let locations = vec![
            PointLocation::new(0.0, 45.0, 0.0),    // 45°N, 0°E (lon, lat, alt)
            PointLocation::new(-120.0, 30.0, 0.0), // 30°N, 120°W
        ];

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let propagators = vec![create_test_propagator(epoch), {
            let oe = Vector6::new(
                R_EARTH + 500e3,
                0.0,
                45.0_f64.to_radians(),
                60.0_f64.to_radians(),
                0.0,
                0.0,
            );
            KeplerianPropagator::new(
                epoch,
                oe,
                crate::trajectories::traits::OrbitFrame::ECI,
                crate::trajectories::traits::OrbitRepresentation::Keplerian,
                Some(AngleFormat::Radians),
                60.0,
            )
        }];

        let period = 5674.0;
        let search_end = epoch + (period * 3.0);

        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        let config = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
            parallel: true,
            num_threads: None,
        };

        let windows = location_accesses(
            &locations,
            &propagators,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
            Some(0.1),
        )
        .unwrap();

        // Should find windows for multiple location-satellite pairs
        assert!(
            !windows.is_empty(),
            "Expected at least 1 window, found {}",
            windows.len()
        );

        // Verify windows are sorted
        for i in 1..windows.len() {
            assert!(windows[i - 1].window_open <= windows[i].window_open);
        }
    }

    #[test]
    fn test_location_accesses_sequential() {
        setup_global_test_eop();

        let location = PointLocation::new(45.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = create_test_propagator(epoch);

        let period = 5674.0;
        let search_end = epoch + (period * 2.0);

        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        // Test with parallel: false to exercise sequential code path
        let config = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
            parallel: false, // Use sequential computation
            num_threads: None,
        };

        let windows = location_accesses(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
            Some(0.1),
        )
        .unwrap();

        // Should find at least one window
        assert!(
            !windows.is_empty(),
            "Expected at least 1 window, found {}",
            windows.len()
        );

        // Verify windows are sorted
        for i in 1..windows.len() {
            assert!(windows[i - 1].window_open <= windows[i].window_open);
        }
    }
}
