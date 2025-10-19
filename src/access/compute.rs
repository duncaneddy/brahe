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
use crate::orbits::traits::IdentifiableStateProvider;
use crate::time::Epoch;

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

/// Trait to convert various propagator inputs into a slice of references.
///
/// This trait enables the unified `location_accesses` function to accept
/// either single propagators or slices/vectors of propagators.
pub(crate) trait ToPropagatorRefs<P: IdentifiableStateProvider> {
    fn to_refs(&self) -> Vec<&P>;
}

// Single propagator reference
impl<P: IdentifiableStateProvider> ToPropagatorRefs<P> for P {
    fn to_refs(&self) -> Vec<&P> {
        vec![self]
    }
}

// Slice of propagators
impl<P: IdentifiableStateProvider> ToPropagatorRefs<P> for [P] {
    fn to_refs(&self) -> Vec<&P> {
        self.iter().collect()
    }
}

// Vec of propagators
impl<P: IdentifiableStateProvider> ToPropagatorRefs<P> for Vec<P> {
    fn to_refs(&self) -> Vec<&P> {
        self.iter().collect()
    }
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
/// Vector of `AccessWindow` objects sorted by start time
///
/// # Examples
/// ```ignore
/// use brahe::access::*;
/// use brahe::orbits::KeplerianPropagator;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let location = PointLocation::new(0.0, 45.0, 0.0);
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let propagator = KeplerianPropagator::new(/* ... */);
/// let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();
///
/// // Single location, single propagator
/// let windows = location_accesses(
///     &location,
///     &propagator,
///     epoch,
///     epoch + 86400.0,
///     &constraint,
///     None,
///     None,
///     None,
/// );
///
/// // Single location, multiple propagators
/// let propagators = vec![propagator1, propagator2];
/// let windows = location_accesses(
///     &location,
///     &propagators,
///     epoch,
///     epoch + 86400.0,
///     &constraint,
///     None,
///     None,
///     None,
/// );
///
/// // Multiple locations, single propagator
/// let locations = vec![location1, location2];
/// let windows = location_accesses(
///     &locations,
///     &propagator,
///     epoch,
///     epoch + 86400.0,
///     &constraint,
///     None,
///     None,
///     None,
/// );
///
/// // Multiple locations, multiple propagators
/// let windows = location_accesses(
///     &locations,
///     &propagators,
///     epoch,
///     epoch + 86400.0,
///     &constraint,
///     None,
///     None,
///     None,
/// );
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
) -> Vec<AccessWindow>
where
    L: AccessibleLocation,
    P: IdentifiableStateProvider,
    Locs: ToLocationRefs<L> + ?Sized,
    Props: ToPropagatorRefs<P> + ?Sized,
{
    let search_config = config.copied().unwrap_or_default();

    let loc_refs = locations.to_refs();
    let prop_refs = propagators.to_refs();

    // Process all location-propagator combinations
    let mut all_windows = Vec::new();

    for location in &loc_refs {
        for propagator in &prop_refs {
            let mut windows = find_access_windows(
                *location,
                *propagator,
                search_start,
                search_end,
                constraint,
                property_computers,
                Some(search_config.initial_time_step),
                time_tolerance,
            );
            all_windows.append(&mut windows);
        }
    }

    // Sort by window start time
    all_windows.sort_by(|a, b| {
        a.window_open
            .partial_cmp(&b.window_open)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    all_windows
}

// ================================
// Tests
// ================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::constraints::ElevationConstraint;
    use crate::access::location::PointLocation;
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::orbits::keplerian_propagator::KeplerianPropagator;
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
        );

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
        );

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
        );

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
        );

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
}
