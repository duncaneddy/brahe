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
fn compute_accesses_sequential<L, P>(
    locations: &[&L],
    propagators: &[&P],
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    search_config: &AccessSearchConfig,
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
                Some(search_config),
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
fn compute_accesses_parallel<L, P>(
    locations: &[&L],
    propagators: &[&P],
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    search_config: &AccessSearchConfig,
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
                Some(search_config),
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
/// * `config` - Optional search configuration (time step, adaptive parameters, tolerance, subdivisions)
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
/// ).unwrap();
///
/// // Windows contains all periods when satellite is above 10 degrees elevation
/// for window in windows {
///     println!("Access from {} to {}", window.window_open, window.window_close);
/// }
/// ```
#[allow(private_bounds)]
pub fn location_accesses<L, P, Locs, Props>(
    locations: &Locs,
    propagators: &Props,
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    config: Option<&AccessSearchConfig>,
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
            time_tolerance: 0.1,
            ..Default::default()
        };

        let windows = location_accesses(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
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
            time_tolerance: 0.1,
            ..Default::default()
        };

        let windows = location_accesses(
            &location,
            &propagators,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
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
            time_tolerance: 0.1,
            ..Default::default()
        };

        let windows = location_accesses(
            &locations,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
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
            time_tolerance: 0.1,
            ..Default::default()
        };

        let windows = location_accesses(
            &locations,
            &propagators,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
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
            time_tolerance: 0.1,
            ..Default::default()
        };

        let windows = location_accesses(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
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
    fn test_access_identification_traceability() {
        use crate::propagators::sgp_propagator::SGPPropagator;
        use crate::utils::Identifiable;

        setup_global_test_eop();

        // -- Locations --
        let new_york = PointLocation::new(-74.006, 40.7128, 0.0)
            .with_name("NewYork")
            .with_id(1);
        let london = PointLocation::new(-0.1276, 51.5074, 0.0)
            .with_name("London")
            .with_id(2);
        let locations = vec![new_york, london];

        // -- Propagators (SGP4 from 3LE) --
        // ISS (NORAD 25544) - 2026 epoch
        let iss = SGPPropagator::from_3le(
            Some("ISS"),
            "1 25544U 98067A   26071.86901803  .00011348  00000-0  21655-3 0  9990",
            "2 25544  51.6324  56.6367 0007924 186.1410 173.9482 15.48614629556825",
            60.0,
        )
        .unwrap();
        // Hubble (NORAD 20580) - 2026 epoch
        let hubble = SGPPropagator::from_3le(
            Some("HST"),
            "1 20580U 90037B   26071.94420327  .00008743  00000-0  28877-3 0  9998",
            "2 20580  28.4723  17.7975 0001801 157.8636 202.2037 15.29540863773810",
            60.0,
        )
        .unwrap();

        // Verify default identification after construction
        assert_eq!(iss.get_name(), Some("ISS"));
        assert_eq!(iss.get_id(), Some(25544));
        assert_eq!(hubble.get_name(), Some("HST"));
        assert_eq!(hubble.get_id(), Some(20580));

        // Verify PointLocation defaults before builder methods
        let bare = PointLocation::new(0.0, 0.0, 0.0);
        assert_eq!(bare.get_name(), None);
        assert_eq!(bare.get_id(), None);
        assert!(bare.get_uuid().is_some()); // Auto-generated in constructor

        // Verify locations have identity set
        assert_eq!(locations[0].get_name(), Some("NewYork"));
        assert_eq!(locations[0].get_id(), Some(1));
        assert_eq!(locations[1].get_name(), Some("London"));
        assert_eq!(locations[1].get_id(), Some(2));

        let propagators = vec![iss, hubble];

        // -- Search window: 24 hours from a 2026 epoch --
        let search_start = Epoch::from_datetime(2026, 3, 13, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let search_end = search_start + 86400.0;

        let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();

        let windows = location_accesses(
            &locations,
            &propagators,
            search_start,
            search_end,
            &constraint,
            None,
            None,
        )
        .unwrap();

        // Should find access windows
        assert!(
            !windows.is_empty(),
            "Expected at least 1 access window, found 0"
        );

        // -- Traceability assertions --
        for window in &windows {
            // Every window must have location and satellite identification
            assert!(
                window.location_name.is_some(),
                "Window missing location_name"
            );
            assert!(window.location_id.is_some(), "Window missing location_id");
            assert!(
                window.satellite_name.is_some(),
                "Window missing satellite_name"
            );
            assert!(window.satellite_id.is_some(), "Window missing satellite_id");

            let loc_id = window.location_id.unwrap();
            let sat_id = window.satellite_id.unwrap();

            // Location ID must be one of our locations
            assert!(
                loc_id == 1 || loc_id == 2,
                "Unexpected location_id: {loc_id}"
            );
            // Satellite ID must be one of our NORAD IDs
            assert!(
                sat_id == 25544 || sat_id == 20580,
                "Unexpected satellite_id: {sat_id}"
            );

            // No cross-contamination: name must match ID
            let loc_name = window.location_name.as_deref().unwrap();
            let sat_name = window.satellite_name.as_deref().unwrap();

            match loc_id {
                1 => assert_eq!(loc_name, "NewYork"),
                2 => assert_eq!(loc_name, "London"),
                _ => unreachable!(),
            }
            match sat_id {
                25544 => assert_eq!(sat_name, "ISS"),
                20580 => assert_eq!(sat_name, "HST"),
                _ => unreachable!(),
            }

            // Auto-generated window name should contain both location and satellite names
            let window_name = window.name.as_deref().unwrap();
            assert!(
                window_name.contains(loc_name),
                "Window name '{window_name}' should contain location name '{loc_name}'"
            );
            assert!(
                window_name.contains(sat_name),
                "Window name '{window_name}' should contain satellite name '{sat_name}'"
            );
            assert!(
                window_name.contains("Access"),
                "Window name '{window_name}' should contain 'Access'"
            );
        }
    }

    #[test]
    fn test_access_default_uuid_traceability() {
        use crate::utils::Identifiable;
        use std::collections::HashSet;

        setup_global_test_eop();

        // Create locations with NO explicit identity — only auto-generated UUIDs
        let loc1 = PointLocation::new(0.0, 45.0, 0.0);
        let loc2 = PointLocation::new(-120.0, 30.0, 0.0);

        // Create propagators with NO explicit name/id/uuid
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let prop1 = create_test_propagator(epoch);
        let prop2 = {
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
        };

        // All objects should have auto-generated UUIDs
        let loc1_uuid = loc1
            .get_uuid()
            .expect("loc1 should have auto-generated UUID");
        let loc2_uuid = loc2
            .get_uuid()
            .expect("loc2 should have auto-generated UUID");
        let prop1_uuid = prop1
            .get_uuid()
            .expect("prop1 should have auto-generated UUID");
        let prop2_uuid = prop2
            .get_uuid()
            .expect("prop2 should have auto-generated UUID");

        // All UUIDs should be unique
        let all_uuids: HashSet<_> = [loc1_uuid, loc2_uuid, prop1_uuid, prop2_uuid]
            .into_iter()
            .collect();
        assert_eq!(all_uuids.len(), 4, "All 4 UUIDs should be unique");

        let loc_uuids: HashSet<_> = [loc1_uuid, loc2_uuid].into_iter().collect();
        let sat_uuids: HashSet<_> = [prop1_uuid, prop2_uuid].into_iter().collect();

        // Compute access windows
        let period = 5674.0;
        let search_end = epoch + (period * 3.0);
        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();
        let config = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            time_tolerance: 0.1,
            ..Default::default()
        };

        let windows = location_accesses(
            &vec![loc1, loc2],
            &vec![prop1, prop2],
            epoch,
            search_end,
            &constraint,
            None,
            Some(&config),
        )
        .unwrap();

        assert!(!windows.is_empty(), "Expected at least 1 access window");

        for window in &windows {
            // Every window should carry location and satellite UUIDs
            assert!(
                window.location_uuid.is_some(),
                "Window missing location_uuid"
            );
            assert!(
                window.satellite_uuid.is_some(),
                "Window missing satellite_uuid"
            );

            let loc_uuid = window.location_uuid.unwrap();
            let sat_uuid = window.satellite_uuid.unwrap();

            // UUIDs should match one of our source objects
            assert!(
                loc_uuids.contains(&loc_uuid),
                "Window location_uuid {loc_uuid} doesn't match any source location"
            );
            assert!(
                sat_uuids.contains(&sat_uuid),
                "Window satellite_uuid {sat_uuid} doesn't match any source propagator"
            );
        }

        // Group windows by satellite UUID to verify filtering works
        let unique_sat_uuids: HashSet<_> =
            windows.iter().filter_map(|w| w.satellite_uuid).collect();
        assert!(
            !unique_sat_uuids.is_empty(),
            "Should be able to group windows by satellite UUID"
        );

        // Group windows by location UUID
        let unique_loc_uuids: HashSet<_> = windows.iter().filter_map(|w| w.location_uuid).collect();
        assert!(
            !unique_loc_uuids.is_empty(),
            "Should be able to group windows by location UUID"
        );
    }
}
