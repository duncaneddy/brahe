/*!
 * Access window finding and management
 *
 * This module provides algorithms for finding and refining access windows
 * between satellites and ground locations, along with the complete
 * AccessWindow structure.
 */

use crate::access::constraints::AccessConstraint;
use crate::access::geometry::*;
use crate::access::location::AccessibleLocation;
use crate::access::properties::{AccessProperties, AccessPropertyComputer};
use crate::constants::{AngleFormat, GM_EARTH};
use crate::coordinates::position_ecef_to_geodetic;
use crate::orbits::keplerian::orbital_period_from_state;
use crate::orbits::traits::IdentifiableStateProvider;
use crate::time::Epoch;
use crate::traits::Identifiable;
use uuid::Uuid;

// ================================
// AccessWindow Structure
// ================================

/// An access window between a satellite and a location
///
/// Contains timing information, location/satellite identification,
/// and computed access properties.
#[derive(Debug, Clone)]
pub struct AccessWindow {
    /// Start of access window
    pub window_open: Epoch,

    /// End of access window
    pub window_close: Epoch,

    // ===== Location identification =====
    /// Location name (from Identifiable trait)
    pub location_name: Option<String>,

    /// Location ID (from Identifiable trait)
    pub location_id: Option<u64>,

    /// Location UUID (from Identifiable trait)
    pub location_uuid: Option<Uuid>,

    // ===== Satellite identification =====
    /// Satellite name (from Identifiable trait)
    pub satellite_name: Option<String>,

    /// Satellite ID (from Identifiable trait)
    pub satellite_id: Option<u64>,

    /// Satellite UUID (from Identifiable trait)
    pub satellite_uuid: Option<Uuid>,

    // ===== Computed properties =====
    /// Access properties (geometric + custom)
    pub properties: AccessProperties,
}

impl AccessWindow {
    /// Create a new access window
    ///
    /// # Arguments
    /// * `window_open` - Start time
    /// * `window_close` - End time
    /// * `location` - Location being accessed
    /// * `satellite` - Satellite/object with Identifiable trait
    /// * `properties` - Computed access properties
    pub fn new<L: AccessibleLocation, S: Identifiable>(
        window_open: Epoch,
        window_close: Epoch,
        location: &L,
        satellite: &S,
        properties: AccessProperties,
    ) -> Self {
        Self {
            window_open,
            window_close,
            location_name: location.get_name().map(|s| s.to_string()),
            location_id: location.get_id(),
            location_uuid: location.get_uuid(),
            satellite_name: satellite.get_name().map(|s| s.to_string()),
            satellite_id: satellite.get_id(),
            satellite_uuid: satellite.get_uuid(),
            properties,
        }
    }

    /// Get window start time
    pub fn start(&self) -> Epoch {
        self.window_open
    }

    /// Get window end time
    pub fn end(&self) -> Epoch {
        self.window_close
    }

    /// Get window midtime
    pub fn midtime(&self) -> Epoch {
        self.window_open + (self.window_close - self.window_open) / 2.0
    }

    /// Get window duration in seconds
    pub fn duration(&self) -> f64 {
        self.window_close - self.window_open
    }
}

// ================================
// Access Search Configuration
// ================================

/// Configuration for access window search algorithms.
///
/// Controls the grid search parameters and whether to enable adaptive stepping
/// based on orbital period.
///
/// # Examples
/// ```
/// use brahe::access::AccessSearchConfig;
///
/// // Default configuration: 60 second fixed grid, no adaptation
/// let config = AccessSearchConfig::default();
///
/// // Adaptive configuration: starts at 60s, then uses 3/4 orbital period
/// let config = AccessSearchConfig {
///     initial_time_step: 60.0,
///     adaptive_step: true,
///     adaptive_fraction: 0.75,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AccessSearchConfig {
    /// Initial fixed grid step (seconds)
    ///
    /// Used for the first grid search before any access is found.
    /// After the first access, if `adaptive_step` is enabled, the
    /// step size will change to `adaptive_fraction * orbital_period`.
    pub initial_time_step: f64,

    /// Enable adaptive stepping after first access found
    ///
    /// When true, after finding the first access window, the time step
    /// will be adjusted to a fraction of the orbital period. This can
    /// significantly speed up searches over long time periods with
    /// periodic revisits.
    pub adaptive_step: bool,

    /// Fraction of orbital period to use for adaptive step
    ///
    /// Typical values: 0.5-0.9. Higher values (e.g., 0.75-0.9) take larger
    /// steps and are faster but might miss short access windows. Lower
    /// values (e.g., 0.5-0.6) are more thorough but slower.
    ///
    /// Recommended: 0.75 (3T/4) provides good balance of speed and coverage.
    pub adaptive_fraction: f64,
}

impl Default for AccessSearchConfig {
    fn default() -> Self {
        Self {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
        }
    }
}

// ================================
// Window Finding Algorithms
// ================================

/// Find candidate access windows using coarse grid search with optional adaptive stepping.
///
/// Searches the time range to identify intervals where constraints are satisfied.
/// Supports adaptive stepping: after finding the first access window, the search
/// can automatically adjust the time step based on the orbital period to speed up
/// long searches with periodic revisits.
///
/// # Arguments
/// * `location` - Ground location
/// * `propagator` - Satellite propagator
/// * `search_start` - Start of search window
/// * `search_end` - End of search window
/// * `constraint` - Access constraints
/// * `config` - Search configuration (time step, adaptive parameters)
///
/// # Returns
/// List of (start, end) candidate windows (unrefined)
///
/// # Examples
/// ```ignore
/// // Fixed grid search at 60 second intervals
/// let config = AccessSearchConfig::default();
/// let windows = find_access_candidates(&location, &prop, start, end, &constraint, &config);
///
/// // Adaptive search: 60s initially, then 3T/4 after first access
/// let config = AccessSearchConfig {
///     initial_time_step: 60.0,
///     adaptive_step: true,
///     adaptive_fraction: 0.75,
/// };
/// let windows = find_access_candidates(&location, &prop, start, end, &constraint, &config);
/// ```
pub fn find_access_candidates<L: AccessibleLocation, P: IdentifiableStateProvider>(
    location: &L,
    propagator: &P,
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    config: &AccessSearchConfig,
) -> Vec<(Epoch, Epoch)> {
    let mut candidates = Vec::new();
    let location_ecef = location.center_ecef();

    // Compute orbital period once at start if adaptive stepping is enabled
    let orbital_period = if config.adaptive_step {
        let state_eci = propagator.state_eci(search_start);
        Some(orbital_period_from_state(&state_eci, GM_EARTH))
    } else {
        None
    };

    // Grid search over time range
    let mut current_time = search_start;
    let mut in_window = false;
    let mut window_start = search_start;

    while current_time <= search_end {
        // Propagate satellite to current time
        let sat_state_ecef = propagator.state_ecef(current_time);

        // Evaluate constraints
        let is_satisfied = constraint.evaluate(&current_time, &sat_state_ecef, &location_ecef);

        // Track window boundaries
        if is_satisfied && !in_window {
            // Start of new window
            window_start = current_time;
            in_window = true;
            current_time += config.initial_time_step;
        } else if !is_satisfied && in_window {
            // End of window
            candidates.push((window_start, current_time - config.initial_time_step));
            in_window = false;

            // After window closes, take ONE adaptive jump if enabled
            if let Some(period) = orbital_period {
                // Jump forward by adaptive_fraction * period, accounting for the normal step we'll take
                current_time += config.adaptive_fraction * period - config.initial_time_step;
            }
            current_time += config.initial_time_step;
        } else {
            // Normal step
            current_time += config.initial_time_step;
        }
    }

    // Handle case where window extends to end of search
    if in_window {
        candidates.push((window_start, search_end));
    }

    candidates
}

/// Direction for stepping during boundary search
pub enum StepDirection {
    Forward,
    Backward,
}

/// Refine window boundary using recursive bisection search
///
/// Steps in a given direction until the constraint condition changes, then
/// recursively refines with opposite direction and half step size. This approach
/// prevents missing boundaries that fall outside an initial search box.
///
/// # Arguments
/// * `location` - Ground location
/// * `propagator` - Satellite propagator
/// * `time` - Starting time
/// * `direction` - Direction to step (forward or backward in time)
/// * `step` - Step size in seconds
/// * `condition` - Current constraint condition at starting time (true/false)
/// * `constraint` - Access constraints
/// * `tolerance` - Minimum step size (seconds) - recursion stops when step < tolerance
/// * `min_bound` - Minimum time bound (don't search before this)
/// * `max_bound` - Maximum time bound (don't search after this)
///
/// # Returns
/// Refined boundary time where constraint transitions
#[allow(clippy::too_many_arguments)]
pub fn bisection_search<L: AccessibleLocation, P: IdentifiableStateProvider>(
    location: &L,
    propagator: &P,
    time: Epoch,
    direction: StepDirection,
    step: f64,
    condition: bool,
    constraint: &dyn AccessConstraint,
    tolerance: f64,
    min_bound: Epoch,
    max_bound: Epoch,
) -> Epoch {
    let location_ecef = location.center_ecef();

    // Step in given direction until condition changes or we hit bounds
    let mut current_time = time;
    let mut steps_taken = 0;
    const MAX_STEPS: usize = 1000; // Safety limit to prevent infinite loops

    loop {
        // Take a step
        let next_time = match direction {
            StepDirection::Forward => current_time + step,
            StepDirection::Backward => current_time - step,
        };

        // Check bounds
        if next_time < min_bound || next_time > max_bound || steps_taken >= MAX_STEPS {
            // Hit boundary without finding transition - return current best estimate
            return current_time;
        }

        current_time = next_time;
        steps_taken += 1;

        // Evaluate constraint at new time
        let sat_state_ecef = propagator.state_ecef(current_time);
        let current_condition = constraint.evaluate(&current_time, &sat_state_ecef, &location_ecef);

        // Check if condition changed
        if current_condition != condition {
            // Found transition - check if we should recurse or stop
            if step < tolerance {
                return current_time;
            } else {
                // Recurse with opposite direction and half step size
                let new_direction = match direction {
                    StepDirection::Forward => StepDirection::Backward,
                    StepDirection::Backward => StepDirection::Forward,
                };
                return bisection_search(
                    location,
                    propagator,
                    current_time,
                    new_direction,
                    step / 2.0,
                    current_condition,
                    constraint,
                    tolerance,
                    min_bound,
                    max_bound,
                );
            }
        }
    }
}

/// Compute access properties for a refined window
///
/// Computes core geometric properties and optionally calls custom
/// property computers.
///
/// # Arguments
/// * `window_open` - Window start time
/// * `window_close` - Window end time
/// * `location` - Ground location
/// * `propagator` - Satellite propagator
/// * `property_computers` - Optional custom property computers
///
/// # Returns
/// Complete AccessProperties or error
pub fn compute_window_properties<L: AccessibleLocation, P: IdentifiableStateProvider>(
    window_open: Epoch,
    window_close: Epoch,
    location: &L,
    propagator: &P,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
) -> Result<AccessProperties, crate::utils::BraheError> {
    let location_ecef = location.center_ecef();
    let location_geodetic = position_ecef_to_geodetic(location_ecef, AngleFormat::Radians);

    // Compute midtime
    let midtime = window_open + (window_close - window_open) / 2.0;

    // Compute states at key times
    let state_open_ecef = propagator.state_ecef(window_open);

    let state_close_ecef = propagator.state_ecef(window_close);

    let state_mid_ecef = propagator.state_ecef(midtime);

    let sat_pos_open = state_open_ecef.fixed_rows::<3>(0).into_owned();
    let sat_pos_close = state_close_ecef.fixed_rows::<3>(0).into_owned();
    let sat_pos_mid = state_mid_ecef.fixed_rows::<3>(0).into_owned();

    // Core geometric properties
    let azimuth_open = compute_azimuth(&sat_pos_open, &location_ecef);
    let azimuth_close = compute_azimuth(&sat_pos_close, &location_ecef);

    let elevation_open = compute_elevation(&sat_pos_open, &location_ecef);
    let elevation_close = compute_elevation(&sat_pos_close, &location_ecef);
    let elevation_mid = compute_elevation(&sat_pos_mid, &location_ecef);

    // Elevation min/max (sample a few more points for better accuracy)
    let mut elevation_samples = vec![elevation_open, elevation_close, elevation_mid];
    for i in 1..4 {
        let t = window_open + (window_close - window_open) * (i as f64 / 4.0);
        let state_ecef = propagator.state_ecef(t);

        let pos = state_ecef.fixed_rows::<3>(0).into_owned();
        elevation_samples.push(compute_elevation(&pos, &location_ecef));
    }
    let elevation_min = elevation_samples
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let elevation_max = elevation_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let off_nadir_open = compute_off_nadir(&sat_pos_open, &location_ecef);
    let off_nadir_close = compute_off_nadir(&sat_pos_close, &location_ecef);
    let off_nadir_mid = compute_off_nadir(&sat_pos_mid, &location_ecef);

    let off_nadir_min = off_nadir_open.min(off_nadir_close).min(off_nadir_mid);
    let off_nadir_max = off_nadir_open.max(off_nadir_close).max(off_nadir_mid);

    let local_time = compute_local_time(&midtime, &location_geodetic);
    let look_direction = compute_look_direction(&state_mid_ecef, &location_ecef);
    let asc_dsc = compute_asc_dsc(&state_mid_ecef);

    // Create base properties
    let mut properties = AccessProperties::new(
        azimuth_open,
        azimuth_close,
        elevation_min,
        elevation_max,
        off_nadir_min,
        off_nadir_max,
        local_time,
        look_direction,
        asc_dsc,
    );

    // Call custom property computers if provided
    if let Some(computers) = property_computers {
        // Create temporary AccessWindow for property computation
        let temp_window = AccessWindow {
            window_open,
            window_close,
            location_name: location.get_name().map(|s| s.to_string()),
            location_id: location.get_id(),
            location_uuid: location.get_uuid(),
            satellite_name: propagator.get_name().map(|s| s.to_string()),
            satellite_id: propagator.get_id(),
            satellite_uuid: propagator.get_uuid(),
            properties: properties.clone(),
        };

        // Use propagator's StateProvider trait directly
        for computer in computers {
            let additional =
                computer.compute(&temp_window, propagator, &location_ecef, &location_geodetic)?;
            for (key, value) in additional {
                properties.add_property(key, value);
            }
        }
    }

    Ok(properties)
}

/// Wrapper for compute_window_properties that propagates errors
fn compute_window_properties_internal<L: AccessibleLocation, P: IdentifiableStateProvider>(
    window_open: Epoch,
    window_close: Epoch,
    location: &L,
    propagator: &P,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
) -> Result<AccessProperties, crate::utils::BraheError> {
    compute_window_properties(
        window_open,
        window_close,
        location,
        propagator,
        property_computers,
    )
}

/// Find and refine access windows for a location
///
/// Complete workflow: coarse grid search, boundary refinement,
/// and property computation.
///
/// # Arguments
/// * `location` - Ground location
/// * `propagator` - Satellite propagator
/// * `search_start` - Start of search window
/// * `search_end` - End of search window
/// * `constraint` - Access constraints
/// * `property_computers` - Optional custom property computers
/// * `time_step` - Search grid step (default: 60 seconds)
/// * `time_tolerance` - Boundary refinement tolerance (default: 0.01 seconds)
///
/// # Returns
/// List of complete AccessWindow objects
#[allow(clippy::too_many_arguments)]
pub fn find_access_windows<L: AccessibleLocation, P: IdentifiableStateProvider>(
    location: &L,
    propagator: &P,
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    time_step: Option<f64>,
    time_tolerance: Option<f64>,
) -> Vec<AccessWindow> {
    let time_tolerance = time_tolerance.unwrap_or(0.01);

    // Create search config from time_step parameter
    let config = AccessSearchConfig {
        initial_time_step: time_step.unwrap_or(60.0),
        adaptive_step: false, // Default: no adaptation
        adaptive_fraction: 0.75,
    };

    // Find candidate windows
    let candidates = find_access_candidates(
        location,
        propagator,
        search_start,
        search_end,
        constraint,
        &config,
    );

    // Refine each candidate
    let mut windows = Vec::new();

    for (coarse_start, coarse_end) in candidates {
        // Get location for constraint evaluation
        let location_ecef = location.center_ecef();

        // Refine opening boundary
        // Start at coarse_start (condition=true) and search backward
        let refined_start = if coarse_start > search_start {
            // Evaluate condition at coarse_start to confirm
            let sat_state = propagator.state_ecef(coarse_start);
            let start_condition = constraint.evaluate(&coarse_start, &sat_state, &location_ecef);

            bisection_search(
                location,
                propagator,
                coarse_start,
                StepDirection::Backward,
                config.initial_time_step,
                start_condition,
                constraint,
                time_tolerance,
                search_start, // min_bound
                coarse_start, // max_bound
            )
        } else {
            coarse_start
        };

        // Refine closing boundary
        // Start at coarse_end (condition=false) and search backward
        let refined_end = if coarse_end < search_end {
            // Evaluate condition at coarse_end to confirm
            let sat_state = propagator.state_ecef(coarse_end);
            let end_condition = constraint.evaluate(&coarse_end, &sat_state, &location_ecef);

            bisection_search(
                location,
                propagator,
                coarse_end,
                StepDirection::Backward,
                config.initial_time_step,
                end_condition,
                constraint,
                time_tolerance,
                coarse_end, // min_bound
                search_end, // max_bound
            )
        } else {
            coarse_end
        };

        // Compute properties (propagate errors)
        let properties = match compute_window_properties_internal(
            refined_start,
            refined_end,
            location,
            propagator,
            property_computers,
        ) {
            Ok(props) => props,
            Err(_) => continue, // Skip windows that fail property computation
        };

        // Create complete window
        windows.push(AccessWindow::new(
            refined_start,
            refined_end,
            location,
            propagator,
            properties,
        ));
    }

    windows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::constraints::ElevationConstraint;
    use crate::access::location::PointLocation;
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::orbits::keplerian_propagator::KeplerianPropagator;
    use crate::orbits::traits::StateProvider;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;
    use nalgebra::Vector6;

    #[test]
    fn test_find_access_candidates() {
        setup_global_test_eop();

        // Create a location at 45° latitude (more likely to have access with 45° inclination orbit)
        let location = PointLocation::new(45.0, 0.0, 0.0);

        // Create a LEO satellite (500 km altitude, 45° inclination)
        let oe = Vector6::new(
            R_EARTH + 500e3,       // a
            0.0,                   // e
            45.0_f64.to_radians(), // i (radians)
            0.0,                   // RAAN
            0.0,                   // argp
            0.0,                   // M
        );
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = KeplerianPropagator::new(
            epoch,
            oe,
            crate::trajectories::traits::OrbitFrame::ECI,
            crate::trajectories::traits::OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );

        // Search for two orbital periods to increase chances of finding access
        let period = 5674.0; // ~94 minutes for 500 km LEO
        let search_end = epoch + (period * 2.0);

        // Low elevation constraint to ensure we find access
        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        // Find candidates using default config (60s, no adaptation)
        let config = AccessSearchConfig::default();
        let candidates = find_access_candidates(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            &config,
        );

        // Should find at least one access window
        assert!(
            !candidates.is_empty(),
            "Expected to find at least 1 access window, found {}",
            candidates.len()
        );

        // Each window should have start <t end
        for (start, end) in candidates {
            assert!(start < end);
        }
    }

    #[test]
    fn test_bisection_search() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);

        let oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0_f64.to_radians(), 0.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = KeplerianPropagator::new(
            epoch,
            oe,
            crate::trajectories::traits::OrbitFrame::ECI,
            crate::trajectories::traits::OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );

        let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();

        // Test recursive bisection search
        // Start from a time where condition is known, search backward to find transition
        let t_start = epoch + 300.0; // Start 5 minutes in

        // Evaluate initial condition
        let location_ecef = location.center_ecef();
        let sat_state = propagator.state_ecef(t_start);
        let initial_condition = constraint.evaluate(&t_start, &sat_state, &location_ecef);

        // Search backward from t_start with initial step of 60 seconds
        // Use bounds to constrain the search
        let refined = bisection_search(
            &location,
            &propagator,
            t_start,
            StepDirection::Backward,
            60.0,
            initial_condition,
            &constraint,
            0.01,
            epoch,   // min_bound - don't search before epoch
            t_start, // max_bound
        );

        // Refined time should be within bounds
        assert!(refined <= t_start);
        assert!(refined >= epoch);
    }

    #[test]
    fn test_compute_window_properties() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 45.0, 0.0);

        let oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0_f64.to_radians(), 0.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = KeplerianPropagator::new(
            epoch,
            oe,
            crate::trajectories::traits::OrbitFrame::ECI,
            crate::trajectories::traits::OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );

        let window_open = epoch;
        let window_close = epoch + 300.0;

        let properties =
            compute_window_properties(window_open, window_close, &location, &propagator, None)
                .unwrap();

        // Verify properties are in valid ranges
        assert!((0.0..=360.0).contains(&properties.azimuth_open));
        assert!((0.0..=360.0).contains(&properties.azimuth_close));
        assert!((-90.0..=90.0).contains(&properties.elevation_min));
        assert!((-90.0..=90.0).contains(&properties.elevation_max));
        assert!((0.0..=180.0).contains(&properties.off_nadir_min));
        assert!((0.0..=180.0).contains(&properties.off_nadir_max));
        assert!((0.0..=86400.0).contains(&properties.local_time));
    }

    #[test]
    fn test_find_access_windows() {
        setup_global_test_eop();

        let location = PointLocation::new(45.0, 0.0, 0.0);

        let oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0_f64.to_radians(), 0.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let propagator = KeplerianPropagator::new(
            epoch,
            oe,
            crate::trajectories::traits::OrbitFrame::ECI,
            crate::trajectories::traits::OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );

        let period = 5674.0;
        let search_end = epoch + (period * 2.0);

        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        let windows = find_access_windows(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(60.0),
            Some(0.1),
        );

        // Should find at least one window
        assert!(
            !windows.is_empty(),
            "Expected to find at least 1 access window, found {}",
            windows.len()
        );

        // Verify window structure
        for window in windows {
            assert!(window.window_open < window.window_close);
            assert!(window.duration() > 0.0);

            // Properties should be valid
            assert!((0.0..=360.0).contains(&window.properties.azimuth_open));
            assert!((-90.0..=90.0).contains(&window.properties.elevation_max));
        }
    }
}
