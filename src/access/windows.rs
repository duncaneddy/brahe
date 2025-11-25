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
use crate::time::Epoch;
use crate::traits::Identifiable;
use crate::utils::BraheError;
use crate::utils::state_providers::DIdentifiableStateProvider;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

// Static counter for auto-generating AccessWindow names
static ACCESS_COUNTER: AtomicUsize = AtomicUsize::new(1);

// ================================
// AccessWindow Structure
// ================================

/// An access window between a satellite and a location
///
/// Contains timing information, location/satellite identification,
/// and computed access properties. Implements `Identifiable` trait
/// with auto-generated names.
///
/// # Auto-generated Names
/// When created via `new()`, names are automatically generated as:
/// - If both location and satellite have names: `"{location}-{satellite}-Access-{counter:03}"`
/// - Otherwise: `"Access-{counter:03}"`
///
/// The counter increments sequentially for each access window created.
///
/// # Examples
/// ```
/// // With named location and satellite: "Svalbard-Sentinel1-Access-001"
/// // Without names: "Access-042"
/// ```
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

    // ===== AccessWindow identification (Identifiable trait) =====
    /// Access window name (auto-generated or user-set)
    pub name: Option<String>,

    /// Access window numeric ID
    pub id: Option<u64>,

    /// Access window UUID
    pub uuid: Option<Uuid>,

    // ===== Computed properties =====
    /// Access properties (geometric + custom)
    pub properties: AccessProperties,
}

impl AccessWindow {
    /// Create a new access window with auto-generated name
    ///
    /// # Arguments
    /// * `window_open` - Start time
    /// * `window_close` - End time
    /// * `location` - Location being accessed
    /// * `satellite` - Satellite/object with Identifiable trait
    /// * `properties` - Computed access properties
    ///
    /// # Auto-generated Name
    /// - If both location and satellite have names: `"{location}-{satellite}-Access-{counter:03}"`
    /// - Otherwise: `"Access-{counter:03}"`
    ///
    /// Counter increments atomically for each window created.
    pub fn new<L: AccessibleLocation, S: Identifiable>(
        window_open: Epoch,
        window_close: Epoch,
        location: &L,
        satellite: &S,
        properties: AccessProperties,
    ) -> Self {
        // Get counter value and increment atomically
        let counter = ACCESS_COUNTER.fetch_add(1, Ordering::SeqCst);

        // Extract identification from location and satellite
        let loc_name = location.get_name().map(|s| s.to_string());
        let sat_name = satellite.get_name().map(|s| s.to_string());

        // Generate access window name
        let name = match (&loc_name, &sat_name) {
            (Some(loc), Some(sat)) => Some(format!("{}-{}-Access-{:03}", loc, sat, counter)),
            _ => Some(format!("Access-{:03}", counter)),
        };

        Self {
            window_open,
            window_close,
            location_name: loc_name,
            location_id: location.get_id(),
            location_uuid: location.get_uuid(),
            satellite_name: sat_name,
            satellite_id: satellite.get_id(),
            satellite_uuid: satellite.get_uuid(),
            name,
            id: None,
            uuid: None,
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

    /// Get window start time (alias for `start`)
    pub fn t_start(&self) -> Epoch {
        self.window_open
    }

    /// Get window end time (alias for `end`)
    pub fn t_end(&self) -> Epoch {
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
// Identifiable Trait Implementation
// ================================

impl Identifiable for AccessWindow {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_uuid(mut self, uuid: Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(Uuid::new_v4());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
    }

    fn with_identity(mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) -> Self {
        if let Some(n) = name {
            self.name = Some(n.to_string());
        }
        self.uuid = uuid;
        self.id = id;
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) {
        if let Some(n) = name {
            self.name = Some(n.to_string());
        } else {
            self.name = None;
        }
        self.uuid = uuid;
        self.id = id;
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(Uuid::new_v4());
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<Uuid> {
        self.uuid
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
/// // Default configuration: 60 second fixed grid, no adaptation, parallel enabled
/// let config = AccessSearchConfig::default();
///
/// // Adaptive configuration: starts at 60s, then uses 3/4 orbital period
/// let config = AccessSearchConfig {
///     initial_time_step: 60.0,
///     adaptive_step: true,
///     adaptive_fraction: 0.75,
///     parallel: true,
///     num_threads: None,
/// };
///
/// // Sequential (non-parallel) with custom threads
/// let config = AccessSearchConfig {
///     initial_time_step: 60.0,
///     adaptive_step: false,
///     adaptive_fraction: 0.75,
///     parallel: false,
///     num_threads: Some(4),
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

    /// Enable parallel computation
    ///
    /// When true (default), access computation for multiple locations
    /// and/or propagators will be parallelized using rayon. When false,
    /// computation is sequential.
    ///
    /// Parallel computation is typically faster for multiple locations/satellites,
    /// but sequential may be preferred for debugging or when managing threads
    /// externally.
    pub parallel: bool,

    /// Number of threads for parallel computation
    ///
    /// When `Some(n)`, uses exactly n threads for this computation.
    /// When `None` (default), uses the global thread pool setting
    /// (see `set_max_threads()`, defaults to 90% of available cores).
    ///
    /// Only applies when `parallel = true`.
    pub num_threads: Option<usize>,
}

impl Default for AccessSearchConfig {
    fn default() -> Self {
        Self {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.75,
            parallel: true,
            num_threads: None,
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
///     parallel: true,
///     num_threads: None,
/// };
/// let windows = find_access_candidates(&location, &prop, start, end, &constraint, &config);
/// ```
pub fn find_access_candidates<L: AccessibleLocation, P: DIdentifiableStateProvider>(
    location: &L,
    propagator: &P,
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    config: &AccessSearchConfig,
) -> Result<Vec<(Epoch, Epoch)>, BraheError> {
    let mut candidates = Vec::new();
    let location_ecef = location.center_ecef();

    // Compute orbital period once at start if adaptive stepping is enabled
    let orbital_period = if config.adaptive_step {
        let state_eci = propagator.state_eci(search_start)?;
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
        let sat_state_ecef = propagator.state_ecef(current_time)?;

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

    Ok(candidates)
}

/// Direction for stepping during boundary search
pub enum StepDirection {
    /// Step forward in time when searching for window boundary
    Forward,
    /// Step backward in time when searching for window boundary
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
pub fn bisection_search<L: AccessibleLocation, P: DIdentifiableStateProvider>(
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
) -> Result<Epoch, BraheError> {
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
            return Ok(current_time);
        }

        current_time = next_time;
        steps_taken += 1;

        // Evaluate constraint at new time
        let sat_state_ecef = propagator.state_ecef(current_time)?;
        let current_condition = constraint.evaluate(&current_time, &sat_state_ecef, &location_ecef);

        // Check if condition changed
        if current_condition != condition {
            // Found transition - check if we should recurse or stop
            if step < tolerance {
                return Ok(current_time);
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
pub fn compute_window_properties<L: AccessibleLocation, P: DIdentifiableStateProvider>(
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
    let state_open_ecef = propagator.state_ecef(window_open)?;

    let state_close_ecef = propagator.state_ecef(window_close)?;

    let state_mid_ecef = propagator.state_ecef(midtime)?;

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
        let state_ecef = propagator.state_ecef(t)?;

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

    // Extract location coordinates for storage (convert angles to degrees for consistency)
    let center_lon = location_geodetic[0].to_degrees();
    let center_lat = location_geodetic[1].to_degrees();
    let center_alt = location_geodetic[2];
    let center_ecef = [location_ecef[0], location_ecef[1], location_ecef[2]];

    // Create base properties
    let mut properties = AccessProperties::new(
        azimuth_open,
        azimuth_close,
        elevation_min,
        elevation_max,
        elevation_open,
        elevation_close,
        off_nadir_min,
        off_nadir_max,
        local_time,
        look_direction,
        asc_dsc,
        center_lon,
        center_lat,
        center_alt,
        center_ecef,
    );

    // Call custom property computers if provided
    if let Some(computers) = property_computers {
        // Create temporary AccessWindow for property computation
        // Note: This is temporary and won't increment the counter
        let temp_window = AccessWindow {
            window_open,
            window_close,
            location_name: location.get_name().map(|s| s.to_string()),
            location_id: location.get_id(),
            location_uuid: location.get_uuid(),
            satellite_name: propagator.get_name().map(|s| s.to_string()),
            satellite_id: propagator.get_id(),
            satellite_uuid: propagator.get_uuid(),
            name: None, // Temporary window, no name needed
            id: None,
            uuid: None,
            properties: properties.clone(),
        };

        // Use propagator's StateProvider trait with sampling
        for computer in computers {
            // Get sampling configuration
            let sampling_config = computer.sampling_config();

            // Generate sample epochs based on configuration
            let sample_epochs = sampling_config.generate_sample_epochs(window_open, window_close);

            // Get states at sample epochs
            let sample_states: Vec<nalgebra::SVector<f64, 6>> = sample_epochs
                .iter()
                .map(|&epoch| propagator.state_ecef(epoch))
                .collect::<Result<Vec<_>, _>>()?;

            // Convert epochs to MJD for property computer interface
            let sample_epochs_mjd: Vec<f64> = sample_epochs.iter().map(|e| e.mjd()).collect();

            // Compute properties with sampled states
            let additional = computer.compute(
                &temp_window,
                &sample_epochs_mjd,
                &sample_states,
                &location_ecef,
                &location_geodetic,
            )?;

            for (key, value) in additional {
                properties.add_property(key, value);
            }
        }
    }

    Ok(properties)
}

/// Wrapper for compute_window_properties that propagates errors
fn compute_window_properties_internal<L: AccessibleLocation, P: DIdentifiableStateProvider>(
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
/// * `time_tolerance` - Boundary refinement tolerance (default: 0.001 seconds, ~0.01° elevation precision)
///
/// # Returns
/// Result containing list of complete AccessWindow objects, or error if property computation fails
#[allow(clippy::too_many_arguments)]
pub fn find_access_windows<L: AccessibleLocation, P: DIdentifiableStateProvider>(
    location: &L,
    propagator: &P,
    search_start: Epoch,
    search_end: Epoch,
    constraint: &dyn AccessConstraint,
    property_computers: Option<&[&dyn AccessPropertyComputer]>,
    time_step: Option<f64>,
    time_tolerance: Option<f64>,
) -> Result<Vec<AccessWindow>, BraheError> {
    let time_tolerance = time_tolerance.unwrap_or(0.001);

    // Create search config from time_step parameter
    let config = AccessSearchConfig {
        initial_time_step: time_step.unwrap_or(60.0),
        adaptive_step: false, // Default: no adaptation
        adaptive_fraction: 0.75,
        parallel: true,    // Default: parallel enabled (not used in this function)
        num_threads: None, // Default: use global setting (not used in this function)
    };

    // Find candidate windows
    let candidates = find_access_candidates(
        location,
        propagator,
        search_start,
        search_end,
        constraint,
        &config,
    )?;

    // Refine each candidate
    let mut windows = Vec::new();

    for (coarse_start, coarse_end) in candidates {
        // Get location for constraint evaluation
        let location_ecef = location.center_ecef();

        // Refine opening boundary
        // Start at coarse_start (condition=true) and search backward
        let refined_start = if coarse_start > search_start {
            // Evaluate condition at coarse_start to confirm
            let sat_state = propagator.state_ecef(coarse_start)?;
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
            )?
        } else {
            coarse_start
        };

        // Refine closing boundary
        // Start at coarse_end (last point where condition=true) and search forward
        let refined_end = if coarse_end < search_end {
            // Evaluate condition at coarse_end to confirm
            let sat_state = propagator.state_ecef(coarse_end)?;
            let end_condition = constraint.evaluate(&coarse_end, &sat_state, &location_ecef);

            bisection_search(
                location,
                propagator,
                coarse_end,
                StepDirection::Forward, // Search forward to find where constraint becomes false
                config.initial_time_step,
                end_condition,
                constraint,
                time_tolerance,
                coarse_end, // min_bound (starting point)
                search_end, // max_bound (allow stepping forward to search end)
            )?
        } else {
            coarse_end
        };

        // Compute properties (propagate errors instead of skipping)
        let properties = compute_window_properties_internal(
            refined_start,
            refined_end,
            location,
            propagator,
            property_computers,
        )?;

        // Create complete window
        windows.push(AccessWindow::new(
            refined_start,
            refined_end,
            location,
            propagator,
            properties,
        ));
    }

    Ok(windows)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::access::constraints::ElevationConstraint;
    use crate::access::location::PointLocation;
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::propagators::keplerian_propagator::KeplerianPropagator;
    use crate::time::TimeSystem;
    use crate::utils::state_providers::DOrbitStateProvider;
    use crate::utils::testing::setup_global_test_eop;
    use nalgebra::Vector6;
    use serial_test::serial;

    #[test]
    #[serial]
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
        )
        .unwrap();

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
        let sat_state = propagator.state_ecef(t_start).unwrap();
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
        )
        .unwrap();

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
    #[serial]
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
        )
        .unwrap();

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

    #[test]
    fn test_access_window_implements_identifiable() {
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

        let window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Test that Identifiable trait methods work
        assert!(window.get_name().is_some());
        assert!(window.get_id().is_none());
        assert!(window.get_uuid().is_none());

        // Test builder methods
        let window_with_id = window.clone().with_id(123);
        assert_eq!(window_with_id.get_id(), Some(123));

        let window_with_uuid = window.clone().with_new_uuid();
        assert!(window_with_uuid.get_uuid().is_some());

        let window_with_custom_name = window.clone().with_name("CustomAccess");
        assert_eq!(window_with_custom_name.get_name(), Some("CustomAccess"));
    }

    #[test]
    fn test_access_window_auto_naming_with_location_and_satellite() {
        setup_global_test_eop();

        // Create location and satellite with names
        let location = PointLocation::new(15.4, 78.2, 0.0).with_name("Svalbard");
        let oe = Vector6::new(R_EARTH + 500e3, 0.0, 45.0_f64.to_radians(), 0.0, 0.0, 0.0);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut propagator = KeplerianPropagator::new(
            epoch,
            oe,
            crate::trajectories::traits::OrbitFrame::ECI,
            crate::trajectories::traits::OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        );
        propagator.set_name(Some("Sentinel1"));

        let window_open = epoch;
        let window_close = epoch + 300.0;
        let properties =
            compute_window_properties(window_open, window_close, &location, &propagator, None)
                .unwrap();

        let window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Should have format: "{location}-{satellite}-Access-{counter:03}"
        let name = window.get_name().unwrap();
        assert!(name.starts_with("Svalbard-Sentinel1-Access-"));
        assert!(name.contains("-Access-"));
    }

    #[test]
    fn test_access_window_auto_naming_without_names() {
        setup_global_test_eop();

        // Create location and satellite WITHOUT names
        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        let window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Should have format: "Access-{counter:03}"
        let name = window.get_name().unwrap();
        assert!(name.starts_with("Access-"));
        assert!(!name.contains("-Access-")); // Should NOT have the double dash pattern
    }

    #[test]
    fn test_access_window_counter_increments() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        // Create multiple windows and verify counter increments
        let window1 = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties.clone(),
        );
        let window2 = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties.clone(),
        );
        let window3 = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        let name1 = window1.get_name().unwrap();
        let name2 = window2.get_name().unwrap();
        let name3 = window3.get_name().unwrap();

        // All should be different due to counter
        assert_ne!(name1, name2);
        assert_ne!(name2, name3);
        assert_ne!(name1, name3);

        // All should have the "Access-" prefix
        assert!(name1.starts_with("Access-"));
        assert!(name2.starts_with("Access-"));
        assert!(name3.starts_with("Access-"));
    }

    /// Test that validates elevation at access window boundaries matches the constraint threshold.
    /// This test documents Issue: Elevation values at window open/close should match the constraint
    /// threshold within a tight tolerance (0.001°).
    #[test]
    #[serial]
    fn test_elevation_boundary_precision() {
        setup_global_test_eop();

        // Create NYC ground station (lon, lat, alt)
        let location = PointLocation::new(-74.0060, 40.7128, 0.0);

        // Create LEO satellite (500 km altitude, 97.8° inclination - typical sun-synchronous)
        let oe = Vector6::new(
            R_EARTH + 500e3,       // a (meters)
            0.001,                 // e (nearly circular)
            97.8_f64.to_radians(), // i (radians) - sun-synchronous
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

        // Search for 24 hours with 5.0° elevation constraint
        let search_end = epoch + 86400.0; // 24 hours in seconds
        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        // Find access windows with default settings (currently 0.01s time tolerance)
        let windows = find_access_windows(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            None,
            Some(60.0), // Grid step
            None,       // Use default time tolerance
        )
        .unwrap();

        // Should find at least one window
        assert!(
            !windows.is_empty(),
            "Expected to find at least 1 access window for LEO satellite over NYC in 24 hours"
        );

        // Validate each window's boundary elevations match the 5.0° constraint
        let tolerance = 0.001; // degrees
        let constraint_elevation = 5.0; // degrees

        for (i, window) in windows.iter().enumerate() {
            let elevation_open = window.properties.elevation_open;
            let elevation_close = window.properties.elevation_close;

            let error_open = (elevation_open - constraint_elevation).abs();
            let error_close = (elevation_close - constraint_elevation).abs();

            assert!(
                error_open <= tolerance,
                "Window {}: elevation_open = {:.6}° differs from constraint ({:.1}°) by {:.6}° (tolerance: {:.3}°)",
                i,
                elevation_open,
                constraint_elevation,
                error_open,
                tolerance
            );

            assert!(
                error_close <= tolerance,
                "Window {}: elevation_close = {:.6}° differs from constraint ({:.1}°) by {:.6}° (tolerance: {:.3}°)",
                i,
                elevation_close,
                constraint_elevation,
                error_close,
                tolerance
            );
        }

        // Print summary for debugging
        println!(
            "\nValidated {} access windows - all boundary elevations within {:.3}° of {:.1}° constraint",
            windows.len(),
            tolerance,
            constraint_elevation
        );
    }

    #[test]
    fn test_access_window_with_uuid() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        let window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Initially should have no UUID
        assert!(window.get_uuid().is_none());

        // Create a specific UUID and set it
        let test_uuid = Uuid::new_v4();
        let window_with_uuid = window.with_uuid(test_uuid);

        // Should have the exact UUID we set
        assert_eq!(window_with_uuid.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_access_window_with_identity() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        let window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Set complete identity (name, UUID, ID)
        let test_uuid = Uuid::new_v4();
        let window_with_identity =
            window
                .clone()
                .with_identity(Some("TestAccess"), Some(test_uuid), Some(42));

        // Verify all identity fields are set correctly
        assert_eq!(window_with_identity.get_name(), Some("TestAccess"));
        assert_eq!(window_with_identity.get_uuid(), Some(test_uuid));
        assert_eq!(window_with_identity.get_id(), Some(42));

        // Test partial identity (only name and ID, no UUID)
        let window_partial = window.with_identity(Some("PartialAccess"), None, Some(99));
        assert_eq!(window_partial.get_name(), Some("PartialAccess"));
        assert_eq!(window_partial.get_uuid(), None);
        assert_eq!(window_partial.get_id(), Some(99));
    }

    #[test]
    fn test_access_window_set_identity() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        let mut window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Initially has auto-generated name, no UUID or ID
        assert!(window.get_name().is_some());
        assert!(window.get_uuid().is_none());
        assert!(window.get_id().is_none());

        // Set complete identity using mutable method
        let test_uuid = Uuid::new_v4();
        window.set_identity(Some("MutableAccess"), Some(test_uuid), Some(123));

        // Verify all fields updated
        assert_eq!(window.get_name(), Some("MutableAccess"));
        assert_eq!(window.get_uuid(), Some(test_uuid));
        assert_eq!(window.get_id(), Some(123));

        // Test clearing name by passing None
        window.set_identity(None, Some(test_uuid), Some(456));
        assert_eq!(window.get_name(), None);
        assert_eq!(window.get_uuid(), Some(test_uuid));
        assert_eq!(window.get_id(), Some(456));
    }

    #[test]
    fn test_access_window_generate_uuid() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        let mut window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Initially should have no UUID
        assert!(window.get_uuid().is_none());

        // Generate a UUID
        window.generate_uuid();

        // Should now have a UUID
        assert!(window.get_uuid().is_some());

        // Verify it's a valid UUID (has correct version and variant)
        let uuid = window.get_uuid().unwrap();
        assert_eq!(uuid.get_version_num(), 4); // UUIDv4

        // Generate another UUID and verify it's different
        let first_uuid = uuid;
        window.generate_uuid();
        let second_uuid = window.get_uuid().unwrap();
        assert_ne!(first_uuid, second_uuid);
    }

    #[test]
    fn test_access_window_time_methods() {
        setup_global_test_eop();

        let location = PointLocation::new(0.0, 0.0, 0.0);
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

        let window = AccessWindow::new(
            window_open,
            window_close,
            &location,
            &propagator,
            properties,
        );

        // Test start() and t_start() are equivalent
        assert_eq!(window.start(), window_open);
        assert_eq!(window.t_start(), window_open);
        assert_eq!(window.start(), window.t_start());

        // Test end() and t_end() are equivalent
        assert_eq!(window.end(), window_close);
        assert_eq!(window.t_end(), window_close);
        assert_eq!(window.end(), window.t_end());

        // Test midtime() calculation
        let expected_midtime = window_open + 150.0; // 300s / 2 = 150s
        assert_eq!(window.midtime(), expected_midtime);
    }

    #[test]
    fn test_compute_window_properties_with_property_computer() {
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

        // Create property computers
        use crate::access::properties::{DopplerComputer, RangeComputer, SamplingConfig};

        let doppler_computer = DopplerComputer::new(
            Some(2.2e9), // Uplink frequency: 2.2 GHz
            Some(8.4e9), // Downlink frequency: 8.4 GHz
            SamplingConfig::Midpoint,
        );

        let range_computer = RangeComputer::new(SamplingConfig::FixedCount(5));

        // Create slice of property computer references
        let computers: [&dyn AccessPropertyComputer; 2] = [&doppler_computer, &range_computer];

        // Compute properties with custom property computers
        let properties = compute_window_properties(
            window_open,
            window_close,
            &location,
            &propagator,
            Some(&computers),
        )
        .unwrap();

        // Verify standard geometric properties are computed
        assert!((0.0..=360.0).contains(&properties.azimuth_open));
        assert!((0.0..=360.0).contains(&properties.azimuth_close));
        assert!((-90.0..=90.0).contains(&properties.elevation_min));
        assert!((-90.0..=90.0).contains(&properties.elevation_max));

        // Verify custom properties were computed
        assert!(
            properties.get_property("doppler_uplink").is_some(),
            "Uplink Doppler property should be computed"
        );
        assert!(
            properties.get_property("doppler_downlink").is_some(),
            "Downlink Doppler property should be computed"
        );
        assert!(
            properties.get_property("range").is_some(),
            "Range property should be computed"
        );

        // Verify Doppler values are reasonable (should be scalar at midpoint)
        if let Some(doppler_uplink) = properties.get_property("doppler_uplink") {
            match doppler_uplink {
                crate::access::properties::PropertyValue::Scalar(value) => {
                    // Doppler shift for LEO should be in the range of ±10 kHz typically
                    assert!(
                        value.abs() < 20000.0,
                        "Uplink Doppler should be reasonable: {}",
                        value
                    );
                }
                _ => panic!("Expected Scalar property for doppler_uplink"),
            }
        }

        // Verify range values are time series (FixedCount(5))
        if let Some(range) = properties.get_property("range") {
            match range {
                crate::access::properties::PropertyValue::TimeSeries { times, values } => {
                    assert_eq!(times.len(), 5, "Should have 5 time samples");
                    assert_eq!(values.len(), 5, "Should have 5 range values");
                    // All ranges should be positive and reasonable for LEO
                    // (can be much larger than altitude due to slant angle)
                    for value in values {
                        assert!(*value > 0.0, "Range should be positive");
                        assert!(
                            *value < 10000e3,
                            "Range should be less than 10000 km for visible LEO satellite"
                        );
                    }
                }
                _ => panic!("Expected TimeSeries property for range"),
            }
        }
    }

    #[test]
    #[serial]
    fn test_find_access_candidates_with_adaptive_stepping() {
        setup_global_test_eop();

        // Create a location at 45° latitude
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

        // Search for several orbital periods
        let period = 5674.0; // ~94 minutes for 500 km LEO
        let search_end = epoch + (period * 5.0); // Search for 5 orbits

        // Low elevation constraint to ensure we find access
        let constraint = ElevationConstraint::new(Some(5.0), None).unwrap();

        // Find candidates WITHOUT adaptive stepping
        let config_no_adaptive = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: false,
            adaptive_fraction: 0.5,
            parallel: false,
            num_threads: Some(1),
        };
        let candidates_no_adaptive = find_access_candidates(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            &config_no_adaptive,
        )
        .unwrap();

        // Find candidates WITH adaptive stepping
        let config_adaptive = AccessSearchConfig {
            initial_time_step: 60.0,
            adaptive_step: true,
            adaptive_fraction: 0.5, // Jump forward by 50% of orbital period after window closes
            parallel: false,
            num_threads: Some(1),
        };
        let candidates_adaptive = find_access_candidates(
            &location,
            &propagator,
            epoch,
            search_end,
            &constraint,
            &config_adaptive,
        )
        .unwrap();

        // Both methods should find windows
        assert!(
            !candidates_no_adaptive.is_empty(),
            "Expected to find access windows without adaptive stepping"
        );
        assert!(
            !candidates_adaptive.is_empty(),
            "Expected to find access windows with adaptive stepping"
        );

        // Adaptive stepping should find similar number of windows
        // (might differ slightly due to grid alignment)
        let count_diff =
            (candidates_no_adaptive.len() as i32 - candidates_adaptive.len() as i32).abs();
        assert!(
            count_diff <= 2,
            "Window counts should be similar: no_adaptive={}, adaptive={}, diff={}",
            candidates_no_adaptive.len(),
            candidates_adaptive.len(),
            count_diff
        );

        // Verify windows have reasonable overlap
        // For each window from adaptive search, there should be a corresponding window
        // in the non-adaptive search that overlaps
        for (adaptive_start, adaptive_end) in &candidates_adaptive {
            let has_overlap = candidates_no_adaptive.iter().any(|(start, end)| {
                // Check if windows overlap
                !(adaptive_end < start || adaptive_start > end)
            });
            assert!(
                has_overlap,
                "Adaptive window ({} to {}) should overlap with at least one non-adaptive window",
                adaptive_start, adaptive_end
            );
        }

        println!(
            "Adaptive stepping test: found {} windows without adaptive, {} windows with adaptive",
            candidates_no_adaptive.len(),
            candidates_adaptive.len()
        );
    }
}
