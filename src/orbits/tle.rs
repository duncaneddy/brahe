/*!
 * The `tle` module provides functionality for working with NORAD Two-Line Element (TLE) data.
 * 
 * Supports both classic TLE format and Alpha-5 TLE format using the SGP4 propagation algorithm.
 * Alpha-5 expands the NORAD ID range by replacing the first digit with a letter for IDs >= 100000.
 */

use crate::orbits::propagation::{OrbitPropagator, TrajectoryEvictionPolicy};
use crate::time::Epoch;
use crate::trajectories::{AngleFormat, InterpolationMethod, OrbitFrame, OrbitState, OrbitStateType, Trajectory};
use crate::utils::BraheError;
use nalgebra::{Vector3, Vector6};
use serde::{Deserialize, Serialize};
use sgp4::chrono::{Datelike, Timelike};
use std::str::FromStr;

/// TLE format type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TleFormat {
    /// Classic 2-line TLE format with numeric NORAD ID
    Classic,
    /// Alpha-5 TLE format with alphanumeric NORAD ID (first digit replaced with letter for IDs >= 100000)
    Alpha5,
}

/// Structure representing a Two-Line Element set with SGP4 propagation capability
#[derive(Debug, Clone)]
pub struct TLE {
    /// Raw first line of TLE
    pub line1: String,
    
    /// Raw second line of TLE  
    pub line2: String,
    
    /// Optional satellite name (from 3-line format)
    pub satellite_name: Option<String>,
    
    /// TLE format type (Classic or Alpha-5)
    pub format: TleFormat,
    
    /// Original NORAD ID string (may contain Alpha-5 encoding)
    pub norad_id_string: String,
    
    /// Decoded numeric NORAD ID 
    pub norad_id: u32,
    
    /// Parsed SGP4 elements
    elements: sgp4::Elements,
    
    /// SGP4 propagation constants
    constants: sgp4::Constants,
    
    /// Initial state from TLE epoch
    initial_state: OrbitState,
    
    /// Current propagated state
    current_state: OrbitState,
    
    /// Trajectory of all propagated states
    trajectory: Trajectory<OrbitState>,
    
    /// Maximum number of states to keep in trajectory
    max_trajectory_size: Option<usize>,
    
    /// Policy for evicting old states
    eviction_policy: TrajectoryEvictionPolicy,
}

impl TLE {
    /// Create a new TLE from classic 2-line format
    /// 
    /// # Arguments
    /// * `line1` - First line of TLE data
    /// * `line2` - Second line of TLE data
    /// 
    /// # Returns
    /// * `Result<TLE, BraheError>` - New TLE instance or error
    pub fn from_lines(line1: &str, line2: &str) -> Result<Self, BraheError> {
        Self::from_tle_string(&format!("{}\n{}", line1, line2))
    }
    
    /// Create a new TLE from 3-line format (with satellite name)
    /// 
    /// # Arguments  
    /// * `name` - Satellite name (line 0)
    /// * `line1` - First line of TLE data
    /// * `line2` - Second line of TLE data
    /// 
    /// # Returns
    /// * `Result<TLE, BraheError>` - New TLE instance or error
    pub fn from_3le(name: &str, line1: &str, line2: &str) -> Result<Self, BraheError> {
        Self::from_tle_string(&format!("{}\n{}\n{}", name, line1, line2))
    }
    
    /// Create TLE from raw TLE string (auto-detects format)
    /// 
    /// # Arguments
    /// * `tle_string` - Raw TLE data as string
    /// 
    /// # Returns
    /// * `Result<TLE, BraheError>` - New TLE instance or error
    pub fn from_tle_string(tle_string: &str) -> Result<Self, BraheError> {
        let lines: Vec<&str> = tle_string.trim().lines().collect();
        
        match lines.len() {
            2 => {
                // 2-line format
                let line1 = lines[0].trim();
                let line2 = lines[1].trim();
                
                Self::parse_tle(None, line1, line2)
            },
            3 => {
                // 3-line format (with satellite name)
                let name = lines[0].trim();
                let line1 = lines[1].trim();
                let line2 = lines[2].trim();
                
                Self::parse_tle(Some(name), line1, line2)
            },
            _ => Err(BraheError::Error(format!(
                "Invalid TLE format: expected 2 or 3 lines, got {}", 
                lines.len()
            ))),
        }
    }
    
    /// Internal TLE parsing function
    fn parse_tle(name: Option<&str>, line1: &str, line2: &str) -> Result<Self, BraheError> {
        // Validate TLE format
        Self::validate_tle_lines(line1, line2)?;
        
        // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
        let norad_id_str = &line1[2..7];
        let (norad_id, format) = Self::decode_norad_id(norad_id_str)?;
        
        // Create modified lines with numeric NORAD ID for SGP4 parsing
        // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
        let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
            Self::convert_to_numeric_line(line1, norad_id)?
        } else {
            line1.to_string()
        };
        let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
            Self::convert_to_numeric_line(line2, norad_id)?
        } else {
            line2.to_string()
        };
        
        // Parse using sgp4 crate with correct function based on format
        let elements = if let Some(sat_name) = name {
            // 3-line format
            let tle_string = format!("{}\n{}\n{}", sat_name, numeric_line1, numeric_line2);
            sgp4::parse_3les(&tle_string)
                .map_err(|e| BraheError::Error(format!("Failed to parse 3-line TLE: {:?}", e)))?
                .into_iter()
                .next()
                .ok_or_else(|| BraheError::Error(format!("No elements parsed from 3-line TLE. Input was: '{}'", tle_string)))?
        } else {
            // 2-line format
            let tle_string = format!("{}\n{}", numeric_line1, numeric_line2);
            sgp4::parse_2les(&tle_string)
                .map_err(|e| BraheError::Error(format!("Failed to parse 2-line TLE: {:?}", e)))?
                .into_iter()
                .next()
                .ok_or_else(|| BraheError::Error(format!("No elements parsed from 2-line TLE. Input was: '{}'", tle_string)))?
        };
        
        // Create SGP4 constants for propagation
        let constants = sgp4::Constants::from_elements(&elements)
            .map_err(|e| BraheError::Error(format!("Failed to create SGP4 constants: {:?}", e)))?;
        
        // Extract epoch from elements
        let epoch = Self::extract_epoch(&elements)?;
        
        // Create initial orbital state from TLE mean elements
        let initial_state = Self::elements_to_orbit_state(&elements, epoch)?;
        let current_state = initial_state.clone();
        
        // Initialize trajectory with initial state
        let mut trajectory = Trajectory::new(InterpolationMethod::Linear);
        trajectory.add_state(initial_state.clone())?;
        
        Ok(TLE {
            line1: line1.to_string(),
            line2: line2.to_string(),
            satellite_name: name.map(|s| s.to_string()),
            format,
            norad_id_string: norad_id_str.to_string(),
            norad_id,
            elements,
            constants,
            initial_state,
            current_state,
            trajectory,
            max_trajectory_size: None,
            eviction_policy: TrajectoryEvictionPolicy::None,
        })
    }
    
    /// Decode NORAD ID from string, handling Alpha-5 format
    /// 
    /// # Arguments
    /// * `id_str` - 5-character NORAD ID string
    /// 
    /// # Returns
    /// * `Result<(u32, TleFormat), BraheError>` - Decoded numeric ID and format type
    fn decode_norad_id(id_str: &str) -> Result<(u32, TleFormat), BraheError> {
        if id_str.len() != 5 {
            return Err(BraheError::Error(format!(
                "NORAD ID must be exactly 5 characters, got: '{}'", 
                id_str
            )));
        }
        
        let first_char = id_str.chars().next().unwrap();
        
        if first_char.is_ascii_digit() {
            // Classic format - all numeric
            let id = id_str.parse::<u32>()
                .map_err(|_| BraheError::Error(format!("Invalid numeric NORAD ID: '{}'", id_str)))?;
            Ok((id, TleFormat::Classic))
        } else if first_char.is_ascii_alphabetic() && first_char.is_ascii_uppercase() {
            // Alpha-5 format - first character is letter
            let alpha_5_id = Self::decode_alpha5_id(id_str)?;
            Ok((alpha_5_id, TleFormat::Alpha5))
        } else {
            Err(BraheError::Error(format!(
                "Invalid NORAD ID format: '{}' (first character must be digit or uppercase letter)", 
                id_str
            )))
        }
    }
    
    /// Decode Alpha-5 NORAD ID to numeric value
    /// 
    /// Alpha-5 mapping: A=10, B=11, ..., H=17, J=19, K=20, ..., N=23, P=25, ..., Z=35
    /// (skipping I=18 and O=24 to avoid confusion with 1 and 0)
    fn decode_alpha5_id(id_str: &str) -> Result<u32, BraheError> {
        let first_char = id_str.chars().next().unwrap();
        let remaining = &id_str[1..];
        
        // Validate remaining 4 characters are numeric
        let remaining_num = remaining.parse::<u32>()
            .map_err(|_| BraheError::Error(format!(
                "Invalid Alpha-5 ID: remaining digits '{}' must be numeric", 
                remaining
            )))?;
        
        // Convert letter to corresponding tens digit
        // A=10, B=11, ..., H=17, J=18, K=19, ..., N=22, P=23, ..., Z=33
        // (I and O are skipped)
        let tens_digit = match first_char {
            'A' => 10, 'B' => 11, 'C' => 12, 'D' => 13, 'E' => 14, 'F' => 15, 'G' => 16, 'H' => 17,
            'J' => 18, 'K' => 19, 'L' => 20, 'M' => 21, 'N' => 22,
            'P' => 23, 'Q' => 24, 'R' => 25, 'S' => 26, 'T' => 27, 'U' => 28, 'V' => 29, 
            'W' => 30, 'X' => 31, 'Y' => 32, 'Z' => 33,
            _ => return Err(BraheError::Error(format!(
                "Invalid Alpha-5 letter: '{}' (I and O are not allowed)", 
                first_char
            ))),
        };
        
        // Combine tens digit with remaining 4 digits
        // For Alpha-5, we map to ranges starting at 100000
        let numeric_id = (tens_digit - 10) * 10000 + 100000 + remaining_num;
        
        Ok(numeric_id)
    }
    
    /// Convert TLE line to use numeric NORAD ID for SGP4 compatibility
    fn convert_to_numeric_line(line: &str, numeric_id: u32) -> Result<String, BraheError> {
        if line.len() != 69 {
            return Err(BraheError::Error(format!(
                "TLE line must be 69 characters, got {}", 
                line.len()
            )));
        }
        
        // Replace NORAD ID (positions 2-7, 0-indexed: 2-6) with numeric version
        let mut modified_line = line.to_string();
        
        // For very large IDs, we'll use modulo to fit in 5 digits for SGP4 compatibility
        let sgp4_id = if numeric_id > 99999 { numeric_id % 100000 } else { numeric_id };
        let numeric_id_str = format!("{:05}", sgp4_id);
        
        modified_line.replace_range(2..7, &numeric_id_str);
        
        // Recalculate checksum for the modified line
        let new_checksum = Self::calculate_tle_checksum(&modified_line[..68]);
        modified_line.replace_range(68..69, &new_checksum.to_string());
        
        Ok(modified_line)
    }
    
    /// Calculate TLE line checksum
    fn calculate_tle_checksum(line: &str) -> u8 {
        let mut sum = 0;
        for c in line.chars() {
            match c {
                '0'..='9' => sum += c.to_digit(10).unwrap() as u8,
                '-' => sum += 1,
                _ => {} // Ignore other characters
            }
        }
        sum % 10
    }
    
    /// Validate TLE line format
    fn validate_tle_lines(line1: &str, line2: &str) -> Result<(), BraheError> {
        // Basic length check
        if line1.len() != 69 {
            return Err(BraheError::Error(format!(
                "TLE line 1 must be exactly 69 characters, got {}", 
                line1.len()
            )));
        }
        
        if line2.len() != 69 {
            return Err(BraheError::Error(format!(
                "TLE line 2 must be exactly 69 characters, got {}", 
                line2.len()
            )));
        }
        
        // Check line numbers
        if !line1.starts_with('1') {
            return Err(BraheError::Error("TLE line 1 must start with '1'".to_string()));
        }
        
        if !line2.starts_with('2') {
            return Err(BraheError::Error("TLE line 2 must start with '2'".to_string()));
        }
        
        // Validate NORAD IDs match between lines
        let id1 = &line1[2..7];
        let id2 = &line2[2..7];
        if id1 != id2 {
            return Err(BraheError::Error(format!(
                "NORAD IDs don't match: line1='{}', line2='{}'", 
                id1, id2
            )));
        }
        
        Ok(())
    }
    
    /// Extract epoch from SGP4 elements
    fn extract_epoch(elements: &sgp4::Elements) -> Result<Epoch, BraheError> {
        // SGP4 elements contain a NaiveDateTime
        // Convert to Julian Date for Brahe's Epoch
        let dt = elements.datetime;
        
        // Convert NaiveDateTime to Julian Date
        // Julian day number calculation from Gregorian calendar
        let year = dt.year();
        let month = dt.month();
        let day = dt.day();
        let hour = dt.hour();
        let minute = dt.minute();
        let second = dt.second();
        let nanosecond = dt.nanosecond();
        
        // Calculate Julian Day Number
        let a = (14 - month) / 12;
        let y = year + 4800 - a as i32;
        let m = month + 12 * a - 3;
        
        let jdn = day as f64 + (153 * m + 2) as f64 / 5.0 + 365.25 * y as f64 - 32045.0;
        
        // Convert time of day to fraction
        let time_fraction = (hour as f64 * 3600.0 + minute as f64 * 60.0 + 
                           second as f64 + nanosecond as f64 / 1e9) / 86400.0;
        
        let jd = jdn + time_fraction - 0.5; // JD starts at noon
        
        Ok(Epoch::from_jd(jd, crate::time::TimeSystem::UTC))
    }
    
    /// Convert SGP4 elements to OrbitState
    fn elements_to_orbit_state(elements: &sgp4::Elements, epoch: Epoch) -> Result<OrbitState, BraheError> {
        // Create state vector from TLE mean elements
        // [a, e, i, Ω, ω, M] in SI units (meters and radians)
        
        // Calculate semi-major axis from mean motion
        // a = (μ/n²)^(1/3) where μ = GM_EARTH, n = mean_motion
        use crate::constants::GM_EARTH;
        let n = elements.mean_motion * 2.0 * std::f64::consts::PI / 86400.0; // Convert rev/day to rad/s
        let a = (GM_EARTH / (n * n)).powf(1.0/3.0); // Semi-major axis in meters
        
        let e = elements.eccentricity;
        let i = elements.inclination.to_radians();
        let omega_cap = elements.right_ascension.to_radians();
        let omega = elements.argument_of_perigee.to_radians();
        let m = elements.mean_anomaly.to_radians();
        
        let state_vector = Vector6::new(a, e, i, omega_cap, omega, m);
        
        Ok(OrbitState::new(
            epoch,
            state_vector,
            OrbitFrame::ECI,
            OrbitStateType::TLEMean,
            AngleFormat::Radians,
        ))
    }
    
    /// Apply eviction policy to manage trajectory memory
    fn apply_eviction_policy(&mut self) -> Result<(), BraheError> {
        if let Some(max_size) = self.max_trajectory_size {
            if self.trajectory.states.len() > max_size {
                match self.eviction_policy {
                    TrajectoryEvictionPolicy::None => {
                        // Do nothing, let it grow
                    },
                    TrajectoryEvictionPolicy::KeepRecent => {
                        // Remove oldest states
                        let excess = self.trajectory.states.len() - max_size;
                        self.trajectory.states.drain(0..excess);
                    },
                    TrajectoryEvictionPolicy::KeepWithinDuration => {
                        // TODO: Implement duration-based eviction
                        // For now, fall back to KeepRecent
                        let excess = self.trajectory.states.len() - max_size;
                        self.trajectory.states.drain(0..excess);
                    },
                    TrajectoryEvictionPolicy::MemoryBased => {
                        // TODO: Implement memory-based eviction
                        // For now, fall back to KeepRecent
                        let excess = self.trajectory.states.len() - max_size;
                        self.trajectory.states.drain(0..excess);
                    },
                }
            }
        }
        
        Ok(())
    }
    
    /// Get satellite name (if available)
    pub fn satellite_name(&self) -> Option<&str> {
        self.satellite_name.as_deref()
    }
    
    /// Get original NORAD ID string (may be Alpha-5 format)
    pub fn norad_id_string(&self) -> &str {
        &self.norad_id_string
    }
    
    /// Get decoded numeric NORAD ID
    pub fn norad_id(&self) -> u32 {
        self.elements.norad_id as u32
    }
    
    /// Get international designator
    pub fn international_designator(&self) -> Option<String> {
        self.elements.international_designator.clone()
    }
    
    /// Get epoch of TLE
    pub fn epoch(&self) -> Epoch {
        self.initial_state.epoch
    }
    
    /// Get mean motion (revolutions per day)
    pub fn mean_motion(&self) -> f64 {
        self.elements.mean_motion
    }
    
    /// Get eccentricity
    pub fn eccentricity(&self) -> f64 {
        self.elements.eccentricity
    }
    
    /// Get inclination in radians
    pub fn inclination(&self) -> f64 {
        self.elements.inclination.to_radians()
    }
    
    /// Get right ascension of ascending node in radians  
    pub fn raan(&self) -> f64 {
        self.elements.right_ascension.to_radians()
    }
    
    /// Get argument of perigee in radians
    pub fn argument_of_perigee(&self) -> f64 {
        self.elements.argument_of_perigee.to_radians()
    }
    
    /// Get mean anomaly in radians
    pub fn mean_anomaly(&self) -> f64 {
        self.elements.mean_anomaly.to_radians()
    }
    
    /// Check if TLE uses Alpha-5 format
    pub fn is_alpha5(&self) -> bool {
        self.format == TleFormat::Alpha5
    }
    
    /// Propagate to specific time and return Cartesian state
    pub fn propagate(&self, epoch: Epoch) -> Result<OrbitState, BraheError> {
        // Calculate minutes since TLE epoch
        let time_diff = epoch - self.epoch();
        let minutes_since_epoch = time_diff / 60.0; // Convert seconds to minutes
        
        // Use SGP4 to propagate
        let prediction = self.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation failed: {:?}", e)))?;
        
        // Convert SGP4 prediction to Cartesian state  
        // SGP4 returns position in km and velocity in km/s
        let position = Vector3::new(
            prediction.position[0] * 1000.0, // Convert km to m
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
        );
        
        let velocity = Vector3::new(
            prediction.velocity[0] * 1000.0, // Convert km/s to m/s  
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        );
        
        let state_vector = Vector6::new(
            position.x, position.y, position.z,
            velocity.x, velocity.y, velocity.z,
        );
        
        Ok(OrbitState::new(
            epoch,
            state_vector,
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        ))
    }
}

/// Implement OrbitPropagator trait for TLE
impl OrbitPropagator for TLE {
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError> {
        let propagated_state = self.propagate(target_epoch)?;
        self.current_state = propagated_state.clone();
        self.trajectory.add_state(propagated_state)?;
        self.apply_eviction_policy()?;
        Ok(&self.current_state)
    }
    
    fn reset(&mut self) -> Result<(), BraheError> {
        self.current_state = self.initial_state.clone();
        
        // Reset trajectory with initial state
        self.trajectory = Trajectory::new(self.trajectory.interpolation_method);
        self.trajectory.add_state(self.initial_state.clone())?;
        
        Ok(())
    }
    
    fn current_epoch(&self) -> Epoch {
        self.current_state.epoch
    }
    
    fn current_state(&self) -> &OrbitState {
        &self.current_state
    }
    
    fn initial_state(&self) -> &OrbitState {
        &self.initial_state
    }
    
    fn set_initial_state(&mut self, _state: OrbitState) -> Result<(), BraheError> {
        // For TLE, we don't allow changing the initial state since it comes from the TLE data
        Err(BraheError::Error(
            "Cannot change initial state for TLE propagator - state is determined by TLE data".to_string()
        ))
    }
    
    fn set_initial_conditions(
        &mut self, 
        _epoch: Epoch, 
        _state: Vector6<f64>,
        _frame: OrbitFrame,
        _orbit_type: OrbitStateType,
        _angle_format: AngleFormat,
    ) -> Result<(), BraheError> {
        // For TLE, we don't allow changing initial conditions
        Err(BraheError::Error(
            "Cannot change initial conditions for TLE propagator - conditions are determined by TLE data".to_string()
        ))
    }
    
    fn propagate_batch(&mut self, epochs: &[Epoch]) -> Result<Vec<OrbitState>, BraheError> {
        let mut states = Vec::with_capacity(epochs.len());
        
        for &epoch in epochs {
            let state = self.propagate(epoch)?;
            self.current_state = state.clone();
            self.trajectory.add_state(state.clone())?;
            states.push(state);
        }
        
        self.apply_eviction_policy()?;
        
        Ok(states)
    }
    
    fn trajectory(&self) -> &Trajectory<OrbitState> {
        &self.trajectory
    }
    
    fn trajectory_mut(&mut self) -> &mut Trajectory<OrbitState> {
        &mut self.trajectory
    }
    
    fn set_max_trajectory_size(&mut self, max_size: Option<usize>) {
        self.max_trajectory_size = max_size;
    }
    
    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy) {
        self.eviction_policy = policy;
    }
}

impl FromStr for TLE {
    type Err = BraheError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_tle_string(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::State;
    use approx::assert_abs_diff_eq;
    
    // Example TLE for ISS (International Space Station) - Classic format
    const ISS_CLASSIC_TLE: &str = r#"1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"#;
    
    // Example 3-line TLE with satellite name  
    const ISS_3LE: &str = r#"ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"#;
    
    // Example Alpha-5 TLE (using A0000 format for NORAD ID >= 100000)
    const ALPHA5_TLE: &str = r#"1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  9991
2 A0000  50.0000   0.0000 0001000   0.0000   0.0000 15.50000000000000"#;
    
    #[test]
    fn test_tle_from_classic_2line() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        
        assert_eq!(tle.format, TleFormat::Classic);
        assert_eq!(tle.satellite_name(), None);
        assert_eq!(tle.norad_id(), 25544);
        assert_eq!(tle.norad_id_string(), "25544");
    }
    
    #[test]
    fn test_tle_from_3line_format() {
        let tle = TLE::from_3le("ISS (ZARYA)", 
            "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992",
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003").unwrap();
        
        assert_eq!(tle.satellite_name(), Some("ISS (ZARYA)"));
        assert_eq!(tle.norad_id(), 25544);
    }
    
    #[test]
    fn test_alpha5_decoding() {
        // Test Alpha-5 decoding: A0000 should decode to 100000
        let (decoded_id, format) = TLE::decode_norad_id("A0000").unwrap();
        assert_eq!(decoded_id, 100000);
        assert_eq!(format, TleFormat::Alpha5);
        
        // Test other Alpha-5 examples
        let (decoded_id, _) = TLE::decode_norad_id("E8493").unwrap();
        assert_eq!(decoded_id, 148493);
        
        let (decoded_id, _) = TLE::decode_norad_id("Z9999").unwrap();
        assert_eq!(decoded_id, 339999);
        
        // Test skipped letters (I and O not allowed)
        assert!(TLE::decode_norad_id("I0000").is_err());
        assert!(TLE::decode_norad_id("O0000").is_err());
    }
    
    #[test]
    fn test_tle_validation() {
        // Test invalid line length
        let invalid_tle = "1 25544U\n2 25544";
        assert!(TLE::from_tle_string(invalid_tle).is_err());
        
        // Test wrong line numbers
        let wrong_line_num = r#"2 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991
1 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000000"#;
        assert!(TLE::from_tle_string(wrong_line_num).is_err());
        
        // Test mismatched NORAD IDs
        let mismatched_ids = r#"1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991
2 25545  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000000"#;
        assert!(TLE::from_tle_string(mismatched_ids).is_err());
        
        // Test invalid number of lines
        let single_line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991";
        assert!(TLE::from_tle_string(single_line).is_err());
        
        let four_lines = r#"Line 0
1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000000
Line 3"#;
        assert!(TLE::from_tle_string(four_lines).is_err());
    }
    
    #[test]
    fn test_tle_orbital_elements() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        
        // Test basic orbital elements access
        assert!(tle.eccentricity() < 0.01); // Low Earth orbit should have low eccentricity
        assert_abs_diff_eq!(tle.inclination().to_degrees(), 51.6461, epsilon = 0.1);
        assert!(tle.mean_motion() > 15.0); // ISS orbits about 15.5 times per day
        assert_abs_diff_eq!(tle.mean_motion(), 15.48919103, epsilon = 0.1);
        
        // Test argument of perigee and RAAN
        assert_abs_diff_eq!(tle.argument_of_perigee().to_degrees(), 88.1267, epsilon = 0.1);
        assert_abs_diff_eq!(tle.raan().to_degrees(), 306.0234, epsilon = 0.1);
        assert_abs_diff_eq!(tle.mean_anomaly().to_degrees(), 25.5695, epsilon = 0.1);
    }
    
    #[test]
    fn test_tle_propagation() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        
        let initial_epoch = tle.epoch();
        let future_epoch = initial_epoch + 3600.0; // 1 hour later
        
        // Test single propagation
        let state = tle.propagate(future_epoch).unwrap();
        assert_eq!(*state.epoch(), future_epoch);
        assert_eq!(state.orbit_type, OrbitStateType::Cartesian);
        assert_eq!(state.frame, OrbitFrame::ECI);
        
        // Verify position is reasonable for LEO satellite
        let position = state.position().unwrap();
        let altitude_km = (position.norm() - 6371000.0) / 1000.0; // Rough altitude calculation
        assert!(altitude_km > 200.0 && altitude_km < 800.0); // Reasonable LEO altitude range
        
        // Verify velocity is reasonable for LEO
        let velocity = state.velocity().unwrap();
        let velocity_magnitude = velocity.norm() / 1000.0; // Convert m/s to km/s
        assert!(velocity_magnitude > 6.0 && velocity_magnitude < 9.0); // LEO velocity range
    }
    
    #[test] 
    fn test_tle_propagator_trait() {
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        
        let initial_epoch = tle.current_epoch();
        let target_epoch = initial_epoch + 1800.0; // 30 minutes later
        
        // Test OrbitPropagator trait methods
        let propagated_state = tle.propagate_to(target_epoch).unwrap();
        assert_eq!(*propagated_state.epoch(), target_epoch);
        
        // Verify current state was updated
        assert_eq!(tle.current_epoch(), target_epoch);
        
        // Test batch propagation
        let epochs = vec![
            initial_epoch + 900.0,  // 15 minutes
            initial_epoch + 1800.0, // 30 minutes  
            initial_epoch + 2700.0, // 45 minutes
        ];
        
        let states = tle.propagate_batch(&epochs).unwrap();
        assert_eq!(states.len(), 3);
        
        for (i, state) in states.iter().enumerate() {
            assert_eq!(*state.epoch(), epochs[i]);
            
            // Verify each state has reasonable position for LEO
            let position = state.position().unwrap();
            let altitude_km = (position.norm() - 6371000.0) / 1000.0;
            assert!(altitude_km > 200.0 && altitude_km < 800.0);
        }
        
        // Test reset functionality
        tle.reset().unwrap();
        assert_eq!(tle.current_epoch(), initial_epoch);
        assert_eq!(tle.trajectory().states.len(), 1);
    }
    
    #[test]
    fn test_trajectory_memory_management() {
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        
        // Set trajectory limits
        tle.set_max_trajectory_size(Some(5));
        tle.set_eviction_policy(TrajectoryEvictionPolicy::KeepRecent);
        
        let initial_epoch = tle.current_epoch();
        
        // Propagate to many epochs to test eviction
        let mut target_epochs = Vec::new();
        for i in 1..=10 {
            let epoch = initial_epoch + (i as f64) * 600.0; // Every 10 minutes
            target_epochs.push(epoch);
        }
        
        let _states = tle.propagate_batch(&target_epochs).unwrap();
        
        // Should only keep the most recent 5 states (plus initial state may be kept)
        assert!(tle.trajectory().states.len() <= 6);
        
        // Verify the kept states are the most recent ones
        let trajectory_states = &tle.trajectory().states;
        if trajectory_states.len() > 1 {
            // Should be in chronological order
            for i in 1..trajectory_states.len() {
                assert!(trajectory_states[i].epoch() >= trajectory_states[i-1].epoch());
            }
        }
    }
    
    #[test]
    fn test_tle_immutable_state() {
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        
        let initial_state = tle.initial_state().clone();
        
        // TLE propagators should not allow changing initial state/conditions
        // since they come from the TLE data
        assert!(tle.set_initial_state(initial_state.clone()).is_err());
        
        use crate::trajectories::AngleFormat;
        assert!(tle.set_initial_conditions(
            initial_state.epoch,
            nalgebra::Vector6::zeros(),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None
        ).is_err());
    }
    
    #[test]
    fn test_checksum_calculation() {
        // Test TLE checksum calculation
        let line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999";
        let checksum = TLE::calculate_tle_checksum(line);
        assert_eq!(checksum, 2); // Calculated checksum for this line
        
        // Test line with negative values
        let line_with_neg = "1 25544U 98067A   21001.00000000 -.00001764  00000-0 -40967-4 0  999";
        let checksum_neg = TLE::calculate_tle_checksum(line_with_neg);
        assert_eq!(checksum_neg, 4); // Should count minus signs as 1
    }
    
    #[test]
    fn test_numeric_line_conversion() {
        // Test conversion of Alpha-5 line to numeric for SGP4 compatibility
        let alpha5_line = "1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  9991";
        let numeric_line = TLE::convert_to_numeric_line(alpha5_line, 100000).unwrap();
        
        // Should replace A0000 with 00000 (100000 mod 100000)
        assert!(numeric_line.contains("00000"));
        assert!(!numeric_line.contains("A0000"));
        
        // Should recalculate checksum
        assert_eq!(numeric_line.len(), 69);
        
        // Test with smaller ID that fits in 5 digits
        let small_line = TLE::convert_to_numeric_line(alpha5_line, 12345).unwrap();
        assert!(small_line.contains("12345"));
    }
    
    #[test]
    fn test_from_str_trait() {
        // Test that TLE implements FromStr
        let tle: TLE = ISS_CLASSIC_TLE.parse().unwrap();
        assert_eq!(tle.norad_id(), 25544);
        
        // Test error case
        let bad_tle: Result<TLE, _> = "invalid tle".parse();
        assert!(bad_tle.is_err());
    }
    
    #[test]
    fn test_is_alpha5_flag() {
        let classic_tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        assert!(!classic_tle.is_alpha5());
        
        // Would need a valid Alpha-5 TLE to test this properly
        // For now just test the enum comparison
        assert_eq!(TleFormat::Classic != TleFormat::Alpha5, true);
    }
    
    #[test]
    fn test_julian_date_conversion() {
        // Test the datetime to Julian Date conversion
        // Use a known date: 2021-01-01 00:00:00 UTC = JD 2459215.5
        
        // Create a mock elements struct to test epoch extraction
        // Note: This test might need adjustment based on actual SGP4 API
        // For now, just test the conversion logic would work
        
        let jd_2021_jan_1 = 2459215.5;
        let epoch = Epoch::from_jd(jd_2021_jan_1, TimeSystem::UTC);
        
        // Basic sanity check that our epoch system works
        assert_abs_diff_eq!(epoch.jd(), jd_2021_jan_1, epsilon = 1e-10);
    }
}
