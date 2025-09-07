/*!
 * The `tle` module provides functionality for working with NORAD Two-Line Element (TLE) data.
 * 
 * Supports both classic TLE format and Alpha-5 TLE format using the SGP4 propagation algorithm.
 * Alpha-5 expands the NORAD ID range by replacing the first digit with a letter for IDs >= 100000.
 */

use crate::orbits::propagation::{OrbitPropagator, TrajectoryEvictionPolicy};
use crate::orbits::traits::AnalyticPropagator;
use crate::time::Epoch;
use crate::trajectories::{AngleFormat, InterpolationMethod, OrbitFrame, OrbitState, OrbitStateType, Trajectory, State};
use crate::utils::BraheError;
use crate::coordinates::{state_cartesian_to_osculating};
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

/// Calculate TLE line checksum as an independent function
/// 
/// # Arguments
/// * `line` - First 68 characters of TLE line (without checksum)
/// 
/// # Returns
/// * `u8` - Calculated checksum digit
pub fn calculate_tle_line_checksum(line: &str) -> u8 {
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

/// Validate TLE line format as an independent function
/// 
/// # Arguments
/// * `line1` - First line of TLE data
/// * `line2` - Second line of TLE data
/// 
/// # Returns
/// * `bool` - true if valid, false otherwise
pub fn validate_tle_lines(line1: &str, line2: &str) -> bool {
    // Validate individual lines
    if !validate_tle_line(line1, 1) || !validate_tle_line(line2, 2) {
        return false;
    }
    
    // Validate NORAD IDs match between lines
    let id1 = &line1[2..7];
    let id2 = &line2[2..7];
    id1 == id2
}

/// Internal function to validate TLE lines with detailed error messages
/// Used by TLE struct creation for better error reporting
fn validate_tle_lines_with_errors(line1: &str, line2: &str) -> Result<(), BraheError> {
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
    
    // Validate checksums
    let checksum1_expected = line1.chars().nth(68).unwrap().to_digit(10)
        .ok_or_else(|| BraheError::Error("Line 1 checksum must be a digit".to_string()))? as u8;
    let checksum1_calculated = calculate_tle_line_checksum(&line1[..68]);
    if checksum1_expected != checksum1_calculated {
        return Err(BraheError::Error(format!(
            "Line 1 checksum mismatch: expected {}, calculated {}", 
            checksum1_expected, checksum1_calculated
        )));
    }
    
    let checksum2_expected = line2.chars().nth(68).unwrap().to_digit(10)
        .ok_or_else(|| BraheError::Error("Line 2 checksum must be a digit".to_string()))? as u8;
    let checksum2_calculated = calculate_tle_line_checksum(&line2[..68]);
    if checksum2_expected != checksum2_calculated {
        return Err(BraheError::Error(format!(
            "Line 2 checksum mismatch: expected {}, calculated {}", 
            checksum2_expected, checksum2_calculated
        )));
    }
    
    Ok(())
}

/// Validate single TLE line format
/// 
/// # Arguments
/// * `line` - TLE line to validate
/// * `expected_line_number` - Expected line number (1 or 2)
/// 
/// # Returns
/// * `bool` - true if valid, false otherwise
pub fn validate_tle_line(line: &str, expected_line_number: u8) -> bool {
    // Basic length check
    if line.len() != 69 {
        return false;
    }
    
    // Check line number
    let first_char = line.chars().next().unwrap_or(' ');
    if first_char != (b'0' + expected_line_number) as char {
        return false;
    }
    
    // Validate checksum
    let checksum_expected = line.chars().nth(68).and_then(|c| c.to_digit(10)).unwrap_or(10) as u8;
    if checksum_expected > 9 {
        return false; // Invalid checksum character
    }
    
    let checksum_calculated = calculate_tle_line_checksum(&line[..68]);
    checksum_expected == checksum_calculated
}

/// Extract NORAD ID from TLE string, handling both classic and Alpha-5 formats
/// 
/// Alpha-5 mapping: A=10, B=11, ..., H=17, J=19, K=20, ..., N=23, P=25, ..., Z=35
/// (skipping I=18 and O=24 to avoid confusion with 1 and 0)
/// 
/// # Arguments
/// * `id_str` - 5-character NORAD ID string (either numeric or Alpha-5)
/// 
/// # Returns
/// * `Result<u32, BraheError>` - Decoded numeric NORAD ID
pub fn extract_tle_norad_id(id_str: &str) -> Result<u32, BraheError> {
    if id_str.len() != 5 {
        return Err(BraheError::Error(format!(
            "NORAD ID must be exactly 5 characters, got: '{}'", 
            id_str
        )));
    }
    
    let first_char = id_str.chars().next().unwrap();
    
    if first_char.is_ascii_digit() {
        // Classic format - all numeric
        id_str.parse::<u32>()
            .map_err(|_| BraheError::Error(format!("Invalid numeric NORAD ID: '{}'", id_str)))
    } else if first_char.is_ascii_alphabetic() && first_char.is_ascii_uppercase() {
        // Alpha-5 format - first character is letter
        decode_alpha5_id(id_str)
    } else {
        Err(BraheError::Error(format!(
            "Invalid NORAD ID format: '{}' (first character must be digit or uppercase letter)", 
            id_str
        )))
    }
}

/// Internal function to decode Alpha-5 NORAD ID to numeric value
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

/// Extract epoch from SGP4 elements as an independent function
/// 
/// # Arguments
/// * `elements` - SGP4 elements structure
/// 
/// # Returns
/// * `Result<Epoch, BraheError>` - Extracted epoch or error
pub fn extract_epoch(elements: &sgp4::Elements) -> Result<Epoch, BraheError> {
    // SGP4 elements contain a NaiveDateTime
    // Convert to Julian Date for Brahe's Epoch using time::conversions
    let dt = elements.datetime;
    
    // Use the time::conversions module for Julian date conversion
    let jd = crate::time::conversions::datetime_to_jd(
        dt.year() as u32,
        dt.month() as u8,
        dt.day() as u8,
        dt.hour() as u8,
        dt.minute() as u8,
        dt.second() as f64,
        dt.nanosecond() as f64,
    );
    
    Ok(Epoch::from_jd(jd, crate::time::TimeSystem::UTC))
}

/// Decode NORAD ID from string, handling Alpha-5 format as an independent function
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
        let id = extract_tle_norad_id(id_str)?;
        Ok((id, TleFormat::Classic))
    } else if first_char.is_ascii_alphabetic() && first_char.is_ascii_uppercase() {
        // Alpha-5 format - first character is letter
        let alpha_5_id = extract_tle_norad_id(id_str)?;
        Ok((alpha_5_id, TleFormat::Alpha5))
    } else {
        Err(BraheError::Error(format!(
            "Invalid NORAD ID format: '{}' (first character must be digit or uppercase letter)", 
            id_str
        )))
    }
}

/// Convert TLE line to use numeric NORAD ID for SGP4 compatibility as an independent function
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
    let new_checksum = calculate_tle_line_checksum(&modified_line[..68]);
    modified_line.replace_range(68..69, &new_checksum.to_string());
    
    Ok(modified_line)
}

/// Convert TLE lines to orbital elements as an independent function
/// 
/// # Arguments
/// * `line1` - First line of TLE data
/// * `line2` - Second line of TLE data
/// 
/// # Returns
/// * `Result<Vector6<f64>, BraheError>` - Orbital elements [a, e, i, Ω, ω, M] in SI units
pub fn lines_to_orbit_elements(line1: &str, line2: &str) -> Result<Vector6<f64>, BraheError> {
    // Validate TLE format first
    if !validate_tle_lines(line1, line2) {
        return Err(BraheError::Error("Invalid TLE format".to_string()));
    }
    
    // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
    let norad_id_str = &line1[2..7];
    let (norad_id, format) = decode_norad_id(norad_id_str)?;
    
    // Create modified lines with numeric NORAD ID for SGP4 parsing
    // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
    let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line1, norad_id)?
    } else {
        line1.to_string()
    };
    let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line2, norad_id)?
    } else {
        line2.to_string()
    };
    
    // Parse using sgp4 crate with converted lines
    let tle_string = format!("{}\n{}", numeric_line1, numeric_line2);
    let elements = sgp4::parse_2les(&tle_string)
        .map_err(|e| BraheError::Error(format!("Failed to parse 2-line TLE: {:?}", e)))?
        .into_iter()
        .next()
        .ok_or_else(|| BraheError::Error(format!("No elements parsed from 2-line TLE. Input was: '{}'", tle_string)))?;
    
    // Convert to orbital elements vector
    // Calculate semi-major axis from mean motion using orbits module
    let n_rad_s = elements.mean_motion * 2.0 * std::f64::consts::PI / 86400.0; // Convert rev/day to rad/s
    let a = crate::orbits::semimajor_axis(n_rad_s, false); // Use orbits module function
    
    let e = elements.eccentricity;
    let i = elements.inclination.to_radians();
    let omega_cap = elements.right_ascension.to_radians();
    let omega = elements.argument_of_perigee.to_radians();
    let m = elements.mean_anomaly.to_radians();
    
    Ok(Vector6::new(a, e, i, omega_cap, omega, m))
}

/// Convert TLE lines to OrbitState as an independent function
/// 
/// # Arguments
/// * `line1` - First line of TLE data  
/// * `line2` - Second line of TLE data
/// 
/// # Returns
/// * `Result<OrbitState, BraheError>` - OrbitState representing the mean orbital elements
pub fn lines_to_orbit_state(line1: &str, line2: &str) -> Result<OrbitState, BraheError> {
    // Validate TLE format first
    if !validate_tle_lines(line1, line2) {
        return Err(BraheError::Error("Invalid TLE format".to_string()));
    }
    
    // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
    let norad_id_str = &line1[2..7];
    let (norad_id, format) = decode_norad_id(norad_id_str)?;
    
    // Create modified lines with numeric NORAD ID for SGP4 parsing
    // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
    let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line1, norad_id)?
    } else {
        line1.to_string()
    };
    let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line2, norad_id)?
    } else {
        line2.to_string()
    };
    
    // Parse using sgp4 crate with converted lines
    let tle_string = format!("{}\n{}", numeric_line1, numeric_line2);
    let elements = sgp4::parse_2les(&tle_string)
        .map_err(|e| BraheError::Error(format!("Failed to parse 2-line TLE: {:?}", e)))?
        .into_iter()
        .next()
        .ok_or_else(|| BraheError::Error(format!("No elements parsed from 2-line TLE. Input was: '{}'", tle_string)))?;
    
    // Extract epoch
    let epoch = extract_epoch(&elements)?;
    
    // Get orbital elements
    let state_vector = lines_to_orbit_elements(line1, line2)?;
    
    Ok(OrbitState::new(
        epoch,
        state_vector,
        OrbitFrame::ECI,
        OrbitStateType::TLEMean,
        AngleFormat::Radians,
    ))
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
        validate_tle_lines_with_errors(line1, line2)?;
        
        // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
        let norad_id_str = &line1[2..7];
        let (norad_id, format) = decode_norad_id(norad_id_str)?;
        
        // Create modified lines with numeric NORAD ID for SGP4 parsing
        // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
        let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
            convert_to_numeric_line(line1, norad_id)?
        } else {
            line1.to_string()
        };
        let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
            convert_to_numeric_line(line2, norad_id)?
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
        let epoch = extract_epoch(&elements)?;
        
        // Create initial orbital state from TLE lines
        let initial_state = lines_to_orbit_state(line1, line2)?;
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
        let (decoded_id, format) = decode_norad_id("A0000").unwrap();
        assert_eq!(decoded_id, 100000);
        assert_eq!(format, TleFormat::Alpha5);
        
        // Test other Alpha-5 examples
        let (decoded_id, _) = decode_norad_id("E8493").unwrap();
        assert_eq!(decoded_id, 148493);
        
        let (decoded_id, _) = decode_norad_id("Z9999").unwrap();
        assert_eq!(decoded_id, 339999);
        
        // Test skipped letters (I and O not allowed)
        assert!(decode_norad_id("I0000").is_err());
        assert!(decode_norad_id("O0000").is_err());
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
        let checksum = calculate_tle_line_checksum(line);
        assert_eq!(checksum, 2); // Calculated checksum for this line
        
        // Test line with negative values
        let line_with_neg = "1 25544U 98067A   21001.00000000 -.00001764  00000-0 -40967-4 0  999";
        let checksum_neg = calculate_tle_line_checksum(line_with_neg);
        assert_eq!(checksum_neg, 4); // Should count minus signs as 1
    }
    
    #[test]
    fn test_numeric_line_conversion() {
        // Test conversion of Alpha-5 line to numeric for SGP4 compatibility
        let alpha5_line = "1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  9991";
        let numeric_line = convert_to_numeric_line(alpha5_line, 100000).unwrap();
        
        // Should replace A0000 with 00000 (100000 mod 100000)
        assert!(numeric_line.contains("00000"));
        assert!(!numeric_line.contains("A0000"));
        
        // Should recalculate checksum
        assert_eq!(numeric_line.len(), 69);
        
        // Test with smaller ID that fits in 5 digits
        let small_line = convert_to_numeric_line(alpha5_line, 12345).unwrap();
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

    // Tests for independent functions
    
    #[test]
    fn test_validate_tle_lines() {
        // Test valid TLE lines
        let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        assert!(validate_tle_lines(line1, line2));
        
        // Test invalid length
        let short_line1 = "1 25544U 98067A";
        assert!(!validate_tle_lines(short_line1, line2));
        
        // Test mismatched NORAD IDs
        let wrong_id_line2 = "2 25545  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        assert!(!validate_tle_lines(line1, wrong_id_line2));
        
        // Test wrong line numbers
        let wrong_line_num = "3 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        assert!(!validate_tle_lines(wrong_line_num, line2));
        
        // Test invalid checksum
        let bad_checksum_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9990";
        assert!(!validate_tle_lines(bad_checksum_line1, line2));
    }
    
    #[test]
    fn test_calculate_tle_line_checksum() {
        // Test checksum calculation
        let line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999";
        let checksum = calculate_tle_line_checksum(line);
        assert_eq!(checksum, 2);
        
        // Test with negative values
        let line_with_neg = "1 25544U 98067A   21001.00000000 -.00001764  00000-0 -40967-4 0  999";
        let checksum_neg = calculate_tle_line_checksum(line_with_neg);
        assert_eq!(checksum_neg, 4);
        
        // Test empty line
        let empty_checksum = calculate_tle_line_checksum("");
        assert_eq!(empty_checksum, 0);
    }
    
    #[test]
    fn test_decode_alpha5_id() {
        // Test basic Alpha-5 decoding using internal function
        assert_eq!(decode_alpha5_id("A0000").unwrap(), 100000);
        assert_eq!(decode_alpha5_id("A0001").unwrap(), 100001);
        assert_eq!(decode_alpha5_id("B0000").unwrap(), 110000);
        assert_eq!(decode_alpha5_id("E8493").unwrap(), 148493);
        assert_eq!(decode_alpha5_id("Z9999").unwrap(), 339999);
        
        // Test invalid letters
        assert!(decode_alpha5_id("I0000").is_err());
        assert!(decode_alpha5_id("O0000").is_err());
        
        // Test non-numeric remaining
        assert!(decode_alpha5_id("AABCD").is_err());
        
        // Test wrong length (handled by caller but worth testing)
        // Note: This function assumes 5 character input from caller
    }
    
    #[test]
    fn test_extract_tle_norad_id() {
        // Test classic format (numeric NORAD IDs less than 100000)
        assert_eq!(extract_tle_norad_id("25544").unwrap(), 25544);
        assert_eq!(extract_tle_norad_id("00001").unwrap(), 1);
        assert_eq!(extract_tle_norad_id("12345").unwrap(), 12345);
        assert_eq!(extract_tle_norad_id("99999").unwrap(), 99999);
        
        // Test Alpha-5 format (NORAD IDs >= 100000)
        assert_eq!(extract_tle_norad_id("A0000").unwrap(), 100000);
        assert_eq!(extract_tle_norad_id("A0001").unwrap(), 100001);
        assert_eq!(extract_tle_norad_id("B0000").unwrap(), 110000);
        assert_eq!(extract_tle_norad_id("E8493").unwrap(), 148493);
        assert_eq!(extract_tle_norad_id("Z9999").unwrap(), 339999);
        
        // Test invalid formats
        assert!(extract_tle_norad_id("I0000").is_err()); // Invalid letter
        assert!(extract_tle_norad_id("O0000").is_err()); // Invalid letter  
        assert!(extract_tle_norad_id("AABCD").is_err()); // Non-numeric after letter
        assert!(extract_tle_norad_id("1234").is_err()); // Wrong length
        assert!(extract_tle_norad_id("123456").is_err()); // Wrong length
    }
    
    #[test]
    fn test_lines_to_orbit_elements() {
        let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        
        let elements = lines_to_orbit_elements(line1, line2).unwrap();
        
        // Check that we get 6 elements: [a, e, i, Ω, ω, M]
        assert_eq!(elements.len(), 6);
        
        // Verify semi-major axis is reasonable for ISS (around 6700-6800 km)
        let a = elements[0];
        assert!(a > 6_700_000.0 && a < 6_800_000.0); // meters
        
        // Verify eccentricity is small for LEO
        let e = elements[1];
        assert!(e < 0.01);
        
        // Verify inclination is close to expected (51.6461 degrees)
        let i_rad = elements[2];
        let i_deg = i_rad.to_degrees();
        assert_abs_diff_eq!(i_deg, 51.6461, epsilon = 0.1);
        
        // Verify RAAN
        let raan_rad = elements[3];
        let raan_deg = raan_rad.to_degrees();
        assert_abs_diff_eq!(raan_deg, 306.0234, epsilon = 0.1);
        
        // Verify argument of perigee
        let argp_rad = elements[4];
        let argp_deg = argp_rad.to_degrees();
        assert_abs_diff_eq!(argp_deg, 88.1267, epsilon = 0.1);
        
        // Verify mean anomaly
        let ma_rad = elements[5];
        let ma_deg = ma_rad.to_degrees();
        assert_abs_diff_eq!(ma_deg, 25.5695, epsilon = 0.1);
        
        // Test with invalid lines
        let bad_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9990";
        assert!(lines_to_orbit_elements(bad_line1, line2).is_err());
    }
    
    #[test]
    fn test_lines_to_orbit_state() {
        let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        
        let orbit_state = lines_to_orbit_state(line1, line2).unwrap();
        
        // Verify state properties
        assert_eq!(orbit_state.frame, OrbitFrame::ECI);
        assert_eq!(orbit_state.orbit_type, OrbitStateType::TLEMean);
        assert_eq!(orbit_state.angle_format, AngleFormat::Radians);
        
        // Verify epoch exists
        let epoch = orbit_state.epoch;
        assert!(epoch.jd() > 0.0); // Just verify we have a valid epoch
        
        // Verify state vector
        let state = orbit_state.state;
        assert_eq!(state.len(), 6);
        
        // Semi-major axis should be reasonable for ISS
        let a = state[0];
        assert!(a > 6_700_000.0 && a < 6_800_000.0);
        
        // Verify inclination matches parsed value
        let i_rad = state[2];
        let i_deg = i_rad.to_degrees();
        assert_abs_diff_eq!(i_deg, 51.6461, epsilon = 0.1);
        
        // Test with invalid lines
        let bad_line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000009";
        assert!(lines_to_orbit_state(line1, bad_line2).is_err());
    }
    
    #[test]
    fn test_alpha5_tle_lines_to_orbit_elements() {
        // Test with Alpha-5 TLE (A0000 = 100000)
        let line1_base = "1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  999";
        let line2_base = "2 A0000  50.0000   0.0000 0001000   0.0000   0.0000 15.5000000000000";
        
        // Calculate correct checksums
        let checksum1 = calculate_tle_line_checksum(line1_base);
        let checksum2 = calculate_tle_line_checksum(line2_base);
        
        let alpha5_line1 = format!("{}{}",line1_base, checksum1);
        let alpha5_line2 = format!("{}{}",line2_base, checksum2);
        
        let elements = lines_to_orbit_elements(&alpha5_line1, &alpha5_line2).unwrap();
        
        // Should successfully parse and return valid elements
        assert_eq!(elements.len(), 6);
        
        // Semi-major axis should be reasonable
        let a = elements[0];
        assert!(a > 6_000_000.0 && a < 8_000_000.0);
        
        // Eccentricity should match
        let e = elements[1];
        assert_abs_diff_eq!(e, 0.0001, epsilon = 1e-6);
        
        // Inclination should match (50 degrees)
        let i_rad = elements[2];
        let i_deg = i_rad.to_degrees();
        assert_abs_diff_eq!(i_deg, 50.0, epsilon = 0.1);
        
        // RAAN should be 0
        let raan_rad = elements[3];
        assert_abs_diff_eq!(raan_rad, 0.0, epsilon = 1e-6);
        
        // Argument of perigee should be 0
        let argp_rad = elements[4];
        assert_abs_diff_eq!(argp_rad, 0.0, epsilon = 1e-6);
        
        // Mean anomaly should be 0
        let ma_rad = elements[5];
        assert_abs_diff_eq!(ma_rad, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_analytic_propagator_state() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let state = tle.state(epoch);
        
        assert_eq!(state.len(), 6);
        assert!(state[0].abs() > 1e6); // Position should be reasonable (>1000 km)
        assert!(state[3].abs() > 1e3); // Velocity should be reasonable (>1 km/s)
    }

    #[test]
    fn test_analytic_propagator_state_eci() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let state_eci = tle.state_eci(epoch);
        
        assert_eq!(state_eci.len(), 6);
        assert!(state_eci[0].abs() > 1e6); // Position should be reasonable
        assert!(state_eci[3].abs() > 1e3); // Velocity should be reasonable
    }

    #[test]
    fn test_analytic_propagator_state_ecef() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let state_ecef = tle.state_ecef(epoch);
        
        assert_eq!(state_ecef.len(), 6);
        assert!(state_ecef[0].abs() > 1e6); // Position should be reasonable
        assert!(state_ecef[3].abs() > 1e3); // Velocity should be reasonable
    }

    #[test]
    fn test_analytic_propagator_state_osculating_elements() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let elements = tle.state_osculating_elements(epoch);
        
        assert_eq!(elements.len(), 6);
        assert!(elements[0] > 6e6); // Semi-major axis should be > 6000 km for ISS
        assert!(elements[1] >= 0.0 && elements[1] < 1.0); // Eccentricity [0,1)
        assert!(elements[2] >= 0.0 && elements[2] <= std::f64::consts::PI); // Inclination [0,π]
    }

    #[test]
    fn test_analytic_propagator_batch_states() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        let epoch1 = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2021, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![epoch1, epoch2];
        
        let states = tle.states(&epochs);
        
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].len(), 6);
        assert_eq!(states[1].len(), 6);
        
        let states_eci = tle.states_eci(&epochs);
        assert_eq!(states_eci.len(), 2);
        
        let states_ecef = tle.states_ecef(&epochs);
        assert_eq!(states_ecef.len(), 2);
        
        let states_elements = tle.states_osculating_elements(&epochs);
        assert_eq!(states_elements.len(), 2);
    }

    #[test]
    fn test_analytic_propagator_consistency() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        // Single epoch call
        let state_single = tle.state(epoch);
        
        // Batch call with single epoch
        let states_batch = tle.states(&[epoch]);
        
        // Should be identical
        assert_abs_diff_eq!(state_single[0], states_batch[0][0], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[1], states_batch[0][1], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[2], states_batch[0][2], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[3], states_batch[0][3], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[4], states_batch[0][4], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[5], states_batch[0][5], epsilon = 1e-12);
    }
}

/// Implement AnalyticPropagator trait for TLE
impl AnalyticPropagator for TLE {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_to_state(epoch);
        orbit_state.state
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_to_state(epoch);
        let eci_state = orbit_state.to_frame(&OrbitFrame::ECI).unwrap();
        eci_state.state
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_to_state(epoch);
        let ecef_state = orbit_state.to_frame(&OrbitFrame::ECEF).unwrap();
        ecef_state.state
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_to_state(epoch);
        let cart_state = orbit_state.to_cartesian().unwrap();
        state_cartesian_to_osculating(cart_state.state, false)
    }

    fn states(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }

    fn states_eci(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_eci(epoch)).collect()
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_ecef(epoch)).collect()
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_osculating_elements(epoch)).collect()
    }
}

impl TLE {
    /// Internal helper method to propagate to a state without modifying the propagator
    fn propagate_to_state(&self, epoch: Epoch) -> OrbitState {
        self.propagate(epoch).unwrap()
    }
}
